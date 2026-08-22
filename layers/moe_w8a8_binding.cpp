// PyTorch-facing binding for moe_w8a8.cu's fused_moe_w8a8_wgmma_up_down_acc().
// UNCOMPILED, UNTESTED -- written without access to a CUDA toolchain (this
// dev machine has none). Needs a real compile+run pass on H200 before any of
// the shape/dtype assumptions below can be trusted. See moe_w8a8.h for the
// full contract this is enforcing.
//
// Tensor shapes asserted below are INFERRED from reading moe_w8a8.cu's TMA
// setup code (the CUtensorMap size/stride arguments), not confirmed against
// the kernel author's actual intent -- treat as a documented hypothesis to
// verify on first real run, not an established fact.
//   x:      (M, K)                       fp8 (torch.float8_e4m3fn)
//   w:      (num_experts, N, K)          fp8   -- gate_up_proj, row-major, K fastest
//   w2:     (num_experts, K, N/2)        fp8   -- down_proj, row-major, N/2 fastest
//   out:    (M, K)                       bf16  -- MUST be pre-zeroed, kernel accumulates into it
//   x_scale, w_scale, w2_scale, topk_weights: float32
//   sorted_token_ids, expert_ids, num_tokens_post_padded: int32

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "moe_w8a8.h"

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (0)

void moe_w8a8_forward(
        torch::Tensor x,
        torch::Tensor x_scale,
        torch::Tensor w,
        torch::Tensor w_scale,
        torch::Tensor w2,
        torch::Tensor w2_scale,
        torch::Tensor out,
        torch::Tensor sorted_token_ids,
        torch::Tensor expert_ids,
        torch::Tensor num_tokens_post_padded,
        torch::Tensor topk_weights,
        int64_t top_k,
        int64_t M,
        int64_t K,
        int64_t N,
        int64_t num_experts,
        int64_t sorted_num,
        int64_t block_m,
        int64_t block_n,
        int64_t warp_n,
        int64_t stages,
        double scaling_factor)
{
    CHECK_INPUT(x);
    CHECK_INPUT(x_scale);
    CHECK_INPUT(w);
    CHECK_INPUT(w_scale);
    CHECK_INPUT(w2);
    CHECK_INPUT(w2_scale);
    CHECK_INPUT(out);
    CHECK_INPUT(sorted_token_ids);
    CHECK_INPUT(expert_ids);
    CHECK_INPUT(num_tokens_post_padded);
    CHECK_INPUT(topk_weights);

    TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "x must be torch.float8_e4m3fn");
    TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn, "w must be torch.float8_e4m3fn");
    TORCH_CHECK(w2.scalar_type() == at::kFloat8_e4m3fn, "w2 must be torch.float8_e4m3fn");
    TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "x_scale must be float32");
    TORCH_CHECK(w_scale.scalar_type() == at::kFloat, "w_scale must be float32");
    TORCH_CHECK(w2_scale.scalar_type() == at::kFloat, "w2_scale must be float32");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights must be float32");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be torch.bfloat16");
    TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt, "sorted_token_ids must be int32");
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt, "expert_ids must be int32");
    TORCH_CHECK(num_tokens_post_padded.scalar_type() == at::kInt, "num_tokens_post_padded must be int32");

    // Only two (block_n, warp_n) pairs are wired up in dispatch_bn_wn() --
    // anything else hits its default branch, prints to stderr, and silently
    // skips the launch (out stays zeroed, not an exception). Catch it here
    // instead, loudly, before that happens.
    TORCH_CHECK((block_n == 32 && warp_n == 8) || (block_n == 64 && warp_n == 4),
                "block_n/warp_n must be (32, 8) or (64, 4) -- the only configs "
                "dispatch_bn_wn() in moe_w8a8.cu actually launches; any other pair "
                "silently no-ops instead of erroring");
    TORCH_CHECK(stages >= 1 && stages <= 5, "stages must be in [1, 5] (dispatch_stages() range)");
    TORCH_CHECK(block_m >= 8 && block_m <= 128 && block_m % 8 == 0,
                "block_m must be a multiple of 8 in [8, 128] (dispatch_bn_wn<BM> instantiation range)");

    // Shape checks against the inferred contract above -- M/K/N/num_experts
    // are caller-supplied rather than derived from tensor shapes, precisely
    // so a caller mistake surfaces as a loud TORCH_CHECK failure here instead
    // of a silent shape mismatch inside the kernel.
    TORCH_CHECK(x.dim() == 2 && x.size(0) == M && x.size(1) == K, "x must be shape (M, K)");
    TORCH_CHECK(w.dim() == 3 && w.size(0) == num_experts && w.size(1) == N && w.size(2) == K,
                "w must be shape (num_experts, N, K)");
    TORCH_CHECK(w2.dim() == 3 && w2.size(0) == num_experts && w2.size(1) == K && w2.size(2) == N / 2,
                "w2 must be shape (num_experts, K, N/2)");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == K, "out must be shape (M, K)");
    TORCH_CHECK(sorted_token_ids.numel() >= sorted_num, "sorted_token_ids shorter than sorted_num");
    TORCH_CHECK(topk_weights.numel() >= M * top_k, "topk_weights shorter than M*top_k");

    fused_moe_w8a8_wgmma_up_down_acc(
            reinterpret_cast<const fp8*>(x.data_ptr()),
            x_scale.data_ptr<float>(),
            reinterpret_cast<fp8*>(w.data_ptr()),
            w_scale.data_ptr<float>(),
            reinterpret_cast<fp8*>(w2.data_ptr()),
            w2_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            sorted_token_ids.data_ptr<int>(),
            expert_ids.data_ptr<int>(),
            num_tokens_post_padded.data_ptr<int>(),
            topk_weights.data_ptr<float>(),
            static_cast<int>(top_k),
            static_cast<int>(M),
            static_cast<int>(K),
            static_cast<int>(N),
            static_cast<int>(num_experts),
            static_cast<int>(sorted_num),
            static_cast<int>(block_m),
            static_cast<int>(block_n),
            static_cast<int>(warp_n),
            static_cast<int>(stages),
            static_cast<float>(scaling_factor),
            at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_w8a8_forward", &moe_w8a8_forward,
          "Hopper wgmma/TMA fused W8A8 MoE forward (gate_up -> SiLU -> dynamic FP8 "
          "requant -> down_proj). Accumulates into `out` via reduce-add -- caller "
          "must zero `out` first. UNVALIDATED -- never run against a reference, "
          "see moe_w8a8.h.");
}
