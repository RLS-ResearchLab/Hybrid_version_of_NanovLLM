#pragma once

// Shared declaration for moe_w8a8.cu's public entry point, so the .cu
// definition and any caller (moe_w8a8_binding.cpp) are checked against the
// SAME signature by the compiler, instead of the binding re-declaring it by
// hand and silently drifting out of sync with the real definition -- exactly
// the kind of mismatch that's easy to introduce and hard to notice given
// nobody here can compile-test this on the current dev machine.
//
// CONTRACT, read before calling this from anywhere:
//
//   - `out` is written via cp_reduce_async_bulk(..., op_add, ...) inside the
//     kernel -- it ACCUMULATES into `out`, it does not overwrite it. The
//     caller MUST zero `out` before every call, or leftover memory (a
//     previous call's result, or uninitialized allocator garbage) corrupts
//     the output silently. This is the single easiest way to get a
//     plausible-looking wrong answer out of this kernel.
//
//   - `w` (gate_up_proj) is expected as a (K, N*num_experts) row-major FP8
//     tensor with stride K*sizeof(fp8) -- matches the tensor_map_w setup in
//     moe_w8a8.cu's launch function exactly. `w2` (down_proj) is expected as
//     (K2, N2*num_experts) with K2=N/2, N2=K, same convention. Getting this
//     layout wrong does not crash -- it silently reads the wrong bytes.
//
//   - `w_scale`/`w2_scale` are per-128x128-block scales (block_shape={128,128}
//     is hardcoded in the kernel), consistent with this project's existing
//     group_size=128 INT8 scheme on Ampere -- NOT vLLM's original whole-row
//     scale convention. Worth confirming explicitly once this is compiled
//     and testable, not just inferred from reading the source.
//
//   - `block_n`/`warp_n` (BN/WN in the kernel) must be one of exactly two
//     supported pairs: (32, 8) or (64, 4). Any other pair hits the
//     `dispatch_bn_wn` default branch, prints an error to stderr, and
//     silently does NOT launch the kernel -- `out` stays whatever it was
//     (zeroed, per the contract above), not an error the caller can catch
//     programmatically. Same failure shape for `stages` outside 1-5 and
//     `block_m` outside {8,16,...,128} (their own dispatch defaults).
//
//   - `x`/`w`/`w2` are __nv_fp8_e4m3 (PyTorch: torch.float8_e4m3fn).
//     `x_scale`/`w_scale`/`w2_scale`/`topk_weights`/`scaling_factor` are
//     float32. `out` is __nv_bfloat16 (torch.bfloat16). `sorted_token_ids`/
//     `expert_ids`/`num_tokens_post_padded` are int32 -- matches this
//     project's existing moe_align_block_size.py contract (same three
//     tensors, same dtypes, same semantics), so that piece should be
//     reusable as-is once this is wired up.
//
//   - Never validated against a reference. Do not trust output correctness
//     until it's been checked in isolation (see H200_test_day_checklist.md /
//     the smoke-test pattern layers/smoke_test_fused_moe_triton.py already
//     established for the Ampere kernel) -- this header documents the
//     intended contract, not a verified one.

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using fp8 = __nv_fp8_e4m3;

void fused_moe_w8a8_wgmma_up_down_acc(
        const fp8* x,
        const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const float* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int num_experts,
        int sorted_num,
        int block_m,
        int block_n,
        int warp_n,
        int stages,
        float scaling_factor,
        cudaStream_t stream
        );
