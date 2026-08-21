"""GPU smoke test for the grouped-scale fused_moe_triton.py kernel -- checks
correctness (vs. this project's existing, already-trusted
dequantize_weight_int8_grouped + plain matmul reference) and prints a timing
comparison. Does NOT test the full Qwen35MoE integration (EP local_slots
indexing, CUDA graph capture, the two-GEMM+SiLU pipeline) -- that's real
follow-on work once this foundation is confirmed solid. This is specifically
about answering: does the grouped-scale kernel modification actually produce
correct numbers on real hardware, and is it faster than what we have today.

Usage:
    python layers/smoke_test_fused_moe_triton.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from moe_align_block_size import moe_align_block_size          # noqa: E402
from fused_moe_triton import invoke_fused_moe_kernel            # noqa: E402
from moe_int8_quantize import (                                  # noqa: E402
    quantize_weight_int8_grouped,
    dequantize_weight_int8_grouped,
)


def reference_output(x, w_int8, w_scale, group_size, topk_ids, topk_weights, top_k):
    """Deliberately naive per-token Python loop -- NOT the production path,
    kept only as a slow-but-obviously-correct ground truth to validate the
    vectorized reference below against, one level removed from trusting a
    single implementation's self-consistency."""
    M = x.shape[0]
    K = x.shape[1]
    N = w_int8.shape[1]
    out = torch.zeros((M, top_k, N), dtype=x.dtype, device=x.device)
    for m in range(M):
        for k in range(top_k):
            e = int(topk_ids[m, k].item())
            w = dequantize_weight_int8_grouped(w_int8[e], w_scale[e], group_size, x.dtype)
            out[m, k] = x[m] @ w.t()
    return out


def vectorized_reference_output(x, w_int8, w_scale, group_size, topk_ids, top_k):
    """The ACTUAL production path -- models/qwen3_5.py's _forward_gathered_ep,
    the hasattr(self.experts, "gate_up_proj_int8") branch, same shape as its
    down_proj usage: one batched fancy-index gather across all (M, top_k)
    selections at once, one batched dequant call (today's fixed, verified
    version -- no fp32 intermediate, fused cast+multiply), one einsum. No
    per-token Python loop. This is what actually measured 37.1 tok/s in
    production earlier this session -- THIS is the number the kernel needs
    to beat, not the naive loop above."""
    gu_i8 = w_int8[topk_ids]   # (M, top_k, N, K) int8
    gu_sc = w_scale[topk_ids]  # (M, top_k, N, K//group_size)
    w = dequantize_weight_int8_grouped(gu_i8, gu_sc, group_size, x.dtype)  # (M, top_k, N, K)
    out = torch.einsum('mkoh,mh->mko', w, x)
    return out


def main():
    if not torch.cuda.is_available():
        print("CUDA required for this smoke test.")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Real-checkpoint-scale shapes: local_num_experts=128 at tp=2,
    # hidden_size=2048, top_k=8. Using down_proj's shape (H=2048 in,
    # MI=512 out) rather than gate_up_proj's (2x wider) to keep this smoke
    # test's memory/runtime modest -- same kernel code path either way.
    E = 128
    K = 2048
    N = 512
    top_k = 8
    M = 16  # concurrency level
    group_size = 128

    print(f"Config: E={E} K={K} N={N} top_k={top_k} M={M} group_size={group_size}")

    # Real weight + our existing, validated quantizer -- NOT synthetic int8
    # noise, so this test exercises the same quantization this project
    # already trusts, not a different one invented for this test.
    w_bf16 = torch.randn((E, N, K), dtype=dtype, device=device) * 0.02
    w_int8, w_scale = quantize_weight_int8_grouped(w_bf16, group_size)
    print(f"w_int8: {tuple(w_int8.shape)} {w_int8.dtype}  "
          f"w_scale: {tuple(w_scale.shape)} {w_scale.dtype}")

    x = torch.randn((M, K), dtype=dtype, device=device) * 0.02
    topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.rand((M, top_k), dtype=torch.float32, device=device)

    # ---- Reference (naive loop, ground truth only) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ref = reference_output(x, w_int8, w_scale, group_size, topk_ids, topk_weights, top_k)
    torch.cuda.synchronize()
    ref_s = time.perf_counter() - t0
    print(f"Naive loop (ground truth, not production): {ref_s*1000:.2f} ms")

    # ---- Vectorized reference (the ACTUAL production path) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    vec_ref = vectorized_reference_output(x, w_int8, w_scale, group_size, topk_ids, top_k)
    torch.cuda.synchronize()
    vec_ref_s = time.perf_counter() - t0
    print(f"Vectorized (PRODUCTION path, first call): {vec_ref_s*1000:.2f} ms")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    vec_ref = vectorized_reference_output(x, w_int8, w_scale, group_size, topk_ids, top_k)
    torch.cuda.synchronize()
    vec_ref_warm_s = time.perf_counter() - t0
    print(f"Vectorized (PRODUCTION path, warm): {vec_ref_warm_s*1000:.2f} ms")

    cos_vec_vs_naive = torch.nn.functional.cosine_similarity(
        ref.float().reshape(-1), vec_ref.float().reshape(-1), dim=0
    ).item()
    print(f"Sanity: naive vs. vectorized reference agree, cosine={cos_vec_vs_naive:.6f} "
          f"(both should be ~identical -- this just confirms the vectorized reference "
          f"itself isn't the thing introducing an error)")

    # ---- Kernel path ----
    block_size_m = 16
    block_size_n = 64
    block_size_k = 64  # divides group_size=128 evenly, required
    assert group_size % block_size_k == 0

    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, E
    )

    A = x  # (M, K)
    B = w_int8  # (E, N, K)
    C = torch.zeros((M, top_k, N), dtype=dtype, device=device)

    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }

    def run_kernel():
        invoke_fused_moe_kernel(
            A, B, C,
            None, w_scale,
            topk_weights, topk_ids,
            sorted_ids, expert_ids, num_tokens_post_padded,
            False, top_k, config, tl_compute_type(dtype),
            use_fp8_w8a8=False, use_int8_w8a16=True,
            quant_group_size=group_size,
        )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_kernel()
    torch.cuda.synchronize()
    kernel_s = time.perf_counter() - t0
    print(f"Kernel (grouped-scale fused_moe_triton, first call incl. compile): {kernel_s*1000:.2f} ms")

    # Second call -- compiled/warm, the number that actually matters.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_kernel()
    torch.cuda.synchronize()
    kernel_warm_s = time.perf_counter() - t0
    print(f"Kernel (warm, second call): {kernel_warm_s*1000:.2f} ms")

    # ---- Correctness, against the naive ground truth ----
    kernel_out = C  # (M, top_k, N)
    cos = torch.nn.functional.cosine_similarity(
        ref.float().reshape(-1), kernel_out.float().reshape(-1), dim=0
    ).item()
    max_abs_err = (ref.float() - kernel_out.float()).abs().max().item()
    print(f"\nCorrectness (kernel vs. naive ground truth): cosine_similarity={cos:.6f}  "
          f"max_abs_err={max_abs_err:.6f}")
    print("PASS" if cos > 0.999 else "FAIL -- investigate before trusting this kernel")

    # ---- The number that actually matters: kernel vs. PRODUCTION ----
    print(f"\n{'='*70}")
    print("REAL COMPARISON -- kernel vs. the actual production path")
    print(f"{'='*70}")
    print(f"  Vectorized/production (warm): {vec_ref_warm_s*1000:.3f} ms")
    print(f"  Kernel (warm):                {kernel_warm_s*1000:.3f} ms")
    print(f"  Speedup: {vec_ref_warm_s/kernel_warm_s:.2f}x")
    print(f"{'='*70}")
    print("This is a single-GEMM microbenchmark (matching down_proj's shape), not the "
          "full two-GEMM+SiLU decode step or an end-to-end tok/s number -- real inside "
          "the actual engine may differ (kernel-launch overhead, CUDA graph capture "
          "behavior, the second GEMM, EP local_slots indexing not yet wired in). This is "
          "the right first checkpoint before investing in full integration, not the final "
          "production number.")


def tl_compute_type(dtype):
    import triton.language as tl
    return {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype]


if __name__ == "__main__":
    main()
