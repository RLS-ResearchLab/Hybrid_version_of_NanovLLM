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
    """Exactly what models/qwen3_5.py's _forward_gathered_ep does today:
    gather per-token expert weights, dequantize (our validated, fixed
    fused-multiply version), matmul. Ungrouped by block -- a straightforward
    per-token loop, deliberately simple/obviously-correct rather than fast,
    since this IS the correctness reference everything else is judged against."""
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

    # ---- Reference (existing, trusted path) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ref = reference_output(x, w_int8, w_scale, group_size, topk_ids, topk_weights, top_k)
    torch.cuda.synchronize()
    ref_s = time.perf_counter() - t0
    print(f"Reference (existing dequant-then-matmul path): {ref_s*1000:.2f} ms")

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
            A, B, C.view(M * top_k, N),
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

    # ---- Correctness ----
    kernel_out = C  # (M, top_k, N)
    cos = torch.nn.functional.cosine_similarity(
        ref.float().reshape(-1), kernel_out.float().reshape(-1), dim=0
    ).item()
    max_abs_err = (ref.float() - kernel_out.float()).abs().max().item()
    print(f"\nCorrectness: cosine_similarity={cos:.6f}  max_abs_err={max_abs_err:.6f}")
    print("PASS" if cos > 0.999 else "FAIL -- investigate before trusting this kernel")

    print(f"\nSpeed (warm): reference={ref_s*1000:.2f}ms  kernel={kernel_warm_s*1000:.2f}ms  "
          f"speedup={ref_s/kernel_warm_s:.2f}x")
    print("NOTE: reference uses a naive per-token Python loop -- NOT the same as this "
          "project's actual vectorized _forward_gathered_ep gather+dequant+einsum path. "
          "This speed number is illustrative of the kernel's own raw performance, not a "
          "direct A/B against the 37.1 tok/s figure already measured in production.")


def tl_compute_type(dtype):
    import triton.language as tl
    return {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype]


if __name__ == "__main__":
    main()
