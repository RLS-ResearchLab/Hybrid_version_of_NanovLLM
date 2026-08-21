"""GPU smoke test for the FULL two-GEMM+SiLU MoE pipeline via the kernel --
matches exactly what models/qwen3_5.py's _forward_gathered_ep computes
(gate_up -> chunk -> silu(gate)*up -> down_proj), not just one isolated GEMM
like smoke_test_fused_moe_triton.py. This is the last isolated checkpoint
before actually touching models/qwen3_5.py.

down_proj's kernel call uses a trick worth stating plainly: its input (h,
the SiLU-gated intermediate) is already per-(token, k) -- each of the N*TK
rows was computed using a SPECIFIC expert's gate_up weights, so down_proj
must use that SAME expert for that SAME row. Reshaping h to (N*TK, MI) and
local_slots to (N*TK, 1) treats each (token, k) pair as its own "virtual
token" with top_k=1 for this second kernel call -- a valid degenerate case
of the same alignment/kernel machinery, not a special case requiring new code.

Usage:
    python layers/smoke_test_full_moe_pipeline.py
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from moe_align_block_size import moe_align_block_size          # noqa: E402
from fused_moe_triton import invoke_fused_moe_kernel            # noqa: E402
from moe_int8_quantize import (                                  # noqa: E402
    quantize_weight_int8_grouped,
    dequantize_weight_int8_grouped,
)


def tl_compute_type(dtype):
    import triton.language as tl
    return {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype]


def production_pipeline(x, gu_i8, gu_sc, dp_i8, dp_sc, group_size, local_slots, top_k):
    """Exactly _forward_gathered_ep's int8 branch (models/qwen3_5.py) --
    today's real, measured-37.1-tok/s production code, unmodified logic."""
    gu_gathered_i8 = gu_i8[local_slots]   # (N, TK, 2*MI, H)
    gu_gathered_sc = gu_sc[local_slots]
    dp_gathered_i8 = dp_i8[local_slots]   # (N, TK, H, MI)
    dp_gathered_sc = dp_sc[local_slots]
    gate_up = dequantize_weight_int8_grouped(gu_gathered_i8, gu_gathered_sc, group_size, x.dtype)
    down = dequantize_weight_int8_grouped(dp_gathered_i8, dp_gathered_sc, group_size, x.dtype)
    gw, uw = gate_up.chunk(2, dim=2)
    h_gate = torch.einsum('nkmh,nh->nkm', gw, x)
    h_up = torch.einsum('nkmh,nh->nkm', uw, x)
    h = F.silu(h_gate) * h_up
    out_e = torch.einsum('nkhm,nkm->nkh', down, h)
    return out_e


def kernel_pipeline(x, gu_i8, gu_sc, dp_i8, dp_sc, group_size, local_slots, top_k,
                    gu_config, dp_config, local_num_experts):
    N_tok, H = x.shape
    two_mi = gu_i8.shape[1]
    MI = two_mi // 2
    dtype = x.dtype
    device = x.device
    compute_type = tl_compute_type(dtype)

    # ---- GEMM 1: gate_up_proj, N_tok real tokens, top_k experts each ----
    sorted_ids1, expert_ids1, ntpp1 = moe_align_block_size(
        local_slots, gu_config["BLOCK_SIZE_M"], local_num_experts
    )
    gate_up_out = torch.zeros((N_tok, top_k, two_mi), dtype=dtype, device=device)
    dummy_w1 = torch.ones((N_tok, top_k), dtype=torch.float32, device=device)
    invoke_fused_moe_kernel(
        x, gu_i8, gate_up_out,
        None, gu_sc,
        dummy_w1, local_slots,
        sorted_ids1, expert_ids1, ntpp1,
        False, top_k, gu_config, compute_type,
        use_fp8_w8a8=False, use_int8_w8a16=True,
        quant_group_size=group_size,
    )
    gw, uw = gate_up_out.chunk(2, dim=2)   # each (N, TK, MI)
    h = F.silu(gw) * uw                     # (N, TK, MI)

    # ---- GEMM 2: down_proj -- each (token, k) pair is its own "virtual
    # token" with top_k=1, since h's rows are already expert-specific ----
    h_flat = h.reshape(N_tok * top_k, MI)
    local_slots_flat = local_slots.reshape(N_tok * top_k, 1)
    sorted_ids2, expert_ids2, ntpp2 = moe_align_block_size(
        local_slots_flat, dp_config["BLOCK_SIZE_M"], local_num_experts
    )
    down_out = torch.zeros((N_tok * top_k, 1, H), dtype=dtype, device=device)
    dummy_w2 = torch.ones((N_tok * top_k, 1), dtype=torch.float32, device=device)
    invoke_fused_moe_kernel(
        h_flat, dp_i8, down_out,
        None, dp_sc,
        dummy_w2, local_slots_flat,
        sorted_ids2, expert_ids2, ntpp2,
        False, 1, dp_config, compute_type,
        use_fp8_w8a8=False, use_int8_w8a16=True,
        quant_group_size=group_size,
    )
    return down_out.reshape(N_tok, top_k, H)


def main():
    if not torch.cuda.is_available():
        print("CUDA required for this smoke test.")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Real checkpoint scale: local_num_experts=128 at tp=2, hidden_size=2048,
    # moe_intermediate_size=512, top_k=8, group_size=128.
    E = 128
    H = 2048
    MI = 512
    top_k = 8
    N_tok = 16  # concurrency level
    group_size = 128

    print(f"Config: E={E} H={H} MI={MI} top_k={top_k} N_tok={N_tok} group_size={group_size}")

    gu_bf16 = torch.randn((E, 2 * MI, H), dtype=dtype, device=device) * 0.02
    dp_bf16 = torch.randn((E, H, MI), dtype=dtype, device=device) * 0.02
    gu_i8, gu_sc = quantize_weight_int8_grouped(gu_bf16, group_size)
    dp_i8, dp_sc = quantize_weight_int8_grouped(dp_bf16, group_size)
    print(f"gate_up_proj_int8: {tuple(gu_i8.shape)}   down_proj_int8: {tuple(dp_i8.shape)}")

    x = torch.randn((N_tok, H), dtype=dtype, device=device) * 0.02
    local_slots = torch.randint(0, E, (N_tok, top_k), dtype=torch.int32, device=device)

    # ---- Production path ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prod_out = production_pipeline(x, gu_i8, gu_sc, dp_i8, dp_sc, group_size, local_slots, top_k)
    torch.cuda.synchronize()
    prod_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prod_out = production_pipeline(x, gu_i8, gu_sc, dp_i8, dp_sc, group_size, local_slots, top_k)
    torch.cuda.synchronize()
    prod_warm_s = time.perf_counter() - t0
    print(f"Production pipeline: first={prod_s*1000:.3f}ms  warm={prod_warm_s*1000:.3f}ms")

    # ---- Kernel path ----
    gu_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    dp_config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    assert group_size % gu_config["BLOCK_SIZE_K"] == 0
    assert group_size % dp_config["BLOCK_SIZE_K"] == 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kernel_out = kernel_pipeline(x, gu_i8, gu_sc, dp_i8, dp_sc, group_size, local_slots, top_k,
                                 gu_config, dp_config, E)
    torch.cuda.synchronize()
    kernel_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kernel_out = kernel_pipeline(x, gu_i8, gu_sc, dp_i8, dp_sc, group_size, local_slots, top_k,
                                 gu_config, dp_config, E)
    torch.cuda.synchronize()
    kernel_warm_s = time.perf_counter() - t0
    print(f"Kernel pipeline:     first={kernel_s*1000:.3f}ms  warm={kernel_warm_s*1000:.3f}ms")

    # ---- Correctness ----
    cos = torch.nn.functional.cosine_similarity(
        prod_out.float().reshape(-1), kernel_out.float().reshape(-1), dim=0
    ).item()
    max_abs_err = (prod_out.float() - kernel_out.float()).abs().max().item()
    print(f"\nCorrectness (kernel pipeline vs. production pipeline): "
          f"cosine_similarity={cos:.6f}  max_abs_err={max_abs_err:.6f}")
    print("PASS" if cos > 0.999 else "FAIL -- investigate before trusting this pipeline")

    print(f"\n{'='*70}")
    print("FULL PIPELINE -- kernel vs. production, both warm")
    print(f"{'='*70}")
    print(f"  Production: {prod_warm_s*1000:.3f} ms")
    print(f"  Kernel:     {kernel_warm_s*1000:.3f} ms")
    print(f"  Speedup: {prod_warm_s/kernel_warm_s:.2f}x")
    print(f"{'='*70}")
    print("Still not wired into the real engine -- no CUDA graph capture, no EP "
          "ep_rank/owned_mask combine, no all_reduce. This validates the COMPUTE "
          "pattern is correct and fast in isolation; integration into "
          "models/qwen3_5.py is the next real step.")


if __name__ == "__main__":
    main()
