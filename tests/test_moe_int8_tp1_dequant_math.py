"""CPU-only correctness check for the int8 dequant branch added to
_forward_gathered() tonight (models/qwen3_5.py), to unblock use_moe_w8a8=True
at tensor_parallel_size=1 (previously EP-only, ep_size>1).

WHY THIS EXISTS RATHER THAN JUST IMPORTING THE REAL FUNCTION: confirmed
empirically (not assumed) that models/qwen3_5.py cannot be imported on this
Windows dev machine at all -- it unconditionally imports triton at module
level (via layers/fused_moe_int8.py -> layers/fused_moe_triton.py), even
when the fused-kernel flag is unset, and this machine has no triton
installed. So the real _forward_gathered() can't be called here, full stop.

What this DOES check: forward_gathered_int8_branch() below is a line-for-line
copy of _forward_gathered's new int8 branch (models/qwen3_5.py, the
gu_i8/gu_sc/dp_i8/dp_sc gather -> dequantize_weight_int8_grouped ->
chunk -> einsum sequence) -- copied from the actual file, not retyped from
memory, using the REAL dequantize_weight_int8_grouped (tests/moe_int8_quantize.py,
not reimplemented). If there's a shape or indexing bug in what got written
into models/qwen3_5.py, this reproduction has it too and this test catches
it. Compared against an independently-structured reference (an explicit
per-(token,k) Python loop, no einsum, no batched gather -- deliberately a
DIFFERENT computation shape, so agreement isn't just two copies of the same
mistake).

What this does NOT check: the real _forward_gathered() function itself (import
blocked, see above), the fused-Triton-kernel branch (needs triton+GPU), or
anything about _forward_dispatch's separate (already-similar, per-expert-loop
based) int8 branch. Before fully trusting this covers the real fix, diff this
file's forward_gathered_int8_branch() against models/qwen3_5.py's actual
_forward_gathered() by eye once more -- a copy can silently drift from its
source.

Usage:
    python tests/test_moe_int8_tp1_dequant_math.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_int8_quantize import quantize_weight_int8_grouped, dequantize_weight_int8_grouped  # noqa: E402


def forward_gathered_int8_branch(x, gate_up_proj_int8, gate_up_proj_scale,
                                  down_proj_int8, down_proj_scale, group_size, idx):
    """Line-for-line copy of _forward_gathered's int8 dequant branch
    (models/qwen3_5.py) -- `idx` here plays the same role `idx` (== local
    slots at ep_size=1) plays there."""
    gu_i8 = gate_up_proj_int8[idx]              # (N, TK, 2*MI, H) int8
    gu_sc = gate_up_proj_scale[idx]
    dp_i8 = down_proj_int8[idx]                 # (N, TK, H, MI) int8
    dp_sc = down_proj_scale[idx]
    gate_up = dequantize_weight_int8_grouped(gu_i8, gu_sc, group_size, x.dtype)
    down = dequantize_weight_int8_grouped(dp_i8, dp_sc, group_size, x.dtype)

    gw, uw = gate_up.chunk(2, dim=2)             # each (N, TK, MI, H)
    h_gate = torch.einsum('nkmh,nh->nkm', gw, x)  # (N, TK, MI)
    h_up = torch.einsum('nkmh,nh->nkm', uw, x)    # (N, TK, MI)
    h = F.silu(h_gate) * h_up                     # (N, TK, MI)
    out_e = torch.einsum('nkhm,nkm->nkh', down, h)  # (N, TK, H)
    return out_e


def reference_per_token_per_k_loop(x, gate_up_proj, down_proj, idx):
    """Independent ground truth: explicit double loop over (token, k), no
    einsum, no batched gather -- structurally different from the function
    under test on purpose. Runs in float64 for a tight numerical reference."""
    N, TK = idx.shape
    H = x.shape[1]
    out = torch.zeros(N, TK, H, dtype=torch.float64)
    for n in range(N):
        for k in range(TK):
            e = idx[n, k].item()
            gw, uw = gate_up_proj[e].double().chunk(2, dim=0)  # each (MI, H)
            xn = x[n].double()
            h_gate = gw @ xn
            h_up = uw @ xn
            h = F.silu(h_gate) * h_up
            out[n, k] = down_proj[e].double() @ h
    return out


def main():
    torch.manual_seed(0)
    # Modest synthetic dims, real-dims-proportional -- this is a shape/logic
    # check, not a scale check (that's what the isolated kernel smoke tests
    # are for).
    E, H, MI, TK, N, group_size = 8, 256, 256, 4, 6, 128

    gu_bf16 = (torch.randn(E, 2 * MI, H, dtype=torch.float32) * 0.02)
    dp_bf16 = (torch.randn(E, H, MI, dtype=torch.float32) * 0.02)
    gu_i8, gu_sc = quantize_weight_int8_grouped(gu_bf16, group_size)
    dp_i8, dp_sc = quantize_weight_int8_grouped(dp_bf16, group_size)

    x = torch.randn(N, H, dtype=torch.float32) * 0.02
    # idx plays the role of _forward_gathered's `idx` tensor directly (the
    # thing that's now passed straight into the int8 gather instead of
    # local_slots = idx // ep_size, correct only because ep_size=1 there).
    idx = torch.randint(0, E, (N, TK), dtype=torch.int64)

    out_e = forward_gathered_int8_branch(x, gu_i8, gu_sc, dp_i8, dp_sc, group_size, idx)

    print(f"Shape check: out_e={tuple(out_e.shape)}  expected=({N}, {TK}, {H})")
    assert out_e.shape == (N, TK, H), "SHAPE MISMATCH -- the gather/einsum indexing is wrong"

    # Reference against the DEQUANTIZED (not original bf16) weights -- this
    # isolates whether the GATHER/EINSUM/COMBINE logic is right, separate
    # from int8 quantization error itself (already validated elsewhere,
    # tests/moe_int8_quantize.py's own suite, and the real GSM8K checks).
    gu_deq_full = dequantize_weight_int8_grouped(gu_i8, gu_sc, group_size, torch.float32)
    dp_deq_full = dequantize_weight_int8_grouped(dp_i8, dp_sc, group_size, torch.float32)
    ref = reference_per_token_per_k_loop(x, gu_deq_full, dp_deq_full, idx)

    cos = F.cosine_similarity(out_e.double().reshape(-1), ref.reshape(-1), dim=0).item()
    max_abs_err = (out_e.double() - ref).abs().max().item()
    print(f"cosine_similarity={cos:.10f}  max_abs_err={max_abs_err:.3e}")
    ok = cos > 0.999999 and out_e.shape == (N, TK, H)
    print("PASS" if ok else "FAIL -- investigate before trusting the tp=1 fix")
    print("\nScope reminder: this validates the GATHER/DEQUANT/EINSUM tensor math only, "
          "reproduced from models/qwen3_5.py's real int8 branch but not imported from it "
          "(triton import wall on this machine, see module docstring). Does NOT validate "
          "the fused-kernel branch, the real function's integration with the rest of "
          "Qwen35MoE.forward(), or anything on real GPU hardware.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
