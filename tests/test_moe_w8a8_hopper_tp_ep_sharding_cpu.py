"""CPU-only proof that the Hopper kernel's 2D-blocked FP8 weight scale
(quantize_weight_fp8_grouped, {128,128} tiles) survives TP/EP sharding the
same way the existing production INT8 scheme was already proven to
(moe_quantization_memo.md Q3: quantize-then-shard == shard-then-quantize).

This is NOT the same proof reused -- it's a new one, for a structurally
different scale layout (2D, blocked across both N and K, vs. INT8's 1D,
grouped along K only). See w8a8_activation_quant_scoping_memo.md Phase 1:
"whether [2D-block sharding] survives EP sharding cleanly needs its own
proof, run CPU-only before any GPU time, exactly like Q3 was."

Uses the REAL production sharding function (utils.loader.shard_experts_tensor
-- round-robin along dim 0, e % tp_size == rank), not a reimplementation, so
this proof is only as good as that function's actual behavior, not an
assumption about it. Uses the smoke test's own quantize_weight_fp8_grouped /
dequantize_weight_fp8_grouped_gathered (layers/smoke_test_moe_w8a8_hopper.py)
for the same reason.

Does NOT validate: moe_w8a8.cu itself (needs real compile+run on Hopper), or
that this scale convention is actually what the kernel expects (still an
inferred hypothesis per moe_w8a8.h's own docstring). This only proves the
SHARDING math is safe, independent of everything else still open.

Usage:
    python tests/test_moe_w8a8_hopper_tp_ep_sharding_cpu.py
"""
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "layers"))
sys.path.insert(0, _REPO_ROOT)

from utils.loader import shard_experts_tensor            # noqa: E402
from moe_w8a8_hopper_quantize import (                    # noqa: E402
    quantize_weight_fp8_grouped,
    dequantize_weight_fp8_grouped_gathered,
)


def check_tp_size(num_experts: int, N: int, K: int, group_size: int, tp_size: int) -> bool:
    """For every rank at this tp_size: shard-then-quantize must equal
    quantize-then-shard, bitwise, for both the fp8 weight and its scale."""
    torch.manual_seed(0)
    full_w = torch.randn(num_experts, N, K, dtype=torch.float32) * 0.02

    # Path A: quantize the FULL tensor first, then shard both outputs.
    full_fp8, full_scale = quantize_weight_fp8_grouped(full_w, group_size)

    ok = True
    for rank in range(tp_size):
        # Path B: shard the full-precision tensor first (real production
        # sharding function), then quantize the local shard independently --
        # this is what actually happens at load time, one rank at a time,
        # each rank never seeing the other ranks' experts.
        local_w = shard_experts_tensor(full_w, rank, tp_size)
        local_fp8_B, local_scale_B = quantize_weight_fp8_grouped(local_w, group_size)

        # Path A's shard: same round-robin index, applied post-quantization.
        local_ids = [e for e in range(num_experts) if e % tp_size == rank]
        idx = torch.tensor(local_ids, dtype=torch.long)
        local_fp8_A = full_fp8.index_select(0, idx)
        local_scale_A = full_scale.index_select(0, idx)

        fp8_match = torch.equal(local_fp8_A.view(torch.uint8), local_fp8_B.view(torch.uint8))
        scale_match = torch.equal(local_scale_A, local_scale_B)
        shape_match = (local_fp8_B.shape == (len(local_ids), N, K)
                        and local_scale_B.shape == (len(local_ids), N // group_size, K // group_size))

        rank_ok = fp8_match and scale_match and shape_match
        ok &= rank_ok
        status = "OK" if rank_ok else "MISMATCH"
        print(f"    tp_size={tp_size} rank={rank}: experts={local_ids[:3]}{'...' if len(local_ids) > 3 else ''} "
              f"n_local={len(local_ids)}  fp8_match={fp8_match}  scale_match={scale_match}  "
              f"shape_match={shape_match}  [{status}]")

        if not rank_ok:
            # Downstream check: does the mismatch actually change dequantized
            # values, or is it purely representational (e.g. NaN formatting)?
            # If shapes/equality already failed this is diagnostic, not a
            # pass/fail input.
            deq_A = dequantize_weight_fp8_grouped_gathered(local_fp8_A, local_scale_A, group_size, torch.float32)
            deq_B = dequantize_weight_fp8_grouped_gathered(local_fp8_B, local_scale_B, group_size, torch.float32)
            max_diff = (deq_A - deq_B).abs().max().item()
            print(f"      dequantized max_abs_diff={max_diff:.6e}")

    return ok


def main():
    # num_experts=256 is the real production count (the dimension actually
    # being sharded -- worth keeping realistic). H/MI are shrunk from real
    # production dims (H=2048, MI=512) to keep this CPU-RAM-safe: the full
    # bf16-scale (E, 2*MI, H) tensor at real dims is a 2GB single float32
    # allocation, which OOM'd the default CPU allocator on first attempt --
    # a smaller, real-world echo of the exact memory-pressure warning
    # already in shard_experts_tensor's own docstring. H=256/MI=256 still
    # exercises 2 blocks per dim at group_size=128, which is what this proof
    # actually needs (the sharding math has no size dependence -- only
    # dim-0-only reduction and divisibility matter), without re-proving the
    # separately-known CPU-RAM constraint this test isn't trying to test.
    num_experts = 256
    H = 256
    MI = 256
    group_size = 128
    tp_sizes = [1, 2, 4, 8]  # matches the EP-imbalance simulation's own sweep

    print(f"Config: num_experts={num_experts} H={H} MI={MI} (2*MI={2*MI}) group_size={group_size}\n")

    all_ok = True

    print("=== gate_up_proj shape: (E, 2*MI, H) ===")
    for tp in tp_sizes:
        all_ok &= check_tp_size(num_experts, 2 * MI, H, group_size, tp)

    print("\n=== down_proj shape: (E, H, MI) ===")
    for tp in tp_sizes:
        all_ok &= check_tp_size(num_experts, H, MI, group_size, tp)

    print("\nPASS -- 2D-blocked FP8 scale is sharding-commutative for both expert "
          "tensors at tp_size in {1,2,4,8}, same conclusion as the INT8 scheme's "
          "Q3 proof, now independently confirmed for this different scale layout."
          if all_ok else
          "\nFAIL -- 2D-block sharding is NOT commutative here. Do not assume this "
          "transfers from the INT8 case; something about the {128,128} block layout "
          "interacts with round-robin dim-0 sharding in a way the 1D case didn't. "
          "Investigate before writing any production loader code against this scheme.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
