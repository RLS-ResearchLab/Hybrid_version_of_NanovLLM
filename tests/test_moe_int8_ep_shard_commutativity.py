"""Q3 -- EP-sharding correctness for quantized MoE experts, CPU-only.

============================================================================
WHAT THIS PROVES, AND WHY IT MATTERS FOR INTEGRATION
============================================================================
utils/loader.py's expert_local_slot() / shard_experts_tensor() shard a
batched Experts tensor purely along dim 0 (the expert axis), round-robin
(expert e -> rank e % tp_size). shard_experts_tensor is shape-agnostic
beyond dim 0 -- it is literally full_tensor.index_select(0, idx) -- so it
works UNCHANGED on the (E, out_features, num_groups) scale tensor this
project's INT8 quantizer produces, with zero new sharding code.

That leaves exactly one real question, and it is the one this file answers:
does it matter WHEN quantization happens relative to sharding?

  PATH A (quantize-then-shard): quantize the full (256, ...) weight into
  int8 + scale, THEN shard both via the existing shard_experts_tensor.

  PATH B (shard-then-quantize): shard the full-precision weight via the
  existing, unmodified, already-tested shard_experts_tensor (exactly what
  load_model() does today, at ep_size>1, for bf16 weights -- see
  utils/loader.py's load_model() shape-mismatch branch), THEN each rank
  quantizes only its own local slice independently.

If A and B give bitwise-identical results, Path B is the correct
integration choice: it requires ZERO changes to load_model() or
shard_experts_tensor, reuses 100% already-tested production sharding code,
and never needs the full 256-expert tensor materialized an extra time
purely for quantization. If A and B do NOT match, that would mean
quantization has some cross-expert interaction this design didn't account
for, and Path A (quantize globally, then shard) would be required instead
-- a materially more invasive integration.

This is exactly the kind of assumption this project's own incident history
says to verify rather than take on faith (see the in_proj_qkv merged-chunk
bug, the conv1d mispairing bug -- both were "should be fine" assumptions
about how sharding interacts with a computation, that turned out not to
hold without being checked).

============================================================================
SCOPE
============================================================================
  COVERED   A == B, bitwise, for both int8 weight and scale, at every rank,
            for tp_size in {1, 2, 4}.
  COVERED   Negative-contamination check: no rank's shard contains any
            trace of another rank's expert data (weight OR scale) --
            same standard tests/test_expert_round_robin_loader.py already
            established for bf16 weights, extended to the quantized
            artifacts.
  COVERED   Round-trip: scattering every rank's Path-B shard back together
            and dequantizing reproduces quantize-then-dequantize of the
            original full tensor, exactly.
  COVERED   The REAL utils/loader.py functions are imported and used
            directly -- this file contains no reimplementation of sharding
            logic, on purpose, so it cannot silently drift from what
            load_model() actually does.
  NOT       GPU, distributed process groups, or the real checkpoint. Pure
            tensor operations, small synthetic expert counts (E=8), real
            hidden_size/intermediate_size/group_size (2048/512/128) so the
            group-boundary arithmetic is exercised at true dimensions.
  NOT       The decode-path integration itself (_forward_gathered_ep
            dequant-on-gather). That is Q4, and this file's PASS is its
            precondition, not a substitute for it.

Usage: python tests/test_moe_int8_ep_shard_commutativity.py
"""
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [ROOT]
    _pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg

from nanovllm.utils.loader import expert_local_slot, shard_experts_tensor  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_int8_quantize import (  # noqa: E402
    quantize_weight_int8_grouped,
    dequantize_weight_int8_grouped,
)

# Real dims (Q0), small synthetic expert count for speed. E=8 keeps every
# tp_size in {1,2,4} an exact divisor, matching this project's existing
# Q4/Checkpoint-2-style precedent of "real per-expert dims, small expert
# count" for CPU-tractable sharding tests.
NUM_EXPERTS = 8
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
GROUP_SIZE = 128
SEED = 4242


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def build_tagged_reference():
    """Each expert's gate_up_proj/down_proj slice is filled with a distinct
    base value plus small per-element noise -- distinct enough to make
    cross-expert contamination trivially detectable (same tagging technique
    as tests/test_expert_round_robin_loader.py), with enough per-element
    variation that quantization has real per-group dynamic range to work
    with (a perfectly flat tensor would trivially quantize with zero
    error, which would not exercise the group-boundary arithmetic at all)."""
    torch.manual_seed(SEED)
    gate_up = torch.empty(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)
    down = torch.empty(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.bfloat16)
    for e in range(NUM_EXPERTS):
        base = float(e + 1) * 0.01
        gate_up[e] = (torch.randn(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE) * 0.002 + base).to(torch.bfloat16)
        down[e] = (torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE) * 0.002 + base).to(torch.bfloat16)
    return gate_up, down


def test_commutativity(full_weight: torch.Tensor, name: str, tp_size: int):
    section(f"Commutativity: quantize-then-shard vs shard-then-quantize -- "
            f"{name}, tp_size={tp_size}")

    # PATH A: quantize the full tensor, then shard weight and scale using
    # the SAME real production function, applied to each.
    full_int8, full_scale = quantize_weight_int8_grouped(full_weight, GROUP_SIZE)

    all_ok = True
    for rank in range(tp_size):
        path_a_weight = shard_experts_tensor(full_int8, rank, tp_size)
        path_a_scale = shard_experts_tensor(full_scale, rank, tp_size)

        # PATH B: shard the full-precision tensor first (the REAL,
        # unmodified production function -- exactly what load_model()
        # already does for bf16 weights today), then quantize only this
        # rank's local slice.
        local_full_precision = shard_experts_tensor(full_weight, rank, tp_size)
        path_b_weight, path_b_scale = quantize_weight_int8_grouped(local_full_precision, GROUP_SIZE)

        weight_match = torch.equal(path_a_weight, path_b_weight)
        scale_match = torch.equal(path_a_scale, path_b_scale)

        status = "OK" if (weight_match and scale_match) else "MISMATCH"
        print(f"  rank {rank}: weight_bitwise_equal={weight_match}  "
              f"scale_bitwise_equal={scale_match}  shapes weight={tuple(path_a_weight.shape)} "
              f"scale={tuple(path_a_scale.shape)}  [{status}]")

        if not (weight_match and scale_match):
            all_ok = False

    assert all_ok, (
        f"Path A and Path B disagree for {name} at tp_size={tp_size}. "
        f"Quantization is NOT commuting with round-robin expert sharding -- "
        f"post-shard quantization (the simple integration path) is NOT "
        f"equivalent to pre-shard quantization here, and needs investigating "
        f"before Q4 integration proceeds on that assumption."
    )
    print(f"  [PASS] Path A == Path B, bitwise, at every rank for tp_size={tp_size}")


def test_negative_contamination(full_weight: torch.Tensor, name: str, tp_size: int):
    section(f"Negative contamination check -- {name}, tp_size={tp_size}")

    full_int8, full_scale = quantize_weight_int8_grouped(full_weight, GROUP_SIZE)

    shards = {}
    for rank in range(tp_size):
        shards[rank] = (
            shard_experts_tensor(full_int8, rank, tp_size),
            shard_experts_tensor(full_scale, rank, tp_size),
        )

    for rank in range(tp_size):
        rank_weight, rank_scale = shards[rank]
        owned = {e for e in range(NUM_EXPERTS) if e % tp_size == rank}
        forbidden = set(range(NUM_EXPERTS)) - owned

        for other_expert in forbidden:
            other_rank = other_expert % tp_size
            other_slot = expert_local_slot(other_expert, other_rank, tp_size)
            other_weight_slice = shards[other_rank][0][other_slot]
            other_scale_slice = shards[other_rank][1][other_slot]

            for local_slot in range(rank_weight.shape[0]):
                assert not torch.equal(rank_weight[local_slot], other_weight_slice), (
                    f"rank {rank}'s local slot {local_slot} matches expert "
                    f"{other_expert}'s weight, which belongs to rank {other_rank}"
                )
                assert not torch.equal(rank_scale[local_slot], other_scale_slice), (
                    f"rank {rank}'s local slot {local_slot} matches expert "
                    f"{other_expert}'s scale, which belongs to rank {other_rank}"
                )

    print(f"  [OK] no rank's shard contains another rank's expert data, "
          f"weight or scale, tp_size={tp_size}")
    print("  [PASS]")


def test_round_trip(full_weight: torch.Tensor, name: str, tp_size: int):
    section(f"Round-trip: scatter Path-B shards back, dequantize, compare "
            f"to direct full-tensor quantize+dequantize -- {name}, tp_size={tp_size}")

    reference_int8, reference_scale = quantize_weight_int8_grouped(full_weight, GROUP_SIZE)
    reference_dequant = dequantize_weight_int8_grouped(
        reference_int8, reference_scale, GROUP_SIZE, torch.bfloat16
    )

    reconstructed = torch.empty_like(full_weight)
    for rank in range(tp_size):
        local_full_precision = shard_experts_tensor(full_weight, rank, tp_size)
        local_int8, local_scale = quantize_weight_int8_grouped(local_full_precision, GROUP_SIZE)
        local_dequant = dequantize_weight_int8_grouped(
            local_int8, local_scale, GROUP_SIZE, torch.bfloat16
        )
        for local_slot in range(local_dequant.shape[0]):
            global_expert = None
            for e in range(NUM_EXPERTS):
                if expert_local_slot(e, rank, tp_size) == local_slot:
                    global_expert = e
                    break
            assert global_expert is not None
            reconstructed[global_expert] = local_dequant[local_slot]

    match = torch.equal(reconstructed, reference_dequant)
    print(f"  scattered-and-dequantized reconstruction == direct dequantize: {match}")
    assert match, (
        f"Round-trip mismatch for {name} at tp_size={tp_size} -- scattering "
        f"per-rank quantize+dequantize shards back together does not "
        f"reproduce quantizing and dequantizing the full tensor directly."
    )
    print("  [PASS]")


def main():
    gate_up, down = build_tagged_reference()

    for tp_size in (1, 2, 4):
        assert NUM_EXPERTS % tp_size == 0

        for name, full_weight in (("gate_up_proj", gate_up), ("down_proj", down)):
            test_commutativity(full_weight, name, tp_size)
            test_negative_contamination(full_weight, name, tp_size)
            test_round_trip(full_weight, name, tp_size)

    print("\n" + "=" * 74)
    print("ALL Q3 EP-SHARDING COMMUTATIVITY CHECKS PASSED")
    print("=" * 74)
    print("Conclusion: quantize-then-shard and shard-then-quantize are bitwise")
    print("identical, at every tp_size tested, for both expert tensors. The")
    print("integration path for Q4 is therefore: quantize AFTER the existing,")
    print("unmodified load_model()/shard_experts_tensor bf16 sharding path --")
    print("zero changes needed to production loader code, and no new sharding")
    print("logic for scales. Still requires: real-checkpoint precision (Q6)")
    print("and the decode-path dequant-on-gather integration itself (Q4).")


if __name__ == "__main__":
    main()