"""tp=4 KV-cache SIZING coverage -- the one site tests/test_gqa_kv_replication_tp4.py
explicitly declines to exercise.

============================================================================
WHAT THIS COVERS, AND WHAT IT DOES NOT -- read before citing it
============================================================================
tests/test_gqa_kv_replication_tp4.py validates the three GQA-replication
sites at the level of shard math, weight-loader selection, and real
multi-process module construction. Its own docstring records that
ModelRunner.allocate_kv_cache() is NOT exercised directly -- only
local_num_kv_heads(), the helper it calls, is. That leaves the KV-cache
SIZING arithmetic at tp=4 never executed anywhere.

This file closes exactly that gap, and nothing wider:

  COVERED   the block_bytes / num_kvcache_blocks / kv_cache-shape
            arithmetic of allocate_kv_cache(), replicated line-for-line
            against a fake config, at tp = 1, 2 and 4.
  COVERED   that the arithmetic reproduces the REAL tp=2 values measured
            on the real checkpoint (see MEASURED_TP2 below) -- so the
            replication below is anchored to a known-good observation,
            not just to itself.
  NOT       allocate_kv_cache() itself. It reads torch.cuda.mem_get_info()
            and torch.cuda.memory_stats(), and wires k_cache/v_cache onto
            live attention modules -- neither is reachable without CUDA and
            a constructed model. This file reimplements the arithmetic; it
            does not call the function.
  NOT       anything about attention numerics, NCCL, or cross-rank
            behaviour at tp=4. Still requires 4 GPUs.

The replication risk is real and worth stating: if allocate_kv_cache()'s
formula changes and this file's copy does not, this test passes while
testing a formula that is no longer in use. That is mitigated by
MEASURED_TP2 (a real observed tuple that both must reproduce) but not
eliminated. If you change the formula in engine/model_runner.py, change it
here in the same commit.

============================================================================
THE BUG THIS GUARDS AGAINST, stated accurately
============================================================================
Before the GQA-replication fix, allocate_kv_cache() computed

    num_kv_heads = hf_config.num_key_value_heads // self.world_size

which at num_key_value_heads=2, world_size=4 gives 0. That makes
block_bytes == 0, and the next line divides by it:

    config.num_kvcache_blocks = int(...) // block_bytes   # ZeroDivisionError

So the pre-fix failure mode was a LOUD ZeroDivisionError, not a silently
allocated zero-width cache -- an earlier note in the planning docs
described it as silent, and that was wrong. It is recorded correctly here
because the distinction matters: this site belonged to the "would have
crashed confusingly" category, not the "silent wrong state" category that
ARCH_DISPATCH and load_model()'s key-prefix skip belong to.

In practice it was never even reached at tp=4: Qwen35FullAttention's own
`assert num_kv_heads % tp_size == 0` fired first, during model
construction, which is the 9.5s/1776MiB failure recorded in the cluster-day
measurements.

Usage: python tests/test_gqa_kv_cache_sizing_tp4.py
"""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.linear import local_num_kv_heads


# ---------------------------------------------------------------------------
# Real-checkpoint geometry. Values are the actual Qwen3.5-35B-A3B config.
# ---------------------------------------------------------------------------
NUM_KEY_VALUE_HEADS = 2
NUM_ATTENTION_HEADS = 16
HEAD_DIM = 256
NUM_HIDDEN_LAYERS = 40
FULL_ATTENTION_INTERVAL = 4
NUM_KV_LAYERS = NUM_HIDDEN_LAYERS // FULL_ATTENTION_INTERVAL  # 10 full-attention layers
BLOCK_SIZE = 256
DTYPE = torch.bfloat16

# Observed on the real checkpoint at tp=2, from
# tests/_cluster_day_cache/logs/decode_profile_tp2_eager.log:
#   [KV DEBUG] num_kv_layers(full-attn only)=10
#   [KV DEBUG] block_bytes (per block, per layer count of 10)=2621440
#   [KV DEBUG] num_kvcache_blocks=2192
#   [KV DEBUG] kv_cache tensor shape=(2, 10, 2192, 256, 1, 256)
# The `1` in that shape is local_num_kv_heads(2, 2) -- the whole point.
MEASURED_TP2 = {
    "num_kv_layers": 10,
    "block_bytes": 2621440,
    "num_kvcache_blocks": 2192,
    "kv_cache_shape": (2, 10, 2192, 256, 1, 256),
}


def compute_block_bytes(num_kv_heads: int) -> int:
    """Line-for-line copy of engine/model_runner.py:allocate_kv_cache()'s
    block_bytes computation. Kept as its own function so the tp-independent
    part of the formula is stated once."""
    return 2 * NUM_KV_LAYERS * BLOCK_SIZE * num_kv_heads * HEAD_DIM * DTYPE.itemsize


def compute_num_blocks(budget_bytes: int, block_bytes: int) -> int:
    """allocate_kv_cache() computes
        int(total * gpu_memory_utilization - used - peak + current) // block_bytes
    The bracketed term is a pure CUDA memory query with no tp dependence, so
    it is passed in here as `budget_bytes` rather than faked -- what is under
    test is the division and the block_bytes that feeds it."""
    return budget_bytes // block_bytes


def old_formula(total_num_kv_heads: int, world_size: int) -> int:
    """The pre-fix computation, kept so the regression is demonstrated
    rather than asserted from memory."""
    return total_num_kv_heads // world_size


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------

def test_local_kv_heads_across_tp():
    section("num_kv_heads used for KV-cache sizing, tp = 1 / 2 / 4")

    expected = {1: 2, 2: 1, 4: 1}
    for tp in (1, 2, 4):
        got = local_num_kv_heads(NUM_KEY_VALUE_HEADS, tp)
        old = old_formula(NUM_KEY_VALUE_HEADS, tp)
        note = ""
        if tp <= NUM_KEY_VALUE_HEADS:
            assert got == old, (
                f"tp={tp}: new helper ({got}) disagrees with the old formula "
                f"({old}) in the sharding regime -- this must be a no-op change "
                f"for tp <= num_key_value_heads, or every tp=1/tp=2 number ever "
                f"measured is invalidated."
            )
            note = f"(matches old formula {old} -- sharding regime, unchanged)"
        else:
            assert old == 0, f"expected the old formula to degenerate at tp={tp}"
            note = f"(old formula gave {old} -- the bug; now replicates 1 whole head)"
        assert got == expected[tp], f"tp={tp}: expected {expected[tp]}, got {got}"
        print(f"  tp={tp}: local_num_kv_heads({NUM_KEY_VALUE_HEADS},{tp}) = {got}  {note}")

    print("  [PASS]")


def test_block_bytes_nonzero_and_matches_measurement():
    section("block_bytes: nonzero at every tp, and reproduces the measured tp=2 value")

    for tp in (1, 2, 4):
        nkv = local_num_kv_heads(NUM_KEY_VALUE_HEADS, tp)
        bb = compute_block_bytes(nkv)
        assert bb > 0, (
            f"tp={tp}: block_bytes == 0. This is the pre-fix failure -- the next "
            f"line in allocate_kv_cache() divides by it."
        )
        print(f"  tp={tp}: num_kv_heads={nkv}  block_bytes={bb}")

    measured = MEASURED_TP2["block_bytes"]
    computed = compute_block_bytes(local_num_kv_heads(NUM_KEY_VALUE_HEADS, 2))
    assert computed == measured, (
        f"tp=2 block_bytes computed here ({computed}) does not match the value "
        f"observed on the real checkpoint ({measured}). Either this file's copy "
        f"of the formula has drifted from engine/model_runner.py, or the geometry "
        f"constants at the top of this file are wrong. Both are worth stopping for."
    )
    print(f"  [OK] tp=2 block_bytes == {measured}, matching the real-checkpoint log")

    print("  [PASS]")


def test_old_formula_would_have_raised_at_tp4():
    section("Regression demonstration: the pre-fix formula at tp=4")

    old_nkv = old_formula(NUM_KEY_VALUE_HEADS, 4)
    assert old_nkv == 0
    old_bb = compute_block_bytes(old_nkv)
    assert old_bb == 0
    print(f"  old num_kv_heads = 2 // 4 = {old_nkv}  ->  block_bytes = {old_bb}")

    raised = False
    try:
        compute_num_blocks(10_000_000_000, old_bb)
    except ZeroDivisionError:
        raised = True
    assert raised, (
        "Expected ZeroDivisionError from the pre-fix path. If this no longer "
        "raises, the formula has changed and this test's premise needs revisiting."
    )
    print("  [OK] pre-fix path raises ZeroDivisionError -- a LOUD failure, not a")
    print("       silently zero-width cache. Recorded accurately: this site is NOT")
    print("       in the same 'silent wrong state' category as ARCH_DISPATCH's")
    print("       default fallback or load_model()'s continue-on-unmatched-key.")

    new_bb = compute_block_bytes(local_num_kv_heads(NUM_KEY_VALUE_HEADS, 4))
    blocks = compute_num_blocks(10_000_000_000, new_bb)
    assert blocks > 0
    print(f"  [OK] fixed path: block_bytes={new_bb}, blocks={blocks} > 0")

    print("  [PASS]")


def test_kv_cache_shape_and_allocatability():
    section("kv_cache tensor shape at tp = 1 / 2 / 4 (allocated small, on CPU)")

    # A deliberately tiny block count: what is under test is the SHAPE the
    # formula produces and that torch accepts it, not real capacity.
    tiny_blocks = 4

    for tp in (1, 2, 4):
        nkv = local_num_kv_heads(NUM_KEY_VALUE_HEADS, tp)
        shape = (2, NUM_KV_LAYERS, tiny_blocks, BLOCK_SIZE, nkv, HEAD_DIM)
        kv = torch.empty(*shape, dtype=DTYPE, device="cpu")
        assert kv.shape[4] == nkv >= 1, (
            f"tp={tp}: kv-head dimension is {kv.shape[4]}. A zero here is the "
            f"degenerate cache this test exists to prevent."
        )
        assert kv.numel() > 0, f"tp={tp}: allocated a zero-element KV cache"
        print(f"  tp={tp}: shape={tuple(kv.shape)}  numel={kv.numel()}")
        del kv

    print("  [OK] kv-head dimension is >= 1 at every tp; no zero-width allocation")

    # Anchor the shape against the real measurement.
    nkv2 = local_num_kv_heads(NUM_KEY_VALUE_HEADS, 2)
    reconstructed = (
        2, NUM_KV_LAYERS, MEASURED_TP2["num_kvcache_blocks"], BLOCK_SIZE, nkv2, HEAD_DIM
    )
    assert reconstructed == MEASURED_TP2["kv_cache_shape"], (
        f"Reconstructed tp=2 shape {reconstructed} != observed "
        f"{MEASURED_TP2['kv_cache_shape']}"
    )
    print(f"  [OK] reconstructed tp=2 shape == {MEASURED_TP2['kv_cache_shape']},")
    print("       matching the real-checkpoint log exactly")

    print("  [PASS]")


def test_per_rank_kv_cost_does_not_shrink_from_tp2_to_tp4():
    section("Per-rank KV cost: tp=2 vs tp=4 (the expectation worth correcting)")

    bb2 = compute_block_bytes(local_num_kv_heads(NUM_KEY_VALUE_HEADS, 2))
    bb4 = compute_block_bytes(local_num_kv_heads(NUM_KEY_VALUE_HEADS, 4))

    assert bb2 == bb4, (
        f"block_bytes changed between tp=2 ({bb2}) and tp=4 ({bb4}). If this "
        f"assertion ever fails the KV-memory story in the planning docs needs "
        f"rewriting -- it currently states that per-rank KV cost is identical."
    )
    print(f"  tp=2 block_bytes = {bb2}")
    print(f"  tp=4 block_bytes = {bb4}")
    print("  [OK] IDENTICAL -- each rank holds one whole kv head in both regimes,")
    print("       so per-rank KV-cache cost per block does NOT halve at tp=4.")
    print("       tp=4's headroom comes from sharding WEIGHTS (~35 -> ~17.5 GB/rank),")
    print("       not from the cache. num_kvcache_blocks will still rise at tp=4,")
    print("       but only because freed weight memory enlarges the budget the")
    print("       block count is divided out of.")

    print("  [PASS]")


def main():
    test_local_kv_heads_across_tp()
    test_block_bytes_nonzero_and_matches_measurement()
    test_old_formula_would_have_raised_at_tp4()
    test_kv_cache_shape_and_allocatability()
    test_per_rank_kv_cost_does_not_shrink_from_tp2_to_tp4()

    print("\n" + "=" * 74)
    print("ALL tp=4 KV-CACHE SIZING CHECKS PASSED")
    print("=" * 74)
    print("Scope reminder: this covers allocate_kv_cache()'s ARITHMETIC only.")
    print("The function itself, NCCL, cross-rank dispatch, and real attention")
    print("numerics at tp=4 remain untested and still require 4 GPUs.")


if __name__ == "__main__":
    main()