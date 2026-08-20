"""EP rank-imbalance simulation at ep_size 4 and 8 -- offline, from real
per-expert routing counts already measured at ep_size=2.

============================================================================
WHAT THIS IS, AND WHAT IT ISN'T
============================================================================
tests/cluster_a3_ep_correctness.py --phase histogram measured, on the real
checkpoint with real GSM8K routing, how many (token, k) pairs land on each
of the 256 experts, per layer. That measurement is a property of the MODEL
and the WORKLOAD -- it does not depend on how many ranks the experts are
later sharded across. This script re-shards those same per-expert counts
under round-robin (expert_id % P) at P = 4 and P = 8, and reports the
resulting per-rank imbalance.

This is SIMULATION, not a new measurement of the model. It answers "if we
ran this exact routing at ep_size=4 or 8, how imbalanced would dropless
dispatch be" using data already on disk. No GPU, no new forward passes.

VALIDATION: at P=2 this script's own re-sharding must reproduce the
per-rank totals recorded in the source JSON's `rank_counts` field exactly
(int for int, on every layer). That is not incidental -- it is the check
that this script's round-robin implementation matches the one the engine
actually uses (utils/loader.py's expert_id % tp_size convention), before
its P=4/P=8 output is trusted at all.

Source: tests/_cluster_day_cache/a3_ep/histogram_real_ep2.json (rerun with
the `per_expert_counts` field added -- see git log / commit message for the
one-line patch to cluster_a3_ep_correctness.py that produced it).

Usage: python tests/simulate_ep_imbalance.py
"""
import json
import os
import sys

HISTOGRAM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "_cluster_day_cache", "a3_ep", "histogram_real_ep2.json",
)


def reshard(per_expert_counts: list[int], ep_size: int) -> list[int]:
    """Round-robin re-sharding: expert e is owned by rank e % ep_size.
    Matches utils/loader.py's expert_local_slot convention (expert_id % P) --
    this is the SAME assignment rule the engine's real EP dispatch uses,
    not an independent approximation of it."""
    rank_totals = [0] * ep_size
    for expert_id, count in enumerate(per_expert_counts):
        rank_totals[expert_id % ep_size] += count
    return rank_totals


def max_over_mean(rank_totals: list[int]) -> float:
    total = sum(rank_totals)
    if total == 0:
        return float("nan")
    mean = total / len(rank_totals)
    return max(rank_totals) / mean


def validate_against_measured_ep2(per_layer: list[dict]) -> None:
    """Re-shard at P=2 and require an EXACT match against the measured
    rank_counts already in the source file. If this does not hold, nothing
    below is trustworthy -- the re-sharding logic disagrees with whatever
    produced the original ep_size=2 numbers, for reasons that need
    resolving before P=4/P=8 output means anything."""
    print("=" * 78)
    print("VALIDATION: re-sharding at P=2 must exactly reproduce the")
    print("measured rank_counts already recorded for ep_size=2")
    print("=" * 78)

    mismatches = 0
    for layer in per_layer:
        computed = reshard(layer["per_expert_counts"], 2)
        measured = layer["rank_counts"]
        if computed != measured:
            mismatches += 1
            print(f"  [MISMATCH] layer {layer['layer_idx']}: "
                  f"computed={computed}  measured={measured}")

    if mismatches:
        print(f"\n  [FAIL] {mismatches}/{len(per_layer)} layers disagree.")
        print("  Stopping -- P=4/P=8 numbers below would not be trustworthy.")
        sys.exit(1)

    print(f"  [OK] all {len(per_layer)} layers match exactly.")
    print("  [PASS] round-robin re-sharding logic confirmed correct.\n")


def main():
    if not os.path.isfile(HISTOGRAM_PATH):
        print(f"ERROR: {HISTOGRAM_PATH} not found.")
        print("Run: python tests/cluster_a3_ep_correctness.py --phase histogram --tp 2")
        sys.exit(1)

    with open(HISTOGRAM_PATH, encoding="utf-8") as f:
        data = json.load(f)

    per_layer = data["per_layer"]
    num_experts = data["num_experts"]
    top_k = data["top_k"]
    n_prompts = data["n_prompts"]

    if "per_expert_counts" not in per_layer[0]:
        print("ERROR: source JSON has no per_expert_counts field.")
        print("This is the pre-patch histogram file. Re-run cluster_a3_ep_correctness.py")
        print("--phase histogram after adding per_expert_counts to the saved summary.")
        sys.exit(1)

    print(f"Source: {HISTOGRAM_PATH}")
    print(f"num_experts={num_experts}  top_k={top_k}  n_prompts={n_prompts}  "
          f"layers={len(per_layer)}\n")

    validate_against_measured_ep2(per_layer)

    for ep_size in (2, 4, 8):
        if num_experts % ep_size != 0:
            print(f"ep_size={ep_size}: SKIPPED -- {num_experts} not divisible by {ep_size}")
            continue

        print("=" * 78)
        print(f"ep_size = {ep_size}  ({num_experts // ep_size} experts/rank)")
        print("=" * 78)

        ratios = []
        worst_layer = None
        worst_ratio = 0.0
        over_120 = 0

        for layer in per_layer:
            counts = layer["per_expert_counts"]
            totals = reshard(counts, ep_size)
            ratio = max_over_mean(totals)
            ratios.append(ratio)
            if ratio > 1.20:
                over_120 += 1
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_layer = (layer["layer_idx"], totals)

        mean_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)

        print(f"  max/mean across 40 layers: min={min_ratio:.3f}  "
              f"mean={mean_ratio:.3f}  max={max_ratio:.3f}")
        print(f"  layers with max/mean > 1.20: {over_120}/{len(per_layer)}")
        print(f"  worst layer: {worst_layer[0]}  totals={worst_layer[1]}  "
              f"ratio={worst_ratio:.3f}")

        if ep_size > 2:
            # Idle-time framing: dropless dispatch means every rank waits for
            # the slowest. A ratio of R means the lightest-loaded rank sits
            # idle for roughly (R-1)/R of that layer's compute, assuming
            # per-token compute cost is uniform across experts (it is not
            # exactly -- MoE intermediate size is fixed per expert here, so
            # this is a reasonable proxy, not an exact latency prediction).
            worst_idle_frac = (worst_ratio - 1.0) / worst_ratio
            print(f"  worst-layer idle-time proxy for lightest rank: "
                  f"~{worst_idle_frac * 100:.1f}%")

        print()

    print("=" * 78)
    print("READING")
    print("=" * 78)
    print("""
This is simulation over the SAME real-checkpoint, real-GSM8K routing
already measured at ep_size=2 -- not a new measurement of a different
workload. The per-expert distribution is fixed; only the rank assignment
changes.

If the max/mean ratio and the over-1.20 layer count both increase from
ep_size=2 to ep_size=4 to ep_size=8, that confirms the expected mechanism:
fewer experts per rank under round-robin means less averaging, so a rank
that happens to draw a few heavily-used experts is proportionally more
exposed. Since dispatch is dropless, every rank waits for the slowest one,
so this ratio is a direct proxy for wasted compute time at that ep_size.

This does not by itself tell you which ep_size to run at tp=4 or on H200 --
that also depends on communication cost, which this simulation does not
model. It tells you what the load-imbalance COST of a given ep_size is,
holding communication fixed, which is the piece that was previously
unmeasured and stated only as a hypothesis.
""")


if __name__ == "__main__":
    main()