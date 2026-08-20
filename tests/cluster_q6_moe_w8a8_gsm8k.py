"""Q6 -- MoE INT8 (use_moe_w8a8) GSM8K non-regression, real weights, same
n=40 protocol as A4 (tests/cluster_a4_fused_gdr_gsm8k.py).

============================================================================
WHY THIS REUSES A4 RATHER THAN REBUILDING IT
============================================================================
_run_arm, _select_indices, SUBSAMPLE_SEED, MAX_TOKENS, MAX_MODEL_LEN,
DEFAULT_CKPT, and _maybe_install_fake_config_loader are imported directly
from cluster_a4_fused_gdr_gsm8k.py, not reimplemented. Two reasons:

  1. Same discipline this codebase already established (cluster_a3_ep_
     correctness.py imports REFERENCE_PATH/PROMPTS/etc. from cluster_a2_tp_
     correctness.py rather than redefining them) -- reuse a validated
     helper, don't risk a second, subtly different copy drifting from it.

  2. SUBSAMPLE_SEED being the SAME seed means this script's n=40 subset is
     the EXACT SAME 40 GSM8K examples A4 tested. That makes the two
     accumulation-precision findings (fused-GDR's 13/40 flips, and whatever
     this run finds) directly comparable example-for-example, not just
     comparable in aggregate rate -- if any index flips under BOTH
     interventions, that is worth knowing (a genuinely fragile decision
     point, sensitive to any reduction-order change), and this makes that
     question answerable by a simple index intersection, for free.

============================================================================
WHAT'S NEW HERE, NOT PRESENT IN A4
============================================================================
A4's script prints the raw per-example flip list but does NOT compute
McNemar's test -- the p*approx*0.58 figure already in the measurement
notebook for A4 was derived from that printed list in a separate step, not
produced by the script itself. That gap is closed here: mcnemar_exact_test()
below is a first-class part of this script, computed automatically, and
--verify-mcnemar lets it be re-run against ANY two cached result files
(including A4's own n40_tp2_fused_off/on.json) with zero GPU cost --
matching tests/gsm8k_rescore_fixed_extractor.py's existing pattern of
re-analyzing already-generated model_output text without regenerating it.
Running --verify-mcnemar against A4's own files is a cheap, immediate way
to confirm this script's McNemar code reproduces the already-cited ~0.58
figure before trusting it on a fresh quantization result.

============================================================================
WHAT THIS DOES AND DOES NOT CHECK
============================================================================
NOT an accuracy-bar test (same posture as A4) -- the 87.5% correctness gate
is a separate, already-passing check (95.30% under chat-no-think). This is
flag-off vs. flag-on non-regression on the SAME examples, matching A4's own
prompt format exactly (build_prompt, i.e. the 'raw' format A4 used, not the
production chat-no-think format) -- deliberately, for apples-to-apples
comparability with A4's own numbers, not because raw is the shipping
config. If a production-config (chat-no-think) accuracy comparison is
wanted separately, that is a different, follow-on script -- not silently
substituted here.

Exercises BOTH quantized forward paths: _forward_gathered_ep (decode, most
of the workload) and _forward_dispatch_ep (prefill, needed for every
example's first step) -- unlike Q4/A2's single-forward-pass check, which
only touched prefill in isolation.

Usage:
    python tests/cluster_q6_moe_w8a8_gsm8k.py --tp 2 --num-examples 40

Re-analyze existing results with no GPU (McNemar + flips only):
    python tests/cluster_q6_moe_w8a8_gsm8k.py --verify-mcnemar \\
        tests/_cluster_day_cache/a4_fused_gdr/n40_tp2_fused_off.json \\
        tests/_cluster_day_cache/a4_fused_gdr/n40_tp2_fused_on.json
    # ^ sanity check: should reproduce the already-cited p~=0.58 for A4

    python tests/cluster_q6_moe_w8a8_gsm8k.py --verify-mcnemar \\
        tests/_cluster_day_cache/q6_moe_w8a8/n40_tp2_w8a8_off.json \\
        tests/_cluster_day_cache/q6_moe_w8a8/n40_tp2_w8a8_on.json
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from time import perf_counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cluster_a4_fused_gdr_gsm8k import (  # noqa: E402
    _run_arm,
    _select_indices,
    SUBSAMPLE_SEED,
    MAX_TOKENS,
    MAX_MODEL_LEN,
    DEFAULT_CKPT,
    _maybe_install_fake_config_loader,
)

CACHE_DIR = os.path.join(ROOT, "tests", "_cluster_day_cache", "q6_moe_w8a8")


def mcnemar_exact_test(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs, stdlib-only (no
    scipy dependency). b = arm-A-wrong -> arm-B-correct count, c = the
    reverse. Under H0 (the two arms disagree only by symmetric noise, no
    systematic direction), each discordant pair is an independent coin
    flip, n = b+c ~ Binomial(n, 0.5).

    Two-sided exact p-value: 2 * P(X <= min(b,c)) under Binomial(n, 0.5),
    capped at 1.0 -- the standard exact-binomial form of McNemar's test,
    used here instead of the chi-squared approximation because n=40-scale
    discordant-pair counts (b+c typically well under 20) are too small for
    the chi-squared approximation to be trustworthy.
    """
    n = b + c
    if n == 0:
        return 1.0  # no discordant pairs at all -- nothing to test
    k = min(b, c)
    cdf = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    p = min(1.0, 2 * cdf)
    return p


def analyze(records_a, records_b, label_a: str, label_b: str):
    """Shared by both the live-run path and --verify-mcnemar. Reproduces
    A4's own accuracy / extraction-method-distribution / per-example-flip
    printing verbatim in spirit (same fields, same framing), then adds the
    McNemar computation A4's script never had."""
    by_idx_a = {r["idx"]: r for r in records_a}
    by_idx_b = {r["idx"]: r for r in records_b}
    common = sorted(set(by_idx_a) & set(by_idx_b))
    assert common == sorted(by_idx_a) == sorted(by_idx_b), "index mismatch between the two arms"

    n_a_correct = sum(1 for i in common if by_idx_a[i]["correct"])
    n_b_correct = sum(1 for i in common if by_idx_b[i]["correct"])
    acc_a = 100 * n_a_correct / len(common)
    acc_b = 100 * n_b_correct / len(common)
    print(f"\nACCURACY  {label_a}: {n_a_correct}/{len(common)} ({acc_a:.1f}%)   "
          f"{label_b}: {n_b_correct}/{len(common)} ({acc_b:.1f}%)   delta: {acc_b - acc_a:+.1f}pp")

    method_a = Counter(by_idx_a[i]["extraction_method"] for i in common)
    method_b = Counter(by_idx_b[i]["extraction_method"] for i in common)
    print("\nEXTRACTION-METHOD DISTRIBUTION")
    print(f"  {'method':<22} {label_a:>12} {label_b:>12}")
    for method in ["hash", "answer_is", "fallback_last_number", "failed"]:
        print(f"  {method:<22} {method_a.get(method, 0):>12} {method_b.get(method, 0):>12}")

    flipped = [i for i in common if by_idx_a[i]["correct"] != by_idx_b[i]["correct"]]
    b_to_a_correct = [i for i in flipped if not by_idx_a[i]["correct"] and by_idx_b[i]["correct"]]
    a_to_b_wrong = [i for i in flipped if by_idx_a[i]["correct"] and not by_idx_b[i]["correct"]]

    print(f"\nPER-EXAMPLE FLIPS (correct changed between arms): {len(flipped)}/{len(common)}")
    print(f"  {label_a}-wrong -> {label_b}-correct: {len(b_to_a_correct)}  {b_to_a_correct}")
    print(f"  {label_a}-correct -> {label_b}-wrong: {len(a_to_b_wrong)}  {a_to_b_wrong}")
    for i in flipped:
        print(f"  idx={i}: {label_a} correct={by_idx_a[i]['correct']} (method={by_idx_a[i]['extraction_method']})  "
              f"{label_b} correct={by_idx_b[i]['correct']} (method={by_idx_b[i]['extraction_method']})")

    p = mcnemar_exact_test(len(b_to_a_correct), len(a_to_b_wrong))
    print(f"\nMcNEMAR EXACT TEST (two-sided): b={len(b_to_a_correct)}  c={len(a_to_b_wrong)}  "
          f"n_discordant={len(flipped)}  p={p:.4f}")
    if len(flipped) == 0:
        print("  No discordant pairs at all -- strongest possible non-regression result.")
    elif p >= 0.05:
        print(f"  p={p:.4f} >= 0.05: no evidence of a systematic direction in the flips. "
              f"Consistent with symmetric noise (e.g. reduction-order-dependent bf16 "
              f"rounding), not a directional regression -- READ ALONGSIDE the accuracy "
              f"delta above, not as a substitute for it: a small p is not required for "
              f"a real problem to exist if the accuracy delta itself is large.")
    else:
        print(f"  p={p:.4f} < 0.05: the flip direction is NOT symmetric -- {label_b} is "
              f"systematically {'better' if len(b_to_a_correct) > len(a_to_b_wrong) else 'worse'} "
              f"than chance would predict on the discordant pairs. Treat as a real signal, "
              f"not noise, and investigate the flipped examples specifically.")

    return {
        "n": len(common), "acc_a": acc_a, "acc_b": acc_b,
        "n_flipped": len(flipped), "b": len(b_to_a_correct), "c": len(a_to_b_wrong),
        "mcnemar_p": p,
    }


def _build_examples(num_examples: int, dry_run: bool):
    if dry_run:
        return [{"idx": i, "question": f"What is {i} plus {i + 1}?", "answer": f"#### {2 * i + 1}"}
                for i in range(num_examples)]
    from datasets import load_dataset
    print("Loading openai/gsm8k (main, test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if len(ds) != 1319:
        print(f"STOP: expected 1319 test examples, got {len(ds)}. Not proceeding.")
        sys.exit(1)
    indices = _select_indices(len(ds), num_examples)
    examples = [{"idx": i, "question": ds[i]["question"], "answer": ds[i]["answer"]} for i in indices]
    print(f"Subsampled {len(examples)}/{len(ds)} examples (seed={SUBSAMPLE_SEED}, SAME seed as A4 "
          f"-> same 40 examples as the fused-GDR check): {indices}")
    return examples


def _run_and_cache_q6(args, examples, use_w8a8: bool, results_path: str):
    if os.path.exists(results_path):
        print(f"  [{'on' if use_w8a8 else 'off'}] results already cached at {results_path}, "
              f"skipping run (delete the file to force a re-run)")
        with open(results_path, encoding="utf-8") as f:
            return json.load(f)

    _maybe_install_fake_config_loader(args)
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    tag = "on" if use_w8a8 else "off"
    print(f"\n[arm={tag}] loading engine (tensor_parallel_size={args.tp}, "
          f"use_moe_w8a8={use_w8a8}, group_size={args.moe_w8a8_group_size}) "
          f"from {args.checkpoint} ...")
    llm = LLM(
        args.checkpoint,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=8,
        max_model_len=MAX_MODEL_LEN,
        use_moe_w8a8=use_w8a8,
        moe_w8a8_weight_group_size=args.moe_w8a8_group_size,
    )
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    t0 = perf_counter()
    records = _run_arm(llm, sp, examples, args.dry_run)
    elapsed = perf_counter() - t0
    print(f"[arm={tag}] {len(records)} example(s) in {elapsed:.1f}s")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[arm={tag}] saved to {results_path}")

    import atexit
    atexit.unregister(llm.exit)
    llm.exit()
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-mcnemar", nargs=2, metavar=("PATH_A", "PATH_B"), default=None,
                     help="Re-run the analysis (accuracy, extraction distribution, flips, "
                          "McNemar) on two ALREADY-CACHED result json files, no GPU/LLM "
                          "construction at all. PATH_A is treated as the baseline/off arm.")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--num-examples", type=int, default=40,
                     help="Same default and SAME seed as A4 -- same 40 examples.")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--moe-w8a8-group-size", type=int, default=128)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--fake-config-loader", action="store_true", default=False)
    args = ap.parse_args()

    if args.verify_mcnemar:
        path_a, path_b = args.verify_mcnemar
        with open(path_a, encoding="utf-8") as f:
            records_a = json.load(f)
        with open(path_b, encoding="utf-8") as f:
            records_b = json.load(f)
        print("\n" + "=" * 78)
        print(f"RE-ANALYSIS (no GPU) -- {path_a}  vs.  {path_b}")
        print("=" * 78)
        analyze(records_a, records_b, "A(off)", "B(on)")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    examples = _build_examples(args.num_examples, args.dry_run)

    off_path = os.path.join(CACHE_DIR, f"n{args.num_examples}_tp{args.tp}_w8a8_off.json")
    on_path = os.path.join(CACHE_DIR, f"n{args.num_examples}_tp{args.tp}_w8a8_on.json")

    records_off = _run_and_cache_q6(args, examples, use_w8a8=False, results_path=off_path)
    records_on = _run_and_cache_q6(args, examples, use_w8a8=True, results_path=on_path)

    print("\n" + "=" * 78)
    print(f"Q6 NON-REGRESSION -- MoE INT8 W8A8 flag-off vs. flag-on, "
          f"n={len(examples)}, tp={args.tp}, group_size={args.moe_w8a8_group_size}")
    print("=" * 78)

    if args.dry_run:
        by_idx_off = {r["idx"]: r for r in records_off}
        by_idx_on = {r["idx"]: r for r in records_on}
        common = sorted(set(by_idx_off) & set(by_idx_on))
        n_finite = sum(1 for i in common if by_idx_off[i]["model_output"] and by_idx_on[i]["model_output"])
        print(f"DRY-RUN CHECK: {n_finite}/{len(common)} examples produced non-empty output in BOTH arms.")
        assert n_finite == len(common)
        print("DRY-RUN CHECK: PASS")
        return

    analyze(records_off, records_on, "off", "on")


if __name__ == "__main__":
    main()