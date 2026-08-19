"""A5 -- reduced concurrency sweep {1,2,4,8,16}, FUNCTIONAL, at
tensor_parallel_size=4 (or 2). Purpose is narrower than it looks: prove the
sweep HARNESS works correctly at rank 4 on real hardware, so the eventual
H200 sweep is a scale-up of something known-working, not a first run of both
"does the harness work" and "what's the number" at once.

TREAT EVERY tok/s NUMBER FROM THIS SCRIPT AS FUNCTIONAL VALIDATION, NOT A
RESULT (task's own framing, repeated here so it can't be missed downstream):
A6000 is Ampere with PHB interconnect, no NVLink -- TP/EP scaling measured
here says very little about NVLink-connected H200s. Do not cite these
numbers as "the" throughput result for this engine.

Reuses bench_throughput.py's build_engine()/make_prompts()/run_trial()
directly (imported, not re-derived) -- that script already supports
--tensor-parallel-size and already hardcodes SamplingParams(ignore_eos=True)
in run_trial(), so the "use --ignore-eos so completions actually reach the
1024-token target" requirement is ALREADY satisfied for the internal-engine
path this script uses (--ignore-eos as a named flag exists on
bench_http_concurrency.py, the separate HTTP-server sweep script; this one
talks to the engine directly, no server process to manage on a
one-day cluster window).

What this ADDS on top of bench_throughput.py's own CSV logging:
  - An explicit PASS/FAIL gate per level: did it complete without
    crashing/hanging, did every trial produce tok_s > 0 and finite, and
    (since ignore_eos=True) did every completion actually reach
    --output-len tokens -- catching the exact failure mode README.md
    documents for the ORIGINAL (pre-ignore_eos) throughput numbers: real
    EOS silently truncating completions and suppressing the reported rate
    without looking like a failure.
  - Results written incrementally (one JSON line per completed level,
    flushed immediately) -- survivable if killed mid-sweep, unlike waiting
    for the whole --concurrency list to finish before any verdict exists.
  - A final PASS/FAIL summary instead of only a CSV of numbers to interpret
    by eye under cluster-day time pressure.

Note (per README.md / bench_throughput.py's own docstring): Phase 2 batching
has a still-open cosine~=0.947 item that may share a root cause with this
task's Part C preemption finding (see that report) -- both are read here as
reasons to treat this sweep's numbers as FUNCTIONAL VALIDATION ONLY, exactly
the posture this script already takes for the A6000-vs-H200 reason above,
not a new blocker to work around.

Usage:
    python tests/cluster_a5_concurrency_sweep.py --checkpoint /path/to/real/checkpoint --tp 4
    python tests/cluster_a5_concurrency_sweep.py --checkpoint /path/to/real/checkpoint --tp 2

Dry run (small model, single GPU). PREREQUISITE, run once (needs CUDA)
before the first dry run -- see cluster_a2_tp_correctness.py's module
docstring for the full writeup:
    python tests/make_fake_hf_config.py     # already done if config.json exists
    python tests/make_fake_checkpoint.py    # writes model.safetensors -- REQUIRED
    python tests/make_fake_tokenizer.py     # already done if tokenizer.json exists
Without it every parameter is torch.empty() garbage rather than merely
untrained -- bench_throughput.py's own build_engine() has no finiteness
check either, so a missing checkpoint here would silently "pass" on
meaningless numbers instead of failing loudly.

    python tests/cluster_a5_concurrency_sweep.py --checkpoint tests/fake_qwen35_small --tp 1 \\
        --levels 1 2 4 --prompt-len 32 --output-len 64 --gpu-memory-utilization 0.2
"""
import argparse
import json
import math
import os
import sys
import types
from time import perf_counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bench_throughput as bt  # noqa: E402 -- reused directly, see module docstring

DEFAULT_CKPT = os.path.join(ROOT, "qwen35_checkpoint")
DEFAULT_OUT = os.path.join(ROOT, "tests", "_cluster_day_cache", "a5_concurrency_sweep.jsonl")


class _Args:
    """Plain attribute bag matching what bench_throughput.build_engine()
    reads off argparse.Namespace -- constructed here instead of calling
    bench_throughput.main() so this script controls the per-level loop,
    incremental writes, and PASS/FAIL gate itself."""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tp", type=int, default=4, dest="tensor_parallel_size")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--warmup-trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-fake-config-loader", dest="fake_config_loader", action="store_false", default=True,
                     help="Pass this against the REAL checkpoint (registered config, no monkeypatch "
                          "needed). Default True is for the small fake model's unregistered model_type.")
    cli = ap.parse_args()

    args = _Args()
    args.model = cli.checkpoint
    args.max_num_batched_tokens = cli.max_num_batched_tokens
    args.concurrency = cli.levels
    args.max_model_len = cli.max_model_len
    args.gpu_memory_utilization = cli.gpu_memory_utilization
    args.tensor_parallel_size = cli.tensor_parallel_size
    args.enforce_eager = False
    args.use_fused_gdr_kernel = True
    args.fake_config_loader = cli.fake_config_loader

    print(f"Building engine: tensor_parallel_size={args.tensor_parallel_size} (== ep_size) "
          f"from {args.model} ...")
    engine = bt.build_engine(args)

    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    print(f"Results will be appended incrementally to: {cli.out}")

    level_results = []
    try:
        with open(cli.out, "a", encoding="utf-8") as out_fh:
            for level in cli.levels:
                print(f"\n=== concurrency={level} ===")
                for w in range(cli.warmup_trials):
                    r = bt.run_trial(engine, level, cli.prompt_len, cli.output_len, seed=cli.seed)
                    print(f"  warmup {w + 1}/{cli.warmup_trials}: wall_s={r['wall_s']:.2f} "
                          f"tok/s={r['tok_s']:.1f} (discarded)")

                trial_records = []
                for t in range(cli.trials):
                    t0 = perf_counter()
                    r = bt.run_trial(engine, level, cli.prompt_len, cli.output_len, seed=cli.seed + t + 1)
                    dt = perf_counter() - t0
                    finite = math.isfinite(r["tok_s"]) and r["tok_s"] > 0
                    reached_target = all(n == cli.output_len for n in r["completion_lengths"])
                    ok = finite and reached_target
                    print(f"  trial {t + 1}/{cli.trials}: wall_s={r['wall_s']:.2f} "
                          f"tok/s={r['tok_s']:.1f} finite={finite} "
                          f"all_reached_output_len={reached_target} -> {'OK' if ok else 'ANOMALY'} "
                          f"(harness call took {dt:.2f}s)")
                    if not reached_target:
                        print(f"    completion_lengths={r['completion_lengths']} (expected all == "
                              f"{cli.output_len} -- ignore_eos=True is hardcoded in "
                              f"bench_throughput.run_trial(), so a short completion here means "
                              f"something OTHER than EOS truncated it: max_tokens misconfigured, "
                              f"a stop condition firing unexpectedly, or a real bug)")
                    trial_records.append({"trial": t, "wall_s": r["wall_s"], "tok_s": r["tok_s"],
                                           "completion_lengths": r["completion_lengths"], "ok": ok})

                level_tok_s = [tr["tok_s"] for tr in trial_records]
                level_ok = all(tr["ok"] for tr in trial_records)
                mean_tok_s = sum(level_tok_s) / len(level_tok_s)
                result = {
                    "concurrency": level, "tensor_parallel_size": args.tensor_parallel_size,
                    "mean_tok_s": mean_tok_s, "trials": trial_records, "passed": level_ok,
                }
                level_results.append(result)
                out_fh.write(json.dumps(result) + "\n")
                out_fh.flush()
                print(f"  concurrency={level}: mean tok/s over {cli.trials} trials = {mean_tok_s:.1f} "
                      f"-> {'PASS' if level_ok else 'FAIL'}")
    finally:
        engine.exit()

    print("\n" + "=" * 78)
    print(f"A5 SUMMARY -- tensor_parallel_size={args.tensor_parallel_size} "
          f"(FUNCTIONAL VALIDATION ONLY -- see module docstring, NOT an H200-predictive number)")
    print("=" * 78)
    print(f"{'concurrency':>11}  {'verdict':>7}  {'mean_tok_s':>10}")
    for r in level_results:
        print(f"{r['concurrency']:>11}  {'PASS' if r['passed'] else 'FAIL':>7}  {r['mean_tok_s']:>10.1f}")

    all_passed = all(r["passed"] for r in level_results)
    increasing = all(
        level_results[i]["mean_tok_s"] <= level_results[i + 1]["mean_tok_s"] * 1.0
        for i in range(len(level_results) - 1)
    ) if len(level_results) > 1 else True
    print()
    print(f"All levels functionally PASS: {all_passed}")
    if not increasing:
        print("NOTE: tok/s did not monotonically increase with concurrency -- on Ampere/PHB "
              "hardware without NVLink this is plausible and NOT automatically a bug (see module "
              "docstring), but worth a second look before assuming it's expected.")
    print(f"\nVERDICT: {'PASS' if all_passed else 'FAIL'} -- harness is "
          f"{'' if all_passed else 'NOT '}confirmed working at tensor_parallel_size="
          f"{args.tensor_parallel_size} through rank {max(cli.levels)}-way concurrency. "
          f"{'Ready to scale up to the H200 sweep.' if all_passed else 'Fix before scaling up.'}")
    print(f"\nFull per-trial record (including completion_lengths, for the exact truncation check "
          f"above): {cli.out}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
