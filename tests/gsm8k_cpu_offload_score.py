"""Comparison table for tests/gsm8k_cpu_offload_run.py's 3-arm output.

Standalone, no GPU/model dependency -- reads the JSONL files each arm writes
(cpu_offload_n{N}_{arm}.jsonl) and prints per-arm exact-match accuracy and
extraction-method breakdown side by side, plus the engine-path baseline
(full_results_n32_mt706_stop.jsonl's 'raw' arm) for reference where available.

Safe to run against PARTIAL results (a run still in progress, or killed
partway) -- each arm is reported independently with its own completed count,
never padded or assumed complete.

Usage:
    python tests/gsm8k_cpu_offload_score.py
    python tests/gsm8k_cpu_offload_score.py --output-dir tests/_gsm8k_cache/cpu_offload/ --num-examples 32
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "tests", "_gsm8k_cache", "cpu_offload")
ENGINE_BASELINE_32_PATH = os.path.join(ROOT, "tests", "_gsm8k_cache", "full_results_n32_mt706_stop.jsonl")
ARM_CHOICES = ["raw", "chat-think", "chat-no-think"]
EXTRACTION_METHODS = ["hash", "answer_is", "fallback_last_number", "failed"]


def load_records(path: str) -> list:
    if not os.path.exists(path):
        return []
    records = []
    seen_idx = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record["idx"] in seen_idx:
                continue
            seen_idx.add(record["idx"])
            records.append(record)
    return records


def summarize(label: str, path: str, records: list, expected_total: int):
    n = len(records)
    print(f"--- {label} ---")
    print(f"  file: {path}")
    if n == 0:
        print("  (no results yet)\n")
        return None
    n_correct = sum(1 for r in records if r["correct"])
    pct = 100 * n_correct / n
    is_partial = n < expected_total
    status = f"PARTIAL {n}/{expected_total}" if is_partial else f"COMPLETE {n}/{expected_total}"
    print(f"  status: {status}")
    print(f"  exact-match accuracy: {n_correct}/{n} ({pct:.1f}%)")
    print("  by extraction method:")
    method_stats = {}
    for method in EXTRACTION_METHODS:
        method_records = [r for r in records if r["extraction_method"] == method]
        if not method_records:
            continue
        m_correct = sum(1 for r in method_records if r["correct"])
        m_pct = 100 * m_correct / len(method_records)
        method_stats[method] = (len(method_records), m_correct)
        print(f"    {method:<22} n={len(method_records):>3}  correct={m_correct:>3}  accuracy={m_pct:.1f}%")
    print()
    return {
        "n": n, "n_correct": n_correct, "pct": pct,
        "n_fallback": method_stats.get("fallback_last_number", (0, 0))[0],
        "fallback_correct": method_stats.get("fallback_last_number", (0, 0))[1],
        "is_partial": is_partial,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-examples", type=int, default=32)
    args = parser.parse_args()

    print("=" * 78)
    print(f"GSM8K CPU-OFFLOAD 3-ARM COMPARISON -- n={args.num_examples}")
    print("=" * 78)
    print()

    baseline_records = load_records(ENGINE_BASELINE_32_PATH)
    baseline_summary = None
    if baseline_records:
        baseline_summary = summarize(
            "engine-path baseline ('raw' arm, via nano-vLLM engine, for reference only "
            "-- NOT numerically identical setup, see gsm8k_cpu_offload_run.py's docstring)",
            ENGINE_BASELINE_32_PATH, baseline_records, args.num_examples,
        )
    else:
        print(f"(engine-path baseline not found at {ENGINE_BASELINE_32_PATH} -- skipping reference row)\n")

    arm_summaries = {}
    for arm in ARM_CHOICES:
        path = os.path.join(args.output_dir, f"cpu_offload_n{args.num_examples}_{arm}.jsonl")
        records = load_records(path)
        arm_summaries[arm] = summarize(f"cpu-offload arm: {arm}", path, records, args.num_examples)

    print("=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    header = f"{'arm':<16}{'n':>5}{'correct':>9}{'acc%':>8}{'fallback':>10}{'fallback_acc%':>15}"
    print(header)
    rows = []
    if baseline_summary:
        rows.append(("engine-raw (ref)", baseline_summary))
    for arm in ARM_CHOICES:
        if arm_summaries[arm] is not None:
            rows.append((arm, arm_summaries[arm]))
    for name, s in rows:
        fb_acc = (100 * s["fallback_correct"] / s["n_fallback"]) if s["n_fallback"] else float("nan")
        fb_acc_str = f"{fb_acc:.1f}" if s["n_fallback"] else "n/a"
        partial_flag = " (partial)" if s["is_partial"] else ""
        print(f"{name + partial_flag:<16}{s['n']:>5}{s['n_correct']:>9}{s['pct']:>7.1f}%"
              f"{s['n_fallback']:>10}{fb_acc_str:>15}")
    print()

    print("Reading guide (see the pre-registered predictions in gsm8k_full_run.py's module")
    print("docstring, written before any of these arms were run):")
    print("  chat-no-think fallback count ~1  -> consistent with the thinking-prefix hypothesis")
    print("  chat-no-think fallback count 0   -> check whether idx 608 recovered or just failed")
    print("                                       differently (wrong answer via fallback)")
    print("  chat-no-think fallback count 5+  -> thinking-prefix is not the dominant mechanism")
    print("  chat-think ALSO dropping fallback vs. raw -> the fix is 'move to a chat turn',")
    print("                                       not 'disable thinking' specifically (confound)")
    print("  A fallback drop with FLAT OR LOWER accuracy than 'raw' is a red flag, not a win.")


if __name__ == "__main__":
    main()
