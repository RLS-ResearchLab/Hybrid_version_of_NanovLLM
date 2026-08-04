"""Does the empty-<think></think> (skip-reasoning) branch actually cost
accuracy in practice? gsm8k_decode_vs_hf_check.py found the engine takes
that branch at 4/10 vs HF's 1/10 on 30-token truncated samples -- suggestive
but not yet strong evidence at that n, and truncated at 30 tokens so
"still_reasoning" there just means "hadn't closed yet," not a real
classification of the FULL completion.

This script needs zero new GPU time: gsm8k_full_run.py's results files
already contain full (up to max_tokens=512) completions AND correctness
labels. Reads one of those, classifies each completion's think-branch from
the FULL text (a much cleaner classification than the 30-token truncated
one -- at 512 tokens, "still_reasoning" should be rare; most completions
will have either closed </think> with real content, closed it empty, or
never opened it at all), and reports exact-match accuracy PER BRANCH.

If empty_think's accuracy is meaningfully lower than closed_with_content's,
that's direct evidence the skip-reasoning branch is a real accuracy tax on
this checkpoint/prompt format -- not just a cosmetic decode-path curiosity.

No GPU needed -- pure Python, reads a JSONL file, same as gsm8k_score.py.

Usage:
    python tests/gsm8k_think_branch_accuracy.py --results tests/_gsm8k_cache/full_results_n32.jsonl
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_PATH = os.path.join(ROOT, "tests", "_gsm8k_cache", "full_results.jsonl")


def _classify_think(text: str) -> str:
    """Same logic as gsm8k_decode_vs_hf_check.py's _classify_think, kept as
    an independent copy here since this script is meant to run standalone
    (no GPU-dependent imports) against any results JSONL, not just this
    session's decode_vs_hf cache."""
    open_tag, close_tag = "<think>", "</think>"
    i = text.find(open_tag)
    if i == -1:
        return "no_think_tag"
    j = text.find(close_tag, i + len(open_tag))
    if j == -1:
        return "still_reasoning"  # shouldn't happen much at max_tokens=512, but possible
    inner = text[i + len(open_tag):j]
    return "empty_think" if inner.strip() == "" else "closed_with_content"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"STOP: {args.results} does not exist.")
        return

    records = []
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print(f"STOP: {args.results} has no records.")
        return

    buckets: dict[str, list[dict]] = {}
    for r in records:
        branch = _classify_think(r["model_output"])
        buckets.setdefault(branch, []).append(r)

    print("=" * 78)
    print(f"THINK-BRANCH ACCURACY -- {args.results} ({len(records)} examples)")
    print("=" * 78)
    print()
    print(f"{'branch':<20} {'n':>5} {'correct':>8} {'accuracy':>10} {'fallback_used':>15}")
    print("-" * 78)
    overall_correct = sum(1 for r in records if r["correct"])
    for branch in sorted(buckets, key=lambda b: -len(buckets[b])):
        rs = buckets[branch]
        n = len(rs)
        n_correct = sum(1 for r in rs if r["correct"])
        n_fallback = sum(1 for r in rs if r["extraction_method"] == "fallback_last_number")
        acc = 100 * n_correct / n
        print(f"{branch:<20} {n:>5} {n_correct:>8} {acc:>9.1f}% {n_fallback:>14}/{n}")
    print("-" * 78)
    print(f"{'ALL':<20} {len(records):>5} {overall_correct:>8} "
          f"{100 * overall_correct / len(records):>9.1f}%")
    print()

    empty = buckets.get("empty_think", [])
    content = buckets.get("closed_with_content", [])
    if empty and content:
        empty_acc = 100 * sum(1 for r in empty if r["correct"]) / len(empty)
        content_acc = 100 * sum(1 for r in content if r["correct"]) / len(content)
        gap = content_acc - empty_acc
        print(f"empty_think accuracy ({empty_acc:.1f}%, n={len(empty)}) vs. "
              f"closed_with_content accuracy ({content_acc:.1f}%, n={len(content)}): "
              f"gap={gap:+.1f}pts")
        if abs(gap) < 15 or min(len(empty), len(content)) < 5:
            print("NOTE: small n and/or small gap here -- not strong evidence either way at "
                  "this sample size, but directionally informative.")
        elif gap > 0:
            print("READ: empty_think completions score meaningfully worse -- consistent with "
                  "CoT-skipping being a real accuracy tax, not just a cosmetic decode-path "
                  "curiosity.")
        else:
            print("READ: empty_think completions do NOT score worse here -- for these "
                  "particular problems, skipping the boilerplate reasoning didn't cost "
                  "accuracy. Worth remembering before assuming shorter reasoning is always bad.")
    else:
        print("Not enough examples in both empty_think and closed_with_content buckets to "
              "compare -- rerun against a larger results file (e.g. full_results_n440.jsonl, "
              "once its provenance relative to the Bug A/B fixes is confirmed) for a real read.")


if __name__ == "__main__":
    main()
