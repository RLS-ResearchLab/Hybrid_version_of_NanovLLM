"""STEP 5 -- scoring script. Reads the incremental results file written by
gsm8k_full_run.py (or gsm8k_smoke_test.py, via --results) and reports:
  - overall exact-match % (the headline number)
  - extraction-fallback-used count and extraction-failure count, broken out
    SEPARATELY from math-correctness (never collapsed into one number)
  - whether the result clears the 87.5% threshold

Extraction-failure cases are COUNTED AS WRONG in the denominator, not
excluded -- this is the default per spec. Excluding them would inflate the
score by silently ignoring genuine failures; every example that was
attempted must appear in the denominator one way or another.

No GPU needed -- pure Python, reads a JSONL file.

Usage:
    python tests/gsm8k_score.py
    python tests/gsm8k_score.py --results tests/_gsm8k_cache/smoke_results.jsonl
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_PATH = os.path.join(ROOT, "tests", "_gsm8k_cache", "full_results.jsonl")
EXPECTED_TOTAL = 1319
THRESHOLD_PCT = 87.5


def load_records(path: str) -> list[dict]:
    records = []
    seen_idx = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"WARNING: skipping malformed line {line_no} in {path} "
                    f"(likely a crash mid-write of the last line) -- {line[:80]!r}..."
                )
                continue
            if record["idx"] in seen_idx:
                print(f"WARNING: duplicate idx={record['idx']} in {path} -- keeping the FIRST occurrence")
                continue
            seen_idx.add(record["idx"])
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"STOP: {args.results} does not exist yet -- nothing to score.")
        return

    records = load_records(args.results)
    n = len(records)
    if n == 0:
        print(f"STOP: {args.results} has no valid records.")
        return

    n_correct = sum(1 for r in records if r["correct"])
    n_wrong = n - n_correct
    n_fallback = sum(1 for r in records if r["extraction_method"] == "fallback_last_number")
    n_extract_failed = sum(1 for r in records if r["extraction_method"] == "failed")
    exact_match_pct = 100 * n_correct / n

    is_partial = n < EXPECTED_TOTAL
    print("=" * 70)
    print(f"GSM8K SCORE -- {args.results}")
    print("=" * 70)
    if is_partial:
        print(f"PARTIAL RUN: {n}/{EXPECTED_TOTAL} examples completed so far.")
        print("(Numbers below are over the completed subset only -- the 87.5% threshold")
        print(" verdict below only applies once all 1319 examples are done.)")
    elif n > EXPECTED_TOTAL:
        print(f"WARNING: {n} records found, expected exactly {EXPECTED_TOTAL} -- "
              f"check for a stale/mixed results file.")
    print()
    print(f"Examples scored:            {n}")
    print(f"Correct (exact match):      {n_correct}")
    print(f"Wrong (incl. unparseable):  {n_wrong}")
    print(f"Exact-match accuracy:       {exact_match_pct:.2f}%")
    print()
    print("Extraction diagnostics (informational, already folded into 'wrong' above where applicable):")
    print(f"  Fallback used (last-number, no '####'/'the answer is' marker found): {n_fallback}/{n}")
    print(f"  Extraction failure (no number found at all -- always counted WRONG): {n_extract_failed}/{n}")
    print()
    print("NOTE: extraction-failure cases are counted as WRONG in the denominator above,")
    print("      not excluded -- excluding them would inflate the score by ignoring genuine failures.")
    print()

    if not is_partial:
        verdict = "PASS" if exact_match_pct >= THRESHOLD_PCT else "FAIL"
        print(f"THRESHOLD: {THRESHOLD_PCT}% exact-match accuracy")
        print(f"VERDICT: {verdict} -- {exact_match_pct:.2f}% {'>=' if verdict == 'PASS' else '<'} {THRESHOLD_PCT}%")


if __name__ == "__main__":
    main()
