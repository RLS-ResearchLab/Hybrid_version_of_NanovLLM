"""Re-score an EXISTING gsm8k_full_run.py results JSONL against the FIXED
_ANSWER_IS_PATTERN (gsm8k_extract.py -- now requires a trailing delimiter
after the number, so an intermediate "...is 60% of..."/"...is 60
minutes..." mention inside ongoing reasoning can no longer be mistaken for
the terminal answer). The bug was found via gsm8k_decode_vs_hf_check.py's/
gsm8k_answer_position_check.py's --stop-strings verification run: idx=14's
completion contained an intermediate "...is 60..." aside AND the real final
"The answer is $70,000."; the old leftmost-match rule with no delimiter
requirement silently extracted the wrong (intermediate) value.

ZERO GPU cost, deliberately: gsm8k_full_run.py already wrote each example's
full model_output text to disk -- that text is still valid, only the
EXTRACTION logic that ran over it at generation time was wrong. This script
re-runs extract_answer_detailed() on the stored text with the fixed
pattern; nothing needs to be regenerated. gold_answer is reused as-is from
the stored record (computed via _HASH_PATTERN against the dataset's own
"#### N" gold text, which this fix does not touch, so re-deriving it would
just reproduce the same number).

Reports the NEW overall accuracy against the OLD (stored, pre-fix) accuracy,
and itemizes every example whose verdict actually FLIPPED either direction
-- the flip count and direction is the real answer to "did this bug affect
the invalidated run's score, and which way," not a guess from the pattern
alone.

Usage:
    python tests/gsm8k_rescore_fixed_extractor.py [--results PATH]
    python tests/gsm8k_rescore_fixed_extractor.py --results tests/_gsm8k_cache/full_results_n32.jsonl
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from gsm8k_extract import extract_answer_detailed  # noqa: E402 -- the FIXED version

DEFAULT_RESULTS_PATH = os.path.join(ROOT, "tests", "_gsm8k_cache", "full_results.jsonl")

_ANSWER_IS_PHRASE = re.compile(r"the answer is", re.IGNORECASE)


def _every_occurrence_context(text: str, radius: int = 60) -> list[str]:
    """Every "the answer is" occurrence in the text with surrounding
    context, not just the one/two the extractor happened to pick -- for
    diagnosing WHY a match was accepted/rejected, not just what value came
    out. Only called for the small number of flipped examples, not the
    whole dataset, so the extra scan cost is negligible."""
    out = []
    for m in _ANSWER_IS_PHRASE.finditer(text):
        lo = max(0, m.start() - 10)
        hi = min(len(text), m.end() + radius)
        out.append(text[lo:hi].replace("\n", "\\n"))
    return out


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
                print(f"WARNING: skipping malformed line {line_no} in {path} "
                      f"(likely a crash mid-write of the last line) -- {line[:80]!r}...")
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
        print(f"STOP: {args.results} does not exist yet -- nothing to re-score.")
        return

    records = load_records(args.results)
    n = len(records)
    if n == 0:
        print(f"STOP: {args.results} has no valid records.")
        return

    old_n_correct = sum(1 for r in records if r["correct"])
    old_pct = 100 * old_n_correct / n

    flips_to_correct = []
    flips_to_wrong = []
    new_n_correct = 0
    new_n_fallback = 0
    new_n_failed = 0

    for r in records:
        gold = r["gold_answer"]
        new_result = extract_answer_detailed(r["model_output"])
        new_correct = new_result.value is not None and abs(new_result.value - gold) < 1e-6
        if new_correct:
            new_n_correct += 1
        if new_result.method == "fallback_last_number":
            new_n_fallback += 1
        elif new_result.method == "failed":
            new_n_failed += 1

        old_correct = r["correct"]
        if old_correct != new_correct:
            entry = {
                "idx": r["idx"], "gold": gold,
                "old_value": r["extracted_answer"], "old_method": r["extraction_method"],
                "new_value": new_result.value, "new_method": new_result.method,
                "model_output": r["model_output"],
            }
            (flips_to_correct if new_correct else flips_to_wrong).append(entry)

    new_pct = 100 * new_n_correct / n

    print("=" * 78)
    print(f"RE-SCORE (fixed _ANSWER_IS_PATTERN) -- {args.results} ({n} examples)")
    print("=" * 78)
    print(f"OLD (stored, pre-fix):  {old_n_correct}/{n}  ({old_pct:.2f}%)")
    print(f"NEW (re-extracted):     {new_n_correct}/{n}  ({new_pct:.2f}%)")
    print(f"DELTA: {new_pct - old_pct:+.2f} points")
    print()
    print(f"Flipped wrong -> correct ({len(flips_to_correct)}):")
    for e in flips_to_correct:
        print(f"  idx={e['idx']}: gold={e['gold']}  old={e['old_value']} ({e['old_method']}) -> "
              f"new={e['new_value']} ({e['new_method']})")
    print()
    print(f"Flipped correct -> wrong ({len(flips_to_wrong)}):")
    for e in flips_to_wrong:
        print(f"  idx={e['idx']}: gold={e['gold']}  old={e['old_value']} ({e['old_method']}) -> "
              f"new={e['new_value']} ({e['new_method']})")
        # This direction is the concerning one -- the fix REJECTED a match
        # that used to be correct. Show every "the answer is" occurrence in
        # the text so it's visible exactly which one was the genuine answer
        # and why the new pattern didn't accept it there.
        for i, ctx in enumerate(_every_occurrence_context(e["model_output"])):
            print(f"      occurrence {i}: ...{ctx}...")
    print()
    print(f"New fallback_last_number count: {new_n_fallback}/{n}  "
          f"(informational -- these were never trusted as answer_is either way)")
    print(f"New extraction-failed count: {new_n_failed}/{n}")
    print()
    if new_n_correct == old_n_correct:
        print("VERDICT: no net accuracy change -- either the bug didn't affect this run at all, "
              "or an equal number of flips happened in both directions and cancelled out (check "
              "the flip lists above to tell which).")
    else:
        direction = "UP" if new_pct > old_pct else "DOWN"
        print(f"VERDICT: this run's TRUE score under correct extraction is {new_pct:.2f}%, not "
              f"{old_pct:.2f}% -- the extraction bug was masking this, moving the score {direction} "
              f"by {abs(new_pct - old_pct):.2f} points. Compare against the 87.5% threshold using "
              f"THIS number, not the original.")


if __name__ == "__main__":
    main()
