"""Unit tests for gsm8k_extract.extract_answer() -- pure Python, no GPU, no
model, no dataset download. Run standalone (no pytest dependency, matching
this repo's existing tests/ convention of plain assert + a main() runner):

    python tests/test_gsm8k_extract.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gsm8k_extract import extract_answer, extract_answer_detailed  # noqa: E402


CASES = [
    # (name, model_output, expected_value, expected_method)
    ("hash_basic", "She has 3 apples left.\n#### 3", 3.0, "hash"),
    ("hash_negative", "The temperature dropped.\n#### -5", -5.0, "hash"),
    ("hash_decimal", "Price per unit.\n#### 3.5", 3.5, "hash"),
    ("hash_comma_thousands", "Total revenue this year.\n#### 1,234", 1234.0, "hash"),
    (
        "hash_comma_large",
        "After all transactions.\n#### 12,345.67",
        12345.67,
        "hash",
    ),
    # Canonical CoT style (matches the exemplars in gsm8k_prompt.py) -- no
    # "####" at all, must fall through to the "answer_is" path, not the
    # last-number fallback.
    (
        "answer_is_basic",
        "Jason started with 20 lollipops. He gave some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
        8.0,
        "answer_is",
    ),
    (
        "answer_is_dollar",
        "Olivia had 23 dollars. She spent 15. The answer is $8.",
        8.0,
        "answer_is",
    ),
    (
        "answer_is_case_insensitive",
        "Some reasoning here. THE ANSWER IS 42.",
        42.0,
        "answer_is",
    ),
    # Hash takes priority over answer_is when both are present, and the
    # FIRST hash match wins (guards against a hallucinated continuation
    # after max_tokens lets the model ramble past its answer).
    (
        "hash_priority_over_answer_is",
        "#### 7\nThe answer is 99 in some other unrelated continuation.",
        7.0,
        "hash",
    ),
    (
        "first_match_not_last_on_rambled_continuation",
        "Q: What is 2+2?\n\nA: 2 + 2 = 4. The answer is 4.\n\n"
        "Q: What is 3+3? \n\nA: 3 + 3 = 6. The answer is 6.",
        4.0,
        "answer_is",
    ),
    # No marker at all -- fallback to last number in the text.
    (
        "fallback_last_number",
        "She started with 3 apples and bought 4 oranges, giving her 7 pieces of fruit total.",
        7.0,
        "fallback_last_number",
    ),
    (
        "fallback_last_number_with_distractor",
        "There were 15 trees. 21 trees now. They planted 6 more trees.",
        6.0,
        "fallback_last_number",
    ),
    # Genuinely unparseable -- must return None, NOT 0 or any placeholder.
    ("unparseable_no_numbers", "I don't know the answer to this one.", None, "failed"),
    ("unparseable_empty_string", "", None, "failed"),
]


def main():
    failures = []
    for name, text, expected_value, expected_method in CASES:
        result = extract_answer_detailed(text)
        ok = result.value == expected_value and result.method == expected_method
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {name}: got value={result.value!r} method={result.method!r} "
            f"(expected value={expected_value!r} method={expected_method!r})"
        )
        if not ok:
            failures.append(name)

        # extract_answer() (the plain, non-detailed API) must agree with the
        # detailed one's value on every case.
        plain = extract_answer(text)
        if plain != expected_value:
            print(
                f"[FAIL] {name}: extract_answer() plain API returned {plain!r}, "
                f"expected {expected_value!r}"
            )
            failures.append(f"{name} (plain API)")

    print()
    if failures:
        print(f"RESULT: {len(failures)}/{len(CASES)} case(s) FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"RESULT: all {len(CASES)} cases PASSED")


if __name__ == "__main__":
    main()
