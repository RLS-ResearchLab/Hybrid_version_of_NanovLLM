"""GSM8K answer extraction -- pure Python, no GPU/model dependency.

Two markers are checked, in this priority order, before falling back to
"last number in the text":

  1. "#### <number>" -- how the ORIGINAL DATASET stores gold answers (every
     `answer` field in openai/gsm8k ends this way). Checked first so this
     module also works unmodified on gold-answer text.
  2. "The answer is <number>" -- how the model's own completions will
     actually end, because the 8-shot exemplars we prompt with (see
     gsm8k_prompt.py, sourced from lm-evaluation-harness's gsm8k-cot task)
     themselves end every worked example with that exact phrase. The model
     mimics the shots it's shown; it will essentially never spontaneously
     emit "####" on its own.
  3. Last number appearing anywhere in the text, as a last resort.

Both markers use re.search (leftmost/FIRST match), not the last one. This
matters when max_tokens lets the model ramble past its answer into a
hallucinated continuation (this engine has no "stop on string" support --
see gsm8k_smoke_test.py's docstring -- only EOS-token/max_tokens cutoffs).
The intended answer to the ACTUAL question is always the first "#### N" /
"The answer is N" the model writes; anything after that is a hallucinated
continuation we must not accidentally prefer.
"""
import re
from typing import NamedTuple, Optional

_HASH_PATTERN = re.compile(r"####\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_ANSWER_IS_PATTERN = re.compile(r"the answer is\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
_ANY_NUMBER_PATTERN = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")

# For SamplingParams.stop (engine/scheduler.py's _check_stop_string) --
# deliberately kept in sync with _HASH_PATTERN/_ANSWER_IS_PATTERN above
# (same marker + number shape) so "the engine stops generating" and "the
# extractor would have found a real hash/answer_is match, not just
# fallback_last_number" are the same condition. One addition beyond the
# extraction patterns: a REQUIRED trailing delimiter right after the number.
# Without it, a partially-generated multi-digit number (e.g. just "1" of
# "18", if the tokenizer splits it across steps) would already satisfy
# `\d[\d,]*` and fire a token early, truncating the answer mid-digit.
# Requiring a character AFTER the number means the pattern can't match until
# the tokenizer has already emitted something past the last digit -- which
# structurally can't happen until the number is actually finished. No
# separate "wait one more token and re-check" state machine needed; the
# regex's own shape enforces it.
#
# Delimiter set is [.\s$], not just [.\s]: this checkpoint frequently writes
# math in LaTeX inline mode ("the answer is $60$." -- closing '$' right
# after the number, not '.'/whitespace). A live n=24 verification run
# (gsm8k_answer_position_check.py --stop-strings) caught exactly this --
# one example's stop fired 143 tokens after its marker instead of the
# expected 1, because [.\s] alone never matched the "$60$" case and had to
# wait for a later, cleanly-delimited restatement. [.\s$] closes that gap.
GSM8K_STOP_PATTERNS = [
    r"####\s*\$?-?[0-9][0-9,]*(?:\.[0-9]+)?[.\s$]",
    r"the answer is\s*\$?-?[0-9][0-9,]*(?:\.[0-9]+)?[.\s$]",
]


class ExtractionResult(NamedTuple):
    value: Optional[float]
    method: str  # "hash" | "answer_is" | "fallback_last_number" | "failed"
    match_end: Optional[int] = None  # character offset just past the matched marker,
                                      # None for "failed" -- used by gsm8k_answer_position_check.py
                                      # to measure how many tokens are needed to reach the answer,
                                      # not used by scoring (correctness only needs .value/.method)


def _to_float(num_str: str) -> Optional[float]:
    cleaned = num_str.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_answer_detailed(model_output: str) -> ExtractionResult:
    """Like extract_answer(), but also reports which extraction path fired --
    used by the run scripts to report fallback/failure counts separately
    from correctness (never silently collapse these into one number)."""
    if not model_output:
        return ExtractionResult(None, "failed")

    m = _HASH_PATTERN.search(model_output)
    if m:
        val = _to_float(m.group(1))
        if val is not None:
            return ExtractionResult(val, "hash", m.end())

    m = _ANSWER_IS_PATTERN.search(model_output)
    if m:
        val = _to_float(m.group(1))
        if val is not None:
            return ExtractionResult(val, "answer_is", m.end())

    last_match = None
    for last_match in _ANY_NUMBER_PATTERN.finditer(model_output):
        pass
    if last_match is not None:
        val = _to_float(last_match.group(0))
        if val is not None:
            return ExtractionResult(val, "fallback_last_number", last_match.end())

    return ExtractionResult(None, "failed", None)


def extract_answer(model_output: str) -> Optional[float]:
    """Return the extracted numeric answer, or None if truly nothing could
    be parsed (never silently defaults to 0 or any placeholder)."""
    return extract_answer_detailed(model_output).value
