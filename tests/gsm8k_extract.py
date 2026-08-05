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
hallucinated continuation. The intended answer to the ACTUAL question is
always the first "#### N" / "The answer is N" the model writes; anything
after that is a hallucinated continuation we must not accidentally prefer
-- BUT ONLY once that first occurrence is confirmed to actually be a
terminal statement, not an intermediate mention (see _ANSWER_IS_PATTERN's
trailing-delimiter requirement below).

HISTORY WORTH KEEPING VISIBLE, because the wrong fix here silently costs
real accuracy: this trailing-delimiter requirement was added after
gsm8k_decode_vs_hf_check.py's/gsm8k_answer_position_check.py's
--stop-strings verification surfaced one completion with an intermediate
"...is 60..." aside AND a genuine final "The answer is $70,000."; the old
leftmost-match rule with NO delimiter requirement silently extracted the
wrong (intermediate) value. The FIRST fix tried here excluded plain
whitespace from the delimiter set (keeping only '.'/newline/'$'/end-of-
string), reasoning that all 16 genuine answers in a 24-example sample ended
in "N." directly. That was wrong at scale: re-scoring the real invalidated
1319-example run with that version flipped 18 examples from correct to
WRONG and recovered zero, a measured -1.36 point regression -- because
"The answer is 60 dollars."/"...8 hours." (a genuine final answer with a
plain trailing unit word before the period) is common, and excluding bare
whitespace rejected all of those too, same as the intermediate mentions it
was meant to catch. The delimiter set below is back to including
whitespace (matching GSM8K_STOP_PATTERNS exactly) -- '%' is still excluded
(not a delimiter char at all), which is what actually rejects "...is 60% of
the total...", the specific case that motivated this fix in the first
place. "...is 60 minutes, but wait..." is NOT reliably rejected by this
(bare whitespace before a continuing word still matches) -- a real,
accepted, narrower gap; distinguishing that case would need something
smarter than a trailing-delimiter char class, and isn't worth reintroducing
the whitespace-exclusion regression to chase.
"""
import re
from typing import NamedTuple, Optional

_HASH_PATTERN = re.compile(r"####\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
# Trailing-delimiter requirement: a number immediately followed by '%' or
# similar (not in [.\s$], not end-of-string) means this "the answer is N" is
# an intermediate mention (e.g. "...is 60% of the total...") embedded in
# ongoing reasoning, not the real final answer -- see the module docstring
# for the measured-regression history of why this is [.\s$], matching
# GSM8K_STOP_PATTERNS exactly, and not a narrower set. End-of-string is
# additionally allowed here (`(?:...|$)` -- the alternation's trailing `$`
# is the regex end-of-input anchor, not the literal dollar-sign character in
# the char class) because extraction runs on a COMPLETE, already-generated
# text: if the number is the literal last thing in the text (e.g. truncated
# right there), there's no more text that could turn it into a false match
# -- unlike GSM8K_STOP_PATTERNS, checked mid-generation, where "nothing here
# yet" must NOT be treated as "confirmed no continuation" (more tokens are
# still coming).
_ANSWER_IS_PATTERN = re.compile(
    r"the answer is\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)(?:[.\s$]|$)", re.IGNORECASE
)
_ANY_NUMBER_PATTERN = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")

# For SamplingParams.stop (engine/scheduler.py's _check_stop_string) --
# deliberately kept in sync with _HASH_PATTERN/_ANSWER_IS_PATTERN above
# (same marker + number shape + trailing-delimiter requirement, same
# [.\s$] delimiter set) so "the engine stops generating" and "the
# extractor would have found a real hash/answer_is match, not just
# fallback_last_number" are the same condition. Trailing delimiter is
# REQUIRED for a different reason here than in _ANSWER_IS_PATTERN: without
# it, a partially-generated multi-digit number (e.g. just "1" of "18", if
# the tokenizer splits it across steps) would already satisfy `\d[\d,]*`
# and fire a token early, truncating the answer mid-digit. Requiring a
# character AFTER the number means the pattern can't match until the
# tokenizer has already emitted something past the last digit -- which
# structurally can't happen until the number is actually finished. No
# separate "wait one more token and re-check" state machine needed; the
# regex's own shape enforces it. '$' is included in the delimiter set
# because this checkpoint frequently writes math in LaTeX inline mode
# ("the answer is $60$." -- closing '$' right after the number).
#
# Same "...is 60 minutes..." gap as _ANSWER_IS_PATTERN above, NOT proven
# safe here either (no case like that appeared in the n=24 verification
# run) -- a mid-reasoning "...is 60 minutes..." aside could in principle
# trigger a premature stop. Left as-is pending real evidence one way or
# the other, same reasoning as the docstring above: don't narrow this
# again on a hypothesis alone after the last one cost -1.36 points.
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
