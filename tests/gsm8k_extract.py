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
after that is a hallucinated continuation we must not accidentally prefer.

HISTORY WORTH KEEPING VISIBLE, because two different "fixes" here each
silently cost real accuracy before being caught: a --stop-strings
verification run surfaced one completion where extraction's leftmost match
(value 60) didn't match the value gsm8k_decode_vs_hf_check.py's stop
mechanism confirmed as the completion's actual ending ($70,000), which
looked like a real leftmost-match bug. Two successive attempts added a
required trailing delimiter after the number to _ANSWER_IS_PATTERN, first
excluding bare whitespace, then (after that regressed) excluding just '%'.
BOTH were re-scored against the real invalidated 1319-example run's stored
model_output text (zero GPU cost -- see gsm8k_rescore_fixed_extractor.py)
and BOTH produced the exact same result: 18 examples flipped from correct
to WRONG, 0 recovered, a measured -1.36 point regression each time. Pulling
full per-occurrence context for those 18 (not just the two matched values)
showed why: GSM8K genuinely has percentage-valued answers, the model
correctly states them as "The answer is 60%.", and EVERY ONE of the 18 was
this exact shape -- a correct "N%." answer being rejected (since '%' isn't
a delimiter), falling through to a hallucinated self-invented follow-up
Q&A pair the model appends after correctly answering the real question
(its own new "Q: ..." completely unrelated to the prompt, complete with
its own internally-coherent worked arithmetic and its own "The answer is
N." conclusion). That last part matters beyond just explaining these 18:
it means the ORIGINAL single-anecdote case that started this whole
investigation was never actually confirmed by reading its real text --
only inferred from a plausible-sounding narrative -- and given how
convincingly this model hallucinates a FULLY coherent second question
(not just a stray fragment), that original case may well have been the
same shape, with the old leftmost-match extractor having been correct all
along. Two measured regressions and zero measured recoveries is not
"delimiter needs tuning" -- it's evidence the premise was wrong.
_ANSWER_IS_PATTERN below is reverted to its original, delimiter-free form.
Don't re-add a trailing-delimiter requirement here without first reading
the actual raw text of a specific real failure -- guessing from a
plausible story has now cost two rounds of regressions on the real
dataset.
"""
import re
from typing import NamedTuple, Optional

_HASH_PATTERN = re.compile(r"####\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_ANSWER_IS_PATTERN = re.compile(r"the answer is\s*\$?(-?[0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
_ANY_NUMBER_PATTERN = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")

# For SamplingParams.stop (engine/scheduler.py's _check_stop_string) --
# UNCHANGED and NOT reverted: this one was validated independently, with
# real generation data (gsm8k_answer_position_check.py --stop-strings,
# n=24 on the live GPU checkpoint), and every one of 15/16 marker-found
# completions stopped exactly 1 token after its own genuine answer marker,
# with matched_text confirmed by direct engine-side debug output -- not
# inferred from a narrative the way the reverted _ANSWER_IS_PATTERN
# experiment above was. Trailing delimiter is required here for a
# completely different, load-bearing reason: without it, a
# partially-generated multi-digit number (e.g. just "1" of "18", if the
# tokenizer splits it across decode steps) would already satisfy `\d[\d,]*`
# and fire a token early, truncating the answer mid-digit. Requiring a
# character AFTER the number means the pattern can't match until the
# tokenizer has already emitted something past the last digit -- which
# structurally can't happen until the number is actually finished. '$' is
# included in the delimiter set because this checkpoint frequently writes
# math in LaTeX inline mode ("the answer is $60$." -- closing '$' right
# after the number).
#
# KNOWN, still-open gap (same root cause as the now-reverted extraction
# regression above, not yet independently confirmed to matter here): a
# genuine percentage answer ("the answer is 60%.") won't satisfy this
# pattern either ('%' isn't a delimiter), so generation won't stop right
# there -- it'll keep going until a LATER match (e.g. a hallucinated
# follow-up's own "the answer is N.") or max_tokens. This is an EFFICIENCY
# gap only, not a correctness one: whatever text ends up captured,
# _ANSWER_IS_PATTERN's leftmost-match (now delimiter-free again) will still
# correctly find the real "N%." answer regardless of how much extra text
# follows it. Not fixed here since it costs decode time, not accuracy --
# revisit only with real evidence of how much it actually costs.
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
