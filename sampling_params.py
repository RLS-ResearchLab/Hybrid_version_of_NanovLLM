from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    # Each entry is a REGEX pattern (not a literal substring), matched
    # case-insensitively via re.search against a trailing window of the
    # sequence's DECODED COMPLETION text (never the prompt) once per decode
    # step -- see engine/scheduler.py's _check_stop_string. None/empty means
    # no stop-string checking is done at all (zero added cost). A match is
    # treated identically to hitting EOS: the sequence finishes immediately,
    # keeping every token generated up to and including the match.
    stop: list[str] | None = None

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be >= 0 (0 = greedy)"
