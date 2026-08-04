"""Empirical check: how many tokens into a completion does the actual
answer marker ("####" or "the answer is") appear, versus max_tokens=512?

Built to answer a specific question before touching max_tokens: the timing
benchmark (gsm8k_timing_benchmark.py) found 91.7% of examples run to the
512-token cap without ever emitting EOS -- meaning the model is very
likely rambling past its answer into a hallucinated continuation for most
examples. extract_answer only needs the FIRST marker match, so anything
after it is wasted decode time. But guessing a smaller max_tokens without
checking WHERE the marker actually falls risks silently truncating the
real answer for some example before its marker appears -- a correctness
regression with no error message, exactly the kind of silent failure this
whole investigation has been trying to avoid. This measures it instead of
guessing.

Runs a modest sample (default 24, matching gsm8k_smoke_test.py's scale) at
the FULL max_tokens=512 (so nothing gets cut off in this measurement run
itself), then for each example finds the marker's character position
(gsm8k_extract.py's ExtractionResult.match_end) and re-tokenizes the
prefix up to that point to get an exact token count. Reports the
distribution (mean/median/p90/p95/max) -- pick max_tokens from the tail of
this distribution plus margin, not the mean, so rare slow-to-answer
examples don't get truncated.

--stop-strings (default ON): passes gsm8k_extract.py's GSM8K_STOP_PATTERNS
as SamplingParams.stop, so generation now stops itself right after the
answer marker instead of rambling to max_tokens=512 regardless. Verification
against the no-stop-string baseline (this script's own prior output, before
this flag existed) is by eye: per-example lines below now print completion_len
AND the extracted value/method, in the same format as before, so the two runs'
output can be diffed line-for-line -- completion_len should drop sharply for
examples that found a marker, and (method, value) must be IDENTICAL to the
baseline for every example that found a marker there. Pass --no-stop-strings
to reproduce the old always-run-to-512 behavior.

Usage:
    python tests/gsm8k_answer_position_check.py [--num-examples N]
    python tests/gsm8k_answer_position_check.py --no-stop-strings   # old baseline behavior
"""
import argparse
import os
import statistics
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm" and "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

sys.path.insert(0, os.path.dirname(__file__))
from gsm8k_prompt import build_prompt  # noqa: E402
from gsm8k_extract import extract_answer_detailed, GSM8K_STOP_PATTERNS  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")
MAX_MODEL_LEN = 2048
MAX_TOKENS = 512  # deliberately the FULL cap for this measurement run -- must not truncate anything
BATCH_SIZE = 8
EXPECTED_TOTAL = 1319


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-examples", type=int, default=24)
    parser.add_argument(
        "--stop-strings", action=argparse.BooleanOptionalAction, default=True,
        help="Default True -- generation stops itself right after the answer marker "
             "(gsm8k_extract.py's GSM8K_STOP_PATTERNS) instead of always running to "
             "max_tokens. --no-stop-strings reproduces the old baseline behavior.",
    )
    args = parser.parse_args()
    num_examples = args.num_examples

    from datasets import load_dataset
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    print("Loading openai/gsm8k (main, test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if len(ds) != EXPECTED_TOTAL:
        print(f"STOP: expected {EXPECTED_TOTAL} test examples, got {len(ds)}.")
        sys.exit(1)

    print(f"Loading engine from {CKPT_DIR} (tensor_parallel_size=2) ...")
    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=BATCH_SIZE,
        max_model_len=MAX_MODEL_LEN,
    )
    sp = SamplingParams(
        temperature=0, max_tokens=MAX_TOKENS,
        stop=GSM8K_STOP_PATTERNS if args.stop_strings else None,
    )
    print(f"stop-strings: {'ON (' + str(len(GSM8K_STOP_PATTERNS)) + ' patterns)' if args.stop_strings else 'OFF (baseline behavior)'}")

    answer_token_positions = []  # tokens needed to reach the marker, per example (only where found)
    completion_lengths = []
    n_no_marker = 0
    n_batches = (num_examples + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(n_batches):
        lo, hi = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, num_examples)
        prompts = [build_prompt(ds[i]["question"]) for i in range(lo, hi)]
        outputs = llm.generate(prompts, sp, use_tqdm=False)

        for i, out in zip(range(lo, hi), outputs):
            text = out["text"]
            token_ids = out["token_ids"]
            completion_lengths.append(len(token_ids))
            result = extract_answer_detailed(text)
            if result.match_end is None or result.method == "fallback_last_number":
                # No "####"/"the answer is" marker at all, or only the weak
                # last-number fallback fired -- can't measure a marker
                # position, and this example wouldn't be safe to truncate
                # against anyway.
                n_no_marker += 1
                print(f"  idx={i}: NO MARKER FOUND (method={result.method}, value={result.value}) -- "
                      f"completion_len={len(token_ids)} tokens")
                print(f"    last 400 chars: {text[-400:]!r}")
                continue
            # Re-tokenize the prefix up to (and including) the marker to get
            # an exact token count -- character position alone doesn't map
            # 1:1 to token count.
            prefix_token_ids = llm.tokenizer.encode(text[:result.match_end])
            n_tokens_to_answer = len(prefix_token_ids)
            answer_token_positions.append(n_tokens_to_answer)
            print(f"  idx={i}: answer marker at token ~{n_tokens_to_answer} "
                  f"(method={result.method}, value={result.value}, completion_len={len(token_ids)} tokens)")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Examples measured: {num_examples}")
    print(f"Examples with a real marker found: {len(answer_token_positions)}")
    print(f"Examples with NO marker (fallback or failed): {n_no_marker}")
    if not answer_token_positions:
        print("No markers found at all -- cannot recommend a max_tokens value from this sample.")
        return

    mean = statistics.mean(answer_token_positions)
    median = statistics.median(answer_token_positions)
    p90 = _percentile(answer_token_positions, 0.90)
    p95 = _percentile(answer_token_positions, 0.95)
    worst = max(answer_token_positions)
    print(f"Tokens-to-answer-marker: mean={mean:.1f}  median={median:.1f}  "
          f"p90={p90:.1f}  p95={p95:.1f}  max={worst}")
    print(f"Current max_tokens={MAX_TOKENS} -- ratio of tokens actually needed at max: "
          f"{100 * worst / MAX_TOKENS:.1f}%")

    recommended = int(worst * 1.5)  # generous margin above the worst-case observed in this sample
    print()
    print(f"Suggested max_tokens (worst observed x1.5 margin): {recommended}")
    print(f"NOTE: this is a sample of {num_examples}/{EXPECTED_TOTAL} examples -- the true tail could be "
          f"longer than what this sample saw. Treat this as informative, not a hard guarantee; a value "
          f"picked here should still be comfortably above the worst case observed, not just the mean/median.")


if __name__ == "__main__":
    main()
