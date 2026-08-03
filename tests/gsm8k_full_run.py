"""STEP 4 -- full GSM8K test-set run (all 1319 examples, batches of 8),
checkpointed: each example's result is appended to RESULTS_PATH and flushed
immediately, so a crash partway through only costs the in-flight batch, not
everything completed so far. Re-running this script resumes automatically --
it reads RESULTS_PATH first, skips any idx already present, and only
processes what's left.

Same load/generate pattern as gsm8k_smoke_test.py (validated there first):
LLM(..., tensor_parallel_size=2, enforce_eager=True, max_model_len=2048),
SamplingParams(temperature=0, max_tokens=512), batches of exactly 8 prompts
per generate() call. These are the VALIDATED defaults -- --enforce-eager,
--batch-size, and --max-tokens below let you try faster settings, but each
is a real, unvalidated change to the methodology this eval's correctness
was built on this session, not a free speedup:

  --no-enforce-eager: tries CUDA graph capture/replay. This code path
    (engine/model_runner.py's capture_cudagraph/use_graph) has NEVER been
    exercised against the real checkpoint this entire session -- validate
    with `gsm8k_decode_contamination_check.py --no-enforce-eager` (exact
    token-for-token match expected, same as the eager-mode validation)
    BEFORE trusting a run that used it.

  --batch-size N: concurrency=8 is the value phase6_packed_contamination_check.py
    and this session's decode-contamination investigation actually validated.
    A different value needs its own contamination-check run
    (`gsm8k_decode_contamination_check.py --batch-size N`) before trusting it.

  --max-tokens N: 512 is the safe default. Use
    gsm8k_answer_position_check.py to find an empirically-justified smaller
    value (based on where the "####"/"the answer is" marker actually falls)
    instead of guessing -- guessing risks silently truncating a real answer
    before its marker appears, corrupting the score with no error message.

Usage (resumes automatically if RESULTS_PATH already has partial results):
    python tests/gsm8k_full_run.py
    python tests/gsm8k_full_run.py --no-enforce-eager --batch-size 16 --max-tokens 256   # after validating each
"""
import argparse
import json
import os
import sys
import types
from time import perf_counter

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
from gsm8k_extract import extract_answer_detailed  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")
CACHE_DIR = os.path.join(ROOT, "tests", "_gsm8k_cache")
RESULTS_PATH = os.path.join(CACHE_DIR, "full_results.jsonl")

MAX_MODEL_LEN = 2048
EXPECTED_TOTAL = 1319


def _load_completed(path: str) -> dict:
    completed = {}
    if not os.path.exists(path):
        return completed
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
            completed[record["idx"]] = record
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True,
        help="Default True (the validated setting). --no-enforce-eager tries CUDA graphs -- "
             "validate with gsm8k_decode_contamination_check.py first, see module docstring.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Default 8 (the validated concurrency). A different value needs its own "
             "contamination-check validation first, see module docstring.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Default 512 (safe upper bound). Use gsm8k_answer_position_check.py to pick a "
             "smaller, empirically-justified value instead of guessing.",
    )
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    from datasets import load_dataset
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    print("Loading openai/gsm8k (main, test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(ds)} examples (expect {EXPECTED_TOTAL}).")
    if len(ds) != EXPECTED_TOTAL:
        print(f"STOP: expected {EXPECTED_TOTAL} test examples, got {len(ds)}. Not proceeding.")
        sys.exit(1)

    completed = _load_completed(RESULTS_PATH)
    n_correct_total = sum(1 for r in completed.values() if r["correct"])
    print(f"Resuming: {len(completed)}/{len(ds)} examples already completed in {RESULTS_PATH}")

    remaining_indices = [i for i in range(len(ds)) if i not in completed]
    if not remaining_indices:
        print("All examples already completed -- nothing to do. Run tests/gsm8k_score.py to see results.")
        return

    if not args.enforce_eager or args.batch_size != 8 or args.max_tokens != 512:
        print(
            f"NON-DEFAULT SETTINGS IN USE: enforce_eager={args.enforce_eager}, "
            f"batch_size={args.batch_size}, max_tokens={args.max_tokens} -- make sure these were "
            f"validated with gsm8k_decode_contamination_check.py before trusting this run's score."
        )

    print(f"Loading engine from {CKPT_DIR} (tensor_parallel_size=2) ...")
    llm = LLM(
        CKPT_DIR,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=args.batch_size,
        max_model_len=MAX_MODEL_LEN,
    )
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    n_batches = (len(remaining_indices) + args.batch_size - 1) // args.batch_size
    t_run_start = perf_counter()
    n_done_this_run = 0

    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        for b in range(n_batches):
            batch_indices = remaining_indices[b * args.batch_size:(b + 1) * args.batch_size]
            batch_examples = [ds[i] for i in batch_indices]
            prompts = [build_prompt(ex["question"]) for ex in batch_examples]

            t0 = perf_counter()
            try:
                outputs = llm.generate(prompts, sp, use_tqdm=False)
            except Exception:
                print(
                    f"ERROR during generate() on batch {b + 1}/{n_batches} "
                    f"(indices {batch_indices}) -- results up to this point are safely "
                    f"checkpointed in {RESULTS_PATH}; re-run this script to resume."
                )
                raise
            batch_time = perf_counter() - t0

            for idx, ex, out in zip(batch_indices, batch_examples, outputs):
                model_text = out["text"]
                gold_result = extract_answer_detailed(ex["answer"])
                assert gold_result.method == "hash", (
                    f"example {idx}: gold answer text did not contain '#### N' as expected: "
                    f"{ex['answer']!r}"
                )
                model_result = extract_answer_detailed(model_text)
                correct = (
                    model_result.value is not None
                    and abs(model_result.value - gold_result.value) < 1e-6
                )
                record = {
                    "idx": idx,
                    "question": ex["question"],
                    "gold_answer": gold_result.value,
                    "model_output": model_text,
                    "extracted_answer": model_result.value,
                    "extraction_method": model_result.method,
                    "correct": correct,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                n_done_this_run += 1
                if correct:
                    n_correct_total += 1

            done_total = len(completed) + n_done_this_run
            elapsed = perf_counter() - t_run_start
            rate_per_example = elapsed / n_done_this_run
            eta_seconds = rate_per_example * (len(ds) - done_total)
            print(
                f"[batch {b + 1}/{n_batches}] indices {batch_indices[0]}-{batch_indices[-1]} "
                f"({len(batch_indices)} examples) in {batch_time:.1f}s | "
                f"progress {done_total}/{len(ds)} | running exact-match "
                f"{n_correct_total}/{done_total} ({100 * n_correct_total / done_total:.1f}%) | "
                f"ETA {eta_seconds / 3600:.2f}h"
            )

    print("\nFULL RUN COMPLETE.")
    print(f"Results in {RESULTS_PATH}. Run tests/gsm8k_score.py for the final scored report.")


if __name__ == "__main__":
    main()
