"""STEP 3 -- GPU smoke test on the FIRST 24 GSM8K test examples only (3
batches of 8), built directly on the validated load/generate patterns from
run_construct.py and tests/reference_check_phase6.py (same LLM(...) kwargs,
same tensor_parallel_size=2 / enforce_eager=True setup) and the
concurrency=8 packed-batching already verified clean on this checkpoint by
tests/phase6_packed_contamination_check.py (8/8 PASS, cosine 0.9976-0.9997).
Each batch of 8 prompts is passed to llm.generate() in ONE call, so the
scheduler packs them together the same way real concurrent serving would --
not a hand-rolled batching loop.

NO STOP-STRING SUPPORT: this engine's SamplingParams (nanovllm/sampling_params.py)
only has temperature/max_tokens/ignore_eos -- no "until"/stop-sequence list
like lm-evaluation-harness's generation_kwargs.until (["Q:", "</s>", "<|im_end|>"]).
Generation only stops early on the real EOS token; otherwise it runs to
max_tokens=512. gsm8k_extract.py's extract_answer takes the FIRST "####" or
"The answer is N" match in the text specifically to stay correct even if the
model rambles past its answer into a hallucinated next Q/A pair.

Sampling: temperature=0 (deterministic argmax, verified bitwise-identical
across repeated runs -- see layers/sampler.py), max_tokens=512 (see
gsm8k_dry_run.py's docstring for justification). top_p is not a parameter
this engine's SamplingParams exposes; it's moot at temperature=0 anyway
(argmax ignores the sampling distribution entirely).

Usage:
    python tests/gsm8k_smoke_test.py
"""
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
from gsm8k_extract import extract_answer, extract_answer_detailed  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")
CACHE_DIR = os.path.join(ROOT, "tests", "_gsm8k_cache")
RESULTS_PATH = os.path.join(CACHE_DIR, "smoke_results.jsonl")

MAX_MODEL_LEN = 2048
MAX_TOKENS = 512
BATCH_SIZE = 8
NUM_EXAMPLES = 24  # 3 batches of 8


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)  # smoke test always starts clean, unlike the checkpointed full run

    from datasets import load_dataset
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    print("Loading openai/gsm8k (main, test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(ds)} examples (expect 1319).")
    if len(ds) != 1319:
        print(f"STOP: expected 1319 test examples, got {len(ds)}. Not proceeding.")
        sys.exit(1)

    subset = ds.select(range(NUM_EXAMPLES))

    print(f"Loading engine from {CKPT_DIR} (tensor_parallel_size=2) ...")
    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=BATCH_SIZE,
        max_model_len=MAX_MODEL_LEN,
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    all_records = []
    batch_timings = []
    n_batches = (NUM_EXAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        for b in range(n_batches):
            lo, hi = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, NUM_EXAMPLES)
            batch_examples = [subset[i] for i in range(lo, hi)]
            prompts = [build_prompt(ex["question"]) for ex in batch_examples]

            print(f"\n=== Batch {b + 1}/{n_batches} (examples {lo}-{hi - 1}, {len(prompts)} prompts) ===")
            t0 = perf_counter()
            try:
                outputs = llm.generate(prompts, sp)
            except Exception:
                print(f"ERROR during generate() on batch {b + 1} (examples {lo}-{hi - 1}):")
                raise
            batch_time = perf_counter() - t0
            batch_timings.append(batch_time)
            print(f"Batch {b + 1} wall-clock: {batch_time:.2f}s ({batch_time / len(prompts):.2f}s/example)")

            for i, ex, out in zip(range(lo, hi), batch_examples, outputs):
                model_text = out["text"]
                gold_result = extract_answer_detailed(ex["answer"])
                assert gold_result.method == "hash", (
                    f"example {i}: gold answer text did not contain '#### N' as expected "
                    f"-- dataset format assumption violated: {ex['answer']!r}"
                )
                model_result = extract_answer_detailed(model_text)
                correct = (
                    model_result.value is not None
                    and abs(model_result.value - gold_result.value) < 1e-6
                )
                record = {
                    "idx": i,
                    "question": ex["question"],
                    "gold_answer": gold_result.value,
                    "model_output": model_text,
                    "extracted_answer": model_result.value,
                    "extraction_method": model_result.method,
                    "correct": correct,
                }
                all_records.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                status = "PASS" if correct else "FAIL"
                print(
                    f"  [{status}] idx={i} gold={gold_result.value} "
                    f"extracted={model_result.value} (method={model_result.method})"
                )

    n = len(all_records)
    n_correct = sum(1 for r in all_records if r["correct"])
    n_fallback = sum(1 for r in all_records if r["extraction_method"] == "fallback_last_number")
    n_failed = sum(1 for r in all_records if r["extraction_method"] == "failed")

    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    print(f"Examples: {n}")
    print(f"Exact match (correct): {n_correct}/{n} ({100 * n_correct / n:.1f}%)")
    print(f"Extraction fallback used (last-number, no marker found): {n_fallback}/{n}")
    print(f"Extraction failure (no number found at all): {n_failed}/{n}")
    print(f"Per-batch wall-clock: {[f'{t:.2f}s' for t in batch_timings]}")
    print(f"Mean per-batch: {sum(batch_timings) / len(batch_timings):.2f}s, "
          f"mean per-example: {sum(batch_timings) / n:.2f}s")
    print(f"Results written incrementally to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
