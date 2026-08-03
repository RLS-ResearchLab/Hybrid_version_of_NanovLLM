"""GSM8K decode-loop timing benchmark -- splits wall-clock into PREFILL vs
DECODE time (by hooking LLMEngine.step()'s own is_prefill signal directly,
not llm.generate()'s convenience wrapper), and reports how many tokens each
example ACTUALLY generated before finishing (EOS vs the 512-token cap), on
a small subset. Answers, with real numbers, the two directly-measurable
questions behind the ~7.5h full-run estimate:
  - Is time dominated by decode (sequential, up to 512 steps/example) or
    prefill (the shared ~800-token 8-shot exemplar prefix)?
  - Do examples typically run close to the 512-token cap, or stop much
    earlier via EOS?

PREFIX CACHING NOTE: as of this session's bug-fix work, prefix-cache reuse
is DELIBERATELY, PERMANENTLY DISABLED whenever a StateManager is active
(engine/block_manager.py's disable_prefix_cache flag) -- Scheduler.schedule()
unconditionally resets recurrent/conv state to zero on any fresh
block_table allocation, so reusing cached KV for a shared prefix while
resetting state produced a confirmed real bug (a degenerate repetition
loop on a partially-cached sequence). This means every batch here
recomputes the full exemplar prefix from scratch in prefill -- not a bug,
a deliberate, already-decided correctness trade-off. This benchmark
quantifies exactly what that costs, it does not question whether it's
happening.

OUT OF SCOPE (needs real profiling, not a wall-clock split, to quantify
without guessing): CUDA graphs vs enforce_eager delta, PCIe/no-NVLink
per-step all-reduce/all-to-all cost, batch-size scaling/GPU saturation.
This script reports the two numbers a simple timer CAN answer honestly.

Usage:
    python tests/gsm8k_timing_benchmark.py
"""
import os
import statistics
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

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")
MAX_MODEL_LEN = 2048
MAX_TOKENS = 512
BATCH_SIZE = 8
NUM_EXAMPLES = 24  # 3 batches -- same scale as gsm8k_smoke_test.py
EXPECTED_TOTAL = 1319
FULL_RUN_BATCHES = 165


def run_batch_with_timing(llm, prompts, sp):
    for prompt in prompts:
        llm.add_request(prompt, sp)
    prefill_time = 0.0
    decode_time = 0.0
    n_prefill_steps = 0
    n_decode_steps = 0
    outputs = {}
    while not llm.is_finished():
        t0 = perf_counter()
        output, num_tokens = llm.step()
        dt = perf_counter() - t0
        if num_tokens > 0:  # matches LLMEngine.generate()'s own prefill/decode signal
            prefill_time += dt
            n_prefill_steps += 1
        else:
            decode_time += dt
            n_decode_steps += 1
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
    return outputs, prefill_time, decode_time, n_prefill_steps, n_decode_steps


def main():
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
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    n_batches = (NUM_EXAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    all_completion_lengths = []
    batch_summaries = []

    for b in range(n_batches):
        lo, hi = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, NUM_EXAMPLES)
        prompts = [build_prompt(ds[i]["question"]) for i in range(lo, hi)]

        t0 = perf_counter()
        outputs, prefill_time, decode_time, n_prefill_steps, n_decode_steps = run_batch_with_timing(
            llm, prompts, sp
        )
        total_time = perf_counter() - t0

        lengths = [len(tok_ids) for tok_ids in outputs.values()]
        all_completion_lengths.extend(lengths)
        batch_summaries.append({
            "total_time": total_time, "prefill_time": prefill_time, "decode_time": decode_time,
            "n_prefill_steps": n_prefill_steps, "n_decode_steps": n_decode_steps,
            "completion_lengths": lengths,
        })

        print(f"\n=== Batch {b + 1}/{n_batches} (examples {lo}-{hi - 1}) ===")
        print(f"  total={total_time:.1f}s  prefill={prefill_time:.1f}s ({100 * prefill_time / total_time:.1f}%)  "
              f"decode={decode_time:.1f}s ({100 * decode_time / total_time:.1f}%)")
        print(f"  prefill steps={n_prefill_steps}  decode steps={n_decode_steps}  (cap={MAX_TOKENS})")
        print(f"  per-example completion lengths: {lengths}")
        print(f"  mean={statistics.mean(lengths):.1f}  median={statistics.median(lengths):.1f}  "
              f"max={max(lengths)}  min={min(lengths)}")

    print("\n" + "=" * 78)
    print("OVERALL SUMMARY")
    print("=" * 78)
    total_wall = sum(s["total_time"] for s in batch_summaries)
    total_prefill = sum(s["prefill_time"] for s in batch_summaries)
    total_decode = sum(s["decode_time"] for s in batch_summaries)
    print(f"Total wall-clock: {total_wall:.1f}s over {n_batches} batches "
          f"({total_wall / n_batches:.1f}s/batch average)")
    print(f"Prefill: {total_prefill:.1f}s ({100 * total_prefill / total_wall:.1f}%)")
    print(f"Decode:  {total_decode:.1f}s ({100 * total_decode / total_wall:.1f}%)")
    print(f"Decode steps taken per batch: {[s['n_decode_steps'] for s in batch_summaries]}  (cap={MAX_TOKENS})")
    print(f"Completion length across all {len(all_completion_lengths)} examples: "
          f"mean={statistics.mean(all_completion_lengths):.1f}  "
          f"median={statistics.median(all_completion_lengths):.1f}  "
          f"max={max(all_completion_lengths)}  min={min(all_completion_lengths)}  (cap={MAX_TOKENS})")
    pct_at_cap = 100 * sum(1 for l in all_completion_lengths if l >= MAX_TOKENS) / len(all_completion_lengths)
    print(f"Examples that hit the {MAX_TOKENS}-token cap without EOS: {pct_at_cap:.1f}%")

    print()
    est_hours = (total_wall / n_batches) * FULL_RUN_BATCHES / 3600
    print(f"Rough extrapolation to the full {EXPECTED_TOTAL}-example run ({FULL_RUN_BATCHES} batches), "
          f"using this subset's per-batch average: {est_hours:.2f} hours")
    print(f"NOTE: n={NUM_EXAMPLES} is small -- treat as a rough estimate, not a tight bound. This run has "
          f"BOTH correctness fixes active (rank-dispatched state + prefix-cache disabled), so the prefill "
          f"time above reflects the real, now-permanent per-batch cost of recomputing the exemplar prefix "
          f"from scratch -- not a stale/optimistic number from before those fixes.")


if __name__ == "__main__":
    main()
