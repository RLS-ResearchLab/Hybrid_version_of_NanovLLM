"""A4 -- fused-GDR (use_fused_gdr_kernel) GSM8K non-regression, real weights,
n=32-50 subset. Deferred from the single-GPU window because it needs real
weights (the fused kernel is only reached for real multi-token prefill --
see models/qwen3_5.py's `use_fused = self.use_fused_gdr_kernel and not
is_decode_shape`) and this engine has no CPU-offload path (LLMEngine/
ModelRunner require CUDA; the CPU-offload GSM8K scripts in this repo --
tests/gsm8k_cpu_offload_run.py -- deliberately bypass the engine entirely
via plain transformers/accelerate, so they can't exercise use_fused_gdr_kernel
at all, which is engine-internal, models/qwen3_5.py-only code).

NOT an accuracy-bar test. The correctness gate (87.5% exact-match,
tests/gsm8k_score.py) is separately known to be failing for an unrelated
reason (answer-termination behavior, see README's "Known limitations") --
that failure would show up identically flag-on or flag-off and is NOT what
this script checks. This is FLAG-OFF vs. FLAG-ON non-regression on the SAME
examples, nothing else: does turning use_fused_gdr_kernel on change accuracy
or the extraction-method distribution, relative to flag-off on this run.

Diffs TWO things, deliberately not collapsed into one number:
  1. Exact-match accuracy, flag-off vs. flag-on (the headline non-regression
     check).
  2. Extraction-method distribution (hash / answer_is / fallback_last_number
     / failed -- see gsm8k_extract.py) per flag value. A shifted
     marker-found rate with FLAT accuracy is still a meaningful signal per
     this task's own framing: it would mean the fused kernel changes
     *where* the model stops being clearly extractable, even if it doesn't
     change whether the underlying arithmetic was right -- worth flagging
     even though it wouldn't fail the non-regression bar by itself.

n=32-50: small subset (default 40, fixed seed matching gsm8k_full_run.py's
SUBSAMPLE_SEED=42 convention so results are comparable/reproducible), same
prompt construction (gsm8k_prompt.build_prompt) and scoring
(gsm8k_extract.extract_answer_detailed) as this project's other real-checkpoint
GSM8K scripts -- reused, not reinvented.

Usage (two runs, same examples, then diff -- can share ONE engine process,
generating both arms back to back, since both flag values load the SAME
checkpoint at the SAME tensor_parallel_size and there's no reason to pay the
~69GB load cost twice):

    python tests/cluster_a4_fused_gdr_gsm8k.py --tp 2 --num-examples 40

Dry run (small model, single GPU -- meaningless for accuracy, since
fake_qwen35_small was never trained on GSM8K, but DOES validate that
flag-on/flag-off produce equal-length, non-crashing, well-formed output
through the real multi-step decode path). PREREQUISITE, run once (needs
CUDA) before the first dry run -- see cluster_a2_tp_correctness.py's
module docstring for the full writeup:
    python tests/make_fake_hf_config.py     # already done if config.json exists
    python tests/make_fake_checkpoint.py    # writes model.safetensors -- REQUIRED
    python tests/make_fake_tokenizer.py     # already done if tokenizer.json exists
Without it every parameter is torch.empty() garbage rather than merely
untrained, and outputs can be non-finite, not just architecturally
meaningless.

    python tests/cluster_a4_fused_gdr_gsm8k.py --checkpoint tests/fake_qwen35_small \\
        --tp 1 --num-examples 8 --dry-run --fake-config-loader
"""
import argparse
import json
import os
import sys
import types
from collections import Counter
from time import perf_counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

sys.path.insert(0, os.path.dirname(__file__))
from gsm8k_prompt import build_prompt  # noqa: E402
from gsm8k_extract import extract_answer_detailed  # noqa: E402
from cluster_a2_tp_correctness import _maybe_install_fake_config_loader  # noqa: E402

DEFAULT_CKPT = os.path.join(ROOT, "qwen35_checkpoint")
CACHE_DIR = os.path.join(ROOT, "tests", "_cluster_day_cache", "a4_fused_gdr")
MAX_MODEL_LEN = 2048
MAX_TOKENS = 512
SUBSAMPLE_SEED = 42  # matches gsm8k_full_run.py's convention


def _select_indices(total: int, num_examples: int) -> list[int]:
    import random
    assert 0 < num_examples <= total
    rng = random.Random(SUBSAMPLE_SEED)
    all_indices = list(range(total))
    rng.shuffle(all_indices)
    return sorted(all_indices[:num_examples])


def _run_arm(llm, sp, examples, dry_run: bool) -> list[dict]:
    records = []
    batch_size = 8
    for b in range(0, len(examples), batch_size):
        batch = examples[b:b + batch_size]
        prompts = [build_prompt(ex["question"]) for ex in batch]
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        for ex, out in zip(batch, outputs):
            model_text = out["text"]
            model_result = extract_answer_detailed(model_text)
            record = {"idx": ex["idx"], "model_output": model_text,
                      "extracted_answer": model_result.value, "extraction_method": model_result.method}
            if not dry_run:
                gold_result = extract_answer_detailed(ex["answer"])
                assert gold_result.method == "hash", (
                    f"example {ex['idx']}: gold answer did not contain '#### N': {ex['answer']!r}"
                )
                record["gold_answer"] = gold_result.value
                record["correct"] = (
                    model_result.value is not None
                    and abs(model_result.value - gold_result.value) < 1e-6
                )
            records.append(record)
        print(f"    batch {b // batch_size + 1}/{(len(examples) + batch_size - 1) // batch_size} done")
    return records


def _run_and_cache(args, examples, use_fused: bool, results_path: str):
    if os.path.exists(results_path):
        print(f"  [{'on' if use_fused else 'off'}] results already cached at {results_path}, skipping run "
              f"(delete the file to force a re-run)")
        with open(results_path, encoding="utf-8") as f:
            return json.load(f)

    _maybe_install_fake_config_loader(args)
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    tag = "on" if use_fused else "off"
    print(f"\n[arm={tag}] loading engine (tensor_parallel_size={args.tp}, "
          f"use_fused_gdr_kernel={use_fused}) from {args.checkpoint} ...")
    llm = LLM(
        args.checkpoint,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=8,
        max_model_len=MAX_MODEL_LEN,
        use_fused_gdr_kernel=use_fused,
    )
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    t0 = perf_counter()
    records = _run_arm(llm, sp, examples, args.dry_run)
    elapsed = perf_counter() - t0
    print(f"[arm={tag}] {len(records)} example(s) in {elapsed:.1f}s")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[arm={tag}] saved to {results_path}")

    import atexit
    atexit.unregister(llm.exit)
    llm.exit()
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--num-examples", type=int, default=40,
                     help="n=32-50 per task; default 40. Fixed seed -> same subset every run.")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true", default=False,
                     help="Small fake model: skip gold-answer scoring (meaningless on random "
                          "untrained weights), only check flag-on/flag-off both run end-to-end "
                          "and produce well-formed, non-crashing output through real multi-step decode.")
    ap.add_argument("--fake-config-loader", action="store_true", default=False,
                     help="Required against tests/fake_qwen35_small (default OFF -- never needed "
                          "against the real checkpoint). See cluster_a2_tp_correctness.py's "
                          "_AttrDict docstring for why.")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    if args.dry_run:
        examples = [{"idx": i, "question": f"What is {i} plus {i + 1}?", "answer": f"#### {2 * i + 1}"}
                    for i in range(args.num_examples)]
    else:
        from datasets import load_dataset
        print("Loading openai/gsm8k (main, test split) ...")
        ds = load_dataset("openai/gsm8k", "main", split="test")
        if len(ds) != 1319:
            print(f"STOP: expected 1319 test examples, got {len(ds)}. Not proceeding.")
            sys.exit(1)
        indices = _select_indices(len(ds), args.num_examples)
        examples = [{"idx": i, "question": ds[i]["question"], "answer": ds[i]["answer"]} for i in indices]
        print(f"Subsampled {len(examples)}/{len(ds)} examples (seed={SUBSAMPLE_SEED}): {indices}")

    off_path = os.path.join(CACHE_DIR, f"n{args.num_examples}_tp{args.tp}_fused_off.json")
    on_path = os.path.join(CACHE_DIR, f"n{args.num_examples}_tp{args.tp}_fused_on.json")

    records_off = _run_and_cache(args, examples, use_fused=False, results_path=off_path)
    records_on = _run_and_cache(args, examples, use_fused=True, results_path=on_path)

    print("\n" + "=" * 78)
    print(f"A4 NON-REGRESSION -- fused-GDR flag-off vs. flag-on, n={len(examples)}, tp={args.tp}")
    print("=" * 78)

    by_idx_off = {r["idx"]: r for r in records_off}
    by_idx_on = {r["idx"]: r for r in records_on}
    common = sorted(set(by_idx_off) & set(by_idx_on))
    assert common == sorted(by_idx_off) == sorted(by_idx_on), "index mismatch between the two arms"

    if args.dry_run:
        n_finite = sum(1 for i in common if by_idx_off[i]["model_output"] and by_idx_on[i]["model_output"])
        print(f"DRY-RUN CHECK: {n_finite}/{len(common)} examples produced non-empty output in BOTH arms.")
        assert n_finite == len(common), "at least one example produced empty output in some arm"
        print("DRY-RUN CHECK: PASS -- flag-off and flag-on both run end-to-end without crashing.")
        return

    n_off_correct = sum(1 for i in common if by_idx_off[i]["correct"])
    n_on_correct = sum(1 for i in common if by_idx_on[i]["correct"])
    acc_off = 100 * n_off_correct / len(common)
    acc_on = 100 * n_on_correct / len(common)
    print(f"\nACCURACY  flag-off: {n_off_correct}/{len(common)} ({acc_off:.1f}%)   "
          f"flag-on: {n_on_correct}/{len(common)} ({acc_on:.1f}%)   delta: {acc_on - acc_off:+.1f}pp")

    method_off = Counter(by_idx_off[i]["extraction_method"] for i in common)
    method_on = Counter(by_idx_on[i]["extraction_method"] for i in common)
    print("\nEXTRACTION-METHOD DISTRIBUTION")
    print(f"  {'method':<22} {'flag-off':>10} {'flag-on':>10}")
    for method in ["hash", "answer_is", "fallback_last_number", "failed"]:
        print(f"  {method:<22} {method_off.get(method, 0):>10} {method_on.get(method, 0):>10}")

    flipped = [i for i in common if by_idx_off[i]["correct"] != by_idx_on[i]["correct"]]
    print(f"\nPER-EXAMPLE FLIPS (correct changed between arms): {len(flipped)}/{len(common)}")
    for i in flipped:
        print(f"  idx={i}: off correct={by_idx_off[i]['correct']} (method={by_idx_off[i]['extraction_method']})  "
              f"on correct={by_idx_on[i]['correct']} (method={by_idx_on[i]['extraction_method']})")

    print("\n" + "-" * 78)
    print("READ (per this task's framing -- NOT an accuracy-bar test):")
    if abs(acc_on - acc_off) < 1e-9 and method_off == method_on:
        print("  IDENTICAL accuracy AND extraction-method distribution -- cleanest possible "
              "non-regression result.")
    elif abs(n_on_correct - n_off_correct) <= max(1, len(common) // 20) and method_off == method_on:
        print("  Accuracy within a small margin (<=5% of n) and extraction-method distribution "
              "UNCHANGED -- consistent with ordinary sampling noise from a code path already "
              "known to reassociate floating point differently (see models/qwen3_5.py's fused "
              "kernel docstring); no distribution-level signal of a regression.")
    elif method_off != method_on and abs(n_on_correct - n_off_correct) <= max(1, len(common) // 20):
        print("  Extraction-method distribution SHIFTED while accuracy stayed flat -- per this "
              "task's own framing, this IS a meaningful signal even though it wouldn't fail a "
              "pure accuracy-delta bar: worth reporting and investigating which examples moved "
              "and why (see PER-EXAMPLE FLIPS and the two arms' raw .json files), not silently "
              "waved through as 'accuracy is fine'.")
    else:
        print("  Accuracy delta is NOT small -- treat as a real candidate regression, not noise. "
              "Investigate the flipped examples above before shipping use_fused_gdr_kernel=True.")

    print(f"\nRaw per-example output saved at:\n  {off_path}\n  {on_path}")


if __name__ == "__main__":
    main()
