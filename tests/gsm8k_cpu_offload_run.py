"""Standalone, engine-INDEPENDENT diagnostic: does enable_thinking=False fix the
GSM8K no-marker-found failure mode, run via plain `transformers` + `accelerate`
CPU/GPU offload instead of the nano-vLLM engine?

WHY THIS SCRIPT EXISTS, SEPARATELY FROM tests/gsm8k_full_run.py:
The real checkpoint (~69GB bf16) does not fit on the single available A6000
(48GB) alongside anything else, and this diagnostic needs to run in the
BACKGROUND, unattended, for hours, while GPU-active engine work (QLLM) happens
separately. It intentionally does NOT touch engine/, ModelRunner, LLMEngine,
StateManager, the scheduler, or models/qwen3_5.py's nano-vLLM integration --
only plain `transformers.AutoModelForCausalLM.from_pretrained(...,
device_map="auto")`, which lets `accelerate` split the model across GPU VRAM
and system RAM automatically. It reuses the EXACT SAME prompt-construction
functions as the engine-path script (build_prompt() / build_chat_messages()
from gsm8k_prompt.py), and the exact same 32 example indices where possible,
so results are comparable -- but the two paths are NOT expected to produce
numerically identical output (different attention/MoE kernels, different
batching, no CUDA graphs here). Treat this as a SIMILAR, independent
data point, not a byte-for-byte cross-check of the engine.

STOP-STRING CAVEAT: GSM8K_STOP_PATTERNS (gsm8k_extract.py) are regex
patterns, and HF `generate()`'s built-in `stop_strings=` only matches literal
strings, not regex -- so this script implements a small custom
StoppingCriteria (see RegexStopCriteria below) that mirrors
engine/scheduler.py's `_check_stop_string` as closely as possible without
importing engine code: decode a bounded trailing window of newly-generated
tokens each step, regex-search (case-insensitive) for GSM8K_STOP_PATTERNS,
stop on first match. This is read from engine/scheduler.py, not imported
from it. One real difference: the engine checks this once per DECODE STEP
across a batch; this script generates one example at a time (batch_size=1
throughout -- deliberate, see "Why batch_size=1" below), so the check instead
runs once per generated token via HF's per-step StoppingCriteria callback.
Behavior should match; if a completion's stop point ever looks off, compare
against gsm8k_answer_position_check.py's engine-side validation before
trusting either as ground truth.

WHY BATCH_SIZE=1 THROUGHOUT: HF's `StoppingCriteria` returns one bool for
the WHOLE batch call, not per-sequence -- there's no clean way to let one
GSM8K example in a batch stop early on its own stop pattern while its
batch-mates keep decoding, the way engine/scheduler.py's per-sequence
`postprocess()` does. Rather than approximate that (padding + a
worse-than-engine "stop whole batch when ANY sequence matches" behavior),
this script sidesteps it entirely by generating one example at a time.
Slower, but this run doesn't need to be fast (see module purpose above) and
correctness/comparability matters more here than throughput.

USAGE:
    Sanity check FIRST, always -- confirms the checkpoint loads, the device
    map looks reasonable (especially for MoE expert params, which naive
    accelerate splitting can get wrong), and one real example produces
    finite, non-garbage output. Review this output BEFORE launching the full
    background run -- see this repo's README / the calling conversation for
    the review step.

        python tests/gsm8k_cpu_offload_run.py --checkpoint qwen35_checkpoint --sanity-check

    Full 3-arm run (only after the sanity check above looks right), backgrounded:

        nohup python -u tests/gsm8k_cpu_offload_run.py \\
            --checkpoint qwen35_checkpoint \\
            --arms raw chat-think chat-no-think \\
            --num-examples 32 \\
            --output-dir tests/_gsm8k_cache/cpu_offload/ \\
            > tests/_gsm8k_cache/cpu_offload/run.log 2>&1 &
        echo "Started, PID: $!"

    Checking on it later:
        tail -f tests/_gsm8k_cache/cpu_offload/run.log     # live progress
        ps -p <PID>                                         # still running?

    Resumes automatically per-arm if killed partway (same convention as
    gsm8k_full_run.py): each arm's JSONL is read first, completed indices
    are skipped, only what's left for that arm is generated.

    Scoring, once results exist (partial results are fine to score):
        python tests/gsm8k_cpu_offload_score.py --output-dir tests/_gsm8k_cache/cpu_offload/
"""
import argparse
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))  # tests/ -- for gsm8k_prompt.py / gsm8k_extract.py only
from gsm8k_prompt import build_prompt, build_chat_messages  # noqa: E402
from gsm8k_extract import extract_answer_detailed, GSM8K_STOP_PATTERNS  # noqa: E402

DEFAULT_CACHE_DIR = os.path.join(ROOT, "tests", "_gsm8k_cache", "cpu_offload")
ENGINE_BASELINE_32_PATH = os.path.join(ROOT, "tests", "_gsm8k_cache", "full_results_n32_mt706_stop.jsonl")
EXPECTED_TOTAL = 1319
SUBSAMPLE_SEED = 42  # mirrors gsm8k_full_run.py's SUBSAMPLE_SEED exactly -- same seed, same
                      # random.Random(seed).shuffle() call, so an independent re-derivation
                      # lands on the identical index set (verified against ENGINE_BASELINE_32_PATH
                      # below; recover_indices() prefers reading that file directly when it exists,
                      # this is only the fallback for when it doesn't).
ARM_CHOICES = ["raw", "chat-think", "chat-no-think"]

# Mirrors engine/scheduler.py's _STOP_CHECK_WINDOW_TOKENS -- read from that file, not
# imported (this script must not depend on engine/). Same reasoning applies: bounds the
# cost of the stop check to a fixed decode+scan per step instead of growing with
# completion length.
_STOP_CHECK_WINDOW_TOKENS = 32


def select_indices_seeded(total: int, num_examples: int, seed: int = SUBSAMPLE_SEED) -> list:
    rng = random.Random(seed)
    all_indices = list(range(total))
    rng.shuffle(all_indices)
    return sorted(all_indices[:num_examples])


def recover_indices(total: int, num_examples: int) -> list:
    """Prefer the LITERAL index list the engine-path 32-example run actually used
    (read straight from its cached results file) over re-deriving it, so this script
    is reusing the exact same problems for comparability, not just a same-seed
    approximation that assumes nothing about dataset loading differs between paths."""
    if num_examples == 32 and os.path.exists(ENGINE_BASELINE_32_PATH):
        idxs = set()
        with open(ENGINE_BASELINE_32_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    idxs.add(json.loads(line)["idx"])
        if len(idxs) == num_examples:
            print(f"Reusing the exact {num_examples} example indices from {ENGINE_BASELINE_32_PATH}")
            return sorted(idxs)
        print(
            f"WARNING: {ENGINE_BASELINE_32_PATH} exists but has {len(idxs)} distinct indices, "
            f"not {num_examples} -- falling back to seeded selection instead of trusting it."
        )
    print(
        f"Falling back to seeded selection (seed={SUBSAMPLE_SEED}), matching "
        f"gsm8k_full_run.py's _select_indices exactly -- verified to reproduce the same "
        f"32-example set as {os.path.basename(ENGINE_BASELINE_32_PATH)} when that file is absent."
    )
    return select_indices_seeded(total, num_examples)


def _results_path(output_dir: str, num_examples: int, arm: str) -> str:
    return os.path.join(output_dir, f"cpu_offload_n{num_examples}_{arm}.jsonl")


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


def build_arm_prompt_ids(tokenizer, arm: str, question: str):
    """Returns a 1D list[int] of prompt token ids for the given arm. Uses the SAME
    build_prompt()/build_chat_messages() content as gsm8k_full_run.py -- see this
    module's docstring for why that matters for comparability."""
    if arm == "raw":
        text = build_prompt(question)
        return tokenizer(text, add_special_tokens=True)["input_ids"]
    elif arm == "chat-think":
        return tokenizer.apply_chat_template(
            build_chat_messages(question), tokenize=True, add_generation_prompt=True,
        )
    elif arm == "chat-no-think":
        return tokenizer.apply_chat_template(
            build_chat_messages(question), tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
    raise ValueError(f"unknown arm {arm!r}")


class RegexStopCriteria:
    """One instance per generate() call (batch_size=1 -- see module docstring for why).
    Decodes a bounded trailing window of newly-generated tokens each step and
    regex-searches (case-insensitive) for GSM8K_STOP_PATTERNS, mirroring
    engine/scheduler.py's _check_stop_string. HF's StoppingCriteria protocol calls
    __call__(input_ids, scores, **kwargs) with the FULL sequence-so-far (prompt +
    generated) each step; prompt_len lets us only ever look at the generated tail,
    same as the engine scanning seq.completion_token_ids, never seq.token_ids (the
    GSM8K prompts contain "The answer is N." literally in the exemplars -- scanning
    the prompt would false-trigger before generation even starts)."""

    def __init__(self, tokenizer, prompt_len: int, patterns=GSM8K_STOP_PATTERNS):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.matched_text = None
        self.matched_pattern = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        seq_len = input_ids.shape[1]
        if seq_len <= self.prompt_len:
            return False
        window_start = max(self.prompt_len, seq_len - _STOP_CHECK_WINDOW_TOKENS)
        window_ids = input_ids[0, window_start:seq_len].tolist()
        window_text = self.tokenizer.decode(window_ids, skip_special_tokens=True)
        for p in self.patterns:
            m = p.search(window_text)
            if m:
                self.matched_pattern = p.pattern
                self.matched_text = m.group(0)
                return True
        return False


def generate_one(model, tokenizer, prompt_ids: list, max_new_tokens: int, use_stop_strings: bool):
    import torch

    if isinstance(prompt_ids, str):
        prompt_ids = tokenizer.encode(prompt_ids, add_special_tokens=False)
    elif hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    if len(prompt_ids) and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    attention_mask = torch.ones_like(input_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
    )
    stop_criterion = None
    if use_stop_strings:
        from transformers import StoppingCriteriaList

        stop_criterion = RegexStopCriteria(tokenizer, prompt_len=input_ids.shape[1])
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_criterion])

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    new_tokens = out[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    stopped_on_pattern = stop_criterion is not None and stop_criterion.matched_text is not None
    return text, stopped_on_pattern, int(new_tokens.shape[0])


def run_sanity_check(args):
    print("=" * 70)
    print("SANITY CHECK -- load model, dump device map, run ONE example.")
    print("This does NOT proceed to the full run. Review this output first.")
    print("=" * 70)

    try:
        import accelerate  # noqa: F401
    except ImportError:
        print(
            "STOP: `accelerate` is not importable. device_map=\"auto\" CPU offload "
            "requires it. This is a real blocker -- install it "
            "(`pip install accelerate`, already in requirements.txt) before proceeding. "
            "Not routing around this."
        )
        sys.exit(1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading tokenizer from {args.checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    print("Tokenizer loaded OK.")

    print(f"\nLoading model from {args.checkpoint} (torch_dtype=bfloat16, device_map='auto') ...")
    print("This offloads across GPU VRAM + system RAM automatically via accelerate -- "
          "if system RAM is also insufficient, the error below will typically mention "
          "disk offload or an OOM during weight materialization, not just CUDA OOM. "
          "If that happens, STOP and report it -- do not add offload_folder/disk-offload "
          "workarounds without approval.")
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
    except Exception as e:
        print(f"\nSTOP: model load FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
        print(
            "If this looks like insufficient system RAM (not just GPU VRAM) for the "
            "offload split, that is a real hardware blocker to report back, not "
            "something to silently work around here."
        )
        raise
    model.eval()
    print(f"Model loaded OK in {time.time() - t0:.1f}s.")

    print("\n--- Device map ---")
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        print(
            "WARNING: model has no hf_device_map attribute -- device_map='auto' may not "
            "have actually engaged accelerate's dispatch. Investigate before trusting "
            "anything below."
        )
    else:
        devices_used = sorted({str(v) for v in device_map.values()})
        print(f"Total device_map entries: {len(device_map)}")
        print(f"Distinct devices used: {devices_used}")
        expert_entries = {k: v for k, v in device_map.items() if "expert" in k.lower()}
        print(f"\nEntries containing 'expert' ({len(expert_entries)}):")
        if not expert_entries:
            print(
                "  NONE FOUND -- if this checkpoint's MoE layers are named something "
                "other than '...expert...', this filter is missing them; grep "
                "device_map's full key list manually before assuming experts are placed "
                "sanely."
            )
        else:
            for k in sorted(expert_entries):
                print(f"  {k} -> {expert_entries[k]}")
            expert_devices = sorted({str(v) for v in expert_entries.values()})
            print(f"\nExpert params span {len(expert_devices)} distinct device(s): {expert_devices}")
            if len(expert_devices) > 1:
                print(
                    "  NOTE: experts split across multiple devices is not necessarily wrong "
                    "(accelerate CAN split a single layer's weight-heavy MoE block), but it's "
                    "worth eyeballing whether the split looks like it cut a single "
                    "logical expert-parameter tensor in half vs. cleanly assigning whole "
                    "experts/layers to devices -- the former is more likely to be a real "
                    "correctness risk than the latter."
                )

    print("\n--- Single example, forward-pass finite check ---")
    ds = None
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"WARNING: could not load openai/gsm8k ({e}); using a hardcoded sanity question instead.")

    if ds is not None and len(ds) == EXPECTED_TOTAL:
        question = ds[args.sanity_idx]["question"]
        print(f"Using real GSM8K example idx={args.sanity_idx}: {question[:120]}...")
    else:
        question = "There are 15 trees in the grove. Grove workers plant more, and now there are 21 trees. How many trees did the workers plant?"
        print(f"Using hardcoded fallback question: {question}")

    prompt_ids = build_arm_prompt_ids(tokenizer, "chat-no-think", question)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        fwd_out = model(input_ids=input_ids)
    logits = fwd_out.logits
    all_finite = bool(torch.isfinite(logits).all().item())
    print(f"Forward-pass logits shape: {tuple(logits.shape)}, all finite: {all_finite}")
    if not all_finite:
        n_nan = int(torch.isnan(logits).sum().item())
        n_inf = int(torch.isinf(logits).sum().item())
        print(f"STOP CANDIDATE: {n_nan} NaN, {n_inf} Inf values in logits. Do not proceed to the full run.")

    print("\n--- Single example, full generate() (chat-no-think arm, max_new_tokens={}) ---".format(args.max_tokens))
    t0 = time.time()
    text, stopped_on_pattern, n_new = generate_one(
        model, tokenizer, prompt_ids, args.max_tokens, use_stop_strings=args.stop_strings,
    )
    elapsed = time.time() - t0
    result = extract_answer_detailed(text)
    print(f"Generated {n_new} new tokens in {elapsed:.1f}s, stopped_on_pattern={stopped_on_pattern}")
    print(f"extraction_method={result.method}, extracted_value={result.value}")
    print("\n--- Full generated text ---")
    print(text)
    print("\n" + "=" * 70)
    print("SANITY CHECK COMPLETE. Review the device map and generated text above.")
    print("If this looks right, launch the background full run per this script's")
    print("module docstring. If anything looks wrong, STOP here.")
    print("=" * 70)


def run_full(args):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading openai/gsm8k (main, test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(ds)} examples (expect {EXPECTED_TOTAL}).")
    if len(ds) != EXPECTED_TOTAL:
        print(f"STOP: expected {EXPECTED_TOTAL} test examples, got {len(ds)}. Not proceeding.")
        sys.exit(1)

    selected_indices = recover_indices(len(ds), args.num_examples)
    print(f"Selected {len(selected_indices)} example indices: {selected_indices}")

    print(f"\nLoading tokenizer + model from {args.checkpoint} "
          f"(torch_dtype=bfloat16, device_map='auto') ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s. Beginning generation.\n")

    for arm in args.arms:
        results_path = _results_path(args.output_dir, args.num_examples, arm)
        completed = _load_completed(results_path)
        n_correct = sum(1 for r in completed.values() if r["correct"])
        n_fallback = sum(1 for r in completed.values() if r["extraction_method"] == "fallback_last_number")
        remaining = [i for i in selected_indices if i not in completed]
        print(f"[arm={arm}] resuming: {len(completed)}/{len(selected_indices)} already done "
              f"in {results_path} ({n_correct} correct, {n_fallback} fallback so far)")
        if not remaining:
            print(f"[arm={arm}] nothing left to do.\n")
            continue

        with open(results_path, "a", encoding="utf-8") as f:
            for i, idx in enumerate(remaining):
                ex = ds[idx]
                question = ex["question"]
                t0 = time.time()
                prompt_ids = build_arm_prompt_ids(tokenizer, arm, question)
                try:
                    text, stopped_on_pattern, n_new = generate_one(
                        model, tokenizer, prompt_ids, args.max_tokens, use_stop_strings=args.stop_strings,
                    )
                except Exception:
                    print(
                        f"[arm={arm}] ERROR generating idx={idx} ({i + 1}/{len(remaining)} this run) -- "
                        f"results up to this point are safely checkpointed in {results_path}; "
                        f"re-run this script to resume."
                    )
                    raise
                elapsed = time.time() - t0

                gold_result = extract_answer_detailed(ex["answer"])
                assert gold_result.method == "hash", (
                    f"example {idx}: gold answer text did not contain '#### N' as expected: {ex['answer']!r}"
                )
                model_result = extract_answer_detailed(text)
                correct = (
                    model_result.value is not None
                    and abs(model_result.value - gold_result.value) < 1e-6
                )

                record = {
                    "idx": idx,
                    "arm": arm,
                    "question": question,
                    "gold_answer": gold_result.value,
                    "model_output": text,
                    "extracted_answer": model_result.value,
                    "extraction_method": model_result.method,
                    "correct": correct,
                    "stopped_on_pattern": stopped_on_pattern,
                    "n_new_tokens": n_new,
                    "gen_seconds": elapsed,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

                n_correct += int(correct)
                n_fallback += int(model_result.method == "fallback_last_number")
                done = len(completed) + i + 1
                print(
                    f"[arm={arm}] idx={idx} ({done}/{len(selected_indices)}) "
                    f"correct={correct} method={model_result.method} "
                    f"n_new_tokens={n_new} elapsed={elapsed:.1f}s | "
                    f"running: correct={n_correct}/{done} fallback={n_fallback}/{done}",
                    flush=True,
                )

        print(f"[arm={arm}] DONE. Results in {results_path}.\n")

    print("ALL ARMS COMPLETE. Run tests/gsm8k_cpu_offload_score.py "
          f"--output-dir {args.output_dir} for a comparison table.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to the real checkpoint directory.")
    parser.add_argument("--arms", nargs="+", choices=ARM_CHOICES, default=list(ARM_CHOICES))
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--output-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--max-tokens", type=int, default=706,
        help="Default 706 -- matches full_results_n32_mt706_stop.jsonl, the engine-path "
             "run this script's results are meant to be compared against.",
    )
    parser.add_argument(
        "--stop-strings", action=argparse.BooleanOptionalAction, default=True,
        help="Default True, matching full_results_n32_mt706_stop.jsonl's '_stop' suffix "
             "(that run used --stop-strings). See RegexStopCriteria and this module's "
             "docstring for how this is implemented here vs. in the engine.",
    )
    parser.add_argument(
        "--sanity-check", action="store_true",
        help="Load the model, dump the device map (with special attention to MoE expert "
             "params), run one forward pass finite-check, and run ONE full generate() call. "
             "Prints everything and EXITS -- does not proceed to the full run. Always do "
             "this first and review the output before launching the background full run.",
    )
    parser.add_argument(
        "--sanity-idx", type=int, default=15,
        help="Which GSM8K test-set index to use for --sanity-check (default 15, the first "
             "no-marker-found example from the investigation that motivated this script).",
    )
    args = parser.parse_args()

    if args.sanity_check:
        run_sanity_check(args)
        return

    run_full(args)


if __name__ == "__main__":
    main()
