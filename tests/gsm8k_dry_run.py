"""STEP 2 -- prompt construction + tokenization dry run. NO GPU needed (only
the tokenizer is loaded, not the model): confirms the 8-shot GSM8K prompt +
expected completion length comfortably fits under max_model_len=2048 BEFORE
any GPU time is spent on Step 3.

max_tokens budget: 512. Justification -- the 8-shot exemplars in
gsm8k_prompt.py average ~55 words / ~3 arithmetic steps each; GSM8K test
questions are drawn from the same distribution (grade-school multi-step word
problems), so a full CoT trace + "The answer is N." typically runs well
under 200 tokens. 512 leaves >2x headroom for a verbose 35B model without
being so large it wastes wall-clock on runaway generations.

Usage (run on the machine with the real checkpoint's tokenizer files):
    python tests/gsm8k_dry_run.py
"""
import os
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

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")
MAX_MODEL_LEN = 2048
MAX_TOKENS = 512
NUM_SAMPLE_PROMPTS = 5


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {CKPT_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)

    print("Loading openai/gsm8k (main, test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(ds)} examples (expect 1319).")
    if len(ds) != 1319:
        print(
            f"STOP: expected 1319 test examples, got {len(ds)} -- wrong split/config "
            f"was loaded. Do not proceed."
        )
        sys.exit(1)

    print(f"\nBuilding {NUM_SAMPLE_PROMPTS} real prompts from real test-set questions:\n")
    worst_total = 0
    worst_idx = None
    print(f"{'#':>3}  {'prompt_tokens':>13}  {'+max_tokens':>11}  {'total':>7}  {'margin_to_2048':>15}")
    print("-" * 60)
    for i in range(NUM_SAMPLE_PROMPTS):
        question = ds[i]["question"]
        prompt = build_prompt(question)
        token_ids = tokenizer.encode(prompt)
        n_prompt = len(token_ids)
        total = n_prompt + MAX_TOKENS
        margin = MAX_MODEL_LEN - total
        print(f"{i:>3}  {n_prompt:>13}  {MAX_TOKENS:>11}  {total:>7}  {margin:>15}")
        if total > worst_total:
            worst_total = total
            worst_idx = i

    print()
    print(f"Worst case among these {NUM_SAMPLE_PROMPTS}: prompt #{worst_idx}, "
          f"total={worst_total} tokens, margin={MAX_MODEL_LEN - worst_total} under max_model_len={MAX_MODEL_LEN}")
    if worst_total >= MAX_MODEL_LEN:
        print(
            "STOP: prompt + max_tokens does NOT fit under max_model_len=2048 -- "
            "max_model_len needs revisiting before any GPU time is spent."
        )
        sys.exit(1)
    else:
        print("OK: comfortably fits with margin. Safe to proceed to Step 3.")

    print("\nFirst prompt, full text (sanity-check formatting by eye):")
    print("=" * 78)
    print(build_prompt(ds[0]["question"]))
    print("=" * 78)


if __name__ == "__main__":
    main()
