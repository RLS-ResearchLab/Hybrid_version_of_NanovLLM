"""Isolated test: does crossing a 2nd KV cache block (block_size=256) during
decode produce consistent output, completely independent of preemption?

No scarcity, no forced eviction — just one sequence, generous KV cache,
generated long enough (300 tokens) to definitely need a 2nd block. Run the
IDENTICAL scenario twice (fresh engine each time, same seed, same prompt) and
check the two runs produce byte-identical tokens. Since sampling is patched
to plain greedy argmax (deterministic), any divergence between two runs of
the exact same setup means there's a real bug somewhere in the multi-block
code path (slot_mapping/block_tables construction once block_table has 2+
entries, or how flash_attn_with_kvcache consumes a multi-block block_tables
tensor) — not a preemption-recompute bug, since preemption never fires here.

Usage:
    python tests/make_fake_hf_config.py   # if not already run
    python tests/test_qwen35_multiblock.py
"""

import sys
import os
import json
import torch
import torch.nn as nn

# --- Path & virtual nanovllm package setup ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm":
    import types
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

# --- Determinism: greedy argmax sampler ---
import nanovllm.layers.sampler as sampler_mod
sampler_mod.Sampler.forward = lambda self, logits, temperatures: logits.float().argmax(dim=-1)

# --- Fake model loading / config / tokenizer (same as the preemption test) ---
import nanovllm.utils.loader as loader_mod
loader_mod.load_model = lambda model, path: None

import nanovllm.config as config_mod


class AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _fake_from_pretrained(path, *args, **kwargs):
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d


config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)


class DummyTokenizer:
    eos_token_id = 999999

    def encode(self, prompt):
        if isinstance(prompt, list):
            return prompt
        return [int(x) for x in prompt.split() if x.isdigit()]

    def decode(self, token_ids):
        return " ".join(str(t) for t in token_ids)


import transformers
transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *args, **kwargs: DummyTokenizer())

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


def init_deterministic_weights(engine: LLMEngine, seed: int = 42):
    with torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = engine.model_runner.model
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(param)


def run_once(fake_dir, prompt, max_tokens, seed):
    engine = LLMEngine(
        fake_dir,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=512,
        gpu_memory_utilization=0.5,   # generous, no scarcity, no preemption expected
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    try:
        init_deterministic_weights(engine, seed=seed)

        preempt_count = [0]
        orig_preempt = engine.scheduler.preempt
        def count_preempt(seq):
            preempt_count[0] += 1
            return orig_preempt(seq)
        engine.scheduler.preempt = count_preempt

        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=1.0)
        result = engine.generate([prompt], sampling_params, use_tqdm=False)
        tokens = result[0]["token_ids"]

        # Sanity: confirm this sequence actually crossed into a 2nd KV block,
        # i.e. this test is exercising what it claims to exercise.
        total_len = len(prompt) + len(tokens)
        block_size = engine.model_runner.config.kvcache_block_size
        crossed_block_boundary = total_len > block_size

        return tokens, preempt_count[0], crossed_block_boundary, total_len, block_size
    finally:
        engine.exit()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Requires CUDA (flash-attn, paged KV cache)"

    fake_dir = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
    assert os.path.isdir(fake_dir), "Run tests/make_fake_hf_config.py first"

    prompt = [10, 11, 12, 13, 14, 15, 16, 17]
    max_tokens = 300  # forces total length past 256 (block_size), needs 2nd block

    print("=" * 70)
    print("ISOLATED MULTI-BLOCK TEST: single sequence, no scarcity, no preemption")
    print("=" * 70)

    print("\n--- RUN 1 ---")
    tokens_1, preempt_1, crossed_1, len_1, block_size = run_once(fake_dir, prompt, max_tokens, seed=42)
    print(f"  Preemption count: {preempt_1} (expected 0 — no scarcity here)")
    print(f"  Total sequence length: {len_1}, block_size: {block_size}, crossed boundary: {crossed_1}")
    assert crossed_1, (
        f"Sequence length {len_1} did not exceed block_size {block_size} — "
        f"this test isn't actually exercising the 2nd-block path. Increase max_tokens."
    )
    assert preempt_1 == 0, f"Unexpected preemption in a generous-memory run: {preempt_1}"

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n--- RUN 2 (identical scenario) ---")
    tokens_2, preempt_2, crossed_2, len_2, _ = run_once(fake_dir, prompt, max_tokens, seed=42)
    print(f"  Preemption count: {preempt_2} (expected 0)")
    print(f"  Total sequence length: {len_2}, crossed boundary: {crossed_2}")
    assert preempt_2 == 0, f"Unexpected preemption in a generous-memory run: {preempt_2}"

    print("\n--- Comparing Run 1 vs Run 2 (should be byte-identical) ---")
    if tokens_1 == tokens_2:
        print(f"  [PASS] {len(tokens_1)} tokens byte-identical across two independent runs.")
        print("  Multi-block decode path is internally consistent —")
        print("  the earlier preemption-test mismatch is NOT caused by simply")
        print("  crossing a 2nd KV block; the bug must be specific to eviction/recompute.")
    else:
        # Find the first point of divergence for a precise bug report.
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(tokens_1, tokens_2)) if a != b),
            min(len(tokens_1), len(tokens_2)),
        )
        print(f"  [FAIL] Divergence at token index {first_diff}")
        print(f"    Prompt length: {len(prompt)}, block_size: {block_size}")
        print(f"    Divergence at absolute position: {len(prompt) + first_diff}"
              f" (relative to block boundary at {block_size}: "
              f"{len(prompt) + first_diff - block_size:+d})")
        print(f"    Run 1 around divergence: {tokens_1[max(0,first_diff-3):first_diff+5]}")
        print(f"    Run 2 around divergence: {tokens_2[max(0,first_diff-3):first_diff+5]}")
        print("\n  This confirms a real bug in ordinary multi-block decode,")
        print("  completely independent of preemption — likely in slot_mapping/")
        print("  block_tables construction once block_table has 2+ entries, or")
        print("  in how flash_attn_with_kvcache consumes a multi-block block_tables tensor.")
        raise AssertionError(f"Multi-block decode is non-deterministic: diverged at index {first_diff}")

    print("=" * 70)


if __name__ == "__main__":
    main()