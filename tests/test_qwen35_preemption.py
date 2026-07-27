"""Phase 3 Preemption Correctness Test for Qwen3.5 Hybrid Model.

Exercises the REAL Scheduler/ModelRunner/StateManager pipeline via LLMEngine
under forced memory pressure to validate that preemption:
  1. Correctly triggers Scheduler.preempt() when KV blocks are constrained.
  2. Frees state slots in StateManager without leaking slot allocations.
  3. Correctly re-prefills and rebuilds GDR linear attention state from scratch.
  4. Produces byte-identical completion tokens compared to an unconstrained run.

Note on kvcache_block_size: FlashAttention's paged-KV-cache path
(flash_attn_with_kvcache) hard-requires block size to be a multiple of 256
internally. We can't shrink the block granularity to force scarcity — instead
we keep block_size=256 (the real default) and make sequences long enough
(max_tokens=300) to actually cross a 256-token block boundary, then shrink the
*block count* (not size) to force multiple concurrent long sequences to
compete for a scarce pool of full-size blocks.

Usage:
    python tests/make_fake_hf_config.py
    python tests/test_qwen35_preemption.py
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

# --- Determinism requirement #2: Monkeypatch Sampler to plain greedy argmax ---
import nanovllm.layers.sampler as sampler_mod
sampler_mod.Sampler.forward = lambda self, logits, temperatures: logits.float().argmax(dim=-1)

# --- Monkeypatch model loading (fake config, no .safetensors) ---
import nanovllm.utils.loader as loader_mod
loader_mod.load_model = lambda model, path: None

import nanovllm.config as config_mod

# NOTE: the block-size-bypass patch from earlier attempts has been removed.
# Config's real __post_init__ (with its `kvcache_block_size % 256 == 0`
# assertion) is used as-is — flash-attn's paged-cache kernel requires this,
# so bypassing it just moves the failure from Config to flash_attn with a
# less clear error message.

class AttrDict(dict):
    """Minimal stand-in for HF's PretrainedConfig."""
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

# --- Tokenizer Monkeypatch ---
class DummyTokenizer:
    eos_token_id = 999999  # high EOS ID so max_tokens bounds generation

    def encode(self, prompt):
        if isinstance(prompt, list):
            return prompt
        return [int(x) for x in prompt.split() if x.isdigit()]

    def decode(self, token_ids):
        return " ".join(str(t) for t in token_ids)

import transformers
transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *args, **kwargs: DummyTokenizer())

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
from nanovllm.models.qwen3_5 import Experts


def init_deterministic_weights(engine: LLMEngine, seed: int = 42):
    """Initialize all model weights deterministically so uninitialized memory
    (torch.empty in the custom Column/Row/Replicated/MergedColumn linear
    layers and the embedding classes — none of which are plain nn.Linear or
    nn.Embedding, so isinstance-based init misses them entirely) doesn't
    introduce zeros/NaN/inf or unintended randomness into logits."""
    with torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = engine.model_runner.model
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(param)  # biases, A_log, dt_bias, 1D norm weights


def shrink_kv_cache(engine, num_blocks):
    """Force num_kvcache_blocks down to a small number post-construction,
    so preemption is triggered deterministically regardless of GPU size.
    Block SIZE (256, flash-attn's hard requirement) is left untouched —
    only the block COUNT is shrunk."""
    mr = engine.model_runner
    cfg = mr.config
    cfg.num_kvcache_blocks = num_blocks

    _, num_kv_layers, _, block_size, num_kv_heads, head_dim = mr.kv_cache.shape
    mr.kv_cache = torch.empty(
        2, num_kv_layers, num_blocks, block_size, num_kv_heads, head_dim,
        device="cuda", dtype=mr.kv_cache.dtype,
    )
    layer_id = 0
    for module in mr.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = mr.kv_cache[0, layer_id]
            module.v_cache = mr.kv_cache[1, layer_id]
            layer_id += 1

    from nanovllm.engine.block_manager import BlockManager
    engine.scheduler.block_manager = BlockManager(num_blocks, cfg.kvcache_block_size)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Preemption test requires CUDA (flash-attn, paged KV cache)"

    fake_dir = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
    assert os.path.isdir(fake_dir), "Run tests/make_fake_hf_config.py first"

    # Test prompt token lists (4 distinct prompts of varying lengths)
    prompts = [
        [10, 11, 12, 13, 14, 15, 16, 17],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        [40, 41, 42, 43],
        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
    ]
    # 300 > 256 (the fixed block_size) so a sequence's total length actually
    # crosses a block boundary and needs a 2nd block during decode — without
    # this, can_append() never needs a new block and preempt() never fires,
    # no matter how few blocks exist globally.
    max_tokens = 300
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=1.0)

    print("=" * 70)
    print("PHASE 3 PREEMPTION TEST: Qwen3.5 Hybrid Model")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # RUN A: Unconstrained (Generous KV cache, zero preemption)
    # -------------------------------------------------------------------------
    print("\n--- RUN A: Unconstrained Execution ---")
    engine_a = LLMEngine(
        fake_dir,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=512,   # room for prompt + up to 300 generated tokens
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    try:
        init_deterministic_weights(engine_a, seed=42)

        preempt_count_a = [0]
        orig_preempt_a = engine_a.scheduler.preempt
        def count_preempt_a(seq):
            preempt_count_a[0] += 1
            return orig_preempt_a(seq)
        engine_a.scheduler.preempt = count_preempt_a

        res_a = engine_a.generate(prompts, sampling_params, use_tqdm=False)
        tokens_a = [out["token_ids"] for out in res_a]

        print(f"  Run A Preemption Count: {preempt_count_a[0]}")
        assert preempt_count_a[0] == 0, f"Run A unexpected preemption: {preempt_count_a[0]}"
        print(f"  Run A Generated Tokens (sample seq 0): {tokens_a[0][:8]}...")
    finally:
        # Clean up distributed process group and CUDA resources before starting Run B
        engine_a.exit()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # -------------------------------------------------------------------------
    # RUN B: Preemption-Forced (tiny block COUNT, real block SIZE)
    # -------------------------------------------------------------------------
    print("\n--- RUN B: Preemption-Forced Execution ---")
    engine_b = LLMEngine(
        fake_dir,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=512,
        gpu_memory_utilization=0.5,  # doesn't matter — shrink_kv_cache overrides directly below
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    init_deterministic_weights(engine_b, seed=42)
    # 4 sequences need >=1 block each just to start (4 total), and will each
    # need a 2nd block once they cross token 256. 6 blocks is enough for all
    # 4 to start, but not enough for all 4 to hold 2 blocks simultaneously —
    # forcing real competition/eviction once sequences grow past the first block.
    shrink_kv_cache(engine_b, num_blocks=6)

    try:
        actual_blocks = engine_b.model_runner.config.num_kvcache_blocks
        print(f"  Run B Allocated KV Cache Blocks (via allocate_kv_cache): {actual_blocks}")

        preempt_count_b = [0]
        orig_preempt_b = engine_b.scheduler.preempt
        def count_preempt_b(seq):
            preempt_count_b[0] += 1
            return orig_preempt_b(seq)
        engine_b.scheduler.preempt = count_preempt_b

        res_b = engine_b.generate(prompts, sampling_params, use_tqdm=False)
        tokens_b = [out["token_ids"] for out in res_b]

        print(f"  Run B Preemption Count: {preempt_count_b[0]}")
        assert preempt_count_b[0] > 0, "Run B failed to trigger preemption — budget was not constrained enough!"
        print(f"  [OK] Confirmed preemption triggered {preempt_count_b[0]} time(s)")

        sm = engine_b.model_runner.state_manager
        if sm is not None:
            assert len(sm.free_slot_ids) == sm.max_num_seqs, (
                f"Run B leaked state slots! {sm.max_num_seqs - len(sm.free_slot_ids)} slots still in use."
            )
            print("  [OK] StateManager pool fully cleaned up after preemption run.")
    finally:
        engine_b.exit()

    # -------------------------------------------------------------------------
    # ASSERTION: Byte-identical outputs between Run A and Run B
    # -------------------------------------------------------------------------
    print("\n--- Verifying Exact Token Equivalence ---")
    mismatches = 0
    for idx, (seq_a, seq_b) in enumerate(zip(tokens_a, tokens_b)):
        if seq_a != seq_b:
            print(f"  [FAIL] Sequence {idx} mismatch!")
            print(f"    Run A: {seq_a}")
            print(f"    Run B: {seq_b}")
            mismatches += 1
        else:
            print(f"  [PASS] Sequence {idx}: exact match ({len(seq_a)} tokens)")

    assert mismatches == 0, f"Preemption test failed: {mismatches} sequence(s) mismatched!"

    print("=" * 70)
    print("ALL PREEMPTION TESTS PASSED — Preemption is 100% correct!")
    print("=" * 70)

if __name__ == "__main__":
    main()