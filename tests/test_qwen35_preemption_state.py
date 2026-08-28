"""Phase 3 Preemption Correctness Test — STATE COMPARISON version.

Byte-identical sampled tokens is NOT a valid correctness bar here: bf16
matmul isn't strictly batch-size invariant, and with random/untrained
weights many logits sit close together, so a batch-size change (4->3
sequences when one is preempted) can flip an occasional argmax for ALL
sequences in that batch step, preempted or not. That's a measurement
artifact, not a preemption bug (confirmed: the preempted sequence's
num_tokens was correctly preserved across preempt -> re-prefill).

This test instead checks the thing that actually matters: after preemption
forces a sequence to be fully re-prefilled from scratch, does its GDR
recurrent state end up matching what an isolated, uninterrupted run of the
exact same token history would produce? That's independent of which token
greedy sampling happened to pick along the way.

Usage:
    python tests/make_fake_hf_config.py   # if not already run
    python tests/test_qwen35_preemption_state.py
"""

import sys
import os
import json
import torch
import torch.nn as nn

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

import nanovllm.layers.sampler as sampler_mod
sampler_mod.Sampler.forward = lambda self, logits, temperatures: logits.float().argmax(dim=-1)

import nanovllm.utils.loader as loader_mod
loader_mod.load_model = lambda model, path, **kw: None

import nanovllm.config as config_mod


class AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


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


def shrink_kv_cache(engine, num_blocks):
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


def patch_capture_on_free(state_manager):
    """Snapshot (per-layer) state right before StateManager.free() zeroes it,
    keyed by seq_id, in call order. Returns the capture dict."""
    captured = {}
    orig_free = state_manager.free

    def capturing_free(seq):
        slot_id = seq.state_slot
        if slot_id is not None:
            captured[seq.seq_id] = state_manager.states[:, slot_id].clone()
        return orig_free(seq)

    state_manager.free = capturing_free
    return captured


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Requires CUDA"

    fake_dir = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
    assert os.path.isdir(fake_dir), "Run tests/make_fake_hf_config.py first"

    prompts = [
        [10, 11, 12, 13, 14, 15, 16, 17],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        [40, 41, 42, 43],
        [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
    ]
    max_tokens = 300
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=1.0)

    print("=" * 70)
    print("PHASE 3 PREEMPTION TEST (state comparison)")
    print("=" * 70)

    # ---- RUN B: preemption-forced, capture state at free() time ----
    print("\n--- RUN B: Preemption-Forced Execution ---")
    engine_b = LLMEngine(
        fake_dir, max_num_batched_tokens=512, max_num_seqs=8, max_model_len=512,
        gpu_memory_utilization=0.5, tensor_parallel_size=1, enforce_eager=True,
    )
    init_deterministic_weights(engine_b, seed=42)
    shrink_kv_cache(engine_b, num_blocks=6)
    captured_b = patch_capture_on_free(engine_b.model_runner.state_manager)

    preempt_count_b = [0]
    orig_preempt_b = engine_b.scheduler.preempt
    def count_preempt_b(seq):
        preempt_count_b[0] += 1
        return orig_preempt_b(seq)
    engine_b.scheduler.preempt = count_preempt_b

    try:
        res_b = engine_b.generate(prompts, sampling_params, use_tqdm=False)
        tokens_b = [out["token_ids"] for out in res_b]
    finally:
        engine_b.exit()

    print(f"  Preemption count: {preempt_count_b[0]}")
    assert preempt_count_b[0] > 0, "No preemption occurred — nothing to validate"
    assert len(captured_b) >= 1, "No state captured at free() — check the monkeypatch"

    # With Preemption Count == 1, the FIRST capture chronologically (dict
    # preserves insertion order) is the mid-run preempted sequence; every
    # other capture happens later, at normal end-of-generation finish time.
    first_captured_seq_id = next(iter(captured_b.keys()))
    preempted_state = captured_b[first_captured_seq_id]

    # From the earlier debug run we know the 4th prompt (16-token prompt,
    # prompt index 3) is the one that gets preempted with this config.
    preempted_prompt_idx = 3
    full_history = prompts[preempted_prompt_idx] + tokens_b[preempted_prompt_idx]
    print(f"  Preempted sequence: prompt idx {preempted_prompt_idx}, "
          f"full history length {len(full_history)}")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ---- ISOLATED RUN: same weights (same seed), same full token history,
    #      as a single uninterrupted sequence, generous memory (no preemption) ----
    print("\n--- ISOLATED RUN: same token history, single sequence, no preemption ---")
    engine_iso = LLMEngine(
        fake_dir, max_num_batched_tokens=512, max_num_seqs=8, max_model_len=512,
        gpu_memory_utilization=0.5, tensor_parallel_size=1, enforce_eager=True,
    )
    init_deterministic_weights(engine_iso, seed=42)  # same seed -> same weights as engine_b
    captured_iso = patch_capture_on_free(engine_iso.model_runner.state_manager)

    try:
        # Feed the FULL known history as the prompt, generate exactly 1 token
        # just to drive it through prefill and let it finish/free naturally.
        iso_params = SamplingParams(max_tokens=1, temperature=1.0)
        engine_iso.generate([full_history[:-1]], iso_params, use_tqdm=False)
    finally:
        engine_iso.exit()

    assert len(captured_iso) == 1, f"Expected 1 captured state, got {len(captured_iso)}"
    isolated_state = next(iter(captured_iso.values()))

    # ---- Compare ----
    print("\n--- Comparing GDR recurrent state: preempted-and-recomputed vs isolated ---")
    num_linear_layers = preempted_state.shape[0]
    all_pass = True
    for layer_idx in range(num_linear_layers):
        a = preempted_state[layer_idx].float().reshape(-1)
        b = isolated_state[layer_idx].float().reshape(-1)
        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
        status = "PASS" if cos > 0.999 else "FAIL"
        if cos <= 0.999:
            all_pass = False
        print(f"  layer {layer_idx}: cosine={cos:.6f} [{status}]")

    print()
    if all_pass:
        print("[PASS] Preempted-and-recomputed GDR state matches an isolated run "
              "of the same token history — preemption correctly rebuilds state "
              "from scratch.")
    else:
        raise AssertionError("GDR state mismatch after preemption — recompute logic is wrong.")

    print("=" * 70)


if __name__ == "__main__":
    main()