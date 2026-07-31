# tests/test_state_slot_reuse.py
"""Issue 1: StateManager slot-reuse contamination check, end-to-end through
the real engine (not a hand-built StateManager like
tests/test_qwen35_batching.py's test_slot_reuse_no_leakage -- this exercises
the actual Scheduler/ModelRunner/StateManager wiring with the real small
checkpoint).

engine/state_manager.py's allocate()/free() both call .zero_() on the
slot's (state, conv_state) slices, so on paper a reused slot should never
carry over a previous tenant's recurrent/conv state. This test verifies that
empirically end-to-end rather than trusting the zero_() calls are actually
on every code path that matters, by forcing an actual reuse of slot 0 across
three distinct sequences and comparing generation output against a fully
isolated run of the same prompt.

Sampling is monkeypatched to plain argmax (greedy) -- SamplingParams forbids
temperature<=1e-10 ("greedy sampling is not permitted"), so determinism has
to come from replacing Sampler.forward, not from the temperature knob.

max_num_seqs is deliberately small (2) so slot cycling back to slot 0 is
forced after exactly one intermediate sequence, not left to chance:
    free_slot_ids starts as deque([0, 1])
    A.allocate() -> slot 0;  A.free() -> free=[1, 0]
    B.allocate() -> slot 1;  B.free() -> free=[0, 1]
    C.allocate() -> slot 0   <- same slot A used, this is the reuse under test

Usage:
    python tests/make_fake_hf_config.py     # if not already run
    python tests/make_fake_checkpoint.py    # if not already run
    python tests/make_fake_tokenizer.py     # if not already run
    python tests/test_state_slot_reuse.py
"""
import atexit
import gc
import json
import os
import sys

import torch

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

import nanovllm.config as config_mod

FAKE_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _fake_from_pretrained(path, *args, **kwargs):
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d


def make_prompt(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(100, 5000, (n,), generator=g).tolist()


def build_engine():
    from nanovllm.engine.llm_engine import LLMEngine
    return LLMEngine(
        FAKE_DIR,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        max_model_len=512,
        gpu_memory_utilization=0.2,
        tensor_parallel_size=1,
        enforce_eager=True,
    )


def main():
    assert torch.cuda.is_available(), "requires CUDA"
    assert os.path.isdir(FAKE_DIR), "run tests/make_fake_hf_config.py first"
    assert os.path.isfile(os.path.join(FAKE_DIR, "model.safetensors")), \
        "run tests/make_fake_checkpoint.py first"
    assert os.path.isfile(os.path.join(FAKE_DIR, "tokenizer_config.json")), \
        "run tests/make_fake_tokenizer.py first"

    config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)

    from nanovllm.sampling_params import SamplingParams

    prompt_a = make_prompt(10, seed=1)
    prompt_b = make_prompt(10, seed=2)
    prompt_c = make_prompt(10, seed=3)  # the "new, different prompt" that lands on slot 0 again
    sp = SamplingParams(max_tokens=20, temperature=1.0, ignore_eos=True)

    print("=" * 70)
    print("Building engine (max_num_seqs=2) -- run A, B, C sequentially")
    print("=" * 70)
    engine = build_engine()
    try:
        out_a = engine.generate([prompt_a], sp, use_tqdm=False)[0]
        slot_a = "freed"  # can't observe seq.state_slot after free(), but allocation order is deterministic
        print(f"  A: prompt={prompt_a}")
        print(f"     completion={out_a['token_ids']}")

        out_b = engine.generate([prompt_b], sp, use_tqdm=False)[0]
        print(f"  B: prompt={prompt_b}")
        print(f"     completion={out_b['token_ids']}")

        out_c_together = engine.generate([prompt_c], sp, use_tqdm=False)[0]
        print(f"  C (after A, B -- reuses slot 0): prompt={prompt_c}")
        print(f"     completion={out_c_together['token_ids']}")
    finally:
        # LLMEngine.__init__ does atexit.register(self.exit) -- that keeps a
        # permanent strong reference to this instance (model weights,
        # kv_cache, StateManager buffers) alive for the life of the process
        # no matter what, so plain `del engine` would NOT free GPU memory
        # before the second engine below tries to allocate its own KV cache.
        atexit.unregister(engine.exit)
        engine.exit()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Building FRESH engine -- run C in total isolation")
    print("=" * 70)
    engine_iso = build_engine()
    try:
        out_c_isolated = engine_iso.generate([prompt_c], sp, use_tqdm=False)[0]
        print(f"  C (isolated): prompt={prompt_c}")
        print(f"     completion={out_c_isolated['token_ids']}")
    finally:
        engine_iso.exit()

    print("\n" + "=" * 70)
    print("Comparing C (after slot-0 reuse) vs C (isolated)")
    print("=" * 70)
    together = out_c_together["token_ids"]
    isolated = out_c_isolated["token_ids"]
    print(f"  together: {together}")
    print(f"  isolated: {isolated}")

    if together == isolated:
        print("\n[PASS] slot-0 reuse produced identical output to an isolated run -- "
              "no detectable state contamination.")
    else:
        first_diff = next((i for i, (x, y) in enumerate(zip(together, isolated)) if x != y), None)
        print("\n[FAIL] slot-0 reuse output DIFFERS from an isolated run of the same prompt.")
        print(f"       first differing position: {first_diff}")
        print(f"       together[{first_diff}]={together[first_diff] if first_diff is not None else 'N/A'} "
              f"isolated[{first_diff}]={isolated[first_diff] if first_diff is not None else 'N/A'}")
        print("       This indicates StateManager slot-reuse contamination. Do NOT "
              "patch StateManager's reset logic based on this script alone -- report "
              "it and investigate the actual code path first.")
        raise AssertionError("slot-0 reuse output mismatch -- see printed diff above")


if __name__ == "__main__":
    main()
