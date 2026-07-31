# tests/run_small_model_smoke_test.py
"""Correctness gate: drives the REAL LLMEngine.generate() end-to-end on the
small hybrid config, using the checkpoint from make_fake_checkpoint.py and the
tokenizer from make_fake_tokenizer.py.

Reports four measurements separately -- deliberately not collapsed into a
single pass/fail:
  1. ModelRunner constructs without exception (layer dispatch, KV-cache
     allocation for full-attention layers, StateManager allocation for GDR
     layers, all at this size).
  2. load_model() actually loaded the checkpoint -- compares loaded params
     against a freshly constructed, never-loaded Qwen35ForCausalLM.
     load_model() silently no-ops if it finds zero .safetensors files, so
     absence-of-crash is not evidence it worked.
  3. generate() completes without exception -- eager mode, single prompt,
     single sequence.
  4. Output token ids are non-constant and finite (printed raw), and
     decoding to text doesn't crash.

Usage:
    python tests/make_fake_hf_config.py     # if not already run
    python tests/make_fake_checkpoint.py    # writes model.safetensors
    python tests/make_fake_tokenizer.py     # attaches gpt2 tokenizer
    python tests/run_small_model_smoke_test.py
"""
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


def main():
    assert torch.cuda.is_available(), "requires CUDA"
    assert os.path.isdir(FAKE_DIR), "run tests/make_fake_hf_config.py first"
    assert os.path.isfile(os.path.join(FAKE_DIR, "model.safetensors")), \
        "run tests/make_fake_checkpoint.py first"
    assert os.path.isfile(os.path.join(FAKE_DIR, "tokenizer_config.json")), \
        "run tests/make_fake_tokenizer.py first"

    config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)

    results = {}
    engine = None

    # ---- Measurement 1: real ModelRunner construction (via LLMEngine) ----
    print("=" * 70)
    print("Building LLMEngine (real ModelRunner, real load_model)")
    print("=" * 70)
    try:
        from nanovllm.engine.llm_engine import LLMEngine
        from nanovllm.sampling_params import SamplingParams

        engine = LLMEngine(
            FAKE_DIR,
            max_num_batched_tokens=512,
            max_num_seqs=4,
            max_model_len=512,
            gpu_memory_utilization=0.2,
            tensor_parallel_size=1,
            enforce_eager=True,
        )
        results["1_construction"] = "PASS"
        print("[1] ModelRunner construction: PASS")
    except Exception as e:
        results["1_construction"] = f"FAIL: {e!r}"
        print(f"[1] ModelRunner construction: FAIL -- {e!r}")
        print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
        for k, v in results.items():
            print(f"  {k}: {v}")
        raise

    # ---- Measurement 2: did load_model() actually load real weights? ----
    try:
        from nanovllm.models.qwen3_5 import Qwen35ForCausalLM

        fresh_config = _fake_from_pretrained(FAKE_DIR)
        prior_dtype = torch.get_default_dtype()
        torch.set_default_dtype(fresh_config.dtype)
        fresh_model = Qwen35ForCausalLM(fresh_config)  # never touched by load_model
        torch.set_default_dtype(prior_dtype)

        loaded_params = dict(engine.model_runner.model.named_parameters())
        fresh_params = dict(fresh_model.named_parameters())
        sample_names = [n for n in loaded_params if loaded_params[n].numel() > 0][:2]

        all_differ = True
        for n in sample_names:
            loaded = loaded_params[n].detach().float().cpu()
            fresh = fresh_params[n].detach().float().cpu()
            same = torch.equal(loaded, fresh)
            all_differ &= (not same)
            print(f"    param {n}: shape={tuple(loaded.shape)} identical-to-never-loaded={same}")

        results["2_weights_loaded"] = "PASS" if all_differ else "FAIL (matches uninitialized/fresh)"
        print(f"[2] load_model() actually loaded weights: {'PASS' if all_differ else 'FAIL'}")
    except Exception as e:
        results["2_weights_loaded"] = f"FAIL: {e!r}"
        print(f"[2] load_model() weight-loaded check: FAIL -- {e!r}")

    # ---- Measurement 3 + 4: generate() ----
    try:
        prompt = "The quick brown fox"
        sp = SamplingParams(max_tokens=20, temperature=1.0, ignore_eos=True)
        out = engine.generate([prompt], sp, use_tqdm=False)
        results["3_generate_completes"] = "PASS"
        print("[3] generate() completed without exception: PASS")

        token_ids = out[0]["token_ids"]
        text = out[0]["text"]
        print(f"    raw token_ids: {token_ids}")

        finite = all(isinstance(t, int) for t in token_ids)
        non_constant = len(set(token_ids)) > 1
        print(f"    all ids finite ints: {finite}, non-constant: {non_constant}")
        print(f"    decoded text: {text!r}")

        ok4 = finite and non_constant
        results["4_output_valid"] = "PASS" if ok4 else f"FAIL (finite={finite}, non_constant={non_constant})"
        print(f"[4] Output non-constant/finite + decode OK: {'PASS' if ok4 else 'FAIL'}")
    except Exception as e:
        results["3_generate_completes"] = f"FAIL: {e!r}"
        results["4_output_valid"] = "SKIPPED (generate failed)"
        print(f"[3] generate() completed without exception: FAIL -- {e!r}")
    finally:
        if engine is not None:
            engine.exit()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
