"""Verify utils/loader.py's prefix-translation step against the real
Qwen3.5-MoE checkpoint's key names (qwen35_checkpoint/model.safetensors.index.json).

No real tensor data is available locally (index.json only lists keys), so this
does not run load_model() end-to-end. Instead it:
  1. Runs every real checkpoint key through translate_weight_name() (+ the
     unmodified packed_modules_mapping resolution loop copied verbatim from
     load_model(), only touching the prefix-stripping step as instructed).
  2. Constructs a real Qwen35ForCausalLM(config.hf_config) on the meta device
     (real 40-layer / 256-expert shape, no actual memory for weight data) and
     compares resolved names against model.named_parameters().keys().

Usage:
    python tests/test_loader_prefix_translation.py
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist  # noqa: E402  (sets up nanovllm namespace + dist)

INDEX_PATH = os.path.join(ROOT, "qwen35_checkpoint", "model.safetensors.index.json")


def resolve(param_name: str, packed_modules_mapping: dict) -> str:
    """Mirrors load_model()'s packed_modules_mapping resolution loop exactly."""
    for k in packed_modules_mapping:
        if k in param_name:
            v, _shard_id = packed_modules_mapping[k]
            return param_name.replace(k, v)
    return param_name


def _stub_attention_module():
    """layers/attention.py imports triton, which isn't installed on this CPU
    dev box. Attention holds no nn.Parameter/nn.Linear of its own (pure
    flash-attn/triton compute wrapper -- verified via grep), so for a
    parameter-name/shape check it's safe to stub it out rather than require
    a GPU+triton install just to enumerate module names."""
    import types
    import torch.nn as nn

    stub = types.ModuleType("nanovllm.layers.attention")

    class Attention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise NotImplementedError("stub Attention -- structure-only test, no forward")

    stub.Attention = Attention
    sys.modules["nanovllm.layers.attention"] = stub


def main():
    init_dist()
    _stub_attention_module()

    from nanovllm.config import Config
    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM
    from nanovllm.utils.loader import translate_weight_name

    config = Config("qwen35_checkpoint")

    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    all_keys = list(weight_map.keys())
    print(f"checkpoint has {len(all_keys)} total keys")

    # No assert_all_parameters_initialized here (see tests/test_utils.py):
    # torch.device("meta") allocates no real storage, only names/shapes --
    # there is nothing for that guardrail to check, and calling it would
    # just error on meta tensors rather than catch a real bug. This test
    # only ever reads model.named_parameters() for NAMES, never numeric
    # values.
    with torch.device("meta"):
        model = Qwen35ForCausalLM(config.hf_config)
    param_names = set(name for name, _ in model.named_parameters())
    print(f"Qwen35ForCausalLM has {len(param_names)} parameters")

    packed_modules_mapping = model.packed_modules_mapping

    text_model_keys = []
    skipped_keys = []
    for k in all_keys:
        t = translate_weight_name(k)
        if t is None:
            skipped_keys.append(k)
        else:
            text_model_keys.append((k, t))

    print(f"\ntranslate_weight_name: {len(text_model_keys)} text-model keys, "
          f"{len(skipped_keys)} skipped (non text-model)")
    print("skipped-key prefixes sample:", sorted(set(k.split('.')[0] + '.' + k.split('.')[1]
          if '.' in k else k for k in skipped_keys))[:5])

    unmatched_checkpoint_keys = []
    matched_param_names = set()
    for raw_key, translated in text_model_keys:
        resolved = resolve(translated, packed_modules_mapping)
        if resolved in param_names:
            matched_param_names.add(resolved)
        else:
            unmatched_checkpoint_keys.append((raw_key, translated, resolved))

    unmatched_model_params = param_names - matched_param_names

    print(f"\nmatched: {len(matched_param_names)} / {len(param_names)} model parameters")
    print(f"unmatched checkpoint keys (text-model, no corresponding param): {len(unmatched_checkpoint_keys)}")
    for raw_key, translated, resolved in unmatched_checkpoint_keys[:20]:
        print(f"  {raw_key!r} -> {translated!r} -> {resolved!r} (NOT FOUND)")

    print(f"\nunmatched model parameters (no checkpoint key ever resolves to them): {len(unmatched_model_params)}")
    for p in sorted(unmatched_model_params)[:20]:
        print(f"  {p}")

    ok = not unmatched_checkpoint_keys and not unmatched_model_params
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
