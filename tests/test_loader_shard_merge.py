# tests/test_loader_shard_merge.py
"""Issue 3: verify utils/loader.py's packed_modules_mapping + shard-aware
weight_loader path actually merges separately-named HF-style shard tensors
correctly, using real on-disk safetensors I/O -- not just that it runs.

Context: tests/make_fake_checkpoint.py already splits Qwen35ForCausalLM's
fused shared_expert.gate_up_proj back into HF-style gate_proj/up_proj shards
before saving (see that script's docstring) specifically so load_model()'s
packed_modules_mapping branch fires. So tests/fake_qwen35_small/model.safetensors
is ALREADY the "sharded, HF-style-named" checkpoint -- there's no separate
"original non-sharded" checkpoint on disk to compare it against.

This script builds the inverse: a fused-name checkpoint
(tests/fake_qwen35_small_fused/), by concatenating each pair of on-disk shard
tensors back together along the dimension packed_modules_mapping implies.
Loading the EXISTING sharded checkpoint exercises the real code path
(packed_modules_mapping lookup + MergedColumnParallelLinear.weight_loader's
shard_offset/shard_size slicing). Loading the new fused checkpoint exercises
the plain direct-name path (model.get_parameter + default_weight_loader, no
packed mapping). If packed_modules_mapping's logic is correct, the two
resulting in-memory models must be parameter-for-parameter IDENTICAL --
exact torch.equal, not cosine similarity, since this is pure concatenation/
slicing with no numerical computation anywhere in either path.

Uses direct Qwen35ForCausalLM(config) + utils.loader.load_model() rather than
the full LLMEngine/ModelRunner -- packed_modules_mapping and weight_loader
live entirely at the nn.Module level and have no interaction with KV-cache
sizing, the scheduler, or StateManager, so going through the full engine
would add unrelated failure surface (we've repeatedly hit KV-cache budget
assertions for reasons having nothing to do with checkpoint loading) without
covering anything more of the actual question.

Usage:
    python tests/make_fake_hf_config.py     # if not already run
    python tests/make_fake_checkpoint.py    # if not already run
    python tests/test_loader_shard_merge.py
"""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config
from test_utils import known_zero_initialized_param_names, assert_all_parameters_initialized
from make_fake_checkpoint import _to_checkpoint_name

FAKE_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
FUSED_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small_fused")
SHARDED_PATH = os.path.join(FAKE_DIR, "model.safetensors")
FUSED_PATH = os.path.join(FUSED_DIR, "model.safetensors")


def build_fused_tensors(model, sharded_tensors: dict) -> dict:
    """Inverse of make_fake_checkpoint.py's build_save_dict(): reconstruct
    each of the model's own (fused-name) parameters from the on-disk,
    HF-style split shard tensors, by concatenating shards in packed_modules_mapping's
    shard_id order along dim 0 -- the same dim MergedColumnParallelLinear's
    weight_loader slices along.

    sharded_tensors is keyed by the real-checkpoint-style names
    (make_fake_checkpoint.py's _to_checkpoint_name, "model.language_model."
    prefix) that load_model()'s translate_weight_name expects on disk -- not
    by the model's own internal parameter names. Every lookup below must
    translate through _to_checkpoint_name first, same as build_save_dict()
    did when it wrote the file."""
    packed = model.packed_modules_mapping  # split_key -> (fused_suffix, shard_id)
    fused_suffixes = {}
    for split_key, (fused_suffix, shard_id) in packed.items():
        fused_suffixes.setdefault(fused_suffix, {})[shard_id] = split_key

    fused_tensors = {}
    for name, _ in model.named_parameters():
        matched_fused = next((s for s in fused_suffixes if s in name), None)
        if matched_fused is not None:
            shard_map = fused_suffixes[matched_fused]
            pieces = []
            for shard_id in sorted(shard_map.keys()):
                split_name = name.replace(matched_fused, shard_map[shard_id])
                checkpoint_name = _to_checkpoint_name(split_name)
                assert checkpoint_name in sharded_tensors, (
                    f"expected shard tensor {checkpoint_name!r} (for fused param {name!r}) "
                    f"not found in {SHARDED_PATH}"
                )
                pieces.append(sharded_tensors[checkpoint_name])
            fused_tensors[name] = torch.cat(pieces, dim=0)
        else:
            checkpoint_name = _to_checkpoint_name(name)
            assert checkpoint_name in sharded_tensors, (
                f"expected tensor {checkpoint_name!r} (for param {name!r}) not found in {SHARDED_PATH}"
            )
            fused_tensors[name] = sharded_tensors[checkpoint_name]
    return fused_tensors


def main():
    assert torch.cuda.is_available(), "requires CUDA"
    assert os.path.isfile(SHARDED_PATH), "run tests/make_fake_checkpoint.py first"

    init_dist()
    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM

    config = make_small_config()

    print("=" * 70)
    print("Building fused-name checkpoint from the existing sharded one")
    print("=" * 70)
    os.makedirs(FUSED_DIR, exist_ok=True)

    torch.manual_seed(2024)
    # No guardrail here (additive-only judgment call, not rule 4's literal
    # meta-device case but the same spirit): this instance is real (not
    # meta) but is used ONLY for its parameter NAMES (packed_modules_mapping
    # lookup below) and is deleted immediately after, never forward-passed
    # or otherwise relied on for numeric correctness -- there's nothing here
    # a torch.empty()-garbage check would meaningfully protect.
    ref_model_for_names = Qwen35ForCausalLM(config).to("cuda").to(torch.bfloat16)

    sharded_tensors = load_file(SHARDED_PATH, device="cuda")
    fused_tensors = build_fused_tensors(ref_model_for_names, sharded_tensors)
    save_file(
        {k: v.detach().clone().contiguous().cpu() for k, v in fused_tensors.items()},
        FUSED_PATH,
    )
    print(f"  wrote {len(fused_tensors)} fused-name tensors to {FUSED_PATH}")
    del ref_model_for_names

    import shutil
    for fname in ("config.json", "tokenizer_config.json", "tokenizer.json",
                  "vocab.json", "merges.txt", "special_tokens_map.json"):
        src = os.path.join(FAKE_DIR, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(FUSED_DIR, fname))

    print("\n" + "=" * 70)
    print("Loading SHARDED checkpoint (exercises packed_modules_mapping)")
    print("=" * 70)
    from nanovllm.utils.loader import load_model

    model_sharded = Qwen35ForCausalLM(config).to("cuda").to(torch.bfloat16)
    load_model(model_sharded, FAKE_DIR)
    print("  load_model() completed without error")

    # Guardrail (additive only -- see tests/test_utils.py): confirms
    # load_model() actually populated every parameter from the real fake
    # checkpoint, not just enough to avoid crashing.
    assert_all_parameters_initialized(
        model_sharded, whitelist_zero=known_zero_initialized_param_names(model_sharded)
    )

    print("\n" + "=" * 70)
    print("Loading FUSED checkpoint (control -- direct copy_, bypassing load_model())")
    print("=" * 70)
    # NOT routed through load_model(): MergedColumnParallelLinear.weight_loader
    # requires a loaded_shard_id argument that load_model()'s non-packed
    # ("else") branch never supplies, so it has no code path that can accept
    # a checkpoint using the model's own fused parameter name -- confirmed
    # below, this is exactly what crashed on the first attempt. That's a real,
    # separate gap in the loader (out of scope here), not something to route
    # around silently. This control only needs a numerically-correct
    # reference to compare against, so it copies tensors in directly.
    model_fused = Qwen35ForCausalLM(config).to("cuda").to(torch.bfloat16)
    fused_on_disk = load_file(FUSED_PATH, device="cuda")
    with torch.no_grad():
        for name, param in model_fused.named_parameters():
            param.data.copy_(fused_on_disk[name])
    print(f"  copied {len(fused_on_disk)} tensors directly (weight_loader bypassed by design)")

    # Guardrail (additive only -- see tests/test_utils.py): the manual copy
    # loop above targets every named_parameter by name, so this should
    # trivially pass -- confirms the "bypass" path didn't silently miss
    # anything either.
    assert_all_parameters_initialized(
        model_fused, whitelist_zero=known_zero_initialized_param_names(model_fused)
    )

    # No guardrail on the model constructed just below: it's passed straight
    # into a load_model() call that's EXPECTED to raise TypeError (a real,
    # documented loader gap with fused-name checkpoints, out of scope here)
    # before any parameter would be populated -- wiring a check here would
    # either never run (exception fires first) or, if moved before the call,
    # trivially fail on an untouched model and add nothing to what's already
    # being tested. See the try/except immediately below.
    try:
        load_model(Qwen35ForCausalLM(config).to("cuda").to(torch.bfloat16), FUSED_DIR)
        print("  [UNEXPECTED] load_model() accepted the fused-name checkpoint without error")
    except TypeError as e:
        print(f"  [CONFIRMED] load_model() cannot accept fused-name checkpoints -- {e}")
        print("  (real loader gap, out of scope for this test -- not fixed here)")

    print("\n" + "=" * 70)
    print("Comparing parameter-by-parameter: sharded-loaded vs fused-loaded")
    print("=" * 70)
    sharded_params = dict(model_sharded.named_parameters())
    fused_params = dict(model_fused.named_parameters())
    assert sharded_params.keys() == fused_params.keys(), (
        f"parameter name sets differ: "
        f"only-in-sharded={sharded_params.keys() - fused_params.keys()} "
        f"only-in-fused={fused_params.keys() - sharded_params.keys()}"
    )

    mismatches = []
    for name in sharded_params:
        a = sharded_params[name].detach()
        b = fused_params[name].detach()
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs()
            mismatches.append((name, tuple(a.shape), diff.max().item(), (diff != 0).sum().item(), diff.numel()))

    if not mismatches:
        print(f"  [PASS] all {len(sharded_params)} parameters bitwise identical "
              f"between sharded-load and fused-load")
    else:
        print(f"  [FAIL] {len(mismatches)} / {len(sharded_params)} parameters differ:")
        for name, shape, max_diff, n_mismatched, numel in mismatches:
            print(f"    {name} shape={shape}: max_abs_diff={max_diff:.6g} "
                  f"mismatched_elements={n_mismatched}/{numel}")
        raise AssertionError(f"{len(mismatches)} parameter(s) mismatched -- see detail above")


if __name__ == "__main__":
    main()
