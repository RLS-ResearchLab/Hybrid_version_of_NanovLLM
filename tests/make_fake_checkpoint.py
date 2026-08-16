# tests/make_fake_checkpoint.py
"""Builds tests/fake_qwen35_small/model.safetensors so ModelRunner has real,
finite weights to load instead of uninitialized nn.Parameter(torch.empty(...))
memory (LinearBase, Experts, VocabParallelEmbedding/ParallelLMHead all skip
default init).

Reuses test_qwen35_full_model.py's copy_weights_to_port(port_model, ref_model)
-- validated against the reference implementation, see that test's own
printed cosine for the current number (the 0.999967 figure once quoted here
was measured before Experts.gate_up_proj/down_proj were initialized -- see
init_reference_moe_experts_ below -- and is stale; do not re-quote it
without re-running the test) -- to populate a Qwen35ForCausalLM from the
reference Qwen35MoESmall (plain nn.Linear, so it initializes correctly with
finite values).

The saved tensor names must match what nanovllm.utils.loader.load_model()
expects: every port parameter name, PREFIX-TRANSLATED to the real
checkpoint's "model.language_model." convention (see
_to_checkpoint_name's docstring below -- load_model()'s
translate_weight_name silently drops any key that isn't under that prefix
or "lm_head.", a real bug this script used to trip over silently, found
and fixed during pre-cluster-day dry-run prep), EXCEPT the shared-expert
gate/up projection, which the port model fuses into a single gate_up_proj
but load_model's packed_modules_mapping only fires on the split HF-style
names ("shared_expert.gate_proj" / "shared_expert.up_proj"). Saving the
fused name directly would skip the packed-mapping branch and call
MergedColumnParallelLinear.weight_loader with a missing loaded_shard_id
argument. So gate_up_proj is split back into two chunks before saving.

Usage:
    python tests/make_fake_hf_config.py   # if not already run
    python tests/make_fake_checkpoint.py
"""
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, load_reference_module, make_small_config
from test_qwen35_full_model import copy_weights_to_port
from test_utils import (
    known_zero_initialized_param_names,
    assert_all_parameters_initialized,
    init_reference_moe_experts_,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")


def _to_checkpoint_name(internal_name: str) -> str:
    """Inverse of nanovllm.utils.loader.translate_weight_name: internal
    parameter names (e.g. "model.embed_tokens.weight") -> the real-
    checkpoint-style key load_model() actually recognizes
    ("model.language_model.embed_tokens.weight"). "lm_head.*" is already
    the real checkpoint's own convention (see translate_weight_name),
    unchanged.

    BUG THIS WORKS AROUND, found during pre-cluster-day dry-run prep
    (cluster_a2_tp_correctness.py's --phase compare failing with
    non-finite prefill logits, traced with tests/diag_a2_prefill_nan_trace.py):
    this function used to save keys AS-IS (bare "model." prefix, matching
    named_parameters()'s own internal naming) -- but load_model()'s
    translate_weight_name only recognizes "model.language_model." (real
    HF-VLM-style checkpoints) or "lm_head." keys, silently DROPPING every
    other key (`if param_name is None: continue`, utils/loader.py). The
    saved checkpoint itself was never the problem (this script's own
    finiteness check already passed on every saved tensor) -- load_model()
    simply never loaded ANY of it back in when the engine was later
    constructed, leaving every parameter at whatever the fresh CUDA
    allocation happened to contain. Confirmed directly, not guessed:
    embed_tokens.weight read back as bitwise all-zero (0/127139840
    nonzero) after a full engine construction against the OLD, bare-
    "model."-prefixed checkpoint this function used to produce.

    Isolated to this local fixture path -- the REAL checkpoint's own keys
    already use "model.language_model." (confirmed by reading
    qwen35_checkpoint/model.safetensors.index.json directly), so this
    never affected, and does not change, real-checkpoint loading.
    """
    if internal_name.startswith("lm_head."):
        return internal_name
    assert internal_name.startswith("model."), f"unexpected top-level param name: {internal_name!r}"
    return "model.language_model." + internal_name[len("model."):]


def build_save_dict(model):
    """Return {name: cpu_tensor} for safetensors, splitting any fused
    packed_modules_mapping targets back into their HF-style split names,
    and renaming every key to the real-checkpoint-style prefix load_model()
    actually recognizes (see _to_checkpoint_name)."""
    packed = model.packed_modules_mapping  # split_key -> (fused_suffix, shard_id)
    fused_suffixes = {}
    for split_key, (fused_suffix, shard_id) in packed.items():
        fused_suffixes.setdefault(fused_suffix, {})[shard_id] = split_key

    save_dict = {}
    for name, param in model.named_parameters():
        matched_fused = next((s for s in fused_suffixes if s in name), None)
        if matched_fused is not None:
            shard_map = fused_suffixes[matched_fused]
            chunks = param.data.chunk(len(shard_map), dim=0)
            for shard_id, split_key in shard_map.items():
                split_name = name.replace(matched_fused, split_key)
                save_dict[_to_checkpoint_name(split_name)] = chunks[shard_id].detach().clone().contiguous().cpu()
        else:
            save_dict[_to_checkpoint_name(name)] = param.detach().clone().contiguous().cpu()
    return save_dict


def main():
    assert torch.cuda.is_available(), "requires CUDA (Qwen35FullAttention needs flash_attn)"
    assert os.path.isdir(OUT_DIR), "run tests/make_fake_hf_config.py first"

    init_dist()
    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM

    config = make_small_config()

    torch.manual_seed(2024)
    ref_mod = load_reference_module()
    ref_model = ref_mod.Qwen35MoESmall().to("cuda").to(torch.bfloat16)

    # FIXED (was: CONFIRMED FAILURE below) -- ref_model's Experts class
    # (src/model_small_qwen3.5.py) never initializes gate_up_proj/down_proj,
    # they read back all-zero straight out of construction, and
    # copy_weights_to_port would otherwise faithfully carry that zero into
    # port_model -- which is the direct source of
    # tests/fake_qwen35_small/model.safetensors, so that fixture file had
    # all-zero expert weights baked into it on disk. src/model_small_qwen3.5.py
    # is the reference and out of scope to modify, so this initializes the
    # constructed ref_model instance instead -- see
    # init_reference_moe_experts_'s docstring in test_utils.py (shared with
    # test_qwen35_full_model.py and test_qwen35_standalone.py, which hit the
    # identical issue) for why this is the targeted `.normal_(0, 0.02)` fix
    # test_qwen35_vectorized_moe.py already validated locally, not the
    # blanket init_model_weights_with_norms_ (which would incorrectly zero
    # this reference module's RMSNormGated.weight instead, since its
    # isinstance-based norm override only matches nanovllm's own layernorm
    # classes, not the reference's).
    #
    # 17 files under tests/ reference "fake_qwen35_small" (grep-confirmed):
    # this file and test_qwen35_full_model.py (the ones that construct this
    # checkpoint), plus test_loader_shard_merge.py, test_qwen35_preemption.py,
    # test_qwen35_preemption_state.py, test_state_slot_reuse.py,
    # test_qwen35_multiblock.py, cuda_graph_consistency_test.py,
    # run_small_model_smoke_test.py, decode_stagger_contamination_check.py,
    # gsm8k_fused_gdr_check.py, debug_warmup_state.py, debug_kvcache.py,
    # test_qwen34_model_runner.py, plus the fixture-generation scripts
    # (make_fake_hf_config.py, make_fake_tokenizer.py,
    # add_fake_chat_template.py). Per-file check now done (see the intervention
    # that added this fix): most of the ModelRunner/LLMEngine-based ones
    # monkeypatch load_model() to a no-op and initialize their own weights
    # directly, so they never actually read this fixture's .safetensors
    # content and were unaffected by the zero-Experts bug either way.
    # test_state_slot_reuse.py, run_small_model_smoke_test.py, and
    # gsm8k_fused_gdr_check.py DO load this fixture for real (no override)
    # and route tokens through MoE with it -- those three are the genuine
    # dependents of this fix and were re-run against the regenerated fixture.
    init_reference_moe_experts_(ref_model, seed=42)

    port_model = Qwen35ForCausalLM(config).to("cuda").to(torch.bfloat16)

    copied, missed = copy_weights_to_port(port_model, ref_model)
    print(f"copy_weights_to_port: copied {len(copied)}, missed {len(missed)}")
    if missed:
        for name, ref_name, port_shape, ref_shape in missed[:10]:
            print(f"  MISSED: {name} (ref={ref_name}) port={port_shape} ref={ref_shape}")
    assert not missed, f"{len(missed)} parameters failed to copy -- see above"

    # Guardrail (additive only -- see tests/test_utils.py): "not missed"
    # only proves every port parameter NAME/SHAPE matched something in
    # copy_weights_to_port's "copied" list, same caveat as
    # test_qwen35_full_model.py -- doesn't independently prove the copied
    # values themselves are real, finite, non-degenerate numbers. Now passes
    # for real (init_reference_moe_experts_ above), not degenerately on
    # zero == zero.
    assert_all_parameters_initialized(
        port_model, whitelist_zero=known_zero_initialized_param_names(port_model)
    )

    save_dict = build_save_dict(port_model)

    non_finite = [name for name, t in save_dict.items() if not torch.isfinite(t).all()]
    if non_finite:
        raise AssertionError(f"non-finite values found in: {non_finite}")

    save_file(save_dict, OUT_PATH)

    print(f"Wrote {len(save_dict)} tensors to {OUT_PATH}")
    print("All tensors verified finite (no NaN/Inf).")

    import torch.distributed as dist
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
