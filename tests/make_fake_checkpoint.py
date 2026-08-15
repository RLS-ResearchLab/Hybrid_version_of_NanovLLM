# tests/make_fake_checkpoint.py
"""Builds tests/fake_qwen35_small/model.safetensors so ModelRunner has real,
finite weights to load instead of uninitialized nn.Parameter(torch.empty(...))
memory (LinearBase, Experts, VocabParallelEmbedding/ParallelLMHead all skip
default init).

Reuses test_qwen35_full_model.py's copy_weights_to_port(port_model, ref_model)
-- validated at cosine 0.999967 against the reference implementation -- to
populate a Qwen35ForCausalLM from the reference Qwen35MoESmall (plain
nn.Linear, so it initializes correctly with finite values).

The saved tensor names must match what nanovllm.utils.loader.load_model()
expects: every port parameter name as-is, EXCEPT the shared-expert gate/up
projection, which the port model fuses into a single gate_up_proj but
load_model's packed_modules_mapping only fires on the split HF-style names
("shared_expert.gate_proj" / "shared_expert.up_proj"). Saving the fused name
directly would skip the packed-mapping branch and call
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
from test_utils import known_zero_initialized_param_names, assert_all_parameters_initialized

OUT_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")


def build_save_dict(model):
    """Return {name: cpu_tensor} for safetensors, splitting any fused
    packed_modules_mapping targets back into their HF-style split names."""
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
                save_dict[split_name] = chunks[shard_id].detach().clone().contiguous().cpu()
        else:
            save_dict[name] = param.detach().clone().contiguous().cpu()
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
    # values themselves are real, finite, non-degenerate numbers.
    #
    # CONFIRMED FAILURE, NOT WORKED AROUND (per task instructions -- report,
    # don't silently fix): this DOES fail here, same root cause as
    # test_qwen35_full_model.py -- ref_model's Experts class
    # (src/model_small_qwen3.5.py) never initializes gate_up_proj/down_proj,
    # they read back all-zero, and copy_weights_to_port faithfully carries
    # that zero into port_model. This model is the direct source of
    # tests/fake_qwen35_small/model.safetensors, so that FIXTURE FILE has
    # all-zero expert weights baked into it on disk. 17 files under tests/
    # reference "fake_qwen35_small" (grep-confirmed): this file and
    # test_qwen35_full_model.py (the ones directly implicated above), plus
    # test_loader_shard_merge.py, test_qwen35_preemption.py,
    # test_qwen35_preemption_state.py, test_state_slot_reuse.py,
    # test_qwen35_multiblock.py, cuda_graph_consistency_test.py,
    # run_small_model_smoke_test.py, decode_stagger_contamination_check.py,
    # gsm8k_fused_gdr_check.py, debug_warmup_state.py, debug_kvcache.py,
    # test_qwen34_model_runner.py, plus the fixture-generation scripts
    # (make_fake_hf_config.py, make_fake_tokenizer.py,
    # add_fake_chat_template.py). NOT all of those necessarily exercise the
    # MoE expert forward path (some only touch KV-cache/scheduler mechanics)
    # -- that would need a per-file check this pass didn't do -- but any of
    # them that construct a model from this checkpoint and route tokens
    # through MoE are getting zero-valued expert weights, silently. Not
    # fixed here: src/model_small_qwen3.5.py is the reference and out of
    # scope to modify.
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
