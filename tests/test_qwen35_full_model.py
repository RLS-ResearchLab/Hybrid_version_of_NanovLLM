# tests/test_qwen35_full_model.py
"""Phase 1 full-model standalone test.

Instantiates Qwen35ForCausalLM directly, copies weights from the reference
Qwen35MoESmall, and compares final-token logits on a single-shot forward pass.

Requires flash_attn/triton (real GPU) since Qwen35FullAttention needs them.

Usage:
    python tests/test_qwen35_full_model.py
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import (
    init_dist,
    load_reference_module,
    make_small_config,
)
from test_utils import known_zero_initialized_param_names, assert_all_parameters_initialized


def copy_weights_to_port(port_model, ref_model):
    """Copy reference weights into the port model, name-mapped.

    The port wraps everything under `model.` (Qwen35ForCausalLM.model.*)
    except lm_head, which sits at the top level in both. The one structural
    mismatch is the shared expert: the port fuses gate_proj/up_proj into a
    single gate_up_proj (MergedColumnParallelLinear), while the reference
    keeps them separate — same special-case handled in Test 7.
    """
    ref_sd = dict(ref_model.named_parameters())
    copied, missed = [], []

    for name, param in port_model.named_parameters():
        ref_name = name[len("model."):] if name.startswith("model.") else name

        if ref_name.endswith("mlp.shared_expert.gate_up_proj.weight"):
            gate_name = ref_name.replace("gate_up_proj", "gate_proj")
            up_name = ref_name.replace("gate_up_proj", "up_proj")
            if gate_name in ref_sd and up_name in ref_sd:
                param.data.copy_(torch.cat(
                    [ref_sd[gate_name].data, ref_sd[up_name].data], dim=0
                ))
                copied.append(name)
                continue

        if ref_name in ref_sd and param.shape == ref_sd[ref_name].shape:
            param.data.copy_(ref_sd[ref_name].data)
            copied.append(name)
        else:
            shape_info = ref_sd[ref_name].shape if ref_name in ref_sd else "MISSING"
            missed.append((name, ref_name, tuple(param.shape), shape_info))

    return copied, missed


def test_full_model_single_shot(device):
    """Full Qwen35ForCausalLM vs reference Qwen35MoESmall — final-token logits."""
    print("\n" + "=" * 70)
    print("TEST 9: Full Model — Single-Shot Logits vs Reference")
    print("=" * 70)

    from nanovllm.utils.context import set_context, reset_context
    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM

    ref_mod = load_reference_module()
    config = make_small_config()

    torch.manual_seed(2024)
    ref_model = ref_mod.Qwen35MoESmall().to(device).to(torch.bfloat16)

    port_model = Qwen35ForCausalLM(config).to(device).to(torch.bfloat16)

    copied, missed = copy_weights_to_port(port_model, ref_model)
    print(f"  Copied {len(copied)} params, missed {len(missed)}")
    if missed:
        for name, ref_name, port_shape, ref_shape in missed[:10]:
            print(f"    MISSED: {name} (ref={ref_name}) port={port_shape} ref={ref_shape}")
    assert not missed, f"{len(missed)} parameters failed to copy — see above"

    # Guardrail (additive only -- see tests/test_utils.py): "not missed"
    # above only proves every port parameter NAME/SHAPE matched something in
    # copy_weights_to_port's "copied" list -- it doesn't independently prove
    # the copied values themselves are real, finite, non-degenerate numbers.
    #
    # CONFIRMED FAILURE, NOT WORKED AROUND (per task instructions -- report,
    # don't silently fix): this DOES fail here, on every layer's
    # mlp.experts.gate_up_proj / experts.down_proj being all-zero. Root
    # cause: src/model_small_qwen3.5.py's Experts class constructs those as
    # `nn.Parameter(torch.empty(...))` with no explicit init anywhere in the
    # reference, and ref_model.layers.*.mlp.experts.gate_up_proj/down_proj
    # are ALREADY all-zero (confirmed directly) before copy_weights_to_port
    # ever runs -- the copy faithfully carries zero into port_model, "not
    # missed" included, since name/shape matching succeeds regardless of the
    # VALUE being copied. This test's headline cosine number (0.999967, per
    # make_fake_checkpoint.py's docstring) has therefore only ever validated
    # attention/norm/shared-expert/gating agreement, never the actual
    # per-expert gate_up_proj/down_proj matmul -- the core MoE computation.
    # Same root cause as tests/test_qwen35_standalone.py::test_moe_vs_reference
    # and it also poisons tests/fake_qwen35_small/model.safetensors (see
    # make_fake_checkpoint.py), since that fixture is built by this same
    # copy_weights_to_port off the same reference class. Not fixed here:
    # src/model_small_qwen3.5.py is the reference and out of scope to modify.
    assert_all_parameters_initialized(
        port_model, whitelist_zero=known_zero_initialized_param_names(port_model)
    )

    torch.manual_seed(321)
    T = 12
    input_ids_ref = torch.randint(100, 5000, (1, T), device=device)  # reference wants (B,T)
    flat_ids = input_ids_ref.view(-1)                                 # port wants flat (N,)
    positions = torch.arange(T, device=device)

    with torch.no_grad():
        # Reference: single-shot forward
        logits_ref, _, _, _ = ref_model(input_ids_ref)
        logits_ref_last = logits_ref[0, -1, :].float()

        # Port: single-shot prefill, no KV cache (k_cache/v_cache stay empty
        # tensors since we never went through ModelRunner.allocate_kv_cache)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
        set_context(True, cu_seqlens, cu_seqlens, T, T, None, None, None)
        hidden, _, _ = port_model(flat_ids, positions)
        logits_port = port_model.compute_logits(hidden)  # (1, VOCAB) — ParallelLMHead
        reset_context()                                   # already extracted last token
        logits_port_last = logits_port[0, :].float()

    cos = F.cosine_similarity(
        logits_ref_last.unsqueeze(0), logits_port_last.unsqueeze(0)
    ).item()
    top1_ref = logits_ref_last.argmax().item()
    top1_port = logits_port_last.argmax().item()
    top5_ref = set(logits_ref_last.topk(5).indices.tolist())
    top5_port = set(logits_port_last.topk(5).indices.tolist())
    overlap = len(top5_ref & top5_port) / 5

    print(f"  Cosine similarity (final token): {cos:.6f}")
    print(f"  Top-1 match: {top1_ref == top1_port} (ref={top1_ref}, port={top1_port})")
    print(f"  Top-5 overlap: {overlap:.2f}")

    assert cos > 0.95, f"Cosine too low: {cos}"
    assert top1_ref == top1_port, f"Top-1 mismatch: ref={top1_ref} port={top1_port}"
    print("  [PASS]")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Qwen3.5 Full-Model Test — device={device}")
    if device == "cpu":
        print("  WARNING: Qwen35FullAttention requires flash_attn (GPU only).")
        print("  This test will fail to construct the model on CPU.")

    init_dist()
    test_full_model_single_shot(device)

    print("\n" + "=" * 70)
    print("FULL MODEL TEST PASSED")
    print("=" * 70)

    import torch.distributed as dist
    dist.destroy_process_group()


if __name__ == "__main__":
    main()