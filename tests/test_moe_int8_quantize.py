"""Q5 -- CPU-only correctness test for MoE INT8 weight quantization.

============================================================================
WHAT THIS TESTS, AND WHAT IT DOES NOT
============================================================================
Tests quantize_weight_int8_grouped / dequantize_weight_int8_grouped from
tests/moe_int8_quantize.py at the REAL checkpoint's dimensions
(hidden_size=2048, moe_intermediate_size=512, group_size=128 -- see Q0),
using RANDOM weights, not real checkpoint weights.

  COVERED   Reconstruction error: quantize then dequantize a weight tensor,
            compare to the original. This bounds what quantization alone
            costs, before any forward pass is involved.
  COVERED   Downstream matmul error: x @ w.T (bf16) vs x @ dequant(w).T,
            for both gate_up_proj-shaped and down_proj-shaped tensors. This
            is the number that actually matters -- reconstruction error on
            the weight doesn't directly say what happens to a forward pass,
            because errors can partially cancel or partially compound
            across the contraction dimension.
  COVERED   Zero-weight-group edge case (the divide-by-zero guard).
  COVERED   Group-size divisibility assertion actually fires on a
            mismatched shape.
  NOT       Real checkpoint weights. RANDOM weights, matching this
            project's own established distinction (see the small-model
            docstrings throughout tests/) between infrastructure/shape
            correctness (this file) and precision-sensitivity questions
            that require real weights (deferred -- would need the real
            checkpoint loaded, which is a GPU-memory-bound operation this
            test deliberately avoids so it can run in seconds on CPU).
  NOT       The full MoE forward path, EP dispatch, or the decode
            integration. Those are Q4 and its own separate test.
  NOT       An accuracy ablation (GSM8K-style). That is Q6, and requires
            real weights through the engine, same as every other
            correctness-gate check in this project.

Precedent for random-weight-first, real-weight-second sequencing: the
fused GDR kernel and vectorized MoE dispatch were both first validated at
small/random-weight scale before any real-checkpoint measurement (see
qllm_plan.tex's experimental-protocol section). This follows the same
order, not a shortcut around it.

Usage: python tests/test_moe_int8_quantize.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_int8_quantize import (
    quantize_weight_int8_grouped,
    dequantize_weight_int8_grouped,
    quantize_experts_module,
)

# Real checkpoint dims, confirmed via Q0 (config.json: hidden_size=2048,
# moe_intermediate_size=512, num_experts=256, num_hidden_layers=40).
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512
GROUP_SIZE = 128
NUM_EXPERTS_SAMPLE = 8  # a subset, not all 256 -- quantization is per-expert
                         # independent, so this does not lose generality for
                         # a correctness check; it keeps the test fast.
SEED = 1234


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a-b| / max(|a|, eps), a single scalar summary. Deliberately
    crude (matches this project's own "max abs diff" / "max rel error"
    reporting style elsewhere) rather than a fancier norm, so the number is
    directly comparable to the MoE combine-step and EP-dispatch precision
    numbers already in the measurement notebook."""
    diff = (a.float() - b.float()).abs()
    denom = a.float().abs().clamp_min(1e-8)
    return (diff / denom).max().item()


def test_group_size_divides_real_dims():
    section("Group-size divisibility at real checkpoint dims")

    assert HIDDEN_SIZE % GROUP_SIZE == 0, (
        f"hidden_size={HIDDEN_SIZE} not divisible by group_size={GROUP_SIZE}"
    )
    assert INTERMEDIATE_SIZE % GROUP_SIZE == 0, (
        f"moe_intermediate_size={INTERMEDIATE_SIZE} not divisible by "
        f"group_size={GROUP_SIZE}"
    )
    print(f"  gate_up_proj in_features={HIDDEN_SIZE}: "
          f"{HIDDEN_SIZE // GROUP_SIZE} groups, exact")
    print(f"  down_proj    in_features={INTERMEDIATE_SIZE}: "
          f"{INTERMEDIATE_SIZE // GROUP_SIZE} groups, exact")
    print("  [PASS]")


def test_mismatched_shape_asserts_loudly():
    section("Divisibility assertion actually fires on a bad shape")

    bad = torch.randn(4, 10, 130)  # 130 not divisible by 128
    raised = False
    try:
        quantize_weight_int8_grouped(bad, GROUP_SIZE)
    except AssertionError:
        raised = True
    assert raised, "Expected an AssertionError for a non-divisible in_features"
    print("  [OK] AssertionError raised for in_features=130, group_size=128")
    print("  [PASS]")


def test_zero_group_does_not_produce_nan():
    section("Zero-weight group: divide-by-zero guard")

    w = torch.randn(2, 4, GROUP_SIZE * 2)
    w[0, 0, :GROUP_SIZE] = 0.0  # first group of first output channel, all zero

    int8_w, scale = quantize_weight_int8_grouped(w, GROUP_SIZE)
    assert torch.isfinite(scale).all(), "scale contains non-finite values"
    assert torch.isfinite(int8_w.float()).all()
    assert (int8_w[0, 0, :GROUP_SIZE] == 0).all(), (
        "an all-zero group should quantize to all-zero int8 values"
    )
    print("  [OK] all-zero group: scale finite, int8 values are 0, no NaN")
    print("  [PASS]")


def test_reconstruction_error_gate_up_proj_shape():
    section("Reconstruction error -- gate_up_proj shape "
            f"({NUM_EXPERTS_SAMPLE}, {2*INTERMEDIATE_SIZE}, {HIDDEN_SIZE})")

    torch.manual_seed(SEED)
    # Real trained weights are not unit-Gaussian; scale roughly matches
    # typical post-init transformer weight magnitudes (~0.02 std), not
    # because this proves anything about the real checkpoint's actual
    # distribution, but because quantization error scales with the
    # weight's dynamic range and this is a more representative range than
    # std=1.
    w = torch.randn(NUM_EXPERTS_SAMPLE, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE) * 0.02
    w = w.to(torch.bfloat16)

    int8_w, scale = quantize_weight_int8_grouped(w, GROUP_SIZE)
    w_hat = dequantize_weight_int8_grouped(int8_w, scale, GROUP_SIZE, torch.bfloat16)

    rel_err = relative_error(w, w_hat)
    max_abs_err = (w.float() - w_hat.float()).abs().max().item()

    print(f"  shape: {tuple(w.shape)}  dtype: {w.dtype}")
    print(f"  max relative error: {rel_err:.6f}")
    print(f"  max absolute error: {max_abs_err:.6e}")

    # INT8 symmetric RTN with 128-wide groups on roughly-Gaussian weights:
    # expect max relative error well under 1% for the vast majority of
    # values, with outliers near individual near-zero weights (where
    # relative error is not a meaningful metric -- see the EP dispatch
    # top_k=8 near-zero-reference-element caveat already documented in the
    # measurement notebook for exactly this reason). Threshold set loosely
    # to catch a broken implementation, not to certify a tight bound.
    assert rel_err < 0.5, (
        f"Reconstruction relative error {rel_err:.4f} is implausibly high "
        f"for INT8 RTN -- likely an implementation bug, not expected "
        f"quantization noise."
    )
    print("  [PASS] (threshold is a bug-catcher, not a precision claim --")
    print("         see test_matmul_error_* below for the number that matters)")


def test_reconstruction_error_down_proj_shape():
    section("Reconstruction error -- down_proj shape "
            f"({NUM_EXPERTS_SAMPLE}, {HIDDEN_SIZE}, {INTERMEDIATE_SIZE})")

    torch.manual_seed(SEED + 1)
    w = torch.randn(NUM_EXPERTS_SAMPLE, HIDDEN_SIZE, INTERMEDIATE_SIZE) * 0.02
    w = w.to(torch.bfloat16)

    int8_w, scale = quantize_weight_int8_grouped(w, GROUP_SIZE)
    w_hat = dequantize_weight_int8_grouped(int8_w, scale, GROUP_SIZE, torch.bfloat16)

    rel_err = relative_error(w, w_hat)
    print(f"  shape: {tuple(w.shape)}  dtype: {w.dtype}")
    print(f"  max relative error: {rel_err:.6f}")
    assert rel_err < 0.5
    print("  [PASS]")


def test_matmul_error_gate_up_proj():
    section("Downstream matmul error -- gate_up_proj, x @ w.T "
            "(bf16 vs quantized-dequantized)")

    torch.manual_seed(SEED + 2)
    n_tokens = 64
    w = torch.randn(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE) * 0.02  # single expert
    x = torch.randn(n_tokens, HIDDEN_SIZE) * 1.0
    w_bf16, x_bf16 = w.to(torch.bfloat16), x.to(torch.bfloat16)

    int8_w, scale = quantize_weight_int8_grouped(w_bf16, GROUP_SIZE)
    w_hat = dequantize_weight_int8_grouped(int8_w, scale, GROUP_SIZE, torch.bfloat16)

    out_ref = (x_bf16.float() @ w_bf16.float().t())
    out_quant = (x_bf16.float() @ w_hat.float().t())

    rel_err = relative_error(out_ref, out_quant)
    cos = torch.nn.functional.cosine_similarity(
        out_ref.flatten().unsqueeze(0), out_quant.flatten().unsqueeze(0)
    ).item()

    print(f"  x: ({n_tokens}, {HIDDEN_SIZE})  w: {tuple(w.shape)}")
    print(f"  output max relative error: {rel_err:.6f}")
    print(f"  output cosine similarity:  {cos:.6f}")

    # This is the number that actually predicts downstream impact, unlike
    # the raw weight reconstruction error above -- it is what a real
    # gate_up_proj forward call would see. Threshold again a bug-catcher:
    # a broken quantizer would show cosine well below 0.99; healthy INT8
    # RTN on a single linear layer typically lands >0.999.
    assert cos > 0.99, (
        f"Downstream matmul cosine similarity {cos:.4f} is too low for "
        f"healthy INT8 RTN quantization -- check the quantizer, not just "
        f"accept this number."
    )
    print("  [PASS]")


def test_matmul_error_down_proj():
    section("Downstream matmul error -- down_proj, x @ w.T")

    torch.manual_seed(SEED + 3)
    n_tokens = 64
    w = torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE) * 0.02
    x = torch.randn(n_tokens, INTERMEDIATE_SIZE) * 1.0
    w_bf16, x_bf16 = w.to(torch.bfloat16), x.to(torch.bfloat16)

    int8_w, scale = quantize_weight_int8_grouped(w_bf16, GROUP_SIZE)
    w_hat = dequantize_weight_int8_grouped(int8_w, scale, GROUP_SIZE, torch.bfloat16)

    out_ref = (x_bf16.float() @ w_bf16.float().t())
    out_quant = (x_bf16.float() @ w_hat.float().t())

    rel_err = relative_error(out_ref, out_quant)
    cos = torch.nn.functional.cosine_similarity(
        out_ref.flatten().unsqueeze(0), out_quant.flatten().unsqueeze(0)
    ).item()

    print(f"  x: ({n_tokens}, {INTERMEDIATE_SIZE})  w: {tuple(w.shape)}")
    print(f"  output max relative error: {rel_err:.6f}")
    print(f"  output cosine similarity:  {cos:.6f}")
    assert cos > 0.99
    print("  [PASS]")


def test_quantize_experts_module_wrapper():
    section("quantize_experts_module wrapper -- shapes and roundtrip")

    torch.manual_seed(SEED + 4)
    gate_up = (torch.randn(NUM_EXPERTS_SAMPLE, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE) * 0.02).to(torch.bfloat16)
    down = (torch.randn(NUM_EXPERTS_SAMPLE, HIDDEN_SIZE, INTERMEDIATE_SIZE) * 0.02).to(torch.bfloat16)

    packed = quantize_experts_module(gate_up, down, GROUP_SIZE)

    assert packed["gate_up_proj_int8"].shape == gate_up.shape
    assert packed["down_proj_int8"].shape == down.shape
    assert packed["gate_up_proj_int8"].dtype == torch.int8
    assert packed["down_proj_int8"].dtype == torch.int8

    gu_hat = dequantize_weight_int8_grouped(
        packed["gate_up_proj_int8"], packed["gate_up_proj_scale"], GROUP_SIZE, torch.bfloat16
    )
    dp_hat = dequantize_weight_int8_grouped(
        packed["down_proj_int8"], packed["down_proj_scale"], GROUP_SIZE, torch.bfloat16
    )

    print(f"  gate_up_proj_int8: {tuple(packed['gate_up_proj_int8'].shape)}  dtype={packed['gate_up_proj_int8'].dtype}")
    print(f"  gate_up_proj_scale: {tuple(packed['gate_up_proj_scale'].shape)}")
    print(f"  down_proj_int8:    {tuple(packed['down_proj_int8'].shape)}  dtype={packed['down_proj_int8'].dtype}")
    print(f"  down_proj_scale:    {tuple(packed['down_proj_scale'].shape)}")

    gu_rel_err = relative_error(gate_up, gu_hat)
    dp_rel_err = relative_error(down, dp_hat)
    print(f"  gate_up_proj reconstruction rel err: {gu_rel_err:.6f}")
    print(f"  down_proj reconstruction rel err:    {dp_rel_err:.6f}")

    print("  [PASS]")


def test_memory_footprint_estimate():
    section("Memory footprint: bf16 vs int8+scale, at REAL full-checkpoint scale")

    E, I, H, L = 256, INTERMEDIATE_SIZE, HIDDEN_SIZE, 40

    gate_up_params = E * (2 * I) * H
    down_params = E * H * I
    total_expert_params = L * (gate_up_params + down_params)

    bf16_bytes = total_expert_params * 2

    # int8 weight: 1 byte/param. Scale: 1 value per (out_channel, group) in
    # bf16 (2 bytes) -- group_size=128 means scale storage is 1/128th the
    # element count of the weight it covers, negligible in comparison.
    int8_bytes = total_expert_params * 1
    gate_up_scale_params = E * (2 * I) * (H // GROUP_SIZE)
    down_scale_params = E * H * (I // GROUP_SIZE)
    scale_bytes = L * (gate_up_scale_params + down_scale_params) * 2
    int8_total_bytes = int8_bytes + scale_bytes

    print(f"  total routed-expert params (all 40 layers): {total_expert_params/1e9:.2f}B")
    print(f"  bf16 size:  {bf16_bytes/1e9:.2f} GB")
    print(f"  int8 size:  {int8_bytes/1e9:.2f} GB (weights) + "
          f"{scale_bytes/1e9:.3f} GB (scales) = {int8_total_bytes/1e9:.2f} GB")
    print(f"  reduction:  {bf16_bytes/int8_total_bytes:.2f}x")
    print(f"  scale overhead: {scale_bytes/int8_bytes*100:.2f}% of the int8 weight size")

    assert int8_total_bytes < bf16_bytes / 1.8, (
        "Expected close to 2x reduction; scale overhead is eating too much "
        "of the win -- check GROUP_SIZE."
    )
    print("  [PASS]")


def main():
    test_group_size_divides_real_dims()
    test_mismatched_shape_asserts_loudly()
    test_zero_group_does_not_produce_nan()
    test_reconstruction_error_gate_up_proj_shape()
    test_reconstruction_error_down_proj_shape()
    test_matmul_error_gate_up_proj()
    test_matmul_error_down_proj()
    test_quantize_experts_module_wrapper()
    test_memory_footprint_estimate()

    print("\n" + "=" * 74)
    print("ALL Q5 CPU-ONLY QUANTIZATION CORRECTNESS CHECKS PASSED")
    print("=" * 74)
    print("Scope reminder: RANDOM weights at real dimensions, CPU only.")
    print("Real-checkpoint precision (Q6, GSM8K ablation) and the decode-path")
    print("integration (Q4, dequant-on-gather) remain to be done, and require")
    print("the real checkpoint and a GPU respectively.")


if __name__ == "__main__":
    main()