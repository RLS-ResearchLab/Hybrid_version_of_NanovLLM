"""CPU-only correctness check for lm_head_int8_integration.py's
quantize_lm_head_inplace / apply_lm_head_int8_quantization -- exercised
without CUDA, without triton, and without the real Qwen35ForCausalLM class
(models/qwen3_5.py has an unconditional triton import at module level,
confirmed absent on this Windows dev machine, so it cannot be imported here).

Uses minimal stand-in modules with the same attribute names/shapes the real
ParallelLMHead and Qwen35ForCausalLM have (.weight Parameter on the former,
.lm_head attribute on the latter) -- a faithful exercise of the actual
mutation/lookup logic, not a mock of behavior this test then just asserts
happened.

Does NOT validate: the real ParallelLMHead class, real model integration, or
the forward-path throughput question (see lm_head_int8_integration.py's own
module docstring -- that's explicitly not claimed to be resolved by this or
any CPU-only test).

Usage:
    python tests/test_lm_head_int8_integration_cpu.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_head_int8_integration import quantize_lm_head_inplace, apply_lm_head_int8_quantization  # noqa: E402


class _FakeParallelLMHead(nn.Module):
    """Minimal stand-in for layers/embed_head.py's real ParallelLMHead --
    same .weight Parameter name/shape quantize_lm_head_inplace actually
    reads/deletes, nothing else."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.02)


class _FakeCausalLM(nn.Module):
    """Minimal stand-in for Qwen35ForCausalLM -- same .lm_head attribute
    name apply_lm_head_int8_quantization's getattr(model, "lm_head", None)
    actually looks for."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lm_head = _FakeParallelLMHead(vocab_size, hidden_size)


def main():
    torch.manual_seed(0)
    ok = True

    V, H, group_size = 4096, 2048, 128
    lm_head = _FakeParallelLMHead(V, H)
    w_before = lm_head.weight.data.clone()

    quantize_lm_head_inplace(lm_head, group_size)

    # ---- 1. bf16/fp32 Parameter genuinely gone ----
    has_weight_param = "weight" in lm_head._parameters and lm_head._parameters["weight"] is not None
    print(f"[1] .weight still a Parameter: {has_weight_param}")
    assert not has_weight_param, "original weight Parameter should be deleted, not merely emptied"
    try:
        lm_head.weight
        raise AssertionError("lm_head.weight should raise AttributeError after deletion")
    except AttributeError:
        pass

    # ---- 2. int8 buffers present, correct dtype/shape ----
    assert hasattr(lm_head, "weight_int8") and lm_head.weight_int8.dtype == torch.int8
    assert lm_head.weight_int8.shape == (V, H)
    assert lm_head.weight_scale.shape == (V, H // group_size)
    print(f"[2] weight_int8={tuple(lm_head.weight_int8.shape)} weight_scale={tuple(lm_head.weight_scale.shape)}")

    # ---- 3. registered as buffers, not Parameters ----
    assert "weight_int8" in lm_head._buffers and "weight_int8" not in lm_head._parameters
    print("[3] registered as buffers, not Parameters -- confirmed")

    # ---- 4. group_size recorded ----
    assert lm_head.lm_head_int8_group_size == group_size
    print(f"[4] lm_head_int8_group_size={lm_head.lm_head_int8_group_size}")

    # ---- 5. dequantized values close to the original ----
    from moe_int8_quantize import dequantize_weight_int8_grouped
    deq = dequantize_weight_int8_grouped(lm_head.weight_int8, lm_head.weight_scale, group_size, torch.float32)
    cos = F.cosine_similarity(w_before.reshape(-1), deq.reshape(-1), dim=0).item()
    print(f"[5] dequant-vs-original cosine={cos:.6f}")
    assert cos > 0.999
    ok &= cos > 0.999

    # ---- 6. apply_lm_head_int8_quantization: presence-check wrapper ----
    model = _FakeCausalLM(V, H)
    n = apply_lm_head_int8_quantization(model, group_size)
    print(f"[6] apply_lm_head_int8_quantization returned {n} (expect 1)")
    assert n == 1
    assert hasattr(model.lm_head, "weight_int8")
    ok &= (n == 1)

    # ---- 7. no-lm_head model returns 0, doesn't crash ----
    class _NoLMHead(nn.Module):
        pass
    n0 = apply_lm_head_int8_quantization(_NoLMHead(), group_size)
    print(f"[7] no-lm_head model returned {n0} (expect 0)")
    assert n0 == 0
    ok &= (n0 == 0)

    print("\nPASS" if ok else "\nFAIL")
    print("\nScope reminder: validates the in-place mutation logic against stand-in modules only "
          "-- says nothing about the real ParallelLMHead class, engine integration, or the "
          "forward-path throughput question.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
