"""CPU-only correctness check for moe_w8a8_hopper_integration.py's
quantize_experts_module_fp8_inplace / apply_moe_w8a8_hopper_quantization --
the load-time FP8 weight-quantization pass, exercised without CUDA, without
triton, and without the real Experts class (models/qwen3_5.py has an
unconditional triton import at module level, confirmed empirically absent on
this Windows dev machine, so it cannot be imported here at all).

Uses a minimal stand-in nn.Module with the same two Parameter names
(gate_up_proj, down_proj) the real Experts module has -- this is a faithful
exercise of quantize_experts_module_fp8_inplace's actual logic (it only ever
touches those two attribute names), not a mock of behavior this test then
just asserts happened.

Does NOT validate: moe_w8a8.cu itself, the real Experts class, or engine
integration (nothing calls apply_moe_w8a8_hopper_quantization from
ModelRunner yet -- see moe_w8a8_hopper_integration.py's own module
docstring). Validates only that the in-place mutation this pass performs --
delete bf16 Parameters, register FP8 buffers, set the group_size attribute --
does what it claims, correctly and without leaking the old memory reference.

Usage:
    python tests/test_moe_w8a8_hopper_integration_cpu.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_w8a8_hopper_integration import quantize_experts_module_fp8_inplace  # noqa: E402


class _FakeExperts(nn.Module):
    """Minimal stand-in for models/qwen3_5.py's real Experts module -- same
    two Parameter names and shapes quantize_experts_module_fp8_inplace
    actually reads/deletes, nothing else. Not importing the real class is a
    scope limitation (see module docstring), not a design choice."""

    def __init__(self, E, N, K):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, N, K) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(E, K, N // 2) * 0.02)


def main():
    torch.manual_seed(0)
    ok = True

    E, N, K, group_size = 8, 256, 256, 128
    experts = _FakeExperts(E, N, K)

    gu_before = experts.gate_up_proj.data.clone()
    dp_before = experts.down_proj.data.clone()

    quantize_experts_module_fp8_inplace(experts, group_size)

    # ---- 1. bf16 Parameters genuinely gone, not just set to None ----
    has_gu_param = "gate_up_proj" in experts._parameters and experts._parameters["gate_up_proj"] is not None
    has_dp_param = "down_proj" in experts._parameters and experts._parameters["down_proj"] is not None
    print(f"[1] gate_up_proj still a Parameter: {has_gu_param}  down_proj still a Parameter: {has_dp_param}")
    assert not has_gu_param and not has_dp_param, "bf16 Parameters should be deleted, not merely emptied"
    try:
        experts.gate_up_proj
        raise AssertionError("experts.gate_up_proj should raise AttributeError after deletion")
    except AttributeError:
        pass
    ok &= True

    # ---- 2. FP8 buffers present, correct dtype, correct shape ----
    assert hasattr(experts, "gate_up_proj_fp8") and experts.gate_up_proj_fp8.dtype == torch.float8_e4m3fn
    assert hasattr(experts, "down_proj_fp8") and experts.down_proj_fp8.dtype == torch.float8_e4m3fn
    assert experts.gate_up_proj_fp8.shape == (E, N, K)
    assert experts.down_proj_fp8.shape == (E, K, N // 2)
    assert experts.gate_up_proj_scale_fp8.shape == (E, N // group_size, K // group_size)
    assert experts.down_proj_scale_fp8.shape == (E, K // group_size, (N // 2) // group_size)
    print(f"[2] gate_up_proj_fp8={tuple(experts.gate_up_proj_fp8.shape)} "
          f"scale={tuple(experts.gate_up_proj_scale_fp8.shape)}  "
          f"down_proj_fp8={tuple(experts.down_proj_fp8.shape)} "
          f"scale={tuple(experts.down_proj_scale_fp8.shape)}")

    # ---- 3. registered as buffers (persistent state, not Parameters --
    # quantized weights shouldn't receive gradients) ----
    assert "gate_up_proj_fp8" in experts._buffers and "gate_up_proj_fp8" not in experts._parameters
    assert "down_proj_fp8" in experts._buffers and "down_proj_fp8" not in experts._parameters
    print("[3] FP8 weights registered as buffers, not Parameters -- confirmed")

    # ---- 4. group_size recorded, matches what was passed ----
    assert experts.moe_w8a8_hopper_group_size == group_size
    print(f"[4] moe_w8a8_hopper_group_size={experts.moe_w8a8_hopper_group_size}")

    # ---- 5. dequantized values are close to the original bf16 weights --
    # not bit-exact (lossy quantization is the point), but in the right
    # ballpark. Reuses dequantize_weight_fp8_grouped_gathered with no
    # leading batch dims (E acts as the batch dim here, same function). ----
    from moe_w8a8_hopper_quantize import dequantize_weight_fp8_grouped_gathered
    gu_deq = dequantize_weight_fp8_grouped_gathered(
        experts.gate_up_proj_fp8, experts.gate_up_proj_scale_fp8, group_size, torch.float32)
    cos = torch.nn.functional.cosine_similarity(gu_before.reshape(-1), gu_deq.reshape(-1), dim=0).item()
    print(f"[5] gate_up_proj dequant-vs-original cosine={cos:.6f}")
    assert cos > 0.99, "dequantized weights should closely match the original bf16 values"
    ok &= cos > 0.99

    # ---- 6. apply_moe_w8a8_hopper_quantization's per-module count is
    # exercised separately (it imports the real Experts class, which cannot
    # be imported on this machine) -- explicitly out of scope here, not
    # silently skipped. ----
    print("\n[skipped] apply_moe_w8a8_hopper_quantization (walks real model.modules(), "
          "requires importing models.qwen3_5.Experts -- blocked by the unconditional "
          "triton import on this machine, needs a real GPU environment).")

    print("\nPASS" if ok else "\nFAIL")
    print("\nScope reminder: validates quantize_experts_module_fp8_inplace's in-place "
          "mutation logic against a stand-in module only -- says nothing about the real "
          "Experts class, moe_w8a8.cu, or engine wiring (none of which exist yet for this path).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
