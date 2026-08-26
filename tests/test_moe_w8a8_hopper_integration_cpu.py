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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layers"))
from moe_w8a8_hopper_integration import (               # noqa: E402
    quantize_experts_module_fp8_inplace,
    _KERNEL_BLOCK_N,
    _KERNEL_WARP_N,
)
from moe_w8a8_hopper_quantize import gate_up_interleave_permutation  # noqa: E402


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

    # ---- 6. gate_up_proj_fp8_KERNEL: shape checks, plus the correctness
    # property the whole real-checkpoint fix (2026-08-26) rests on -- that
    # the interleaved buffer is the SAME logical weight as the contiguous
    # one, just with rows reordered by gate_up_interleave_permutation, not
    # some other accidental transformation. Checked two ways: (a) the
    # permutation is a genuine bijection of range(N), not e.g. a
    # many-to-one bug that would silently drop rows; (b) dequantizing the
    # kernel buffer and comparing it to the ORIGINAL bf16 weights permuted
    # by that same perm agrees closely (cosine > 0.99, same bar as check
    # [5]'s round-trip). This is exactly the kind of per-stage exact-value
    # check that found the original interleave bug on real hardware --
    # doing it here, on CPU, against synthetic weights, is what makes the
    # eventual real-Hopper validation a confirmation rather than a first
    # look. ----
    assert hasattr(experts, "gate_up_proj_fp8_kernel")
    assert experts.gate_up_proj_fp8_kernel.dtype == torch.float8_e4m3fn
    assert experts.gate_up_proj_fp8_kernel.shape == (E, N, K)
    assert experts.gate_up_proj_scale_fp8_kernel.shape == (E, N // group_size, K // group_size)

    MI = N // 2
    perm = gate_up_interleave_permutation(MI, _KERNEL_BLOCK_N, _KERNEL_WARP_N)
    assert perm.shape == (N,)
    assert torch.equal(torch.sort(perm).values, torch.arange(N)), (
        "gate_up_interleave_permutation must be a bijection of range(N) -- "
        "a duplicate or missing index here would mean some physical rows are "
        "silently dropped or double-counted when the kernel reads this buffer"
    )

    gu_deq_kernel = dequantize_weight_fp8_grouped_gathered(
        experts.gate_up_proj_fp8_kernel, experts.gate_up_proj_scale_fp8_kernel, group_size, torch.float32)
    gu_before_permuted = gu_before[:, perm, :]
    cos_kernel = torch.nn.functional.cosine_similarity(
        gu_before_permuted.reshape(-1), gu_deq_kernel.reshape(-1), dim=0).item()
    print(f"[6] gate_up_proj_fp8_kernel dequant-vs-permuted-original cosine={cos_kernel:.6f}")
    assert cos_kernel > 0.99, "interleaved buffer should decode back to the permuted original weights"
    ok &= cos_kernel > 0.99

    # Cross-check in the OTHER direction too: un-permuting the kernel
    # buffer's dequant via the inverse permutation should recover the same
    # values as the plain contiguous buffer's own dequant (gu_deq from
    # check [5]) -- confirms the two registered buffers really are two
    # views of the same logical weight, not independently-drifted copies.
    inv_perm = torch.argsort(perm)
    cos_cross = torch.nn.functional.cosine_similarity(
        gu_deq.reshape(-1), gu_deq_kernel[:, inv_perm, :].reshape(-1), dim=0).item()
    print(f"[6b] contiguous vs. un-permuted-kernel cross-check cosine={cos_cross:.6f}")
    assert cos_cross > 0.99, "contiguous and interleaved buffers disagree on the underlying weight"
    ok &= cos_cross > 0.99

    # ---- 7. The OTHER supported kernel config, (block_n, warp_n)=(64, 4) --
    # moe_w8a8_hopper_integration.py's _KERNEL_BLOCK_N/_KERNEL_WARP_N are
    # hardcoded to (32, 8) (matching models/qwen3_5.py's production launch
    # config), so this exercises gate_up_interleave_permutation and
    # quantize_weight_fp8_grouped directly at (64, 4) instead -- the same
    # round-trip check as [6]/[6b], for the config nothing else here uses.
    # This does NOT prove the kernel itself is correct at (64, 4) (needs
    # real Hopper hardware, see layers/smoke_test_moe_w8a8_hopper.py's
    # --block-n/--warp-n flags for that) -- it proves the PERMUTATION
    # formula stays a valid, balanced, self-consistent bijection at the
    # config nothing here had tested before 2026-08-26. ----
    from moe_w8a8_hopper_quantize import quantize_weight_fp8_grouped as _qwfg
    perm_64_4 = gate_up_interleave_permutation(MI, 64, 4)
    assert perm_64_4.shape == (N,)
    assert torch.equal(torch.sort(perm_64_4).values, torch.arange(N)), (
        "gate_up_interleave_permutation(MI, 64, 4) must also be a bijection -- "
        "(64, 4) is a real dispatched kernel config (moe_w8a8.cu:dispatch_bn_wn), "
        "not a hypothetical one"
    )
    n_gate_64_4 = int((perm_64_4 < MI).sum())
    assert n_gate_64_4 == MI, f"expected exactly MI={MI} gate rows, got {n_gate_64_4}"
    gu_fp8_64_4, gu_scale_64_4 = _qwfg(gu_before[:, perm_64_4, :], group_size)
    gu_deq_64_4 = dequantize_weight_fp8_grouped_gathered(gu_fp8_64_4, gu_scale_64_4, group_size, torch.float32)
    cos_64_4 = torch.nn.functional.cosine_similarity(
        gu_before[:, perm_64_4, :].reshape(-1), gu_deq_64_4.reshape(-1), dim=0).item()
    print(f"[7] (block_n,warp_n)=(64,4) permutation dequant-vs-permuted-original cosine={cos_64_4:.6f}")
    assert cos_64_4 > 0.99
    ok &= cos_64_4 > 0.99

    # ---- 8. apply_moe_w8a8_hopper_quantization's per-module count is
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
