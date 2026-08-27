"""CPU-only correctness check for moe_w8a8_hopper_integration.py's
quantize_experts_module_fp8_inplace / apply_moe_w8a8_hopper_quantization --
the load-time FP8 weight-quantization pass, exercised without CUDA, without
triton, and without compiling moe_w8a8.cu.

Sections 1-7 use a minimal _FakeExperts stand-in (same two Parameter names
the pass touches) for the low-level permutation/scale-layout math. Section 8
runs apply_moe_w8a8_hopper_quantization end-to-end against the REAL
Qwen35MoE / Experts -- possible on a CPU-only box since layers/fused_moe_int8.py's
triton import was made lazy 2026-08-27 (models/qwen3_5.py imports on CPU now).
engine/model_runner.py DOES call this pass when config.use_moe_w8a8_hopper is
set (2026-08-23), and _forward_gathered_w8a8_hopper / the FP8 elif branches
in _forward_dispatch* read the buffers it registers.

Does NOT validate: moe_w8a8.cu itself -- that still needs a real compile+run
on Hopper (Phase 0). Validates the in-place mutation (delete bf16 Parameters,
register the 6 FP8 buffers + kernel-permuted variant, set the group_size
attr), the model-walk, and the double-quantization guard.

Usage:
    python tests/test_moe_w8a8_hopper_integration_cpu.py
"""
import os
import sys
import types

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [_ROOT]
    _pkg.__file__ = os.path.join(_ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "layers"))
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

    # ---- 8. apply_moe_w8a8_hopper_quantization end-to-end against the REAL
    # Qwen35MoE / Experts. Previously skipped because models/qwen3_5.py had an
    # unconditional module-level `import triton`; that was made lazy
    # 2026-08-27 (layers/fused_moe_int8.py), so the real class imports on a
    # CPU-only box now. This exercises the actual model-walk + real-Experts
    # mutation, not just the _FakeExperts stand-in. ----
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29547")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")
    from nanovllm.models.qwen3_5 import Qwen35MoE, Experts
    from moe_w8a8_hopper_integration import apply_moe_w8a8_hopper_quantization

    torch.manual_seed(1)
    moe = Qwen35MoE(hidden_size=128, intermediate_size=256, shared_intermediate_size=256,
                    num_experts=8, top_k=2)
    with torch.no_grad():
        moe.experts.gate_up_proj.normal_(0, 0.02)
        moe.experts.down_proj.normal_(0, 0.02)

    n = apply_moe_w8a8_hopper_quantization(moe, group_size=128)
    print(f"[8] apply_moe_w8a8_hopper_quantization -> quantized {n} Experts module(s)")
    assert n == 1, f"expected 1 Experts module quantized, got {n}"
    exp = moe.experts
    assert not hasattr(exp, "gate_up_proj") and not hasattr(exp, "down_proj"), \
        "bf16 Experts Parameters must be deleted after quantization"
    for buf in ("gate_up_proj_fp8", "gate_up_proj_scale_fp8", "gate_up_proj_fp8_kernel",
                "gate_up_proj_scale_fp8_kernel", "down_proj_fp8", "down_proj_scale_fp8"):
        assert hasattr(exp, buf), f"missing FP8 buffer after quantization: {buf}"
    assert exp.gate_up_proj_fp8.dtype == torch.float8_e4m3fn
    assert getattr(exp, "moe_w8a8_hopper_group_size", None) == 128
    # Re-running must fail loudly (bf16 params already gone), not silently double-quantize.
    try:
        apply_moe_w8a8_hopper_quantization(moe, group_size=128)
        raise AssertionError("expected RuntimeError on double-quantization")
    except RuntimeError as e:
        assert "already gone" in str(e)
    print("    [OK] real Experts quantized, bf16 params deleted, 6 FP8 buffers registered, "
          "double-quant fails loudly")

    print("\nPASS" if ok else "\nFAIL")
    print("\nScope reminder: validates the load-time FP8 quantization pass (in-place mutation "
          "logic + model-walk against the real Experts class) -- says NOTHING about moe_w8a8.cu "
          "itself, which still needs a real compile+run on Hopper (Phase 0).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
