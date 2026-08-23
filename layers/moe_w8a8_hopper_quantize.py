"""Production FP8 e4m3 quantization for the Hopper wgmma/TMA fused MoE kernel
(moe_w8a8.cu). Promoted out of smoke_test_moe_w8a8_hopper.py, which defined
these inline as test fixtures -- given a production home here instead,
deliberately not repeating the existing INT8 scheme's layout wart
(moe_int8_quantize.py lives under tests/, "not on any production sys.path...
an implicit dependency on import order, not a guarantee" per
models/qwen3_5.py's own comment on it).

UNVALIDATED against the real kernel -- see moe_w8a8.h's docstring: this is
"this test's own best-effort interpretation" of what moe_w8a8.cu expects,
inferred from reading its TMA setup and scale-indexing code, not confirmed
against it running (no CUDA toolchain on this dev machine). The CPU-only
checks in tests/test_moe_w8a8_hopper_quant_helpers_cpu.py and
tests/test_moe_w8a8_hopper_tp_ep_sharding_cpu.py validate these functions'
internal consistency and TP/EP-sharding safety, not their correctness against
the actual kernel -- that needs a real compile+run on Hopper (P1 in
H200_test_day_checklist.md).

smoke_test_moe_w8a8_hopper.py imports these from here now rather than
defining them -- single source of truth for both the isolated kernel smoke
test and any production integration code.
"""
import torch

FP8_MAX = 448.0  # e4m3's max representable magnitude -- matches moe_w8a8.cu's
                  # hardcoded fp8_max constant exactly, must stay in sync with it.


def quantize_activation_fp8_dynamic(x: torch.Tensor):
    """Per-token (per-row) dynamic FP8 e4m3 quantization, max-abs scale.

    x: (M, K) any float dtype.
    Returns: (x_fp8: (M, K) float8_e4m3fn, scale: (M,) float32).

    No calibration data, no offline statistics pass -- computed fresh from
    the actual activation values every call. This is this project's OWN
    reference quantization scheme for activations -- moe_w8a8.cu does not
    quantize its own input (see moe_w8a8.h), so there is no "real" activation
    quantizer to match against yet. The kernel's own per-token dynamic
    requantization of the SiLU-gated intermediate (the down_proj input,
    computed mid-kernel) uses this exact max-abs/448 convention, per its
    `token_scale[tm][t] = float(token_max[tm].t) / fp8_max` line -- so this
    mirrors that pattern for the INPUT activation too, as the most consistent
    assumption available, not because it's been confirmed as what an eventual
    real quantizer will do.

    Outlier handling: NOT addressed here. This is plain per-token max-abs,
    the scheme most exposed to a few outlier channels dominating a token's
    scale and starving the rest of that row's precision -- see
    w8a8_activation_quant_scoping_memo.md §2c for the open decision on
    whether this needs SmoothQuant-style offline rebalancing before it's
    trustworthy on real activations. Shipped as-is here because it's the
    cheapest starting point and the accuracy validation chain (Phase 3) is
    what will actually tell us if it's a problem, not speculation now.
    """
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / FP8_MAX
    x_fp8 = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1).contiguous()


def quantize_weight_fp8_grouped(weight: torch.Tensor, group_size: int = 128):
    """FP8 e4m3 quantization, 2D-blocked (group_size x group_size), matching
    moe_w8a8.cu's hardcoded `block_shape[2] = {128, 128}` -- ONE scale per
    128x128 tile of the (N, K) weight matrix, not per-row/per-group along a
    single dimension like this project's existing INT8 scheme
    (moe_int8_quantize.py, grouped along K only). Getting this 1D-vs-2D
    distinction wrong is exactly the kind of thing that would silently
    misalign scales against the kernel's own indexing -- see moe_w8a8.h.

    weight: (E, N, K), any float dtype. N and K must both be divisible by
        group_size.
    Returns: (w_fp8: (E, N, K) float8_e4m3fn,
              scale: (E, N // group_size, K // group_size) float32).

    TP/EP-sharding safety CONFIRMED (tests/test_moe_w8a8_hopper_tp_ep_sharding_cpu.py,
    2026-08-22, CPU-only, real shard_experts_tensor, tp_size in {1,2,4,8}):
    quantize-then-shard is bitwise identical to shard-then-quantize, because
    quantization has no cross-expert (dim-0) interaction and
    shard_experts_tensor only ever indexes along dim 0. Safe to call this on
    an already-sharded local Experts slice, same as the INT8 scheme.
    """
    E, N, K = weight.shape
    assert N % group_size == 0 and K % group_size == 0, (
        f"N={N}, K={K} must both be divisible by group_size={group_size} -- "
        f"moe_w8a8.cu's block_shape={{128,128}} is hardcoded, not parameterized."
    )
    w = weight.float().view(E, N // group_size, group_size, K // group_size, group_size)
    amax = w.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-8)
    scale = amax / FP8_MAX
    w_fp8 = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w_fp8.view(E, N, K).contiguous(), scale.view(E, N // group_size, K // group_size).contiguous()


def dequantize_weight_fp8_grouped_gathered(w_fp8_gathered: torch.Tensor, scale_gathered: torch.Tensor,
                                            group_size: int, out_dtype: torch.dtype):
    """Inverse of quantize_weight_fp8_grouped, generalized to accept extra
    leading (batch) dims -- e.g. after gathering per-expert weights via
    `weight[local_slots]`, shape becomes (..., N, K) instead of (N, K).

    w_fp8_gathered: (..., N, K) float8_e4m3fn.
    scale_gathered: (..., N // group_size, K // group_size) float32.
    """
    *lead, N, K = w_fp8_gathered.shape
    w = w_fp8_gathered.float().view(*lead, N // group_size, group_size, K // group_size, group_size)
    s = scale_gathered.view(*lead, N // group_size, 1, K // group_size, 1)
    return (w * s).view(*lead, N, K).to(out_dtype)
