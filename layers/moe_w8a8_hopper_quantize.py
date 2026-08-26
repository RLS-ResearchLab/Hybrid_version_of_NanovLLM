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


def quantize_activation_fp8_dynamic(x: torch.Tensor, block_size: int = 128):
    """Per-token, per-128-K-block dynamic FP8 e4m3 quantization, max-abs scale.

    x: (M, K) any float dtype, K divisible by block_size.
    Returns: (x_fp8: (M, K) float8_e4m3fn, scale: (M, K // block_size) float32).

    CORRECTED 2026-08-24: previously computed one scale per WHOLE ROW (shape
    (M,)), not one per (token, 128-K-block) -- found via a first real-hardware
    run of moe_w8a8_hopper's smoke test that a "value bug" (cosine~0.01,
    kernel_out ~227,000x too large on tokens whose index landed the read
    out-of-bounds) traced all the way back to this function. moe_w8a8.cu:811
    reads x_scale as `x_scale[token*(K/block_shape[1]) + k_block]` -- the
    SAME 128x128-block convention already used (and confirmed, per moe_w8a8.h)
    for w_scale/w2_scale -- so it needs K/block_size values per token, not 1.
    The old (M,) shape meant every read past token (M*block_size/K rows) or
    so wrapped into a neighboring token's scale (in-bounds but wrong -- small,
    plausible-looking, uncorrelated output) or straight past the end of the
    tensor (out-of-bounds -- picked up whatever GPU memory happened to be
    adjacent, producing the ~227,000x blowup). This shape is what the kernel
    has always expected; nothing in moe_w8a8.cu changed.

    No calibration data, no offline statistics pass -- computed fresh from
    the actual activation values every call, same per-block max-abs
    convention as quantize_weight_fp8_grouped (weights) and the kernel's own
    down_proj-input requantization (`token_scale[tm][t] = token_max/fp8_max`).

    Outlier handling: NOT addressed here beyond the 128-block granularity
    itself. See w8a8_activation_quant_scoping_memo.md §2c for the open
    decision on whether this needs SmoothQuant-style offline rebalancing
    before it's trustworthy on real activations.
    """
    M, K = x.shape
    assert K % block_size == 0, (
        f"K={K} must be divisible by block_size={block_size} -- moe_w8a8.cu's "
        f"block_shape={{128,128}} is hardcoded, not parameterized."
    )
    xb = x.float().view(M, K // block_size, block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / FP8_MAX
    x_fp8 = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8.view(M, K).contiguous(), scale.squeeze(-1).contiguous()


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
    assert group_size == 128, (
        f"group_size={group_size} requested, but moe_w8a8.cu's block_shape={{128,128}} is "
        f"hardcoded, not parameterized -- Config.moe_w8a8_hopper_weight_group_size looks like "
        f"an independent tuning knob (mirroring the INT8 scheme's own group_size field) but "
        f"isn't one for this kernel. A different value here would silently misalign this "
        f"function's scale tiles against the kernel's fixed indexing rather than error -- "
        f"caught here instead."
    )
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


def gate_up_interleave_permutation(MI, block_n=32, warp_n=8):
    """Row-permutation for moe_w8a8.cu's up-proj wgmma, found 2026-08-24.
    Promoted here from layers/smoke_test_moe_w8a8_hopper.py (single source of
    truth, same reasoning as every other function in this file) -- both the
    isolated kernel smoke test and tests/moe_w8a8_hopper_integration.py's
    real-checkpoint quantization path import it from here now.

    The kernel's SwiGLU combine pairs wgmma accumulator register c0
    (physical gate_up_proj weight-row r) with c2 (row r+8) as (gate, up) for
    the SAME logical down-proj-input feature -- derived from the Hopper
    wgmma m64nNk32 F32-accumulator fragment layout (thread 0 owns
    (0,0),(0,1),(8,0),(8,1), repeating every 8 N-columns) combined with this
    kernel's own warpgroup/tn/warp_local tiling of the N axis. That means
    the kernel needs gate and up interleaved every 8 physical rows, NOT laid
    out as the conventional contiguous halves (gate=[0,MI), up=[MI,2*MI))
    that both this test's naive synthetic weights and any standard
    gate_up_proj checkpoint use. Confirmed empirically: hand-computing
    SwiGLU from the kernel's own (independently verified-correct) gate/up
    f_acc values under the contiguous assumption gave a value ~59x off and
    opposite-signed from the reference at the matching coordinate.

    Returns a LongTensor `perm` of length 2*MI such that indexing a
    contiguous-layout (2*MI, ...) weight as `w[perm]` produces the
    interleaved layout the kernel needs. CONFIG-DEPENDENT: this permutation
    is only valid for the (block_n, warp_n) pair the kernel is actually
    launched with -- recompute it if those change (models/qwen3_5.py's
    _W8A8_HOPPER_BLOCK_N / _W8A8_HOPPER_WARP_N are the production values,
    32/8 -- kept as this function's defaults so a caller that forgets to
    pass them explicitly still matches production rather than silently
    diverging).
    """
    rows_per_block = block_n * warp_n
    rows_per_warpgroup = block_n * 4
    rows_per_tn = 64
    logical_per_block = rows_per_block // 2
    logical_per_warpgroup = logical_per_block // 2
    N = 2 * MI
    assert N % rows_per_block == 0, (
        f"2*MI={N} must be divisible by block_n*warp_n={rows_per_block}"
    )
    num_blocks = N // rows_per_block

    perm = torch.empty(N, dtype=torch.long)
    for block in range(num_blocks):
        for p_local in range(rows_per_block):
            warpgroup = p_local // rows_per_warpgroup
            p_in_wg = p_local % rows_per_warpgroup
            tn = p_in_wg // rows_per_tn
            p_in_tn = p_in_wg % rows_per_tn
            band = p_in_tn // 16       # warp_local, 0-3
            offset = p_in_tn % 16      # 0-15
            is_gate = offset < 8
            lane4 = offset % 8
            logical_in_tn = band * 8 + lane4   # 0-31 (within this tn)
            logical_feature = (block * logical_per_block + warpgroup * logical_per_warpgroup
                                + tn * 32 + logical_in_tn)
            p_global = block * rows_per_block + p_local
            perm[p_global] = logical_feature if is_gate else (MI + logical_feature)
    return perm
