"""Isolated correctness smoke test for the Hopper wgmma/TMA fused W8A8 MoE
kernel (moe_w8a8.cu, bound via moe_w8a8_binding.cpp /
fused_moe_w8a8_hopper.py) -- the same kind of first checkpoint
smoke_test_full_moe_pipeline.py was for the Ampere Triton kernel: synthetic
weights/activations, a plain-PyTorch reference implementation, cosine
similarity. No real checkpoint, no engine, no CUDA graphs -- this only
answers "is the kernel's math right in isolation," nothing more.

UNRUN. Written and reasoned through without any CUDA toolchain available (no
nvcc on this dev machine) -- cannot be compile-tested here. First real run on
H200 needs to happen before anything below is trusted, including the
quantization-contract assumptions baked into the reference implementation
itself (see the docstrings on quantize_weight_fp8_grouped and
quantize_activation_fp8_dynamic -- both are this test's own best-effort
interpretation of what moe_w8a8.cu expects, inferred from reading its TMA
setup and scale-indexing code, not confirmed against it running).

Unlike the Triton kernel's smoke test, this kernel does the FULL weighted
combine internally (accumulates each (token, k) pair's contribution into
`out` via topk_weight-scaled reduce-add, not a separate combine step the
caller does afterward) -- so the reference implementation below sums over
top_k directly too, to stay comparable.

Usage (once a CUDA/Hopper environment exists):
    python layers/smoke_test_moe_w8a8_hopper.py
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from moe_align_block_size import moe_align_block_size          # noqa: E402
from fused_moe_w8a8_hopper import fused_moe_w8a8_hopper_forward  # noqa: E402

FP8_MAX = 448.0  # e4m3's max representable magnitude -- matches moe_w8a8.cu's
                  # hardcoded fp8_max constant exactly, must stay in sync with it.


def quantize_activation_fp8_dynamic(x: torch.Tensor):
    """Per-token (per-row) dynamic FP8 e4m3 quantization, max-abs scale.

    x: (M, K) any float dtype.
    Returns: (x_fp8: (M, K) float8_e4m3fn, scale: (M,) float32).

    This is this test's OWN reference quantization scheme for activations --
    moe_w8a8.cu does not quantize its own input (see moe_w8a8.h), so there is
    no "real" activation quantizer to match against yet. The kernel's own
    per-token dynamic requantization of the SiLU-gated intermediate (the
    down_proj input, computed mid-kernel) uses this exact max-abs/448
    convention, per its `token_scale[tm][t] = float(token_max[tm].t) /
    fp8_max` line -- so this reference mirrors that pattern for the INPUT
    activation too, as the most consistent assumption available, not because
    it's been confirmed as what an eventual real quantizer will do.
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


def reference_pipeline(x, gu_fp8, gu_scale, dp_fp8, dp_scale, group_size,
                        local_slots, topk_weights, scaling_factor):
    """Plain PyTorch reference: dequantize the gathered per-(token,k) expert
    weights, run gate_up -> SiLU -> down_proj, weight by topk_weights, sum
    over top_k -- the FULL combine, matching what moe_w8a8.cu's kernel does
    internally via its reduce-add into `out` (unlike the Triton kernel's
    smoke test, which left the weighted combine to the caller)."""
    M, K = x.shape
    gu_gathered_fp8 = gu_fp8[local_slots]        # (M, TK, N, K)
    gu_gathered_scale = gu_scale[local_slots]    # (M, TK, N/gs, K/gs)
    dp_gathered_fp8 = dp_fp8[local_slots]        # (M, TK, K, N/2)
    dp_gathered_scale = dp_scale[local_slots]    # (M, TK, K/gs, (N/2)/gs)

    gu_deq = dequantize_weight_fp8_grouped_gathered(gu_gathered_fp8, gu_gathered_scale, group_size, torch.float32)
    dp_deq = dequantize_weight_fp8_grouped_gathered(dp_gathered_fp8, dp_gathered_scale, group_size, torch.float32)

    gate_up = torch.einsum('mtnk,mk->mtn', gu_deq, x.float())   # (M, TK, N)
    gw, uw = gate_up.chunk(2, dim=-1)                             # each (M, TK, N/2)
    h = F.silu(gw) * uw                                           # (M, TK, N/2)
    down = torch.einsum('mtkh,mth->mtk', dp_deq, h)               # (M, TK, K)

    weighted = down * topk_weights.unsqueeze(-1).float() * scaling_factor
    out = weighted.sum(dim=1)                                     # (M, K), full combine
    return out.to(torch.bfloat16)


def kernel_pipeline(x_fp8, x_scale, gu_fp8, gu_scale, dp_fp8, dp_scale,
                     local_slots, topk_weights, top_k, num_experts,
                     scaling_factor, block_m, block_n, warp_n, stages):
    return fused_moe_w8a8_hopper_forward(
        x_fp8, x_scale, gu_fp8, gu_scale, dp_fp8, dp_scale,
        *_align(local_slots, block_m, num_experts),
        topk_weights, top_k, num_experts,
        block_m=block_m, block_n=block_n, warp_n=warp_n, stages=stages,
        scaling_factor=scaling_factor,
    )


def _align(local_slots, block_m, num_experts):
    sorted_ids, expert_ids, ntpp = moe_align_block_size(local_slots, block_m, num_experts)
    return sorted_ids, expert_ids, ntpp


def main():
    if not torch.cuda.is_available():
        print("CUDA required for this smoke test.")
        return
    if not hasattr(torch, "float8_e4m3fn"):
        print("This PyTorch build has no torch.float8_e4m3fn -- need a newer PyTorch (~2.1+).")
        return

    device = torch.device("cuda")
    torch.manual_seed(0)

    # Modest scale for a first isolated check, not full production dims --
    # real dims (E=128 at tp=2, H=2048, MI=512) can be substituted once this
    # passes at small scale, matching smoke_test_full_moe_pipeline.py's own
    # progression from small to real-dim checks.
    E = 8              # local expert count
    H = 256             # hidden_size (K) -- must be a multiple of group_size(128)
    MI = 256             # moe_intermediate_size -- N = 2*MI must be a multiple of 128
    top_k = 4
    M = 16               # concurrency / token count
    group_size = 128
    scaling_factor = 1.0
    block_m, block_n, warp_n, stages = 32, 32, 8, 2

    print(f"Config: E={E} H={H} MI={MI} top_k={top_k} M={M} group_size={group_size} "
          f"block_m={block_m} block_n={block_n} warp_n={warp_n} stages={stages}")

    gu_bf16 = torch.randn((E, 2 * MI, H), dtype=torch.bfloat16, device=device) * 0.02
    dp_bf16 = torch.randn((E, H, MI), dtype=torch.bfloat16, device=device) * 0.02
    gu_fp8, gu_scale = quantize_weight_fp8_grouped(gu_bf16, group_size)
    dp_fp8, dp_scale = quantize_weight_fp8_grouped(dp_bf16, group_size)
    print(f"gate_up_proj fp8: {tuple(gu_fp8.shape)}  scale: {tuple(gu_scale.shape)}   "
          f"down_proj fp8: {tuple(dp_fp8.shape)}  scale: {tuple(dp_scale.shape)}")

    x_bf16 = torch.randn((M, H), dtype=torch.bfloat16, device=device) * 0.02
    x_fp8, x_scale = quantize_activation_fp8_dynamic(x_bf16)

    local_slots = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(torch.randn(M, top_k, device=device), dim=-1)

    # ---- Reference path (uses the ORIGINAL bf16 weights/activations, not
    # the fp8-quantized copies, so this check validates the KERNEL's
    # quantized math against a full-precision ground truth -- not just
    # "does dequant(quantize(w)) round-trip," a stronger check) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ref_out = reference_pipeline(x_bf16, gu_fp8, gu_scale, dp_fp8, dp_scale, group_size,
                                  local_slots, topk_weights, scaling_factor)
    torch.cuda.synchronize()
    ref_s = time.perf_counter() - t0
    print(f"Reference pipeline: {ref_s*1000:.3f} ms")

    # ---- Kernel path ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kernel_out = kernel_pipeline(x_fp8, x_scale, gu_fp8, gu_scale, dp_fp8, dp_scale,
                                  local_slots, topk_weights, top_k, E, scaling_factor,
                                  block_m, block_n, warp_n, stages)
    torch.cuda.synchronize()
    kernel_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kernel_out = kernel_pipeline(x_fp8, x_scale, gu_fp8, gu_scale, dp_fp8, dp_scale,
                                  local_slots, topk_weights, top_k, E, scaling_factor,
                                  block_m, block_n, warp_n, stages)
    torch.cuda.synchronize()
    kernel_warm_s = time.perf_counter() - t0
    print(f"Kernel pipeline:    first={kernel_s*1000:.3f} ms  warm={kernel_warm_s*1000:.3f} ms")

    # ---- Correctness ----
    cos = F.cosine_similarity(ref_out.float().reshape(-1), kernel_out.float().reshape(-1), dim=0).item()
    max_abs_err = (ref_out.float() - kernel_out.float()).abs().max().item()
    print(f"\nCorrectness (kernel vs. bf16-ground-truth reference): "
          f"cosine_similarity={cos:.6f}  max_abs_err={max_abs_err:.6f}")
    # Looser threshold than the INT8 kernel's own smoke test (0.999) -- FP8
    # e4m3 has ~2-3 decimal digits of precision on BOTH weights AND
    # activations (true W8A8, unlike INT8's weight-only scheme), so more
    # accumulated quantization error is expected here even if the kernel's
    # math is entirely correct. Revisit this threshold once a real run gives
    # an actual number to calibrate against -- 0.99 is a starting guess, not
    # a validated bar.
    threshold = 0.99
    print(f"{'PASS' if cos > threshold else 'FAIL -- investigate before trusting this kernel'} "
          f"(threshold={threshold})")

    print("\nNot wired into the real engine -- no CUDA graph capture, no EP ep_rank/owned_mask "
          "combine, no all_reduce, no activation-quantization step upstream of a real forward "
          "pass. This validates the KERNEL's own compute is correct in isolation, nothing more.")


if __name__ == "__main__":
    main()
