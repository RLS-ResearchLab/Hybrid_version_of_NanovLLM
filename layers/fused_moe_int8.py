"""High-level entry point wrapping the fused Triton MoE kernel
(fused_moe_triton.py + moe_align_block_size.py) into the exact shape
models/qwen3_5.py's _forward_gathered_ep needs: given the ORIGINAL small
activation tensor and the FULL local-expert weight tensors (no pre-gather),
returns the post-combine-ready (N, TK, H) expert output tensor -- a drop-in
replacement for that method's current gather-then-dequantize-then-einsum
sequence, verified correct and faster in isolation first
(layers/smoke_test_full_moe_pipeline.py, cosine=0.999988, 4.71x on a single
GPU, 2026-08-21) before being wired in here.

NOT YET validated: CUDA graph capture compatibility, real EP
ep_rank/owned_mask combine, real-checkpoint GSM8K correctness, or an actual
engine throughput number. This module is deliberately flag-gated in
models/qwen3_5.py (not a hard replacement) specifically because those things
are unverified -- see the flag's own docstring there for exactly what's
still open before this should be trusted as the default.
"""
import os
import time

import torch
import torch.nn.functional as F

from nanovllm.layers.moe_align_block_size import moe_align_block_size
from nanovllm.layers.fused_moe_triton import invoke_fused_moe_kernel

# Diagnostic-only timing, added 2026-08-21 after the first real-engine test
# (eager, concurrency=16) measured 29.3 tok/s -- SLOWER than the 37.1 tok/s
# baseline this was supposed to beat, contradicting the isolated
# microbenchmark's 4.71x. Accumulates wall time per stage across calls and
# prints a running breakdown every _PRINT_EVERY calls, so we can see WHICH
# stage (alignment vs. kernel launch vs. silu) actually dominates at real
# scale (40 layers x up to 1024 decode steps) instead of guessing from an
# isolated warm-loop benchmark that never paid this call volume's launch
# overhead. Env-var gated, off by default -- adds real sync overhead of its
# own, not meant to stay on once the bottleneck is identified.
_PROFILE = os.environ.get("NANOVLLM_PROFILE_FUSED_MOE", "0") == "1"
_PRINT_EVERY = 200
_timing_acc = {"align1": 0.0, "kernel1": 0.0, "silu": 0.0, "kernel2": 0.0, "calls": 0}


def _tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def _maybe_print_timing():
    if _timing_acc["calls"] > 0 and _timing_acc["calls"] % _PRINT_EVERY == 0:
        c = _timing_acc["calls"]
        total = sum(v for k, v in _timing_acc.items() if k != "calls")
        print(f"[FUSED MOE PROFILE] after {c} calls, avg per call (ms), "
              f"total_avg={total/c*1000:.3f}ms:")
        for stage in ("align1", "kernel1", "silu", "kernel2"):
            v = _timing_acc[stage]
            print(f"    {stage:8s}: {v/c*1000:7.3f} ms/call  "
                  f"({v/total*100:5.1f}% of this function's own time)")

# Single, fixed, conservative config -- validated in the smoke tests, not yet
# autotuned per shape/concurrency. layers/fused_moe_triton_raw.py's own
# autotuner script (adapted from a prior project) is the template for doing
# that properly later; using one safe config everywhere for now trades some
# performance for not adding an untested autotuning dependency to the first
# real integration attempt.
_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,   # must divide the quantization group_size (128) evenly
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}


def _tl_compute_type(dtype: torch.dtype):
    import triton.language as tl
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    raise ValueError(f"fused_moe_int8 only supports bf16/fp16 compute, got {dtype}")


def fused_moe_int8_forward(
    x: torch.Tensor,
    gate_up_proj_int8: torch.Tensor,
    gate_up_proj_scale: torch.Tensor,
    down_proj_int8: torch.Tensor,
    down_proj_scale: torch.Tensor,
    group_size: int,
    local_slots: torch.Tensor,
    top_k: int,
    local_num_experts: int,
    config: dict | None = None,
) -> torch.Tensor:
    """Replaces _forward_gathered_ep's:
        gu_i8 = self.experts.gate_up_proj_int8[local_slots]
        ... dequantize_weight_int8_grouped(...) x2 ...
        gw, uw = gate_up.chunk(2, dim=2)
        h_gate = einsum(...); h_up = einsum(...); h = silu(h_gate) * h_up
        out_e = einsum(...)
    with the same math, computed by the fused kernel instead of
    gather+dequantize+einsum. Caller still does the weighted
    combine/all_reduce afterward, unchanged -- this function's contract ends
    at producing the same-shaped (N, TK, H) out_e.

    Args:
        x: (N, H) bf16/fp16 activations -- the ORIGINAL, ungathered input.
        gate_up_proj_int8/scale, down_proj_int8/scale: the FULL local-expert
            tensors (shape (local_num_experts, ...)), NOT pre-gathered by
            local_slots -- the kernel gathers implicitly.
        local_slots: (N, top_k) int -- which local expert each (token, k)
            pair selects. Same tensor _forward_gathered_ep already computes.
        local_num_experts: self.experts.gate_up_proj_int8.shape[0].

    Returns:
        (N, top_k, H) -- same shape/meaning as the current out_e.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    N, H = x.shape
    two_mi = gate_up_proj_int8.shape[1]
    MI = two_mi // 2
    dtype = x.dtype
    device = x.device
    compute_type = _tl_compute_type(dtype)

    block_k = config["BLOCK_SIZE_K"]
    assert group_size % block_k == 0, (
        f"group_size={group_size} must be a multiple of BLOCK_SIZE_K={block_k} "
        f"for the grouped-scale kernel to apply scale correctly -- see "
        f"fused_moe_triton.py's docstring on why."
    )

    if _PROFILE:
        t0 = _tick()

    # ---- GEMM 1: gate_up_proj ----
    sorted_ids1, expert_ids1, ntpp1 = moe_align_block_size(
        local_slots, config["BLOCK_SIZE_M"], local_num_experts
    )
    if _PROFILE:
        t1 = _tick()
        _timing_acc["align1"] += t1 - t0
        t0 = t1
    gate_up_out = torch.empty((N, top_k, two_mi), dtype=dtype, device=device)
    # MUL_ROUTED_WEIGHT=False -- the real per-(token,k) softmax weight is
    # applied later in _forward_gathered_ep's own combine step, in fp32,
    # exactly as it is today; this dummy tensor's values are never read.
    dummy_w1 = torch.ones((N, top_k), dtype=torch.float32, device=device)
    invoke_fused_moe_kernel(
        x, gate_up_proj_int8, gate_up_out,
        None, gate_up_proj_scale,
        dummy_w1, local_slots,
        sorted_ids1, expert_ids1, ntpp1,
        False, top_k, config, compute_type,
        use_fp8_w8a8=False, use_int8_w8a16=True,
        quant_group_size=group_size,
    )
    if _PROFILE:
        t1 = _tick()
        _timing_acc["kernel1"] += t1 - t0
        t0 = t1
    gw, uw = gate_up_out.chunk(2, dim=2)   # each (N, TK, MI)
    h = F.silu(gw) * uw                     # (N, TK, MI)
    if _PROFILE:
        t1 = _tick()
        _timing_acc["silu"] += t1 - t0
        t0 = t1

    # ---- GEMM 2: down_proj -- each (token, k) pair is its own "virtual
    # token" with top_k=1, since h's rows are already expert-specific (see
    # module docstring). Reuses GEMM 1's sorted_ids1/expert_ids1/ntpp1
    # as-is, does NOT recompute them: moe_align_block_size only depends on
    # topk_ids.reshape(-1)'s VALUES, and local_slots.reshape(-1) is
    # bit-identical to local_slots.reshape(N*top_k, 1).reshape(-1) -- same
    # values, same block_size, same local_num_experts, so a second call
    # here was a full argsort+scatter_add+cumsum+searchsorted spent
    # reproducing a result already sitting in sorted_ids1/expert_ids1/ntpp1.
    # Same reasoning covers local_slots vs. the old local_slots_flat as the
    # topk_ids arg below -- invoke_fused_moe_kernel only reads .numel() off
    # it (both are N*top_k), never the tensor's values or declared shape. ----
    h_flat = h.reshape(N * top_k, MI)
    down_out = torch.empty((N * top_k, 1, H), dtype=dtype, device=device)
    dummy_w2 = torch.ones((N * top_k, 1), dtype=torch.float32, device=device)
    invoke_fused_moe_kernel(
        h_flat, down_proj_int8, down_out,
        None, down_proj_scale,
        dummy_w2, local_slots,
        sorted_ids1, expert_ids1, ntpp1,
        False, 1, config, compute_type,
        use_fp8_w8a8=False, use_int8_w8a16=True,
        quant_group_size=group_size,
    )
    if _PROFILE:
        t1 = _tick()
        _timing_acc["kernel2"] += t1 - t0
        _timing_acc["calls"] += 1
        _maybe_print_timing()
    return down_out.reshape(N, top_k, H)
