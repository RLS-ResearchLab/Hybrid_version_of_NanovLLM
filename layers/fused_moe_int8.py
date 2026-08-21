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
import torch
import torch.nn.functional as F

from nanovllm.layers.moe_align_block_size import moe_align_block_size
from nanovllm.layers.fused_moe_triton import invoke_fused_moe_kernel

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

    # ---- GEMM 1: gate_up_proj ----
    sorted_ids1, expert_ids1, ntpp1 = moe_align_block_size(
        local_slots, config["BLOCK_SIZE_M"], local_num_experts
    )
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
    gw, uw = gate_up_out.chunk(2, dim=2)   # each (N, TK, MI)
    h = F.silu(gw) * uw                     # (N, TK, MI)

    # ---- GEMM 2: down_proj -- each (token, k) pair is its own "virtual
    # token" with top_k=1, since h's rows are already expert-specific (see
    # module docstring) ----
    h_flat = h.reshape(N * top_k, MI)
    local_slots_flat = local_slots.reshape(N * top_k, 1)
    sorted_ids2, expert_ids2, ntpp2 = moe_align_block_size(
        local_slots_flat, config["BLOCK_SIZE_M"], local_num_experts
    )
    down_out = torch.empty((N * top_k, 1, H), dtype=dtype, device=device)
    dummy_w2 = torch.ones((N * top_k, 1), dtype=torch.float32, device=device)
    invoke_fused_moe_kernel(
        h_flat, down_proj_int8, down_out,
        None, down_proj_scale,
        dummy_w2, local_slots_flat,
        sorted_ids2, expert_ids2, ntpp2,
        False, 1, config, compute_type,
        use_fp8_w8a8=False, use_int8_w8a16=True,
        quant_group_size=group_size,
    )
    return down_out.reshape(N, top_k, H)
