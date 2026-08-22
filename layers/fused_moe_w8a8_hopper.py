"""JIT-compiles and loads the Hopper wgmma/TMA fused W8A8 MoE kernel
(moe_w8a8.cu + moe_w8a8_binding.cpp) as a PyTorch CUDA extension, and exposes
a clean Python entry point.

UNVALIDATED. This kernel has never been compiled, run, or checked against a
reference -- see moe_w8a8.h for the full contract and moe_w8a8_binding.cpp
for what's enforced at the Python/C++ boundary. Do not wire this into the
real model (models/qwen3_5.py) until it has been through the same
isolated-correctness-then-GSM8K validation chain every other kernel in this
project went through first -- see H200_test_day_checklist.md's P1 item on
this kernel for what that chain looks like.

Requires: an H200 (or other Hopper, sm_90a) GPU, CUDA 12.3+ (the cuda::ptx
namespace calls in moe_w8a8.cu need a fairly recent toolkit -- confirm the
exact minimum on first build attempt, not assumed here), and a PyTorch build
with torch.float8_e4m3fn support (roughly >=2.1). Cannot be JIT-compiled or
even imported on this project's Windows dev machine -- no nvcc, no CUDA
toolkit there at all, so none of this has been exercised.
"""
import os

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="moe_w8a8_hopper_ext",
            sources=[
                os.path.join(_THIS_DIR, "moe_w8a8.cu"),
                os.path.join(_THIS_DIR, "moe_w8a8_binding.cpp"),
            ],
            extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90a"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=True,
        )
    return _ext


def fused_moe_w8a8_hopper_forward(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    num_experts: int,
    block_m: int = 32,
    block_n: int = 32,
    warp_n: int = 8,
    stages: int = 2,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Runs the fused gate_up -> SiLU -> dynamic-FP8-requant -> down_proj
    Hopper kernel. Allocates and zeroes `out` itself -- the kernel
    accumulates into it via reduce-add rather than overwriting (see
    moe_w8a8.h), so a fresh, zeroed tensor here removes that footgun for
    every caller rather than documenting it and hoping.

    Args:
        x: (M, K) fp8 (torch.float8_e4m3fn) activations, already quantized --
            this function does not quantize its own input. That quantization
            step (dynamic per-token FP8 activation scaling) doesn't exist
            anywhere in this project yet; building it is separate, real work,
            not implied by this wrapper existing.
        w: (num_experts, N, K) fp8 gate_up_proj. w2: (num_experts, K, N/2)
            fp8 down_proj. N = 2 * moe_intermediate_size (gate+up combined).
            Both must already be FP8-quantized and laid out to match what
            moe_w8a8.cu's TMA setup expects -- see moe_w8a8.h's layout note.
        sorted_token_ids/expert_ids/num_tokens_post_padded: same dtype/shape
            contract as layers/moe_align_block_size.py already produces for
            the Ampere Triton kernel -- intended to be reusable as-is, not
            yet confirmed.
        block_m/block_n/warp_n/stages: default to one conservative, untested
            starting config -- (block_n, warp_n)=(32, 8) is one of only two
            pairs the kernel supports at all (the other is (64, 4));
            block_m=32 and stages=2 mirror the same "pick one safe config,
            autotune later, only after correctness is established" posture
            this project used for the Ampere Triton kernel. Not chosen from
            any measurement on this kernel -- there isn't one yet.

    Returns:
        (M, K) bf16 tensor -- the combined MoE output for this expert group,
        matching down_proj's output shape (residual-stream sized).
    """
    ext = _get_ext()
    M, K = x.shape
    N = w.shape[1]
    sorted_num = sorted_token_ids.numel()

    out = torch.zeros((M, K), dtype=torch.bfloat16, device=x.device)

    ext.moe_w8a8_forward(
        x, x_scale, w, w_scale, w2, w2_scale, out,
        sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
        top_k, M, K, N, num_experts, sorted_num,
        block_m, block_n, warp_n, stages, scaling_factor,
    )
    return out
