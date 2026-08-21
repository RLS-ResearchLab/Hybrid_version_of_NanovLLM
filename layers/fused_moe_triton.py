"""Fused MoE Triton kernel -- adapted from layers/fused_moe_triton_raw.py
(vLLM's real production kernel, copied in 2026-07-27, never wired up -- see
that file's own docstring). This file strips the vLLM-package dependency:
the original's module-level `import vllm.envs`, `from vllm import
_custom_ops`, `from vllm.logger import init_logger`, `from vllm.platforms
import current_platform` execute just by importing the file, whether or not
the vLLM-dependent functions are ever called -- pulling in real vLLM as a
dependency for that alone is backwards for a project whose whole point is
being a lightweight, from-scratch alternative to vLLM.

Only fused_moe_kernel (the actual @triton.jit kernel) and
invoke_fused_moe_kernel (its launcher) are carried over. Both are already
vLLM-op-free for the use_int8_w8a16 path this project needs -- the ONLY
vLLM op invoke_fused_moe_kernel calls (ops.scaled_fp8_quant) is gated behind
`if use_fp8_w8a8:`, which this project never sets (weight-only INT8, no FP8
-- A6000 has no FP8 tensor cores anyway, see moe_quantization_memo.md).
Everything else vLLM's original module provided (get_moe_configs, the
top-level fused_moe() orchestration, topk_softmax/silu_and_mul fused ops) is
NOT carried over -- this project has its own topk/softmax/silu-mul logic
already in models/qwen3_5.py's Qwen35MoE.forward(), and its own EP-sharded,
quantized Experts data layout that needs custom orchestration around this
kernel, not vLLM's.

moe_align_block_size (the other real dependency, a compiled CUDA extension
in vLLM) has its own pure-PyTorch replacement in
layers/moe_align_block_size.py, CPU-tested separately
(layers/test_moe_align_block_size.py) -- not needed here since this file
only carries the kernel + launcher, not the alignment step.

NOT yet validated on GPU as of this writing. Kernel logic is vLLM's own,
battle-tested code, unmodified -- risk here is in the adaptation (stripped
imports, calling convention) and in this project's own integration around
it, not in the GEMM math itself.
"""
from typing import Any, Dict, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bse,
        stride_bsg,
        stride_bsn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        QUANT_GROUP_SIZE: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.

    use_int8_w8a16 GROUPED-SCALE CHANGE (2026-08-21): vLLM's original here
    applied b_scale ONCE, after the full K-reduction loop -- correct only for
    one scale per (expert, output_channel) covering the WHOLE row. This
    project's actual quantization (tests/moe_int8_quantize.py) is grouped
    along K (group_size=128, one scale per (expert, output_channel, group)),
    which is finer-grained and is what the already-validated accuracy numbers
    (reconstruction error, GSM8K non-regression) were measured against --
    switching to whole-row scale would be a real accuracy change requiring
    re-validation, not just a kernel-integration detail. Instead, b_scale is
    now looked up and applied ONCE PER K-ITERATION inside the loop, using
    whichever quantization group that iteration's BLOCK_SIZE_K-wide chunk
    falls in. This is exact, not an approximation, ONLY IF QUANT_GROUP_SIZE
    is a multiple of BLOCK_SIZE_K -- enforced by an assert in
    invoke_fused_moe_kernel below, not just assumed here. If BLOCK_SIZE_K
    didn't divide QUANT_GROUP_SIZE evenly, a single loaded chunk could span
    two different scale groups and there would be no single correct scale to
    apply to it -- the assert exists specifically to make that config
    impossible to reach silently.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    # use_int8_w8a16's b_scale is now looked up PER K-ITERATION, inside the
    # loop below (grouped scale) -- not here, unlike the original.

    if use_fp8_w8a8:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            # Grouped scale: this k-iteration's BLOCK_SIZE_K-wide chunk
            # belongs to exactly one quantization group (QUANT_GROUP_SIZE is
            # asserted to be a multiple of BLOCK_SIZE_K by the launcher), so
            # one scale lookup per iteration is exact. Compute this
            # iteration's partial dot product UNSCALED, then scale it before
            # folding into the running fp32 accumulator -- can't use tl.dot's
            # own acc= accumulation here since each iteration needs a
            # DIFFERENT scale applied before summing, not one scale at the end.
            group_idx = (k * BLOCK_SIZE_K) // QUANT_GROUP_SIZE
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            group_idx * stride_bsg + offs_bn * stride_bsn)
            b_scale = tl.load(b_scale_ptrs)  # (BLOCK_SIZE_N,)
            partial = tl.dot(a, b.to(compute_type))
            accumulator += partial * b_scale[None, :]
        elif use_fp8_w8a8:
            accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        # Scale already folded in per-K-iteration above -- do NOT multiply
        # by b_scale again here, that would double-apply it.
        accumulator = accumulator.to(compute_type)
    elif use_fp8_w8a8:
        accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int,
                            config: Dict[str, Any], compute_type: tl.dtype,
                            use_fp8_w8a8: bool, use_int8_w8a16: bool,
                            quant_group_size: Optional[int] = None) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        raise NotImplementedError(
            "FP8 path stripped out of this adapted copy -- this project is "
            "weight-only INT8 (use_int8_w8a16), and A6000 has no FP8 tensor "
            "cores to run this path on anyway. Use layers/fused_moe_triton_raw.py "
            "(with real vLLM installed) if FP8 is ever actually needed."
        )
    elif use_int8_w8a16:
        assert B_scale is not None
        assert B_scale.ndim == 3, (
            f"expected grouped scale shape (E, out_features, num_groups), got "
            f"{tuple(B_scale.shape)} -- this adapted kernel applies scale "
            f"per-K-group, not once per row; a 2D (E, out_features) scale "
            f"means you want the ORIGINAL whole-row-scale behavior, which "
            f"this file no longer implements (see the docstring on why)."
        )
        assert quant_group_size is not None
        block_size_k = config["BLOCK_SIZE_K"]
        assert quant_group_size % block_size_k == 0, (
            f"QUANT_GROUP_SIZE={quant_group_size} is not a multiple of "
            f"this config's BLOCK_SIZE_K={block_size_k} -- a single loaded "
            f"K-chunk would span more than one scale group, and there would "
            f"be no single correct scale to apply to it. Pick a BLOCK_SIZE_K "
            f"that divides quant_group_size evenly (e.g. 32/64/128 for "
            f"group_size=128), not silently produce wrong numbers."
        )
    else:
        assert A_scale is None
        assert B_scale is None

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )

    fused_moe_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0) if B_scale is not None and use_int8_w8a16 else 0,
        B_scale.stride(2) if B_scale is not None and use_int8_w8a16 else 0,
        B_scale.stride(1) if B_scale is not None and use_int8_w8a16 else 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        QUANT_GROUP_SIZE=quant_group_size if quant_group_size is not None else 0,
        **config,
    )
