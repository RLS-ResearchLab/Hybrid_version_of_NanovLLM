"""Pure-PyTorch replacement for vllm._custom_ops.moe_align_block_size.

Originally written against layers/fused_moe_triton_raw.py, vLLM's real
fused-MoE Triton kernel copied in as a reference (2026-07-27, three weeks
before this project's quantization work); that file's own
moe_align_block_size() called ops.moe_align_block_size(...), a compiled CUDA
extension inside the actual vllm package -- pulling in real vLLM as a
dependency just for one supporting op is backwards for a project whose whole
point is being a lightweight, from-scratch alternative to vLLM. This file
reimplements just that one function in plain PyTorch, matching the exact
input/output contract the Triton kernel expects. The kernel itself was since
wired in as layers/fused_moe_triton.py; the raw reference copy was deleted
2026-08-26 (git history has it if needed).

CUDA-graph safety, the constraint that shaped every design choice here: every
output tensor is allocated at the SAME fixed worst-case size the original
vLLM wrapper already used (max_num_tokens_padded = numel + num_experts *
(block_size - 1), max_num_m_blocks = cdiv(max_num_tokens_padded, block_size))
-- never a data-dependent size. The one place that's easy to get wrong is
building expert_ids (which expert each padded BLOCK belongs to): a naive
`torch.repeat_interleave(experts, num_blocks_per_expert)` produces a
dynamically-sized tensor (the same class of problem that ruled out the
gather-deduplication idea earlier this session -- see moe_quantization_memo.md
if this file is read out of context) and would break graph capture. Used
`torch.searchsorted` against fixed-size (num_experts,) cumulative bounds
instead -- same output shape (max_num_m_blocks,) every call, regardless of
the actual routing.

VALIDATED on real GPU hardware (2026-08-21 A6000 session, updated
2026-08-26 -- this docstring previously said "not yet," stale after that
session landed): this is the alignment step inside the fused Triton MoE
kernel path (fused_moe_int8.py's fused_moe_int8_forward,
NANOVLLM_USE_FUSED_MOE_KERNEL=1), which reached cosine=0.999988 against a
trusted reference, real-checkpoint GSM8K non-regression (8/8, then 40/40
under CUDA graphs), and the production 172.7->204.1 tok/s numbers in
README.md and moe_quantization_memo.md. Also still covered by the
CPU-only unit test below (test_moe_align_block_size.py), unaffected by
this update.
"""
import torch


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for vllm._custom_ops.moe_align_block_size's Python
    wrapper (layers/fused_moe_triton_raw.py's own moe_align_block_size()).
    Same signature, same return contract:

        sorted_ids: (max_num_tokens_padded,) int32 -- flat indices into
            topk_ids.view(-1), grouped by assigned expert, block-padded with
            the sentinel value topk_ids.numel() (out of range on purpose --
            the kernel's token_mask = offs_token < num_valid_tokens relies on
            this to skip padding slots).
        expert_ids: (max_num_m_blocks,) int32 -- which expert each
            BLOCK_SIZE-wide block of sorted_ids belongs to. Only the first
            cdiv(num_tokens_post_padded, block_size) entries are meaningful;
            the kernel's own `if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return` guard means entries beyond that are never read, same as
            the original's torch.empty() (uninitialized) tail.
        num_tokens_post_padded: (1,) int32 -- real (data-dependent) total
            after padding. A 0-d/1-element tensor, computed without a host
            sync (.item()) -- stays a device tensor throughout, safe to read
            inside a captured graph the same way the kernel already reads it
            via tl.load(num_tokens_post_padded_ptr).
    """
    device = topk_ids.device
    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    numel = flat_ids.numel()

    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    # Per-expert token counts -- fixed size (num_experts,), regardless of
    # how tokens actually route.
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids))

    num_blocks_per_expert = (counts + block_size - 1) // block_size
    padded_counts = num_blocks_per_expert * block_size

    # Where each expert's PADDED segment starts/ends in the output, and
    # where its UNPADDED segment starts in the (expert-sorted) token order.
    # All (num_experts,) -- fixed size.
    padded_offsets = torch.zeros(num_experts, dtype=torch.int64, device=device)
    padded_offsets[1:] = torch.cumsum(padded_counts, 0)[:-1]
    unpadded_offsets = torch.zeros(num_experts, dtype=torch.int64, device=device)
    unpadded_offsets[1:] = torch.cumsum(counts, 0)[:-1]
    block_end = torch.cumsum(num_blocks_per_expert, 0)  # (num_experts,)

    # Group tokens by assigned expert (stable, so ties keep original token
    # order -- matches the original's "flatten then sort by expert" example).
    sort_order = torch.argsort(flat_ids, stable=True)  # (numel,) -- fixed size
    sorted_expert_of_token = flat_ids[sort_order]

    # Destination slot for each sorted token: this expert's padded-segment
    # start, plus this token's rank within its own expert's unpadded run.
    rank_within_group = torch.arange(numel, device=device) - unpadded_offsets[sorted_expert_of_token]
    dest = padded_offsets[sorted_expert_of_token] + rank_within_group

    sorted_ids = torch.full(
        (max_num_tokens_padded,), numel, dtype=torch.int32, device=device
    )
    sorted_ids[dest] = sort_order.to(torch.int32)

    # Which expert owns each fixed-size block index. searchsorted over
    # block_end (monotonic, (num_experts,)) against a fixed (max_num_m_blocks,)
    # arange -- output shape is always max_num_m_blocks, never data-dependent,
    # unlike a repeat_interleave-based construction would be.
    block_idx = torch.arange(max_num_m_blocks, device=device)
    expert_ids = torch.searchsorted(block_end, block_idx, right=True)
    expert_ids = expert_ids.clamp(max=num_experts - 1).to(torch.int32)

    num_tokens_post_padded = padded_counts.sum().to(torch.int32).reshape(1)

    return sorted_ids, expert_ids, num_tokens_post_padded
