"""CPU-only self-check for layers/moe_align_block_size.py -- the pure-PyTorch
replacement for vllm._custom_ops.moe_align_block_size, written so
layers/fused_moe_triton_raw.py's kernel can be reused without depending on
real vLLM. No GPU, no Triton needed here: this function is plain PyTorch, and
what's being checked is the sorted_ids/expert_ids/num_tokens_post_padded
CONTRACT the kernel relies on, not the kernel itself.

Run: python layers/test_moe_align_block_size.py
"""
import torch

from moe_align_block_size import moe_align_block_size


def check_invariants(topk_ids, block_size, num_experts, label):
    flat_ids = topk_ids.reshape(-1)
    numel = flat_ids.numel()

    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, num_experts
    )

    n_post = int(num_tokens_post_padded.item())
    assert n_post % block_size == 0, (
        f"[{label}] num_tokens_post_padded={n_post} not a multiple of "
        f"block_size={block_size}"
    )
    assert n_post >= numel, f"[{label}] post-padding count shrank below numel"

    # Every real token index must appear EXACTLY once, and only within the
    # first n_post slots (the kernel never reads past num_tokens_post_padded).
    real_region = sorted_ids[:n_post]
    is_padding = real_region == numel
    real_tokens = real_region[~is_padding]
    assert real_tokens.numel() == numel, (
        f"[{label}] expected {numel} real token slots in the post-padding "
        f"region, found {real_tokens.numel()}"
    )
    seen = torch.zeros(numel, dtype=torch.bool)
    seen[real_tokens.long()] = True
    assert seen.all(), f"[{label}] some real token indices never appear"

    # Slots beyond n_post must all be the padding sentinel (nothing "leaks"
    # past the boundary the kernel actually trusts).
    tail = sorted_ids[n_post:]
    assert (tail == numel).all(), (
        f"[{label}] found non-sentinel values past num_tokens_post_padded"
    )

    # For every real (non-padding) slot, the block it's in must be assigned
    # to the SAME expert that token was actually routed to -- this is the
    # actual correctness property the Triton kernel depends on: it reads
    # expert_ids[pid_m] once per block and uses that single expert's weights
    # for every token loaded from that block.
    for slot in range(n_post):
        tok = int(sorted_ids[slot].item())
        if tok == numel:
            continue  # padding within a block that's otherwise real
        block = slot // block_size
        assigned_expert = int(expert_ids[block].item())
        real_expert = int(flat_ids[tok].item())
        assert assigned_expert == real_expert, (
            f"[{label}] slot {slot} (token {tok}, real expert {real_expert}) "
            f"landed in block {block} assigned to expert {assigned_expert}"
        )

    print(f"[{label}] PASS -- numel={numel} block_size={block_size} "
          f"num_experts={num_experts} num_tokens_post_padded={n_post}")


def test_small_deterministic():
    # 4 tokens, top_k=3, 5 experts (ids 0..4) -- small enough to reason about
    # by hand if this ever needs debugging.
    topk_ids = torch.tensor([
        [2, 3, 4],
        [1, 2, 4],
        [1, 3, 4],
        [1, 2, 3],
    ], dtype=torch.int64)
    check_invariants(topk_ids, block_size=4, num_experts=5, label="small_deterministic")


def test_random_realistic_scale():
    # Matches this project's real EP-sharded scale: local_num_experts=128 at
    # tp=2 (256 total / 2), top_k=8, N up to 32 (a concurrency level actually
    # tested this session).
    torch.manual_seed(0)
    for N in [1, 8, 16, 32]:
        for block_size in [16, 32, 64]:
            topk_ids = torch.randint(0, 128, (N, 8), dtype=torch.int64)
            check_invariants(
                topk_ids, block_size=block_size, num_experts=128,
                label=f"random_N{N}_bs{block_size}",
            )


def test_single_expert_degenerate():
    # Every token routes to the same expert -- the other 127 experts get
    # zero tokens each. Checks the num_blocks_per_expert=0 path doesn't
    # corrupt padded_offsets/block_end for later experts.
    topk_ids = torch.zeros((16, 8), dtype=torch.int64)  # all route to expert 0
    check_invariants(topk_ids, block_size=32, num_experts=128, label="single_expert")


def test_no_tokens_for_last_expert():
    # Token ids deliberately avoid the LAST expert -- checks block_end's
    # final cumsum entry and the clamp in expert_ids construction.
    torch.manual_seed(1)
    topk_ids = torch.randint(0, 127, (16, 8), dtype=torch.int64)  # never picks expert 127
    check_invariants(topk_ids, block_size=16, num_experts=128, label="last_expert_unused")


if __name__ == "__main__":
    test_small_deterministic()
    test_random_realistic_scale()
    test_single_expert_degenerate()
    test_no_tokens_for_last_expert()
    print("\nALL CHECKS PASSED (CPU-only, pure PyTorch -- Triton kernel side still unverified)")
