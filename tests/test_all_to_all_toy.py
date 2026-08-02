"""Q3: standalone, isolated correctness test for torch.distributed.all_to_all_single.

Purely mechanical -- moves small tagged rows between 2 ranks and verifies they
come back in exactly the right place. No MoE code, no models/qwen3_5.py
involvement. This is the primitive Q5's dispatch/combine will be built on.

Two-hop pattern per direction (dispatch and combine each do this once):
  1. Exchange per-rank send counts (fixed-size 1-element-per-rank
     all_to_all_single) so each rank learns how many rows it will *receive*
     from each peer -- required because dropless routing makes counts
     data-dependent, not known in advance.
  2. The real all_to_all_single, using those exchanged counts as
     output_split_sizes / the locally-known counts as input_split_sizes.

Row payload is (tag, orig_idx, orig_rank):
  - tag: globally unique per token (0-7 across both ranks) so any positional
    mix-up changes the value at that position, not just "some value at some
    position" -- deliberately not a repeated/symmetric pattern.
  - orig_idx: this token's position in its OWN rank's starting buffer (0-3).
  - orig_rank: which rank this token started on.

Dispatch routes each row by tag % world_size (deterministic, hand-computable).
Combine routes each row back to its orig_rank. After combine, orig_idx is used
to scatter each token back into its exact original position -- the final
assertion is that this reconstructed buffer equals the original tag buffer
*exactly*, not just "contains the same values."

Usage: python tests/test_all_to_all_toy.py
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def exchange_counts(send_counts: torch.Tensor, world_size: int) -> torch.Tensor:
    """send_counts[i] = rows this rank sends to rank i.
    Returns recv_counts[i] = rows this rank will receive from rank i."""
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(
        recv_counts, send_counts,
        output_split_sizes=[1] * world_size,
        input_split_sizes=[1] * world_size,
    )
    return recv_counts


def route(rows: torch.Tensor, dest_rank_per_row: torch.Tensor, world_size: int):
    """Stable-sort rows by destination rank. Returns (sorted_rows, send_counts)."""
    order = torch.argsort(dest_rank_per_row, stable=True)
    sorted_rows = rows[order]
    send_counts = torch.bincount(dest_rank_per_row, minlength=world_size).to(torch.int64)
    return sorted_rows, send_counts


def all_to_all_rows(send_rows: torch.Tensor, send_counts: torch.Tensor, world_size: int) -> torch.Tensor:
    recv_counts = exchange_counts(send_counts, world_size)
    output = torch.empty(int(recv_counts.sum()), send_rows.shape[1], dtype=send_rows.dtype)
    dist.all_to_all_single(
        output, send_rows,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
    )
    return output


# Hand-computed expected values (see conversation for the by-hand derivation).
EXPECTED_DISPATCH = {
    0: torch.tensor([[0, 0, 0], [2, 2, 0], [4, 0, 1], [6, 2, 1]], dtype=torch.int64),
    1: torch.tensor([[1, 1, 0], [3, 3, 0], [5, 1, 1], [7, 3, 1]], dtype=torch.int64),
}
EXPECTED_COMBINE = {
    0: torch.tensor([[0, 0, 0], [2, 2, 0], [1, 1, 0], [3, 3, 0]], dtype=torch.int64),
    1: torch.tensor([[4, 0, 1], [6, 2, 1], [5, 1, 1], [7, 3, 1]], dtype=torch.int64),
}


def worker(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    base = rank * 4
    tags = torch.tensor([base, base + 1, base + 2, base + 3], dtype=torch.int64)
    orig_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    orig_rank = torch.full((4,), rank, dtype=torch.int64)
    rows = torch.stack([tags, orig_idx, orig_rank], dim=1)  # (4, 3): [tag, orig_idx, orig_rank]

    print(f"[rank{rank}] initial rows (tag, orig_idx, orig_rank):\n{rows.tolist()}")

    # ---- DISPATCH: route by tag % world_size ----
    dest = tags % world_size
    sorted_rows, send_counts = route(rows, dest, world_size)
    print(f"[rank{rank}] dispatch send_counts={send_counts.tolist()} sorted_rows={sorted_rows.tolist()}")

    dispatched = all_to_all_rows(sorted_rows, send_counts, world_size)
    print(f"[rank{rank}] post-dispatch received rows: {dispatched.tolist()}")

    expected_dispatch = EXPECTED_DISPATCH[rank]
    assert torch.equal(dispatched, expected_dispatch), (
        f"[rank{rank}] DISPATCH MISMATCH: got {dispatched.tolist()} "
        f"expected {expected_dispatch.tolist()}"
    )
    print(f"[rank{rank}] DISPATCH OK -- matches hardcoded expected {expected_dispatch.tolist()}")

    # ---- COMBINE: route back to orig_rank ----
    combine_dest = dispatched[:, 2].contiguous()  # orig_rank column
    sorted_combine_rows, combine_send_counts = route(dispatched, combine_dest, world_size)
    print(f"[rank{rank}] combine send_counts={combine_send_counts.tolist()} "
          f"sorted_rows={sorted_combine_rows.tolist()}")

    combined = all_to_all_rows(sorted_combine_rows, combine_send_counts, world_size)
    print(f"[rank{rank}] post-combine received rows: {combined.tolist()}")

    expected_combine = EXPECTED_COMBINE[rank]
    assert torch.equal(combined, expected_combine), (
        f"[rank{rank}] COMBINE MISMATCH: got {combined.tolist()} "
        f"expected {expected_combine.tolist()}"
    )
    print(f"[rank{rank}] COMBINE OK -- matches hardcoded expected {expected_combine.tolist()}")

    # ---- Scatter back into original order using orig_idx ----
    reconstructed = torch.empty(4, dtype=torch.int64)
    reconstructed[combined[:, 1]] = combined[:, 0]
    original_tags = torch.tensor([base, base + 1, base + 2, base + 3], dtype=torch.int64)
    print(f"[rank{rank}] reconstructed (scattered by orig_idx): {reconstructed.tolist()}  "
          f"original: {original_tags.tolist()}")
    assert torch.equal(reconstructed, original_tags), (
        f"[rank{rank}] ROUND-TRIP MISMATCH: got {reconstructed.tolist()} "
        f"expected {original_tags.tolist()}"
    )
    print(f"[rank{rank}] ROUND-TRIP OK -- exact original order recovered")

    dist.destroy_process_group()


def main():
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("ALL RANKS PASSED")


if __name__ == "__main__":
    main()
