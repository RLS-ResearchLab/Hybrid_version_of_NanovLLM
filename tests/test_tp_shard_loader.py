# tests/test_tp_shard_loader.py
"""Issue 3 follow-up: verify weight_loader's shard-selection math (which
slice of the on-disk full tensor this rank keeps, given tp_size/tp_rank) at
tensor_parallel_size > 1.

test_loader_shard_merge.py validated packed_modules_mapping + load_model()
end-to-end, but only at tensor_parallel_size=1, where every weight_loader's
tp_size/tp_rank slicing collapses to a no-op (narrow(dim, 0, full_size) or
chunk(1, dim)[0] both just return the whole tensor) -- an off-by-one, wrong
axis, or wrong rank-ordering bug in the real slicing math is invisible at
TP=1 and would first surface at TP=2 (or TP=4), which is exactly what
tonight's real checkpoint run depends on.

weight_loader methods do no collective communication -- each rank
independently narrows/chunks the SAME fully-loaded source tensor -- so this
is pure CPU tensor-slicing logic, verifiable without real multi-GPU/NCCL
hardware. Calls each weight_loader as an unbound method against a minimal
duck-typed stand-in for `self`/`param` (just the attributes each method
actually reads: tp_dim, tp_size, tp_rank, output_sizes, .data) rather than
constructing real nn.Module instances, which would need dist.get_rank()/
get_world_size() faked via a real or monkeypatched process group for no
added coverage of the math itself.

For each class: build the "full" unsharded tensor, run weight_loader once
per rank (tp_size=2), then verify that concatenating every rank's resulting
slice back together exactly reconstructs the original full tensor -- and
that rank 0's slice matches a directly-computed chunk/narrow independently
of weight_loader, as a cross-check against the test's own reconstruction
logic being wrong in the same way the code might be.

Usage:
    python tests/test_tp_shard_loader.py
"""
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm":
    import types
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

from nanovllm.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding


def test_merged_column_parallel_tp2():
    print("=" * 70)
    print("MergedColumnParallelLinear.weight_loader shard-selection at tp_size=2")
    print("=" * 70)

    hidden_size, intermediate_size, tp_size = 16, 24, 2
    output_sizes = [intermediate_size, intermediate_size]

    torch.manual_seed(0)
    gate_full = torch.randn(intermediate_size, hidden_size)
    up_full = torch.randn(intermediate_size, hidden_size)

    per_rank_total = sum(output_sizes) // tp_size
    per_rank_shard = intermediate_size // tp_size
    rank_params = []
    for tp_rank in range(tp_size):
        p = SimpleNamespace(
            data=torch.zeros(per_rank_total, hidden_size),
            tp_dim=0, tp_size=tp_size, tp_rank=tp_rank, output_sizes=output_sizes,
        )
        MergedColumnParallelLinear.weight_loader(p, p, gate_full, 0)
        MergedColumnParallelLinear.weight_loader(p, p, up_full, 1)
        rank_params.append(p)

    reconstructed_gate = torch.cat([rp.data[:per_rank_shard] for rp in rank_params], dim=0)
    reconstructed_up = torch.cat([rp.data[per_rank_shard:] for rp in rank_params], dim=0)

    gate_ok = torch.equal(reconstructed_gate, gate_full)
    up_ok = torch.equal(reconstructed_up, up_full)
    print(f"  reconstructed gate (across ranks) == full gate: {gate_ok}")
    print(f"  reconstructed up   (across ranks) == full up:   {up_ok}")

    expected_rank0_gate = gate_full.chunk(tp_size, dim=0)[0]
    rank0_ok = torch.equal(expected_rank0_gate, rank_params[0].data[:per_rank_shard])
    print(f"  rank0 gate slice matches independently-computed chunk(2,0)[0]: {rank0_ok}")

    assert gate_ok, "gate_proj shard reconstruction across ranks FAILED"
    assert up_ok, "up_proj shard reconstruction across ranks FAILED"
    assert rank0_ok, "rank0 slice doesn't match an independently-computed chunk"
    print("  [PASS]")


def test_column_parallel_tp2():
    print("\n" + "=" * 70)
    print("ColumnParallelLinear.weight_loader shard-selection at tp_size=2")
    print("=" * 70)
    hidden_size, output_size, tp_size = 16, 24, 2

    torch.manual_seed(1)
    full = torch.randn(output_size, hidden_size)
    per_rank = output_size // tp_size

    parts = []
    for tp_rank in range(tp_size):
        p = SimpleNamespace(data=torch.zeros(per_rank, hidden_size), tp_dim=0, tp_size=tp_size, tp_rank=tp_rank)
        ColumnParallelLinear.weight_loader(p, p, full)
        parts.append(p.data)

    reconstructed = torch.cat(parts, dim=0)
    ok = torch.equal(reconstructed, full)
    print(f"  reconstructed == full: {ok}")
    assert ok
    print("  [PASS]")


def test_row_parallel_tp2():
    print("\n" + "=" * 70)
    print("RowParallelLinear.weight_loader shard-selection at tp_size=2 (splits input dim)")
    print("=" * 70)
    hidden_size, input_size, tp_size = 16, 24, 2

    torch.manual_seed(2)
    full = torch.randn(hidden_size, input_size)  # (output, input) -- RowParallel splits dim 1
    per_rank = input_size // tp_size

    parts = []
    for tp_rank in range(tp_size):
        p = SimpleNamespace(data=torch.zeros(hidden_size, per_rank), tp_dim=1, tp_size=tp_size, tp_rank=tp_rank)
        RowParallelLinear.weight_loader(p, p, full)
        parts.append(p.data)

    reconstructed = torch.cat(parts, dim=1)
    ok = torch.equal(reconstructed, full)
    print(f"  reconstructed == full: {ok}")
    assert ok
    print("  [PASS]")


def test_vocab_parallel_embedding_tp2():
    print("\n" + "=" * 70)
    print("VocabParallelEmbedding.weight_loader shard-selection at tp_size=2")
    print("=" * 70)
    vocab_size, hidden_size, tp_size = 32, 16, 2

    torch.manual_seed(3)
    full = torch.randn(vocab_size, hidden_size)
    per_rank = vocab_size // tp_size

    parts = []
    for tp_rank in range(tp_size):
        p = SimpleNamespace(data=torch.zeros(per_rank, hidden_size), tp_rank=tp_rank, tp_size=tp_size)
        VocabParallelEmbedding.weight_loader(p, p, full)
        parts.append(p.data)

    reconstructed = torch.cat(parts, dim=0)
    ok = torch.equal(reconstructed, full)
    print(f"  reconstructed == full: {ok}")
    assert ok
    print("  [PASS]")


def main():
    test_merged_column_parallel_tp2()
    test_column_parallel_tp2()
    test_row_parallel_tp2()
    test_vocab_parallel_embedding_tp2()
    print("\nALL TP=2 SHARD-SELECTION CHECKS PASSED")


if __name__ == "__main__":
    main()
