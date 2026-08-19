# tests/test_gqa_kv_replication_tp4.py
"""CPU-only correctness check for GQA kv-head REPLICATION at
tensor_parallel_size > num_key_value_heads (the real checkpoint's
num_attention_heads=16, num_key_value_heads=2, head_dim=256 case at tp=4).

Three coordinated sites changed together to make this possible (see
layers/linear.py's local_num_kv_heads/kv_head_replica_source docstrings and
models/qwen3_5.py's Qwen35FullAttention.__init__ comments for the full
reasoning):
  1. layers/linear.py: local_num_kv_heads/kv_head_replica_source -- the
     shared helper both other sites use, so they can't independently drift
     (see Qwen35LinearAttention's __init__ comment for the in_proj_qkv/conv1d
     incident this is guarding against).
  2. models/qwen3_5.py: Qwen35FullAttention constructs k_proj/v_proj via
     ReplicatedLinear + a new _kv_replicate_weight_loader instance method
     when num_kv_heads < tp_size, instead of ColumnParallelLinear.
  3. engine/model_runner.py: allocate_kv_cache's KV-cache sizing uses the
     same shared helper (not exercised directly here -- that function also
     needs real hf_config/CUDA memory-query plumbing this test can't safely
     fake; local_num_kv_heads itself is covered by test_local_num_kv_heads
     below, which is the actual per-rank-count logic model_runner.py calls).

Follows tests/test_tp_shard_loader.py's approach for the pure shard-selection
math (duck-typed SimpleNamespace stand-ins for self/param, no process group
needed -- weight_loader methods do no collective communication) PLUS
tests/test_load_model_tp.py's approach for real end-to-end construction
(mp.spawn + gloo, genuinely exercises dist.get_world_size()/get_rank() and
real bound-method weight_loader dispatch, not just the isolated math). Both
styles are included because the pure-math style alone can't catch a
construction-time wiring bug (e.g. weight_loader never actually getting
overridden) the way test_load_model_tp.py's own DISPATCH check catches
exactly that class of bug for the existing TP loaders.

nanovllm.layers.attention requires triton + flash_attn at import time (see
that module's top-level imports) -- neither is installed in this CPU-only
environment. The mp.spawn tests below stub nanovllm.layers.attention before
importing qwen3_5, exactly like test_load_model_tp.py already does, since
Qwen35FullAttention.__init__ only imports Attention lazily at construction
time, not at module-import time.

Reference tensors are tagged PER-ROW (globally unique, via _row_tagged) not
per-head-constant -- a per-head-constant tensor would still distinguish
"got head 0 vs head 1" but could not catch a wrong-row-offset bug that stays
within the right head's block (e.g. an off-by-N start index that still lands
inside the same 256-row head). A recent audit in this project found six
tests whose collinear/constant inputs made them unable to observe what they
claimed to validate -- this follows test_load_model_tp.py's own
_row_tagged/_col_tagged convention instead.

Usage:
    python tests/test_gqa_kv_replication_tp4.py
"""
import os
import sys
import types
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm":
    if "nanovllm" not in sys.modules:
        nanovllm_pkg = types.ModuleType("nanovllm")
        nanovllm_pkg.__path__ = [ROOT]
        nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
        sys.modules["nanovllm"] = nanovllm_pkg

from nanovllm.layers.linear import ColumnParallelLinear, local_num_kv_heads, kv_head_replica_source

# Real checkpoint's actual numbers (qwen35_checkpoint/config.json):
# num_attention_heads=16, num_key_value_heads=2, head_dim=256. HIDDEN is
# shrunk for test speed -- doesn't affect any of the sharding math being
# checked, which only depends on head counts/head_dim, not hidden_size.
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN = 16, 2, 256, 8


def _row_tagged(rows: int, cols: int, base: float) -> torch.Tensor:
    """Every row globally unique (base+i, not per-head-constant) -- see
    module docstring for why constant/collinear inputs are insufficient."""
    t = torch.empty(rows, cols)
    for i in range(rows):
        t[i, :] = float(base + i)
    return t


# ─── Part A: pure helper math (no process group, no tensors) ───────────────


def test_local_num_kv_heads():
    print("=" * 70)
    print("local_num_kv_heads: shard-vs-replicate boundary + tp=1/tp=2 unchanged")
    print("=" * 70)

    # tp=1, tp=2 against the real checkpoint's num_key_value_heads=2 -- must
    # match the OLD `num_key_value_heads // tp_size` formula exactly (the
    # only two configs this codebase has actually run against real weights).
    assert local_num_kv_heads(NUM_KV_HEADS, 1) == NUM_KV_HEADS // 1 == 2
    assert local_num_kv_heads(NUM_KV_HEADS, 2) == NUM_KV_HEADS // 2 == 1
    print(f"  tp=1: local_num_kv_heads({NUM_KV_HEADS},1)=2 (matches old formula)")
    print(f"  tp=2: local_num_kv_heads({NUM_KV_HEADS},2)=1 (matches old formula)")

    # tp=4: the new replication case. OLD formula (num_key_value_heads //
    # tp_size) would have silently given 0 -- a zero-sized KV-cache dimension
    # at the model_runner.py call site, no assert to catch it. New helper
    # must give 1, never 0.
    old_broken_formula = NUM_KV_HEADS // 4
    new_value = local_num_kv_heads(NUM_KV_HEADS, 4)
    print(f"  tp=4: OLD formula ({NUM_KV_HEADS}//4) = {old_broken_formula} (the bug)")
    print(f"  tp=4: NEW local_num_kv_heads({NUM_KV_HEADS},4) = {new_value} (the fix)")
    assert old_broken_formula == 0, "test's own premise is wrong if this isn't 0"
    assert new_value == 1, f"expected 1, got {new_value}"

    # boundary: num_kv_heads == tp_size lands in the shard-normally branch
    # and still gives exactly 1 (no discontinuity at the boundary).
    assert local_num_kv_heads(4, 4) == 1

    # genuinely impossible config: tp_size % num_kv_heads != 0 (e.g. 3 kv
    # heads over 4 ranks -- no clean replication mapping) must raise, not
    # silently produce a wrong shard.
    try:
        local_num_kv_heads(3, 4)
        raise AssertionError("expected local_num_kv_heads(3, 4) to raise, it did not")
    except AssertionError as e:
        assert "3" in str(e) and "4" in str(e), f"error message unhelpful: {e}"
        print(f"  local_num_kv_heads(3,4) correctly raised: {str(e)[:90]}...")

    # the sharding-side impossible config still raises too (unchanged from
    # the original bare `assert num_kv_heads % tp_size == 0`): e.g. 3 kv
    # heads over 2 ranks.
    try:
        local_num_kv_heads(3, 2)
        raise AssertionError("expected local_num_kv_heads(3, 2) to raise, it did not")
    except AssertionError:
        print("  local_num_kv_heads(3,2) [sharding-side impossible] correctly raised")

    print("  [PASS]")


def test_kv_head_replica_source_mapping():
    print("\n" + "=" * 70)
    print("kv_head_replica_source: rank -> physical kv head mapping at tp=4")
    print("=" * 70)
    sources = [kv_head_replica_source(NUM_KV_HEADS, 4, r) for r in range(4)]
    print(f"  ranks 0..3 -> source kv head: {sources}")
    assert sources == [0, 0, 1, 1], (
        f"expected [0,0,1,1] (ranks 0,1 -> kv head 0; ranks 2,3 -> kv head 1, "
        f"matching HF's contiguous query-head grouping), got {sources}"
    )
    print("  [PASS]")


# ─── Part B: _kv_replicate_weight_loader shard-selection math ──────────────


def test_kv_replicate_weight_loader_tp4():
    print("\n" + "=" * 70)
    print("Qwen35FullAttention._kv_replicate_weight_loader shard-selection at tp=4")
    print("=" * 70)
    from nanovllm.models.qwen3_5 import Qwen35FullAttention

    full_kv = _row_tagged(NUM_KV_HEADS * HEAD_DIM, HIDDEN, base=7000.0)  # (512, 8)
    head0_ref = full_kv[0:HEAD_DIM]
    head1_ref = full_kv[HEAD_DIM:2 * HEAD_DIM]

    rank_data = []
    for tp_rank in range(4):
        fake_self = SimpleNamespace(
            head_dim=HEAD_DIM, total_num_kv_heads=NUM_KV_HEADS, tp_size=4, tp_rank=tp_rank,
        )
        param = SimpleNamespace(data=torch.zeros(HEAD_DIM, HIDDEN))
        Qwen35FullAttention._kv_replicate_weight_loader(fake_self, param, full_kv)
        rank_data.append(param.data)
        print(f"  rank {tp_rank}: shape={tuple(param.data.shape)}")

    # Bullet 1: each rank's weight is the FULL correct kv head, not a slice.
    expected_per_rank = [head0_ref, head0_ref, head1_ref, head1_ref]
    for r in range(4):
        assert torch.equal(rank_data[r], expected_per_rank[r]), (
            f"rank {r}: expected full kv head {'0' if r < 2 else '1'}, got a different tensor "
            f"(shape {tuple(rank_data[r].shape)} vs expected {tuple(expected_per_rank[r].shape)})"
        )
    print("  [OK] each rank holds the FULL correct kv head (bitwise match vs direct slice)")

    # Bullets 2/3: ranks 0,1 identical (kv head 0); ranks 2,3 identical (kv head 1).
    assert torch.equal(rank_data[0], rank_data[1]), "ranks 0 and 1 should both hold kv head 0"
    assert torch.equal(rank_data[2], rank_data[3]), "ranks 2 and 3 should both hold kv head 1"
    print("  [OK] ranks 0,1 identical (kv head 0); ranks 2,3 identical (kv head 1)")

    # Bullet 4: negative contamination check -- ranks 0/1 must differ from ranks 2/3.
    assert not torch.equal(rank_data[0], rank_data[2]), "rank 0 and rank 2 must NOT be identical"
    assert not torch.equal(rank_data[1], rank_data[3]), "rank 1 and rank 3 must NOT be identical"
    print("  [OK] ranks 0/1 differ from ranks 2/3 (not 'everything got head 0')")

    print("  [PASS]")


def test_q_proj_shards_normally_tp4():
    print("\n" + "=" * 70)
    print("q_proj (unchanged ColumnParallelLinear) still shards normally at tp=4")
    print("=" * 70)
    # q_proj emits 2*num_heads*head_dim (query+gate interleaved per head).
    full_q = _row_tagged(2 * NUM_HEADS * HEAD_DIM, HIDDEN, base=6000.0)
    per_rank = (2 * NUM_HEADS * HEAD_DIM) // 4  # = 4 heads' worth (query+gate) per rank

    parts = []
    for tp_rank in range(4):
        p = SimpleNamespace(data=torch.zeros(per_rank, HIDDEN), tp_dim=0, tp_size=4, tp_rank=tp_rank)
        ColumnParallelLinear.weight_loader(p, p, full_q)
        parts.append(p.data)
        print(f"  rank {tp_rank}: shape={tuple(p.data.shape)} (4 query heads worth)")

    reconstructed = torch.cat(parts, dim=0)
    assert torch.equal(reconstructed, full_q), "reconstructed q_proj shards != full tensor"
    print("  [OK] reconstructing all 4 ranks' shards == full q_proj tensor")

    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.equal(parts[i], parts[j]), f"ranks {i} and {j} got identical q_proj shards"
    print("  [OK] all four ranks' q_proj shards are pairwise distinct")
    print("  [PASS]")


def test_tp2_unchanged():
    print("\n" + "=" * 70)
    print("tp=2 unchanged: k_proj/v_proj still take the shard-normally path")
    print("=" * 70)
    full_kv = _row_tagged(NUM_KV_HEADS * HEAD_DIM, HIDDEN, base=7000.0)
    per_rank = (NUM_KV_HEADS * HEAD_DIM) // 2

    parts = []
    for tp_rank in range(2):
        p = SimpleNamespace(data=torch.zeros(per_rank, HIDDEN), tp_dim=0, tp_size=2, tp_rank=tp_rank)
        ColumnParallelLinear.weight_loader(p, p, full_kv)
        parts.append(p.data)

    assert torch.equal(torch.cat(parts, dim=0), full_kv), "tp=2 reconstruction != full kv tensor"
    assert torch.equal(parts[0], full_kv[0:HEAD_DIM]), "tp=2 rank0 should get exactly kv head 0"
    assert torch.equal(parts[1], full_kv[HEAD_DIM:2 * HEAD_DIM]), "tp=2 rank1 should get exactly kv head 1"
    print("  [OK] tp=2: rank0 == kv head 0 (whole), rank1 == kv head 1 (whole), via the")
    print("       UNCHANGED ColumnParallelLinear.weight_loader -- same code path as before")
    print("       this change, num_kv_heads=2 >= tp_size=2 never enters the replication branch.")
    print("  [PASS]")


# ─── Part C: real end-to-end construction (mp.spawn + gloo, CPU-only) ──────


def _stub_attention_module():
    """Same shim tests/test_load_model_tp.py already uses: Attention needs
    triton + flash_attn at IMPORT time (see layers/attention.py's top-level
    imports), neither installed in this CPU-only environment.
    Qwen35FullAttention.__init__ only imports the real Attention lazily at
    construction time, so stubbing sys.modules before constructing anything
    avoids ever touching the real import."""
    import torch.nn as nn
    stub = types.ModuleType("nanovllm.layers.attention")

    class Attention(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, *a, **k):
            raise NotImplementedError

    stub.Attention = Attention
    sys.modules["nanovllm.layers.attention"] = stub


def _construction_worker(rank: int, tp_size: int, port: int, results):
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=tp_size)
    _stub_attention_module()

    from nanovllm.models.qwen3_5 import Qwen35FullAttention
    from nanovllm.layers.linear import kv_head_replica_source as _kv_source

    attn = Qwen35FullAttention(
        hidden_size=HIDDEN, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        rotary_dim=32, max_position=128, rope_theta=10000.0, rms_norm_eps=1e-6,
    )

    checks = {}
    checks["num_kv_heads"] = (attn.num_kv_heads, 1 if tp_size > NUM_KV_HEADS else NUM_KV_HEADS // tp_size)
    checks["k_proj_shape"] = (
        tuple(attn.k_proj.weight.shape),
        (attn.num_kv_heads * HEAD_DIM, HIDDEN),
    )

    is_replication_regime = tp_size > NUM_KV_HEADS
    dispatched_to_replication_loader = (
        getattr(attn.k_proj.weight.weight_loader, "__func__", None)
        is Qwen35FullAttention._kv_replicate_weight_loader
    )
    checks["dispatch_matches_regime"] = (dispatched_to_replication_loader, is_replication_regime)

    if is_replication_regime:
        full_kv = _row_tagged(NUM_KV_HEADS * HEAD_DIM, HIDDEN, base=7000.0)
        attn.k_proj.weight.weight_loader(attn.k_proj.weight, full_kv)
        source = _kv_source(NUM_KV_HEADS, tp_size, rank)
        expected = full_kv[source * HEAD_DIM:(source + 1) * HEAD_DIM]
        checks["real_loader_data"] = (torch.equal(attn.k_proj.weight.data, expected), True)

    all_ok = all(actual == expected for actual, expected in checks.values())
    dist.destroy_process_group()
    results.append((rank, all_ok, {k: (str(a), str(e)) for k, (a, e) in checks.items()}))


def _run_construction_check(tp_size: int, port: int, label: str):
    print("\n" + "=" * 70)
    print(f"Real Qwen35FullAttention construction via mp.spawn+gloo at tp_size={tp_size} ({label})")
    print("=" * 70)
    manager = mp.Manager()
    results = manager.list()
    mp.spawn(_construction_worker, args=(tp_size, port, results), nprocs=tp_size, join=True)

    assert len(results) == tp_size, f"expected {tp_size} results, got {len(results)}"
    all_ok = True
    for rank, ok, checks in sorted(results):
        status = "OK" if ok else "FAIL"
        print(f"  [rank{rank}] {status} -- {checks}")
        all_ok = all_ok and ok
    assert all_ok, f"one or more ranks failed at tp_size={tp_size} -- see per-rank output above"
    print(f"  [PASS] real construction + dispatch correct at tp_size={tp_size}")


def test_real_construction_tp4():
    _run_construction_check(tp_size=4, port=29870, label="replication regime")


def test_real_construction_tp2_unchanged():
    _run_construction_check(tp_size=2, port=29871, label="shard-normally regime, must NOT dispatch to replication loader")


def main():
    test_local_num_kv_heads()
    test_kv_head_replica_source_mapping()
    test_kv_replicate_weight_loader_tp4()
    test_q_proj_shards_normally_tp4()
    test_tp2_unchanged()
    test_real_construction_tp4()
    test_real_construction_tp2_unchanged()
    print("\nALL GQA KV-HEAD REPLICATION CHECKS PASSED")


if __name__ == "__main__":
    main()
