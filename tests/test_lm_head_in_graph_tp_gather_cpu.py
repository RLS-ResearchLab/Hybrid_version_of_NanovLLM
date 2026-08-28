"""CPU-only correctness check for config.use_lm_head_in_graph's tp>1 support
(engine/model_runner.py's capture_cudagraph()/run(), layers/embed_head.py's
ParallelLMHead._local_logits()).

WHY THIS TEST EXISTS: ParallelLMHead.forward()'s tp>1 combine allocates
fresh tensors every call (`[torch.empty_like(logits) for _ in
range(self.tp_size)]` then `torch.cat(...)`) -- fine in eager code, not
CUDA-graph-capture-safe. 2026-08-28's fix has capture_cudagraph() call
_local_logits() (the pre-gather per-rank computation, extracted from
forward() the same day) and do its OWN combine against STATIC,
pre-allocated buffers: dist.gather writes into a persistent per-rank
gather_dest list instead of a fresh one, and torch.cat's `out=` parameter
writes into a persistent combined buffer instead of allocating a new one.

This test reproduces that exact sequence (mp.spawn + real gloo
dist.gather(), not a mock) and checks the result against an INDEPENDENTLY
RECONSTRUCTABLE dense reference -- each rank's weight shard is a
deterministic, per-row-unique pattern (not collinear/constant, matching
this project's own established discipline against tests that can't
actually distinguish a wrong-offset bug), so the expected full-vocab
result can be computed on any single process without needing its own
collective.

Does NOT validate: actual CUDA graph capture (device="cuda" only,
untestable without a GPU) -- this is Part A/B/C's "Part C" analog for THIS
fix: real end-to-end multi-process construction + a real collective, not
just isolated math, but still short of the actual capture step.

Usage:
    python tests/test_lm_head_in_graph_tp_gather_cpu.py
"""
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

HIDDEN = 8
VOCAB = 32   # small, but divisible by every world_size tried below
N = 5        # batch size (decode bucket proxy)


def _row_tagged(rows: int, cols: int, base: float) -> torch.Tensor:
    """Every row globally unique (base+i), not per-head/per-shard-constant --
    same discipline as tests/test_gqa_kv_replication_tp4.py's helper of the
    same name, for the same reason: a constant-per-shard weight couldn't
    distinguish "gathered shard N into the right slot" from "gathered it
    into the wrong slot but the values happened to match anyway"."""
    t = torch.empty(rows, cols)
    for i in range(rows):
        t[i, :] = float(base + i)
    return t


def _worker(rank: int, world_size: int, port: int, results):
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{port}",
                             rank=rank, world_size=world_size)

    from nanovllm.layers.embed_head import ParallelLMHead

    shard_rows = VOCAB // world_size
    lm_head = ParallelLMHead(VOCAB, HIDDEN)
    # Deterministic, row-tagged shard -- matches what weight_loader would
    # have narrowed out of a full row-tagged dense weight, reconstructed
    # independently below without needing the real loader.
    lm_head.weight.data.copy_(_row_tagged(shard_rows, HIDDEN, base=rank * shard_rows * 10))

    torch.manual_seed(1234)   # same seed on every rank -> bit-identical x
    x = torch.randn(N, HIDDEN)

    logits_local = lm_head._local_logits(x)   # (N, shard_rows) -- this rank's shard only
    assert logits_local.shape == (N, shard_rows), f"rank{rank}: unexpected local shape {logits_local.shape}"

    # Exact sequence engine/model_runner.py's capture_cudagraph()/_step()
    # uses -- pre-allocated buffers, not fresh ones. Allocated here (not
    # once outside the function) only because this is a single-shot test,
    # not a repeated-replay harness; the POINT under test is that
    # dist.gather/torch.cat write into tensors that already exist, which
    # holds regardless of when the allocation happened.
    if rank == 0:
        gather_dest = [torch.zeros(N, shard_rows) for _ in range(world_size)]
        combined = torch.zeros(N, VOCAB)
    else:
        gather_dest = None
        combined = None

    dist.gather(logits_local, gather_dest, 0)

    if rank == 0:
        torch.cat(gather_dest, dim=-1, out=combined)

    ok = True
    checks = {}
    if rank == 0:
        # Independently-reconstructable dense reference -- no collective
        # needed, every rank could compute this on its own since the
        # weight-tagging scheme is deterministic.
        dense_weight = torch.cat(
            [_row_tagged(shard_rows, HIDDEN, base=r * shard_rows * 10) for r in range(world_size)],
            dim=0,
        )
        expected = F.linear(x, dense_weight)   # (N, VOCAB)

        # NOT torch.equal -- isolated and confirmed (see this file's own
        # investigation, reproducible with zero distributed code at all:
        # one (N,H)x(H,VOCAB) matmul vs. world_size separate (N,H)x(H,shard)
        # matmuls concatenated already differ by ~1e-4 at world_size=4,
        # purely from BLAS choosing different internal blocking/summation
        # order for different GEMM shapes -- benign reassociation, the same
        # class of difference this project already tolerates elsewhere
        # (e.g. test_qwen35_gdr_decode_batched.py's cos>0.99999 bar), not a
        # bug in the gather/cat logic under test. world_size=2 happened to
        # land on the same BLAS code path as the dense reference (0.0 diff)
        # -- that's a coincidence of shape, not evidence the bar should be
        # torch.equal.
        cos = F.cosine_similarity(combined.flatten().unsqueeze(0),
                                  expected.flatten().unsqueeze(0), dim=-1).item()
        max_abs_diff = (combined - expected).abs().max().item()
        checks["shape"] = tuple(combined.shape) == (N, VOCAB)
        checks["cosine"] = cos
        checks["max_abs_diff"] = max_abs_diff
        ok = checks["shape"] and cos > 0.999999 and max_abs_diff < 1e-2
    else:
        # Non-dst ranks: gather_dest/combined are None, matching
        # ParallelLMHead.forward()'s own `if self.tp_rank == 0 else None`
        # convention -- nothing more to check here besides "didn't crash".
        checks["non_dst_no_crash"] = True

    results.append((rank, ok, checks))
    dist.destroy_process_group()


def run_check(world_size: int, port: int):
    print("=" * 74)
    print(f"tp>1 lm_head-in-graph static-buffer gather -- world_size={world_size}")
    print("=" * 74)
    manager = mp.Manager()
    results = manager.list()
    mp.spawn(_worker, args=(world_size, port, results), nprocs=world_size, join=True)

    assert len(results) == world_size, f"expected {world_size} results, got {len(results)}"
    all_ok = True
    for rank, ok, checks in sorted(results):
        status = "OK" if ok else "FAIL"
        print(f"  [rank{rank}] {status} -- {checks}")
        all_ok = all_ok and ok
    assert all_ok, f"one or more ranks failed at world_size={world_size} -- see output above"
    print(f"  [PASS] static-buffer gather matches the dense reference (cos>0.999999) at world_size={world_size}")
    return all_ok


def main():
    ok = True
    ok &= run_check(world_size=2, port=29881)
    ok &= run_check(world_size=4, port=29882)

    print("\n" + "-" * 74)
    print("PASS" if ok else "FAIL")
    if ok:
        print("  The pre-allocated dist.gather + torch.cat(out=...) sequence matches (within")
        print("  benign BLAS-reassociation tolerance) what ParallelLMHead.forward()'s original")
        print("  fresh-allocation combine would produce, at multiple world_sizes, via a real")
        print("  gloo collective. Does NOT prove CUDA-graph capture itself succeeds -- needs a GPU.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
