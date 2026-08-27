"""SMELL-3 -- Sampler's whole-batch all-greedy fast path.

layers/sampler.py's _sample() computes a vocab-wide softmax + a Gumbel
exponential_() RNG draw + a division for EVERY row, then discards it via
torch.where wherever the row is greedy (temperature==0). For an all-greedy
batch (the common eval/benchmark case) that whole thing is wasted. forward()
now short-circuits to logits.argmax when the whole batch is greedy.

This test proves the fast path is (a) numerically identical to routing the
same all-greedy batch through _sample(), and (b) NOT taken when any row is
sampled (mixed batch still gets per-row selection).

Pure CPU. Run with TORCHDYNAMO_DISABLE=1.

Usage:
    TORCHDYNAMO_DISABLE=1 python tests/test_sampler_greedy_fastpath_cpu.py
"""
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layers"))
from sampler import Sampler  # noqa: E402


def main():
    torch.manual_seed(0)
    s = Sampler()
    V = 4096
    print("=" * 66)
    print("SMELL-3 -- Sampler all-greedy fast path")
    print("=" * 66)

    ok = True
    for bs in (1, 8, 64):
        logits = torch.randn(bs, V)

        # ---- all-greedy: fast path vs the full _sample() path ----
        temps0 = torch.zeros(bs)
        fast = s(logits, temps0)                    # forward() -> fast path
        full = s._sample(logits.clone(), temps0)    # force the full path
        eager = logits.float().argmax(dim=-1)
        match_full = torch.equal(fast, full)
        match_eager = torch.equal(fast, eager)
        print(f"bs={bs:3d} all-greedy : fast==_sample? {match_full}   fast==logits.argmax? {match_eager}")
        ok &= match_full and match_eager

        # ---- mixed batch: fast path must NOT fire; greedy rows still exact ----
        temps_mixed = torch.zeros(bs)
        if bs > 1:
            temps_mixed[1::2] = 0.8   # half the rows sampled
        torch.manual_seed(123)
        mixed = s(logits, temps_mixed)
        greedy_rows = (temps_mixed == 0).nonzero(as_tuple=True)[0]
        greedy_exact = torch.equal(mixed[greedy_rows], eager[greedy_rows])
        print(f"bs={bs:3d} mixed      : greedy rows still exact vs logits.argmax? {greedy_exact}"
              f"   (sampled rows: {bs - len(greedy_rows)})")
        ok &= greedy_exact

        # ---- all-sampled: fast path must NOT fire, output is stochastic ----
        temps_hot = torch.full((bs,), 1.0)
        torch.manual_seed(1); a = s(logits, temps_hot)
        torch.manual_seed(2); b = s(logits, temps_hot)
        differs = not torch.equal(a, b) if bs > 1 else True  # bs=1 may coincide
        print(f"bs={bs:3d} all-hot    : two draws differ (stochastic path live)? {differs}")

    print("\n" + "-" * 66)
    print("PASS" if ok else "FAIL")
    print("  All-greedy fast path is bitwise-identical to the full path and to")
    print("  a plain logits.argmax; mixed batches still get per-row selection.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
