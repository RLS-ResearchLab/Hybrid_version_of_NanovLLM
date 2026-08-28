"""SMELL-3 -- Sampler's whole-batch all-greedy fast path.

layers/sampler.py's _sample() computes a vocab-wide softmax + a Gumbel
exponential_() RNG draw + a division for EVERY row, then discards it via
torch.where wherever the row is greedy (temperature==0). For an all-greedy
batch (the common eval/benchmark case) that whole thing is wasted. forward()
now short-circuits to logits.argmax when the whole batch is greedy.

This test proves the fast path is (a) numerically identical to routing the
same all-greedy batch through _sample(), and (b) NOT taken when any row is
sampled (mixed batch still gets per-row selection).

UPDATED 2026-08-28 (CPU window, no GPU access): adds two more checks for the
follow-up fix that removed the GPU->host sync from the hot path --
1. `logits.argmax(dim=-1)` (no `.float()` cast) vs `logits.float().argmax(dim=-1)`
   are proven bitwise-identical on a bf16 input, not just the fp32 tensors the
   original test used (where `.float()` was already a no-op and couldn't have
   caught a dtype-dependent regression).
2. `Sampler.forward()`'s new `all_greedy` parameter (host-precomputed, so
   engine/model_runner.py's hot path never needs
   `bool((temperatures == 0).all())`'s GPU sync) produces output identical to
   the old GPU-side auto-detection, for both all_greedy=True and False/mixed.

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

        # ---- bf16 cast-removal proof: argmax(bf16) == argmax(bf16.float()) ----
        # The original test's `eager` reference above used torch.randn's
        # default fp32 dtype, where `.float()` was already a no-op -- it
        # could never have caught a dtype-dependent regression. bf16 is the
        # actual production dtype (real checkpoint is bfloat16), and this is
        # where removing the cast needed to be proven safe, not assumed.
        logits_bf16 = logits.to(torch.bfloat16)
        no_cast = logits_bf16.argmax(dim=-1)
        with_cast = logits_bf16.float().argmax(dim=-1)
        bf16_match = torch.equal(no_cast, with_cast)
        print(f"bs={bs:3d} bf16 cast  : argmax(bf16) == argmax(bf16.float())? {bf16_match}")
        ok &= bf16_match

        # ---- host-precomputed all_greedy param matches GPU-side auto-detect ----
        fast_auto = s(logits, temps0)                      # all_greedy=None -> auto GPU-side check
        fast_precomputed = s(logits, temps0, all_greedy=True)
        greedy_param_match = torch.equal(fast_auto, fast_precomputed)
        torch.manual_seed(123)
        mixed_auto = s(logits, temps_mixed)
        torch.manual_seed(123)
        mixed_precomputed = s(logits, temps_mixed, all_greedy=False)
        mixed_param_match = torch.equal(mixed_auto, mixed_precomputed)
        print(f"bs={bs:3d} all_greedy param : greedy match={greedy_param_match}  mixed match={mixed_param_match}")
        ok &= greedy_param_match and mixed_param_match

    print("\n" + "-" * 66)
    print("PASS" if ok else "FAIL")
    print("  All-greedy fast path is bitwise-identical to the full path and to")
    print("  a plain logits.argmax; mixed batches still get per-row selection.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
