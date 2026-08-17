# tests/diag_topk8_moe_margin.py
"""Diagnostic follow-up to test_qwen35_vectorized_moe.py::test_topk8_regime's
T=37 argmax-mismatch failure (grouped-GEMM dispatch vs. sequential per-expert
loop, top_k=8). That test's own assertion (torch.equal(argmax_on, argmax_off))
correctly crashes on any mismatch -- by design, per project rules, its
threshold is not to be loosened. This script does NOT touch that test or its
assertion; it reruns the identical T=37/top_k=8 case standalone and reports
the one number that decides whether the mismatch is benign reassociation
noise or a real bug: the RELATIVE MARGIN at each diverging token.

Reuses tests/test_qwen35_fused_gdr.py::_assert_argmax_match_or_near_tie's
established relative-margin formula verbatim (same project, same question --
"is this argmax flip a near-tie or a real divergence" -- rather than
inventing a second variant):
    scale  = row.abs().max().clamp_min(1e-6)
    margin = |row[argmax_a] - row[argmax_b]| / scale
computed against y_off's own row (the sequential-loop output, arbitrarily
chosen as the reference row since both sides are otherwise equal-standing --
matches _assert_argmax_match_or_near_tie's y_a/y_b convention).

That file's own near-tie tolerance is margin_tol=0.01, and its docstring
notes ~0.005-scale margins are the observed benign-tie-flip range for this
project's GDR kernel comparisons. Same read applies here: small (~0.005)
means ordinary reassociation noise on a near-tied channel, consistent with
the module docstring's mechanism claim being incomplete-but-not-wrong;
well-separated means the grouped-GEMM path is picking a genuinely different
expert-combination result, a real bug worth chasing before cluster day.

Usage:
    python tests/diag_topk8_moe_margin.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist  # noqa: E402
from test_qwen35_vectorized_moe import _build_pair, make_topk8_config  # noqa: E402


def main():
    init_dist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device={device}")

    T = 37
    config = make_topk8_config()
    moe_off, moe_on = _build_pair(config, device)

    torch.manual_seed(99)
    x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        y_off = moe_off(x)
        y_on = moe_on(x)

    yf_off, yf_on = y_off.float(), y_on.float()
    argmax_off = yf_off.argmax(dim=-1)
    argmax_on = yf_on.argmax(dim=-1)
    mismatch_idx = (argmax_off != argmax_on).nonzero(as_tuple=True)[0]

    print("\n" + "=" * 70)
    print(f"top_k=8, T={T}: {mismatch_idx.numel()}/{T} tokens with argmax mismatch")
    print("=" * 70)

    if mismatch_idx.numel() == 0:
        print("  No mismatch reproduced -- T=37/top_k=8 result may be seed- or "
              "run-order-sensitive; rerun test_topk8_regime to confirm before "
              "trusting this diagnostic's (lack of) findings.")
        return

    max_margin = 0.0
    for idx in mismatch_idx.tolist():
        row = yf_off[idx]
        scale = row.abs().max().clamp_min(1e-6)
        margin = ((row[argmax_off[idx]] - row[argmax_on[idx]]).abs() / scale).item()
        max_margin = max(max_margin, margin)
        val_off = row[argmax_off[idx]].item()
        val_on = row[argmax_on[idx]].item()
        print(f"  token {idx:3d}: argmax_off={argmax_off[idx].item():3d} (val={val_off:+.6f})  "
              f"argmax_on={argmax_on[idx].item():3d} (val={val_on:+.6f})  "
              f"relative_margin={margin:.6f}")

    print(f"\n  max relative margin across {mismatch_idx.numel()} mismatch(es): {max_margin:.6f}")
    print("  (test_qwen35_fused_gdr.py's near-tie tolerance is 0.01; that file's own "
          "docstring cites ~0.005-scale margins as the observed benign-tie-flip range "
          "for this project's GDR kernel)")
    if max_margin < 0.01:
        print("  [READ] small margin -- consistent with ordinary reassociation noise on "
              "a near-tied channel, same class of finding as the GDR kernel's chunk-"
              "boundary reassociation. Mechanism claim in test_qwen35_vectorized_moe.py's "
              "module docstring is incomplete (doesn't account for the top-k combine sum's "
              "own reassociation) but not contradicted by a real bug.")
    else:
        print("  [READ] well-separated margin -- NOT a near-tie. This is not explained by "
              "ordinary reassociation noise; treat as a real divergence and investigate "
              "before cluster day, not as informational-only.")


if __name__ == "__main__":
    main()
