"""TEMPORARY debug tool, 2026-08-24. Matches moe_w8a8.cu's DEBUGXY printf dump
(captured in a log file) against the full reference intermediate saved by
smoke_test_moe_w8a8_hopper.py (h_reference.pt), to empirically derive the
correct thread -> (row, col) mapping for the wgmma accumulator without
needing to perfectly hand-trace the kernel's multi-layer indexing or rely on
partially-fetchable official documentation. Remove once the bug is found and
the kernel's x_row/x_col formula is fixed for real.

Usage:
    python layers/smoke_test_moe_w8a8_hopper.py > smoke_test_full.log 2>&1
    python layers/analyze_wgmma_mapping.py smoke_test_full.log
"""
import re
import sys

import torch

log_path = sys.argv[1] if len(sys.argv) > 1 else "smoke_test_full.log"

ref = torch.load("h_reference.pt")  # (M*top_k, MI)
print(f"Loaded h_reference.pt: shape={tuple(ref.shape)}")

pattern = re.compile(
    r"DEBUGXY row=(-?\d+) col=(-?\d+) val=(-?[\d.eE+-]+) lane=(\d+) warp=(\d+) "
    r"tm=(\d+) tn=(\d+) t=(\d+) scale=([\d.eE+-]+)"
)

rows = []
with open(log_path) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            row, col, val, lane, warp, tm, tn, t, scale = m.groups()
            rows.append(dict(
                row=int(row), col=int(col), val=float(val),
                lane=int(lane), warp=int(warp), tm=int(tm), tn=int(tn), t=int(t),
                scale=float(scale),
            ))

print(f"Parsed {len(rows)} DEBUGXY lines from {log_path}")
if not rows:
    print("No DEBUGXY lines found -- check the log file / grep pattern.")
    sys.exit(1)

# Undo the (buggy) requant-scale division the kernel applies AFTER this point
# in the real code path -- our probe captures `val` BEFORE requant, so it's
# already in the same units as the reference `h` tensor. No adjustment needed.

flat = ref.reshape(-1)
results = []
for r in rows:
    diffs = (flat - r["val"]).abs()
    best_idx = int(diffs.argmin())
    best_val = float(flat[best_idx])
    abs_diff = float(diffs[best_idx])
    rel_diff = abs_diff / (abs(r["val"]) + 1e-12) * 100
    matched_row = best_idx // ref.shape[1]
    matched_col = best_idx % ref.shape[1]
    results.append(dict(
        **r,
        matched_flat_idx=best_idx, matched_row=matched_row, matched_col=matched_col,
        matched_val=best_val, rel_diff_pct=rel_diff,
    ))

# Only trust matches with small relative error -- large-error ones are noise
# (a thread whose value happens to be closest to some unrelated element by
# chance), not real matches.
good = [r for r in results if r["rel_diff_pct"] < 5.0]
print(f"\n{len(good)}/{len(results)} probes matched within 5% relative error.\n")

print(f"{'reported(row,col)':<20} {'matched(row,col)':<20} {'lane':<5} {'warp':<5} {'tm':<4} {'tn':<4} {'t':<3} {'rel_diff%':<10}")
for r in sorted(good, key=lambda r: (r["row"], r["col"])):
    print(f"({r['row']:3d},{r['col']:3d}){'':<10} ({r['matched_row']:3d},{r['matched_col']:3d}){'':<10} "
          f"{r['lane']:<5} {r['warp']:<5} {r['tm']:<4} {r['tn']:<4} {r['t']:<3} {r['rel_diff_pct']:<10.3f}")

# Quick pattern-hunting: does matched_row correlate with (warp, lane) the way
# matched_col currently does with (lane%4), and vice versa? Print a few
# candidate linear-combination fits so the pattern is easier to spot by eye.
print("\n--- Column-mapping probes (row==0 fixed, i.e. tm==0) ---")
for r in sorted([r for r in good if r["row"] == 0], key=lambda r: r["col"]):
    print(f"reported col={r['col']:3d} -> matched k={r['matched_col']:3d}  "
          f"(lane={r['lane']:2d} lane%4={r['lane']%4} lane/4={r['lane']//4} "
          f"warp={r['warp']} warp%4={r['warp']%4} warp/4={r['warp']//4} tn={r['tn']})")

print("\n--- Row-mapping probes (col==0 fixed, i.e. lane%4==0 & warp%4==0-ish) ---")
for r in sorted([r for r in good if r["col"] == 0], key=lambda r: r["row"]):
    print(f"reported row={r['row']:3d} -> matched token*top_k+slot={r['matched_row']:3d}  "
          f"(lane={r['lane']:2d} lane%4={r['lane']%4} lane/4={r['lane']//4} "
          f"warp={r['warp']} tm={r['tm']} t={r['t']})")
