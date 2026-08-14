"""Diagnostic for the NaN found in tests/test_qwen35_vectorized_moe.py at
T=1 on GPU (cosine=nan, bitwise_exact=False) -- isolates torch._grouped_mm
itself from the full Qwen35MoE model, and varies exactly one thing at a
time, to determine WHICH condition actually triggers it before attempting
any fix. CPU testing (see models/qwen3_5.py's _forward_dispatch_vectorized
docstring) verified zero-width groups work correctly, INCLUDING a
many-empty-groups case -- but that was the CPU backend specifically; CUDA
almost certainly uses a different underlying kernel (cutlass-based
grouped-gemm vs. a CPU reference path), so this needs its own verification,
not an assumption that the CPU finding transfers.

Runs a battery of cases, each varying one axis, and reports NaN/Inf
presence for each:
  1. Baseline: many nonzero groups, no empty ones (sanity -- does
     grouped_mm work AT ALL on this GPU/torch build).
  2. One empty group among several nonzero ones (the exact case already
     verified clean on CPU) -- does CUDA agree?
  3. MANY empty groups, few nonzero (mimics the actual T=1/NE=32/TK=4
     failing case: 28+ empty out of 32 groups).
  4. The exact production shapes (hidden=512, intermediate*2=512,
     NE=32) at T=1, isolated from the rest of the model (no gate/softmax/
     routing/combine -- just the two grouped_mm calls _forward_dispatch_
     vectorized actually makes, with a manually constructed 4-pair,
     28-empty-group offs, matching what T=1/TK=4/NE=32 actually produces).
  5. Same as 4 but with ALL experts empty except the LAST one (boundary
     case: does it matter WHICH groups are empty, not just how many).

Usage:
    python tests/diag_grouped_mm_nan.py
"""
import torch


def check(label, self_t, mat2, offs):
    try:
        out = torch._grouped_mm(self_t, mat2, offs=offs)
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        status = "NaN/Inf FOUND" if (has_nan or has_inf) else "clean"
        print(f"  [{status}] {label}  (nan={has_nan}, inf={has_inf}, "
              f"out.shape={tuple(out.shape)})")
        return not (has_nan or has_inf)
    except Exception as e:
        print(f"  [ERROR] {label}: {type(e).__name__}: {e}")
        return False


def main():
    if not torch.cuda.is_available():
        print("[SKIP] no CUDA GPU available")
        return

    device = "cuda"
    torch.manual_seed(0)
    K, N = 512, 512  # matches small config's hidden_size / (2*intermediate)-ish scale
    E = 32

    print("\n=== Case 1: baseline, all 32 groups nonzero, roughly even split ===")
    self_t = torch.randn(320, K, device=device, dtype=torch.bfloat16)  # 10 rows/group
    mat2 = torch.randn(E, N, K, device=device, dtype=torch.bfloat16).transpose(-1, -2)
    offs = torch.arange(10, 321, 10, dtype=torch.int32, device=device)
    check("32 groups, 10 rows each, no empties", self_t, mat2, offs)

    print("\n=== Case 2: ONE empty group among several nonzero (CPU-verified-clean case) ===")
    self_t2 = torch.randn(20, K, device=device, dtype=torch.bfloat16)
    mat2_3 = torch.randn(3, N, K, device=device, dtype=torch.bfloat16).transpose(-1, -2)
    offs2 = torch.tensor([12, 12, 20], dtype=torch.int32, device=device)  # group1 empty
    check("3 groups, 1 empty", self_t2, mat2_3, offs2)

    print("\n=== Case 3: MANY empty groups, few nonzero (mimics T=1/NE=32/TK=4) ===")
    self_t3 = torch.randn(4, K, device=device, dtype=torch.bfloat16)  # 4 total rows
    mat2_32 = torch.randn(E, N, K, device=device, dtype=torch.bfloat16).transpose(-1, -2)
    # 4 rows spread across 4 of 32 groups, 28 groups empty -- matches production T=1 case
    offs3 = torch.tensor(
        [1, 2, 3, 4] + [4] * 28, dtype=torch.int32, device=device
    )
    check("32 groups, only 4 nonzero (1 row each), 28 empty", self_t3, mat2_32, offs3)

    print("\n=== Case 4: same as 3 but empties are FIRST (not interleaved) ===")
    self_t4 = torch.randn(4, K, device=device, dtype=torch.bfloat16)
    offs4 = torch.tensor(
        [0] * 28 + [1, 2, 3, 4], dtype=torch.int32, device=device
    )
    check("32 groups, 28 empty FIRST, 4 nonzero LAST", self_t4, mat2_32, offs4)

    print("\n=== Case 5: same as 3 but empties are LAST ===")
    self_t5 = torch.randn(4, K, device=device, dtype=torch.bfloat16)
    offs5 = torch.tensor(
        [1, 2, 3, 4] + [4] * 28, dtype=torch.int32, device=device
    )
    check("32 groups, 4 nonzero FIRST, 28 empty LAST (same as case 3, re-run for noise check)",
          self_t5, mat2_32, offs5)

    print("\n=== Case 6: production-realistic dims -- hidden=512, 2*intermediate=512, "
          "actual T=1/TK=4/NE=32 gate_up_proj-shaped call ===")
    self_t6 = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
    mat2_6 = torch.randn(32, 512, 512, device=device, dtype=torch.bfloat16).transpose(-1, -2)
    offs6 = torch.tensor([0, 1, 1, 2, 2, 2, 3, 4] + [4] * 24, dtype=torch.int32, device=device)
    check("production-shaped gate_up_proj-style call, T=1 pattern", self_t6, mat2_6, offs6)


if __name__ == "__main__":
    main()
