"""CPU-only correctness check for smoke_test_moe_w8a8_hopper.py's FP8
quantization helpers (quantize_activation_fp8_dynamic, quantize_weight_fp8_grouped,
dequantize_weight_fp8_grouped_gathered) -- runs without CUDA, without triton, and
without compiling moe_w8a8.cu, since these are pure-PyTorch functions and this
PyTorch build (2.13.0+cpu, confirmed empirically) supports torch.float8_e4m3fn
tensor ops on CPU.

Does NOT validate: the actual moe_w8a8.cu kernel (needs a real compile+run on
Hopper), or that these helpers' quantization CONVENTION matches what the
kernel actually expects (that's still an inferred hypothesis, per moe_w8a8.h's
own docstring -- this only checks the helpers are internally consistent and
shape-correct, not that they're "right" in an absolute sense).

Usage:
    python layers/test_moe_w8a8_hopper_quant_helpers_cpu.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from smoke_test_moe_w8a8_hopper import (  # noqa: E402
    quantize_activation_fp8_dynamic,
    quantize_weight_fp8_grouped,
    dequantize_weight_fp8_grouped_gathered,
)


def main():
    torch.manual_seed(0)
    ok = True

    # ---- 1. Activation quantization: shape + round-trip sanity ----
    # NOTE: quantize_activation_fp8_dynamic was CORRECTED 2026-08-24 to return
    # a per-(token, 128-K-block) scale of shape (M, K // 128), replacing the
    # old per-whole-row (M,) that caused an on-hardware out-of-bounds read in
    # moe_w8a8.cu (see that function's docstring). This test still asserted
    # the stale (M,) shape and was failing on that alone -- fixed 2026-08-27.
    M, K, BS = 6, 256, 128
    x = torch.randn(M, K) * 0.02
    x_fp8, x_scale = quantize_activation_fp8_dynamic(x)
    print(f"[1] x_fp8={tuple(x_fp8.shape)} x_scale={tuple(x_scale.shape)} dtype={x_fp8.dtype}")
    assert x_fp8.shape == (M, K), "activation quant fp8 shape mismatch"
    assert x_scale.shape == (M, K // BS), (
        f"activation quant scale shape mismatch: {tuple(x_scale.shape)} != {(M, K // BS)} "
        f"-- expected per-(token, 128-K-block) scale"
    )
    recon = (x_fp8.float().view(M, K // BS, BS) * x_scale.unsqueeze(-1)).view(M, K)
    cos1 = F.cosine_similarity(x.reshape(-1), recon.reshape(-1), dim=0).item()
    print(f"    round-trip cosine={cos1:.6f}")
    assert cos1 > 0.99, "activation quant round-trip too lossy -- likely a real bug, not just fp8 precision"
    ok &= cos1 > 0.99

    # ---- 2. Weight quantization: 2D-blocked shape sanity ----
    E, N, Kw, group_size = 4, 256, 256, 128
    w = torch.randn(E, N, Kw) * 0.02
    w_fp8, w_scale = quantize_weight_fp8_grouped(w, group_size)
    print(f"[2] w_fp8={tuple(w_fp8.shape)} w_scale={tuple(w_scale.shape)}")
    assert w_fp8.shape == (E, N, Kw), "weight quant shape mismatch"
    assert w_scale.shape == (E, N // group_size, Kw // group_size), "weight scale shape mismatch"

    # ---- 3. Gathered-dequant order-independence: gather-then-dequantize
    # must equal dequantize-then-gather bit-for-bit (same underlying tensors,
    # different operation order) -- exactly the class of scale/weight
    # misalignment bug quantization code is prone to after a gather. ----
    idx = torch.randint(0, E, (M, 4), dtype=torch.int64)  # (M, top_k)
    w_gathered_fp8 = w_fp8[idx]
    w_gathered_scale = w_scale[idx]
    deq_gathered = dequantize_weight_fp8_grouped_gathered(
        w_gathered_fp8, w_gathered_scale, group_size, torch.float32)
    print(f"[3] gathered dequant shape={tuple(deq_gathered.shape)}  expected=({M}, 4, {N}, {Kw})")
    assert deq_gathered.shape == (M, 4, N, Kw), "gathered dequant shape mismatch"

    deq_full = dequantize_weight_fp8_grouped_gathered(w_fp8, w_scale, group_size, torch.float32)
    deq_indexed_after = deq_full[idx]
    max_diff = (deq_gathered - deq_indexed_after).abs().max().item()
    print(f"    gather-before-dequant vs. gather-after-dequant max diff: {max_diff:.3e}")
    assert max_diff < 1e-6, "gather/dequant order mismatch -- indicates a real indexing bug"
    ok &= max_diff < 1e-6

    print("\nPASS" if ok else "\nFAIL")
    print("\nScope reminder: validates the quantization HELPER functions only -- says nothing "
          "about moe_w8a8.cu itself, which still needs a real compile+run on Hopper.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
