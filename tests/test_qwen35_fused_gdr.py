"""Correctness tests for the fused-GDR-kernel path (QLLM Stage 2, first
intervention) added to Qwen35LinearAttention.forward() in models/qwen3_5.py.

Compares, at multiple sequence lengths and in the packed multi-segment
(cu_seqlens) batching path:
  1. use_fused_gdr_kernel=True  vs  use_fused_gdr_kernel=False (flag-on vs
     flag-off, same weights, same input) -- the primary regression guard.
  2. use_fused_gdr_kernel=True  vs  src/model_small_qwen3.5.py's LinearAttn
     (ground truth) -- transitively covered by (1) once (1) passes, since
     test_qwen35_standalone.py's test_linear_attention_vs_reference already
     established flag-off matches the reference (note: that specific test,
     and test_linear_attention_incremental, currently call
     Qwen35LinearAttention with a stale (B,T,H)+state=/conv_state= calling
     convention that predates the cu_seqlens packed-batch refactor and will
     raise on the assert at the top of forward() -- unrelated pre-existing
     test debt, not something this file depends on or fixes). This file
     compares against the reference DIRECTLY as well, rather than relying
     solely on that transitivity, per the task's correctness requirements.

Uses the SAME cosine-similarity thresholds already established in
tests/test_qwen35_standalone.py (0.99 for the vs-reference / cross-path
comparison) -- see that file's test_linear_attention_vs_reference.

Requires flash-linear-attention ('fla') installed and a CUDA GPU (the
kernel is Triton-based). Skips with a clear message if unavailable --
this environment (Windows, no GPU, fla not installed) cannot run it; it
is prepared for the user to run on GPU hardware.

Usage:
    python tests/test_qwen35_fused_gdr.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config, load_reference_module, cosine_sim  # noqa: E402

try:
    import fla  # noqa: F401
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False


def _skip(reason):
    print(f"  [SKIP] {reason}")


def _build_pair(config, device, seed=42):
    """Two Qwen35LinearAttention instances (fused off / fused on) with
    identical weights, plus the reference module's LinearAttn with the
    same weights copied in."""
    from nanovllm.models.qwen3_5 import Qwen35LinearAttention

    torch.manual_seed(seed)
    la_off = Qwen35LinearAttention(
        hidden_size=config.hidden_size,
        linear_attn_kq_heads=config.linear_attn_kq_heads,
        linear_attn_v_heads=config.linear_attn_v_heads,
        linear_attn_head_dim=config.linear_attn_head_dim,
        conv_kernel_size=config.conv_kernel_size,
        rms_norm_eps=config.rms_norm_eps,
        use_fused_gdr_kernel=False,
    ).to(device).to(torch.bfloat16)

    la_on = Qwen35LinearAttention(
        hidden_size=config.hidden_size,
        linear_attn_kq_heads=config.linear_attn_kq_heads,
        linear_attn_v_heads=config.linear_attn_v_heads,
        linear_attn_head_dim=config.linear_attn_head_dim,
        conv_kernel_size=config.conv_kernel_size,
        rms_norm_eps=config.rms_norm_eps,
        use_fused_gdr_kernel=True,
    ).to(device).to(torch.bfloat16)
    la_on.load_state_dict(la_off.state_dict())

    ref_mod = load_reference_module()
    ref_la = ref_mod.LinearAttn().to(device).to(torch.bfloat16)
    ref_sd = dict(ref_la.named_parameters())
    for name, param in la_off.named_parameters():
        if name in ref_sd and param.shape == ref_sd[name].shape:
            param.data.copy_(ref_sd[name].data)
    la_on.load_state_dict(la_off.state_dict())

    return la_off, la_on, ref_la


def _run_packed(la, hidden_states, cu_seqlens):
    with torch.no_grad():
        y, states, conv_states = la(hidden_states, cu_seqlens)
    return y, states, conv_states


def test_single_segment(device, T, config=None):
    print("\n" + "=" * 70)
    print(f"Fused GDR vs sequential vs reference -- single segment, T={T}")
    print("=" * 70)
    config = config or make_small_config()
    la_off, la_on, ref_la = _build_pair(config, device)

    torch.manual_seed(99)
    x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

    y_off, s_off, _ = _run_packed(la_off, x, cu_seqlens)
    y_on, s_on, _ = _run_packed(la_on, x, cu_seqlens)

    with torch.no_grad():
        y_ref, s_ref, _ = ref_la(x.unsqueeze(0))

    cos_vs_off = cosine_sim(y_on, y_off)
    cos_state_vs_off = cosine_sim(s_on, s_off)
    cos_vs_ref = cosine_sim(y_on, y_ref)
    argmax_match = (
        y_on.float().argmax(dim=-1) == y_off.float().argmax(dim=-1)
    ).float().mean().item()

    print(f"  fused vs sequential  output cosine: {cos_vs_off:.6f}")
    print(f"  fused vs sequential  state  cosine: {cos_state_vs_off:.6f}")
    print(f"  fused vs reference   output cosine: {cos_vs_ref:.6f}")
    print(f"  fused vs sequential  argmax match:  {argmax_match:.4f}")

    assert cos_vs_off > 0.99, f"fused vs sequential output mismatch: {cos_vs_off}"
    assert cos_state_vs_off > 0.99, f"fused vs sequential state mismatch: {cos_state_vs_off}"
    assert cos_vs_ref > 0.99, f"fused vs reference output mismatch: {cos_vs_ref}"
    assert argmax_match > 0.99, f"fused vs sequential argmax mismatch: {argmax_match}"
    print("  [PASS]")


def test_packed_multi_segment(device, seg_lens=(37, 129, 5), config=None):
    """Correctness under REAL packed batching with unequal segment lengths --
    per this project's repeated finding that a kernel correct for one
    sequence alone can still be wrong once cu_seqlens boundaries are
    involved (segment-crossing leakage, off-by-one chunk boundaries, etc.)."""
    print("\n" + "=" * 70)
    print(f"Fused GDR vs sequential -- packed multi-segment, lens={seg_lens}")
    print("=" * 70)
    config = config or make_small_config()
    la_off, la_on, _ = _build_pair(config, device)

    torch.manual_seed(7)
    N = sum(seg_lens)
    x = torch.randn(N, config.hidden_size, device=device, dtype=torch.bfloat16)
    cu = [0]
    for L in seg_lens:
        cu.append(cu[-1] + L)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)

    y_off, s_off, c_off = _run_packed(la_off, x, cu_seqlens)
    y_on, s_on, c_on = _run_packed(la_on, x, cu_seqlens)

    cos_y = cosine_sim(y_on, y_off)
    cos_s = cosine_sim(s_on, s_off)
    argmax_match = (
        y_on.float().argmax(dim=-1) == y_off.float().argmax(dim=-1)
    ).float().mean().item()

    print(f"  output cosine: {cos_y:.6f}")
    print(f"  state  cosine: {cos_s:.6f}")
    print(f"  argmax match:  {argmax_match:.4f}")

    assert cos_y > 0.99, f"packed output mismatch: {cos_y}"
    assert cos_s > 0.99, f"packed state mismatch: {cos_s}"
    assert argmax_match > 0.99, f"packed argmax mismatch: {argmax_match}"

    # Also check EACH segment in isolation reproduces the same slice of the
    # packed fused output -- catches cross-segment leakage specifically.
    start = 0
    for i, L in enumerate(seg_lens):
        end = start + L
        cu_solo = torch.tensor([0, L], dtype=torch.int32, device=device)
        y_solo, _, _ = _run_packed(la_on, x[start:end], cu_solo)
        cos_solo = cosine_sim(y_solo, y_on[start:end])
        print(f"  segment {i} (len={L}) solo-vs-packed cosine: {cos_solo:.6f}")
        assert cos_solo > 0.99, f"segment {i} solo-vs-packed mismatch: {cos_solo}"
        start = end

    print("  [PASS]")


def main():
    init_dist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not _FLA_AVAILABLE:
        _skip("flash-linear-attention ('fla') is not installed in this environment")
        return
    if device != "cuda":
        _skip("fla's chunked kernel is Triton-based and requires a CUDA GPU; none available here")
        return

    config = make_small_config()
    test_single_segment(device, T=8, config=config)
    test_single_segment(device, T=512, config=config)
    test_single_segment(device, T=1024, config=config)
    test_packed_multi_segment(device, seg_lens=(37, 129, 5), config=config)
    test_packed_multi_segment(device, seg_lens=(1, 1, 1, 1), config=config)  # decode-shaped: fused must fall back
    print("\nAll fused-GDR correctness tests passed.")


if __name__ == "__main__":
    main()
