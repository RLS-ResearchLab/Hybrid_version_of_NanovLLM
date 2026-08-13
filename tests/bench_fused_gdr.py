"""Isolated-layer timing for the fused-GDR-kernel path (QLLM Stage 2, first
intervention): Qwen35LinearAttention.forward(), flag off (sequential scan)
vs flag on (flash-linear-attention chunked kernel), at T in {128, 512, 1024}.

Reuses bench_throughput.py's timing convention: torch.cuda.synchronize()
bracketing time.perf_counter(), separate discarded warm-up trials before
the timed trials (warm-up matters here too -- Qwen35RMSNormGated / the norm
call inside forward() may be compiled, and the fused kernel path itself
JIT-compiles its Triton kernels on first call per shape).

This is layer-level timing only (single Qwen35LinearAttention forward, not
end-to-end prefill through the full model / scheduler) -- for end-to-end
prefill timing, use bench_throughput.py itself with --model pointed at a
checkpoint built with use_fused_gdr_kernel toggled via Config (see the
correctness test / deliverable notes for how that flag threads through).

Requires flash-linear-attention ('fla') installed and a CUDA GPU. Prints a
[SKIP] message and exits cleanly if unavailable, same as
tests/test_qwen35_fused_gdr.py.

--real-dims switches from the small 290M synthetic config (hidden_size=512,
LKH=8, LVH=16, LHD=64 -- convenient for the correctness tests, which this
script's default matched for consistency) to the real Qwen3.5-35B-A3B GDR
dims (hidden_size=2048, LKH=16, LVH=32, LHD=128, per
src/model_small_qwen3.5.py's own docstring listing both). This matters:
at the small dims, actual per-token compute is tiny relative to the fixed
Python/Triton-kernel-launch overhead chunk_gated_delta_rule pays per call
(it issues several separate Triton kernels internally -- cumsum, intra-
chunk, recurrent-state, output-combination), so a small-dims run can show
fused time that's flat across T (dispatch-overhead-dominated) rather than
scaling with compute -- exactly what the default config's first run showed
(2.5ms flat from T=128 to T=1024, while sequential scaled ~9x). --real-dims
uses random-init weights at the real shapes (timing doesn't depend on
weight values) without needing the actual checkpoint downloaded.

Usage:
    python tests/bench_fused_gdr.py
    python tests/bench_fused_gdr.py --seq-lens 128 512 1024 --trials 10 --warmup-trials 3
    python tests/bench_fused_gdr.py --real-dims --seq-lens 128 512 1024 --trials 10 --warmup-trials 3
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config  # noqa: E402

try:
    import fla  # noqa: F401
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False


def make_real_dims_config():
    """Real Qwen3.5-35B-A3B GDR dims (src/model_small_qwen3.5.py's docstring:
    'H 2048 (was 512)', 'LKH 16 (was 8)', 'LVH 32 (was 16)', 'LHD 128
    (was 64)'). conv_kernel_size (CK=4) is unchanged between small/real per
    that same docstring. Only the fields build_layer() actually reads are
    populated -- this is NOT a full model config, just enough to construct
    a standalone Qwen35LinearAttention at real shapes for timing."""
    return SimpleNamespace(
        hidden_size=2048,
        linear_attn_kq_heads=16,
        linear_attn_v_heads=32,
        linear_attn_head_dim=128,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
    )


def build_layer(config, device, use_fused):
    from nanovllm.models.qwen3_5 import Qwen35LinearAttention

    torch.manual_seed(42)
    la = Qwen35LinearAttention(
        hidden_size=config.hidden_size,
        linear_attn_kq_heads=config.linear_attn_kq_heads,
        linear_attn_v_heads=config.linear_attn_v_heads,
        linear_attn_head_dim=config.linear_attn_head_dim,
        conv_kernel_size=config.conv_kernel_size,
        rms_norm_eps=config.rms_norm_eps,
        use_fused_gdr_kernel=use_fused,
    ).to(device).to(torch.bfloat16)
    la.eval()
    return la


def time_forward(la, x, cu_seqlens, trials, warmup_trials):
    with torch.no_grad():
        for w in range(warmup_trials):
            la(x, cu_seqlens)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(trials):
            la(x, cu_seqlens)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0

    return wall_s / trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup-trials", type=int, default=3)
    ap.add_argument("--real-dims", action="store_true",
                     help="Use real Qwen3.5-35B-A3B GDR dims instead of the small "
                          "290M synthetic config -- see module docstring for why "
                          "this matters before trusting a speedup number.")
    args = ap.parse_args()

    init_dist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("  [SKIP] no CUDA GPU available in this environment")
        return

    config = make_real_dims_config() if args.real_dims else make_small_config()
    print(f"  config: hidden_size={config.hidden_size}, "
          f"LKH={config.linear_attn_kq_heads}, LVH={config.linear_attn_v_heads}, "
          f"LHD={config.linear_attn_head_dim} ({'real' if args.real_dims else 'small'}-dims)")
    la_off = build_layer(config, device, use_fused=False)

    if not _FLA_AVAILABLE:
        print("  [SKIP] flash-linear-attention ('fla') not installed -- "
              "reporting sequential-only timings, no speedup ratio")
        la_on = None
    else:
        la_on = build_layer(config, device, use_fused=True)
        la_on.load_state_dict(la_off.state_dict())

    print(f"{'T':>6}  {'sequential (ms)':>16}  {'fused (ms)':>12}  {'speedup':>8}")
    for T in args.seq_lens:
        torch.manual_seed(99)
        x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

        t_off = time_forward(la_off, x, cu_seqlens, args.trials, args.warmup_trials)
        row = f"{T:>6}  {t_off * 1000:>16.3f}"
        if la_on is not None:
            t_on = time_forward(la_on, x, cu_seqlens, args.trials, args.warmup_trials)
            row += f"  {t_on * 1000:>12.3f}  {t_off / t_on:>7.2f}x"
        else:
            row += f"  {'--':>12}  {'--':>8}"
        print(row)


if __name__ == "__main__":
    main()
