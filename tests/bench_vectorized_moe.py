"""Isolated-layer timing for the vectorized-MoE-dispatch path (QLLM Stage 2,
grouped-GEMM intervention): Qwen35MoE.forward(), flag off (per-expert Python
loop) vs flag on (torch._grouped_mm), at T in {128, 512, 1024}.

Host+device timing cross-check is now the standard for trust on this
project (established after the GDR kernel bench -- an async-completion gap
would silently produce a fast-looking but wrong number, and the fix is
cheap: CUDA events measure device-side elapsed time directly, immune to
any host-side dispatch/GIL artifact). Built directly into this script this
time rather than as a separate follow-up -- every timed cell reports both
numbers and flags disagreement instead of silently trusting one.

Reuses bench_throughput.py's warm-up-before-timing convention (same
rationale as bench_fused_gdr.py: torch._grouped_mm JIT/kernel-selects on
first call per shape, and Qwen35SharedExpert's projections may be compiled).

--real-dims switches from the small synthetic MoE config (num_experts=32,
top_k=4, moe_intermediate_size=256, hidden_size=512) to the real
Qwen3.5-35B-A3B MoE dims (num_experts=256, top_k=8, moe_intermediate_size=512,
hidden_size=2048), per src/model_small_qwen3.5.py's own docstring listing
both. This matters even more here than for the GDR bench: MoE dispatch cost
scales directly with num_experts and top_k (256 vs 32, 8 vs 4 -- not a minor
factor), and this project has already found once that small-model timing
behavior does not reliably predict real-checkpoint behavior. Uses
random-init weights at the real shapes (timing doesn't depend on weight
values) without needing the actual checkpoint downloaded.

Usage:
    python tests/bench_vectorized_moe.py
    python tests/bench_vectorized_moe.py --seq-lens 128 512 1024 --trials 10 --warmup-trials 3
    python tests/bench_vectorized_moe.py --real-dims --seq-lens 128 512 1024 --trials 10 --warmup-trials 3
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config  # noqa: E402


def make_real_dims_config():
    """Real Qwen3.5-35B-A3B MoE dims (src/model_small_qwen3.5.py's docstring:
    'H 2048 (was 512)', 'NE 256 (was 32)', 'TK 8 (was 4)', 'MI 512
    (was 256)', 'SI 512 (was 256)'). Only the fields build_moe() actually
    reads are populated."""
    return SimpleNamespace(
        hidden_size=2048,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        num_experts_per_tok=8,
    )


def build_moe(config, device, use_vectorized):
    from nanovllm.models.qwen3_5 import Qwen35MoE

    torch.manual_seed(42)
    moe = Qwen35MoE(
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        shared_intermediate_size=config.shared_expert_intermediate_size,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        use_vectorized_moe=use_vectorized,
    ).to(device).to(torch.bfloat16)
    with torch.no_grad():
        moe.experts.gate_up_proj.normal_(0, 0.02)
        moe.experts.down_proj.normal_(0, 0.02)
    moe.eval()
    return moe


def time_forward(moe, x, trials, warmup_trials):
    """Returns (host_ms, device_ms, ratio). device_ms is ground truth
    (CUDA-event elapsed time, cannot be fooled by an async-incomplete
    call); host_ms is what a plain time.perf_counter() bench would report.
    A ratio far from 1.0 means don't trust the host number."""
    with torch.no_grad():
        for _ in range(warmup_trials):
            moe(x)
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(trials):
            moe(x)
        end_evt.record()
        torch.cuda.synchronize()
        device_ms = start_evt.elapsed_time(end_evt) / trials

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(trials):
            moe(x)
        torch.cuda.synchronize()
        host_ms = (time.perf_counter() - t0) / trials * 1000

    ratio = host_ms / device_ms if device_ms > 0 else float("nan")
    return host_ms, device_ms, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup-trials", type=int, default=3)
    ap.add_argument("--real-dims", action="store_true",
                     help="Use real Qwen3.5-35B-A3B MoE dims instead of the small "
                          "synthetic config -- see module docstring for why this "
                          "matters more here than for the GDR bench (num_experts "
                          "and top_k scale MoE dispatch cost directly).")
    args = ap.parse_args()

    init_dist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("  [SKIP] no CUDA GPU available in this environment "
              "(CUDA events require a GPU; correctness was already verified on "
              "CPU separately via test_qwen35_vectorized_moe.py)")
        return

    config = make_real_dims_config() if args.real_dims else make_small_config()
    print(f"  config: hidden_size={config.hidden_size}, num_experts={config.num_experts}, "
          f"top_k={config.num_experts_per_tok}, moe_intermediate_size={config.moe_intermediate_size} "
          f"({'real' if args.real_dims else 'small'}-dims)")

    moe_off = build_moe(config, device, use_vectorized=False)
    moe_on = build_moe(config, device, use_vectorized=True)
    moe_on.load_state_dict(moe_off.state_dict())

    print(f"{'T':>6}  {'off host(ms)':>13}  {'off dev(ms)':>12}  {'off h/d':>8}  "
          f"{'on host(ms)':>13}  {'on dev(ms)':>12}  {'on h/d':>8}  {'speedup(dev)':>12}")
    for T in args.seq_lens:
        torch.manual_seed(99)
        x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

        off_host, off_dev, off_ratio = time_forward(moe_off, x, args.trials, args.warmup_trials)
        on_host, on_dev, on_ratio = time_forward(moe_on, x, args.trials, args.warmup_trials)

        for label, ratio in [("off", off_ratio), ("on", on_ratio)]:
            if not (0.7 < ratio < 1.4):
                print(f"  [SUSPICIOUS] {label} host/device ratio={ratio:.3f} at T={T} -- "
                      f"host and device timings disagree substantially, don't trust this row "
                      f"until investigated.")

        speedup_dev = off_dev / on_dev if on_dev > 0 else float("nan")
        print(f"{T:>6}  {off_host:>13.4f}  {off_dev:>12.4f}  {off_ratio:>8.3f}  "
              f"{on_host:>13.4f}  {on_dev:>12.4f}  {on_ratio:>8.3f}  {speedup_dev:>11.2f}x")


if __name__ == "__main__":
    main()
