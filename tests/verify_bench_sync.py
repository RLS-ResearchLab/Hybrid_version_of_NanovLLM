"""Independent cross-check for tests/bench_fused_gdr.py's timing: measures
the SAME fused-GDR forward call two ways -- host-side time.perf_counter()
bracketed by torch.cuda.synchronize() (what bench_fused_gdr.py uses) vs.
torch.cuda.Event-based device-side timing (measures actual GPU elapsed time
directly, immune to any host-side dispatch/GIL artifact that could make an
async-incomplete call look fast). If these two numbers disagree
substantially, the perf_counter-based bench numbers reported so far
(21.5x end-to-end, 36-63x isolated-layer) should not be trusted -- an
async-completion gap would show host time much lower than device time.

Usage:
    python tests/verify_bench_sync.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist  # noqa: E402
from bench_fused_gdr import make_real_dims_config, build_layer  # noqa: E402


def main():
    init_dist()
    if not torch.cuda.is_available():
        print("[SKIP] no CUDA GPU available")
        return

    config = make_real_dims_config()
    la = build_layer(config, "cuda", use_fused=True)

    T = 1024
    torch.manual_seed(99)
    x = torch.randn(T, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        for _ in range(3):  # warm up JIT-compiled Triton kernels
            la(x, cu_seqlens)
        torch.cuda.synchronize()

        # Device-side event timing -- ground truth, cannot be fooled by an
        # async gap since it directly measures GPU stream elapsed time.
        n = 10
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(n):
            la(x, cu_seqlens)
        end_evt.record()
        torch.cuda.synchronize()
        event_ms = start_evt.elapsed_time(end_evt) / n

        # Host-side perf_counter timing -- what bench_fused_gdr.py reports.
        import time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            la(x, cu_seqlens)
        torch.cuda.synchronize()
        host_ms = (time.perf_counter() - t0) / n * 1000

    print(f"T={T}  device-event ms/call: {event_ms:.4f}   host-perf_counter ms/call: {host_ms:.4f}")
    ratio = host_ms / event_ms if event_ms > 0 else float("nan")
    print(f"ratio (host/device): {ratio:.3f}  -- should be close to 1.0 if sync is correct")
    if not (0.7 < ratio < 1.4):
        print("[SUSPICIOUS] host and device timings disagree substantially -- "
              "investigate before trusting bench_fused_gdr.py's numbers.")
    else:
        print("[OK] host and device timings agree -- no evidence of an async-completion gap.")


if __name__ == "__main__":
    main()
