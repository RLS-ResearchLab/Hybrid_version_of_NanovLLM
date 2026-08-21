"""GPU-kernel-level profile of the fused MoE kernel path UNDER CUDA GRAPH
REPLAY -- the one profiling angle this project hasn't looked at yet.

WHY THIS SCRIPT EXISTS AND WHY THE EXISTING INSTRUMENTATION CAN'T ANSWER
THIS: layers/fused_moe_int8.py already has NANOVLLM_PROFILE_FUSED_MOE=1,
which found alignment ops eating ~57% of time in EAGER mode. That
instrumentation is Python-level: each stage is timed with
torch.cuda.synchronize() + time.perf_counter() wrapped around the Python
call to that stage. This does not work for graph mode, for two independent
reasons, not one:

  1. torch.cuda.synchronize() during CUDA graph CAPTURE is illegal (capture
     forbids stream synchronization) -- turning that env var on for a
     graph-mode run would crash during engine construction
     (model_runner.py's capture_cudagraph(), not during actual generation).
  2. Even if the sync calls were removed: CUDA graph REPLAY is a single
     cudaGraphLaunch from the CPU. fused_moe_int8_forward() (the Python
     function) is only ever called during the ONE-TIME warmup+capture pass
     at engine construction -- it is never invoked again during decode.
     Python-level timers are structurally blind to what graph replay
     actually spends time on, no matter how they're written.

The only way to see time breakdown INSIDE a replayed graph is a GPU-side
kernel tracer (CUPTI, via torch.profiler here) -- it hooks kernel launches
below the Python/aten dispatch layer, so it sees every kernel a graph
replay executes, just without the Python-level "aten::cumsum"-style
operator name wrapping each one (there's no Python call to attribute it
to). This script wraps a real generate() call -- AFTER graph capture has
already happened at engine construction, so what gets profiled is actual
replay, not capture -- in torch.profiler, then buckets the resulting
kernel-level trace into three groups by kernel name:

  - "alignment"     -- moe_align_block_size's ~8-10 small ops (scatter_add,
                        cumsum, argsort, searchsorted, index_put, gather)
  - "fused_moe_gemm" -- the actual Triton kernel (fused_moe_triton.py's
                        fused_moe_kernel)
  - "other"          -- attention, norms, silu, embedding, NCCL comm,
                        everything else in the model

CAVEAT, stated plainly: the bucketing below is a best-effort substring
match against kernel names, written without being able to run this on GPU
myself (no CUDA/triton on this dev machine -- see the project's own
gotchas doc). Exact kernel name strings vary by PyTorch/CUDA/Triton
version. The script ALSO prints the raw top-N kernels by self CUDA time
unconditionally, specifically so a mis-bucketed kernel is still visible and
fixable by eye rather than silently absorbed into the wrong category.

If the profiler trace comes back showing one dominant "cudaGraphLaunch" (or
similar) CPU event and nothing informative on the CUDA side, that's a sign
this PyTorch/driver combination isn't exposing per-kernel replay detail to
CUPTI -- fall back to `nsys profile -o <out> python
tests/diag_fused_kernel_graph_profile.py ...` (Nsight Systems), which
traces at the driver level and reliably sees inside graph replay regardless
of PyTorch profiler support.

Usage:
    NANOVLLM_USE_FUSED_MOE_KERNEL=1 python tests/diag_fused_kernel_graph_profile.py \\
        --checkpoint "$HOME/Hybrid_version_of_NanovLLM/qwen35_checkpoint" \\
        --concurrency 32 --gpu-memory-utilization 0.60
"""
import argparse
import os
import sys
import types

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

import torch  # noqa: E402
import bench_throughput as bt  # noqa: E402

# Substring buckets, case-insensitive. Order matters: checked top to bottom,
# first match wins, so put the most specific pattern for OUR kernel first
# in case a generic substring elsewhere would otherwise steal it.
_ALIGNMENT_PATTERNS = [
    "scatter_add", "cumsum", "devicescan", "scan_kernel", "argsort",
    "sort_kernel", "radix", "searchsorted", "index_put", "index_select",
    "gather_kernel", "indexkernel",
]
_MOE_GEMM_PATTERNS = ["fused_moe_kernel", "triton"]


def _bucket(kernel_name: str) -> str:
    name = kernel_name.lower()
    for pat in _MOE_GEMM_PATTERNS:
        if pat in name:
            return "fused_moe_gemm"
    for pat in _ALIGNMENT_PATTERNS:
        if pat in name:
            return "alignment"
    return "other"


class _Args:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "qwen35_checkpoint"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=32,
                     help="Default 32 -- matches the validated A5 concurrency=32 result "
                          "(202.0 tok/s) this profile is meant to explain.")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64,
                     help="Kept small on purpose: this is a decode-heavy trace, and each "
                          "decode step through 40 layers of the MoE path emits on the order "
                          "of ~20 kernels just from alignment+GEMM; 64 steps already gives "
                          "thousands of profiled kernel launches without an unmanageable "
                          "trace file. Raise it if the top-N breakdown looks noisy.")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.60,
                     help="0.60 default -- matches the setting that avoided the graph-capture "
                          "OOM seen at concurrency 32/64 with this kernel earlier this session; "
                          "the script's own default elsewhere (0.82-0.90) is NOT safe here.")
    ap.add_argument("--moe-w8a8-group-size", type=int, default=128)
    ap.add_argument("--no-fake-config-loader", dest="fake_config_loader", action="store_false", default=True)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--trace-out", default=None,
                     help="Optional: also write a chrome://tracing-compatible JSON trace here "
                          "for manual inspection beyond this script's own summary.")
    cli = ap.parse_args()

    args = _Args()
    args.model = cli.checkpoint
    args.concurrency = [cli.concurrency]
    args.max_num_batched_tokens = 4096
    args.max_model_len = cli.max_model_len
    args.gpu_memory_utilization = cli.gpu_memory_utilization
    args.tensor_parallel_size = cli.tp
    args.enforce_eager = False  # graph mode -- the whole point of this script
    args.fake_config_loader = cli.fake_config_loader
    args.use_fused_gdr_kernel = False
    args.use_moe_w8a8 = True
    args.moe_w8a8_weight_group_size = cli.moe_w8a8_group_size

    print(f"NANOVLLM_USE_FUSED_MOE_KERNEL={os.environ.get('NANOVLLM_USE_FUSED_MOE_KERNEL', '0')}  "
          f"(must be 1 for this profile to be measuring the fused kernel at all)")
    print(f"Building engine: tp={args.tensor_parallel_size}  concurrency={cli.concurrency}  "
          f"gpu_memory_utilization={args.gpu_memory_utilization}  graph_mode=True ...")
    engine = bt.build_engine(args)

    try:
        print("\nWarmup trial (untraced) -- lets any first-call effects (e.g. lazy Triton "
              "kernel compilation/caching that hasn't already happened during graph capture "
              "warmup) settle before the profiled trial ...")
        bt.run_trial(engine, cli.concurrency, cli.prompt_len, cli.output_len, seed=1)

        print("\nProfiled trial ...")
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as prof:
            result = bt.run_trial(engine, cli.concurrency, cli.prompt_len, cli.output_len, seed=2)

        print(f"\nProfiled trial: wall_s={result['wall_s']:.2f}  tok_s={result['tok_s']:.1f}  "
              f"(NOTE: profiler overhead makes this slower than an untraced run -- ignore this "
              f"tok/s number, it is not a throughput result, only the RELATIVE kernel-time "
              f"breakdown below is meaningful)")

        if cli.trace_out:
            prof.export_chrome_trace(cli.trace_out)
            print(f"Chrome trace written to {cli.trace_out} (open in chrome://tracing or "
                  f"https://ui.perfetto.dev for a visual timeline)")

        events = prof.key_averages()
        cuda_events = [e for e in events if getattr(e, "self_cuda_time_total", 0) > 0]
        if not cuda_events:
            print("\nNo CUDA kernel events with nonzero self time found in the trace.")
            print("This usually means this PyTorch/CUPTI/driver combination is not exposing "
                  "per-kernel detail inside a replayed CUDA graph. Fall back to Nsight Systems:")
            print(f"    nsys profile -o fused_kernel_graph_profile python {__file__} "
                  f"--checkpoint {cli.checkpoint} --concurrency {cli.concurrency} "
                  f"--gpu-memory-utilization {cli.gpu_memory_utilization}")
            sys.exit(1)

        total_cuda_us = sum(e.self_cuda_time_total for e in cuda_events)
        bucket_totals = {"alignment": 0.0, "fused_moe_gemm": 0.0, "other": 0.0}
        bucket_counts = {"alignment": 0, "fused_moe_gemm": 0, "other": 0}
        for e in cuda_events:
            b = _bucket(e.key)
            bucket_totals[b] += e.self_cuda_time_total
            bucket_counts[b] += e.count

        print("\n" + "=" * 78)
        print(f"GRAPH-MODE KERNEL TIME BREAKDOWN (best-effort bucketing by kernel name, "
              f"see script docstring for caveats)")
        print("=" * 78)
        print(f"Total profiled CUDA self time: {total_cuda_us / 1000:.1f} ms "
              f"across {len(cuda_events)} distinct kernel names")
        for b in ("fused_moe_gemm", "alignment", "other"):
            pct = 100 * bucket_totals[b] / total_cuda_us if total_cuda_us else 0.0
            print(f"  {b:16s}: {bucket_totals[b] / 1000:9.2f} ms  ({pct:5.1f}%)  "
                  f"{bucket_counts[b]} kernel launches")

        print(f"\nTop {cli.top_n} individual kernels by self CUDA time (sanity-check the "
              f"bucketing above against this -- if something looks mis-bucketed, the pattern "
              f"lists near the top of this file are what to fix):")
        ranked = sorted(cuda_events, key=lambda e: e.self_cuda_time_total, reverse=True)[:cli.top_n]
        for e in ranked:
            pct = 100 * e.self_cuda_time_total / total_cuda_us if total_cuda_us else 0.0
            print(f"  [{_bucket(e.key):14s}] {pct:5.1f}%  {e.self_cuda_time_total/1000:9.2f} ms  "
                  f"count={e.count:6d}  {e.key}")

    finally:
        engine.exit()


if __name__ == "__main__":
    main()
