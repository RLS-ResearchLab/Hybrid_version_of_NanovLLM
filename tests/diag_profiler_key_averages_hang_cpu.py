"""CPU-only reproduction attempt for the torch.profiler post-processing hang
in tests/diag_fused_kernel_graph_profile.py (SESSION_HANDOFF_2026-08-25.md:
"the profiled trial itself completes fine (wall_s/tok_s print), but
bucketing the torch.profiler trace by kernel name afterward hangs
indefinitely. Reducing --output-len 64->16 (4x smaller trace) did NOT
resolve it, ruling out simple trace-size scaling.").

WHERE THE HANG ACTUALLY IS: reading that script closely
(tests/diag_fused_kernel_graph_profile.py:172-182), wall_s/tok_s print
BEFORE prof.key_averages() is ever called -- the hang is inside
key_averages() itself (or possibly export_chrome_trace(), if --trace-out
was passed), which is PyTorch's own trace-aggregation code, not this
project's bucketing loop (that loop, `for e in cuda_events: ... _bucket(...)`,
is a handful of dict increments over an already-materialized list -- no way
that hangs on its own). key_averages() runs identically whether the
underlying events came from CPU or CUDA activities; this script profiles a
CPU-only synthetic workload and times key_averages() directly, to see
whether it scales pathologically with event count/duplication -- something
reproducible with zero CUDA, on this exact machine, right now.

CAVEAT the handoff itself already raises, restated here: "output_len
64->16 (4x smaller trace) did NOT resolve it" argues against pure event-COUNT
scaling being the whole story -- if key_averages() were simply O(n^2) in
raw event count, a 4x smaller trace should have been dramatically faster
even if not instant. This script checks event-count scaling anyway (cheap,
rules in/out the simplest theory first) and, separately, checks whether
DISTINCT KERNEL NAME CARDINALITY (not raw count) is what matters -- CUDA
graph replay is documented (this project's own scripts, e.g. the docstring
here) to produce many kernel launches sharing a small set of names repeated
thousands of times, which is a different shape than "N distinct one-off
Python function calls" and exercises key_averages()'s per-key aggregation
differently.

Usage:
    python tests/diag_profiler_key_averages_hang_cpu.py
"""
import time

import torch


def _run_profiled_workload(num_events: int, num_distinct_names: int) -> torch.profiler.profile:
    """Generates num_events profiler events drawn from num_distinct_names
    distinct labels (record_function names), simulating a trace where a
    small set of kernel names repeats many times -- the CUDA-graph-replay
    shape described in diag_fused_kernel_graph_profile.py's docstring --
    rather than num_events all-distinct one-off calls."""
    x = torch.randn(8, 8)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        for i in range(num_events):
            name = f"kernel_{i % num_distinct_names}"
            with torch.profiler.record_function(name):
                x = x + 1
    return prof


def _time_key_averages(prof: torch.profiler.profile) -> float:
    t0 = time.perf_counter()
    events = prof.key_averages()
    elapsed = time.perf_counter() - t0
    return elapsed, len(events)


def main():
    print("=" * 78)
    print("PART 1 -- does key_averages() scale linearly or worse with raw event count, "
          "holding distinct-name cardinality fixed (matches the handoff's own "
          "output_len 64->16 test, redone here as a controlled sweep)?")
    print("=" * 78)
    NUM_DISTINCT = 20  # matches diag_fused_kernel_graph_profile.py's own estimate of
                        # "~20 kernels just from alignment+GEMM" per decode step
    prev_elapsed = None
    for num_events in [1_000, 4_000, 16_000, 64_000, 128_000]:
        prof = _run_profiled_workload(num_events, NUM_DISTINCT)
        elapsed, n_keys = _time_key_averages(prof)
        ratio = f"  ({elapsed / prev_elapsed:.2f}x prev)" if prev_elapsed else ""
        print(f"  num_events={num_events:7d}  distinct_names={NUM_DISTINCT:3d}  "
              f"key_averages()={elapsed*1000:9.2f} ms  n_aggregated_keys={n_keys}{ratio}")
        prev_elapsed = elapsed

    print()
    print("=" * 78)
    print("PART 2 -- holding raw event count FIXED, does key_averages() scale with the "
          "NUMBER OF DISTINCT NAMES each event is drawn from (CUDA graph replay's actual "
          "shape: a few kernel names repeated thousands of times, vs. this part's higher-"
          "cardinality end simulating many differently-named ops)?")
    print("=" * 78)
    NUM_EVENTS = 32_000
    prev_elapsed = None
    for num_distinct in [5, 20, 100, 1_000, 8_000]:
        prof = _run_profiled_workload(NUM_EVENTS, num_distinct)
        elapsed, n_keys = _time_key_averages(prof)
        ratio = f"  ({elapsed / prev_elapsed:.2f}x prev)" if prev_elapsed else ""
        print(f"  num_events={NUM_EVENTS:7d}  distinct_names={num_distinct:5d}  "
              f"key_averages()={elapsed*1000:9.2f} ms  n_aggregated_keys={n_keys}{ratio}")
        prev_elapsed = elapsed

    print()
    print("Interpretation: near-linear growth (ratio tracking the input-size multiplier) "
          "in BOTH parts means key_averages() itself is not pathological at these scales on "
          "this PyTorch version, and the real hang is more likely specific to CUDA-graph-"
          "replay trace CONTENT this CPU-only workload can't fabricate (e.g. CUPTI "
          "correlation-id bookkeeping unique to real GPU kernel launches, or "
          "export_chrome_trace rather than key_averages if --trace-out was passed on the "
          "real run) -- narrows where to add a timeout/instrumentation next time there's "
          "GPU access, rather than assuming key_averages() in general is the culprit. "
          "Clearly-worse-than-linear growth in either part would instead point at a real, "
          "CPU-reproducible PyTorch profiler scaling bug independent of CUDA at all -- "
          "worth a minimal repro against PyTorch's own issue tracker.")


if __name__ == "__main__":
    main()
