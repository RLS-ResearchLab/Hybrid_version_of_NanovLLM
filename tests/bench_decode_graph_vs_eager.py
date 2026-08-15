"""QLLM Stage 2, P2 investigation: decode-step latency, eager vs. the
EXISTING monolithic CUDA graph, swept across batch size.

This is the baseline the per-layer-type-graph question turns on: Step 1's
investigation (see tests/profile_decode_launch_overhead.py and the
investigation report) argues from code structure that the current
monolithic graph already collapses the whole decode forward pass into one
graph.replay() call, so splitting it further can only ADD replay calls, not
remove launch overhead -- UNLESS the monolithic graph is barely helping to
begin with, in which case the whole premise needs re-examining before
touching graph structure at all. This script produces the number that
either supports or falsifies "the monolithic graph is already doing the
job": eager-vs-graph decode latency, per batch size, with a host+device
cross-check on every timed cell (this project's standard since a prior
benchmark needed both to rule out an async-completion measurement gap).

Also watches for the diagnostic signature from the MoE work: a baseline
whose wall-clock is FLAT across a large change in input size is measuring
fixed overhead, not compute. If EAGER decode latency is flat across the
batch-size sweep, that's the same signature and should be investigated
before trusting any ratio computed against it.

Requires a real CUDA GPU. Prints [SKIP] and exits cleanly otherwise, same
convention as tests/bench_fused_gdr.py.

Usage (fake small model -- self-contained):
    python tests/make_fake_hf_config.py     # if not already run
    python tests/bench_decode_graph_vs_eager.py

Usage (real checkpoint):
    python tests/bench_decode_graph_vs_eager.py \\
        --model /path/to/qwen3.5-35b-a3b --no-fake-config-loader \\
        --max-model-len 4096 --gpu-memory-utilization 0.85 \\
        --batch-sizes 1 2 4 8 16 32 --csv-out decode_graph_vs_eager.csv
"""

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

from profile_decode_launch_overhead import (  # noqa: E402
    build_runner, init_weights_and_guard, make_prefill_batch,
    run_prefill_to_completion, cleanup_running, time_iters,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "fake_qwen35_small"))
    ap.add_argument("--no-fake-config-loader", dest="fake_config_loader", action="store_false", default=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--prefill-len", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--warmup-iters", type=int, default=5)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--csv-out", default=os.path.join(os.path.dirname(__file__), "decode_graph_vs_eager.csv"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] no CUDA GPU available in this environment")
        return

    max_num_seqs = max(args.batch_sizes)
    print(f"Building ModelRunner (model={args.model}, max_num_seqs={max_num_seqs})...")
    runner, config = build_runner(
        args.model, args.fake_config_loader, max_num_seqs, args.max_model_len,
        args.gpu_memory_utilization, use_fused_gdr_kernel=False,
    )
    print(f"  [OK] graphs captured for batch sizes: {getattr(runner, 'graph_bs', 'NONE')}")
    init_weights_and_guard(runner, args.fake_config_loader)

    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence

    scheduler = Scheduler(config, runner.state_manager, runner)

    write_header = not os.path.exists(args.csv_out)
    csv_file = open(args.csv_out, "a", newline="")
    fieldnames = ["timestamp", "model", "batch_size", "mode", "host_ms_per_step",
                  "device_ms_per_step", "host_device_disagreement_pct"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    results = []
    try:
        for bs in args.batch_sizes:
            print(f"\n### batch_size={bs} ###")
            added = make_prefill_batch(scheduler, Sequence, bs, args.prefill_len, seed=3000 + bs)
            run_prefill_to_completion(runner, scheduler, added)
            decode_seqs = list(scheduler.running)
            assert len(decode_seqs) == bs

            def decode_step():
                seqs, is_prefill = scheduler.schedule()
                assert not is_prefill
                token_ids = runner.run(seqs, False)
                scheduler.postprocess(seqs, token_ids, False)

            row_data = {}
            for mode, eager_flag in (("eager", True), ("graph", False)):
                runner.enforce_eager = eager_flag
                for _ in range(args.warmup_iters):
                    decode_step()
                host_ms, device_ms = time_iters(decode_step, args.trials)
                disagreement = abs(host_ms - device_ms) / max(host_ms, device_ms, 1e-6) * 100
                print(f"  {mode:>6s}: host={host_ms:.4f}ms  device={device_ms:.4f}ms  "
                      f"disagreement={disagreement:.1f}%")
                if disagreement > 15:
                    print(f"    [WARN] host/device disagree by {disagreement:.0f}% for {mode} -- "
                          f"investigate before trusting this cell")
                row_data[mode] = (host_ms, device_ms)
                writer.writerow({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": args.model,
                    "batch_size": bs,
                    "mode": mode,
                    "host_ms_per_step": host_ms,
                    "device_ms_per_step": device_ms,
                    "host_device_disagreement_pct": disagreement,
                })
                csv_file.flush()

            eager_host, _ = row_data["eager"]
            graph_host, _ = row_data["graph"]
            speedup = eager_host / graph_host if graph_host > 0 else float("nan")
            print(f"  speedup (eager/graph, host time): {speedup:.2f}x")
            results.append((bs, eager_host, graph_host, speedup))

            cleanup_running(scheduler, decode_seqs)
    finally:
        csv_file.close()

    print("\n" + "=" * 70)
    print("SUMMARY -- decode latency, eager vs. existing monolithic graph")
    print("=" * 70)
    print(f"{'bs':>6}  {'eager (ms)':>12}  {'graph (ms)':>12}  {'speedup':>8}")
    for bs, eager_ms, graph_ms, speedup in results:
        print(f"{bs:>6}  {eager_ms:>12.4f}  {graph_ms:>12.4f}  {speedup:>7.2f}x")

    if len(results) >= 2:
        eager_vals = [r[1] for r in results]
        spread = (max(eager_vals) - min(eager_vals)) / max(eager_vals)
        if spread < 0.15:
            print(f"\n[WARN] eager decode latency is FLAT across the batch-size sweep "
                  f"(min={min(eager_vals):.4f}ms, max={max(eager_vals):.4f}ms, "
                  f"spread={spread*100:.0f}%) -- this is the diagnostic signature from the "
                  f"MoE work: a baseline flat across a large input-size change is measuring "
                  f"fixed overhead, not compute. Investigate before trusting any ratio computed "
                  f"against this eager baseline (including the speedup column above).")

    print(f"\nResults written to {args.csv_out}")


if __name__ == "__main__":
    main()
