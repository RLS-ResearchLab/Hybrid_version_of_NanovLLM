"""One-off diagnostic -- does concurrency=16 under INT8 W8A8 succeed in
EAGER mode (no CUDA graph capture)? Isolates whether cluster_a5_concurrency_
sweep.py's OOM at level=16 (which bf16+graphs handled fine at 54.0 tok/s,
per the measurement notebook) is graph-capture-specific.

WHY THIS MATTERS: CUDA graph capture does not free its working memory after
capture -- whatever tensors get materialized DURING the captured forward
pass become part of the graph's PRIVATE memory pool for the graph's entire
lifetime, unlike eager mode where the same tensors would be freed
immediately after each call. dequantize_weight_int8_grouped's fp32
intermediate (even after the in-place mul_ fix) is a real, non-trivial
allocation at the (N, TK, 2*MI, H) gathered-buffer scale. If that gets
baked into the graph's private pool once per captured batch-size shape,
it could plausibly consume more memory than quantization saved on the
resident weights (~15 GB/rank) -- which would explain a level that worked
fine under bf16 now OOMing under INT8, an otherwise counterintuitive
result.

This script does NOT patch cluster_a5_concurrency_sweep.py again (that file
has been through enough edit-application trouble this session). It reuses
bench_throughput.py's build_engine/run_trial directly, exactly as
cluster_a5_concurrency_sweep.py itself does, with a real --enforce-eager
switch this time so eager and graph-enabled can be A/B compared at the
SAME concurrency level directly.

Usage:
    # Does eager mode succeed where graphs OOM'd?
    python tests/diag_w8a8_eager_vs_graph.py --concurrency 16 --moe-w8a8 --enforce-eager

    # Reproduce the graph-enabled OOM for direct comparison (same run,
    # graphs on):
    python tests/diag_w8a8_eager_vs_graph.py --concurrency 16 --moe-w8a8
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import bench_throughput as bt  # noqa: E402


class _Args:
    """Same plain attribute-bag pattern cluster_a5_concurrency_sweep.py
    uses -- bench_throughput.build_engine() reads plain attributes, not a
    real argparse.Namespace."""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(ROOT), "qwen35_checkpoint"))
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    ap.add_argument("--moe-w8a8", action="store_true", default=False)
    ap.add_argument("--moe-w8a8-group-size", type=int, default=128)
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    cli = ap.parse_args()

    args = _Args()
    args.model = cli.checkpoint
    args.max_num_batched_tokens = 4096
    args.concurrency = [cli.concurrency]
    args.max_model_len = cli.max_model_len
    args.gpu_memory_utilization = cli.gpu_memory_utilization
    args.tensor_parallel_size = cli.tp
    args.enforce_eager = cli.enforce_eager
    args.use_fused_gdr_kernel = False
    args.use_moe_w8a8 = cli.moe_w8a8
    args.moe_w8a8_weight_group_size = cli.moe_w8a8_group_size
    args.fake_config_loader = False  # real checkpoint, registered config

    print(f"enforce_eager={args.enforce_eager}  use_moe_w8a8={args.use_moe_w8a8}  "
          f"group_size={args.moe_w8a8_weight_group_size}  concurrency={cli.concurrency}  "
          f"tp={args.tensor_parallel_size}  gpu_memory_utilization={args.gpu_memory_utilization}")
    print("Constructing engine (this is where the level-16 OOM previously fired, "
          "during warmup_model()'s graph capture) ...")
    engine = bt.build_engine(args)
    print("[OK] engine constructed and warmed up without OOM")

    print("Running one decode trial ...")
    result = bt.run_trial(engine, cli.concurrency, cli.prompt_len, cli.output_len, seed=1234)
    print(f"[OK] decode trial completed: tok/s={result['tok_s']:.1f}  "
          f"wall_s={result['wall_s']:.2f}  "
          f"all_completions_reached_output_len="
          f"{all(n == cli.output_len for n in result['completion_lengths'])}")

    engine.exit()


if __name__ == "__main__":
    main()