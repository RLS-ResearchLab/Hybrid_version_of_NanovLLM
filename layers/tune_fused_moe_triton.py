"""Kernel config sweep for the grouped-scale fused_moe_triton.py kernel --
finds a better (BLOCK_SIZE_M/N/K, num_warps, num_stages) combination for
whatever GPU this runs on, rather than assuming layers/fused_moe_int8.py's
_DEFAULT_CONFIG (validated on A6000/Ampere) is also good on Hopper. Ampere
and Hopper have different shared-memory sizes and tensor-core generations --
a config tuned for one is not guaranteed efficient on the other, and this
config has never been retuned since it was set.

Reuses layers/smoke_test_fused_moe_triton.py's real-quantizer setup (same
quantize_weight_int8_grouped call, not synthetic int8 noise) so results are
directly comparable to that script's own baseline number. M defaults to 32
to match the concurrency level where the real engine's throughput curve was
actually measured this session, not the smoke test's modest M=16.

Uses CUDA events for the timed comparison, not just host-side
time.perf_counter() -- the session's own GIL-starvation bug (found in
src/server.py's BatchedEngine._loop()) is a direct, concrete reminder that
host-side timing artifacts are a real risk on this codebase specifically,
not a theoretical one. Host time is still printed alongside, and a
disagreement between them is flagged rather than silently trusted.

Usage:
    python layers/tune_fused_moe_triton.py
    python layers/tune_fused_moe_triton.py --M 64 --trials 20 --warmup 5
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from moe_align_block_size import moe_align_block_size          # noqa: E402
from fused_moe_triton import invoke_fused_moe_kernel             # noqa: E402
from moe_int8_quantize import quantize_weight_int8_grouped       # noqa: E402


def tl_compute_type(dtype):
    import triton.language as tl
    return {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype]


# Baseline is _DEFAULT_CONFIG from layers/fused_moe_int8.py -- always first
# in the list, so its rank in the results table is directly visible rather
# than needing a separate reference run to compare against.
CANDIDATE_CONFIGS = [
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},  # current default
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 4},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--E", type=int, default=128, help="local expert count -- matches the smoke test's tp=2 shape by default")
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=8, dest="top_k")
    ap.add_argument("--M", type=int, default=32, help="concurrency proxy -- default matches the level the real sweep was measured at, not the smoke test's M=16")
    ap.add_argument("--group-size", type=int, default=128, dest="group_size")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for this sweep.")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    print(f"Config: E={args.E} K={args.K} N={args.N} top_k={args.top_k} M={args.M} "
          f"group_size={args.group_size} trials={args.trials} warmup={args.warmup}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    w_bf16 = torch.randn((args.E, args.N, args.K), dtype=dtype, device=device) * 0.02
    w_int8, w_scale = quantize_weight_int8_grouped(w_bf16, args.group_size)

    x = torch.randn((args.M, args.K), dtype=dtype, device=device) * 0.02
    topk_ids = torch.randint(0, args.E, (args.M, args.top_k), dtype=torch.int32, device=device)
    topk_weights = torch.rand((args.M, args.top_k), dtype=torch.float32, device=device)

    A = x
    B = w_int8
    C = torch.zeros((args.M, args.top_k, args.N), dtype=dtype, device=device)
    compute_type = tl_compute_type(dtype)

    results = []
    for cfg in CANDIDATE_CONFIGS:
        block_size_m = cfg["BLOCK_SIZE_M"]
        if args.group_size % cfg["BLOCK_SIZE_K"] != 0:
            print(f"SKIP {cfg} -- BLOCK_SIZE_K={cfg['BLOCK_SIZE_K']} does not divide "
                  f"group_size={args.group_size}, invalid per fused_moe_triton.py's own assert")
            continue

        sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, args.E
        )

        def run_kernel():
            invoke_fused_moe_kernel(
                A, B, C,
                None, w_scale,
                topk_weights, topk_ids,
                sorted_ids, expert_ids, num_tokens_post_padded,
                False, args.top_k, cfg, compute_type,
                use_fp8_w8a8=False, use_int8_w8a16=True,
                quant_group_size=args.group_size,
            )

        try:
            # Compile/first call -- discarded from timing, matches the
            # existing smoke test's own warmup convention.
            for _ in range(args.warmup):
                run_kernel()
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 -- record and move to the next config, don't abort the whole sweep
            print(f"FAIL {cfg}: {e!r}")
            continue

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        import time
        host_t0 = time.perf_counter()
        start_evt.record()
        for _ in range(args.trials):
            run_kernel()
        end_evt.record()
        torch.cuda.synchronize()
        host_ms = (time.perf_counter() - host_t0) / args.trials * 1000
        device_ms = start_evt.elapsed_time(end_evt) / args.trials

        ratio = host_ms / device_ms if device_ms > 0 else float("nan")
        flag = "  <-- host/device disagree >20%, treat with suspicion" if abs(ratio - 1.0) > 0.2 else ""
        results.append((cfg, device_ms, host_ms))
        print(f"BLOCK_M={block_size_m:3d} BLOCK_N={cfg['BLOCK_SIZE_N']:3d} BLOCK_K={cfg['BLOCK_SIZE_K']:3d} "
              f"GROUP_M={cfg['GROUP_SIZE_M']} warps={cfg['num_warps']} stages={cfg['num_stages']}  "
              f"device={device_ms:7.4f}ms  host={host_ms:7.4f}ms{flag}")

    if not results:
        print("\nNo configs completed successfully.")
        return

    results.sort(key=lambda r: r[1])
    baseline_device_ms = next((r[1] for r in results if r[0] == CANDIDATE_CONFIGS[0]), None)
    print(f"\n{'='*70}\nRANKED (fastest device-time first)\n{'='*70}")
    for rank, (cfg, device_ms, host_ms) in enumerate(results, 1):
        speedup = f"  ({baseline_device_ms/device_ms:.2f}x vs. current default)" if baseline_device_ms else ""
        print(f"{rank:2d}. device={device_ms:7.4f}ms  {cfg}{speedup}")

    best_cfg, best_ms, _ = results[0]
    print(f"\nBest config: {best_cfg}")
    if baseline_device_ms and best_ms < baseline_device_ms:
        print(f"Improvement over current default: {baseline_device_ms/best_ms:.2f}x faster (device time)")
        print("Before adopting: this is an isolated single-GEMM microbenchmark, same caveat as "
              "smoke_test_fused_moe_triton.py -- confirm the win survives inside the real engine "
              "(full decode step, CUDA graph capture) before changing _DEFAULT_CONFIG in "
              "layers/fused_moe_int8.py.")
    else:
        print("Current default was already the fastest (or tied) among the configs tried here.")


if __name__ == "__main__":
    main()
