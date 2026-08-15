"""A1 worker: attempts ONE (checkpoint, tensor_parallel_size) construction
and reports whether it fits. Always run through cluster_a1_memory_probe.py
(the orchestrator), not directly -- see that file's module docstring for why
this is a separate process per attempt (a failed/OOM'd config must not leave
GPU memory or a stuck NCCL process group behind for the NEXT config to trip
over) and for the exact command sequence.

What "fits" means here, in order:
  1. LLMEngine construction succeeds (weight load, StateManager, KV-cache
     budget calc, warmup -- everything ModelRunner.__init__ does) without
     raising (OOM surfaces here as a CUDA OutOfMemoryError, or as an
     AssertionError from allocate_kv_cache's `num_kvcache_blocks > 0` check
     if construction technically succeeds but leaves no room for any KV
     cache at all).
  2. A tiny real generate() call (1 short prompt, 4 tokens, greedy) completes
     without raising -- catches "constructs but the first real step crashes"
     separately from "never even constructed", per this task's "fail loudly
     rather than silently degrading" requirement. A config that constructs
     but can't actually run a request is NOT a pass.

Peak memory is read from nvidia-smi (--query-gpu=memory.used), captured
ONCE, right after the smoke-test step, before engine.exit() frees anything
-- this is per-GPU-INDEX usage, which lines up 1:1 with per-RANK usage
because ModelRunner.__init__ (engine/model_runner.py) pins rank i to CUDA
device i via torch.cuda.set_device(rank) and nothing else on the box should
be sharing these GPUs during a cluster-day run. This is deliberately used
INSTEAD of parsing this process's own torch.cuda.memory_stats() for every
rank: rank>0 processes are separate OS processes (torch.multiprocessing
spawn, see LLMEngine.__init__) with their own independent CUDA contexts --
nvidia-smi is the one vantage point that sees all of them at once without
needing a purpose-built cross-process reporting channel. Rank0's own
torch-tracked stats are ALSO captured as a cross-reference (allocated bytes
specifically, excluding reserved/fragmentation/CUDA-context overhead that
nvidia-smi's number includes) -- the two numbers answering different
questions, both kept, neither silently preferred.

Usage (called by the orchestrator; can also be run standalone for one config):
    python tests/cluster_a1_probe_worker.py --tensor-parallel-size 4 --out /tmp/tp4.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

DEFAULT_CKPT = os.path.join(ROOT, "qwen35_checkpoint")


def _nvidia_smi_snapshot():
    """[{"index": int, "used_mib": int, "total_mib": int}, ...] or None (with
    a printed reason) if nvidia-smi isn't reachable -- never raises, this is
    a best-effort diagnostic, not something that should mask the real
    construction result above it."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=30, text=True,
        )
    except Exception as e:  # noqa: BLE001 -- diagnostic only, never fatal
        print(f"[a1-worker] WARNING: nvidia-smi snapshot failed: {e!r}")
        return None
    rows = []
    for line in out.strip().splitlines():
        idx, used, total = [p.strip() for p in line.split(",")]
        rows.append({"index": int(idx), "used_mib": int(used), "total_mib": int(total)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tensor-parallel-size", type=int, required=True,
                     help="Also sets ep_size implicitly -- Qwen35MoE shards experts across "
                          "dist.get_world_size() directly, same process group as TP (see "
                          "models/qwen3_5.py); there is no separate --ep-size flag in this codebase.")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--out", required=True, help="Path to write this attempt's RESULT_JSON.")
    args = ap.parse_args()

    result = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "checkpoint": args.checkpoint,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "construct_ok": False,
        "smoke_generate_ok": False,
        "construction_s": None,
        "error_stage": None,
        "error_type": None,
        "error_message": None,
        "nvidia_smi_snapshot": None,
        "rank0_torch_peak_allocated_bytes": None,
    }

    engine = None
    try:
        assert os.path.isdir(args.checkpoint), f"checkpoint dir not found: {args.checkpoint}"
        print(f"[a1-worker] constructing LLMEngine(tensor_parallel_size={args.tensor_parallel_size}) "
              f"from {args.checkpoint} ...")
        from nanovllm.llm import LLM
        from nanovllm.sampling_params import SamplingParams

        t0 = time.perf_counter()
        engine = LLM(
            args.checkpoint,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=True,  # graph capture adds memory + time this probe doesn't need
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
        )
        result["construction_s"] = time.perf_counter() - t0
        result["construct_ok"] = True
        print(f"[a1-worker] construction OK in {result['construction_s']:.1f}s")

        import torch
        result["rank0_torch_peak_allocated_bytes"] = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

        print("[a1-worker] running smoke generate() (1 prompt, 4 tokens) ...")
        sp = SamplingParams(temperature=0, max_tokens=4)
        out = engine.generate(["The capital of France is"], sp, use_tqdm=False)
        assert out and out[0]["token_ids"], "smoke generate() returned no tokens"
        result["smoke_generate_ok"] = True
        print(f"[a1-worker] smoke generate() OK -- {len(out[0]['token_ids'])} token(s) produced")

    except Exception as e:  # noqa: BLE001 -- report every failure mode, never mask it
        stage = "smoke_generate" if result["construct_ok"] else "construct"
        result["error_stage"] = stage
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)[:2000]
        print(f"[a1-worker] FAILED at stage={stage}: {type(e).__name__}: {e}")
        traceback.print_exc()

    finally:
        # Snapshot memory BEFORE freeing anything -- this is the number that
        # answers "does it fit", not whatever's left after cleanup.
        result["nvidia_smi_snapshot"] = _nvidia_smi_snapshot()
        if engine is not None:
            try:
                import atexit
                atexit.unregister(engine.exit)  # see test_qwen35_multiblock.py's identical fix
                engine.exit()
            except Exception as e:  # noqa: BLE001 -- cleanup failure must not hide the real result above
                print(f"[a1-worker] WARNING: engine.exit() raised during cleanup: {e!r}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[a1-worker] RESULT_JSON: {json.dumps(result)}")

    passed = result["construct_ok"] and result["smoke_generate_ok"]
    print(f"[a1-worker] {'PASS' if passed else 'FAIL'} -- tensor_parallel_size={args.tensor_parallel_size}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
