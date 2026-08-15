"""A1 -- MEMORY PROBE. RUN THIS FIRST ON THE CLUSTER. Every other cluster-day
script (A2-A5) depends on its answer: which tensor_parallel_size (== ep_size
in this codebase -- Qwen35MoE shards experts across the same process group TP
uses, see models/qwen3_5.py, there is no separate ep_size flag) the real
checkpoint can actually be constructed at, on the hardware that showed up
that day (assume 4xA6000, degrade to 2 if only 2 are available).

Gating context (from the task this script was written for, not re-derived
here): the last recorded dual-GPU attempt hit ~47.5/47.54 GB used on BOTH
cards at tensor_parallel_size=2 and OOM'd, BEFORE expert-parallel sharding
existed to shrink each rank's MoE weight footprint. EP sharding may make
tp=2 fit now, and may make tp=4 fit for the first time ever -- but "may" is
exactly what this script exists to replace with a real number. Do not assume
either config fits; do not skip this and go straight to A2.

WHY A SEPARATE PROCESS PER ATTEMPT (cluster_a1_probe_worker.py): an
OOM/crash inside torch.multiprocessing-spawned rank>0 processes can leave
them alive and still holding GPU memory, or stuck inside a NCCL collective
waiting for a rank0 that already died (see engine/llm_engine.py's own
exit()/atexit comments on exactly this failure mode) -- trying the NEXT
tensor_parallel_size in the SAME orchestrator process risks a fixed-port
(dist.init_process_group uses tcp://localhost:2333, engine/model_runner.py)
or fixed-SharedMemory-name ("nanovllm") collision with a still-alive leftover
from the previous attempt, or an artificially reduced free-memory reading.
Each attempt gets its own subprocess with a fresh process GROUP (POSIX
start_new_session / Windows CREATE_NEW_PROCESS_GROUP) so a timeout can kill
the WHOLE tree, not just the immediate child.

Results are written incrementally to --out (one JSON object appended per
attempt, flushed immediately) -- survivable if this script or the machine is
killed mid-sweep; a partial file still tells you what was already learned.

Usage (defaults: try tp=4 first, degrade to tp=2; ~15min timeout per attempt):
    python tests/cluster_a1_memory_probe.py --checkpoint /path/to/real/checkpoint

    # Only the configs actually available that day:
    python tests/cluster_a1_memory_probe.py --checkpoint /path/to/real/checkpoint --tp-sizes 2

Dry run (small model, single GPU -- see this task's cluster-day prep):
    python tests/cluster_a1_memory_probe.py --checkpoint tests/fake_qwen35_small --tp-sizes 1 \\
        --out tests/_cluster_day_cache/a1_dry_run.jsonl --timeout-s 300
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "tests", "cluster_a1_probe_worker.py")
DEFAULT_CKPT = os.path.join(ROOT, "qwen35_checkpoint")
DEFAULT_OUT = os.path.join(ROOT, "tests", "_cluster_day_cache", "a1_memory_probe.jsonl")


def _kill_process_tree(proc: subprocess.Popen):
    """Best-effort kill of the WHOLE process group a timed-out attempt
    started (worker.py's own torch.multiprocessing-spawned rank>0 children
    included), not just the immediate `proc` handle -- see module docstring
    for why a lingering rank>0 process is exactly the failure mode the
    next attempt must not inherit."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            # Windows: best-effort via taskkill /T (kills the tree rooted at
            # proc.pid). The real cluster is expected to be Linux (A6000/H200
            # boxes), where os.killpg above is what actually runs -- this
            # branch exists so a Windows dry run against the small model
            # doesn't hang forever on a stuck attempt, not as the primary path.
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                            capture_output=True, timeout=30)
    except Exception as e:  # noqa: BLE001 -- best-effort cleanup, never fatal
        print(f"[a1-probe] WARNING: failed to kill process tree for pid={proc.pid}: {e!r}")
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def run_one_attempt(tp_size: int, args, out_jsonl_fh) -> dict:
    print("\n" + "=" * 78)
    print(f"ATTEMPT: tensor_parallel_size={tp_size} (== ep_size, see module docstring)")
    print("=" * 78)

    with tempfile.NamedTemporaryFile(suffix=f"_tp{tp_size}.json", delete=False) as f:
        worker_out_path = f.name

    cmd = [
        sys.executable, WORKER,
        "--checkpoint", args.checkpoint,
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--out", worker_out_path,
    ]
    print(f"[a1-probe] launching: {' '.join(cmd)}")

    popen_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if hasattr(os, "setsid"):
        popen_kwargs["start_new_session"] = True
    elif sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, **popen_kwargs)
    timed_out = False
    stdout_text = ""
    try:
        stdout_text, _ = proc.communicate(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(f"[a1-probe] TIMEOUT after {args.timeout_s}s -- killing process tree "
              f"(likely a hang, e.g. a NCCL collective waiting on a rank that already "
              f"OOM'd -- see module docstring)")
        _kill_process_tree(proc)
        try:
            stdout_text, _ = proc.communicate(timeout=30)
        except Exception:
            stdout_text = stdout_text or "<no output captured before kill>"
    wall_s = time.perf_counter() - t0

    print(stdout_text)

    worker_result = None
    if os.path.exists(worker_out_path):
        try:
            with open(worker_out_path, encoding="utf-8") as f:
                worker_result = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[a1-probe] WARNING: could not parse worker output file: {e!r}")
        finally:
            os.unlink(worker_out_path)

    attempt = {
        "tensor_parallel_size": tp_size,
        "wall_s": wall_s,
        "timed_out": timed_out,
        "returncode": proc.returncode,
        "worker_result": worker_result,
        "passed": bool(worker_result and worker_result.get("construct_ok") and worker_result.get("smoke_generate_ok")),
    }

    out_jsonl_fh.write(json.dumps(attempt) + "\n")
    out_jsonl_fh.flush()

    verdict = "PASS" if attempt["passed"] else "FAIL"
    print(f"[a1-probe] ATTEMPT RESULT: tensor_parallel_size={tp_size} -- {verdict} "
          f"(wall={wall_s:.1f}s, timed_out={timed_out})")
    if worker_result and worker_result.get("nvidia_smi_snapshot"):
        for row in worker_result["nvidia_smi_snapshot"]:
            pct = 100 * row["used_mib"] / row["total_mib"] if row["total_mib"] else float("nan")
            print(f"    GPU{row['index']}: {row['used_mib']} / {row['total_mib']} MiB used ({pct:.1f}%)")
    return attempt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tp-sizes", type=int, nargs="+", default=[4, 2],
                     help="Tried IN ORDER, first-to-last. Default [4, 2]: try the best case "
                          "first, degrade gracefully. Pass --tp-sizes 2 alone if only 2 GPUs "
                          "showed up that day -- don't waste a timeout window probing tp=4 "
                          "against hardware that can't run it.")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--timeout-s", type=float, default=900.0,
                     help="Per-attempt wall-clock budget. Real checkpoint load + warmup at "
                          "these sizes is expected to take a few minutes; 900s leaves headroom "
                          "for a slow first-touch load from disk without letting a genuine hang "
                          "burn the whole cluster window.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"Results will be appended incrementally to: {args.out}")

    attempts = []
    with open(args.out, "a", encoding="utf-8") as out_fh:
        for tp_size in args.tp_sizes:
            attempts.append(run_one_attempt(tp_size, args, out_fh))

    print("\n" + "=" * 78)
    print("A1 MEMORY PROBE -- SUMMARY")
    print("=" * 78)
    print(f"{'tp_size':>8}  {'verdict':>7}  {'wall_s':>8}  {'construct_s':>12}  GPU used/total (MiB)")
    for a in attempts:
        wr = a["worker_result"] or {}
        construct_s = wr.get("construction_s")
        construct_s_str = f"{construct_s:.1f}" if construct_s is not None else "--"
        snap = wr.get("nvidia_smi_snapshot") or []
        mem_str = ", ".join(f"GPU{r['index']}={r['used_mib']}/{r['total_mib']}" for r in snap) or "n/a"
        print(f"{a['tensor_parallel_size']:>8}  {'PASS' if a['passed'] else 'FAIL':>7}  "
              f"{a['wall_s']:>8.1f}  {construct_s_str:>12}  {mem_str}")
        if not a["passed"] and wr.get("error_message"):
            print(f"           error ({wr.get('error_stage')}): {wr.get('error_type')}: {wr.get('error_message')[:300]}")

    passing = [a["tensor_parallel_size"] for a in attempts if a["passed"]]
    print()
    if not passing:
        print("VERDICT: NONE of the attempted tensor_parallel_size values fit. STOP -- do not "
              "proceed to A2/A3/A4/A5 against the real checkpoint. Options: lower "
              "--gpu-memory-utilization further, lower --max-num-seqs/--max-model-len, confirm "
              "no other process is holding GPU memory, or this needs a real memory-budget "
              "investigation before any more cluster time is spent.")
        sys.exit(1)
    else:
        best = max(passing)
        print(f"VERDICT: PASS at tensor_parallel_size in {sorted(passing)}. Highest viable: {best}.")
        print(f"-> Run A2/A3/A4/A5 at tensor_parallel_size={best}.", end="")
        skipped = [t for t in args.tp_sizes if t not in passing]
        if skipped:
            print(f" SKIP the {sorted(skipped)} config(s) for the rest of A2-A5 -- confirmed "
                  f"not to fit, don't re-attempt them on remaining cluster time.")
        else:
            print()
        sys.exit(0)


if __name__ == "__main__":
    main()
