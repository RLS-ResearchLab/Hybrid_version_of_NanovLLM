"""GPU-only: does torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
affect RowParallelLinear's residual TP relative error in Qwen35SharedExpert,
and does flipping it cost meaningfully more time? Same isolation
methodology as tests/test_shared_expert_allreduce_precision.py (dense
world_size=1 reference vs TP world_size=2, direct shared_expert(x)
comparison, bypassing MoE routing entirely) and the same bootstrap pattern
as run_construct.py -- just ported from CPU/gloo to real CUDA/nccl.

Step 1: measure with the flag at its default (True). This is NOT assumed
to reproduce the CPU-measured 5.917e-03 baseline -- GPU tensor cores'
default bf16 accumulation behavior may already differ from CPU's oneDNN/
native kernels, which is exactly what this step checks before anything
else is concluded.

Step 2: measure again with the flag set to False.

Both relative errors and both wall-clock per-call timings (rough signal
via a short warmup + N-iteration loop, NOT a rigorous benchmark) are
reported directly -- no interpretation baked into the numbers themselves.

Run from the repo root: python3 run_gpu_allreduce_precision_check.py
"""
import os
import sys
import tempfile
import time
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm" and "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

HIDDEN = 6
SHARED_INTERMEDIATE = 8  # divisible by TP_SIZE
N_TOKENS = 8
TP_SIZE = 2
N_WARMUP_ITERS = 10
N_TIMING_ITERS = 50
CPU_BASELINE_REL_ERR = 5.917e-03  # tests/test_shared_expert_allreduce_precision.py


def _relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return ((a.float() - b.float()).abs() / (b.float().abs() + eps)).max().item()


def build_reference():
    """Dense, single-GPU (world_size=1) reference -- same hardware/kernel
    path as the TP-sharded version, just without the split, so the only
    variable under test is dense-vs-sharded, not CPU-vs-GPU."""
    dist.init_process_group("nccl", init_method="tcp://localhost:29911", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from nanovllm.models.qwen3_5 import Qwen35SharedExpert

    ref = Qwen35SharedExpert(HIDDEN, SHARED_INTERMEDIATE).cuda()
    torch.manual_seed(1234)
    for name in sorted(dict(ref.named_parameters()).keys()):
        p = dict(ref.named_parameters())[name]
        p.data.normal_(mean=0.0, std=0.02)
    ref = ref.to(torch.bfloat16)

    x = torch.stack([(i + 1) * torch.ones(HIDDEN) for i in range(N_TOKENS)], dim=0).to(torch.bfloat16).cuda()
    ref_out = ref(x)

    ref_state = {k: v.clone().cpu() for k, v in ref.state_dict().items()}
    payload = {"ref_state": ref_state, "x": x.cpu(), "ref_out": ref_out.cpu()}
    dist.destroy_process_group()
    return payload


def worker(rank: int, tmp_path: str, results: list):
    dist.init_process_group("nccl", init_method="tcp://localhost:29912", rank=rank, world_size=TP_SIZE)
    torch.cuda.set_device(rank)

    from nanovllm.models.qwen3_5 import Qwen35SharedExpert

    payload = torch.load(tmp_path, weights_only=True)
    ref_state = payload["ref_state"]
    x = payload["x"].cuda()
    ref_out = payload["ref_out"].cuda()

    local = Qwen35SharedExpert(HIDDEN, SHARED_INTERMEDIATE).cuda().to(torch.bfloat16)
    gate_half, up_half = ref_state["gate_up_proj.weight"].cuda().chunk(2, dim=0)
    local.gate_up_proj.weight.weight_loader(local.gate_up_proj.weight, gate_half, 0)
    local.gate_up_proj.weight.weight_loader(local.gate_up_proj.weight, up_half, 1)
    local.down_proj.weight.weight_loader(local.down_proj.weight, ref_state["down_proj.weight"].cuda())

    def measure(flag_value: bool, label: str):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_value

        out = local(x)
        max_abs = (out - ref_out).abs().max().item()
        rel = _relative_error(out, ref_out)
        bitwise = torch.equal(out, ref_out)

        torch.cuda.synchronize()
        for _ in range(N_WARMUP_ITERS):
            _ = local(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_TIMING_ITERS):
            _ = local(x)
        torch.cuda.synchronize()
        per_call_ms = (time.perf_counter() - t0) / N_TIMING_ITERS * 1000

        print(f"[rank{rank}] {label}: max_abs_diff={max_abs:.3e}  relative_error={rel:.3e}  "
              f"bitwise={bitwise}  per_call_time={per_call_ms:.4f}ms "
              f"(avg over {N_TIMING_ITERS} calls, {N_WARMUP_ITERS} warmup)")
        return rel, per_call_ms

    rel_true, time_true = measure(True, "FLAG=True (default)")
    rel_false, time_false = measure(False, "FLAG=False")

    results.append((rank, rel_true, time_true, rel_false, time_false))
    dist.destroy_process_group()


def main():
    payload = build_reference()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save(payload, tmp_path)

    try:
        manager = mp.Manager()
        results = manager.list()
        mp.spawn(worker, args=(tmp_path, results), nprocs=TP_SIZE, join=True)
    finally:
        os.unlink(tmp_path)

    print()
    for rank, rel_true, time_true, rel_false, time_false in results:
        reproduces_cpu = 0.1 * CPU_BASELINE_REL_ERR < rel_true < 10 * CPU_BASELINE_REL_ERR
        print(f"[rank{rank}] SUMMARY:")
        print(f"  FLAG=True  (default): relative_error={rel_true:.3e}  per_call={time_true:.4f}ms")
        print(f"  FLAG=False           : relative_error={rel_false:.3e}  per_call={time_false:.4f}ms")
        print(f"  CPU baseline was:      relative_error={CPU_BASELINE_REL_ERR:.3e}")
        print(f"  GPU default reproduces CPU baseline (same order of magnitude)? "
              f"{'YES' if reproduces_cpu else 'NO -- differs'}")
        print(f"  Time delta (False vs True): {time_false - time_true:+.4f}ms "
              f"({(time_false / time_true - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    # Required: mp.spawn's "spawn" start method re-imports this file as
    # __main__ in each child process -- without this guard, children would
    # re-run main() (and thus mp.spawn) themselves, recursing before the
    # parent even finishes starting the first child. Same reasoning as
    # run_construct.py's identical guard.
    main()
