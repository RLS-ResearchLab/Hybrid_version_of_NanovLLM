"""Ablation: does promoting RowParallelLinear's all_reduce (input cast +
accumulation only -- the matmul itself stays bf16) to fp32 resolve the
residual bf16 TP relative error, isolated from MoE routing entirely via a
direct Qwen35SharedExpert(x) comparison (dense world_size=1 vs TP
world_size=2, same weights, same input, bf16 throughout)?

Two measurements:
  (baseline) current RowParallelLinear -- all_reduce operates on the bf16
    matmul output directly.
  (variant a) layers.linear._FP32_ALLREDUCE = True -- y cast to fp32 before
    all_reduce, summed in fp32, cast back to bf16 after. F.linear (the
    matmul) is completely untouched by this flag either way.

If variant (a) drops relative error to ~0 / fp32-floor: that's the fix --
promote unconditionally in RowParallelLinear.forward(), matching the MoE
combine fix's established pattern, then re-verify with the full regression
suite. PASS.

If variant (a) doesn't fully resolve it: the residual comes from the bf16
matmul's own internal accumulation (F.linear with bf16 inputs), a separate,
harder problem -- report the residual precisely. Still TODO, narrowed.

CPU caveat: this runs on CPU (gloo, no GPU here). NVIDIA tensor cores
natively accumulate bf16xbf16 GEMMs in fp32 by hardware design; PyTorch's
CPU bf16 matmul path may behave differently. This measurement establishes
whether the all_reduce step is or isn't the dominant error source; if the
answer is "not fully resolved," the absolute residual should be re-measured
on the real GPU target before treating it as the deployment answer.

Usage: python tests/test_shared_expert_allreduce_precision.py
"""
import os
import sys
import tempfile
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [ROOT]
    _pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg

HIDDEN = 6
SHARED_INTERMEDIATE = 8  # divisible by TP_SIZE
N_TOKENS = 8
TP_SIZE = 2


def _relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return ((a.float() - b.float()).abs() / (b.float().abs() + eps)).max().item()


def build_reference():
    dist.init_process_group("gloo", init_method="tcp://localhost:29901", rank=0, world_size=1)

    from nanovllm.models.qwen3_5 import Qwen35SharedExpert

    ref = Qwen35SharedExpert(HIDDEN, SHARED_INTERMEDIATE)
    torch.manual_seed(1234)
    for name in sorted(dict(ref.named_parameters()).keys()):
        p = dict(ref.named_parameters())[name]
        p.data.normal_(mean=0.0, std=0.02)
    ref = ref.to(torch.bfloat16)

    x = torch.stack([(i + 1) * torch.ones(HIDDEN) for i in range(N_TOKENS)], dim=0).to(torch.bfloat16)
    ref_out = ref(x)

    ref_state = {k: v.clone() for k, v in ref.state_dict().items()}
    dist.destroy_process_group()
    return ref_state, x, ref_out


def worker(rank: int, tmp_path: str, results: list):
    dist.init_process_group("gloo", init_method="tcp://localhost:29902", rank=rank, world_size=TP_SIZE)

    from nanovllm.models.qwen3_5 import Qwen35SharedExpert
    import nanovllm.layers.linear as linear_mod

    payload = torch.load(tmp_path, weights_only=True)
    ref_state = payload["ref_state"]
    x = payload["x"]
    ref_out = payload["ref_out"]

    local = Qwen35SharedExpert(HIDDEN, SHARED_INTERMEDIATE).to(torch.bfloat16)
    gate_half, up_half = ref_state["gate_up_proj.weight"].chunk(2, dim=0)
    local.gate_up_proj.weight.weight_loader(local.gate_up_proj.weight, gate_half, 0)
    local.gate_up_proj.weight.weight_loader(local.gate_up_proj.weight, up_half, 1)
    local.down_proj.weight.weight_loader(local.down_proj.weight, ref_state["down_proj.weight"])

    assert linear_mod._FP32_ALLREDUCE is False
    baseline_out = local(x)
    max_abs_baseline = (baseline_out - ref_out).abs().max().item()
    rel_baseline = _relative_error(baseline_out, ref_out)
    bitwise_baseline = torch.equal(baseline_out, ref_out)
    print(f"[rank{rank}] BASELINE (bf16 all_reduce): max abs diff={max_abs_baseline:.3e} "
          f"relative error={rel_baseline:.3e} bitwise={bitwise_baseline}")

    linear_mod._FP32_ALLREDUCE = True
    variant_a_out = local(x)
    linear_mod._FP32_ALLREDUCE = False
    max_abs_a = (variant_a_out - ref_out).abs().max().item()
    rel_a = _relative_error(variant_a_out, ref_out)
    bitwise_a = torch.equal(variant_a_out, ref_out)
    print(f"[rank{rank}] VARIANT (a) (fp32 all_reduce): max abs diff={max_abs_a:.3e} "
          f"relative error={rel_a:.3e} bitwise={bitwise_a}")

    results.append((rank, rel_baseline, rel_a, bitwise_a))
    dist.destroy_process_group()


def main():
    ref_state, x, ref_out = build_reference()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save({"ref_state": ref_state, "x": x, "ref_out": ref_out}, tmp_path)

    try:
        manager = mp.Manager()
        results = manager.list()
        mp.spawn(worker, args=(tmp_path, results), nprocs=TP_SIZE, join=True)
    finally:
        os.unlink(tmp_path)

    print()
    for rank, rel_baseline, rel_a, bitwise_a in results:
        print(f"[rank{rank}] SUMMARY: baseline rel_err={rel_baseline:.3e} -> "
              f"variant(a) rel_err={rel_a:.3e} (bitwise={bitwise_a})")

    verdict_pass = all(rel_a < 1e-4 for _, _, rel_a, _ in results)
    print(f"\nVERDICT: {'PASS -- variant (a) resolves it' if verdict_pass else 'NOT FULLY RESOLVED -- residual from bf16 matmul accumulation'}")


if __name__ == "__main__":
    main()
