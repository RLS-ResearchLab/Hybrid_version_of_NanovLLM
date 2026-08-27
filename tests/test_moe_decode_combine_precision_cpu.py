"""B1 -- quantify the decode-path MoE combine precision gap.

models/qwen3_5.py's _forward_gathered (the ep_size=1 DECODE path, run for
every decode token of the shipping config) ends with:

    out = (out_e * w.unsqueeze(-1)).sum(dim=1)     # out_e, w both bf16

Its own comment flags this as "NOT promoted to fp32 -- same unpromoted
bf16-sum-across-top_k pattern that measurably caused ~1.3% relative error"
in _forward_dispatch before that path's fp32-promote fix -- but says the
decode path "wasn't ablation-tested". This is that ablation.

_forward_dispatch's fix comment claims: for bf16 inputs, top_k=8, promoting
ONLY the combine to fp32 (expert FFN matmuls stay bf16) takes the
routed-expert output from ~1.3% relative error to bitwise-exact, because
8 bf16 values (~8 mantissa bits) sum exactly in an fp32 (24-bit) accumulator.

This test reproduces _forward_gathered's routed-expert math on a REAL
Qwen35MoE (bf16), then combines three ways:
  - CURRENT:  (out_e * w[...,None]).sum(1)               -- bf16 throughout
  - FP32-FIX: (out_e.float() * w.float()[...,None]).sum(1).to(bf16)
  - REF:      (out_e.double() * w.double()[...,None]).sum(1)   -- ground truth
and reports the relative error of CURRENT vs REF and FP32-FIX vs REF.

Pure CPU. Run with TORCHDYNAMO_DISABLE=1.

Usage:
    TORCHDYNAMO_DISABLE=1 python tests/test_moe_decode_combine_precision_cpu.py
"""
import os
import sys
import types

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.distributed as dist  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [_ROOT]
    _pkg.__file__ = os.path.join(_ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _rel_err(a, b_ref):
    """||a - b_ref|| / ||b_ref||, both flattened, computed in fp64."""
    a, b_ref = a.double().reshape(-1), b_ref.double().reshape(-1)
    return (a - b_ref).norm().item() / b_ref.norm().clamp_min(1e-12).item()


def main():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29553")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")

    from nanovllm.models.qwen3_5 import Qwen35MoE

    # Real checkpoint proportions: top_k=8 of num_experts, moe_intermediate
    # 1/4 of hidden. Smaller absolute dims to keep it a CPU test.
    H, MI, NE, TK = 512, 128, 64, 8
    torch.manual_seed(0)
    moe = Qwen35MoE(hidden_size=H, intermediate_size=MI, shared_intermediate_size=MI,
                    num_experts=NE, top_k=TK).to(torch.bfloat16)
    with torch.no_grad():
        moe.experts.gate_up_proj.normal_(0, 0.02)
        moe.experts.down_proj.normal_(0, 0.02)
        for p in moe.gate.parameters():
            p.normal_(0, 0.02)

    print("=" * 70)
    print("B1 -- decode MoE combine: bf16 vs fp32 vs fp64-ref, real Qwen35MoE")
    print("=" * 70)
    print(f"dims: hidden={H} moe_intermediate={MI} num_experts={NE} top_k={TK}\n")

    worst_cur_vs_ref = 0.0
    worst_fix_vs_ref = 0.0
    worst_cur_vs_fix = 0.0
    # "gate_temp" scales the gate logits: 1.0 = as-routed, >1 = peakier
    # (top-1 dominates, the other 7 contributions are tiny -- the regime
    # where a bf16 sum loses the small terms).
    for N in (1, 8, 64):
        for gate_temp in (1.0, 4.0):
            torch.manual_seed(100 + N)
            x = (torch.randn(N, H) * 0.1).to(torch.bfloat16)

            with torch.no_grad():
                wv, idx = torch.topk(moe.gate(x) * gate_temp, TK, dim=-1)   # (N, TK)
                wv = F.softmax(wv, dim=-1).to(x.dtype)                      # bf16, like the code
                gate_up = moe.experts.gate_up_proj[idx]
                down = moe.experts.down_proj[idx]
                gw, uw = gate_up.chunk(2, dim=2)
                h = F.silu(torch.einsum('nkmh,nh->nkm', gw, x)) * torch.einsum('nkmh,nh->nkm', uw, x)
                out_e = torch.einsum('nkhm,nkm->nkh', down, h)   # (N, TK, H) bf16 -- SAME in all cases

                out_current = (out_e * wv.unsqueeze(-1)).sum(dim=1)                        # bf16
                out_fixed = (out_e.float() * wv.float().unsqueeze(-1)).sum(dim=1).to(x.dtype)
                out_ref = (out_e.double() * wv.double().unsqueeze(-1)).sum(dim=1)          # fp64

            e_cur = _rel_err(out_current, out_ref)
            e_fix = _rel_err(out_fixed, out_ref)
            e_cvf = _rel_err(out_current.to(x.dtype), out_fixed)  # does the fix change the SHIPPED bf16 answer?
            worst_cur_vs_ref = max(worst_cur_vs_ref, e_cur)
            worst_fix_vs_ref = max(worst_fix_vs_ref, e_fix)
            worst_cur_vs_fix = max(worst_cur_vs_fix, e_cvf)
            print(f"N={N:3d} gate_temp={gate_temp:>3}  "
                  f"CURRENT-vs-ref={e_cur:.4%}  FP32FIX-vs-ref={e_fix:.4%}  "
                  f"CURRENT-vs-FP32FIX={e_cvf:.4%}")

    print("\n" + "-" * 70)
    print(f"WORST  CURRENT vs fp64-ref : {worst_cur_vs_ref:.4%}")
    print(f"WORST  FP32-FIX vs fp64-ref: {worst_fix_vs_ref:.4%}   "
          f"(floor -- bf16 output cast of an exact sum of bf16 out_e)")
    print(f"WORST  CURRENT vs FP32-FIX : {worst_cur_vs_fix:.4%}   "
          f"<- how much the one-line fix actually changes the shipped answer")
    print()
    delta = worst_cur_vs_ref - worst_fix_vs_ref
    print("READ:")
    print(f"  - The bf16 combine's error above the fp32-fix floor is ~{delta:.3%} "
          f"(rounding out_e*w products to bf16 before .sum).")
    print(f"  - torch.sum() over a reduction dim already accumulates bf16 in an fp32 "
          f"accumulator (CPU and CUDA both use acc_type=float), so the historical "
          f"_forward_dispatch '~1.3%' came from its pre-fix code ALSO writing bf16 into "
          f"the per-expert buffer -- .sum(dim=1) here doesn't do that.")
    if delta < 0.003 and worst_cur_vs_fix < 0.003:
        print(f"\nVERDICT: LOW severity. The fp32 fix moves the shipped bf16 answer by "
              f"< {max(delta, worst_cur_vs_fix):.2%} -- real but small, no argmax-scale "
              f"concern from this alone. RECOMMEND: apply the one-line fp32 promote anyway "
              f"(strictly better, ~free, matches _forward_dispatch/_forward_gathered_ep's "
              f"own discipline and removes a standing 'known issue' comment) + GSM8K "
              f"non-regression on GPU. Not urgent; not the 1.3% the comment implied.")
    else:
        print(f"\nVERDICT: the fp32 fix changes the shipped answer by up to "
              f"{max(delta, worst_cur_vs_fix):.2%} -- material. FIX IT + GSM8K on GPU.")
    sys.exit(0)


if __name__ == "__main__":
    main()
