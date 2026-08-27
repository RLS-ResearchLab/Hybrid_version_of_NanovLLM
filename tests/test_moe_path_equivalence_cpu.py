"""B2 + B4 -- cross-path equivalence for Qwen35MoE's forward variants.

B4: _forward_gathered (ep_size=1 DECODE path, capture-safe) vs
    _forward_dispatch (PREFILL path) -- same top-k routing + expert FFN +
    weighted combine + shared expert, one via advanced-indexing einsum, the
    other via a sort-by-expert Python loop. The code comment on
    _forward_gathered's combine said it "wasn't ablation-tested"; this is that.

B2: _forward_dispatch_vectorized's INT8 branch (models/qwen3_5.py, bug #1's
    fix from the H100 window -- dequantizes the FULL expert table then
    torch._grouped_mm) vs _forward_dispatch's INT8 branch (per-expert dequant
    in the loop) vs the bf16 reference. torch._grouped_mm works on CPU at
    realistic dims, so this is checkable here.

Both use a real Qwen35MoE (bf16). Threshold: these paths reassociate the
K-dim reduction differently (einsum vs loop vs grouped_mm), so cosine, not
torch.equal -- 0.999 bar, matching tests/test_qwen35_vectorized_moe.py.

Pure CPU. Run with TORCHDYNAMO_DISABLE=1.

Usage:
    TORCHDYNAMO_DISABLE=1 python tests/test_moe_path_equivalence_cpu.py
"""
import copy
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

H, MI, NE, TK, GS = 256, 256, 16, 8, 128


def _cos(a, b):
    return F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def _report(name, a, ref, bar=0.999):
    c = _cos(a, ref)
    m = (a.float() - ref.float()).abs().max().item()
    r = (a.float() - ref.float()).norm().item() / ref.float().norm().clamp_min(1e-9).item()
    ok = c > bar
    print(f"  {name:38s} cosine={c:.6f}  rel_err={r:.4%}  max_abs={m:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


def _build_moe(seed=0):
    from nanovllm.models.qwen3_5 import Qwen35MoE
    torch.manual_seed(seed)
    moe = Qwen35MoE(hidden_size=H, intermediate_size=MI, shared_intermediate_size=MI,
                    num_experts=NE, top_k=TK).to(torch.bfloat16)
    with torch.no_grad():
        for _, p in moe.named_parameters():
            p.normal_(0, 0.02)
    return moe.eval()


def main():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29561")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")

    from moe_int8_integration import quantize_experts_module_inplace

    print("=" * 70)
    print("B2 + B4 -- Qwen35MoE cross-path equivalence")
    print("=" * 70)
    print(f"dims: hidden={H} moe_intermediate={MI} num_experts={NE} top_k={TK} int8_group={GS}\n")

    moe = _build_moe()
    moe_int8 = copy.deepcopy(moe)
    quantize_experts_module_inplace(moe_int8.experts, GS)

    ok = True
    for N in (1, 8, 32):
        torch.manual_seed(500 + N)
        x = (torch.randn(N, H) * 0.1).to(torch.bfloat16)

        with torch.no_grad():
            ref_bf16 = moe._forward_dispatch(x)                 # bf16, the reference

            # ---- B4: gathered (decode) vs dispatch (prefill), bf16 ----
            gathered_bf16 = moe._forward_gathered(x)

            # ---- B2: int8 paths ----
            disp_int8 = moe_int8._forward_dispatch(x)           # per-expert dequant loop
            vect_int8 = moe_int8._forward_dispatch_vectorized(x)  # full dequant + grouped_mm (bug #1 fix)
            gath_int8 = moe_int8._forward_gathered(x)           # decode int8 plain-dequant branch

        print(f"N={N}:")
        ok &= _report("[B4] _forward_gathered vs _dispatch (bf16)", gathered_bf16, ref_bf16)
        ok &= _report("[B2] _dispatch INT8 vs bf16 ref", disp_int8, ref_bf16, bar=0.995)
        ok &= _report("[B2] _dispatch_vectorized INT8 vs _dispatch INT8", vect_int8, disp_int8)
        ok &= _report("[B2] _dispatch_vectorized INT8 vs bf16 ref", vect_int8, ref_bf16, bar=0.995)
        ok &= _report("[B2] _forward_gathered INT8 vs bf16 ref", gath_int8, ref_bf16, bar=0.995)
        print()

    print("-" * 70)
    if ok:
        print("PASS -- all MoE forward paths agree within the reassociation bar.")
        print("  B4: the decode _forward_gathered path computes the same function as the")
        print("      prefill _forward_dispatch (cosine > 0.999) -- the 'not ablation-tested'")
        print("      comment is now covered.")
        print("  B2: bug #1's _forward_dispatch_vectorized INT8 branch matches the per-expert")
        print("      _forward_dispatch INT8 branch (cosine > 0.999) and both track the bf16")
        print("      reference (cosine > 0.995 -- INT8 weight-quant error, not a path bug).")
    else:
        print("FAIL -- a path diverged beyond the reassociation bar. Investigate above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
