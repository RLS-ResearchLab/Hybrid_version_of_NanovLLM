"""A1/A2/A4 -- correctness of the Triton-free batched GDR decode path
(`Qwen35LinearAttention._forward_decode_batched`, models/qwen3_5.py), gated
by `use_batched_gdr_decode`.

WHY THIS MATTERS: at decode, forward()'s `for i in range(num_segments)` loop
runs one iteration per in-flight request, on 30 of 40 layers. Under CUDA
graph capture that unrolls into ~30k tiny kernel launches per decode step
(concurrency 64 x 30 layers x ~15 ops) -- the single biggest reason the
H100 decode step measured ~310 ms when the hardware should do ~15-40 ms.
_forward_decode_batched replaces that loop with one set of batched tensor
ops. Unlike the fla chunked kernel (`use_fused_gdr_kernel`), it does NOT
reassociate any reduction, so it should match the sequential scan bitwise on
CPU -- this file's primary check.

Compares, at N in {1, 8, 64}, same weights, same input, non-zero incoming
state + conv history:
  1. use_batched_gdr_decode=True  vs  False  -- output, new recurrent state,
     new conv state.
  2. A 16-step compounding run (A4): state carried decode step to decode
     step, batched vs sequential, cosine must not drift downward.

Threshold: bitwise-exact at N=1 (no batch dimension -> no matmul
reassociation anywhere). At N>1 the only difference source is the small
in_proj_*/out_proj GEMMs reassociating across batch size on the BLAS
backend -- measured max_abs_diff ~1e-8..1e-7, cosine ~1.0 on CPU fp32. A
hard torch.equal bar is right for N=1; for N>1 the bar is
cosine > 0.99999 AND max_abs_diff < 1e-4 (generous headroom over what was
measured), plus the near-tie argmax check borrowed from
test_qwen35_fused_gdr.py.

Pure CPU -- constructs Qwen35LinearAttention directly, no full model, no
attention stub needed. Run with TORCHDYNAMO_DISABLE=1 (Qwen35RMSNormGated
is not @torch.compile, but keep the convention).

Usage:
    TORCHDYNAMO_DISABLE=1 python tests/test_qwen35_gdr_decode_batched.py
"""
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import types  # noqa: E402
if "nanovllm" not in sys.modules:
    pkg = types.ModuleType("nanovllm")
    pkg.__path__ = [ROOT]
    pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = pkg

from test_utils import init_model_weights_with_norms_  # noqa: E402
from test_qwen35_fused_gdr import _assert_argmax_match_or_near_tie  # noqa: E402

H, LKH, LVH, LHD, CK, EPS = 256, 4, 8, 32, 4, 1e-6


def _init_dist():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29533")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")


def _build(flag):
    from nanovllm.models.qwen3_5 import Qwen35LinearAttention
    torch.manual_seed(0)
    m = Qwen35LinearAttention(
        hidden_size=H, linear_attn_kq_heads=LKH, linear_attn_v_heads=LVH,
        linear_attn_head_dim=LHD, conv_kernel_size=CK, rms_norm_eps=EPS,
        use_batched_gdr_decode=flag,
    )
    init_model_weights_with_norms_(m, seed=1)
    return m.to(torch.float32).eval()


def _pair():
    seq_m = _build(False)
    bat_m = _build(True)
    bat_m.load_state_dict(seq_m.state_dict())
    return seq_m, bat_m


def _decode(m, hs, states, conv_states):
    N = hs.shape[0]
    cu = torch.arange(0, N + 1, dtype=torch.int32)
    with torch.no_grad():
        return m(hs, cu, states=states, conv_states=conv_states)


def _check(a, b, name, n, *, exact):
    d = (a.float() - b.float()).abs().max().item()
    cos = F.cosine_similarity(a.float().flatten().unsqueeze(0),
                              b.float().flatten().unsqueeze(0), dim=-1).item()
    ok_exact = torch.equal(a, b)
    print(f"    N={n:2d} {name:10s} max_abs_diff={d:.3e}  cosine={cos:.10f}  exact={ok_exact}")
    if exact:
        assert ok_exact, f"N={n} {name}: expected bitwise-exact at N=1, got max_abs_diff={d:.3e}"
    else:
        assert cos > 0.99999 and d < 1e-4, (
            f"N={n} {name}: cosine={cos:.8f} max_abs_diff={d:.3e} -- exceeds the "
            f"benign-reassociation bar (cos>0.99999, diff<1e-4)"
        )


def test_single_step():
    print("\n[A1/A2] single decode step -- batched vs sequential")
    seq_m, bat_m = _pair()
    for n in (1, 8, 64):
        torch.manual_seed(100 + n)
        hs = torch.randn(n, H)
        st = torch.randn(n, LVH, LHD, LHD)
        cv = torch.randn(n, seq_m.qkv_dim, CK - 1)
        y0, s0, c0 = _decode(seq_m, hs, st.clone(), cv.clone())
        y1, s1, c1 = _decode(bat_m, hs, st.clone(), cv.clone())
        _check(y0, y1, "output", n, exact=(n == 1))
        _check(s0, s1, "new_state", n, exact=(n == 1))
        _check(c0, c1, "new_conv", n, exact=True)  # pure conv1d, no matmul reassoc
        _assert_argmax_match_or_near_tie(
            y0.reshape(n, -1), y1.reshape(n, -1), f"output(N={n})", n
        )
    print("  [PASS] single-step equivalence")


def test_none_state():
    """StateManager always supplies state/conv at decode, but the fallback
    (None -> zeros) must still match the sequential path's own None branch."""
    print("\n[A1] None incoming state/conv -> zeros fallback")
    seq_m, bat_m = _pair()
    n = 8
    torch.manual_seed(7)
    hs = torch.randn(n, H)
    y0, s0, c0 = _decode(seq_m, hs, None, None)
    y1, s1, c1 = _decode(bat_m, hs, None, None)
    _check(y0, y1, "output", n, exact=False)
    _check(s0, s1, "new_state", n, exact=False)
    _check(c0, c1, "new_conv", n, exact=True)
    print("  [PASS] None-state fallback matches")


def test_compounding_16_steps():
    print("\n[A4] 16 consecutive decode steps, state carried, batched vs sequential")
    seq_m, bat_m = _pair()
    n = 16
    torch.manual_seed(2024)
    st = torch.zeros(n, LVH, LHD, LHD)
    cv = torch.zeros(n, seq_m.qkv_dim, CK - 1)
    st0, cv0 = st.clone(), cv.clone()
    st1, cv1 = st.clone(), cv.clone()
    first_cos = last_cos = None
    for step in range(16):
        torch.manual_seed(5000 + step)
        hs = torch.randn(n, H)
        y0, st0, cv0 = _decode(seq_m, hs, st0, cv0)
        y1, st1, cv1 = _decode(bat_m, hs, st1, cv1)
        cos = F.cosine_similarity(y0.float().flatten().unsqueeze(0),
                                  y1.float().flatten().unsqueeze(0), dim=-1).item()
        sd = (st0.float() - st1.float()).abs().max().item()
        if step == 0:
            first_cos = cos
        last_cos = cos
        print(f"    step {step:2d}: output cosine={cos:.10f}  state max_abs_diff={sd:.3e}")
        assert cos > 0.9999, f"step {step}: cosine {cos:.8f} dropped below 0.9999"
    assert last_cos >= first_cos - 1e-5, (
        f"cosine drifted downward over 16 steps: {first_cos:.8f} -> {last_cos:.8f} "
        f"(compounding error -- investigate before trusting long generations)"
    )
    print(f"  [PASS] no downward drift over 16 steps ({first_cos:.8f} -> {last_cos:.8f})")


def main():
    _init_dist()
    print("=" * 70)
    print("Batched Triton-free GDR decode path -- correctness vs sequential scan")
    print("=" * 70)
    test_single_step()
    test_none_state()
    test_compounding_16_steps()
    print("\nALL BATCHED-GDR-DECODE CHECKS PASSED")


if __name__ == "__main__":
    main()
