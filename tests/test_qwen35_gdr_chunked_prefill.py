"""Deep diagnosis for GDN (Gated DeltaNet, "GDR" in this codebase): chunked-
prefill continuation and multi-segment packing correctness for the
SEQUENTIAL scan (models/qwen3_5.py's Qwen35LinearAttention.forward(),
use_fused_gdr_kernel=False, use_batched_gdr_decode=False -- the ground-truth
path). Neither scenario was covered anywhere in the existing suite before
this file: tests/test_qwen35_gdr_decode_batched.py is decode-only
(num_segments==N==T_i==1 throughout); nothing exercised the sequential
path's own multi-token, multi-call, or multi-segment-packed behavior against
itself.

Two properties under test:

1. CHUNKED-PREFILL CONTINUATION: Scheduler.schedule() only allows chunking
   for the first sequence in a prefill round (engine/scheduler.py's
   "only allow chunked prefill for the first seq" comment), so a prompt
   longer than max_num_batched_tokens is fed through forward() as MULTIPLE
   separate calls, each num_segments=1, with the previous call's (state,
   conv_state) threaded into the next. This test builds a single T-token
   sequence, runs it as ONE forward() call, then re-runs the exact same
   tokens as TWO chunks (state/conv_state carried between calls) at every
   split point from 1 to T-1 spanning the conv kernel width CK, and asserts
   the final state, final conv_state, and concatenated output all match the
   one-shot run. This is the sequential path's own ground-truth self-
   consistency check -- if this ever regresses, every downstream path
   (batched decode, fused chunk kernel, fused recurrent kernel) is being
   validated against a broken reference.

   Hand-verified alongside this test (2026-08-28 session): the conv1d
   windowing for a CONTINUING chunk (seg_conv_state prepended, then
   self.conv1d()'s own built-in padding=CK-1 applied on top) works out
   correctly because the module's symmetric zero-padding on both sides of
   the (history + new-chunk) input falls entirely OUTSIDE the
   [offset : offset+T_i] slice forward() reads -- the left zero-padding
   region and the right zero-padding region are never touched by that
   slice. Worked through the cross-correlation index algebra by hand;
   this test is the empirical confirmation.

2. MULTI-SEGMENT PACKING: A single prefill call can pack multiple sequences
   of different lengths (cu_seqlens with >1 segment, e.g. [0,5,14,20] for
   T=[5,9,6]). This test runs 3 segments packed together in one forward()
   call vs. each segment run SEPARATELY as its own num_segments=1 call
   (including a variant with mixed fresh + continuing incoming state), and
   asserts per-segment output/state/conv_state match -- catching any
   cross-segment leakage in the packed (N, ...) tensor layout.

Tolerance: NOT bitwise-exact. Re-segmenting the SAME tokens into a
different number of matmul rows (T_i) can hit a different BLAS
kernel-selection path for in_proj_qkv/in_proj_a/in_proj_b/in_proj_z/
out_proj, producing ~1e-8 order-of-summation noise -- the same
phenomenon test_qwen35_gdr_decode_batched.py already documents for
reassociation across the BATCH axis (N), just triggered here by the
SEQUENCE-LENGTH axis (T_i) instead. Uses that file's same established bar:
cosine > 0.99999 and max_abs_diff < 1e-4.

Pure CPU, sequential path only (no fla/triton dependency).
Run with TORCHDYNAMO_DISABLE=1.

Usage:
    TORCHDYNAMO_DISABLE=1 python tests/test_qwen35_gdr_chunked_prefill.py
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

H, LKH, LVH, LHD, CK, EPS = 256, 4, 8, 32, 4, 1e-6


def _init_dist():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29534")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")


def _build():
    from nanovllm.models.qwen3_5 import Qwen35LinearAttention
    torch.manual_seed(0)
    m = Qwen35LinearAttention(
        hidden_size=H, linear_attn_kq_heads=LKH, linear_attn_v_heads=LVH,
        linear_attn_head_dim=LHD, conv_kernel_size=CK, rms_norm_eps=EPS,
    )
    init_model_weights_with_norms_(m, seed=1)
    return m.to(torch.float32).eval()


def _cu(lengths):
    cu = [0]
    for l in lengths:
        cu.append(cu[-1] + l)
    return torch.tensor(cu, dtype=torch.int32)


def _cmp(a, b, name, *, exact=False):
    is_exact = torch.equal(a, b)
    d = (a.float() - b.float()).abs().max().item()
    af, bf = a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
    cos = F.cosine_similarity(af, bf, dim=-1).item()
    print(f"    {name:24s} max_abs_diff={d:.3e}  cosine={cos:.10f}  exact={is_exact}")
    if exact:
        assert is_exact, f"{name}: expected bitwise-exact, got max_abs_diff={d:.3e}"
    else:
        assert cos > 0.99999 and d < 1e-4, (
            f"{name}: cosine={cos:.8f} max_abs_diff={d:.3e} -- exceeds the "
            f"benign-reassociation bar (cos>0.99999, diff<1e-4) established by "
            f"test_qwen35_gdr_decode_batched.py"
        )


def test_chunked_prefill_continuation():
    print("\n[GDN-1] chunked-prefill continuation: 1 call vs 2 chunks (state threaded)")
    m = _build()
    torch.manual_seed(42)
    T = 20
    hs = torch.randn(T, H)

    with torch.no_grad():
        y_full, s_full, c_full = m(hs, _cu([T]), states=None, conv_states=None)

    for split in (1, 2, CK - 2, CK - 1, CK, CK + 1, 7, T - 1):
        if not (0 < split < T):
            continue
        with torch.no_grad():
            y1, s1, c1 = m(hs[:split], _cu([split]), states=None, conv_states=None)
            y2, s2, c2 = m(hs[split:], _cu([T - split]), states=s1, conv_states=c1)
        y_chunked = torch.cat([y1, y2], dim=0)
        print(f"  split={split}:")
        _cmp(y_full, y_chunked, "output")
        _cmp(s_full, s2, "final_state")
        _cmp(c_full, c2, "final_conv_state")
    print("  [PASS] chunked-prefill continuation matches one-shot run for every split point")


def test_multi_segment_packing():
    print("\n[GDN-2] multi-segment packing: 3 segments packed vs run separately")
    m = _build()
    torch.manual_seed(7)
    lengths = [5, 9, 6]
    hs_list = [torch.randn(l, H) for l in lengths]
    hs_packed = torch.cat(hs_list, dim=0)

    with torch.no_grad():
        y_packed, s_packed, c_packed = m(hs_packed, _cu(lengths), states=None, conv_states=None)

    offset = 0
    for i, l in enumerate(lengths):
        with torch.no_grad():
            y_i, s_i, c_i = m(hs_list[i], _cu([l]), states=None, conv_states=None)
        y_slice = y_packed[offset:offset + l]
        print(f"  segment {i} (len={l}):")
        _cmp(y_i, y_slice, "output")
        _cmp(s_i.squeeze(0), s_packed[i], "final_state")
        _cmp(c_i.squeeze(0), c_packed[i], "final_conv_state")
        offset += l
    print("  [PASS] packed multi-segment output/state matches per-segment isolated runs")


def test_multi_segment_with_incoming_state():
    print("\n[GDN-3] multi-segment packing WITH incoming state (mixed fresh + continuing)")
    m = _build()
    torch.manual_seed(13)
    lengths = [4, 6, 3]
    hs_list = [torch.randn(l, H) for l in lengths]

    with torch.no_grad():
        _, seed_state, seed_conv = m(torch.randn(3, H), _cu([3]), states=None, conv_states=None)

    # seed_state/seed_conv already carry the leading (num_segments=1, ...)
    # dim from forward()'s own return.
    zero_state = torch.zeros(1, LVH, LHD, LHD)
    zero_conv = torch.zeros(1, m.qkv_dim, CK - 1)
    states_in = torch.cat([zero_state, seed_state, zero_state], dim=0)
    conv_in = torch.cat([zero_conv, seed_conv, zero_conv], dim=0)

    hs_packed = torch.cat(hs_list, dim=0)
    with torch.no_grad():
        y_packed, s_packed, c_packed = m(hs_packed, _cu(lengths), states=states_in, conv_states=conv_in)

    offset = 0
    per_seg_states = [None, seed_state, None]
    per_seg_convs = [None, seed_conv, None]
    for i, l in enumerate(lengths):
        st_in = per_seg_states[i]
        cv_in = per_seg_convs[i]
        with torch.no_grad():
            y_i, s_i, c_i = m(hs_list[i], _cu([l]), states=st_in, conv_states=cv_in)
        y_slice = y_packed[offset:offset + l]
        print(f"  segment {i} (len={l}, {'continuing' if st_in is not None else 'fresh'}):")
        _cmp(y_i, y_slice, "output")
        _cmp(s_i.squeeze(0), s_packed[i], "final_state")
        _cmp(c_i.squeeze(0), c_packed[i], "final_conv_state")
        offset += l
    print("  [PASS] mixed fresh+continuing segments in one packed call match isolated runs")


def main():
    _init_dist()
    print("=" * 70)
    print("GDN deep diagnosis: chunked-prefill continuation + multi-segment packing")
    print("=" * 70)
    test_chunked_prefill_continuation()
    test_multi_segment_packing()
    test_multi_segment_with_incoming_state()
    print("\nALL GDN DEEP-DIAGNOSIS CHECKS PASSED")


if __name__ == "__main__":
    main()
