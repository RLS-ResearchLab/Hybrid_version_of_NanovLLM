"""Isolated correctness smoke test for layers/fused_recurrent.py's
fused_recurrent_gated_delta_rule (decode-time GDR kernel) and
layers/gated_delta_net.py's causal_conv1d_decode_triton -- the same
"synthetic inputs, plain-PyTorch reference, cosine similarity" pattern
layers/smoke_test_moe_w8a8_hopper.py used for the Hopper kernel, applied
here before either new kernel touches models/qwen3_5.py's decode path.

WHY THIS EXISTS: models/qwen3_5.py's Qwen35LinearAttention.forward() always
uses a sequential, per-segment Python loop for decode (is_decode_shape=True
forces this unconditionally -- see that function's own comment on why, both
for correctness-scope and CUDA-graph-capture reasons). These two new files
(added 2026-08-26, not yet wired into the model) are candidate Triton
kernels for batching that loop across the whole decode batch in one launch
each. Per this project's own established discipline (every new kernel this
project has shipped -- the Ampere MoE Triton kernel, the Hopper W8A8 CUDA
kernel -- was isolated and hand-verified against a plain-PyTorch reference
BEFORE being wired into the real model, because "looks like the same math"
has been wrong before here), this is that isolation step for the GDR decode
kernels. Nothing in models/qwen3_5.py is touched by this file.

TWO INDEPENDENT CHECKS, not one end-to-end check -- isolating each kernel
separately (rather than only comparing full decode-step output) is
deliberate: if only the combined result were checked and it failed, there'd
be no way to tell which of the two kernels (conv or recurrence) was at
fault without re-deriving this exact split under time pressure, which is
exactly the kind of debugging this project's "isolate one coordinate,
verify every pipeline stage independently" methodology (see the W8A8
Kernel Postmortem artifact linked from SESSION_HANDOFF_2026-08-25.md) was
built to avoid needing to do live.

REQUIRES A REAL CUDA GPU WITH TRITON to run past the reference computation
-- unlike the MoE Hopper kernel's gate/up permutation (pure tensor-index
math, CPU-runnable), fused_recurrent_gated_delta_rule and
causal_conv1d_decode_triton are @triton.jit kernels: they cannot compile or
execute without CUDA. This file's reference implementations run fine on
CPU (pure PyTorch) and were written/checked by hand-tracing against
models/qwen3_5.py's existing sequential-scan decode branch term by term
(see PR/commit notes) -- but the actual kernel comparison below has never
executed anywhere, on any hardware, as of this writing.

BEFORE trusting a PASS here as "ready for production": this only checks
kernel correctness in isolation. It does NOT check CUDA-graph
capturability (models/qwen3_5.py's own comment on why decode currently
never takes ANY fused path is explicit that fla's prefill kernel silently
fails to be capturable -- that same question is open and unanswered for
these two new kernels), and it does NOT establish whether the sequential
per-segment loop this would replace is actually a measurable cost in
production. On the one real profiling data point that exists
(moe_quantization_memo.md's nsys graph-mode trace), the "many small kernel
launches" eager-mode overhead pattern was attributed to prefill's
_forward_dispatch_ep, not decode's GDR path -- and decode already runs
under CUDA graph capture in production, which that same profile found
eliminates most of this class of overhead for whatever it captures. So a
PASS here establishes the kernels are CORRECT, not that they are WORTH
integrating -- tests/profile_decode_launch_overhead.py (already written,
untested at tp=1, hangs at tp=2) is the right tool to answer that
separately.

Usage:
    python layers/smoke_test_fused_recurrent_gdr.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Both fused_recurrent.py and gated_delta_net.py do `import triton` at module
# level -- this dev machine has no triton (no Windows wheel, same gotcha
# every other Triton-kernel file in this project already documents), so that
# import fails before either module's code can even be reached, regardless
# of CUDA availability. Guard it so the CPU-only reference half of this
# script (which needs neither triton nor CUDA) still runs and reports
# something useful, instead of crashing before main() is ever called.
try:
    from fused_recurrent import fused_recurrent_gated_delta_rule  # noqa: E402
    from gated_delta_net import causal_conv1d_decode_triton        # noqa: E402
    _KERNELS_IMPORTABLE = True
    _IMPORT_ERROR = None
except ImportError as e:
    _KERNELS_IMPORTABLE = False
    _IMPORT_ERROR = e


def reference_gdr_decode_step(q, k, v, a, b, A_log, dt_bias, head_expand, scale):
    """Plain-PyTorch reference for ONE decode step (T=1), batched across N
    independent sequences -- hand-derived from models/qwen3_5.py's
    Qwen35LinearAttention.forward() sequential-scan decode branch
    (lines ~537-586 as of 2026-08-26), generalized from "one segment at a
    time via a Python loop" to "all N segments via broadcasting," since at
    decode every segment is independent (no cross-segment interaction) --
    same math, no loop needed for a reference implementation.

    q, k: (N, lkh, LHD)  -- NOT yet head-expanded
    v:    (N, lvh, LHD)
    a, b: (N, lvh)        -- RAW pre-activation (pre-softplus/pre-sigmoid)
    A_log, dt_bias: (lvh,)
    Returns: y (N, lvh, LHD), new_state (N, lvh, LHD, LHD) -- both float32,
    matching models/qwen3_5.py's existing state-buffer dtype contract.
    """
    N, lkh, lhd = q.shape
    lvh = v.shape[1]

    q = q.float().repeat_interleave(head_expand, dim=1)  # (N, lvh, LHD)
    k = k.float().repeat_interleave(head_expand, dim=1)
    v = v.float()

    def l2norm(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    q = l2norm(q) * scale
    k = l2norm(k)

    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())  # (N, lvh)
    beta = b.float().sigmoid()  # (N, lvh)

    S = torch.zeros(N, lvh, lhd, lhd, dtype=torch.float32, device=q.device)
    g_t = g.exp().unsqueeze(-1).unsqueeze(-1)          # (N, lvh, 1, 1)
    beta_t = beta.unsqueeze(-1)                          # (N, lvh, 1)
    S = S * g_t
    kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)          # (N, lvh, LHD)
    delta = (v - kv_mem) * beta_t
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
    y = (S * q.unsqueeze(-1)).sum(dim=-2)                # (N, lvh, LHD)
    return y, S


def reference_gdr_decode_step_with_state(q, k, v, a, b, A_log, dt_bias, head_expand, scale, S_in):
    """Same as reference_gdr_decode_step but with a real (nonzero) incoming
    state S_in (N, lvh, LHD, LHD) -- the actually-relevant case for decode
    (every real decode step after the first follows at least one prefill
    that established nonzero state); the zero-state variant above only
    matters for the model's cold-start/warmup call."""
    N, lkh, lhd = q.shape
    lvh = v.shape[1]

    q = q.float().repeat_interleave(head_expand, dim=1)
    k = k.float().repeat_interleave(head_expand, dim=1)
    v = v.float()

    def l2norm(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    q = l2norm(q) * scale
    k = l2norm(k)

    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    beta = b.float().sigmoid()

    S = S_in.float()
    g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta.unsqueeze(-1)
    S = S * g_t
    kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v - kv_mem) * beta_t
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
    y = (S * q.unsqueeze(-1)).sum(dim=-2)
    return y, S


def reference_causal_conv1d_decode(x, weight, conv_state):
    """Plain-PyTorch reference for the decode-time causal conv1d + SiLU --
    hand-derived from models/qwen3_5.py's forward(), T_i==1 branch (lines
    ~516-522 + the SiLU at line ~534).

    x: (N, D, 1). weight: (D, 1, K) (nn.Conv1d depthwise weight, bias=False).
    conv_state: (N, D, K-1). Returns (out (N, D, 1), new_state (N, D, K-1)).
    """
    combined = torch.cat([conv_state, x], dim=2)       # (N, D, K)
    new_state = combined[:, :, -(weight.shape[-1] - 1):].clone()
    # F.conv1d already batches over dim 0 (N) natively -- no per-batch-element
    # groups trick needed, groups=D (depthwise) is sufficient.
    out = F.conv1d(combined, weight, bias=None, padding=0, groups=weight.shape[0])
    out = F.silu(out)
    return out, new_state


def main():
    torch.manual_seed(0)

    # Small synthetic dims -- not production scale, matching this project's
    # established progression (smoke_test_moe_w8a8_hopper.py's own comment:
    # "modest scale for a first isolated check").
    N = 6                 # decode batch size (concurrency)
    lkh = 2                # local key/query heads
    lvh = 8                # local value heads (head_expand = lvh/lkh = 4)
    lhd = 16                # head dim
    ck = 4                  # conv kernel size
    qkv_dim = (lkh + lkh + lvh) * lhd
    head_expand = lvh // lkh
    scale = lhd ** -0.5

    print(f"Config: N={N} lkh={lkh} lvh={lvh} lhd={lhd} ck={ck} head_expand={head_expand}")

    # ---- Check 1: causal_conv1d_decode_triton vs. plain-PyTorch reference ----
    x = torch.randn(N, qkv_dim, 1, dtype=torch.float32) * 0.1
    conv_weight = torch.randn(qkv_dim, 1, ck, dtype=torch.float32) * 0.1
    conv_state = torch.randn(N, qkv_dim, ck - 1, dtype=torch.float32) * 0.1

    ref_out, ref_new_state = reference_causal_conv1d_decode(x, conv_weight, conv_state)
    print(f"[1a] reference conv1d-decode ran on CPU: out={tuple(ref_out.shape)} "
          f"new_state={tuple(ref_new_state.shape)}")

    if not _KERNELS_IMPORTABLE:
        print(f"[1b] SKIPPED -- fused_recurrent.py/gated_delta_net.py could not be "
              f"imported ({_IMPORT_ERROR!r}). Reference above is CPU-only structural "
              f"validation (shapes correct, ran without error) -- NOT a correctness PASS.")
    elif not torch.cuda.is_available():
        print("[1b] SKIPPED -- causal_conv1d_decode_triton requires CUDA (Triton JIT), "
              "not available on this machine. Reference above is CPU-only structural "
              "validation (shapes correct, ran without error) -- NOT a correctness PASS.")
    else:
        device = torch.device("cuda")
        kernel_out, kernel_new_state = causal_conv1d_decode_triton(
            x.to(device, torch.bfloat16), conv_weight.to(device, torch.bfloat16),
            conv_state.to(device, torch.bfloat16),
        )
        cos_out = F.cosine_similarity(
            ref_out.reshape(-1), kernel_out.float().cpu().reshape(-1), dim=0).item()
        cos_state = F.cosine_similarity(
            ref_new_state.reshape(-1), kernel_new_state.float().cpu().reshape(-1), dim=0).item()
        print(f"[1b] causal_conv1d_decode_triton vs. reference: "
              f"out_cosine={cos_out:.6f} new_state_cosine={cos_state:.6f}")
        assert cos_out > 0.99 and cos_state > 0.99, "conv1d-decode kernel diverges from reference"

    # ---- Check 2: fused_recurrent_gated_delta_rule vs. plain-PyTorch reference ----
    q = torch.randn(N, lkh, lhd, dtype=torch.float32) * 0.1
    k = torch.randn(N, lkh, lhd, dtype=torch.float32) * 0.1
    v = torch.randn(N, lvh, lhd, dtype=torch.float32) * 0.1
    a = torch.randn(N, lvh, dtype=torch.float32) * 0.1
    b = torch.randn(N, lvh, dtype=torch.float32) * 0.1
    A_log = torch.randn(lvh, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(lvh, dtype=torch.float32) * 0.1
    S_in = torch.randn(N, lvh, lhd, lhd, dtype=torch.float32) * 0.05

    ref_y_zero, ref_S_zero = reference_gdr_decode_step(q, k, v, a, b, A_log, dt_bias, head_expand, scale)
    ref_y_state, ref_S_state = reference_gdr_decode_step_with_state(
        q, k, v, a, b, A_log, dt_bias, head_expand, scale, S_in)
    print(f"[2a] reference GDR decode step ran on CPU (zero-state and real-state cases): "
          f"y={tuple(ref_y_zero.shape)} S={tuple(ref_S_zero.shape)}")

    if not _KERNELS_IMPORTABLE:
        print(f"[2b] SKIPPED -- fused_recurrent.py/gated_delta_net.py could not be "
              f"imported ({_IMPORT_ERROR!r}). Reference above is CPU-only structural "
              f"validation only -- NOT a correctness PASS.")
        print("\nCould not import the Triton kernel modules on this machine -- "
              "structural/shape checks passed, kernel correctness UNCONFIRMED. "
              "Re-run this exact script on real hardware with triton installed "
              "before trusting either kernel.")
        return
    elif not torch.cuda.is_available():
        print("[2b] SKIPPED -- fused_recurrent_gated_delta_rule requires CUDA (Triton JIT), "
              "not available on this machine. Reference above is CPU-only structural "
              "validation only -- NOT a correctness PASS.")
        print("\nNo GPU on this machine -- structural/shape checks passed, kernel "
              "correctness UNCONFIRMED. Re-run this exact script on real hardware "
              "before trusting either kernel.")
        return
    else:
        device = torch.device("cuda")
        dt = torch.bfloat16
        q_d = q.unsqueeze(1).to(device, dt)   # (N, 1, lkh, LHD)
        k_d = k.unsqueeze(1).to(device, dt)
        v_d = v.unsqueeze(1).to(device, dt)
        g_d = a.unsqueeze(1).to(device, dt)   # RAW a -- kernel applies -A_log.exp()*softplus internally
        beta_d = b.unsqueeze(1).to(device, dt)  # RAW b -- kernel applies sigmoid internally
        A_log_d = A_log.to(device)
        dt_bias_d = dt_bias.to(device)

        for label, S0, ref_y, ref_S in [
            ("zero-state", None, ref_y_zero, ref_S_zero),
            ("real-state", S_in.to(device), ref_y_state, ref_S_state),
        ]:
            out, new_state = fused_recurrent_gated_delta_rule(
                q_d, k_d, v_d,
                g=g_d, beta=beta_d,
                scale=scale,
                initial_state=S0,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                A_log=A_log_d, dt_bias=dt_bias_d,
                use_beta_sigmoid_in_kernel=True,
            )
            cos_y = F.cosine_similarity(
                ref_y.reshape(-1), out.float().cpu().reshape(N, lvh, lhd).reshape(-1), dim=0).item()
            cos_S = F.cosine_similarity(
                ref_S.reshape(-1), new_state.float().cpu().reshape(-1), dim=0).item()
            print(f"[2b:{label}] fused_recurrent_gated_delta_rule vs. reference: "
                  f"y_cosine={cos_y:.6f} new_state_cosine={cos_S:.6f}")
            assert cos_y > 0.99 and cos_S > 0.99, f"GDR decode kernel diverges from reference ({label})"

        print("\nPASS -- both kernels agree with the plain-PyTorch reference at this small "
              "scale. Still NOT sufficient on its own to enable either kernel in production: "
              "see this file's module docstring for what's still unconfirmed (CUDA-graph "
              "capturability, and whether the sequential loop this replaces is even a "
              "measurable cost given decode already runs under graph capture in production).")


if __name__ == "__main__":
    main()
