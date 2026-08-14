"""Follow-up to tests/diag_grouped_mm_nan.py: that diagnostic showed
torch._grouped_mm itself is clean in isolation (all 6 cases, including the
exact T=1/TK=4/NE=32 empty-group pattern) -- so the NaN found in
test_qwen35_vectorized_moe.py's T=1 case must come from somewhere else:
either later steps in _forward_dispatch_vectorized (routing, silu*up,
combine, shared_expert), OR -- not yet ruled out -- the UNCHANGED
_forward_dispatch (old/sequential) path itself, since
cosine_sim(y_off, y_on) reads NaN if EITHER side is NaN, and this hasn't
been checked yet.

This script: (1) checks y_off and y_on independently for NaN, so it's clear
which side is actually the problem before assuming it's the new code, then
(2) if y_on is the culprit, manually replicates _forward_dispatch_vectorized
step by step with a NaN check after each step, to localize exactly where it
first appears.

Usage:
    python tests/diag_moe_nan_trace.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config  # noqa: E402


def has_bad(t, name):
    nan = torch.isnan(t).any().item()
    inf = torch.isinf(t).any().item()
    print(f"    {name}: shape={tuple(t.shape)}  nan={nan}  inf={inf}  "
          f"min={t.float().min().item():.4g}  max={t.float().max().item():.4g}")
    return nan or inf


def main():
    if not torch.cuda.is_available():
        print("[SKIP] no CUDA GPU available")
        return

    init_dist()
    device = "cuda"
    config = make_small_config()

    from nanovllm.models.qwen3_5 import Qwen35MoE

    torch.manual_seed(42)
    moe_off = Qwen35MoE(
        hidden_size=config.hidden_size, intermediate_size=config.moe_intermediate_size,
        shared_intermediate_size=config.shared_expert_intermediate_size,
        num_experts=config.num_experts, top_k=config.num_experts_per_tok, use_vectorized_moe=False,
    ).to(device).to(torch.bfloat16)
    with torch.no_grad():
        moe_off.experts.gate_up_proj.normal_(0, 0.02)
        moe_off.experts.down_proj.normal_(0, 0.02)
    moe_on = Qwen35MoE(
        hidden_size=config.hidden_size, intermediate_size=config.moe_intermediate_size,
        shared_intermediate_size=config.shared_expert_intermediate_size,
        num_experts=config.num_experts, top_k=config.num_experts_per_tok, use_vectorized_moe=True,
    ).to(device).to(torch.bfloat16)
    moe_on.load_state_dict(moe_off.state_dict())

    torch.manual_seed(99)
    T = 1
    x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

    print("=== Step 1: check y_off and y_on independently ===")
    with torch.no_grad():
        y_off = moe_off(x)
        y_on = moe_on(x)
    off_bad = has_bad(y_off, "y_off (OLD/sequential path, UNCHANGED code)")
    on_bad = has_bad(y_on, "y_on (NEW/vectorized path)")

    if off_bad and not on_bad:
        print("\n[FINDING] y_off (the untouched sequential path) is the one with NaN/Inf. "
              "This is NOT a bug in the new vectorized code -- something about T=1 on this "
              "GPU/bf16 combination is unstable in the ALREADY-EXISTING code path. "
              "Investigate _forward_dispatch, not _forward_dispatch_vectorized.")
        return
    if not off_bad and not on_bad:
        print("\n[FINDING] Neither y_off nor y_on has NaN/Inf on this run -- the earlier "
              "failure may be intermittent, weight-init-dependent (different random seed "
              "state than this run), or otherwise not reproducing here. Re-run "
              "test_qwen35_vectorized_moe.py itself to confirm whether this diagnostic's "
              "different call sequence changed the outcome.")
        return

    print("\n[FINDING] y_on (the NEW vectorized path) has NaN/Inf. Tracing "
          "_forward_dispatch_vectorized step by step to localize it.")

    print("\n=== Step 2: manual trace of _forward_dispatch_vectorized ===")
    H = moe_on.hidden_size
    NE = moe_on.num_experts
    TK = moe_on.top_k
    xf = x.reshape(-1, H)
    N = xf.shape[0]

    with torch.no_grad():
        gate_logits = moe_on.gate(xf)
        has_bad(gate_logits, "gate_logits")

        w, idx = torch.topk(gate_logits, TK, dim=-1)
        has_bad(w, "topk weights (pre-softmax)")
        print(f"    idx (chosen experts): {idx.tolist()}")

        w = F.softmax(w, dim=-1).to(x.dtype)
        has_bad(w, "w (post-softmax, x.dtype)")

        flat_idx = idx.reshape(-1)
        flat_w = w.reshape(-1)
        token_rep = xf.unsqueeze(1).expand(N, TK, H).reshape(N * TK, H)
        has_bad(token_rep, "token_rep")

        sort_order = torch.argsort(flat_idx, stable=True)
        sorted_idx = flat_idx[sort_order]
        sorted_tokens = token_rep[sort_order]
        sorted_weights = flat_w[sort_order]
        has_bad(sorted_tokens, "sorted_tokens")

        expert_counts = torch.zeros(NE, dtype=torch.long, device=x.device)
        expert_counts.scatter_add_(0, sorted_idx, torch.ones_like(sorted_idx))
        print(f"    expert_counts (nonzero only): "
              f"{[(i, c.item()) for i, c in enumerate(expert_counts) if c.item() > 0]}")

        offs = expert_counts.cumsum(0).to(torch.int32)
        print(f"    offs: {offs.tolist()}")

        gate_up_t = moe_on.experts.gate_up_proj.transpose(-1, -2)
        has_bad(gate_up_t, "gate_up_t (transposed weight VIEW)")
        down_t = moe_on.experts.down_proj.transpose(-1, -2)
        has_bad(down_t, "down_t (transposed weight VIEW)")

        gate_up_out = torch._grouped_mm(sorted_tokens, gate_up_t, offs=offs)
        bad1 = has_bad(gate_up_out, "gate_up_out (1st grouped_mm)")
        if bad1:
            print("    [LOCALIZED] NaN/Inf appears at the FIRST grouped_mm call "
                  "(gate_up_proj) -- inconsistent with diag_grouped_mm_nan.py's clean "
                  "result on a similarly-shaped isolated call; the difference must be "
                  "something about THIS specific call's inputs (real routing-derived "
                  "sorted_tokens/offs, or the actual nn.Parameter transposed view, vs. "
                  "that diagnostic's synthetic tensors). Compare offs/expert_counts "
                  "printed above against diag_grouped_mm_nan.py's case 6 construction.")
            return

        gate, up = gate_up_out.chunk(2, dim=-1)
        has_bad(gate, "gate")
        has_bad(up, "up")

        h = F.silu(gate) * up
        bad2 = has_bad(h, "h (after silu*up)")
        if bad2:
            print("    [LOCALIZED] NaN/Inf appears at silu(gate)*up, AFTER a clean "
                  "first grouped_mm -- check for silu overflow on extreme gate values.")
            return

        h2 = torch._grouped_mm(h, down_t, offs=offs)
        bad3 = has_bad(h2, "h2 (2nd grouped_mm, down_proj)")
        if bad3:
            print("    [LOCALIZED] NaN/Inf appears at the SECOND grouped_mm call "
                  "(down_proj) -- different K/N dims than the first call and than "
                  "diag_grouped_mm_nan.py's cases, worth its own isolated repro at "
                  "these specific dims (K=intermediate_size, N=hidden_size).")
            return

        combine_dtype = torch.float32
        sorted_out = sorted_weights.unsqueeze(-1).to(combine_dtype) * h2.to(combine_dtype)
        bad4 = has_bad(sorted_out, "sorted_out (fp32 combine)")
        if bad4:
            print("    [LOCALIZED] NaN/Inf appears at the fp32-promoted combine step, "
                  "after a clean down_proj matmul.")
            return

        unsort_order = torch.argsort(sort_order, stable=True)
        out = sorted_out[unsort_order].reshape(N, TK, H).sum(dim=1).to(x.dtype)
        bad5 = has_bad(out, "out (post-unsort, pre-shared-expert)")
        if bad5:
            print("    [LOCALIZED] NaN/Inf appears at the unsort/reshape/sum step.")
            return

        sg = torch.sigmoid(moe_on.shared_expert_gate(xf))
        has_bad(sg, "sg (shared expert gate)")
        se = moe_on.shared_expert(xf)
        bad6 = has_bad(se, "shared_expert(xf) output")
        if bad6:
            print("    [LOCALIZED] NaN/Inf appears in shared_expert -- unrelated to "
                  "grouped_mm entirely, this branch is untouched by the vectorized change.")
            return

        print("\n[UNEXPECTED] Manual trace found no NaN/Inf at any step, but the "
              "original y_on check flagged it. Re-run to check for run-to-run "
              "nondeterminism (uninitialized memory / race condition), or an "
              "environment difference between this trace and the original test call.")


if __name__ == "__main__":
    main()
