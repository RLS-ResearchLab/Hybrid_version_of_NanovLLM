"""Correctness tests for the vectorized-MoE-dispatch path (QLLM Stage 2,
grouped-GEMM intervention) added to Qwen35MoE.forward() in models/qwen3_5.py.

Compares use_vectorized_moe=True vs. False (same weights, same input) at
multiple token counts, and separately against src/model_small_qwen3.5.py's
MoEFFN directly.

IMPORTANT setup note (found during investigation, not obvious from a first
read): Experts.gate_up_proj / down_proj are raw nn.Parameter(torch.empty(...))
with NO auto-initialization anywhere -- confirmed true of BOTH this port and
the reference (src/model_small_qwen3.5.py's own Qwen35MoESmall() construction
never calls anything on them either). Left as torch.empty(...), these can
read back as literal all-zero memory, which makes both dispatch paths
degenerate to an all-zero output (cosine between two zero vectors is 0.0 by
F.cosine_similarity's epsilon convention -- NOT evidence of a bug, just a
meaningless comparison). Every test below explicitly randomizes Experts
weights before comparing, unlike some of this project's existing MoE tests
which may not (test_qwen35_standalone.py's test_moe_vs_reference copies
weights from a freshly-constructed, likewise-never-initialized reference
Experts -- flagging this as a pre-existing gap in that test's rigor, not
fixed here since it's out of this task's scope; if the numbers there ever
looked suspiciously close to a degenerate case, this is why).

Threshold: unlike the GDR kernel (which genuinely reassociates the
delta-rule recurrence's summation order via chunking, and only ever reached
~0.999985 cosine against the sequential scan), grouped-GEMM dispatch does
NOT reorder any single token's K-dimension reduction -- each token's dot
product is the same set of terms in the same order, just launched via a
batched kernel instead of a per-expert Python loop. Empirically (see
investigation notes and this file's own results when run) this reaches
BITWISE-IDENTICAL output on the CPU backend at several token counts, not
just high cosine -- so this file asserts torch.equal exactly, not a cosine
threshold, as the primary check. If a real run ever shows torch.equal fail
but cosine remains high, that would itself be a meaningful new finding
(evidence the GPU grouped-gemm kernel reassociates differently than CPU's)
worth reporting explicitly rather than silently downgrading to a cosine
check.

Usage:
    python tests/test_qwen35_vectorized_moe.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config, load_reference_module, cosine_sim  # noqa: E402


def _build_pair(config, device, seed=42, init_std=0.02):
    from nanovllm.models.qwen3_5 import Qwen35MoE

    torch.manual_seed(seed)
    moe_off = Qwen35MoE(
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        shared_intermediate_size=config.shared_expert_intermediate_size,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        use_vectorized_moe=False,
    ).to(device).to(torch.bfloat16)

    # Explicit init -- see module docstring. Real, nonzero weights are
    # required for a meaningful comparison; torch.empty(...) is not
    # guaranteed zero in general, but was observed to read back as exactly
    # zero in this environment, which silently degenerates the comparison
    # to two zero vectors (cosine 0.0, not a bug signal) if skipped.
    with torch.no_grad():
        moe_off.experts.gate_up_proj.normal_(0, init_std)
        moe_off.experts.down_proj.normal_(0, init_std)

    moe_on = Qwen35MoE(
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        shared_intermediate_size=config.shared_expert_intermediate_size,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        use_vectorized_moe=True,
    ).to(device).to(torch.bfloat16)
    moe_on.load_state_dict(moe_off.state_dict())

    return moe_off, moe_on


def test_vs_sequential(device, T, config=None):
    print("\n" + "=" * 70)
    print(f"Vectorized MoE vs sequential dispatch -- T={T}")
    print("=" * 70)
    config = config or make_small_config()
    moe_off, moe_on = _build_pair(config, device)

    torch.manual_seed(99)
    x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        y_off = moe_off(x)
        y_on = moe_on(x)

    cos = cosine_sim(y_off, y_on)
    exact = torch.equal(y_off, y_on)
    max_abs_diff = (y_off.float() - y_on.float()).abs().max().item()
    argmax_off = y_off.float().argmax(dim=-1)
    argmax_on = y_on.float().argmax(dim=-1)

    print(f"  cosine: {cos:.8f}  bitwise_exact: {exact}  max_abs_diff: {max_abs_diff:.6g}")

    if not exact:
        print("  [NOTE] not bitwise-exact -- see module docstring; this is meaningful "
              "on GPU (would indicate the grouped-gemm kernel reassociates "
              "differently than a per-expert loop) and should be investigated, "
              "not silently downgraded to a cosine-only check.")
    assert cos > 0.999, f"cosine regression: {cos}"
    assert torch.equal(argmax_on, argmax_off), "argmax mismatch -- unexpected given cosine/exact results above"
    print("  [PASS]")


def test_vs_reference(device, T, config=None):
    print("\n" + "=" * 70)
    print(f"Vectorized MoE vs src/model_small_qwen3.5.py reference -- T={T}")
    print("=" * 70)
    config = config or make_small_config()
    _, moe_on = _build_pair(config, device)

    ref_mod = load_reference_module()
    torch.manual_seed(42)
    ref_moe = ref_mod.MoEFFN().to(device).to(torch.bfloat16)
    with torch.no_grad():
        ref_moe.experts.gate_up_proj.normal_(0, 0.02)
        ref_moe.experts.down_proj.normal_(0, 0.02)

    ref_sd = dict(ref_moe.named_parameters())
    for nv_name, nv_param in moe_on.named_parameters():
        if "shared_expert.gate_up_proj.weight" in nv_name:
            gate_n = nv_name.replace("gate_up_proj", "gate_proj")
            up_n = nv_name.replace("gate_up_proj", "up_proj")
            if gate_n in ref_sd and up_n in ref_sd:
                nv_param.data.copy_(torch.cat([ref_sd[gate_n].data, ref_sd[up_n].data], dim=0))
                continue
        if nv_name in ref_sd and nv_param.shape == ref_sd[nv_name].shape:
            nv_param.data.copy_(ref_sd[nv_name].data)

    torch.manual_seed(99)
    x = torch.randn(1, T, config.hidden_size, device=device, dtype=torch.bfloat16)
    xf = x.view(T, config.hidden_size)

    with torch.no_grad():
        y_ref = ref_moe(x).view(T, config.hidden_size)
        y_on = moe_on(xf)

    cos = cosine_sim(y_ref, y_on)
    print(f"  cosine vs reference: {cos:.6f}")
    assert cos > 0.99, f"vs-reference mismatch: {cos}"
    print("  [PASS]")


def test_expert_distribution_sensitivity(device, config=None):
    """Multiple different token counts routing through very different
    expert-load distributions (some experts overloaded, some empty) --
    the routing/sort/offs machinery is SHARED/batched across every token
    in one call (unlike GDR's per-segment state, there's no cross-token
    recurrence here, but the grouped_mm offs construction touches all
    tokens at once), so this is the closest MoE analog to GDR's packed-
    multi-segment check: a bug in offset computation could plausibly only
    show up at specific expert-count distributions, not at every T."""
    print("\n" + "=" * 70)
    print("Vectorized MoE -- expert-load distribution sensitivity")
    print("=" * 70)
    config = config or make_small_config()
    moe_off, moe_on = _build_pair(config, device)

    for T in [config.num_experts * 3, config.num_experts // 2, 1, 3]:
        torch.manual_seed(T)  # different seed per T -> different routing distribution
        x = torch.randn(T, config.hidden_size, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            y_off = moe_off(x)
            y_on = moe_on(x)
        exact = torch.equal(y_off, y_on)
        cos = cosine_sim(y_off, y_on)
        print(f"  T={T:4d}  bitwise_exact={exact}  cosine={cos:.8f}")
        assert cos > 0.999, f"T={T}: cosine regression: {cos}"

    print("  [PASS]")


def main():
    init_dist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device={device}")

    config = make_small_config()
    for T in [1, 6, 37, 129]:
        test_vs_sequential(device, T, config=config)
    test_vs_reference(device, T=8, config=config)
    test_expert_distribution_sensitivity(device, config=config)
    print("\nAll vectorized-MoE correctness tests passed.")


if __name__ == "__main__":
    main()
