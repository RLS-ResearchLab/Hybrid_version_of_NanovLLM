"""Shared, scale-parameterized core for the Q5 expert-parallel dispatch/
combine correctness check. See tests/test_moe_ep_dispatch.py (top_k=2,
Checkpoint 3) and tests/test_moe_ep_dispatch_topk8.py (top_k=8 extension)
for the two concrete runs -- same design, same checks, different scale, no
duplicated logic.

Design:
  - Reference: the existing, unmodified Qwen35MoE._forward_dispatch, run
    single-process (ep_size=1) -- literally the pre-existing method, no new
    code involved in producing it.
  - Test subject: Qwen35MoE._forward_dispatch_ep, run via real multi-process
    torch.multiprocessing.spawn with gloo -- genuine multi-rank
    communication (dist.all_reduce), not the single-process RANK=0/WORLD=1
    construction-only trick used elsewhere in this codebase.
  - Hidden-state rows uniquely tagged: row i = (i+1) * ones(H).
  - A companion token_id tensor rides through the *real* dispatch/combine
    path (Qwen35MoE._forward_dispatch_ep sets self._last_ep_token_id_roundtrip
    from the exact same ownership mask used for the real computation, not a
    side-channel reimplementation) and must round-trip to torch.arange(N)
    exactly -- always asserted, regardless of scale.
  - Hidden-state (routed-expert-only) comparison: torch.equal is asserted
    only when scale.assert_bitwise_routed is True (proven exact for
    top_k==2). For other top_k values the result is MEASURED and reported
    (bitwise-equal bool + max abs diff), not assumed -- a generous
    torch.allclose is still asserted as a sanity bound to catch genuine
    bugs, not to claim bitwise exactness that hasn't been proven.
"""
import os
import sys
import tempfile
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [ROOT]
    _pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg


@dataclass(frozen=True)
class MoEScale:
    hidden: int
    intermediate: int
    shared_intermediate: int  # must be divisible by ep_size (shared_expert's
                               # RowParallelLinear/MergedColumnParallelLinear
                               # already correctly shard this dimension)
    num_experts: int          # must be divisible by ep_size (round-robin)
    top_k: int
    n_tokens: int
    ep_size: int
    ref_port: int
    worker_port: int
    assert_bitwise_routed: bool  # True only where bitwise exactness is proven (top_k==2)
    dtype: torch.dtype = torch.float32
    enforce_sanity_bound: bool = True  # False = pure measurement, no pass/fail on numeric closeness


def _isolate_routed_expert_output(moe, x):
    """Temporarily force shared_expert_gate's output very negative so
    sigmoid(...) rounds to exactly 0.0, making sg * shared_expert(x) exactly
    0.0 regardless of any floating-point noise in shared_expert's own
    (possibly TP-sharded) computation -- isolates the routed-expert-only
    contribution using the model's real, unmodified forward() and real
    weights, no reimplementation. Relies on x being all-positive (true for
    this test's row_i = (i+1)*ones(H) tagging), so weight=-1000 dot a
    positive row is guaranteed very negative.
    """
    orig = moe.shared_expert_gate.weight.data.clone()
    moe.shared_expert_gate.weight.data.fill_(-1000.0)
    try:
        return moe(x)
    finally:
        moe.shared_expert_gate.weight.data.copy_(orig)


def _relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Elementwise |a-b| / (|b|+eps), max over all elements."""
    return ((a.float() - b.float()).abs() / (b.float().abs() + eps)).max().item()


def build_reference(scale: MoEScale):
    """Single-process (ep_size=1): build Qwen35MoE, seed deterministic
    weights, compute the dense reference output via the existing, unmodified
    _forward_dispatch -- and verify the hard gate (ep_size=1 never enters
    _forward_dispatch_ep) in the same pass."""
    dist.init_process_group(
        "gloo", init_method=f"tcp://localhost:{scale.ref_port}", rank=0, world_size=1
    )

    from nanovllm.models.qwen3_5 import Qwen35MoE

    moe_ref = Qwen35MoE(scale.hidden, scale.intermediate, scale.shared_intermediate,
                         scale.num_experts, scale.top_k)
    assert moe_ref.ep_size == 1
    assert moe_ref.experts.num_experts == scale.num_experts, \
        "ep_size=1 must allocate the FULL expert count, same as before EP support"

    torch.manual_seed(1234)
    for name in sorted(dict(moe_ref.named_parameters()).keys()):
        p = dict(moe_ref.named_parameters())[name]
        p.data.normal_(mean=0.0, std=0.02)

    # Seed in fp32 for RNG quality, then cast the whole module (all real
    # params, no integer buffers) to the target dtype -- weights, inputs,
    # AND the computation itself all run in scale.dtype from here on.
    moe_ref = moe_ref.to(scale.dtype)
    x = torch.stack([(i + 1) * torch.ones(scale.hidden) for i in range(scale.n_tokens)], dim=0)
    x = x.to(scale.dtype)

    call_count = {"n": 0}
    real_ep_method = moe_ref._forward_dispatch_ep

    def _tracking_ep(*a, **kw):
        call_count["n"] += 1
        return real_ep_method(*a, **kw)

    moe_ref._forward_dispatch_ep = _tracking_ep
    reference_out = moe_ref(x)
    moe_ref._forward_dispatch_ep = real_ep_method
    assert call_count["n"] == 0, (
        f"HARD GATE VIOLATED: _forward_dispatch_ep was called {call_count['n']} times "
        f"at ep_size=1 -- the EP branch must never be entered here"
    )
    print(f"[main] HARD GATE OK -- _forward_dispatch_ep call count at ep_size=1: {call_count['n']}")
    print(f"[main] reference (dense, ep_size=1) output shape={tuple(reference_out.shape)}")

    reference_routed_only = _isolate_routed_expert_output(moe_ref, x)

    ref_state = {k: v.clone() for k, v in moe_ref.state_dict().items()}
    dist.destroy_process_group()
    return ref_state, x, reference_out, reference_routed_only


def worker(rank: int, scale: MoEScale, tmp_path: str):
    dist.init_process_group(
        "gloo", init_method=f"tcp://localhost:{scale.worker_port}",
        rank=rank, world_size=scale.ep_size,
    )

    from nanovllm.models.qwen3_5 import Qwen35MoE
    from nanovllm.utils.loader import shard_experts_tensor

    payload = torch.load(tmp_path, weights_only=True)
    ref_state = payload["ref_state"]
    x = payload["x"]
    reference_out = payload["reference_out"]
    reference_routed_only = payload["reference_routed_only"]

    moe_local = Qwen35MoE(scale.hidden, scale.intermediate, scale.shared_intermediate,
                           scale.num_experts, scale.top_k)
    moe_local = moe_local.to(scale.dtype)  # before loading weights, so param dtype matches ref_state's
    assert moe_local.ep_size == scale.ep_size
    assert moe_local.ep_rank == rank
    assert moe_local.experts.num_experts == scale.num_experts // scale.ep_size

    moe_local.gate.weight.weight_loader(moe_local.gate.weight, ref_state["gate.weight"])
    moe_local.shared_expert_gate.weight.weight_loader(
        moe_local.shared_expert_gate.weight, ref_state["shared_expert_gate.weight"]
    )
    gate_half, up_half = ref_state["shared_expert.gate_up_proj.weight"].chunk(2, dim=0)
    moe_local.shared_expert.gate_up_proj.weight.weight_loader(
        moe_local.shared_expert.gate_up_proj.weight, gate_half, 0
    )
    moe_local.shared_expert.gate_up_proj.weight.weight_loader(
        moe_local.shared_expert.gate_up_proj.weight, up_half, 1
    )
    moe_local.shared_expert.down_proj.weight.weight_loader(
        moe_local.shared_expert.down_proj.weight, ref_state["shared_expert.down_proj.weight"]
    )
    gu_shard = shard_experts_tensor(ref_state["experts.gate_up_proj"], rank, scale.ep_size)
    dn_shard = shard_experts_tensor(ref_state["experts.down_proj"], rank, scale.ep_size)
    moe_local.experts.gate_up_proj.data.copy_(gu_shard)
    moe_local.experts.down_proj.data.copy_(dn_shard)

    ep_out = moe_local(x)  # ep_size>1 -> _forward_dispatch_ep

    max_abs_diff_full = (ep_out - reference_out).abs().max().item()
    rel_err_full = _relative_error(ep_out, reference_out)
    is_bitwise_full = torch.equal(ep_out, reference_out)
    print(f"[rank{rank}] FULL output (incl. shared_expert) dtype={ep_out.dtype} "
          f"max abs diff vs reference: {max_abs_diff_full:.3e}  "
          f"max relative error: {rel_err_full:.3e}  bitwise-equal={is_bitwise_full}")

    # Isolate the routed-expert-only contribution on both sides (same helper,
    # same real forward() -- not a reimplementation) to test EXACTLY the new
    # EP logic, decoupled from shared_expert's pre-existing TP reassociation.
    ep_routed_only = _isolate_routed_expert_output(moe_local, x)
    max_abs_diff_routed = (ep_routed_only - reference_routed_only).abs().max().item()
    rel_err_routed = _relative_error(ep_routed_only, reference_routed_only)
    is_bitwise_routed = torch.equal(ep_routed_only, reference_routed_only)
    print(f"[rank{rank}] ROUTED-EXPERT-ONLY (the actual EP logic under test) dtype={ep_routed_only.dtype}: "
          f"max abs diff={max_abs_diff_routed:.3e}  max relative error={rel_err_routed:.3e}  "
          f"bitwise-equal={is_bitwise_routed}")
    print(f"[rank{rank}] routed-expert-only ep ={ep_routed_only.tolist()}")
    print(f"[rank{rank}] routed-expert-only ref={reference_routed_only.tolist()}")

    if scale.assert_bitwise_routed:
        assert is_bitwise_routed, (
            f"[rank{rank}] BITWISE MISMATCH on routed-expert-only output vs dense reference "
            f"(expected exact for this scale): max abs diff={max_abs_diff_routed:.3e}"
        )
        print(f"[rank{rank}] HIDDEN-STATE CHECK: bitwise exactness CONFIRMED (asserted)")
    elif scale.enforce_sanity_bound:
        # Not assumed exact -- this scale is exploratory. Still assert a
        # generous sanity bound so a genuine indexing/pairing bug (which
        # would produce a LARGE discrepancy, not rounding-level noise)
        # still fails loudly rather than being silently reported as "close."
        assert torch.allclose(ep_routed_only, reference_routed_only, rtol=1e-3, atol=1e-6), (
            f"[rank{rank}] SANITY BOUND VIOLATED on routed-expert-only output: "
            f"max abs diff={max_abs_diff_routed:.3e} -- this is too large to be ordinary "
            f"floating-point reassociation, looks like a real bug"
        )
        print(f"[rank{rank}] HIDDEN-STATE CHECK: MEASURED (not assumed) -- "
              f"bitwise-equal={is_bitwise_routed}, max abs diff={max_abs_diff_routed:.3e}, "
              f"max relative error={rel_err_routed:.3e}")
    else:
        # Pure measurement -- no pass/fail on numeric closeness at all. Used
        # for exploratory dtype runs where the fp32-tuned sanity bound isn't
        # known to be the right bound yet (that's the open design question).
        print(f"[rank{rank}] HIDDEN-STATE CHECK: MEASURED ONLY, no sanity bound enforced -- "
              f"bitwise-equal={is_bitwise_routed}, max abs diff={max_abs_diff_routed:.3e}, "
              f"max relative error={rel_err_routed:.3e}")

    token_id_roundtrip = moe_local._last_ep_token_id_roundtrip
    expected_ids = torch.arange(scale.n_tokens, dtype=torch.int64)
    assert torch.equal(token_id_roundtrip, expected_ids), (
        f"[rank{rank}] TOKEN-ID ROUND-TRIP MISMATCH: got {token_id_roundtrip.tolist()} "
        f"expected {expected_ids.tolist()}"
    )
    print(f"[rank{rank}] TOKEN-ID ROUND-TRIP OK -- exact match to torch.arange({scale.n_tokens})")

    dist.destroy_process_group()
    return {
        "rank": rank,
        "max_abs_diff_routed": max_abs_diff_routed,
        "rel_err_routed": rel_err_routed,
        "is_bitwise_routed": is_bitwise_routed,
    }


def run(scale: MoEScale):
    ref_state, x, reference_out, reference_routed_only = build_reference(scale)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save({
        "ref_state": ref_state, "x": x,
        "reference_out": reference_out,
        "reference_routed_only": reference_routed_only,
    }, tmp_path)

    try:
        mp.spawn(worker, args=(scale, tmp_path), nprocs=scale.ep_size, join=True)
    finally:
        os.unlink(tmp_path)

    print("\nALL CHECKS PASSED")
