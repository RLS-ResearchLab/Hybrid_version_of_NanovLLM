"""EP dispatch/combine correctness under DELIBERATELY constructed edge-case
routing: zero tokens for an entire rank's owned experts, zero tokens for
some (not all) local experts on a rank, and heavily skewed-but-nonzero load
-- none of which the existing EP tests exercise deliberately.

Why the existing tests don't cover this (found during investigation, not
obvious from a first read): moe_ep_dispatch_core.py and
test_moe_ep_dispatch_decode.py both tag tokens as row_i = (i+1) * ones(H).
Since every row is a POSITIVE SCALAR MULTIPLE of the same vector, and
gate(x) = x @ gate.weight.T is linear, positive scaling can never change
which coordinates (expert logits) are largest -- so EVERY token in those
tests routes to the SAME top-k expert set (whichever the random gate ranks
highest). Load distribution across ranks in those tests is an accidental
side effect of where that one fixed set happens to land under round-robin,
never deliberately constructed or verified.

This file breaks that degeneracy directly: gate.weight is set to the
identity matrix (hidden_size == num_experts here, so this is square), and
each token's input row is a 0/1 indicator vector with exactly top_k ones at
hand-chosen coordinate positions. Since gate(x) = x @ I.T = x exactly,
torch.topk(gate(x), top_k) deterministically selects EXACTLY the chosen
coordinates -- full per-token routing control, through the REAL unmodified
gate/topk/softmax/dispatch/combine code, no reimplementation, no
monkeypatching (same "manipulate real weights, call the real forward()"
philosophy moe_ep_dispatch_core.py's _isolate_routed_expert_output already
uses for shared_expert_gate).

Scale: num_experts=16, top_k=8, ep_size=2 (8 local experts/rank), matching
test_moe_ep_dispatch_topk8.py's existing scale for continuity. Round-robin:
expert e -> rank e%2, local slot e//2. So rank0 owns evens {0,2,...,14}
(local slots 0-7), rank1 owns odds {1,3,...,15} (local slots 0-7).

Three scenarios, each run through BOTH the prefill EP path
(_forward_dispatch_ep) and the decode EP path (_forward_gathered_ep) --
maintaining the same equivalent-coverage standard already established
between those two methods elsewhere in this project's EP tests:

  A "rank1_gets_zero": every token picks top-8 = all 8 evens. rank1's
     owned-pair count is exactly 0 across ALL 8 of its local experts --
     exercises the `if cnt == 0: continue` branch for an entire rank's
     worth of local experts simultaneously, and confirms all_reduce
     correctly folds in an all-zero contribution from that rank.

  B "partial_zero_diverse": 3 tokens with DIFFERENT picks (genuine
     per-token diversity, not the same set repeated) constructed so that
     on rank0, local slot 7 (expert 14) is never picked (cnt=0) while
     slots 0-6 each get exactly 3; on rank1, slots 3-7 are never picked
     while slots 0/1/2 each get exactly 1 (from a different token each).
     Exercises zero-count for SOME but not all local experts on a rank,
     simultaneously with genuine per-token routing diversity.

  C "skewed_nonzero": 4 tokens, 3 of which pick all-evens and 1 of which
     picks 7 evens + 1 odd. rank0 ends up with 31 owned pairs, rank1 with
     exactly 1 -- confirms CORRECTNESS (not speed) under heavy, deliberate
     imbalance where dropless dispatch means unbounded per-rank load.

Each scenario's expected per-rank owned-pair counts are computed
independently in pure Python from the hand-chosen picks (not from the
model) and printed for transparency -- confirming the scenario actually
has the property it claims. The PASS/FAIL evidence itself is the same as
the rest of this project's EP tests: routed-expert-only output compared
against a single-process (ep_size=1) reference using the real unmodified
_forward_dispatch/_forward_gathered, plus the token-id round-trip via the
real ownership mask (_last_ep_token_id_roundtrip) -- not internal-state
introspection of the loop.

Bitwise exactness is NOT asserted (top_k=8, same reasoning as
test_moe_ep_dispatch_topk8.py -- not proven for 3+ term sums) -- measured
and sanity-bounded, same discipline as the rest of this project's top_k=8
EP tests.

Usage:
    python tests/test_moe_ep_dispatch_edge_cases.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_ep_dispatch_core import _relative_error  # noqa: E402

NUM_EXPERTS = 16
TOP_K = 8
HIDDEN = NUM_EXPERTS  # square gate.weight -> identity trick
INTERMEDIATE = 4
SHARED_INTERMEDIATE = 6
EP_SIZE = 2

SCENARIOS = {
    "rank1_gets_zero": [
        [0, 2, 4, 6, 8, 10, 12, 14],
        [0, 2, 4, 6, 8, 10, 12, 14],
        [0, 2, 4, 6, 8, 10, 12, 14],
    ],
    "partial_zero_diverse": [
        [0, 2, 4, 6, 8, 10, 12, 1],
        [0, 2, 4, 6, 8, 10, 12, 3],
        [0, 2, 4, 6, 8, 10, 12, 5],
    ],
    "skewed_nonzero": [
        [0, 2, 4, 6, 8, 10, 12, 14],
        [0, 2, 4, 6, 8, 10, 12, 14],
        [0, 2, 4, 6, 8, 10, 12, 14],
        [0, 2, 4, 6, 8, 10, 12, 1],
    ],
}


def _expected_local_counts(picks_per_token, ep_size):
    """Pure-Python, independent of the model -- confirms what each
    scenario's owned-pair counts per (rank, local_slot) SHOULD be, from the
    same round-robin formula (expert e -> rank e%ep_size, slot e//ep_size)
    utils/loader.py's expert_local_slot uses. Printed, not asserted against
    the model -- the model's correctness is judged by output comparison,
    this is confirmation the scenario construction itself is right."""
    counts = {r: {} for r in range(ep_size)}
    for picks in picks_per_token:
        for e in picks:
            r, slot = e % ep_size, e // ep_size
            counts[r][slot] = counts[r].get(slot, 0) + 1
    return counts


def _build_x(picks_per_token, hidden):
    rows = []
    for picks in picks_per_token:
        assert len(picks) == TOP_K, f"expected exactly {TOP_K} picks, got {len(picks)}: {picks}"
        assert len(set(picks)) == TOP_K, f"picks must be distinct: {picks}"
        row = torch.zeros(hidden)
        row[picks] = 1.0
        rows.append(row)
    return torch.stack(rows, dim=0)


def build_reference(x, use_decode, ref_port):
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{ref_port}", rank=0, world_size=1)

    from nanovllm.models.qwen3_5 import Qwen35MoE

    moe_ref = Qwen35MoE(HIDDEN, INTERMEDIATE, SHARED_INTERMEDIATE, NUM_EXPERTS, TOP_K)
    assert moe_ref.ep_size == 1
    assert moe_ref.experts.num_experts == NUM_EXPERTS

    torch.manual_seed(1234)
    for name in sorted(dict(moe_ref.named_parameters()).keys()):
        p = dict(moe_ref.named_parameters())[name]
        p.data.normal_(mean=0.0, std=0.02)
    # Overwrite the (randomly-initialized) gate with an exact identity --
    # this is what actually gives per-token routing control. Everything
    # else (experts, shared_expert, shared_expert_gate) stays randomly
    # initialized, matching the rest of this project's EP tests.
    moe_ref.gate.weight.data.copy_(torch.eye(NUM_EXPERTS))

    dense_method = moe_ref._forward_gathered if use_decode else moe_ref._forward_dispatch
    reference_out = dense_method(x)

    orig = moe_ref.shared_expert_gate.weight.data.clone()
    moe_ref.shared_expert_gate.weight.data.fill_(-1000.0)
    try:
        reference_routed_only = dense_method(x)
    finally:
        moe_ref.shared_expert_gate.weight.data.copy_(orig)

    ref_state = {k: v.clone() for k, v in moe_ref.state_dict().items()}
    dist.destroy_process_group()
    return ref_state, reference_out, reference_routed_only


def worker(rank, x, use_decode, tmp_path, worker_port):
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{worker_port}", rank=rank, world_size=EP_SIZE)

    from nanovllm.models.qwen3_5 import Qwen35MoE
    from nanovllm.utils.loader import shard_experts_tensor

    payload = torch.load(tmp_path, weights_only=True)
    ref_state = payload["ref_state"]
    reference_out = payload["reference_out"]
    reference_routed_only = payload["reference_routed_only"]

    moe_local = Qwen35MoE(HIDDEN, INTERMEDIATE, SHARED_INTERMEDIATE, NUM_EXPERTS, TOP_K,
                          debug_ep_token_roundtrip=True)
    assert moe_local.ep_size == EP_SIZE
    assert moe_local.ep_rank == rank
    assert moe_local.experts.num_experts == NUM_EXPERTS // EP_SIZE

    moe_local.gate.weight.weight_loader(moe_local.gate.weight, ref_state["gate.weight"])
    moe_local.shared_expert_gate.weight.weight_loader(
        moe_local.shared_expert_gate.weight, ref_state["shared_expert_gate.weight"]
    )
    gate_half, up_half = ref_state["shared_expert.gate_up_proj.weight"].chunk(2, dim=0)
    moe_local.shared_expert.gate_up_proj.weight.weight_loader(moe_local.shared_expert.gate_up_proj.weight, gate_half, 0)
    moe_local.shared_expert.gate_up_proj.weight.weight_loader(moe_local.shared_expert.gate_up_proj.weight, up_half, 1)
    moe_local.shared_expert.down_proj.weight.weight_loader(
        moe_local.shared_expert.down_proj.weight, ref_state["shared_expert.down_proj.weight"]
    )
    gu_shard = shard_experts_tensor(ref_state["experts.gate_up_proj"], rank, EP_SIZE)
    dn_shard = shard_experts_tensor(ref_state["experts.down_proj"], rank, EP_SIZE)
    moe_local.experts.gate_up_proj.data.copy_(gu_shard)
    moe_local.experts.down_proj.data.copy_(dn_shard)

    ep_method = moe_local._forward_gathered_ep if use_decode else moe_local._forward_dispatch_ep
    ep_out = ep_method(x)

    max_abs_diff_full = (ep_out - reference_out).abs().max().item()
    is_bitwise_full = torch.equal(ep_out, reference_out)
    print(f"[rank{rank}] FULL output max abs diff vs reference: {max_abs_diff_full:.3e}  "
          f"bitwise-equal={is_bitwise_full}")

    orig = moe_local.shared_expert_gate.weight.data.clone()
    moe_local.shared_expert_gate.weight.data.fill_(-1000.0)
    try:
        ep_routed_only = ep_method(x)
    finally:
        moe_local.shared_expert_gate.weight.data.copy_(orig)

    max_abs_diff_routed = (ep_routed_only - reference_routed_only).abs().max().item()
    rel_err_routed = _relative_error(ep_routed_only, reference_routed_only)
    is_bitwise_routed = torch.equal(ep_routed_only, reference_routed_only)
    print(f"[rank{rank}] ROUTED-ONLY max abs diff={max_abs_diff_routed:.3e}  "
          f"max relative error={rel_err_routed:.3e}  bitwise-equal={is_bitwise_routed}")

    assert torch.allclose(ep_routed_only, reference_routed_only, rtol=1e-3, atol=1e-6), (
        f"[rank{rank}] SANITY BOUND VIOLATED: max abs diff={max_abs_diff_routed:.3e} -- "
        f"too large to be ordinary fp reassociation, looks like a real indexing/pairing bug "
        f"specific to this zero/skew scenario"
    )
    print(f"[rank{rank}] HIDDEN-STATE CHECK OK -- bitwise-equal={is_bitwise_routed}, "
          f"max abs diff={max_abs_diff_routed:.3e}")

    token_id_roundtrip = moe_local._last_ep_token_id_roundtrip
    expected_ids = torch.arange(x.shape[0], dtype=torch.int64)
    assert torch.equal(token_id_roundtrip, expected_ids), (
        f"[rank{rank}] TOKEN-ID ROUND-TRIP MISMATCH: got {token_id_roundtrip.tolist()} "
        f"expected {expected_ids.tolist()}"
    )
    print(f"[rank{rank}] TOKEN-ID ROUND-TRIP OK")

    dist.destroy_process_group()


def run_scenario(name, picks_per_token, use_decode, ref_port, worker_port):
    path_label = "decode (_forward_gathered_ep)" if use_decode else "prefill (_forward_dispatch_ep)"
    print("\n" + "=" * 70)
    print(f"Scenario '{name}' -- {path_label}")
    print("=" * 70)

    expected = _expected_local_counts(picks_per_token, EP_SIZE)
    for r in range(EP_SIZE):
        zero_slots = [s for s in range(NUM_EXPERTS // EP_SIZE) if expected[r].get(s, 0) == 0]
        print(f"  expected rank{r} local-expert owned-pair counts: {expected[r]} "
              f"(zero-count local slots: {zero_slots or 'none'})")

    x = _build_x(picks_per_token, HIDDEN)
    ref_state, reference_out, reference_routed_only = build_reference(x, use_decode, ref_port)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save({
        "ref_state": ref_state, "reference_out": reference_out,
        "reference_routed_only": reference_routed_only,
    }, tmp_path)

    try:
        mp.spawn(worker, args=(x, use_decode, tmp_path, worker_port), nprocs=EP_SIZE, join=True)
    finally:
        os.unlink(tmp_path)

    print(f"Scenario '{name}' ({path_label}): ALL CHECKS PASSED")


def main():
    port = 29700
    for name, picks in SCENARIOS.items():
        for use_decode in (False, True):
            port += 2
            run_scenario(name, picks, use_decode, ref_port=port, worker_port=port + 1)

    print("\n" + "=" * 70)
    print("ALL SCENARIOS x ALL PATHS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
