"""Correctness check for Qwen35MoE._forward_gathered_ep -- the EP-aware
DECODE path added to unblock GSM8K generation (see models/qwen3_5.py; this
used to raise NotImplementedError at ep_size>1, discovered by
tests/gsm8k_smoke_test.py dying on the first decode step after a clean
prefill).

Same design as tests/moe_ep_dispatch_core.py (which validates the sibling
PREFILL EP path, _forward_dispatch_ep, and is intentionally left unmodified
-- this is a separate file, not a shared-core edit, to avoid touching
already-validated tests):
  - Reference: the existing, unmodified Qwen35MoE._forward_gathered, run
    single-process (ep_size=1) -- the pre-existing, already-shipped decode
    method, no new code involved in producing it.
  - Test subject: Qwen35MoE._forward_gathered_ep, run via real
    multi-process torch.multiprocessing.spawn with gloo -- genuine
    multi-rank communication (dist.all_reduce), not a single-process trick.
  - Both methods are called DIRECTLY (not through forward()'s cu_seqlens-
    based is_decode routing) -- routing itself isn't under test here, only
    the two methods' numerical agreement.
  - Hidden-state rows: one-hot indicator vectors at randomly-chosen top-k
    coordinates, paired with an identity gate.weight (HIDDEN==NUM_EXPERTS)
    -- see moe_ep_dispatch_core.py's module docstring. FIXED post-hoc from
    the original row_i=(i+1)*ones(H) tagging: since this is an INDEPENDENT
    duplicate of moe_ep_dispatch_core.py (own build_reference/worker, not
    imported), it had the identical collinear-routing bug -- every token
    routed to the same top-k expert set regardless of N_TOKENS, so
    decode-EP's own reassociation number was never actually measured under
    divergent routing either. _build_divergent_x is imported from
    moe_ep_dispatch_core.py (not re-derived) to avoid a second copy of the
    same fix drifting out of sync.
  - Token-id round-trip (_last_ep_token_id_roundtrip) asserted exact,
    always -- integer-only, dtype-independent.
  - Routed-expert-only comparison (shared_expert_gate forced very negative,
    isolating the actual EP logic under test, same helper as the prefill
    core): bitwise-exact for top_k==2 (proven, same argument as the prefill
    path -- IEEE754 sum of exactly 2 finite floats is order-independent);
    MEASURED (not assumed) for top_k==8, still under a sanity bound loose
    enough to pass ordinary fp reassociation noise but tight enough to
    catch a real indexing/pairing bug.

CPU-only (gloo backend) -- no GPU needed, matches this repo's existing
moe_ep_dispatch_core.py convention. Safe to run on a dev machine with no
CUDA and no real checkpoint.

Usage:
    python tests/test_moe_ep_dispatch_decode.py
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
from moe_ep_dispatch_core import _relative_error, _build_divergent_x  # noqa: E402

HIDDEN = 16  # == NUM_EXPERTS, required by the identity-gate technique -- see module docstring
INTERMEDIATE = 4
SHARED_INTERMEDIATE = 6
NUM_EXPERTS = 16
TOP_K = 8
N_TOKENS = 8  # decode: N_TOKENS sequences, ONE token each (real decode shape)
EP_SIZE = 2
REF_PORT = 29631
WORKER_PORT = 29632


def build_reference():
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{REF_PORT}", rank=0, world_size=1)

    from nanovllm.models.qwen3_5 import Qwen35MoE

    moe_ref = Qwen35MoE(HIDDEN, INTERMEDIATE, SHARED_INTERMEDIATE, NUM_EXPERTS, TOP_K)
    assert moe_ref.ep_size == 1
    assert moe_ref.experts.num_experts == NUM_EXPERTS

    torch.manual_seed(1234)
    for name in sorted(dict(moe_ref.named_parameters()).keys()):
        p = dict(moe_ref.named_parameters())[name]
        p.data.normal_(mean=0.0, std=0.02)
    # Overwrite the (randomly-initialized) gate with an exact identity --
    # this is what gives per-token routing control, see module docstring.
    moe_ref.gate.weight.data.copy_(torch.eye(NUM_EXPERTS))

    x, picks = _build_divergent_x(N_TOKENS, NUM_EXPERTS, TOP_K)
    print(f"[main] per-token top-{TOP_K} picks (genuinely divergent routing): {picks}")

    reference_out = moe_ref._forward_gathered(x)
    print(f"[main] reference (dense, ep_size=1, _forward_gathered) output shape={tuple(reference_out.shape)}")

    # _isolate_routed_expert_output (imported from moe_ep_dispatch_core)
    # calls moe(x) -- forward()'s prefill routing, since no cu_seqlens is
    # passed -- which is NOT what we want for a decode-path check. Apply the
    # same isolation trick directly against _forward_gathered instead.
    orig = moe_ref.shared_expert_gate.weight.data.clone()
    moe_ref.shared_expert_gate.weight.data.fill_(-1000.0)
    try:
        reference_routed_only = moe_ref._forward_gathered(x)
    finally:
        moe_ref.shared_expert_gate.weight.data.copy_(orig)

    ref_state = {k: v.clone() for k, v in moe_ref.state_dict().items()}
    dist.destroy_process_group()
    return ref_state, x, reference_out, reference_routed_only


def worker(rank: int, tmp_path: str):
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{WORKER_PORT}", rank=rank, world_size=EP_SIZE)

    from nanovllm.models.qwen3_5 import Qwen35MoE
    from nanovllm.utils.loader import shard_experts_tensor

    payload = torch.load(tmp_path, weights_only=True)
    ref_state = payload["ref_state"]
    x = payload["x"]
    reference_out = payload["reference_out"]
    reference_routed_only = payload["reference_routed_only"]

    moe_local = Qwen35MoE(HIDDEN, INTERMEDIATE, SHARED_INTERMEDIATE, NUM_EXPERTS, TOP_K)
    assert moe_local.ep_size == EP_SIZE
    assert moe_local.ep_rank == rank
    assert moe_local.experts.num_experts == NUM_EXPERTS // EP_SIZE

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
    gu_shard = shard_experts_tensor(ref_state["experts.gate_up_proj"], rank, EP_SIZE)
    dn_shard = shard_experts_tensor(ref_state["experts.down_proj"], rank, EP_SIZE)
    moe_local.experts.gate_up_proj.data.copy_(gu_shard)
    moe_local.experts.down_proj.data.copy_(dn_shard)

    ep_out = moe_local._forward_gathered_ep(x)  # DECODE EP path under test

    max_abs_diff_full = (ep_out - reference_out).abs().max().item()
    rel_err_full = _relative_error(ep_out, reference_out)
    is_bitwise_full = torch.equal(ep_out, reference_out)
    print(f"[rank{rank}] FULL output (incl. shared_expert) dtype={ep_out.dtype} "
          f"max abs diff vs reference: {max_abs_diff_full:.3e}  "
          f"max relative error: {rel_err_full:.3e}  bitwise-equal={is_bitwise_full}")

    orig = moe_local.shared_expert_gate.weight.data.clone()
    moe_local.shared_expert_gate.weight.data.fill_(-1000.0)
    try:
        ep_routed_only = moe_local._forward_gathered_ep(x)
    finally:
        moe_local.shared_expert_gate.weight.data.copy_(orig)

    max_abs_diff_routed = (ep_routed_only - reference_routed_only).abs().max().item()
    rel_err_routed = _relative_error(ep_routed_only, reference_routed_only)
    is_bitwise_routed = torch.equal(ep_routed_only, reference_routed_only)
    print(f"[rank{rank}] ROUTED-EXPERT-ONLY (the actual EP decode logic under test) "
          f"dtype={ep_routed_only.dtype}: max abs diff={max_abs_diff_routed:.3e}  "
          f"max relative error={rel_err_routed:.3e}  bitwise-equal={is_bitwise_routed}")
    print(f"[rank{rank}] routed-expert-only ep ={ep_routed_only.tolist()}")
    print(f"[rank{rank}] routed-expert-only ref={reference_routed_only.tolist()}")

    # top_k == TOP_K == 8 here (matches the real checkpoint's routing width
    # per prior sessions' context) -- NOT proven bitwise-exact (that's only
    # established for top_k==2), so measure and enforce a generous sanity
    # bound rather than assert bitwise equality.
    assert torch.allclose(ep_routed_only, reference_routed_only, rtol=1e-3, atol=1e-6), (
        f"[rank{rank}] SANITY BOUND VIOLATED on routed-expert-only output: "
        f"max abs diff={max_abs_diff_routed:.3e} -- too large to be ordinary floating-point "
        f"reassociation, looks like a real indexing/pairing bug"
    )
    print(f"[rank{rank}] HIDDEN-STATE CHECK: sanity bound OK -- "
          f"bitwise-equal={is_bitwise_routed}, max abs diff={max_abs_diff_routed:.3e}, "
          f"max relative error={rel_err_routed:.3e}")

    token_id_roundtrip = moe_local._last_ep_token_id_roundtrip
    expected_ids = torch.arange(N_TOKENS, dtype=torch.int64)
    assert torch.equal(token_id_roundtrip, expected_ids), (
        f"[rank{rank}] TOKEN-ID ROUND-TRIP MISMATCH: got {token_id_roundtrip.tolist()} "
        f"expected {expected_ids.tolist()}"
    )
    print(f"[rank{rank}] TOKEN-ID ROUND-TRIP OK -- exact match to torch.arange({N_TOKENS})")

    dist.destroy_process_group()


def run():
    ref_state, x, reference_out, reference_routed_only = build_reference()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    torch.save({
        "ref_state": ref_state, "x": x,
        "reference_out": reference_out,
        "reference_routed_only": reference_routed_only,
    }, tmp_path)

    try:
        mp.spawn(worker, args=(tmp_path,), nprocs=EP_SIZE, join=True)
    finally:
        os.unlink(tmp_path)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    run()
