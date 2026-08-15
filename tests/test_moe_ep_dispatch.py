"""Q5 / Checkpoint 3: expert-parallel dispatch/combine correctness.

Scale: 4 experts, top_k=2, tp_size=2 (2 local experts/rank under round-robin,
matching Q4's utils.loader.expert_local_slot). hidden=4==num_experts is
required by moe_ep_dispatch_core.py's identity-gate routing-control
technique (each token gets a genuinely different, randomly-chosen top-2
expert pair rather than the old collinear row_i=(i+1)*ones(H) tagging under
which every token routed identically -- see that module's docstring).
Bitwise exactness is asserted here -- proven for top_k==2 (sum of exactly 2
finite floats is exactly commutative) REGARDLESS of routing pattern, so this
assertion is unaffected by the routing-construction fix. See
tests/moe_ep_dispatch_core.py for the full design writeup and
tests/test_moe_ep_dispatch_topk8.py for the top_k=8 extension, where
exactness is measured rather than assumed.

Usage: python tests/test_moe_ep_dispatch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_ep_dispatch_core import MoEScale, run

SCALE = MoEScale(
    hidden=4,
    intermediate=3,
    shared_intermediate=4,
    num_experts=4,
    top_k=2,
    n_tokens=6,
    ep_size=2,
    ref_port=29601,
    worker_port=29602,
    assert_bitwise_routed=True,
)

if __name__ == "__main__":
    run(SCALE)
