"""Q5 / Checkpoint 3: expert-parallel dispatch/combine correctness.

Scale: 4 experts, top_k=2, tp_size=2 (2 local experts/rank under round-robin,
matching Q4's utils.loader.expert_local_slot). Bitwise exactness is asserted
here -- proven for top_k==2 (sum of exactly 2 finite floats is exactly
commutative). See tests/moe_ep_dispatch_core.py for the full design writeup
and tests/test_moe_ep_dispatch_topk8.py for the top_k=8 extension, where
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
