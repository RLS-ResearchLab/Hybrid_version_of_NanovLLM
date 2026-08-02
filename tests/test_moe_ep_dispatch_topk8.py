"""top_k=8 extension of Q5 / Checkpoint 3: re-run the exact same design
(tagged tokens, token_id round-trip, dual hidden-state/id checks) at a
synthetic scale matching the real model's top_k=8 (16 experts, top_k=8,
tp_size=2 -> 8 local experts/rank). Still CPU-only, no GPU.

Unlike the top_k=2 checkpoint, bitwise exactness of the routed-expert-only
hidden-state output is NOT assumed here -- the top_k==2 proof relied on a
token's total being a sum of exactly 2 finite floats (exactly commutative
under IEEE754). At top_k=8, up to 8 terms can be summed per token, split
unevenly across ranks and accumulated via index_add_ in expert-loop order on
each rank vs. whatever internal reduction order the dense reference's
`.reshape(N, TK, H).sum(dim=1)` uses over 8 elements -- floating-point
addition is not associative for 3+ terms, so this is a genuinely open
question, not an extrapolation from top_k=2. The result is measured and
reported as-is; the token_id round-trip is still asserted exact regardless
(that check has no floating-point content -- it's integer scatter/gather
correctness, unaffected by top_k).

Usage: python tests/test_moe_ep_dispatch_topk8.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_ep_dispatch_core import MoEScale, run

SCALE = MoEScale(
    hidden=6,
    intermediate=4,
    shared_intermediate=6,
    num_experts=16,
    top_k=8,
    n_tokens=8,
    ep_size=2,
    ref_port=29611,
    worker_port=29612,
    assert_bitwise_routed=False,  # measured, not assumed -- see module docstring
)

if __name__ == "__main__":
    run(SCALE)
