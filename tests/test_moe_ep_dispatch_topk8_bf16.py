"""bf16 variant of the top_k=8 extension: same scale as
test_moe_ep_dispatch_topk8.py (16 experts, top_k=8, tp_size=2, 8 local
experts/rank, hidden=16 -- required by the identity-gate routing-control
technique, see that file's docstring and moe_ep_dispatch_core.py's module
docstring for why hidden must equal num_experts and why this replaced the
old collinear row_i=(i+1)*ones(6) tagging), but ALL tensors -- inputs,
weights, and both the EP-dispatch computation and the dense reference --
explicitly cast to torch.bfloat16, not just one side.

Pure measurement: no sanity-bound assertion on the hidden-state closeness
(enforce_sanity_bound=False). The fp32 run's rtol=1e-3/atol=1e-6 bound isn't
known to be the right bound for bf16 -- that's the open design question this
run's measured number is meant to inform, not something to silently reuse.
The token_id round-trip is still asserted exact (integer-only, dtype-
independent).

Usage: python tests/test_moe_ep_dispatch_topk8_bf16.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_ep_dispatch_core import MoEScale, run

SCALE = MoEScale(
    hidden=16,
    intermediate=4,
    shared_intermediate=6,
    num_experts=16,
    top_k=8,
    n_tokens=8,
    ep_size=2,
    ref_port=29621,
    worker_port=29622,
    assert_bitwise_routed=False,
    dtype=torch.bfloat16,
    enforce_sanity_bound=False,
)

if __name__ == "__main__":
    run(SCALE)
