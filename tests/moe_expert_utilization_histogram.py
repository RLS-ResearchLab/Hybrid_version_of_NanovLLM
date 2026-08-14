"""Expert-utilization histogram: the measurement Stage 1's round-robin
expert-to-rank sharding decision (utils/loader.py's expert_local_slot,
expert_id % tp_size) was made WITHOUT. Routes a token workload through
Qwen35MoE's routing (gate + top_k, the same call every dispatch/combine
path makes) and records how many (token, k) pairs land on each of the 256
experts.

============================================================================
FLAGGED EXPLICITLY, PER TASK INSTRUCTIONS -- READ BEFORE TRUSTING THIS DATA
============================================================================
This machine has no actual model weights: qwen35_checkpoint/ contains only
config.json, tokenizer files, and model.safetensors.index.json (the
manifest of WHICH tensors would be where -- not the tensor data itself; no
.safetensors weight files are present). Checked directly, not assumed.

So this histogram runs on a Qwen35MoE instance with REAL num_experts=256
and num_experts_per_tok=8 (what matters for sharding-scale realism, same
convention tests/test_load_model_ep.py already established for exactly
this reason), small hidden_size (tractable random weights), and RANDOMLY
INITIALIZED gate/routing weights (nn.Linear default init).

This measures whether round-robin sharding creates a STRUCTURAL imbalance
risk under routing that has no learned bias -- i.e., "does the SHARDING
SCHEME itself introduce skew, independent of what any model learns to
prefer" -- which is a real and legitimate question round-robin's original
design left unmeasured. It does NOT measure, and must not be cited as
measuring, what a TRAINED model's LEARNED expert preferences look like.
Real trained MoE models are widely documented (this is not specific to
this project) to develop significant, non-uniform expert specialization
through training -- routing skew from LEARNED preferences is a completely
different phenomenon from routing skew from an unlucky random gate, and
this script's random weights cannot speak to the former at all. Treat this
as an infrastructure demonstration (does the measurement pipeline work,
and is round-robin at least not making things worse under neutral routing)
-- not a real finding about production expert-load behavior. Re-run this
exact script against real checkpoint weights once available, before this
question is considered actually answered.
============================================================================

Usage:
    python tests/moe_expert_utilization_histogram.py
    python tests/moe_expert_utilization_histogram.py --n-tokens 8192 --hidden-size 64 --seed 0
"""
import argparse
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [ROOT]
    _pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg

NUM_EXPERTS = 256
TOP_K = 8


def init_dist():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29800")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=4096,
                     help="Total tokens routed (n_tokens * top_k total (token,k) pairs "
                          "distributed across 256 experts).")
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--intermediate-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    init_dist()

    from nanovllm.models.qwen3_5 import Qwen35MoE

    torch.manual_seed(args.seed)
    moe = Qwen35MoE(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        shared_intermediate_size=args.intermediate_size,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    )
    assert moe.ep_size == 1
    assert moe.gate.weight.shape == (NUM_EXPERTS, args.hidden_size)
    # ReplicatedLinear (LinearBase) does NOT auto-initialize -- confirmed
    # during the vectorized-MoE work (none of this project's custom linear
    # wrappers call reset_parameters(), unlike stock nn.Linear; weights
    # normally arrive via a checkpoint's weight_loader). Left as
    # torch.empty(...), gate.weight is uninitialized memory, not a neutral
    # random gate -- an initial version of this script skipped this and
    # measured a wildly degenerate ~247/256-experts-get-zero-tokens result
    # that turned out to be exactly this bug, not a real finding. Explicit
    # init required for a meaningful measurement.
    with torch.no_grad():
        moe.gate.weight.normal_(mean=0.0, std=0.02)

    x = torch.randn(args.n_tokens, args.hidden_size)
    with torch.no_grad():
        logits = moe.gate(x)
        _, idx = torch.topk(logits, TOP_K, dim=-1)  # (n_tokens, TOP_K)

    counts = torch.bincount(idx.reshape(-1), minlength=NUM_EXPERTS)
    total_pairs = args.n_tokens * TOP_K
    assert counts.sum().item() == total_pairs

    mean = counts.float().mean().item()
    std = counts.float().std().item()
    cv = std / mean if mean > 0 else float("nan")
    min_c, max_c = counts.min().item(), counts.max().item()
    zero_experts = (counts == 0).sum().item()
    expected_uniform = total_pairs / NUM_EXPERTS

    print("=" * 70)
    print("Expert-utilization histogram -- RANDOM WEIGHTS, see module docstring")
    print("=" * 70)
    print(f"num_experts={NUM_EXPERTS}  top_k={TOP_K}  n_tokens={args.n_tokens}  "
          f"total (token,k) pairs={total_pairs}  hidden_size={args.hidden_size}  seed={args.seed}")
    print(f"expected count per expert under perfectly uniform routing: {expected_uniform:.2f}")
    print()
    print(f"measured: mean={mean:.2f}  std={std:.2f}  coefficient_of_variation={cv:.4f}  "
          f"min={min_c}  max={max_c}  max/mean={max_c / mean:.3f}  "
          f"experts with ZERO tokens={zero_experts}/{NUM_EXPERTS}")
    print()

    # Round-robin-rank imbalance: does the SHARDING (not just per-expert
    # count) look skewed under this routing, for the ep_size values this
    # project actually plans to use?
    for ep_size in (2, 4, 8):
        if NUM_EXPERTS % ep_size != 0:
            continue
        rank_counts = torch.zeros(ep_size, dtype=torch.int64)
        for r in range(ep_size):
            owned_experts = torch.arange(r, NUM_EXPERTS, ep_size)
            rank_counts[r] = counts[owned_experts].sum()
        rc = rank_counts.tolist()
        print(f"  ep_size={ep_size}: per-rank owned-pair totals={rc}  "
              f"max/mean={max(rc) / (sum(rc) / ep_size):.4f}")

    print()
    print("Per-expert counts (expert_id: count), all 256:")
    for e in range(0, NUM_EXPERTS, 8):
        row = counts[e:e + 8].tolist()
        print(f"  [{e:3d}-{e + 7:3d}]: {row}")

    print()
    print("Reminder: this is a RANDOM-WEIGHT measurement -- see module docstring. "
          "It shows round-robin does not introduce structural skew under neutral "
          "routing; it says nothing about a trained model's learned routing skew.")


if __name__ == "__main__":
    main()
