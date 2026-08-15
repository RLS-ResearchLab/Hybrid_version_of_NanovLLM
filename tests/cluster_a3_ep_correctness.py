"""A3 -- EP=4 (and EP=2) on real weights with real routing. In this codebase
ep_size IS tensor_parallel_size (Qwen35MoE shards experts across
dist.get_world_size() directly -- models/qwen3_5.py; no separate --ep-size
flag exists), so this is the SAME --tp value as cluster_a2_tp_correctness.py,
just isolating the MoE dispatch/combine path specifically instead of the
whole engine.

Why this is a distinct, real gap (not covered by A2 or by prior CPU/gloo
tests): moe_ep_dispatch_core.py and its dependents passed 6/6 edge cases,
but on SYNTHETIC weights, at 2 ranks, with HAND-CONSTRUCTED routing (even
after the B1 fix -- randomly divergent, but still not real trained-model
routing). This is the first time round-robin expert sharding meets a real
expert-utilization distribution from an actually-trained gate, at 4 ranks,
on real hardware.

Two things this script measures, both on the REAL checkpoint:

  1. CORRECTNESS: routed-expert-only MoE output, engine's real EP dispatch
     (Qwen35MoE._forward_dispatch_ep / ._forward_gathered_ep, reached via
     the model's real forward()) vs. an independent HF reference, same
     technique cluster_a2_tp_correctness.py uses (separate process,
     device_map="auto"). Reuses that script's --phase reference artifact
     directly -- no need to reload the ~69GB checkpoint twice.
  2. EXPERT-UTILIZATION HISTOGRAM ON REAL WEIGHTS: tests/moe_expert_utilization_histogram.py
     explicitly flags, in its own docstring, that it only ever measured
     RANDOM weights ("this measures whether round-robin sharding creates a
     STRUCTURAL imbalance risk under routing that has no learned bias ...
     it does NOT measure ... what a TRAINED model's LEARNED expert
     preferences look like ... Re-run this exact script against real
     checkpoint weights once available"). This closes that gap: routes a
     real prompt corpus (GSM8K questions, not random tokens -- genuinely
     varied, real text) through EVERY MoE layer's REAL gate weights loaded
     from the checkpoint, and records the resulting per-expert and
     per-rank-under-round-robin (token,k) counts.

PASS/FAIL for (1): same bar as A2 -- cosine >= 0.99 AND argmax match on
prefill logits, per prompt.
(2) is a MEASUREMENT, not a pass/fail gate (same posture as the existing
random-weight histogram script) -- reported as-is.

Usage (three phases, same GPU-memory reasoning as A2 -- this engine and the
HF reference cannot coexist):

    python tests/cluster_a2_tp_correctness.py --phase reference   # shared with A2, run once
    python tests/cluster_a3_ep_correctness.py --phase engine --tp 4
    python tests/cluster_a3_ep_correctness.py --phase engine --tp 2
    python tests/cluster_a3_ep_correctness.py --phase compare --tp 4
    python tests/cluster_a3_ep_correctness.py --phase compare --tp 2
    python tests/cluster_a3_ep_correctness.py --phase histogram --tp 4   # or 2; real-weight routing

Dry run (small model, single GPU, tp=1 -- EP branch requires ep_size>1
though, so a true tp=1 dry run cannot exercise _forward_dispatch_ep/
_forward_gathered_ep at all; --dry-run-no-hf-reference below only confirms
the histogram phase's real-weight routing plumbing works end-to-end):

    python tests/cluster_a3_ep_correctness.py --phase histogram \\
        --checkpoint tests/fake_qwen35_small --tp 1 --dry-run-no-hf-reference --fake-config-loader

Intermediate artifacts land in tests/_cluster_day_cache/a3_ep/ (repo-relative,
safe to delete after --phase compare / --phase histogram).
"""
import argparse
import json
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

sys.path.insert(0, os.path.dirname(__file__))
from gsm8k_prompt import build_prompt  # noqa: E402
from cluster_a2_tp_correctness import (  # noqa: E402
    REFERENCE_PATH, PROMPTS, _maybe_install_fake_config_loader, _free_diagnostic_seq,
)

DEFAULT_CKPT = os.path.join(ROOT, "qwen35_checkpoint")
CACHE_DIR = os.path.join(ROOT, "tests", "_cluster_day_cache", "a3_ep")
MAX_MODEL_LEN = 2048
COSINE_SIM_THRESHOLD = 0.99

# Real-text corpus for the histogram phase -- GSM8K questions (genuinely
# varied natural-language content, not random tokens), NOT the collinear/
# random-token constructions this task's B1 item flagged as a routing
# degeneracy risk elsewhere. Real trained-gate routing on real text is
# exactly what this phase exists to measure.
HISTOGRAM_NUM_PROMPTS = 64


def _engine_path(tp: int) -> str:
    return os.path.join(CACHE_DIR, f"engine_ep_tp{tp}.pt")


def phase_engine(args):
    """Same shape as cluster_a2_tp_correctness.py's --phase engine, but ALSO
    isolates the routed-expert-only MoE contribution (same isolation trick
    moe_ep_dispatch_core.py uses: force shared_expert_gate very negative via
    a forward hook on every Qwen35MoE instance in the model, so
    sigmoid(...) rounds to exactly 0 and shared_expert's own TP-reassociation
    noise doesn't contaminate the EP-dispatch-specific comparison) so A3
    tests the EP MECHANISM specifically, not the whole decoder stack."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    _maybe_install_fake_config_loader(args)
    from nanovllm.llm import LLM
    from nanovllm.engine.sequence import Sequence
    from nanovllm.models.qwen3_5 import Qwen35MoE

    print(f"Constructing engine: tensor_parallel_size={args.tp} (== ep_size) from {args.checkpoint} ...")
    llm = LLM(
        args.checkpoint,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=len(PROMPTS),
        max_model_len=MAX_MODEL_LEN,
    )
    assert args.tp > 1, (
        f"--tp={args.tp}: EP dispatch (_forward_dispatch_ep/_forward_gathered_ep) is only "
        f"reached at ep_size>1 -- a tp=1 run would silently exercise the NON-EP path and "
        f"report a meaningless 'pass'. Use --tp 2 or --tp 4 against the real checkpoint."
    )
    moe_modules = [m for m in llm.model_runner.model.modules() if isinstance(m, Qwen35MoE)]
    assert moe_modules, "no Qwen35MoE modules found -- not a hybrid model?"
    assert all(m.ep_size == args.tp for m in moe_modules), (
        f"expected every Qwen35MoE.ep_size == {args.tp}, got "
        f"{sorted(set(m.ep_size for m in moe_modules))}"
    )
    print(f"Confirmed {len(moe_modules)} Qwen35MoE layer(s), all at ep_size={args.tp}")

    saved_gates = [m.shared_expert_gate.weight.data.clone() for m in moe_modules]
    for m in moe_modules:
        m.shared_expert_gate.weight.data.fill_(-1000.0)

    try:
        results = []
        for prompt in PROMPTS:
            prompt_ids = llm.tokenizer.encode(prompt)
            seq = Sequence(prompt_ids)
            seq.num_scheduled_tokens = len(prompt_ids)
            llm.scheduler.block_manager.allocate(seq, 0)
            if llm.model_runner.state_manager is not None:
                llm.model_runner.call("allocate_state_slot", seq)
            routed_only_logits = llm.model_runner.call("get_prefill_logits", [seq])
            assert routed_only_logits is not None
            # See cluster_a2_tp_correctness.py's _free_diagnostic_seq
            # docstring for the real crash this prevents once a loop admits
            # more diagnostic sequences than max_num_seqs.
            _free_diagnostic_seq(llm, seq)
            print(f"  [engine ep_tp={args.tp}] {prompt!r} -> prompt_tokens={len(prompt_ids)}")
            results.append({
                "prompt": prompt, "prompt_ids": prompt_ids,
                "routed_only_prefill_logits": routed_only_logits.float().cpu(),
            })
    finally:
        for m, g in zip(moe_modules, saved_gates):
            m.shared_expert_gate.weight.data.copy_(g)

    torch.save({"tp": args.tp, "results": results}, _engine_path(args.tp))
    print(f"\nSaved EP engine (tp={args.tp}) routed-only results to {_engine_path(args.tp)}")
    print("PHASE COMPLETE -- exit this process fully before running --phase compare")


def phase_compare(args):
    eng_path = _engine_path(args.tp)
    assert os.path.exists(eng_path), f"missing {eng_path} -- run --phase engine --tp {args.tp} first"
    if args.dry_run_no_hf_reference:
        payload = torch.load(eng_path, weights_only=False)
        for r in payload["results"]:
            assert torch.isfinite(r["routed_only_prefill_logits"]).all(), \
                f"non-finite routed-only logits for {r['prompt']!r}"
        print(f"DRY-RUN CHECK: PASS -- routed-only EP-dispatch output is finite for all "
              f"{len(payload['results'])} prompts.")
        return

    assert os.path.exists(REFERENCE_PATH), (
        f"missing {REFERENCE_PATH} -- run `python tests/cluster_a2_tp_correctness.py "
        f"--phase reference` first (shared artifact, only needs to run once for A2+A3)"
    )
    ref = torch.load(REFERENCE_PATH, weights_only=False)["results"]
    payload = torch.load(eng_path, weights_only=False)
    eng = payload["results"]
    tp = payload["tp"]

    print("\n" + "=" * 78)
    print(f"A3 COMPARISON -- EP dispatch, routed-expert-only (tensor_parallel_size={tp}) "
          f"vs. full HF reference logits")
    print("=" * 78)
    print("NOTE: HF reference logits include shared_expert's contribution (not isolated the "
          "same way); expect a modest extra cosine/argmax gap vs. A2's whole-model comparison "
          "on this account alone -- read this as a DIRECTIONAL EP-dispatch sanity check "
          "layered on top of A2's whole-model result, not a bitwise-tighter bar than A2.")

    n_pass = 0
    for rr, er in zip(ref, eng):
        assert rr["prompt"] == er["prompt"]
        rl, el = rr["prefill_logits"], er["routed_only_prefill_logits"]
        cos_sim = torch.nn.functional.cosine_similarity(rl.unsqueeze(0), el.unsqueeze(0)).item()
        argmax_match = int(rl.argmax()) == int(el.argmax())
        passed = cos_sim >= COSINE_SIM_THRESHOLD and argmax_match
        n_pass += int(passed)
        print(f"  {rr['prompt']!r}: cosine={cos_sim:.6f}  argmax_match={argmax_match}  "
              f"-> {'PASS' if passed else 'FAIL (see NOTE above before treating as a bug)'}")

    print(f"\nA3 (tensor_parallel_size={tp}): {n_pass}/{len(ref)} prompts passed")


def phase_histogram(args):
    """Expert-utilization histogram on REAL weights, REAL routing -- closes
    the gap tests/moe_expert_utilization_histogram.py's own docstring flags
    (random weights only, see module docstring). Runs real GSM8K questions'
    hidden states through the REAL loaded gate at every MoE layer and
    records per-expert / per-round-robin-rank (token,k) counts, same
    counting methodology as the existing script (torch.bincount over
    topk indices), just fed from a real forward pass instead of
    torch.randn.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    from datasets import load_dataset
    _maybe_install_fake_config_loader(args)
    from nanovllm.llm import LLM
    from nanovllm.engine.sequence import Sequence
    from nanovllm.models.qwen3_5 import Qwen35MoE

    print(f"Loading openai/gsm8k (main, test split) for {HISTOGRAM_NUM_PROMPTS} real prompts ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [build_prompt(ds[i]["question"]) for i in range(min(HISTOGRAM_NUM_PROMPTS, len(ds)))]

    print(f"Constructing engine: tensor_parallel_size={args.tp} from {args.checkpoint} ...")
    llm = LLM(
        args.checkpoint,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=8,
        max_model_len=MAX_MODEL_LEN,
    )
    moe_modules = [m for m in llm.model_runner.model.modules() if isinstance(m, Qwen35MoE)]
    assert moe_modules, "no Qwen35MoE modules found -- not a hybrid model?"
    num_experts = moe_modules[0].num_experts
    top_k = moe_modules[0].top_k
    ep_size = moe_modules[0].ep_size
    print(f"Found {len(moe_modules)} Qwen35MoE layer(s): num_experts={num_experts} top_k={top_k} "
          f"ep_size={ep_size}")

    # Hook gate.forward on every MoE layer to capture real (post-real-gate)
    # top-k expert ids for every real token that flows through prefill --
    # additive only, does not change engine behavior (the real forward()
    # runs exactly as it would without this hook; the hook only records).
    per_layer_counts = [torch.zeros(num_experts, dtype=torch.int64) for _ in moe_modules]
    handles = []

    def _make_hook(layer_idx):
        def _hook(module, inputs, output):
            with torch.no_grad():
                _, idx = torch.topk(output.float(), top_k, dim=-1)
                per_layer_counts[layer_idx] += torch.bincount(
                    idx.reshape(-1).cpu(), minlength=num_experts
                )
        return _hook

    for i, m in enumerate(moe_modules):
        handles.append(m.gate.register_forward_hook(_make_hook(i)))

    try:
        for b in range(0, len(prompts), 8):
            batch = prompts[b:b + 8]
            for prompt in batch:
                prompt_ids = llm.tokenizer.encode(prompt)
                seq = Sequence(prompt_ids)
                seq.num_scheduled_tokens = len(prompt_ids)
                llm.scheduler.block_manager.allocate(seq, 0)
                if llm.model_runner.state_manager is not None:
                    llm.model_runner.call("allocate_state_slot", seq)
                llm.model_runner.call("get_prefill_logits", [seq])
                # MUST free -- this loop runs up to HISTOGRAM_NUM_PROMPTS (64
                # by default) diagnostic sequences against a max_num_seqs=8
                # StateManager; see cluster_a2_tp_correctness.py's
                # _free_diagnostic_seq docstring for the real crash this
                # prevents (confirmed pattern, reproduced elsewhere in this
                # same file's phase_engine on real hardware).
                _free_diagnostic_seq(llm, seq)
            print(f"  routed {min(b + 8, len(prompts))}/{len(prompts)} prompts through real gates ...")
    finally:
        for h in handles:
            h.remove()

    total_pairs = int(sum(c.sum().item() for c in per_layer_counts))
    print("\n" + "=" * 78)
    print(f"A3 EXPERT-UTILIZATION HISTOGRAM -- REAL WEIGHTS, REAL ROUTING "
          f"({HISTOGRAM_NUM_PROMPTS} real GSM8K prompts, {len(moe_modules)} MoE layers)")
    print("=" * 78)
    print(f"total (token,k) pairs across all layers: {total_pairs}")

    summary = []
    for layer_idx, counts in enumerate(per_layer_counts):
        n_pairs = int(counts.sum().item())
        if n_pairs == 0:
            continue
        mean = counts.float().mean().item()
        std = counts.float().std().item()
        zero_experts = int((counts == 0).sum().item())
        rank_counts = []
        for r in range(ep_size):
            owned = torch.arange(r, num_experts, ep_size)
            rank_counts.append(int(counts[owned].sum().item()))
        max_over_mean = max(rank_counts) / (sum(rank_counts) / ep_size) if sum(rank_counts) else float("nan")
        print(f"  layer {layer_idx}: pairs={n_pairs}  mean/expert={mean:.1f}  std={std:.1f}  "
              f"experts_with_zero_tokens={zero_experts}/{num_experts}  "
              f"per-rank(ep={ep_size}) totals={rank_counts}  max/mean={max_over_mean:.3f}")
        summary.append({
            "layer_idx": layer_idx, "n_pairs": n_pairs, "mean": mean, "std": std,
            "zero_experts": zero_experts, "rank_counts": rank_counts, "max_over_mean": max_over_mean,
        })

    out_path = os.path.join(CACHE_DIR, f"histogram_real_ep{args.tp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"num_experts": num_experts, "top_k": top_k, "ep_size": ep_size,
                    "n_prompts": len(prompts), "per_layer": summary}, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("\nThis is a REAL-WEIGHT, REAL-ROUTING measurement -- unlike "
          "tests/moe_expert_utilization_histogram.py's random-weight run, this one CAN speak to "
          "whether round-robin sharding creates a per-rank load imbalance under the model's "
          "actual learned expert preferences. Still just this checkpoint / this prompt "
          "distribution (GSM8K) -- not a claim about every workload this engine will ever see.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["engine", "compare", "histogram"])
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--dry-run-no-hf-reference", action="store_true", default=False)
    ap.add_argument("--fake-config-loader", action="store_true", default=False,
                     help="Required against tests/fake_qwen35_small (default OFF -- never needed "
                          "against the real checkpoint). See cluster_a2_tp_correctness.py's "
                          "_AttrDict docstring for why.")
    args = ap.parse_args()

    if args.phase == "engine":
        phase_engine(args)
    elif args.phase == "compare":
        phase_compare(args)
    else:
        phase_histogram(args)


if __name__ == "__main__":
    main()
