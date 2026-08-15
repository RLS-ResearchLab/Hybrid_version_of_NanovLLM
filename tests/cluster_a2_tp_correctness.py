"""A2 -- TP=4 (and TP=2) full-engine correctness vs. an independent
single-process reference. HIGHEST-RISK PROVISIONAL ITEM in this project per
the cluster-day task: shard-selection math was verified in isolation
(CPU-only, all four TP-aware weight_loaders -- tests/test_tp_shard_loader.py,
tests/test_load_model_tp.py), but the full multi-process ENGINE has not been
re-confirmed since the last dual-GPU session ended. Every TP-dependent
design decision since then is provisional until this runs.

Covers construction, weight loading, prefill, decode, and output agreement
(cosine + argmax), in that order, against a REFERENCE -- for the real
checkpoint, "single-process reference" cannot mean an actual
tensor_parallel_size=1 nano-vLLM run: the real checkpoint is ~69GB bf16,
larger than a single A6000's 48GB, so tp=1 does not fit on one GPU
regardless of any sharding-logic correctness question. This project's
existing real-checkpoint validation scripts (reference_check_phase6.py,
gsm8k_prefill_logits_check.py, gsm8k_decode_vs_hf_check.py) already solved
this the same way: an INDEPENDENT reference via plain
`transformers.AutoModelForCausalLM.from_pretrained(..., device_map="auto")`
(accelerate auto-shards across whatever GPUs are visible), run in ITS OWN
process, never sharing GPU memory with the nano-vLLM engine at the same
time. Reused here rather than re-derived.

PROMPTS below are reference_check_phase6.py's own varied-length/content set
(short factual completion, GSM8K-style arithmetic, bare numeric expression,
longer narrative, multi-clause sentence) -- deliberately reused rather than
picked fresh, both for continuity with that established check and because
they already satisfy this task's "genuinely varied inputs" requirement (no
two are collinear/near-duplicate).

PASS/FAIL THRESHOLDS -- stated here, before running, matching this
project's existing bars: prefill cosine >= 0.99 AND argmax match, per
prompt (reference_check_phase6.py's bar). Decode: first-token argmax must
match for every prompt (gsm8k_decode_vs_hf_check.py's bar -- the one
position with no chaotic-amplification excuse); a later-token divergence is
diagnostic, not an automatic fail, since temperature=0 argmax feeds each
token back in and a single near-tied flip can send two independently-computed
trajectories down different but equally-valid paths.

GPU MEMORY -- three separate process invocations, each exiting fully before
the next starts (same reason as the scripts above: the HF reference and this
engine cannot coexist in GPU memory):

    python tests/cluster_a2_tp_correctness.py --phase reference
    python tests/cluster_a2_tp_correctness.py --phase engine --tp 4
    python tests/cluster_a2_tp_correctness.py --phase engine --tp 2
    python tests/cluster_a2_tp_correctness.py --phase compare --tp 4
    python tests/cluster_a2_tp_correctness.py --phase compare --tp 2

Dry run (small model, single GPU, tp=1 -- see this task's cluster-day prep;
--phase reference is SKIPPED for the dry run, see --dry-run-no-hf-reference):

    python tests/cluster_a2_tp_correctness.py --phase engine --tp 1 \\
        --checkpoint tests/fake_qwen35_small --dry-run-no-hf-reference --fake-config-loader
    python tests/cluster_a2_tp_correctness.py --phase compare --tp 1 --dry-run-no-hf-reference

Intermediate artifacts land in tests/_cluster_day_cache/a2_tp/ (repo-relative,
safe to delete after --phase compare).
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

DEFAULT_CKPT = os.path.join(ROOT, "qwen35_checkpoint")
CACHE_DIR = os.path.join(ROOT, "tests", "_cluster_day_cache", "a2_tp")
REFERENCE_PATH = os.path.join(CACHE_DIR, "reference.pt")
MAX_MODEL_LEN = 2048

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class _AttrDict(dict):
    """Same shim bench_throughput.py / gsm8k_fused_gdr_check.py /
    test_qwen35_preemption_state.py already use -- needed ONLY against
    tests/fake_qwen35_small, pass --fake-config-loader for that. Root cause
    (confirmed directly on the real cluster GPU box, not guessed): the
    installed transformers registers "qwen3_5_moe" with a Qwen3_5MoeConfig
    that nests a `text_config` sub-object carrying its OWN real-checkpoint-
    shaped default `layer_types` (40 entries). The fake model's flat
    8-layer config.json never overrides text_config, so config.py's
    __post_init__ flattening copies that 40-entry default onto the
    top-level object next to num_hidden_layers=8 -- exactly what crashes
    models/qwen3_5.py's `_get_layer_types`'s `assert len(layers_block_type)
    == num_layers`. Bypassing AutoConfig entirely (read config.json as a
    flat dict, no text_config, no nested defaults) sidesteps this. The REAL
    checkpoint's config.json already has all fields flat at the top level
    (no nested text_config -- it's a text-only MoE checkpoint, not a VLM),
    so this is never needed there -- default OFF.
    """

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _fake_from_pretrained(path, *args, **kwargs):
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = _AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d


def _maybe_install_fake_config_loader(args):
    if getattr(args, "fake_config_loader", False):
        import nanovllm.config as config_mod
        config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)
        print("--fake-config-loader: AutoConfig.from_pretrained monkeypatched to read "
              "config.json as a flat dict (see _AttrDict's docstring)")
DECODE_MAX_TOKENS = 10  # short on purpose -- isolating math agreement, not a real eval

# Same set as reference_check_phase6.py -- varied length/content, kept
# identical for continuity with that already-established check.
PROMPTS = [
    "The capital of France is",
    "If John has 3 apples and buys 5 more, he has",
    "2 + 2 =",
    "In a small village, there lived a farmer who had",
    "The quick brown fox jumps over the lazy dog. This sentence is famous because",
]

COSINE_SIM_THRESHOLD = 0.99


def _engine_path(tp: int) -> str:
    return os.path.join(CACHE_DIR, f"engine_tp{tp}.pt")


def phase_reference(args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading INDEPENDENT HF reference from {args.checkpoint} (device_map='auto') ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    for prompt in PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        prefill_logits = out.logits[0, -1, :].float().cpu()

        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=DECODE_MAX_TOKENS, do_sample=False, num_beams=1,
                temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id,
            )
        decode_token_ids = gen[0, input_ids.shape[1]:].tolist()
        print(f"  [ref] {prompt!r} -> prompt_tokens={len(prompt_ids)}  "
              f"decode_tokens={len(decode_token_ids)}")
        results.append({
            "prompt": prompt, "prompt_ids": prompt_ids,
            "prefill_logits": prefill_logits, "decode_token_ids": decode_token_ids,
        })

    torch.save({"results": results}, REFERENCE_PATH)
    print(f"\nSaved HF reference to {REFERENCE_PATH}")
    print("PHASE COMPLETE -- exit this process fully before running --phase engine")


def phase_engine(args):
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Rank0-only weight-loader DISPATCH check, folded into construction
    # rather than a separate phase (a separate real-weight-loading phase
    # would just re-pay the ~69GB load cost for no new signal): tracks
    # whether ANY parameter falls through to default_weight_loader instead
    # of its own TP-aware weight_loader, at REAL config/REAL checkpoint/REAL
    # world_size for the first time (test_load_model_tp.py already proves
    # the DISPATCH mechanism itself is correct, but only against a tiny
    # synthetic checkpoint). Rank0-only: LLMEngine constructs rank0's
    # ModelRunner in THIS process (engine/llm_engine.py) but spawns rank>0 as
    # SEPARATE processes via torch.multiprocessing -- a monkeypatch here
    # cannot see across that boundary. The mechanism is rank-symmetric by
    # construction (identical load_model()/weight_loader code runs on every
    # rank), so this is corroborating evidence at real scale, not independent
    # proof for ranks>1; output agreement below (which reflects EVERY rank's
    # contribution once sharded weights combine to produce a result) is the
    # load-bearing signal for full-engine weight-loading correctness.
    import nanovllm.utils.loader as loader_mod
    called_via_default = []
    real_default = loader_mod.default_weight_loader

    def _tracking_default(param, loaded_weight):
        called_via_default.append(id(param))
        return real_default(param, loaded_weight)

    loader_mod.default_weight_loader = _tracking_default

    _maybe_install_fake_config_loader(args)
    from nanovllm.llm import LLM
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams

    print(f"Constructing engine: tensor_parallel_size={args.tp} (== ep_size) from {args.checkpoint} ...")
    llm = LLM(
        args.checkpoint,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=len(PROMPTS),
        max_model_len=MAX_MODEL_LEN,
    )
    loader_mod.default_weight_loader = real_default

    # Every leaf nn.Parameter on rank0's model MUST have a real weight_loader
    # attribute (set by ColumnParallelLinear/RowParallelLinear/ReplicatedLinear/
    # MergedColumnParallelLinear's __init__, or explicitly by Experts/norm
    # modules) -- anything that fell through to default_weight_loader for a
    # module type that's supposed to be TP-aware would be a real dispatch bug.
    # Report the raw count; this is diagnostic (see docstring on rank-scope),
    # not an automatic fail by itself -- default_weight_loader IS the correct
    # path for plain (non-sharded) parameters like norm weights.
    print(f"[weight-loader dispatch, rank0 only] {len(called_via_default)} parameter(s) loaded "
          f"via default_weight_loader (expected for plain/non-sharded params; cross-check the "
          f"names below by eye if this count looks larger than the model's norm/bias parameter count)")

    results = []
    for prompt in PROMPTS:
        prompt_ids = llm.tokenizer.encode(prompt)
        seq = Sequence(prompt_ids)
        seq.num_scheduled_tokens = len(prompt_ids)
        # See reference_check_phase6.py's identical lines for why: an empty
        # block_table is treated as the warmup case and skips slot_mapping.
        llm.scheduler.block_manager.allocate(seq, 0)
        if llm.model_runner.state_manager is not None:
            llm.model_runner.call("allocate_state_slot", seq)

        prefill_logits = llm.model_runner.call("get_prefill_logits", [seq])
        assert prefill_logits is not None, "expected rank0 to return gathered logits"

        sp = SamplingParams(temperature=0, max_tokens=DECODE_MAX_TOKENS)
        decode_out = llm.generate([prompt], sp, use_tqdm=False)
        decode_token_ids = decode_out[0]["token_ids"]

        print(f"  [engine tp={args.tp}] {prompt!r} -> prompt_tokens={len(prompt_ids)}  "
              f"decode_tokens={len(decode_token_ids)}")
        results.append({
            "prompt": prompt, "prompt_ids": prompt_ids,
            "prefill_logits": prefill_logits.float().cpu(), "decode_token_ids": decode_token_ids,
        })

    torch.save({
        "tp": args.tp, "results": results,
        "weight_loader_dispatch_default_count": len(called_via_default),
    }, _engine_path(args.tp))
    print(f"\nSaved engine (tp={args.tp}) results to {_engine_path(args.tp)}")
    print("PHASE COMPLETE -- exit this process fully before running --phase compare")


def phase_compare(args):
    ref_path = REFERENCE_PATH
    eng_path = _engine_path(args.tp)
    if args.dry_run_no_hf_reference:
        print("--dry-run-no-hf-reference: SKIPPING comparison against an HF reference (none was "
              "generated -- fake_qwen35_small has random untrained weights, so an HF reference for "
              "it would be meaningless anyway). This dry run only confirms --phase engine itself "
              "runs end-to-end (construction, weight loading, prefill, decode) without crashing.")
        assert os.path.exists(eng_path), f"missing {eng_path} -- run --phase engine first"
        payload = torch.load(eng_path, weights_only=False)
        print(f"Loaded engine(tp={payload['tp']}) results: {len(payload['results'])} prompt(s), "
              f"weight_loader_dispatch_default_count={payload['weight_loader_dispatch_default_count']}")
        for r in payload["results"]:
            assert torch.isfinite(r["prefill_logits"]).all(), f"non-finite prefill logits for {r['prompt']!r}"
            assert len(r["decode_token_ids"]) > 0, f"zero decode tokens for {r['prompt']!r}"
        print("DRY-RUN CHECK: PASS -- construction, weight loading, prefill, and decode all "
              "completed and produced finite, non-empty output for every prompt.")
        return

    assert os.path.exists(ref_path), f"missing {ref_path} -- run --phase reference first"
    assert os.path.exists(eng_path), f"missing {eng_path} -- run --phase engine --tp {args.tp} first"

    ref = torch.load(ref_path, weights_only=False)["results"]
    payload = torch.load(eng_path, weights_only=False)
    eng = payload["results"]
    tp = payload["tp"]

    assert len(ref) == len(eng), f"prompt count mismatch: reference={len(ref)} engine={len(eng)}"

    print("\n" + "=" * 78)
    print(f"A2 COMPARISON -- engine (tensor_parallel_size={tp}) vs. independent dense HF reference")
    print("=" * 78)

    n_prefill_pass = 0
    n_decode_first_token_pass = 0
    for rr, er in zip(ref, eng):
        assert rr["prompt"] == er["prompt"]
        ids_match = rr["prompt_ids"] == er["prompt_ids"]

        rl, el = rr["prefill_logits"], er["prefill_logits"]
        cos_sim = torch.nn.functional.cosine_similarity(rl.unsqueeze(0), el.unsqueeze(0)).item()
        argmax_match = int(rl.argmax()) == int(el.argmax())
        prefill_pass = cos_sim >= COSINE_SIM_THRESHOLD and argmax_match
        n_prefill_pass += int(prefill_pass)

        rd, ed = rr["decode_token_ids"], er["decode_token_ids"]
        first_token_match = len(rd) > 0 and len(ed) > 0 and rd[0] == ed[0]
        n_decode_first_token_pass += int(first_token_match)
        exact = rd == ed
        first_diff = next((i for i, (a, b) in enumerate(zip(rd, ed)) if a != b), None)

        print(f"\n  {rr['prompt']!r}")
        if not ids_match:
            print(f"    WARNING: tokenization mismatch between engine and HF for this prompt -- "
                  f"comparison below is confounded.")
        print(f"    PREFILL  cosine={cos_sim:.6f}  argmax_match={argmax_match}  "
              f"-> {'PASS' if prefill_pass else 'FAIL'} (threshold cosine>={COSINE_SIM_THRESHOLD})")
        print(f"    DECODE   ref={rd}")
        print(f"             eng={ed}")
        print(f"             exact_match={exact}  first_token_match={first_token_match}"
              + (f"  first_diff_at={first_diff}" if not exact and first_diff is not None else ""))

    n = len(ref)
    print("\n" + "-" * 78)
    print(f"PREFILL: {n_prefill_pass}/{n} prompts passed (cosine>={COSINE_SIM_THRESHOLD} AND argmax match)")
    print(f"DECODE:  {n_decode_first_token_pass}/{n} prompts passed (first-token argmax match)")
    print(f"weight_loader dispatch (rank0-only, diagnostic): "
          f"{payload['weight_loader_dispatch_default_count']} param(s) via default_weight_loader")

    overall_pass = (n_prefill_pass == n) and (n_decode_first_token_pass == n)
    print("\n" + "=" * 78)
    print(f"A2 (tensor_parallel_size={tp}) VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 78)
    if not overall_pass:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["reference", "engine", "compare"])
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size (== ep_size). Required for "
                     "--phase engine/compare, ignored for --phase reference.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--dry-run-no-hf-reference", action="store_true", default=False,
                     help="For --phase compare only: skip the HF-reference comparison and just "
                          "confirm --phase engine's own output is finite/non-empty. Use this for "
                          "single-GPU dry runs against the small fake model, where an HF reference "
                          "would be meaningless (random untrained weights).")
    ap.add_argument("--fake-config-loader", action="store_true", default=False,
                     help="Required for --phase engine against tests/fake_qwen35_small (default OFF -- "
                          "never needed against the real checkpoint). See _AttrDict's docstring for why.")
    args = ap.parse_args()

    if args.phase == "reference":
        phase_reference(args)
    elif args.phase == "engine":
        phase_engine(args)
    else:
        phase_compare(args)


if __name__ == "__main__":
    main()
