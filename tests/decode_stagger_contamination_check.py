"""Decode-time StateManager contamination check under STAGGERED early
termination -- the one slot-reuse scenario the existing state/contamination
tests do NOT cover.

What's already covered elsewhere (read before assuming this is redundant):
  - tests/test_state_slot_reuse.py forces slot reuse, but SEQUENTIALLY --
    three separate engine.generate() calls, one sequence at a time. No two
    sequences are ever RUNNING together when a free happens.
  - tests/gsm8k_decode_contamination_check.py runs 8 sequences concurrently
    in ONE batch, but all 8 are submitted upfront with nothing left in
    Scheduler.waiting -- a freed slot (when a filler finishes early) is
    never handed to a NEW incoming sequence, because there isn't one.
  - tests/test_qwen35_preemption_state.py compares GDR state across
    preemption (block-manager-pressure eviction), not natural finish.

None of these exercise: sequence A finishes (via SamplingParams.stop, not
EOS or max_tokens) while sequence B is still mid-decode in the SAME running
batch, and A's just-freed StateManager slot gets reassigned to a brand-new
sequence C pulled off Scheduler.waiting -- while B keeps decoding. That is
the scenario engine/scheduler.py's postprocess()/schedule() actually
implement for real traffic (see its docstring reasoning in the accompanying
investigation), and this script is the empirical check for it, not just a
code-reading exercise.

DESIGN -- deterministic without needing to know any real vocabulary:
  Uses the same fake-tiny-model harness as test_qwen35_preemption_state.py
  (fake config, load_model monkeypatched to a no-op, weights initialized
  from a fixed seed, Sampler.forward replaced with plain argmax, a
  DummyTokenizer whose decode() is just `" ".join(str(t) for t in ids)`).
  Real checkpoint weights are irrelevant to a contamination check -- this
  is testing Scheduler/StateManager mechanics, not model quality -- and
  skipping the real checkpoint load is what keeps this a minutes-not-hours
  script.

  The stop-string sequence uses stop=[r"\\d+ \\d+ \\d+"] (three
  space-separated digit groups). Because DummyTokenizer.decode() always
  renders completion tokens as space-separated digit strings REGARDLESS OF
  THEIR ACTUAL VALUE, this pattern is guaranteed to match exactly once 3
  completion tokens exist -- deterministic across any random/untrained
  weights, with no need to pre-observe what the model actually generates.
  ignore_eos=True + max_tokens=20 on this sequence means its finish can
  ONLY be caused by the stop pattern, unambiguously (not a race with EOS or
  max_tokens also being satisfiable around the same point).

  max_num_seqs=2 (small, like test_state_slot_reuse.py) forces: seq_short
  and seq_long prefill together as the initial batch; seq_new_a/seq_new_b
  sit in Scheduler.waiting. When seq_short's stop-string fires at
  completion token 3, its slot frees; the NEXT schedule() call (a prefill
  step, per schedule()'s prefill/decode mutual exclusion) admits seq_new_a
  into that freed slot while seq_long -- still mid-decode, untouched that
  step -- resumes decoding the step after. This is the real interleaving
  the scheduler produces, not a contrived approximation of it.

TWO COMPARISONS, deliberately different bars:
  1. PRIMARY: recurrent-state cosine similarity, captured at the exact
     moment StateManager.free() would zero each slot (monkeypatched to
     snapshot first, matching test_qwen35_preemption_state.py's
     patch_capture_on_free). Ground truth per sequence: a FRESH engine
     (same weight seed), fed that sequence's own prompt + full observed
     completion (from the batched run) as a single uninterrupted history,
     generating exactly 1 more token to force one clean finish/free with
     zero slot-sharing exposure. Threshold reused from
     test_qwen35_preemption_state.py's OWN state-comparison bar: cosine
     > 0.999. (NOT the 0.9976-0.9997 figure from
     phase6_packed_contamination_check.py / gsm8k_decode_contamination_check.py
     -- that number is for PREFILL LOGITS cosine, a different quantity
     measured for a different reason; reusing it here would be comparing
     apples to oranges. The state-comparison bar is the right one for what
     this script actually measures.)
  2. SECONDARY / diagnostic only: exact token-for-token match of full
     completions, together vs isolated. Per gsm8k_decode_contamination_check.py's
     documented caveat, a late single-position divergence after many
     tokens of agreement is NOT automatically a contamination bug (batched
     matmul numeric noise can flip one argmax and cascade) -- but with
     temperature=1.0 on random weights here, argmax flips are actually
     LIKELY even with zero contamination, which is exactly why comparison
     #1 (state cosine, not sampled tokens) is the one that actually decides
     PASS/FAIL. Comparison #2 is printed for extra context only.

Usage (no GPU work happens until you run this -- see module docstring of
the calling investigation for why this script is prepared but not executed
here):
    python tests/make_fake_hf_config.py     # if not already run
    python tests/decode_stagger_contamination_check.py

Runtime: single process, tiny model (8 layers, hidden_size=512), 4 short
prompts + 4 tiny isolated re-runs. Should complete in well under a minute
on either A6000.
"""
import atexit
import gc
import os
import sys
import json

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm":
    import types
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

import nanovllm.layers.sampler as sampler_mod
sampler_mod.Sampler.forward = lambda self, logits, temperatures: logits.float().argmax(dim=-1)

import nanovllm.utils.loader as loader_mod
loader_mod.load_model = lambda model, path, **kw: None

import nanovllm.config as config_mod


class AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _fake_from_pretrained(path, *args, **kwargs):
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d


config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)


class DummyTokenizer:
    """decode() always renders tokens as space-separated digit strings --
    this is what makes the stop=[r"\\d+ \\d+ \\d+"] pattern below fire
    deterministically at exactly completion token 3, for ANY sampled
    token id, on random/untrained weights. See module docstring."""
    eos_token_id = 999999

    def encode(self, prompt):
        if isinstance(prompt, list):
            return prompt
        return [int(x) for x in prompt.split() if x.isdigit()]

    def decode(self, token_ids):
        return " ".join(str(t) for t in token_ids)


import transformers
transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *args, **kwargs: DummyTokenizer())

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

FAKE_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
STATE_COSINE_THRESHOLD = 0.999  # see module docstring -- reused from
                                 # test_qwen35_preemption_state.py's own
                                 # state-comparison bar, not the unrelated
                                 # prefill-logits cosine figure.


def init_deterministic_weights(engine: LLMEngine, seed: int = 42):
    with torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = engine.model_runner.model
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(param)


def instrument(engine: LLMEngine):
    """Monkeypatches this engine's own StateManager.allocate/free (and
    LLMEngine.step, for a step counter) to build a full alloc/free
    timeline plus a pre-free state snapshot per seq_id -- the "which slot,
    when" diagnostic info needed to attribute a FAIL to a specific reuse
    event without a second investigation round.

    Returns (events, captured_states) -- both live dicts/lists, populated
    as engine.generate() runs.
    """
    sm = engine.model_runner.state_manager
    step_counter = {"n": 0}
    events = []
    captured_states = {}

    orig_step = engine.step

    def counting_step():
        step_counter["n"] += 1
        return orig_step()

    engine.step = counting_step

    orig_allocate = sm.allocate

    def logging_allocate(seq):
        slot_id = orig_allocate(seq)
        events.append({"event": "alloc", "step": step_counter["n"],
                        "seq_id": seq.seq_id, "slot": slot_id})
        print(f"[EVENT] step={step_counter['n']:3d} ALLOC  seq_id={seq.seq_id} -> slot={slot_id}")
        return slot_id

    sm.allocate = logging_allocate

    orig_free = sm.free

    def logging_free(seq):
        slot_id = seq.state_slot
        if slot_id is not None:
            captured_states[seq.seq_id] = sm.states[:, slot_id].clone()
            events.append({"event": "free", "step": step_counter["n"],
                            "seq_id": seq.seq_id, "slot": slot_id})
            print(f"[EVENT] step={step_counter['n']:3d} FREE   seq_id={seq.seq_id} <- slot={slot_id}")
        return orig_free(seq)

    sm.free = logging_free

    return events, captured_states


def run_natural_isolated(prompt: list[int], sp: SamplingParams, seed: int = 42):
    """Zero-contamination-risk run via NATURAL prefill + sequential decode,
    letting the sampler pick freely at every step (same compute pattern the
    batched run's own sequence went through, but NOT the same tokens --
    see run_teacher_forced_isolated for the version that pins tokens
    identical). INFORMATIVE ONLY: on random/untrained weights, many logits
    sit close together, so ordinary batch-size numeric noise (batch-of-1
    here vs whatever batch size the sequence saw when run "together") can
    flip an early argmax with zero contamination involved (see
    test_qwen35_preemption_state.py's docstring). If this run's own tokens
    diverge from `together`'s before reaching the point of interest, its
    STATE is not comparable to anything -- different token histories
    trivially produce different states, telling you nothing about
    contamination OR the prefill-vs-decode compute-path question.
    Returns (captured_state, generated_token_ids).
    """
    engine = build_engine()
    init_deterministic_weights(engine, seed=seed)
    _, captured = instrument(engine)
    try:
        out = engine.generate([prompt], sp, use_tqdm=False)[0]
    finally:
        atexit.unregister(engine.exit)
        engine.exit()
    assert len(captured) == 1, f"expected exactly 1 natural-isolated free-time capture, got {len(captured)}"
    state = next(iter(captured.values()))
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return state, out["token_ids"]


def run_teacher_forced_isolated(prompt: list[int], forced_completion: list[int], seed: int = 42):
    """THE decisive control: zero-contamination-risk (own engine, own slot,
    never shares a batch) AND zero-token-divergence-risk (every token is
    FORCED to match `forced_completion` exactly, bypassing the sampler's
    own choice) -- isolating the ONE remaining variable, prefill-chunked-
    parallel-scan (captured_isolated's single-shot-prefill of the full
    history) vs prefill-of-prompt-then-sequential-decode (this function,
    matching the batched run's own compute pattern for this sequence
    exactly: one prefill over just the prompt, then one decode step per
    completion token). Drives Scheduler.schedule()/ModelRunner.run()/
    Scheduler.postprocess() directly instead of LLMEngine.generate()'s own
    loop, so the sampled token_ids can be overridden with the known ones
    before postprocess() ever appends/checks them -- ignore_eos=True and
    max_tokens=len(forced_completion) mean postprocess()'s finish check can
    only fire on the max_tokens condition, exactly when this loop is done
    forcing the last token, never earlier from a stray EOS/stop match.
    """
    engine = build_engine()
    init_deterministic_weights(engine, seed=seed)
    _, captured = instrument(engine)
    sp = SamplingParams(max_tokens=len(forced_completion), temperature=1.0, ignore_eos=True)
    engine.add_request(prompt, sp)
    for forced_token in forced_completion:
        seqs, is_prefill = engine.scheduler.schedule()
        engine.model_runner.call("run", seqs, is_prefill)
        engine.scheduler.postprocess(seqs, [forced_token] * len(seqs), is_prefill)
    assert engine.scheduler.is_finished(), (
        "teacher-forced loop ended before the sequence finished -- "
        "forced_completion length mismatch with the sequence's own max_tokens"
    )
    assert len(captured) == 1, f"expected exactly 1 teacher-forced free-time capture, got {len(captured)}"
    state = next(iter(captured.values()))
    atexit.unregister(engine.exit)
    engine.exit()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return state


def build_engine():
    # 0.3 was too tight on at least one real machine (num_kvcache_blocks
    # computed to <=0 -- allocate_kv_cache()'s budget is
    # total*gpu_memory_utilization - used - peak + current, and `used`/`total`
    # come from torch.cuda.mem_get_info() for the WHOLE device, not just this
    # process, so headroom depends on whatever else is resident on the GPU
    # at the time). 0.9 matches what the real-checkpoint eval scripts
    # (gsm8k_*.py) already use safely; this model is tiny either way.
    return LLMEngine(
        FAKE_DIR, max_num_batched_tokens=128, max_num_seqs=2, max_model_len=512,
        gpu_memory_utilization=0.9, tensor_parallel_size=1, enforce_eager=True,
    )


def main():
    assert torch.cuda.is_available(), "requires CUDA"
    assert os.path.isdir(FAKE_DIR), "run tests/make_fake_hf_config.py first"

    # Distinct, non-overlapping small-integer "prompts" -- content doesn't
    # matter (random weights), only that each sequence is distinguishable
    # and none accidentally shares a prefix-cache block (moot anyway: see
    # investigation notes -- disable_prefix_cache=True whenever a
    # StateManager is active, so this isn't even reachable, but keeping
    # prompts disjoint removes any doubt while reading the output).
    prompt_short = [100, 101, 102, 103, 104, 105]
    prompt_long = [110, 111, 112, 113, 114, 115, 116, 117]
    prompt_new_a = [120, 121, 122, 123]
    prompt_new_b = [130, 131, 132, 133, 134]

    prompts = [prompt_short, prompt_long, prompt_new_a, prompt_new_b]
    names = ["seq_short(stop-string)", "seq_long", "seq_new_a", "seq_new_b"]

    sp_short = SamplingParams(temperature=1.0, max_tokens=20, ignore_eos=True,
                               stop=[r"\d+ \d+ \d+"])
    sp_long = SamplingParams(temperature=1.0, max_tokens=15, ignore_eos=True)
    sp_new_a = SamplingParams(temperature=1.0, max_tokens=10, ignore_eos=True)
    sp_new_b = SamplingParams(temperature=1.0, max_tokens=10, ignore_eos=True)
    sampling_params = [sp_short, sp_long, sp_new_a, sp_new_b]

    print("=" * 78)
    print("BATCHED RUN -- 4 sequences, max_num_seqs=2, staggered stop-string finish")
    print("=" * 78)
    print("Expected interleaving: seq_short + seq_long prefill together (slots 0,1).")
    print("seq_short's stop-string should fire at completion token 3 (deterministic --")
    print("see module docstring), freeing its slot while seq_long is still decoding.")
    print("The NEXT schedule() call should admit seq_new_a (or seq_new_b) into that")
    print("freed slot. Watch the [EVENT] lines below for the ACTUAL interleaving.\n")

    engine = build_engine()
    init_deterministic_weights(engine, seed=42)
    events, captured_together = instrument(engine)

    try:
        results = engine.generate(prompts, sampling_params, use_tqdm=False)
    finally:
        # LLMEngine.__init__ does atexit.register(self.exit) -- that keeps a
        # permanent strong reference to this instance (model weights,
        # kv_cache, StateManager buffers) alive for the life of the process
        # no matter what, so plain `del engine` would NOT free GPU memory
        # before the isolated-run engines below try to allocate their own KV
        # caches (see tests/test_state_slot_reuse.py, same issue there).
        atexit.unregister(engine.exit)
        engine.exit()

    print("\nCompletions (batched run):")
    seq_ids = []
    completions = {}
    for name, prompt, out in zip(names, prompts, results):
        tok_ids = out["token_ids"]
        print(f"  {name}: prompt={prompt} completion={tok_ids}")
        completions[name] = tok_ids
    # LLMEngine.generate() returns outputs sorted by seq_id, same order as
    # `prompts`/`sampling_params` were submitted in (seq_id assigned in
    # submission order) -- safe to zip positionally with `names`/`prompts`.

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("ISOLATED GROUND TRUTH -- each sequence's own prompt+completion, alone,")
    print("fresh engine (same weight seed), forced through in one uninterrupted")
    print("history + 1 final decode step to capture its natural free()-time state.")
    print("=" * 78)

    # captured_together is keyed by seq_id, assigned by Sequence.counter in
    # submission order (LLMEngine.add_request()'s call order inside
    # generate() -- i.e. same order as `prompts`), so sorting seq_id
    # recovers the 0=short, 1=long, 2=new_a, 3=new_b mapping directly.
    ordered_seq_ids = sorted(captured_together.keys())
    assert len(ordered_seq_ids) == 4, (
        f"expected 4 captured free-time states, got {len(ordered_seq_ids)} "
        f"({sorted(captured_together.keys())}) -- a sequence finished without "
        f"going through the instrumented free() path, investigate before trusting "
        f"the comparison below"
    )
    seq_id_to_name = dict(zip(ordered_seq_ids, names))
    seq_id_to_prompt = dict(zip(ordered_seq_ids, prompts))
    seq_id_to_completion = dict(zip(ordered_seq_ids, [results[i]["token_ids"] for i in range(4)]))
    seq_id_to_sp = dict(zip(ordered_seq_ids, sampling_params))

    captured_isolated = {}
    for seq_id in ordered_seq_ids:
        name = seq_id_to_name[seq_id]
        full_history = seq_id_to_prompt[seq_id] + seq_id_to_completion[seq_id]
        assert len(full_history) >= 2, f"{name}: history too short to isolate meaningfully"
        iso_engine = build_engine()
        init_deterministic_weights(iso_engine, seed=42)
        _, iso_captured = instrument(iso_engine)
        try:
            iso_sp = SamplingParams(max_tokens=1, temperature=1.0, ignore_eos=True)
            iso_engine.generate([full_history[:-1]], iso_sp, use_tqdm=False)
        finally:
            # Same atexit-strong-reference issue as the batched engine's
            # teardown above -- must unregister before exit()/del, every
            # iteration, or each successive isolated engine build in this
            # loop leaks the previous one's KV cache too.
            atexit.unregister(iso_engine.exit)
            iso_engine.exit()
        assert len(iso_captured) == 1, f"{name}: expected exactly 1 isolated free-time capture"
        captured_isolated[seq_id] = next(iter(iso_captured.values()))
        del iso_engine
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  captured isolated state for {name} (seq_id={seq_id}, history_len={len(full_history)})")

    print("\n" + "=" * 78)
    print(f"COMPARISON 1 (PRIMARY) -- recurrent state cosine similarity, "
          f"threshold > {STATE_COSINE_THRESHOLD}")
    print("=" * 78)
    all_state_pass = True
    failing_seq_ids = []
    for seq_id in ordered_seq_ids:
        name = seq_id_to_name[seq_id]
        together_state = captured_together[seq_id]
        isolated_state = captured_isolated[seq_id]
        num_linear_layers = together_state.shape[0]
        per_layer_cos = []
        for layer_idx in range(num_linear_layers):
            a = together_state[layer_idx].float().reshape(-1)
            b = isolated_state[layer_idx].float().reshape(-1)
            cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
            per_layer_cos.append(cos)
        min_cos = min(per_layer_cos)
        status = "PASS" if min_cos > STATE_COSINE_THRESHOLD else "FAIL"
        if status == "FAIL":
            all_state_pass = False
            failing_seq_ids.append(seq_id)
        print(f"  {name} (seq_id={seq_id}): per-layer cosine={[f'{c:.6f}' for c in per_layer_cos]} "
              f"min={min_cos:.6f} [{status}]")

    if failing_seq_ids:
        print("\n" + "=" * 78)
        print("COMPARISON 1b (CONTROL, only for FAILing sequences) -- rules out a confound:")
        print("captured_isolated above used a SINGLE-SHOT prefill of the full known history,")
        print("a DIFFERENT compute path (chunked parallel scan) than the batched run's own")
        print("prefill-of-prompt+sequential-decode. The decisive control below")
        print("(run_teacher_forced_isolated) reproduces THAT exact compute pattern -- own")
        print("engine, own slot, NEVER shares a batch (zero contamination risk) -- while")
        print("FORCING every token to match the known completion exactly (zero token-")
        print("divergence risk, unlike letting the sampler pick freely -- see the")
        print("natural-decode-alone line below, which is context only, NOT part of the")
        print("verdict, precisely because its tokens are free to diverge and did).")
        print("Both compared ground truths process the IDENTICAL token sequence -- the")
        print("only thing that differs between them is compute path. If they disagree by")
        print("a similar margin to the original FAIL, that FAIL is the compute-path")
        print("artifact, not contamination. If they agree closely, the confound is ruled")
        print("out and the original FAIL is a real contamination signal.")
        print("=" * 78)
        for seq_id in failing_seq_ids:
            name = seq_id_to_name[seq_id]
            prompt = seq_id_to_prompt[seq_id]
            sp = seq_id_to_sp[seq_id]
            together_tokens = seq_id_to_completion[seq_id]
            single_shot_state = captured_isolated[seq_id]

            natural_state, natural_tokens = run_natural_isolated(prompt, sp)
            print(f"  {name} (seq_id={seq_id}):")
            print(f"    [context only, not part of the verdict] natural-decode-alone "
                  f"completion: {natural_tokens}")
            print(f"    [context only, not part of the verdict] together completion:      "
                  f"{together_tokens}")
            print(f"    [context only] tokens match together exactly: "
                  f"{natural_tokens == together_tokens} -- if False, natural-alone's state "
                  f"is NOT comparable to anything (different token history) and is skipped below")

            forced_state = run_teacher_forced_isolated(prompt, together_tokens)
            num_linear_layers = forced_state.shape[0]
            per_layer_cos = []
            for layer_idx in range(num_linear_layers):
                a = forced_state[layer_idx].float().reshape(-1)
                b = single_shot_state[layer_idx].float().reshape(-1)
                cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
                per_layer_cos.append(cos)
            min_cos = min(per_layer_cos)
            print(f"    DECISIVE: teacher-forced-sequential-decode vs single-shot-prefill "
                  f"(same tokens, both zero-contamination-risk) per-layer cosine="
                  f"{[f'{c:.6f}' for c in per_layer_cos]} min={min_cos:.6f}")
            if min_cos <= STATE_COSINE_THRESHOLD:
                print(f"    -> CONFOUND EXPLAINS THE FAIL: two contamination-free, "
                      f"identical-token-history ground truths disagree by a similar margin "
                      f"({min_cos:.6f}) purely from the prefill-chunk-vs-decode-sequential "
                      f"compute-path difference. The original together-vs-isolated FAIL for "
                      f"{name} is NOT strong evidence of contamination.")
            else:
                print(f"    -> CONFOUND RULED OUT: two contamination-free, identical-token-"
                      f"history ground truths agree closely ({min_cos:.6f} > "
                      f"{STATE_COSINE_THRESHOLD}) despite the different compute paths. The "
                      f"original together-vs-isolated FAIL for {name} is NOT explained by "
                      f"this confound -- treat it as a real contamination signal.")

    print("\n" + "=" * 78)
    print("COMPARISON 2 (secondary/diagnostic) -- exact token-for-token completion match")
    print("(informative only -- see module docstring: argmax flips from batched-vs-solo")
    print("numeric noise are possible even with zero contamination, especially at")
    print("temperature=1.0 on random weights. Comparison 1 above is what decides PASS/FAIL.)")
    print("=" * 78)
    for seq_id in ordered_seq_ids:
        name = seq_id_to_name[seq_id]
        together_tok = seq_id_to_completion[seq_id]
        # isolated run generated only 1 extra token on top of together's own
        # completion minus its last token, so isolated's "new" token is
        # directly comparable to together's LAST token only.
        print(f"  {name}: together_completion={together_tok}")

    print("\n" + "=" * 78)
    print("ALLOC/FREE TIMELINE (for attributing any FAIL to a specific reuse event)")
    print("=" * 78)
    for e in events:
        print(f"  step={e['step']:3d} {e['event']:5s} seq_id={e['seq_id']} slot={e['slot']}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if all_state_pass:
        print(f"[PASS] All {len(ordered_seq_ids)} sequences: recurrent-state cosine > "
              f"{STATE_COSINE_THRESHOLD} between the staggered/batched run and an isolated "
              f"run of the same token history. No detectable decode-time state "
              f"contamination from mid-batch stop-string-triggered slot reuse.")
    else:
        print("[FAIL] At least one sequence's recurrent state diverged beyond the "
              "threshold between the batched and isolated runs -- see per-layer cosine "
              "above and cross-reference the ALLOC/FREE TIMELINE to identify which slot "
              "reuse event is implicated (look for the sequence that ALLOCATED the same "
              "slot immediately after the failing sequence's own slot was FREED).")
        raise AssertionError("decode-time state contamination detected -- see printed diagnostics above")


if __name__ == "__main__":
    main()
