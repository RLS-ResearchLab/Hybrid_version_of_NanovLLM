"""Real-checkpoint decode-time StateManager slot-reuse check -- the test
the earlier curl comparison (see bench notes from today's session) did NOT
perform, despite looking like validation.

That curl test ran 8 concurrent requests against the real checkpoint with
--max-num-seqs 32. 8 < 32, so every request got its own slot immediately;
NO slot was ever freed and reassigned to a new sequence. It genuinely
confirmed "batching multiple sequences into the same forward pass is fast
and produces coherent, byte-identical-with-FCFS output at temperature=0" --
a real and useful result -- but it is NOT evidence that a REUSED slot is
safe, because no reuse occurred. The only evidence for "reuse is safe" so
far comes from tests/decode_stagger_contamination_check.py, on a tiny
fake model with random/untrained weights.

This script closes that gap on the real Qwen3.5-35B-A3B checkpoint:
  1. max_num_seqs is set BELOW the total request count (3 slots, 8
     requests) so admission MUST queue and reuse freed slots -- not
     optional, not inferred from timing, forced by construction.
  2. Sequences are deliberately staggered: 4 "short" prompts carry a real
     stop=["."] so they finish via the STOP-STRING path specifically (the
     exact mechanism this whole investigation started from) well before
     their max_tokens budget; 4 "long" prompts have no stop pattern and a
     much larger max_tokens, so they're still decoding when a short
     sibling frees its slot.
  3. StateManager.allocate/free are instrumented (same technique as
     decode_stagger_contamination_check.py) to log the exact alloc/free
     timeline -- proof a free->reassign event fired, which slot, which
     sequences, at which step. Not inferred, logged.
  4. Each of the 8 sequences is ALSO run alone (isolated, no batching,
     same engine, sequential, before the packed run) as ground truth.
     Comparison is PAIRED: for the sequences that actually landed in a
     reused slot (per the logged timeline), does packed-run output match
     the isolated baseline? "Reuse observed AND output matches isolated
     baseline" is the claim -- not just "reuse observed."

Methodology difference from the fake-model version, deliberate:
  Isolated ground truth here uses NATURAL prefill+decode (own prompt, own
  sampling params, alone) -- NOT the single-shot-prefill-of-full-history
  reconstruction the fake-model script needed. That trick existed there
  to get a deterministic ground truth out of a model with random/
  untrained weights (many near-tied logits -> highly argmax-sensitive)
  without a teacher-forcing hook. On a REAL, well-trained checkpoint at
  temperature=0, greedy decoding is far more robust to the tiny numeric
  differences batching introduces -- already demonstrated empirically
  today: the FCFS-vs-batched curl comparison produced BYTE-IDENTICAL
  completions on this exact checkpoint. Natural isolated runs also avoid
  the prefill-chunk-vs-decode-sequential compute-path confound entirely,
  since both packed and isolated use the SAME compute pattern (prefill of
  just the prompt, then sequential decode) -- no confound-control run
  should be needed here the way it was for the fake model, but the token-
  match check below still gates whether the state-cosine number is
  trustworthy, exactly like gsm8k_decode_contamination_check.py's own
  calibration: if tokens diverge, states reflect different histories and
  are not comparable, full stop.

ONE engine load (not the 3-separate-process split
gsm8k_decode_contamination_check.py uses): safe here because
disable_prefix_cache=True unconditionally whenever a StateManager is
active (engine/block_manager.py) -- there is no prefix-cache-reuse
confound between the isolated and packed phases the way there was for the
non-hybrid KV-cache check that motivated that 3-process split. Isolated
runs happen first, sequentially, one prompt at a time, before the packed
run -- reusing the already-loaded checkpoint saves a second ~1-2 minute
reload.

Usage:
    python tests/real_checkpoint_slot_reuse_check.py                    # forced reuse (max_num_seqs=3)
    python tests/real_checkpoint_slot_reuse_check.py --max-num-seqs 8   # control: no reuse at all,
                                                                          # measures ordinary batch-size
                                                                          # noise alone for comparison
                                                                          # (see --help)

Runtime: one real checkpoint load (~1-2 min, TP=2) + 8 short isolated
generations + 1 packed generation of the same 8 prompts. Should be low
single-digit minutes, not GSM8K-eval timescale.
"""
import atexit
import gc
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if os.path.basename(ROOT) != "nanovllm" and "nanovllm" not in sys.modules:
    import types
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")
STATE_COSINE_THRESHOLD = 0.999  # same bar used in test_qwen35_preemption_state.py
                                 # and decode_stagger_contamination_check.py's
                                 # state-vs-state comparisons (not the unrelated
                                 # prefill-logits cosine figure from other tests)

# 4 short (real stop-string finish) + 4 long (no stop, bigger max_tokens,
# still decoding when a short sibling frees its slot).
PROMPTS_AND_PARAMS = [
    ("What is the capital of France?", SamplingParams(temperature=0, max_tokens=32, stop=["."])),
    ("What is the capital of Japan?", SamplingParams(temperature=0, max_tokens=32, stop=["."])),
    ("What is the capital of Italy?", SamplingParams(temperature=0, max_tokens=32, stop=["."])),
    ("What is the capital of Egypt?", SamplingParams(temperature=0, max_tokens=32, stop=["."])),
    ("Count from 1 to 50, separated by commas.", SamplingParams(temperature=0, max_tokens=128)),
    ("List the first 15 prime numbers, separated by commas.", SamplingParams(temperature=0, max_tokens=128)),
    ("Write a short paragraph describing the ocean.", SamplingParams(temperature=0, max_tokens=128)),
    ("Explain photosynthesis in two sentences.", SamplingParams(temperature=0, max_tokens=128)),
]
NAMES = [
    "short_capital_france", "short_capital_japan", "short_capital_italy", "short_capital_egypt",
    "long_count_to_50", "long_primes", "long_ocean_paragraph", "long_photosynthesis",
]


def build_llm(max_num_seqs: int):
    return LLM(
        CKPT_DIR,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=max_num_seqs,
        enforce_eager=True,
    )


def apply_chat_template(llm: LLM, user_text: str) -> list[int]:
    """Mirrors src/server.py's chat_completions() preprocessing exactly
    (apply_chat_template(..., enable_thinking=False) then encode with
    add_special_tokens=False) so this test exercises the SAME prompt
    formatting real HTTP traffic would produce, not a simplified stand-in."""
    messages = [{"role": "user", "content": user_text}]
    try:
        text = llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except Exception:
        text = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm.tokenizer.encode(text, add_special_tokens=False)


def instrument(llm: LLM):
    """Same technique as decode_stagger_contamination_check.py's
    instrument(): monkeypatch StateManager.allocate/free to log a full
    alloc/free timeline (step, seq_id, slot) and snapshot each sequence's
    recurrent state at the exact moment it would be zeroed. Returns
    (events, captured_states)."""
    sm = llm.model_runner.state_manager
    step_counter = {"n": 0}
    events = []
    captured_states = {}

    orig_step = llm.step

    def counting_step():
        step_counter["n"] += 1
        return orig_step()

    llm.step = counting_step

    orig_allocate = sm.allocate

    def logging_allocate(seq):
        slot_id = orig_allocate(seq)
        events.append({"event": "alloc", "step": step_counter["n"], "seq_id": seq.seq_id, "slot": slot_id})
        print(f"[EVENT] step={step_counter['n']:3d} ALLOC  seq_id={seq.seq_id} -> slot={slot_id}")
        return slot_id

    sm.allocate = logging_allocate

    orig_free = sm.free

    def logging_free(seq):
        slot_id = seq.state_slot
        if slot_id is not None:
            captured_states[seq.seq_id] = sm.states[:, slot_id].clone()
            events.append({"event": "free", "step": step_counter["n"], "seq_id": seq.seq_id, "slot": slot_id})
            print(f"[EVENT] step={step_counter['n']:3d} FREE   seq_id={seq.seq_id} <- slot={slot_id}")
        return orig_free(seq)

    sm.free = logging_free

    return events, captured_states


def teardown(llm: LLM):
    """Same atexit-strong-reference issue as the fake-model script --
    LLMEngine.__init__ does atexit.register(self.exit), which keeps a
    permanent reference (and the real checkpoint's GPU memory) alive
    otherwise. Not needed between phases here since this script only ever
    builds ONE engine, but included for a clean exit / in case this script
    is imported/reused."""
    atexit.unregister(llm.exit)
    llm.exit()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-num-seqs", type=int, default=3, dest="max_num_seqs",
        help="Default 3 (below the 8 prompts below) -- forces queuing + real slot reuse, "
             "the scenario this script exists to test. Pass a value >= 8 (e.g. 8) instead "
             "to run the DECISIVE CONTROL: no reuse occurs at all (every prompt gets its "
             "own slot immediately, same as the earlier curl test's --max-num-seqs 32), so "
             "any packed-vs-isolated divergence measured in that run is ISOLATED from "
             "reuse -- pure ordinary batch-size numeric noise from co-scheduling with "
             "siblings. Compare that run's cosine numbers for the 'long' sequences "
             "against a --max-num-seqs 3 run's REUSE CONFIRMED sequences: if they're in "
             "the same range, the forced-reuse run's divergence is noise, not "
             "contamination; if the control stays much cleaner, the reuse hypothesis holds.",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "requires CUDA"
    assert os.path.isdir(CKPT_DIR), f"real checkpoint not found at {CKPT_DIR}"
    forces_reuse = args.max_num_seqs < len(PROMPTS_AND_PARAMS)

    print("=" * 78)
    if forces_reuse:
        print(f"Loading real checkpoint from {CKPT_DIR} (tensor_parallel_size=2, "
              f"max_num_seqs={args.max_num_seqs} -- deliberately below the "
              f"{len(PROMPTS_AND_PARAMS)} prompts below, to FORCE queuing + slot reuse)...")
    else:
        print(f"Loading real checkpoint from {CKPT_DIR} (tensor_parallel_size=2, "
              f"max_num_seqs={args.max_num_seqs} -- NO-REUSE CONTROL RUN: every prompt gets "
              f"its own slot immediately, matching the earlier curl test's config. This run "
              f"measures ordinary batch-size numeric noise from co-scheduling ALONE, with "
              f"zero possibility of slot reuse, as a baseline to compare against a "
              f"--max-num-seqs 3 (forced-reuse) run's numbers.)")
    print("=" * 78)
    llm = build_llm(args.max_num_seqs)
    events, captured = instrument(llm)

    prompt_token_ids = [apply_chat_template(llm, text) for text, _ in PROMPTS_AND_PARAMS]
    sampling_params = [sp for _, sp in PROMPTS_AND_PARAMS]

    print("\n" + "=" * 78)
    print("PHASE 1: ISOLATED baseline -- each of the 8 prompts run ALONE, sequentially")
    print("(no batching partner ever present regardless of --max-num-seqs, since only 1")
    print("sequence is ever active at a time in this phase)")
    print("=" * 78)
    isolated_tokens = {}
    isolated_states = {}
    for name, ids, sp in zip(NAMES, prompt_token_ids, sampling_params):
        out = llm.generate([ids], sp, use_tqdm=False)[0]
        # `captured` accumulates across the whole llm instance's lifetime --
        # take whichever NEW seq_id(s) appeared since the last snapshot.
        new_seq_ids = [sid for sid in captured if sid not in isolated_states and sid not in isolated_tokens]
        assert len(new_seq_ids) == 1, (
            f"{name}: expected exactly 1 newly-captured free-time state, got {len(new_seq_ids)} "
            f"({new_seq_ids}) -- isolated run should be a single sequence, alone"
        )
        seq_id = new_seq_ids[0]
        isolated_tokens[seq_id] = out["token_ids"]
        isolated_states[seq_id] = captured[seq_id]
        print(f"  isolated {name} (seq_id={seq_id}): {len(out['token_ids'])} tokens -> "
              f"{out['token_ids'][:12]}{'...' if len(out['token_ids']) > 12 else ''}")

    isolated_seq_ids = sorted(isolated_states.keys())
    name_by_isolated_seq_id = dict(zip(isolated_seq_ids, NAMES))

    print("\n" + "=" * 78)
    print(f"PHASE 2: PACKED run -- all 8 prompts submitted TOGETHER, max_num_seqs="
          f"{args.max_num_seqs}. Watch [EVENT] lines for actual free->reassign events.")
    print("=" * 78)
    packed_results = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    packed_seq_ids = sorted(sid for sid in captured if sid not in isolated_states)
    assert len(packed_seq_ids) == len(NAMES), (
        f"expected {len(NAMES)} newly-captured packed free-time states, got "
        f"{len(packed_seq_ids)} ({packed_seq_ids})"
    )
    name_by_packed_seq_id = dict(zip(packed_seq_ids, NAMES))
    packed_tokens = {sid: packed_results[i]["token_ids"] for i, sid in enumerate(packed_seq_ids)}
    packed_states = {sid: captured[sid] for sid in packed_seq_ids}

    for sid in packed_seq_ids:
        name = name_by_packed_seq_id[sid]
        print(f"  packed {name} (seq_id={sid}): {len(packed_tokens[sid])} tokens -> "
              f"{packed_tokens[sid][:12]}{'...' if len(packed_tokens[sid]) > 12 else ''}")

    print("\n" + "=" * 78)
    print("ALLOC/FREE TIMELINE (proof reuse fired, not inferred from co-existence/timing)")
    print("=" * 78)
    for e in events:
        print(f"  step={e['step']:3d} {e['event']:5s} seq_id={e['seq_id']} slot={e['slot']}")

    # Identify which PACKED sequences actually landed in a slot immediately
    # after some OTHER sequence freed it -- the specific reuse events this
    # script exists to validate, not just "8 things ran in the same session".
    packed_events = [e for e in events if e["seq_id"] in packed_seq_ids]
    reused_seq_ids = set()
    slot_last_freed_by = {}
    for e in sorted(packed_events, key=lambda e: e["step"]):
        if e["event"] == "free":
            slot_last_freed_by[e["slot"]] = e["seq_id"]
        elif e["event"] == "alloc":
            prior_occupant = slot_last_freed_by.get(e["slot"])
            if prior_occupant is not None and prior_occupant != e["seq_id"]:
                reused_seq_ids.add(e["seq_id"])
                print(f"  REUSE CONFIRMED: seq_id={e['seq_id']} ({name_by_packed_seq_id[e['seq_id']]}) "
                      f"allocated slot={e['slot']} at step={e['step']}, immediately after "
                      f"seq_id={prior_occupant} freed that exact slot")

    print("\n" + "=" * 78)
    print("COMPARISON -- packed vs isolated, PAIRED per sequence")
    print("(sequences flagged REUSE above are the ones this script exists to validate --")
    print("the rest are reported too, for completeness, but carry less weight since they")
    print("never actually experienced a slot reassignment)")
    print("=" * 78)
    any_reuse_fail = False
    for packed_sid in packed_seq_ids:
        name = name_by_packed_seq_id[packed_sid]
        # Isolated captures are keyed by their OWN (different, earlier) seq_id --
        # match by NAME, since both loops iterate PROMPTS_AND_PARAMS/NAMES in
        # the same fixed order.
        iso_sid = next(s for s, n in name_by_isolated_seq_id.items() if n == name)

        packed_tok = packed_tokens[packed_sid]
        iso_tok = isolated_tokens[iso_sid]
        tokens_match = packed_tok == iso_tok

        packed_state = packed_states[packed_sid]
        iso_state = isolated_states[iso_sid]
        num_linear_layers = packed_state.shape[0]
        per_layer_cos = []
        for layer_idx in range(num_linear_layers):
            a = packed_state[layer_idx].float().reshape(-1)
            b = iso_state[layer_idx].float().reshape(-1)
            cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            per_layer_cos.append(cos)
        min_cos = min(per_layer_cos)

        was_reused = packed_sid in reused_seq_ids
        tag = "[REUSE CONFIRMED]" if was_reused else "[no reuse observed for this seq]"
        print(f"\n  {name} {tag}")
        print(f"    tokens match isolated exactly: {tokens_match}")
        if not tokens_match:
            first_diff = next((i for i, (x, y) in enumerate(zip(packed_tok, iso_tok)) if x != y), None)
            print(f"    packed:   {packed_tok}")
            print(f"    isolated: {iso_tok}")
            print(f"    first divergence at token position {first_diff}")
            print(f"    state cosine SKIPPED -- token histories differ, state comparison would be meaningless")
            if was_reused:
                any_reuse_fail = True
                print(f"    -> FAIL: this sequence WAS reused and its output does NOT match isolated baseline")
        else:
            status = "PASS" if min_cos > STATE_COSINE_THRESHOLD else "FAIL"
            print(f"    state cosine (threshold > {STATE_COSINE_THRESHOLD}): "
                  f"per-layer={[f'{c:.6f}' for c in per_layer_cos]} min={min_cos:.6f} [{status}]")
            if was_reused and status == "FAIL":
                any_reuse_fail = True
                print(f"    -> FAIL: this sequence WAS reused and its recurrent state diverged "
                      f"beyond threshold from the isolated baseline")

    teardown(llm)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if not reused_seq_ids:
        if forces_reuse:
            print("[INCONCLUSIVE] No slot-reuse event was actually observed in the packed run -- "
                  "the ALLOC/FREE TIMELINE above never shows a sequence allocated into a slot a "
                  "DIFFERENT sequence just freed. --max-num-seqs may need to be lowered further, "
                  "or the short prompts' stop=['.'] didn't fire early enough. This run proves "
                  "nothing about reuse safety either way -- do not report it as a PASS.")
            raise AssertionError("no slot-reuse event observed -- test did not exercise the scenario under test")
        else:
            long_names = {n for n in NAMES if n.startswith("long_")}
            print(f"[CONTROL RUN -- no reuse by design, --max-num-seqs={args.max_num_seqs} >= "
                  f"{len(PROMPTS_AND_PARAMS)} prompts] Use the per-sequence cosine numbers above "
                  f"for the 'long_*' sequences as the ordinary-batch-size-noise baseline. Compare "
                  f"them directly against a --max-num-seqs 3 run's REUSE CONFIRMED sequences of "
                  f"the SAME name: similar magnitude means the forced-reuse run's divergence is "
                  f"noise, not contamination; a control run staying much cleaner than the "
                  f"forced-reuse run means the reuse hypothesis holds and needs deeper investigation.")
    elif any_reuse_fail:
        print(f"[FAIL] {len(reused_seq_ids)} sequence(s) experienced a real slot-reuse event "
              f"(logged above), and at least one of them diverged from its isolated baseline. "
              f"See the per-sequence comparison above for exactly which one and by how much.")
        raise AssertionError("real-checkpoint decode-time state contamination detected -- see diagnostics above")
    else:
        print(f"[PASS] {len(reused_seq_ids)} sequence(s) confirmed reused a slot mid-batch "
              f"(exact events logged above, not inferred from co-existence/timing), and EVERY "
              f"one of them matches its isolated (no-batching) baseline exactly. Reuse fired, "
              f"and it fired safely, on the real checkpoint.")


if __name__ == "__main__":
    main()
