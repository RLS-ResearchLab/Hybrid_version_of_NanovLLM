"""Real-checkpoint packed-batch contamination check, built directly on
Phase 6's already-working engine-loading/prompt plumbing
(reference_check_phase6.py) -- not a new, unvetted harness.

Everything so far (diag_multi_hypothesis.py / diag_testA_followup.py /
diag_layerwise_cascade.py / diag_gate_saturation.py) ran on the small
SYNTHETIC test model (make_small_config + build_model_and_state, random
init, A_log/dt_bias hardcoded to zero). That investigation chased a
per-layer amplification mechanism and ruled out gate saturation as the
cause -- interesting, but it's still a question about a toy model. This
script asks the question that actually matters: on the REAL checkpoint,
with the REAL scheduler-style packing path, does concurrently batching
several prompts change any of their outputs relative to running that same
prompt alone?

THRESHOLD -- stated here, before running, not after seeing the number:
cosine similarity >= 0.99 AND top-1 (argmax) match, per prompt. Same
discipline, same threshold, and the same COSINE_SIM_THRESHOLD constant as
reference_check_phase6.py's HF-vs-engine check.

Uses 4 of Phase 6's own PROMPTS (varied length -- short factual, GSM8K-
style word problem, bare numeric expression, longer narrative). These are
the same prompts reference_check_phase6.py already validated
single-sequence against the HF reference, so any divergence found here is
attributable to PACKING specifically, not a pre-existing correctness bug.

Packing goes through get_prefill_logits(seqs) with a REAL multi-sequence
list -- the exact function ModelRunner.run() calls internally, driven by
prepare_prefill()'s actual flat input_ids/cu_seqlens packing logic. Not a
hand-rolled harness: this is the same code path real concurrent serving
uses when the Scheduler batches several waiting prefills together.

State-slot allocation at tensor_parallel_size=2 uses Phase 6's established
`.call("allocate_state_slot", seq)` dispatch (see
reference_check_phase6.py's phase_engine() docstring for why calling
.allocate() directly on rank0's StateManager alone leaves rank1's copy of
that slot un-zeroed -- each rank is a separate process with its own
independent state buffers). Every sequence in this script -- baseline AND
packed -- gets its own never-reused slot (max_num_seqs = 2x prompt count),
so there's no free()/reallocate bookkeeping to get wrong.

Only the engine needs to be loaded here (no HF reference for this
comparison), so this is a single process, single phase.

Usage: python tests/phase6_packed_contamination_check.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from reference_check_phase6 import CKPT_DIR, PROMPTS, COSINE_SIM_THRESHOLD  # noqa: E402

# Varied length/content, same rationale as reference_check_phase6.py: short
# factual completion, GSM8K-style arithmetic word problem, bare numeric
# expression, longer narrative. Skips the 5th (embedded-quotation) prompt
# to keep this a clean 3-4 prompt batch as requested.
SELECTED_PROMPTS = PROMPTS[:4]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def _new_seq(llm, prompt: str):
    from nanovllm.engine.sequence import Sequence

    prompt_ids = llm.tokenizer.encode(prompt)
    seq = Sequence(prompt_ids)
    seq.num_scheduled_tokens = len(prompt_ids)
    # Same real-KV-cache-block allocation phase_engine() uses -- an empty
    # block_table would be treated as the warmup case and skip slot_mapping
    # construction entirely, breaking store_kvcache's assertion for a real
    # (non-warmup) sequence.
    llm.scheduler.block_manager.allocate(seq, 0)
    if llm.model_runner.state_manager is not None:
        # Dispatch through .call(...) so EVERY TP rank's own StateManager
        # allocates (and zeros) this seq's slot, not just rank0's copy --
        # see reference_check_phase6.py's phase_engine() docstring.
        llm.model_runner.call("allocate_state_slot", seq)
    return seq, prompt_ids


def main():
    from nanovllm.llm import LLM

    print(f"Prompts under test ({len(SELECTED_PROMPTS)}):")
    for p in SELECTED_PROMPTS:
        print(f"  {p!r}")

    llm = LLM(
        CKPT_DIR,
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=2 * len(SELECTED_PROMPTS),  # baseline slots + packed slots, no reuse needed
        max_model_len=2048,
    )

    print("\n" + "=" * 78)
    print("BASELINE -- each prompt run single-sequence (its own dedicated slot)")
    print("=" * 78)
    baseline_logits = []
    for prompt in SELECTED_PROMPTS:
        seq, prompt_ids = _new_seq(llm, prompt)
        logits = llm.model_runner.call("get_prefill_logits", [seq])
        assert logits is not None, "expected rank0 to return gathered logits"
        if logits.dim() == 2 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        print(f"  {prompt!r} -> {len(prompt_ids)} tokens, logits shape {tuple(logits.shape)}")
        baseline_logits.append(logits.float())

    print("\n" + "=" * 78)
    print(f"PACKED -- all {len(SELECTED_PROMPTS)} prompts in ONE batch (real prepare_prefill packing)")
    print("=" * 78)
    packed_seqs = [_new_seq(llm, prompt)[0] for prompt in SELECTED_PROMPTS]
    packed_logits = llm.model_runner.call("get_prefill_logits", packed_seqs)
    assert packed_logits is not None, "expected rank0 to return gathered logits"
    print(f"  packed logits shape: {tuple(packed_logits.shape)}  (expect ({len(SELECTED_PROMPTS)}, vocab))")
    assert packed_logits.shape[0] == len(SELECTED_PROMPTS), (
        f"expected one row per prompt, got {packed_logits.shape}"
    )

    print("\n" + "=" * 78)
    print(f"COMPARISON -- threshold stated before running: cosine >= {COSINE_SIM_THRESHOLD} AND argmax match")
    print("=" * 78)
    print(f"  {'#':>3}  {'prompt':<55}  {'cosine':>9}  {'argmax':>7}  {'verdict':>7}")
    print("  " + "-" * 88)

    results = []
    for i, prompt in enumerate(SELECTED_PROMPTS):
        b = baseline_logits[i]
        p = packed_logits[i].float()
        cos = cosine(b, p)
        argmax_match = int(b.argmax().item()) == int(p.argmax().item())
        passed = cos >= COSINE_SIM_THRESHOLD and argmax_match
        results.append((prompt, cos, argmax_match, passed))
        shown = prompt if len(prompt) <= 52 else prompt[:49] + "..."
        verdict = "PASS" if passed else "FAIL"
        print(f"  {i:>3}  {shown:<55}  {cos:>9.6f}  {'match' if argmax_match else 'DIFF':>7}  {verdict:>7}")

    n_pass = sum(1 for r in results if r[3])
    n_total = len(results)
    print()
    print(f"OVERALL: {n_pass}/{n_total} prompts passed (cosine >= {COSINE_SIM_THRESHOLD} AND argmax match)")
    if n_pass == n_total:
        print("VERDICT: PASS -- no meaningful packed-batch contamination on the real checkpoint, "
              "for these prompts.")
    else:
        worst = min(results, key=lambda r: r[1])
        print(f"VERDICT: FAIL -- {n_total - n_pass} prompt(s) degraded under packing:")
        for prompt, cos, argmax_match, passed in results:
            if not passed:
                print(f"    {prompt!r}: cosine={cos:.6f}  argmax_match={argmax_match}")
        print(f"  Worst: {worst[0]!r}  cosine={worst[1]:.6f}  argmax_match={worst[2]}")

    print("\nDONE")
    # LLMEngine.exit() is registered via atexit -- tears down TP ranks on process exit, nothing else needed here.


if __name__ == "__main__":
    main()
