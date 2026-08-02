"""Multi-hypothesis diagnostics for the Phase 2 batching contamination bug
(~0.947-0.952 cosine on the shortest/position-0 sequence, see README Phase 2
and test_qwen35_batching.py's test_multi_sequence_no_contamination).

Three independent tests, run in this order, before proposing any fix:

  Test C -- equal-length, uniquely-tagged sequence pair, packed both
            (A,B) and (B,A). Isolates POSITION from IDENTITY: if a
            sequence's cosine-vs-baseline changes depending on which
            slot/position it occupies (not which sequence it is), that's
            the smoking gun for a positional/state-slot bug.

  Test B -- the ORIGINAL 4-sequence contamination scenario (lengths
            [7, 11, 5, 9], same seed==99 as test_qwen35_batching.py),
            packed in REVERSED order. Does the len=7 sequence still
            degrade wherever it lands, or does degradation follow
            position 0 regardless of which sequence sits there?

  Test A -- forces the GDR linear-attention layers to run entirely in
            float32 (not just the recurrent scan, which was already
            float32 -- see Qwen35LinearAttention's own docstring and
            StateManager.states' hardcoded float32 dtype; what still ran
            in the model's bf16 dtype was everything UPSTREAM of the
            scan: conv1d, in_proj_qkv/z/a/b, SiLU). Re-runs the ORIGINAL
            4-sequence scenario. Full-model float32 is not an option --
            flash_attn_varlen_func (every 4th full-attention layer) only
            accepts fp16/bf16 (see README's validation-gate bug #4) --
            so this monkey-patches ONLY Qwen35LinearAttention.forward to
            upcast at its own boundary; full-attention/MoE stay bf16.

Usage: python tests/diag_multi_hypothesis.py
"""
import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config   # noqa: E402
from test_qwen35_batching import build_model_and_state, run_packed, run_single  # noqa: E402

from nanovllm.models.qwen3_5 import Qwen35LinearAttention   # noqa: E402


def cosine(a, b):
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def test_C_position_vs_identity(device):
    print("\n" + "=" * 70)
    print("TEST C: equal-length, uniquely-tagged pair -- position swap")
    print("=" * 70)

    config = make_small_config()
    model, sm = build_model_and_state(config, device, max_num_seqs=4)

    length = 7  # same length as the sequence that showed contamination originally
    torch.manual_seed(501)
    seq_A = torch.randint(100, 200, (length,)).tolist()      # uniquely tagged: low range
    seq_B = torch.randint(4800, 4999, (length,)).tolist()    # uniquely tagged: high range

    baseline_A = run_single(model, sm, seq_A, device)
    baseline_B = run_single(model, sm, seq_B, device)

    run_AB = run_packed(model, sm, [seq_A, seq_B], device)   # A at pos0, B at pos1
    run_BA = run_packed(model, sm, [seq_B, seq_A], device)   # B at pos0, A at pos1

    cos_A_pos0 = cosine(baseline_A, run_AB[0])
    cos_B_pos1 = cosine(baseline_B, run_AB[1])
    cos_B_pos0 = cosine(baseline_B, run_BA[0])
    cos_A_pos1 = cosine(baseline_A, run_BA[1])

    print(f"  seq_A at position 0 (run A,B): cosine={cos_A_pos0:.6f}")
    print(f"  seq_A at position 1 (run B,A): cosine={cos_A_pos1:.6f}")
    print(f"  seq_B at position 1 (run A,B): cosine={cos_B_pos1:.6f}")
    print(f"  seq_B at position 0 (run B,A): cosine={cos_B_pos0:.6f}")

    a_delta = abs(cos_A_pos0 - cos_A_pos1)
    b_delta = abs(cos_B_pos0 - cos_B_pos1)
    print("\n  -- verdict --")
    print(f"  seq_A cosine delta across positions: {a_delta:.6f}")
    print(f"  seq_B cosine delta across positions: {b_delta:.6f}")
    if a_delta > 1e-3 or b_delta > 1e-3:
        print("  SMOKING GUN: cosine-vs-baseline changes with POSITION while identity/content "
              "is unchanged -- points to a positional/state-slot bug, not a content-dependent one.")
    else:
        print("  No meaningful position-dependent shift for either sequence in this pair "
              "(both deltas ~0) -- Test C alone does not implicate position.")

    return dict(cos_A_pos0=cos_A_pos0, cos_A_pos1=cos_A_pos1,
                cos_B_pos0=cos_B_pos0, cos_B_pos1=cos_B_pos1)


def test_B_reversed_order(device):
    print("\n" + "=" * 70)
    print("TEST B: original contamination scenario, reversed packing order")
    print("=" * 70)

    config = make_small_config()
    model, sm = build_model_and_state(config, device, max_num_seqs=8)

    torch.manual_seed(99)
    seqs = [torch.randint(100, 5000, (n,)).tolist() for n in [7, 11, 5, 9]]  # identical to test_qwen35_batching.py

    baseline = [run_single(model, sm, s, device) for s in seqs]

    forward_order = run_packed(model, sm, seqs, device)
    reversed_order = run_packed(model, sm, seqs[::-1], device)

    print("\n  -- forward order (original) --")
    fwd_cos = []
    for i, (b, p) in enumerate(zip(baseline, forward_order)):
        c = cosine(b, p)
        fwd_cos.append(c)
        print(f"    seq {i} (len={len(seqs[i])}) at position {i}: cosine={c:.6f}")

    print("\n  -- reversed order --")
    rev_cos = [None] * len(seqs)
    for new_pos, orig_idx in enumerate(range(len(seqs) - 1, -1, -1)):
        b = baseline[orig_idx]
        p = reversed_order[new_pos]
        c = cosine(b, p)
        rev_cos[orig_idx] = c
        print(f"    seq {orig_idx} (len={len(seqs[orig_idx])}) now at position {new_pos}: cosine={c:.6f}")

    print("\n  -- verdict --")
    last_pos = len(seqs) - 1
    print(f"  seq 0 (len=7): forward-order (pos0) cosine={fwd_cos[0]:.6f}, "
          f"reversed-order (pos{last_pos}) cosine={rev_cos[0]:.6f}")
    identity_follows = rev_cos[0] < 0.99 and abs(rev_cos[0] - fwd_cos[0]) < 0.02
    position_follows = rev_cos[0] > 0.99 and rev_cos[last_pos] < 0.99 and fwd_cos[0] < 0.99
    if identity_follows:
        print("  Degradation followed the SEQUENCE (seq 0, len=7) regardless of its position "
              "-- points to something content/length-dependent (e.g. numerical noise tied to "
              "how short the sequence is), not a position/state-slot indexing bug.")
    elif position_follows:
        print(f"  Degradation followed POSITION 0 instead -- whichever sequence now sits first "
              f"(seq {last_pos}, len={len(seqs[last_pos])}) degrades in the reversed run while "
              f"seq 0 (len=7, now last) is fine. That is the positional/state-slot smoking gun.")
    else:
        print("  Mixed/inconclusive pattern -- inspect the full table above manually.")

    return dict(forward=fwd_cos, reversed=rev_cos)


def test_A_fp32_gdr(device):
    print("\n" + "=" * 70)
    print("TEST A: force GDR (linear-attention) layers to float32, rerun original scenario")
    print("=" * 70)
    print("  Note: the delta-rule SCAN itself was already float32 (Qwen35LinearAttention "
          "casts q/k/v/g/beta/S to .float() before the recurrence, and StateManager.states "
          "is hardcoded float32). What still ran in the model's bf16 dtype was everything "
          "UPSTREAM of the scan: conv1d, in_proj_qkv/z/a/b, SiLU. This test forces those to "
          "float32 too. Full-model float32 is not an option -- flash_attn_varlen_func (every "
          "4th full-attention layer) only accepts fp16/bf16 -- so only Qwen35LinearAttention's "
          "forward is patched to upcast at its own boundary; full-attention/MoE stay bf16.")

    config = make_small_config()
    model, sm = build_model_and_state(config, device, max_num_seqs=8)

    # Upcast already-bf16-valued params losslessly (bf16 -> fp32 recovers the exact
    # same real number, just with more mantissa bits available for subsequent ops) --
    # same weight values as the bf16 run, just wider arithmetic from here on.
    for m in model.modules():
        if isinstance(m, Qwen35LinearAttention):
            m.float()

    orig_forward = Qwen35LinearAttention.forward

    def fp32_forward(self, hidden_states, cu_seqlens, states=None, conv_states=None):
        dt = hidden_states.dtype
        hs32 = hidden_states.float()
        conv32 = conv_states.float() if conv_states is not None else None
        out, new_states, new_conv = orig_forward(self, hs32, cu_seqlens, states, conv32)
        return out.to(dt), new_states, new_conv.to(dt)

    Qwen35LinearAttention.forward = fp32_forward
    try:
        torch.manual_seed(99)
        seqs = [torch.randint(100, 5000, (n,)).tolist() for n in [7, 11, 5, 9]]

        baseline = [run_single(model, sm, s, device) for s in seqs]
        packed = run_packed(model, sm, seqs, device)

        cos = []
        for i, (b, p) in enumerate(zip(baseline, packed)):
            c = cosine(b, p)
            cos.append(c)
            print(f"    seq {i} (len={len(seqs[i])}): cosine={c:.6f}")
    finally:
        Qwen35LinearAttention.forward = orig_forward

    print("\n  -- verdict --")
    print(f"  seq 0 (len=7) cosine with GDR forced to float32: {cos[0]:.6f} "
          f"(bf16 baseline from tonight's earlier run: ~0.9521)")
    if cos[0] > 0.999:
        print("  RESOLVED by float32: the divergence disappears when GDR's conv1d/projections run "
              "in float32 -- points to bf16 GEMM/conv kernel-selection noise upstream of the scan "
              "(batch-size-dependent rounding), not a logic/indexing bug.")
    elif cos[0] > 0.96:
        print("  IMPROVES but does not fully resolve -- fp32 removes SOME of the divergence, "
              "suggesting a MIX of bf16 numerical noise and a separate (smaller) logic issue.")
    else:
        print("  UNCHANGED (or nearly so) -- forcing float32 does not fix it, so this is NOT "
              "primarily bf16 rounding noise. The bug is a logic/indexing issue (e.g. state/slot "
              "wiring, cu_seqlens boundaries), not numerical precision.")

    return cos


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Multi-hypothesis Phase 2 contamination diagnostics -- device={device}")
    init_dist()

    test_C_position_vs_identity(device)
    test_B_reversed_order(device)
    test_A_fp32_gdr(device)

    print("\n" + "=" * 70)
    print("ALL THREE DIAGNOSTIC TESTS COMPLETE -- see verdicts above for each")
    print("=" * 70)

    import torch.distributed as dist
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
