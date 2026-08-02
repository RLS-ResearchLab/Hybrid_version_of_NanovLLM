"""Follow-up to diag_multi_hypothesis.py's Test A: seq 3 (len=9) got WORSE
(0.996012 -> 0.968734) under the fp32-GDR patch while seq 0 (len=7) improved
a lot and seq 1/seq 2 barely moved. Before concluding "mixed noise + a
separate logic issue", this isolates WHY seq 3 specifically regressed.

IMPORTANT caveat about the original Test A: its "baseline" list was computed
AFTER Qwen35LinearAttention.forward was monkey-patched to fp32 (see
diag_multi_hypothesis.py's test_A_fp32_gdr -- `Qwen35LinearAttention.forward
= fp32_forward` runs BEFORE `baseline = [run_single(...) ...]`). So Test A's
0.994815/0.968734/etc. numbers are (fp32-patched isolated) vs (fp32-patched
packed) -- correct for asking "does packing still diverge under fp32", but
they say nothing about whether the fp32 patch itself shifted the ISOLATED
output away from the original bf16 isolated output. That's exactly what
Part 1 below checks.

Part 1 -- Isolation regression check (the decisive test): for each of the
  4 original sequences, run ISOLATED (no packing) under (a) the original
  unpatched bf16 forward and (b) the fp32-GDR-patched forward, full model,
  and compare. If seq 3 diverges from itself here, the patch introduces a
  real numerical shift with NO packing involved at all -- unrelated to
  Phase 2's contamination question.

Part 2 -- Does the final cast-back-to-bf16 add its OWN rounding artifact,
  separate from the bf16 noise it removes? Isolates layer 0 (linear-
  attention) directly, bypassing the rest of the model, and compares
  isolated-vs-packed cosine for seq 3 in three modes: native bf16,
  fp32-with-final-cast-to-bf16, and fp32-with-NO-final-cast (kept in raw
  float32). If dropping the final cast doesn't change the isolated-vs-
  packed divergence, the cast isn't the culprit -- the divergence already
  exists inside the fp32 math itself.

Part 3 -- Composition/length correlation: packs seq 3 with different
  subsets/positions of the other three sequences under the fp32 patch and
  compares each result against seq 3's own fp32-patched isolated baseline,
  to see whether the regression needs the full 4-way batch, a specific
  companion, or just total packed size N.

Usage: python tests/diag_testA_followup.py
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


def _make_fp32_forward(orig_forward):
    def fp32_forward(self, hidden_states, cu_seqlens, states=None, conv_states=None):
        dt = hidden_states.dtype
        hs32 = hidden_states.float()
        conv32 = conv_states.float() if conv_states is not None else None
        out, new_states, new_conv = orig_forward(self, hs32, cu_seqlens, states, conv32)
        return out.to(dt), new_states, new_conv.to(dt)
    return fp32_forward


def part1_isolation_regression(device):
    print("\n" + "=" * 70)
    print("PART 1: does the fp32 patch shift the ISOLATED (unpacked) output,")
    print("        with zero packing involved at all?")
    print("=" * 70)

    config = make_small_config()
    model, sm = build_model_and_state(config, device, max_num_seqs=8)
    orig_forward = Qwen35LinearAttention.forward

    torch.manual_seed(99)
    seqs = [torch.randint(100, 5000, (n,)).tolist() for n in [7, 11, 5, 9]]

    bf16_iso = [run_single(model, sm, s, device) for s in seqs]

    for m in model.modules():
        if isinstance(m, Qwen35LinearAttention):
            m.float()
    Qwen35LinearAttention.forward = _make_fp32_forward(orig_forward)
    try:
        fp32_iso = [run_single(model, sm, s, device) for s in seqs]
    finally:
        Qwen35LinearAttention.forward = orig_forward
        for m in model.modules():
            if isinstance(m, Qwen35LinearAttention):
                m.to(torch.bfloat16)

    print("\n  seq | len | bf16-iso vs fp32-patch-iso cosine")
    regressions = []
    for i, s in enumerate(seqs):
        c = cosine(bf16_iso[i], fp32_iso[i])
        print(f"    {i}   | {len(s):3d} | {c:.6f}")
        regressions.append(c)

    print("\n  -- verdict --")
    if regressions[3] < 0.999:
        print(f"  seq 3 (len=9) shifts by itself under the patch, in ISOLATION, cosine={regressions[3]:.6f} "
              "-- this is NOT a packing/contamination effect. The fp32-upcast-at-boundary patch "
              "introduces its own numerical change for this sequence's data, independent of Phase 2 "
              "entirely. Test A's 0.968734 packed number is being compared against an ALREADY-SHIFTED "
              "fp32 baseline, not the original bf16 ground truth -- the true regression relative to "
              "the ORIGINAL bf16 output could be smaller, larger, or roughly the same; it must be "
              "read together with Part 3, not read alone as 'packing got worse'.")
    else:
        print(f"  seq 3 (len=9) isolated output is essentially UNCHANGED by the patch (cosine="
              f"{regressions[3]:.6f}) -- so the 0.968734 packed-vs-fp32-baseline number in Test A "
              "reflects a genuine packed-vs-isolated divergence under fp32, not an artifact of the "
              "patch shifting the baseline itself.")
    others_shift = any(c < 0.999 for i, c in enumerate(regressions) if i != 3)
    if others_shift:
        print("  Note: other sequences ALSO shift somewhat under the patch in isolation -- the patch's "
              "effect on isolated output is not unique to seq 3.")
    else:
        print("  Note: seq 3 is the only one (or the most) affected in isolation among these four -- "
              "whatever is happening is somewhat specific to seq 3's own data/length, not a blanket "
              "effect of the patch.")

    return dict(bf16_iso=bf16_iso, fp32_iso=fp32_iso, seqs=seqs, regressions=regressions)


def part2_layer_isolation_cast_check(device, seqs):
    print("\n" + "=" * 70)
    print("PART 2: does the final cast-back-to-bf16 add its OWN error,")
    print("        separate from the bf16 noise it's removing?")
    print("=" * 70)
    print("  Isolates layer 0 (linear-attention) directly -- bypasses the rest of the")
    print("  model entirely -- and compares seq 3's isolated-vs-packed cosine in three")
    print("  modes on the SAME embedded input: native bf16, fp32-with-final-cast, and")
    print("  fp32-with-NO-final-cast (kept raw float32).")

    config = make_small_config()
    model, sm = build_model_and_state(config, device, max_num_seqs=8)
    la = next(m for m in model.modules() if isinstance(m, Qwen35LinearAttention))

    ids_list = [torch.tensor(s, device=device) for s in seqs]
    hidden_list = [model.model.embed_tokens(ids) for ids in ids_list]
    lengths = [len(s) for s in seqs]
    cu_packed = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                              dtype=torch.int32, device=device)
    hidden_packed = torch.cat(hidden_list, dim=0)

    def run_mode(hidden, cu, num_segs, mode):
        conv = torch.zeros(num_segs, la.qkv_dim, la.ck - 1, device=device, dtype=torch.bfloat16)
        if mode == "bf16":
            with torch.no_grad():
                out, _, _ = la(hidden, cu, None, conv)
            return out
        la.float()
        try:
            with torch.no_grad():
                out, _, _ = la(hidden.float(), cu, None, conv.float())
            return out.to(hidden.dtype) if mode == "fp32_cast" else out
        finally:
            la.to(torch.bfloat16)

    print("\n  mode        | seq3 isolated-vs-packed cosine (layer 0 only)")
    results = {}
    for mode in ["bf16", "fp32_cast", "fp32_nocast"]:
        cu3 = torch.tensor([0, lengths[3]], dtype=torch.int32, device=device)
        iso_out = run_mode(hidden_list[3], cu3, 1, mode)
        packed_out_all = run_mode(hidden_packed, cu_packed, 4, mode)
        seg_start, seg_end = int(cu_packed[3]), int(cu_packed[4])
        packed_out = packed_out_all[seg_start:seg_end]
        c = cosine(iso_out, packed_out)
        results[mode] = c
        print(f"    {mode:11s} | {c:.6f}")

    print("\n  -- verdict --")
    cast_gap = results["fp32_nocast"] - results["fp32_cast"]
    if cast_gap > 0.01:
        print(f"  The final bf16 cast IS a meaningful contributor: dropping it improves the "
              f"isolated-vs-packed cosine by {cast_gap:.6f}. Part of Test A's regression for seq 3 is "
              "an artifact of casting the fp32 result back down to bf16 at the layer boundary.")
    else:
        print(f"  The final bf16 cast is NOT the main contributor (cast vs no-cast differ by only "
              f"{cast_gap:.6f}) -- the isolated-vs-packed divergence already exists inside the fp32 "
              "math itself, before any cast happens.")
    if results["fp32_nocast"] < 0.99:
        print("  Even PURE float32 (no bf16 anywhere in this layer, at either the input or output "
              "boundary) still shows meaningful isolated-vs-packed divergence for seq 3 at the "
              "single-layer level. This is NOT a dtype/precision bug -- it reproduces in full fp32. "
              "Most likely explanation: in_proj_qkv/z/a/b run as ONE batched matmul over the whole "
              "packed N (see Qwen35LinearAttention.forward's 'project once over the whole packed N' "
              "comment) -- GEMM reduction order (and therefore its floating-point rounding, even in "
              "fp32) is batch-shape dependent, and that per-token noise -- tiny on its own -- gets "
              "carried into the 9-step recurrent scan, where each step multiplies the running state "
              "by g_t and adds a delta term, i.e. it's a real (if slow) feedback loop that CAN "
              "amplify a tiny per-token seed difference into a visibly different final state, "
              "especially for a sequence whose particular g_t/beta values happen to sit in a more "
              "sensitive regime. This is likely also why the effect isn't a simple function of "
              "length (seq 2, len=5, stays near-perfect while seq 0, len=7, and seq 3, len=9, don't) "
              "-- it's data/conditioning-dependent, not purely length-dependent.")
    else:
        print("  Pure fp32 isolated-vs-packed divergence is negligible for seq 3 at the single-layer "
              "level -- so whatever produced the 0.968734 full-model regression must come from "
              "accumulation across the other 5 linear-attention layers and/or the full-attention/MoE "
              "layers, not from layer 0 alone.")

    return results


def part3_composition_sweep(device, seqs):
    print("\n" + "=" * 70)
    print("PART 3: does seq 3's regression correlate with WHAT it's packed with,")
    print("        or with total packed size, under the fp32 patch?")
    print("=" * 70)

    config = make_small_config()
    model, sm = build_model_and_state(config, device, max_num_seqs=8)
    orig_forward = Qwen35LinearAttention.forward
    for m in model.modules():
        if isinstance(m, Qwen35LinearAttention):
            m.float()
    Qwen35LinearAttention.forward = _make_fp32_forward(orig_forward)

    try:
        seq0, seq1, seq2, seq3 = seqs  # len 7, 11, 5, 9
        baseline3 = run_single(model, sm, seq3, device)  # fp32-patched isolated baseline, matches Test A's own

        combos = {
            "seq3 + seq0(len7)":                    [seq3, seq0],
            "seq3 + seq1(len11)":                   [seq3, seq1],
            "seq3 + seq2(len5)":                     [seq3, seq2],
            "seq3 + seq0 + seq1":                    [seq3, seq0, seq1],
            "seq3 first, +seq0+seq1+seq2 (all 4)":   [seq3, seq0, seq1, seq2],
            "seq3 last, +seq0+seq1+seq2 (all 4, original order)": [seq0, seq1, seq2, seq3],
        }
        print("\n  combination                                              | N_total | seq3 pos | cosine")
        for name, combo in combos.items():
            out = run_packed(model, sm, combo, device)
            seq3_pos = combo.index(seq3)
            n_total = sum(len(x) for x in combo)
            c = cosine(baseline3, out[seq3_pos])
            print(f"  {name:55s} | {n_total:7d} | {seq3_pos:8d} | {c:.6f}")
    finally:
        Qwen35LinearAttention.forward = orig_forward
        for m in model.modules():
            if isinstance(m, Qwen35LinearAttention):
                m.to(torch.bfloat16)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Test A follow-up: explaining seq 3's regression -- device={device}")
    init_dist()

    p1 = part1_isolation_regression(device)
    part2_layer_isolation_cast_check(device, p1["seqs"])
    part3_composition_sweep(device, p1["seqs"])

    print("\n" + "=" * 70)
    print("FOLLOW-UP COMPLETE -- see verdicts above for each part")
    print("=" * 70)

    import torch.distributed as dist
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
