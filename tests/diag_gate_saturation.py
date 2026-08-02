"""Tests the gate-saturation-sensitivity hypothesis for the per-layer
divergence pattern found by diag_layerwise_cascade.py's local-cast column:
  layer 0: 1.000000   layer 1: 0.999974   layer 2: 0.984097
  layer 4: 0.962550    layer 5: 0.978303   layer 6: 0.930455

IMPORTANT CORRECTION TO THE REQUEST: the model under test in every
diag_*.py script so far (diag_multi_hypothesis / diag_testA_followup /
diag_layerwise_cascade) is the SMALL SYNTHETIC test harness
(make_small_config + build_model_and_state), not the real 35B checkpoint.
In that harness, A_log and dt_bias are explicitly zero-initialized for
EVERY linear-attention layer -- see test_qwen35_batching.py's
build_model_and_state:
    elif "A_log" in name or "dt_bias" in name:
        torch.nn.init.zeros_(param)
So there is no real-checkpoint A_log/dt_bias to pull here, and these two
PARAMETERS carry exactly zero per-layer variation by construction in the
model these divergence numbers came from. Pulling the real checkpoint's
A_log/dt_bias would describe a DIFFERENT model (the real 35B one, used
only by diag_linear_attn_tp_isolation.py / reference_check_phase6*.py for
an unrelated TP-correctness check), not explain anything about the
numbers above.

What DOES vary per layer, despite A_log=dt_bias=0 everywhere, is the
RUNTIME gate values:
    g   = -A_log.exp() * softplus(in_proj_a(hidden) + dt_bias)   (<=0)
    g_t = exp(g)                                                  (in (0,1])
    beta = sigmoid(in_proj_b(hidden))                              (in (0,1))
because in_proj_a/in_proj_b are independently-random per layer (see
build_model_and_state: dim>=2 params get normal_(std=0.02)) and
hidden_states evolves with depth. g_t is the per-step multiplicative
decay applied to the recurrent state (S = S * g_t + ...): g_t near 1
means near-zero decay -- a perturbation introduced at token 0 persists
almost undamped through all 9 steps of seq 3's recurrence (long memory,
more room to compound). g_t near 0 means the state resets almost every
step (short memory, perturbations get washed out quickly).

Hypothesis under test: layers whose g_t sits closer to 1 (weaker decay)
should show HIGHER divergence (lower isolated-vs-packed cosine); layers
whose g_t sits closer to 0 (stronger decay/self-correction) should show
LOWER divergence (higher cosine). This script measures actual g_t/beta
statistics per layer, for seq 3's real tokens, in the real cascaded
forward pass, and reports the correlation plainly -- confirmed or not.

Usage: python tests/diag_gate_saturation.py
"""
import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config   # noqa: E402
from test_qwen35_batching import build_model_and_state   # noqa: E402

from nanovllm.models.qwen3_5 import Qwen35LinearAttention   # noqa: E402
from nanovllm.utils.context import set_context, reset_context   # noqa: E402

_ORIG_LA_FORWARD = Qwen35LinearAttention.forward


def cosine(a, b):
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def run_cascade(model, ids, positions, cu_seqlens, num_segments, device):
    """fp32gdr regime cascade (matches diag_layerwise_cascade.py): every
    linear_attention layer runs its own conv1d/in_proj_*/scan in float32,
    cast back to bf16 right after. Captures, per linear-attention layer:
      - cast_out, nocast_out (float32, (N,H)) -- for the local-divergence check
      - g_t, beta (float32, (N,LVH)) -- for the gate-saturation check
    full_attention layers just advance the cascade (always bf16, untouched).
    """
    hidden_states = model.model.embed_tokens(ids)
    residual = None
    per_layer = {}  # layer_idx -> dict(cast_out, nocast_out, g_t, beta)

    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    set_context(True, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, None, None, None)
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if residual is None:
                normed, residual = layer.input_layernorm(hidden_states), hidden_states
            else:
                normed, residual = layer.input_layernorm(hidden_states, residual)

            if layer.is_full:
                with torch.no_grad():
                    hidden_states = layer.self_attn(positions, normed)
            else:
                la = layer.linear_attn
                conv0 = torch.zeros(num_segments, la.qkv_dim, la.ck - 1, device=device, dtype=torch.bfloat16)
                state0 = torch.zeros(num_segments, la.lvh, la.lhd, la.lhd, device=device, dtype=torch.float32)

                la.float()
                try:
                    normed32 = normed.float()
                    with torch.no_grad():
                        out32, _, _ = _ORIG_LA_FORWARD(la, normed32, cu_seqlens, state0, conv0.float())
                        # Replicate forward()'s own pre-scan gate computation exactly,
                        # over the WHOLE packed input (matches how in_proj_a/in_proj_b
                        # are actually applied -- once, over all N, before the
                        # per-segment loop) -- read-only, no side effects.
                        a = la.in_proj_a(normed32)
                        b = la.in_proj_b(normed32)
                        g = -la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias.float())
                        g_t = g.exp()          # (N, LVH), in (0, 1]
                        beta = b.float().sigmoid()  # (N, LVH), in (0, 1)
                    cast_out = out32.to(normed.dtype)
                    hidden_states = cast_out
                    per_layer[layer_idx] = dict(
                        cast_out=cast_out.float(), nocast_out=out32.float(),
                        g_t=g_t, beta=beta,
                    )
                finally:
                    la.to(torch.bfloat16)

            normed2, residual = layer.post_attention_layernorm(hidden_states, residual)
            with torch.no_grad():
                hidden_states = layer.mlp(normed2, cu_seqlens)
    finally:
        reset_context()

    return per_layer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Gate-saturation-sensitivity diagnostic -- device={device}")
    os.environ["MASTER_PORT"] = "29517"
    init_dist()

    config = make_small_config()
    model, _ = build_model_and_state(config, device, max_num_seqs=8)
    layer_types = model.model.layer_types
    linear_indices = [i for i, t in enumerate(layer_types) if t == "linear_attention"]

    torch.manual_seed(99)
    seqs = [torch.randint(100, 5000, (n,)).tolist() for n in [7, 11, 5, 9]]  # seq3 = seqs[3], len=9
    lengths = [len(s) for s in seqs]

    ids_iso = torch.tensor(seqs[3], dtype=torch.int64, device=device)
    positions_iso = torch.arange(lengths[3], device=device)
    cu_iso = torch.tensor([0, lengths[3]], dtype=torch.int32, device=device)

    ids_packed = torch.tensor(sum(seqs, []), dtype=torch.int64, device=device)
    positions_packed = torch.cat([torch.arange(l, device=device) for l in lengths])
    cu_packed = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device=device)
    seg_start, seg_end = int(cu_packed[3]), int(cu_packed[4])

    print("\nRunning isolated cascade (seq 3 alone) ...")
    iso = run_cascade(model, ids_iso, positions_iso, cu_iso, 1, device)
    print("Running packed cascade (seq0,seq1,seq2,seq3) ...")
    packed = run_cascade(model, ids_packed, positions_packed, cu_packed, 4, device)

    print("\n" + "=" * 100)
    print("PER-LAYER: local isolated-vs-packed divergence vs. gate saturation (seq 3's own tokens)")
    print("=" * 100)
    header = (f"  {'layer':>5} | {'local cosine':>12} | {'mean g_t':>9} | {'min g_t':>8} | "
              f"{'mean(1-g_t)':>12} | {'mean beta':>10} | {'mean dist-to-0/1':>17}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for i in linear_indices:
        cast_iso, nocast_iso = iso[i]["cast_out"], iso[i]["nocast_out"]
        cast_packed = packed[i]["cast_out"][seg_start:seg_end]
        c_local = cosine(cast_iso, cast_packed)

        g_t = packed[i]["g_t"][seg_start:seg_end]     # seq3's own rows, (9, LVH)
        beta = packed[i]["beta"][seg_start:seg_end]
        mean_g = g_t.mean().item()
        min_g = g_t.min().item()
        mean_retention = (1.0 - g_t).mean().item()
        mean_beta = beta.mean().item()
        dist_to_edge = torch.minimum(beta, 1.0 - beta).mean().item()  # 0 = fully saturated at an edge, 0.5 = as unsaturated as possible

        rows.append((i, c_local, mean_g, min_g, mean_retention, mean_beta, dist_to_edge))
        print(f"  {i:5d} | {c_local:12.6f} | {mean_g:9.6f} | {min_g:8.6f} | "
              f"{mean_retention:12.6f} | {mean_beta:10.6f} | {dist_to_edge:17.6f}")

    print("\n" + "=" * 100)
    print("CORRELATION CHECK: does higher g_t (weaker decay / longer memory) predict lower cosine?")
    print("=" * 100)

    cosines = [r[1] for r in rows]
    mean_gs = [r[2] for r in rows]
    n = len(rows)
    mean_c = sum(cosines) / n
    mean_g_all = sum(mean_gs) / n
    cov = sum((c - mean_c) * (g - mean_g_all) for c, g in zip(cosines, mean_gs)) / n
    std_c = (sum((c - mean_c) ** 2 for c in cosines) / n) ** 0.5
    std_g = (sum((g - mean_g_all) ** 2 for g in mean_gs) / n) ** 0.5
    pearson_r = cov / (std_c * std_g) if std_c > 0 and std_g > 0 else float("nan")

    print(f"  Pearson r (mean g_t vs. local cosine) across the 6 linear-attention layers: {pearson_r:.4f}")
    print("  (r close to -1: high g_t strongly predicts LOW cosine -- supports the saturation-")
    print("   sensitivity hypothesis. r close to 0: no relationship -- hypothesis not supported")
    print("   by this data; look elsewhere.)")

    if pearson_r < -0.5:
        print("\n  RESULT: correlates. Layers whose gates decay more weakly (g_t closer to 1, longer")
        print("  effective memory) show more isolated-vs-packed divergence. This supports gate-")
        print("  saturation/retention sensitivity as the amplification mechanism -- small bf16/fp32")
        print("  rounding differences between isolated and packed forward passes get compounded over")
        print("  seq 3's 9-step recurrence more in layers that retain state longer, roughly independent")
        print("  of how much divergence entered upstream -- consistent with Part 3's earlier finding")
        print("  that the penalty was flat (~0.969) regardless of packing composition/size: it's a")
        print("  per-layer amplification property, not a function of how much noise packing itself adds.")
    elif pearson_r > 0.5:
        print("\n  RESULT: correlates, but INVERTED from the hypothesis -- layers with WEAKER decay (g_t")
        print("  closer to 1) show LOWER divergence, and stronger-decay layers show MORE. Report this")
        print("  plainly: the gate-retention mechanism is real but backwards from what was predicted,")
        print("  which needs its own explanation before being used to justify anything.")
    else:
        print("\n  RESULT: does NOT correlate (|r| <= 0.5). Gate saturation/retention strength does not")
        print("  explain which layers show high vs. low local divergence. The per-layer variation has a")
        print("  different, not-yet-identified source -- the search continues from there, not from this")
        print("  explanation.")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)

    import torch.distributed as dist
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
