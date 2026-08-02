"""Extends diag_testA_followup.py's Part 2 (single fresh-state layer
isolation) across the REAL layer stack: runs seq 3 isolated and seq 3
packed (with seq 0/1/2, original order) through the actual sequential
decoder-layer loop (embedding -> layer0 -> layer1 -> ... -> layer7),
capturing the residual-stream state after EACH layer, so we can see
exactly where isolated-vs-packed cosine first drops -- gradual creep
through the linear_attention layers (0,1,2 / 4,5,6), or a cliff right at
the first full_attention layer (layer 3)?

Two cascading regimes, run end-to-end:
  bf16    -- fully native, unpatched, exactly what the real engine runs.
  fp32gdr -- every linear_attention layer's OWN conv1d/in_proj_*/scan runs
             in float32 (weights upcast losslessly, matches
             diag_multi_hypothesis.py's Test A patch), cast back to bf16
             immediately after each such layer so the residual stream
             stays bf16 and full_attention layers (bf16/fp16-only,
             flash_attn) keep working. full_attention layers are NEVER
             patched (can't be -- flash_attn has no fp32 kernel).

At every linear_attention layer, ALSO captures that layer's own local
cast-vs-nocast comparison (same technique as diag_testA_followup.py's
Part 2), but now fed with the REAL cascaded input arriving at that point
under the fp32gdr regime -- not a fresh embedding -- so any accumulation
effect from prior layers is visible in this local check too.

Layer schedule for the small test config (full_attention_interval=4,
num_hidden_layers=8): layer 3 and layer 7 are full_attention; every other
layer (0,1,2,4,5,6) is linear_attention.

Usage: python tests/diag_layerwise_cascade.py
"""
import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from test_qwen35_standalone import init_dist, make_small_config   # noqa: E402
from test_qwen35_batching import build_model_and_state   # noqa: E402

from nanovllm.models.qwen3_5 import Qwen35LinearAttention   # noqa: E402
from nanovllm.utils.context import set_context, reset_context   # noqa: E402

_ORIG_LA_FORWARD = Qwen35LinearAttention.forward  # captured once, never reassigned globally


def cosine(a, b):
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def run_layers_capture(model, ids, positions, cu_seqlens, num_segments, device, regime):
    """Replicates Qwen35Model.forward's layer loop by hand, capturing the
    residual-stream state (hidden_states + residual -- what the NEXT
    layer's fused input_layernorm would add together) after each layer.

    Returns:
      captures[i]    -- (N, H) float32 residual-stream state after layer i
      local_casts[i] -- None for full_attention layers; for linear_attention
                        layers under regime=='fp32gdr', (cast_out, nocast_out)
                        both float32 (N, H) -- that layer's OWN linear_attn
                        output, with vs without the final bf16 cast, given
                        its REAL cascaded input at this point in this run.
    """
    hidden_states = model.model.embed_tokens(ids)
    residual = None
    captures = []
    local_casts = []

    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    set_context(True, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, None, None, None)
    try:
        for layer in model.model.layers:
            if residual is None:
                normed, residual = layer.input_layernorm(hidden_states), hidden_states
            else:
                normed, residual = layer.input_layernorm(hidden_states, residual)

            if layer.is_full:
                with torch.no_grad():
                    hidden_states = layer.self_attn(positions, normed)
                local_casts.append(None)
            else:
                la = layer.linear_attn
                conv0 = torch.zeros(num_segments, la.qkv_dim, la.ck - 1, device=device, dtype=torch.bfloat16)
                state0 = torch.zeros(num_segments, la.lvh, la.lhd, la.lhd, device=device, dtype=torch.float32)

                if regime == "bf16":
                    with torch.no_grad():
                        hidden_states, _, _ = _ORIG_LA_FORWARD(la, normed, cu_seqlens, state0, conv0)
                    local_casts.append(None)
                else:  # fp32gdr
                    la.float()
                    try:
                        with torch.no_grad():
                            out32, _, _ = _ORIG_LA_FORWARD(la, normed.float(), cu_seqlens, state0, conv0.float())
                        cast_out = out32.to(normed.dtype)
                        hidden_states = cast_out
                        local_casts.append((cast_out.float(), out32.float()))
                    finally:
                        la.to(torch.bfloat16)

            normed2, residual = layer.post_attention_layernorm(hidden_states, residual)
            with torch.no_grad():
                hidden_states = layer.mlp(normed2, cu_seqlens)

            captures.append((hidden_states.float() + residual.float()))
    finally:
        reset_context()

    return captures, local_casts


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Layer-by-layer cascade diagnostic -- device={device}")
    init_dist()

    config = make_small_config()
    model, _ = build_model_and_state(config, device, max_num_seqs=8)
    layer_types = model.model.layer_types
    num_layers = len(model.model.layers)

    torch.manual_seed(99)
    seqs = [torch.randint(100, 5000, (n,)).tolist() for n in [7, 11, 5, 9]]  # seq3 = seqs[3], len=9
    lengths = [len(s) for s in seqs]

    ids_iso = torch.tensor(seqs[3], dtype=torch.int64, device=device)
    positions_iso = torch.arange(lengths[3], device=device)
    cu_iso = torch.tensor([0, lengths[3]], dtype=torch.int32, device=device)

    ids_packed = torch.tensor(sum(seqs, []), dtype=torch.int64, device=device)
    positions_packed = torch.cat([torch.arange(l, device=device) for l in lengths])
    cu_packed = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device=device)
    seg_start, seg_end = int(cu_packed[3]), int(cu_packed[4])  # seq3's slice within the packed run

    results = {}
    for regime in ["bf16", "fp32gdr"]:
        print(f"\nRunning regime={regime} ...")
        caps_iso, local_iso = run_layers_capture(model, ids_iso, positions_iso, cu_iso, 1, device, regime)
        caps_packed, local_packed = run_layers_capture(model, ids_packed, positions_packed, cu_packed, 4, device, regime)
        results[regime] = (caps_iso, local_iso, caps_packed, local_packed)

    print("\n" + "=" * 100)
    print("PER-LAYER RESIDUAL-STREAM COSINE: seq3 isolated vs seq3-within-packed-4")
    print("=" * 100)
    header = f"  {'layer':>5} | {'type':<16} | {'bf16 cascade':>13} | {'fp32gdr cascade':>16} | {'local cast':>11} | {'local nocast':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    bf16_caps_iso, _, bf16_caps_packed, _ = results["bf16"]
    fp32_caps_iso, fp32_local_iso, fp32_caps_packed, fp32_local_packed = results["fp32gdr"]

    first_drop_bf16 = None
    first_drop_fp32 = None
    for i in range(num_layers):
        ltype = layer_types[i]
        c_bf16 = cosine(bf16_caps_iso[i], bf16_caps_packed[i][seg_start:seg_end])
        c_fp32 = cosine(fp32_caps_iso[i], fp32_caps_packed[i][seg_start:seg_end])

        if ltype == "linear_attention":
            cast_iso, nocast_iso = fp32_local_iso[i]
            cast_packed, nocast_packed = fp32_local_packed[i]
            c_local_cast = cosine(cast_iso, cast_packed[seg_start:seg_end])
            c_local_nocast = cosine(nocast_iso, nocast_packed[seg_start:seg_end])
            print(f"  {i:5d} | {ltype:<16} | {c_bf16:13.6f} | {c_fp32:16.6f} | {c_local_cast:11.6f} | {c_local_nocast:12.6f}")
        else:
            print(f"  {i:5d} | {ltype:<16} | {c_bf16:13.6f} | {c_fp32:16.6f} | {'--':>11} | {'--':>12}")

        if first_drop_bf16 is None and c_bf16 < 0.999:
            first_drop_bf16 = i
        if first_drop_fp32 is None and c_fp32 < 0.999:
            first_drop_fp32 = i

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  First layer where bf16-cascade cosine drops below 0.999: "
          f"{'layer ' + str(first_drop_bf16) + ' (' + layer_types[first_drop_bf16] + ')' if first_drop_bf16 is not None else 'never (stays >=0.999 through all 8 layers)'}")
    print(f"  First layer where fp32gdr-cascade cosine drops below 0.999: "
          f"{'layer ' + str(first_drop_fp32) + ' (' + layer_types[first_drop_fp32] + ')' if first_drop_fp32 is not None else 'never (stays >=0.999 through all 8 layers)'}")

    if first_drop_bf16 is not None and layer_types[first_drop_bf16] == "linear_attention":
        print("\n  Divergence appears WHILE still inside the linear_attention layers, before any "
              "full_attention layer runs -- GDR IS implicated (whether via its own math or via "
              "cross-layer state/conv accumulation), not just the full_attention layers downstream.")
    elif first_drop_bf16 == 3:
        print("\n  Divergence appears for the FIRST time exactly at layer 3, the first full_attention "
              "layer -- all preceding linear_attention layers (0,1,2) stayed clean. This directly "
              "supports the full_attention/flash_attn hypothesis: the divergence is not accumulating "
              "through GDR at all, it's introduced fresh at the first flash_attn_varlen_func call.")
    elif first_drop_bf16 is None:
        print("\n  No layer in this trace drops below 0.999 -- the divergence measured at the FINAL "
              "logits (cosine ~0.952-0.996 across sequences) must be concentrated in compute_logits "
              "(the lm_head projection) or accumulate below the 0.999 threshold used here; rerun with "
              "a tighter threshold.")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)

    import torch.distributed as dist
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
