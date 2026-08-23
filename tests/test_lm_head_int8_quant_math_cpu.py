"""CPU-only correctness check for lm_head INT8 quantization -- confirms
quantize_weight_int8_grouped/dequantize_weight_int8_grouped (already
validated for the MoE's batched (E, out, in) case) also work correctly on a
PLAIN 2D (out_features, in_features) tensor with no leading expert
dimension, at lm_head's REAL dims (vocab_size=248320, hidden_size=2048,
group_size=128 -> 16 groups exact).

Goes one step further than every other quantization test this session: an
ARGMAX-AGREEMENT check, not just cosine similarity. Cosine can stay high
while still flipping a close top-1/top-2 logit gap -- and for lm_head
specifically, that flip IS the failure mode that matters (it changes which
token gets sampled under greedy/temperature=0 decoding, unlike an
intermediate FFN layer where downstream layers can partially average out
small errors). Tests on synthetic random (hidden_state, weight) pairs, not
real model activations (no GPU/triton access here) -- a proxy, not a
guarantee, but a stricter and more directly relevant one than cosine alone.

Usage:
    python tests/test_lm_head_int8_quant_math_cpu.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_int8_quantize import quantize_weight_int8_grouped, dequantize_weight_int8_grouped  # noqa: E402


def main():
    torch.manual_seed(0)
    ok = True

    # ---- 1. Shape check at REAL lm_head dims, plain 2D (no leading dim) ----
    vocab_size, hidden_size, group_size = 248320, 2048, 128
    # Full vocab_size weight at fp32 is a real allocation (~2GB) -- shrink
    # vocab for the quantization-math check (same "size doesn't change the
    # math, only divisibility does" reasoning already used for the W8A8
    # Hopper sharding proof's CPU-RAM constraint) while keeping hidden_size
    # at its real value, since THAT'S what group_size divisibility depends on.
    V = 4096  # stand-in vocab size, real hidden_size below
    weight = torch.randn(V, hidden_size) * 0.02

    w_int8, scale = quantize_weight_int8_grouped(weight, group_size)
    print(f"[1] weight={tuple(weight.shape)} -> int8={tuple(w_int8.shape)} scale={tuple(scale.shape)}")
    assert w_int8.shape == (V, hidden_size), "plain-2D quantize shape mismatch (no leading dim expected)"
    assert scale.shape == (V, hidden_size // group_size), "scale shape mismatch"
    assert w_int8.dtype == torch.int8

    # ---- 2. Reconstruction cosine (same bar as the MoE case) ----
    deq = dequantize_weight_int8_grouped(w_int8, scale, group_size, torch.float32)
    cos = F.cosine_similarity(weight.reshape(-1), deq.reshape(-1), dim=0).item()
    print(f"[2] reconstruction cosine={cos:.8f}")
    assert cos > 0.999, "reconstruction cosine too low for a working RTN quantizer"
    ok &= cos > 0.999

    # ---- 3. THE check that matters for an output layer: argmax agreement.
    # Simulate N independent (hidden_state, logits) forward passes: random
    # unit-scale hidden vectors matmul'd against the ORIGINAL fp32 weight
    # (ground truth) vs. the DEQUANTIZED weight (what the real forward path
    # would use), compare which vocab entry wins. ----
    N_TRIALS = 2000
    hidden = torch.randn(N_TRIALS, hidden_size) * 0.02
    logits_true = hidden @ weight.T           # (N_TRIALS, V), fp32 ground truth
    logits_deq = hidden @ deq.T                # (N_TRIALS, V), int8-dequantized weight

    argmax_true = logits_true.argmax(dim=-1)
    argmax_deq = logits_deq.argmax(dim=-1)
    agree = (argmax_true == argmax_deq).float().mean().item()
    print(f"[3] argmax agreement over {N_TRIALS} synthetic forward passes: {agree*100:.2f}%")
    # Threshold reasoning: at V=4096 (this test's stand-in vocab) with random
    # untrained-scale weights, many logits sit close together by construction
    # (no real semantic separation the way a trained model's true top token
    # would have) -- this is a harder, more adversarial setting than a real
    # forward pass, where the actual top token is usually a clear winner, not
    # a near-tie. Treat this as a lower-bound stress test, not a prediction
    # of real-model behavior; the honest number to trust is the real-model
    # GSM8K non-regression check (Phase 3 equivalent), still needs GPU.
    print("    (synthetic random weights, not real model logits -- a stress test, "
          "not a prediction of real-model argmax-flip rate; see GPU validation note below)")
    ok &= agree > 0.90

    # ---- 4. Gather/dequant order-independence -- same class of check every
    # other quantization test this session ran, even though lm_head has no
    # gather step in its own forward path (F.linear reads the whole tensor
    # directly) -- checking it anyway costs nothing and rules out a class of
    # indexing bug if this quantizer is ever reused somewhere with a gather. ----
    idx = torch.randint(0, V, (64,))
    deq_gathered = dequantize_weight_int8_grouped(w_int8[idx], scale[idx], group_size, torch.float32)
    deq_indexed_after = deq[idx]
    max_diff = (deq_gathered - deq_indexed_after).abs().max().item()
    print(f"[4] gather-before vs. gather-after dequant max diff: {max_diff:.3e}")
    assert max_diff < 1e-6
    ok &= max_diff < 1e-6

    print("\nPASS" if ok else "\nFAIL")
    print("\nScope reminder: validates the quantization MATH on synthetic weights/activations "
          "only -- says nothing about real model logits, real argmax-flip rate on the actual "
          "checkpoint, or the forward-path dequant-then-matmul's THROUGHPUT impact (see "
          "tests/lm_head_int8_integration.py's own module docstring on the capacity-vs-throughput "
          "distinction). GSM8K non-regression on real hardware is still the check that matters "
          "most for this specific layer, given the argmax-sensitivity reasoning above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
