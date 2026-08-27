"""lm_head INT8 weight-only quantization -- load-time integration, mirrors
tests/moe_int8_integration.py's quantize_experts_module_inplace /
apply_moe_int8_quantization exactly, applied to ParallelLMHead instead of
Experts.

============================================================================
WHY THIS EXISTS
============================================================================
lm_head (layers/embed_head.py's ParallelLMHead) was never touched by any
quantization work this session -- everything went into the MoE experts.
Real numbers, checked against the actual checkpoint: vocab_size=248320,
hidden_size=2048 -> lm_head.weight is ~508.6M params, ~45% of the FULL
40-layer MoE stack's own per-token weight-read traffic (~1.134B params) --
a large, completely unaddressed weight sitting in the single dense matmul
computed once per token, AFTER CUDA graph replay returns
(engine/model_runner.py's run(): `logits = self.model.compute_logits(hidden)`
runs eagerly, outside `if use_graph`).

Structurally, this is the simplest quantization target in the whole model:
no batched-expert tensor, no routing, no EP sharding. It reuses
quantize_weight_int8_grouped / dequantize_weight_int8_grouped
(tests/moe_int8_quantize.py) UNMODIFIED -- that function already generalizes
to a plain 2D (out_features, in_features) tensor via its `*lead` leading-dims
unpacking (lead=() for lm_head, vs. lead=(num_experts,) for the MoE case) --
confirmed by reading it, not assumed.

============================================================================
CAPACITY vs. THROUGHPUT -- read before assuming this is a speed win
============================================================================
This quantization pass gives a real, unconditional CAPACITY win: ~485MiB
less resident VRAM (508.6M params x 1 byte saved/param) at rest, exactly
the kind of KV-cache-headroom win the MoE INT8 work already banked on.

It does NOT come with a proven THROUGHPUT win, and there is a concrete
reason to expect the naive dequant-then-matmul path here could be a
REGRESSION, not an improvement: unlike the MoE experts (which gather only
top_k of 256 experts per token -- a SUBSET), lm_head's forward pass
(`F.linear(x, self.weight)`) reads its ENTIRE weight matrix every single
call, every decode step. Dequantizing the full tensor fresh each call costs
int8-read (1x, ~485MiB) + bf16-write (2x, ~970MiB) = ~1455MiB of NEW
traffic, which the matmul then reads AGAIN (~970MiB) -- roughly 2.4GiB total
vs. bf16-direct's ~970MiB (the matmul just reads the weight once, no
dequant step at all). That is a plausible ~2.5x REGRESSION, not a win --
the EXACT bandwidth trap moe_quantization_memo.md section 4 already found
and fixed for the MoE experts (the pre-fused-kernel fp32-round-trip
mistake), caught here by reasoning before writing code instead of by a
matched A/B on real hardware after shipping it.

The forward-path wiring (layers/embed_head.py's ParallelLMHead.forward)
still uses the naive dequant-then-F.linear approach below -- it is not
WRONG, correctness is unaffected, but its throughput impact is UNKNOWN and
plausibly negative until measured. Do not present `use_lm_head_int8=True`
as a speed optimization without a matched-settings A/B first, same
discipline the MoE memo's own history demands. If it does regress, the
established fix is the same one that worked for MoE: a fused kernel that
dequantizes inside the GEMM's inner loop, never materializing a full bf16
copy -- real, separate follow-on work, not a trivial addition.

============================================================================
WHAT THIS DOES NOT DO
============================================================================
- Does not touch utils/loader.py or the weight-loading path -- quantization
  runs AFTER load_model() completes, same ordering as the MoE integration.
- Does not prove TP-sharding commutativity with a dedicated test the way
  the MoE (Q3) and W8A8 Hopper sharding proofs did -- but the argument is
  simpler here and worth stating plainly rather than skipped: quantization
  groups only along the LAST (hidden_size/in_features) dimension, with zero
  cross-row interaction; VocabParallelEmbedding's own weight_loader shards
  strictly along dim 0 (vocab, a contiguous narrow(), not even round-robin).
  A dim-0-only slice trivially commutes with a quantization scheme that only
  ever reduces within a row -- this is a strictly simpler case than either
  already-proven MoE scheme (round-robin dim-0 sharding of a scheme that
  ALSO had no cross-expert interaction), not a new risk.
- Does not build a fused kernel. Naive dequant-then-matmul only, explicitly
  not claimed as a throughput fix -- see above.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_int8_quantize import quantize_weight_int8_grouped  # noqa: E402


def quantize_lm_head_inplace(lm_head: nn.Module, group_size: int = 128) -> None:
    """Mutates a live, already-loaded ParallelLMHead module in place:
    quantizes its current .weight Parameter to INT8 + per-group scale,
    registers the results as buffers, and DELETES the original bf16
    Parameter -- same delete-is-the-point reasoning as
    quantize_experts_module_inplace: holding both bf16 and int8
    simultaneously nets zero capacity win.

    lm_head.weight shape: (vocab_size // tp_size, hidden_size) --
    VocabParallelEmbedding's own sharding, already applied by the time this
    runs (this is called after load_model(), which already ran
    weight_loader). group_size must evenly divide hidden_size (2048 at
    group_size=128 -> 16 groups exact, same arithmetic as the MoE case,
    same model's hidden_size).

    Device-agnostic, no .cpu()/.cuda() calls -- same CPU-testable
    convention as quantize_experts_module_inplace.
    """
    w_int8, w_scale = quantize_weight_int8_grouped(lm_head.weight.data, group_size)

    del lm_head.weight

    lm_head.register_buffer("weight_int8", w_int8)
    lm_head.register_buffer("weight_scale", w_scale)
    lm_head.lm_head_int8_group_size = group_size


def apply_lm_head_int8_quantization(model: nn.Module, group_size: int = 128) -> int:
    """Walks the model, quantizes the lm_head module in place if found.
    Returns 1 if quantized, 0 if no lm_head attribute exists -- unlike
    apply_moe_int8_quantization (which counts N Experts modules across many
    layers), there is exactly one lm_head per model, so this is a
    presence check, not a count of a repeated structure.

    CALLED from engine/model_runner.py's __init__ when config.use_lm_head_int8
    is set (after load_model, before warmup) -- see config.py's comment on
    the flag. The load-time pass + ParallelLMHead.forward()'s int8
    dequant-on-read branch are both wired; what's unresolved is the
    throughput question (see the module docstring's capacity-vs-throughput
    section), which needs a GPU matched A/B.
    """
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return 0
    # If tie_word_embeddings is set, models/qwen3_5.py's Qwen35ForCausalLM.__init__
    # does `lm_head.weight.data = embed_tokens.weight.data` -- deleting lm_head.weight
    # below then only drops LM_HEAD's reference; embed_tokens still holds the same
    # underlying bf16 storage alive, so the whole point of this pass (freeing that
    # memory) silently doesn't happen. Not a correctness bug -- inference still
    # produces the right numbers -- but the ~485MiB capacity claim this pass exists
    # for would be false in that configuration. Not applicable to the real checkpoint
    # (tie_word_embeddings=False there, confirmed against config.json), but loud here
    # rather than silent in case this ever runs against a tied checkpoint.
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        print("[LM_HEAD INT8] WARNING: tie_word_embeddings=True -- lm_head.weight shares "
              "storage with the input embedding table. Quantizing it will NOT free that "
              "memory (embed_tokens keeps the bf16 data alive), so the capacity win this "
              "flag exists for will not happen, even though quantization itself proceeds "
              "and produces correct output.")
    quantize_lm_head_inplace(lm_head, group_size)
    return 1
