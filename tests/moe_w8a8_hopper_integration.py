"""W8A8 Hopper Phase 1 integration -- turns an already-loaded (bf16, already
EP-sharded if ep_size>1) Experts module into FP8 weights + 2D-blocked scale
buffers, in place, freeing the bf16 memory. FP8 analog of
tests/moe_int8_integration.py's quantize_experts_module_inplace /
apply_moe_int8_quantization, same structure, deliberately -- see that file
for the reasoning behind AFTER load_model() / BEFORE
warmup_model()/allocate_kv_cache() ordering, which applies identically here.

============================================================================
STATUS -- read before assuming this is wired up
============================================================================
UPDATED 2026-08-26 (CPU-only session, no GPU access this window): this
docstring described an earlier, unwired state of this file (Phase 1 written
in isolation, nothing called it). That has since changed --
engine/model_runner.py's __init__ DOES call
apply_moe_w8a8_hopper_quantization when config.use_moe_w8a8_hopper is set,
and models/qwen3_5.py's _forward_gathered_w8a8_hopper /
_forward_gathered_ep_w8a8_hopper (decode) and the FP8 elif branches in
_forward_dispatch / _forward_dispatch_ep (prefill) all read the buffers this
function registers. That part of the integration predates this session
(2026-08-23's "add True W8A8 (Hopper FP8)" commit) and is not new here.

What WAS still missing, and is what this session's edit fixes: the fused
kernel needs gate_up_proj's rows interleaved every 8 physically (see
gate_up_interleave_permutation's docstring in
layers/moe_w8a8_hopper_quantize.py), a requirement discovered 2026-08-24 via
the isolated smoke test, AFTER this file was originally written -- so this
function was quantizing the checkpoint's natural contiguous row order only,
which is correct for the prefill dequant branches but WRONG for the fused
kernel branches, which would have silently read a mismatched layout the
first time `use_moe_w8a8_hopper=True` ran end to end on real weights. Fixed
by registering a SECOND, permuted quantization (gate_up_proj_fp8_kernel /
gate_up_proj_scale_fp8_kernel) alongside the original, and repointing the
two fused-kernel call sites in models/qwen3_5.py at it. See
quantize_experts_module_fp8_inplace's docstring for the full reasoning.

**Still NOT validated**: this fix is CPU-only (permutation math + shape/
round-trip checks in tests/test_moe_w8a8_hopper_integration_cpu.py) --
nothing here has run against the actual compiled moe_w8a8.cu kernel with
these buffers plugged into a real engine. That needs a real Hopper GPU
(compile + either the isolated smoke test with a real checkpoint's dims, or
a full engine construction with use_moe_w8a8_hopper=True) before trusting
this in production, exactly the blocking item SESSION_HANDOFF_2026-08-25.md
flagged.

Storage-layout decision (memo §2a / "Decisions I'd like from you" #3) is
still open: whether these FP8 buffers live ADDITIVELY alongside the INT8
ones or eventually replace them. Implemented additively here (distinct
buffer names, gate_up_proj_fp8 not gate_up_proj_int8) because that's the
non-foreclosing default -- doesn't answer the question, just doesn't block
either answer.

============================================================================
WHAT THIS DOES NOT DO
============================================================================
- Does not touch utils/loader.py, load_model(), or shard_experts_tensor --
  same zero-changes-to-the-loading-path design as the INT8 version, proven
  safe for the 2D-block scale layout by
  tests/test_moe_w8a8_hopper_tp_ep_sharding_cpu.py. That test predates the
  permuted gate_up_proj_fp8_kernel buffer added this session and hasn't been
  re-run against it -- the underlying claim (quantize commutes with
  shard_experts_tensor's dim-0-only indexing) should still hold since
  permutation only reorders dim 1, but this is reasoned, not re-confirmed.
- Does not run an accuracy ablation against the real kernel -- needs real
  Hopper hardware, see STATUS above.
- Does not compute activation scales -- that's
  layers/moe_w8a8_hopper_quantize.py's quantize_activation_fp8_dynamic,
  called per-token in the decode hot path, not a load-time concern at all.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layers"))
from moe_w8a8_hopper_quantize import (          # noqa: E402
    quantize_weight_fp8_grouped,
    gate_up_interleave_permutation,
)

# Must match models/qwen3_5.py's _W8A8_HOPPER_BLOCK_N / _W8A8_HOPPER_WARP_N
# exactly -- gate_up_interleave_permutation's layout is config-dependent on
# this pair (see its docstring). Can't import those constants directly:
# models/qwen3_5.py has an unconditional triton import at module level that
# fails on any machine without triton (this dev machine included), so this
# module -- which needs to stay CPU-testable -- hardcodes the same values
# instead, same "assert rather than silently diverge" posture
# quantize_weight_fp8_grouped already takes with group_size==128.
_KERNEL_BLOCK_N = 32
_KERNEL_WARP_N = 8


def quantize_experts_module_fp8_inplace(experts: nn.Module, group_size: int = 128) -> None:
    """Mutates a live, already-loaded Experts module in place: quantizes its
    current gate_up_proj/down_proj Parameters to FP8 e4m3 + 2D-blocked
    per-tile scale, registers the results as buffers under new names, and
    DELETES the original bf16 Parameters -- same delete-is-the-point
    reasoning as quantize_experts_module_inplace (INT8): holding both bf16
    and FP8 simultaneously nets zero capacity win.

    Registers TWO gate_up_proj quantizations, not one -- same pattern
    layers/smoke_test_moe_w8a8_hopper.py's main() already validated at small
    scale (cosine 0.998 against a real Hopper run, 2026-08-24):
      - gate_up_proj_fp8 / gate_up_proj_scale_fp8: CONTIGUOUS gate=[0,MI),
        up=[MI,2*MI) layout, exactly what quantize_weight_fp8_grouped
        produces from the checkpoint's natural row order. Read by the plain
        dequant (prefill) branches in models/qwen3_5.py, which split
        gate/up via a plain .chunk(2, 0) and would silently produce the
        wrong SwiGLU split if given the interleaved layout instead.
      - gate_up_proj_fp8_kernel / gate_up_proj_scale_fp8_kernel: the SAME
        logical weight with rows reordered by gate_up_interleave_permutation
        before quantization (permute-then-quantize, not quantize-then-
        permute -- this keeps each 128x128 scale tile self-consistent for
        whatever row order it was computed from, rather than needing scale
        tiles reasoned about post-hoc under a row shuffle). Read only by the
        fused moe_w8a8.cu kernel call sites (_forward_gathered_w8a8_hopper /
        _forward_gathered_ep_w8a8_hopper), which need gate/up interleaved
        every 8 physical rows -- see that function's docstring for why.
    This doubles gate_up_proj's post-quantization memory (down_proj is
    unaffected -- it isn't a fused gate+up projection, so it never needed a
    second layout), trading capacity for reusing an already-proven-correct
    pattern instead of inverse-permuting inside the hot decode path.

    Requires experts.gate_up_proj.shape[-2:] and experts.down_proj.shape[-2:]
    both divisible by group_size (128) -- true at this project's real
    production dims (H=2048, MI=512 -> gate_up N=1024) but not guaranteed in
    general; quantize_weight_fp8_grouped asserts this rather than silently
    misbehaving.

    Device-agnostic like its INT8 counterpart -- no .cpu()/.cuda() calls, so
    this stays exercisable by a CPU-only test suite as well as the real CUDA
    call site this is designed for (ModelRunner.__init__, where
    torch.set_default_device("cuda") is already active). gate_up_interleave_
    permutation itself returns a CPU LongTensor; indexing a CUDA tensor with
    it works fine (torch moves the index tensor implicitly), so no explicit
    .to(device) is needed here either.

    Running both quantization schemes on the SAME module is not supported --
    config.py describes the two flags as "additive not a replacement" in the
    sense that they're independent Config fields, not in the sense that
    enabling both together does anything sane. If use_moe_w8a8=True already
    ran first (ModelRunner.__init__'s ordering), experts.gate_up_proj/
    down_proj are already deleted, replaced by *_int8 buffers -- reading them
    here would AttributeError with no indication of why. Fail with a message
    that says so instead of a bare attribute error.
    """
    if not hasattr(experts, "gate_up_proj"):
        raise RuntimeError(
            "quantize_experts_module_fp8_inplace: experts.gate_up_proj is already gone "
            "(quantized by something else first -- likely use_moe_w8a8=True's INT8 pass, "
            "which runs before this one in ModelRunner.__init__). Running both INT8 and "
            "W8A8 Hopper quantization on the same model is not supported; enable only one."
        )
    gu_fp8, gu_scale = quantize_weight_fp8_grouped(experts.gate_up_proj.data, group_size)
    dp_fp8, dp_scale = quantize_weight_fp8_grouped(experts.down_proj.data, group_size)

    MI = experts.gate_up_proj.shape[-2] // 2
    perm = gate_up_interleave_permutation(MI, _KERNEL_BLOCK_N, _KERNEL_WARP_N)
    gu_fp8_kernel, gu_scale_kernel = quantize_weight_fp8_grouped(
        experts.gate_up_proj.data[:, perm, :], group_size
    )

    # Same loud-failure-over-silent-wrong-state reasoning as the INT8
    # version: __delattr__ removes the Parameter entirely (not set to None),
    # so any forward path still expecting bf16 gate_up_proj/down_proj fails
    # with AttributeError instead of silently reading stale or absent state.
    del experts.gate_up_proj
    del experts.down_proj

    experts.register_buffer("gate_up_proj_fp8", gu_fp8)
    experts.register_buffer("gate_up_proj_scale_fp8", gu_scale)
    experts.register_buffer("gate_up_proj_fp8_kernel", gu_fp8_kernel)
    experts.register_buffer("gate_up_proj_scale_fp8_kernel", gu_scale_kernel)
    experts.register_buffer("down_proj_fp8", dp_fp8)
    experts.register_buffer("down_proj_scale_fp8", dp_scale)

    # Named distinctly from moe_w8a8_group_size (the INT8 scheme's own
    # group_size attribute) so a module quantized both ways -- not a real
    # scenario today, but nothing here prevents it structurally -- wouldn't
    # collide. Every forward path that eventually supports this reads it
    # directly off the module, same self-contained-state convention as INT8.
    experts.moe_w8a8_hopper_group_size = group_size


def apply_moe_w8a8_hopper_quantization(model: nn.Module, group_size: int = 128) -> int:
    """Walks the model, quantizes every MoE layer's Experts module to FP8 in
    place. Returns the count quantized, matching apply_moe_int8_quantization
    and this project's "Found N Qwen35MoE layer(s)" logging convention.

    CALLED from engine/model_runner.py's __init__ when config.use_moe_w8a8_hopper
    is set (since 2026-08-23). The load-time pass + the forward-path buffer
    reads are wired; what's still missing is a real moe_w8a8.cu compile+run on
    Hopper (Phase 0) -- see the module docstring.
    """
    from nanovllm.models.qwen3_5 import Experts

    count = 0
    for module in model.modules():
        if isinstance(module, Experts):
            quantize_experts_module_fp8_inplace(module, group_size)
            count += 1
    return count
