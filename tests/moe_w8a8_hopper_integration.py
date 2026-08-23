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
This is Phase 1 of w8a8_activation_quant_scoping_memo.md: the WEIGHT side of
true W8A8, written and CPU-tested, matching the discipline the tp=1 INT8 fix
used (write blind, CPU-validate what's CPU-validatable, GPU-validate when
hardware exists). NOT wired into ModelRunner.__init__ or any Config field --
unlike apply_moe_int8_quantization, nothing calls
apply_moe_w8a8_hopper_quantization anywhere yet. That's deliberate: doing so
requires the forward-path integration (Phase 2/4 of the scoping memo,
activation quantization + the actual moe_w8a8_hopper_forward call sites in
models/qwen3_5.py) to exist first, and requires moe_w8a8.cu to have compiled
and passed its own isolated smoke test (Phase 0) before any of this touches
a real forward pass. Wiring this in ahead of those would let
`use_moe_w8a8_hopper=True` construct successfully and then produce silently
wrong output -- exactly the failure shape this project avoids by design.

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
  same zero-changes-to-the-loading-path design as the INT8 version, now
  proven safe for THIS (2D-block) scale layout too by
  tests/test_moe_w8a8_hopper_tp_ep_sharding_cpu.py, not just assumed to
  transfer from the INT8 case.
- Does not touch models/qwen3_5.py. No forward path reads
  gate_up_proj_fp8/down_proj_fp8 yet.
- Does not run an accuracy ablation -- needs the full Phase 0-4 chain first.
- Does not compute activation scales -- that's
  layers/moe_w8a8_hopper_quantize.py's quantize_activation_fp8_dynamic,
  called per-token in the decode hot path once Phase 2 wires it in, not a
  load-time concern at all.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layers"))
from moe_w8a8_hopper_quantize import quantize_weight_fp8_grouped  # noqa: E402


def quantize_experts_module_fp8_inplace(experts: nn.Module, group_size: int = 128) -> None:
    """Mutates a live, already-loaded Experts module in place: quantizes its
    current gate_up_proj/down_proj Parameters to FP8 e4m3 + 2D-blocked
    per-tile scale, registers the results as buffers under new names, and
    DELETES the original bf16 Parameters -- same delete-is-the-point
    reasoning as quantize_experts_module_inplace (INT8): holding both bf16
    and FP8 simultaneously nets zero capacity win.

    Requires experts.gate_up_proj.shape[-2:] and experts.down_proj.shape[-2:]
    both divisible by group_size (128) -- true at this project's real
    production dims (H=2048, MI=512 -> gate_up N=1024) but not guaranteed in
    general; quantize_weight_fp8_grouped asserts this rather than silently
    misbehaving.

    Device-agnostic like its INT8 counterpart -- no .cpu()/.cuda() calls, so
    this stays exercisable by a CPU-only test suite as well as the real CUDA
    call site this is designed for (ModelRunner.__init__, where
    torch.set_default_device("cuda") is already active).
    """
    gu_fp8, gu_scale = quantize_weight_fp8_grouped(experts.gate_up_proj.data, group_size)
    dp_fp8, dp_scale = quantize_weight_fp8_grouped(experts.down_proj.data, group_size)

    # Same loud-failure-over-silent-wrong-state reasoning as the INT8
    # version: __delattr__ removes the Parameter entirely (not set to None),
    # so any forward path still expecting bf16 gate_up_proj/down_proj fails
    # with AttributeError instead of silently reading stale or absent state.
    del experts.gate_up_proj
    del experts.down_proj

    experts.register_buffer("gate_up_proj_fp8", gu_fp8)
    experts.register_buffer("gate_up_proj_scale_fp8", gu_scale)
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

    NOT called from ModelRunner.__init__ yet -- see module docstring. This
    function existing does not mean use_moe_w8a8_hopper=True works; nothing
    passes that flag or calls this today.
    """
    from nanovllm.models.qwen3_5 import Experts

    count = 0
    for module in model.modules():
        if isinstance(module, Experts):
            quantize_experts_module_fp8_inplace(module, group_size)
            count += 1
    return count
