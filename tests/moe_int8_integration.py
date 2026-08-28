"""Q4 integration -- turns an already-loaded (bf16, already EP-sharded if
ep_size>1) Experts module into INT8 weights + per-group scale buffers, in
place, freeing the bf16 memory.

============================================================================
WHY THIS RUNS WHERE IT RUNS
============================================================================
Called from ModelRunner.__init__, AFTER load_model(self.model, config.model)
and BEFORE warmup_model()/allocate_kv_cache(). Both boundaries matter:

  AFTER load_model() -- Q3 (tests/test_moe_int8_ep_shard_commutativity.py)
  proved quantizing an already-sharded local slice of experts gives
  bitwise-identical int8 weights and scales to quantizing before sharding.
  That is what makes this safe: load_model() and shard_experts_tensor run
  completely UNCHANGED, exactly as they do today for the unquantized path.
  This file adds a pass AFTER them, not a modification to them.

  BEFORE warmup_model()/allocate_kv_cache() -- allocate_kv_cache()'s KV
  budget formula (engine/model_runner.py) is
      int(total * gpu_memory_utilization - used - peak + current) // block_bytes
  where `used = total - free` comes from torch.cuda.mem_get_info(). The
  freed bf16 Parameter memory only counts toward that budget once it is
  visible to mem_get_info() -- which requires torch.cuda.empty_cache()
  after the del, not just Python reference-counting the tensor away.
  Skipping that call would make this pass free memory in a sense
  memory_stats() sees but mem_get_info() (and therefore the KV-cache
  budget) does not -- the capacity win would be real but invisible to the
  one calculation it's meant to unblock.

============================================================================
WHAT THIS DOES NOT DO
============================================================================
- Does not touch utils/loader.py, load_model(), or shard_experts_tensor.
  Zero changes to the bf16 loading/sharding path, by design (see Q3).
- Does not touch models/qwen3_5.py at all -- this file only quantizes the
  weights; the four forward paths that read the resulting int8 buffers
  (_forward_gathered_ep, _forward_dispatch_ep, and -- as of the
  tensor_parallel_size=1 fix -- _forward_gathered and _forward_dispatch)
  each check `hasattr(self.experts, "gate_up_proj_int8")` independently in
  models/qwen3_5.py itself. STALE NOTE, kept for history: this comment
  originally said only _forward_gathered_ep had been patched and the other
  three were separate follow-on work -- that was true when this file was
  first written (Q4), not anymore; all four now support int8, so
  use_moe_w8a8=True works at any tensor_parallel_size, not EP-only.
  _forward_gathered_ep's fused-Triton-kernel path
  (NANOVLLM_USE_FUSED_MOE_KERNEL) remains EP-only, not yet ported to the
  non-EP decode path -- that piece genuinely is still open.
- Does not run an accuracy ablation. That is Q6, real weights through the
  engine, same protocol as every other correctness-gate check in this
  project.
"""
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_int8_quantize import quantize_weight_int8_grouped  # noqa: E402


def quantize_experts_module_inplace(experts: nn.Module, group_size: int) -> None:
    """Mutates a live, already-loaded Experts module in place: quantizes its
    current gate_up_proj/down_proj Parameters to INT8 + per-group scale,
    registers the results as buffers under new names, and DELETES the
    original bf16 Parameters. The delete is the entire point -- holding
    both bf16 and int8 simultaneously nets zero capacity win.

    Runs on whatever device experts.gate_up_proj already lives on (CUDA,
    during the real ModelRunner.__init__ call site this is designed for --
    ModelRunner sets torch.set_default_device("cuda") before model
    construction and load_model(), only resetting to "cpu" after
    allocate_kv_cache(), so this pass's tensors are CUDA tensors like
    everything else at this point in construction). No .cpu()/.cuda() calls
    here on purpose -- this function is device-agnostic and should stay
    that way for the CPU-only test suite that exercises it.
    """
    gu_int8, gu_scale = quantize_weight_int8_grouped(experts.gate_up_proj.data, group_size)
    dp_int8, dp_scale = quantize_weight_int8_grouped(experts.down_proj.data, group_size)

    # nn.Module.__delattr__ removes a registered Parameter from _parameters
    # cleanly. After this, experts.gate_up_proj no longer exists as an
    # attribute at all (not set to None -- genuinely gone), so any code
    # path that still expects it will fail loudly (AttributeError) rather
    # than silently reading a None and producing a confusing downstream
    # error. That is deliberate, matching this project's stated preference
    # for loud failures over silent wrong state.
    del experts.gate_up_proj
    del experts.down_proj

    experts.register_buffer("gate_up_proj_int8", gu_int8)
    experts.register_buffer("gate_up_proj_scale", gu_scale)
    experts.register_buffer("down_proj_int8", dp_int8)
    experts.register_buffer("down_proj_scale", dp_scale)

    # The ONLY new piece of state this pass adds beyond the four buffers
    # above. Every forward path that supports int8 (_forward_gathered_ep,
    # _forward_dispatch_ep, and -- as of the tp=1 fix -- _forward_gathered
    # and _forward_dispatch too) reads this directly off the module -- no
    # config threading needed anywhere else. The buffers' presence and this
    # one int are the complete, self-contained state quantization adds;
    # there is no third place that needs to agree with them.
    experts.moe_w8a8_group_size = group_size


def apply_moe_int8_quantization(model: nn.Module, group_size: int) -> int:
    """Walks the model, quantizes every MoE layer's Experts module in place.
    Returns the count quantized, for logging (matches this project's
    existing "Found N Qwen35MoE layer(s)" logging convention used
    elsewhere in the diagnostic scripts)."""
    from nanovllm.models.qwen3_5 import Experts

    count = 0
    for module in model.modules():
        if isinstance(module, Experts):
            quantize_experts_module_inplace(module, group_size)
            count += 1
    return count