"""INT8 groupwise round-to-nearest quantization for the batched MoE expert
tensors (Experts.gate_up_proj, Experts.down_proj).

============================================================================
SCHEME
============================================================================
Symmetric INT8, grouped along the REDUCTION (input) dimension, group_size
elements per group, one scale per (expert, output_channel, group).

Weight layout, matching models/qwen3_5.py's Experts module exactly:
    gate_up_proj: (num_experts, 2*intermediate_size, hidden_size)
    down_proj:    (num_experts, hidden_size, intermediate_size)
Both are (E, out_features, in_features) -- nn.Linear convention, confirmed
in models/qwen3_5.py's own comments and by utils/loader.py's weight-loading
path (no transpose anywhere in that path, per the vectorized-MoE closeout).
The reduction (matmul contraction) dimension is always the LAST one.

Real checkpoint dims (confirmed via config.json, Q0):
    hidden_size=2048  moe_intermediate_size=512  num_experts=256
At group_size=128:
    gate_up_proj: in_features=2048  -> 16 groups, EXACT
    down_proj:    in_features=512   -> 4 groups,  EXACT
No padding logic is needed for the real checkpoint's shapes. This module
still asserts divisibility rather than silently padding, on the general
principle established throughout this project: a shape mismatch should
fail loudly, not be quietly worked around in a way that could paper over a
config error later (see ARCH_DISPATCH / load_model()'s continue-on-
unmatched-key precedent this project has already been burned by twice).

============================================================================
WHAT THIS FILE DOES NOT DO
============================================================================
- No calibration, no activation statistics, no library dependency
  (AutoAWQ/GPTQModel). Pure RTN on already-trained weights. This is
  deliberate -- see the quantization scoping memo's argument for why this
  makes the "Experts is batched, not per-expert nn.Linear" blocker a
  non-issue for weight-only INT8 specifically.
- No TP/EP-aware scale SHARDING. That is Q3, a separate file, because the
  hazard there (a naive dim-0 chunk mispairing scales against weights
  under sharding) is exactly the class of bug this project's incident
  history warns about (in_proj_qkv merged-chunk bug), and deserves its own
  test in isolation rather than being folded into the quantization math.
- No integration into the decode forward path. That is Q4.

Usage (as a library):
    from tests.moe_int8_quantize import quantize_weight_int8_grouped, dequantize_weight_int8_grouped
"""
import torch


def quantize_weight_int8_grouped(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 RTN quantization, grouped along the last (reduction)
    dimension.

    Args:
        weight: (..., out_features, in_features), any float dtype.
        group_size: must evenly divide in_features.

    Returns:
        int8_weight: same shape as weight, dtype int8, values in [-127, 127].
                     (Not [-128, 127] -- symmetric quantization keeps the
                     range balanced around zero, matching the scale
                     definition below. -128 is deliberately unused.)
        scale: (..., out_features, in_features // group_size), same float
               dtype as the input weight. scale[..., g] is the dequant
               multiplier for group g -- dequantize as
               int8_weight.float() * scale.repeat_interleave(group_size, dim=-1).
    """
    *lead, out_features, in_features = weight.shape
    assert in_features % group_size == 0, (
        f"in_features={in_features} not divisible by group_size={group_size}. "
        f"This function does not pad -- either group_size is wrong for this "
        f"tensor, or this tensor doesn't belong to the MoE experts this "
        f"scheme was designed around."
    )
    num_groups = in_features // group_size
    orig_dtype = weight.dtype

    w = weight.float().reshape(*lead, out_features, num_groups, group_size)
    group_absmax = w.abs().amax(dim=-1)  # (..., out_features, num_groups)

    # Zero-guard: a group whose weights are all exactly zero would divide by
    # zero below. This is not expected for a trained expert's weights (see
    # the Experts.gate_up_proj uninitialized-tensor incident this project
    # already hit once -- that produced NaN/garbage, not clean zeros, and
    # was caught then). Guarded anyway because "should not happen" is not
    # "cannot happen", and a silent NaN here would be exactly the kind of
    # invisible-to-cosine-similarity failure this project's bug trail is
    # full of.
    safe_absmax = torch.where(
        group_absmax > 0, group_absmax, torch.ones_like(group_absmax)
    )
    scale = (safe_absmax / 127.0).to(orig_dtype)

    scale_bcast = scale.unsqueeze(-1).float()  # (..., out, num_groups, 1)
    int8_w = torch.round(w / scale_bcast).clamp(-127, 127).to(torch.int8)
    int8_w = int8_w.reshape(*lead, out_features, in_features)

    return int8_w, scale


@torch.compile
def dequantize_weight_int8_grouped(
    int8_weight: torch.Tensor, scale: torch.Tensor, group_size: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """Inverse of quantize_weight_int8_grouped. Returns a tensor of shape
    and dtype matching the original input to quantize_weight_int8_grouped.

    @torch.compile added 2026-08-21, third pass at the same throughput problem, same
    pattern this project already uses for layers/layernorm.py's rms_forward -- letting
    TorchInductor fuse the reshape/multiply/reshape sequence into fewer kernel launches
    than the hand-fused version above achieves manually, and proven compatible with the
    manual outer `torch.cuda.graph()` capture this project uses for decode (rms_forward
    is itself compiled and already runs correctly inside that capture in production).
    NOT free, though -- real, understood risk: torch._dynamo has a default
    cache_size_limit of 8 distinct compiled shapes before silently falling back to eager
    for further shapes (see bench_throughput.py's own module docstring for the earlier
    incident this caused). This function's tensor shapes vary with N (batch size) and
    local_slots count, which vary per captured CUDA-graph bucket (graph_bs = [1,2,4,8,16,
    ...] in engine/model_runner.py's capture_cudagraph()) -- combined with rms_forward's
    own per-shape compiles, this can plausibly exceed the limit before every bucket gets
    compiled, meaning SOME concurrency levels silently get no speedup from this decorator
    while others do. Not a correctness risk (eager fallback is still correct), but a real
    performance-consistency one -- watch for "torch._dynamo hit config.cache_size_limit"
    warnings in the log and consider raising torch._dynamo.config.cache_size_limit if they
    appear for shapes that matter. Verify with the same matched A/B before trusting any
    number this produces -- not assumed safe from the "other functions already do this"
    argument alone.

    Computes directly in out_dtype (bf16 in production), NOT fp32 -- changed
    2026-08-21 after a matched-settings A/B on real hardware measured the
    fp32 intermediate costing ~2.47x decode throughput (52.7 -> 21.3 tok/s,
    concurrency=16, gpu_memory_utilization=0.75, tp=2, real checkpoint,
    identical otherwise), root-caused to this function: int8-read (1x) ->
    fp32-upcast (4x) -> fp32 in-place multiply (4x+4x) -> fp32-read for the
    final downcast (4x) -> bf16-write (2x) is ~19x the traffic of bf16
    directly reading gate_up_proj/down_proj (2x), on a bandwidth-bound
    decode path.

    Second pass, same day: the int8-cast and the scale-multiply are now ONE
    fused elementwise op (`int8_weight * scale_bcast`) instead of two
    separate passes (a prior version here did `.to(out_dtype)` THEN
    `w.mul_(scale_bcast)` in place -- ~7x traffic, a real improvement over
    the fp32 version's ~19x but still two full read+write passes). PyTorch's
    binary elementwise kernels promote mixed dtypes (int8 * bf16 -> bf16)
    INSIDE the kernel -- read int8, cast in-register, multiply, write bf16,
    all in one pass -- as long as int8_weight is never explicitly `.to()`'d
    first; pre-casting forces materialization of a separate intermediate
    tensor and throws that fusion away. Expected ~3x traffic (int8 read 1x +
    out_dtype write 2x) vs. the ~7x two-pass version -- ANOTHER ~2.3x
    reduction on top of the first fix, not yet confirmed on real hardware
    (this is a plausible-from-first-principles claim, not a profiled one --
    verify with the same matched A/B before trusting the number).

    This does NOT increase peak memory vs. the two-pass version: both
    allocate exactly one out_dtype-sized result tensor (`w * scale_bcast`
    allocates one new tensor; the prior two-pass version's `.to(out_dtype)`
    also allocated one, then mutated it in place). The earlier docstring
    here reasoned that avoiding a second allocation required the in-place
    form -- that reasoning was wrong; there is no second allocation either
    way, only a difference in kernel-launch count.

    Precision trade-off, stated plainly: this rounds the per-group scale to
    out_dtype BEFORE the multiply, instead of multiplying in fp32 and
    rounding only the final product (multiply-then-round beats
    round-then-multiply for accuracy, in general). The int8 values
    themselves are unaffected -- integers in [-127, 127] cast exactly to
    bf16, which represents integers exactly up to 256. Re-validated against
    the same reconstruction/matmul-error checks this function was already
    measured against (Q5) before trusting this on GPU -- see
    tests/test_moe_int8_quantize.py's printed before/after numbers, not
    assumed safe from the traffic argument alone."""
    *lead, out_features, in_features = int8_weight.shape
    num_groups = in_features // group_size
    assert scale.shape == (*lead, out_features, num_groups), (
        f"scale shape {tuple(scale.shape)} does not match expected "
        f"{(*lead, out_features, num_groups)} for int8_weight shape "
        f"{tuple(int8_weight.shape)} at group_size={group_size}."
    )

    w = int8_weight.reshape(*lead, out_features, num_groups, group_size)
    scale_bcast = scale.unsqueeze(-1).to(out_dtype)
    dequant = (w * scale_bcast).reshape(*lead, out_features, in_features)
    return dequant


def quantize_experts_module(gate_up_proj: torch.Tensor, down_proj: torch.Tensor, group_size: int):
    """Convenience wrapper for the two tensors an Experts module actually
    holds (models/qwen3_5.py). Returns a plain dict, not a module -- module
    construction and integration into the forward path is Q4's concern, not
    this file's."""
    gu_int8, gu_scale = quantize_weight_int8_grouped(gate_up_proj, group_size)
    dp_int8, dp_scale = quantize_weight_int8_grouped(down_proj, group_size)
    return {
        "gate_up_proj_int8": gu_int8,
        "gate_up_proj_scale": gu_scale,
        "down_proj_int8": dp_int8,
        "down_proj_scale": dp_scale,
        "group_size": group_size,
        "orig_dtype": str(gate_up_proj.dtype),
    }