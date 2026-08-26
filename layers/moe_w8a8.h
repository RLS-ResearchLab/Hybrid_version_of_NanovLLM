#pragma once

// Shared declaration for moe_w8a8.cu's public entry point, so the .cu
// definition and any caller (moe_w8a8_binding.cpp) are checked against the
// SAME signature by the compiler, instead of the binding re-declaring it by
// hand and silently drifting out of sync with the real definition -- exactly
// the kind of mismatch that's easy to introduce and hard to notice given
// nobody here can compile-test this on the current dev machine.
//
// CONTRACT, read before calling this from anywhere:
//
//   - `out` is written via cp_reduce_async_bulk(..., op_add, ...) inside the
//     kernel -- it ACCUMULATES into `out`, it does not overwrite it. The
//     caller MUST zero `out` before every call, or leftover memory (a
//     previous call's result, or uninitialized allocator garbage) corrupts
//     the output silently. This is the single easiest way to get a
//     plausible-looking wrong answer out of this kernel.
//
//   - `w` (gate_up_proj) is expected as a (K, N*num_experts) row-major FP8
//     tensor with stride K*sizeof(fp8) -- matches the tensor_map_w setup in
//     moe_w8a8.cu's launch function exactly. `w2` (down_proj) is expected as
//     (K2, N2*num_experts) with K2=N/2, N2=K, same convention. Getting this
//     layout wrong does not crash -- it silently reads the wrong bytes.
//
//   - `w`'s N axis (the fused gate_up_proj output, 2*moe_intermediate_size)
//     is NOT the conventional contiguous gate=[0,N/2)/up=[N/2,N) layout.
//     The up-proj's SwiGLU combine pairs wgmma accumulator register c0
//     (physical weight-row r) with c2 (row r+8) as (gate, up) for the same
//     logical down-proj-input feature, so gate and up must be interleaved
//     every 8 physical rows -- a config-dependent permutation (see
//     layers/smoke_test_moe_w8a8_hopper.py's gate_up_interleave_permutation,
//     verified for block_n/warp_n=(32,8) only). Confirmed 2026-08-24: a
//     contiguous-layout weight silently produces plausible-magnitude but
//     wrong (near-zero cosine similarity) output, no crash, no error --
//     REAL checkpoints must have this permutation applied to gate_up_proj
//     before FP8 quantization, or this kernel will be silently wrong once
//     wired into the real model.
//
//   - `w_scale`/`w2_scale` are per-128x128-block scales (block_shape={128,128}
//     is hardcoded in the kernel), consistent with this project's existing
//     group_size=128 INT8 scheme on Ampere -- NOT vLLM's original whole-row
//     scale convention. Worth confirming explicitly once this is compiled
//     and testable, not just inferred from reading the source.
//
//   - `block_n`/`warp_n` (BN/WN in the kernel) must be one of exactly two
//     supported pairs: (32, 8) or (64, 4). Any other pair hits the
//     `dispatch_bn_wn` default branch, prints an error to stderr, and
//     silently does NOT launch the kernel -- `out` stays whatever it was
//     (zeroed, per the contract above), not an error the caller can catch
//     programmatically. Same failure shape for `stages` outside 1-5 and
//     `block_m` outside {8,16,...,128} (their own dispatch defaults).
//
//   - `x`/`w`/`w2` are __nv_fp8_e4m3 (PyTorch: torch.float8_e4m3fn).
//     `x_scale`/`w_scale`/`w2_scale`/`topk_weights`/`scaling_factor` are
//     float32. `out` is __nv_bfloat16 (torch.bfloat16). `sorted_token_ids`/
//     `expert_ids`/`num_tokens_post_padded` are int32 -- matches this
//     project's existing moe_align_block_size.py contract (same three
//     tensors, same dtypes, same semantics), so that piece should be
//     reusable as-is once this is wired up.
//
//   - `x_scale` is per-(token, 128-K-block), shape (M, K/128) -- the SAME
//     128-block convention as w_scale/w2_scale, not a single per-token
//     scalar. Getting this wrong doesn't crash either: for tokens whose
//     stride*token_id lands past the tensor's actual size it reads
//     out-of-bounds global memory silently (confirmed 2026-08-24 -- this was
//     the cause of a ~227,000x output magnitude blowup before being fixed
//     in layers/moe_w8a8_hopper_quantize.py's quantize_activation_fp8_dynamic).
//
//   - Validated in isolation 2026-08-24 (layers/smoke_test_moe_w8a8_hopper.py,
//     cosine_similarity=0.998, compute-sanitizer memcheck clean) for
//     block_n/warp_n=(32,8), M=16, top_k=4, E=8, H=MI=256 -- small synthetic
//     dims, not production scale, and NOT yet run through GSM8K
//     non-regression or wired into the real model. Still worth treating with
//     real skepticism outside that specific validated configuration,
//     especially block_n/warp_n=(64,4) (never run on real hardware) and the
//     gate/up permutation above (only HARDWARE-confirmed for (32,8)).
//
//   - (64,4) specifically: read-through analysis 2026-08-26 (CPU-only, no
//     hardware access) found real, source-level reason to expect the
//     EXISTING gate_up_interleave_permutation to already be correct at
//     (64,4), not needing fresh derivation -- rows_per_tn=64 in that
//     function derives algebraically to 64 for ANY block_n (confirmed
//     against this file's own `sw = ... + tn*64*BK` stride and TN=BN/16
//     loop bound at both dispatched configs), and the fine-grained
//     "interleave every 8 rows" logic comes from wgmma m64nNk32's fixed
//     per-thread accumulator fragment shape (an SM_90 ISA property, not a
//     BN/WN-dependent one) -- both configs read gate/up out of the SAME
//     4-register tile_acc via the identical f_acc[...][0]/[1] split (see
//     the up-proj consumer loop above). Numerically confirmed as a valid,
//     balanced bijection at (64,4) too (tests/test_moe_w8a8_hopper_
//     integration_cpu.py check [7]). NOT proof -- this exact kernel has
//     had equally-plausible-looking wgmma layout theories turn out wrong
//     before (see the postmortem artifact linked in
//     H200_test_day_checklist.md) -- but real reason to try
//     `--block-n 64 --warp-n 4` on the smoke test FIRST on the next Hopper
//     window, before assuming (64,4) needs its own derivation pass.

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using fp8 = __nv_fp8_e4m3;

void fused_moe_w8a8_wgmma_up_down_acc(
        const fp8* x,
        const float* x_scale,
        fp8* w, const float* w_scale,
        fp8* w2, const float* w2_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const float* topk_weights,
        const int top_k,
        int M,
        int K,
        int N,
        int num_experts,
        int sorted_num,
        int block_m,
        int block_n,
        int warp_n,
        int stages,
        float scaling_factor,
        cudaStream_t stream
        );
