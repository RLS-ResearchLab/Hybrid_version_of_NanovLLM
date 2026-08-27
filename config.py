import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    use_fused_gdr_kernel: bool = False
    # Decode-side counterpart to use_fused_gdr_kernel above (that one is
    # PREFILL-only -- is_decode_shape forces decode onto the sequential
    # per-segment scan regardless of it). This routes the DECODE branch's
    # per-request Python loop through fla.ops.gated_delta_rule.
    # fused_recurrent_gated_delta_rule -- the single-step recurrent
    # counterpart of the chunk kernel prefill uses (models/qwen3_5.py's
    # _forward_decode_fused_gdr). REWRITTEN 2026-08-27: the original wired
    # phantom, never-committed modules (nanovllm.layers.fused_recurrent /
    # gated_delta_net) so this flag was dead; it now calls the real installed
    # fla kernel. STILL GPU-UNVALIDATED -- the open question is whether fla's
    # recurrent kernel is capturable inside torch.cuda.graph() (the chunk
    # kernel is NOT: cudaErrorStreamCaptureInvalidated). Prefer
    # use_batched_gdr_decode below (dependency-free, numerically identical,
    # CPU-verified, graph-safe by construction) as the primary path; try
    # this one on GPU only as an A/B against it -- see THROUGHPUT_PUSH_
    # CHECKLIST.md's D2 decision.
    use_fused_gdr_decode_kernel: bool = False
    # Triton-free batched decode path for Qwen35LinearAttention -- replaces
    # forward()'s per-request `for i in range(num_segments)` Python loop (30
    # of 40 layers, ~30k tiny kernel launches per captured decode step at
    # concurrency 64) with one set of batched tensor ops. Unlike
    # use_fused_gdr_decode_kernel above, this has NO missing dependency and
    # is numerically identical to the sequential scan (same reduction order),
    # CPU-verified (tests/test_qwen35_gdr_decode_batched.py: bitwise@N=1,
    # ~1e-8@N=64, no 16-step drift). Still needs a GPU run to confirm it
    # CUDA-graph-captures and to measure the actual throughput effect. Off by
    # default; the sequential scan stays the ground-truth path.
    use_batched_gdr_decode: bool = False
    use_vectorized_moe: bool = False
    use_moe_w8a8: bool = False
    moe_w8a8_weight_group_size: int = 128
    # True W8A8 Hopper path (moe_w8a8.cu) -- separate scheme from use_moe_w8a8
    # above (2D-blocked FP8, not 1D-grouped INT8), additive not a replacement.
    # UNVALIDATED end-to-end -- see w8a8_activation_quant_scoping_memo.md and
    # models/qwen3_5.py's NANOVLLM_USE_MOE_W8A8_HOPPER import-site comment.
    # Setting this True quantizes weights to FP8 at load time
    # (moe_w8a8_hopper_integration.py); the SEPARATE NANOVLLM_USE_MOE_W8A8_HOPPER
    # env var controls whether the decode forward path actually calls the
    # Hopper kernel with them (mirrors use_moe_w8a8's own
    # NANOVLLM_USE_FUSED_MOE_KERNEL split, same reason: two independent
    # questions -- "are weights quantized" vs. "which kernel reads them").
    # Do not set True before Phase 0 (kernel compile + isolated smoke test)
    # has actually run on real Hopper hardware.
    use_moe_w8a8_hopper: bool = False
    moe_w8a8_hopper_weight_group_size: int = 128
    # lm_head INT8 weight-only quantization -- 2026-08-23, see
    # tests/lm_head_int8_integration.py's module docstring for the real numbers
    # (~485MiB capacity win) and the honest caveat (NOT a confirmed throughput
    # win -- lm_head reads its full weight every call, unlike the gathered MoE
    # experts, so naive dequant-then-matmul is a plausible bandwidth regression
    # until a fused kernel exists). Off by default, independent of use_moe_w8a8.
    use_lm_head_int8: bool = False
    lm_head_int8_group_size: int = 128
    # Debug-only: prints argmax vs sampled token for every prefilled seq on
    # every prefill call. Forces an extra argmax() + host-sync .tolist() in
    # the hot path even when nobody reads the output -- default off so
    # normal runs (including anything being benchmarked) don't pay for it.
    debug_print_prefill_samples: bool = False
    # Debug-only: prints seq_id/token counts/state_slot on every scheduler
    # preemption. Off by default -- under real memory pressure (e.g. a
    # concurrency sweep past KV-cache capacity) preemptions can fire in a
    # tight burst, and this print previously ran unconditionally inside
    # Scheduler.preempt(), which BatchedEngine's background loop calls while
    # holding its request-admission lock -- turning a burst of preemptions
    # into a burst of serialized stdout syscalls at exactly the moment
    # things are already degrading.
    debug_print_preemptions: bool = False
    # Debug-only: gates the EP forward paths' (_forward_dispatch_ep,
    # _forward_gathered_ep, _forward_gathered_ep_w8a8_hopper) independent
    # token-id round-trip check, which costs a real extra dist.all_reduce
    # on every single EP forward call (prefill AND decode) purely to
    # populate self._last_ep_token_id_roundtrip for test verification --
    # never read by the numeric output. Off by default so production/
    # benchmarked runs don't pay for a second collective on top of the
    # functionally-necessary one. tests/moe_ep_dispatch_core.py,
    # tests/test_moe_ep_dispatch_decode.py, and
    # tests/test_moe_ep_dispatch_edge_cases.py construct Qwen35MoE directly
    # with this flag explicitly True, bypassing this Config default.
    debug_ep_token_roundtrip: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        hf_config = AutoConfig.from_pretrained(self.model)
        # VLM checkpoints (e.g. Qwen3.5-MoE) nest language-model fields under
        # `text_config`, and recent transformers versions further group RoPE
        # hyperparameters (rope_theta, partial_rotary_factor, ...) inside a
        # `rope_parameters` dict instead of exposing them as flat attributes.
        # Flatten both onto the top-level object so any field downstream code
        # reads directly (hidden_size, vocab_size, rope_theta, ...) resolves
        # correctly regardless of nesting depth.
        #
        # Only fill gaps (top-level attribute missing or None) -- never
        # overwrite a real top-level value. text_config is itself a full
        # PretrainedConfig with its own base-class defaults (e.g.
        # `architectures=None`, `model_type="qwen3_5_moe_text"`), and a
        # blind overwrite clobbers meaningful top-level fields like
        # `architectures` with those meaningless nested defaults.
        # layer_types is derived (Qwen3_5MoeConfig.__post_init__ computes it
        # from num_hidden_layers + full_attention_interval), not a plain
        # config value. text_config is a full PretrainedConfig even when the
        # checkpoint's config.json is flat with no "text_config" section --
        # it auto-materializes with base-class defaults (num_hidden_layers=40),
        # so its layer_types is derived from THAT default count, not the
        # top-level checkpoint's real num_hidden_layers. Copying it over
        # would silently mismatch _get_layer_types()'s
        # `len(layer_types) == num_layers` assert (or worse, if lengths ever
        # coincided, apply the wrong attention pattern). Skip it here and let
        # _get_layer_types()'s own full_attention_interval fallback
        # recompute it against the resolved top-level num_hidden_layers.
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            for key, value in vars(text_config).items():
                if key == "layer_types":
                    continue
                if getattr(hf_config, key, None) is None:
                    setattr(hf_config, key, value)
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            for key, value in rope_parameters.items():
                if getattr(hf_config, key, None) is None:
                    setattr(hf_config, key, value)
        self.hf_config = hf_config
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # Rides along on hf_config since model_cls() is only ever constructed
        # from hf_config, not this Config object -- see Qwen35DecoderLayer's
        # getattr(config, "use_fused_gdr_kernel", False) read.
        self.hf_config.use_fused_gdr_kernel = self.use_fused_gdr_kernel
        self.hf_config.use_fused_gdr_decode_kernel = self.use_fused_gdr_decode_kernel
        self.hf_config.use_batched_gdr_decode = self.use_batched_gdr_decode
        self.hf_config.use_vectorized_moe = self.use_vectorized_moe
        self.hf_config.debug_ep_token_roundtrip = self.debug_ep_token_roundtrip
        # NOTE: use_moe_w8a8 / moe_w8a8_weight_group_size deliberately do NOT
        # mirror onto hf_config like the two lines above -- unlike
        # use_fused_gdr_kernel/use_vectorized_moe (read via
        # getattr(config, ...) inside model classes constructed from
        # hf_config directly), INT8 quantization is applied as a separate
        # pass in engine/model_runner.py (getattr(config, "use_moe_w8a8",
        # False), reading THIS Config object, not hf_config) after the model
        # is already built, and the MoE forward paths detect int8-ness
        # structurally via hasattr(self.experts, "gate_up_proj_int8"), never
        # via a config flag. A mirrored hf_config.use_moe_w8a8 previously
        # existed here as dead, write-only state (2026-08-26 audit: never
        # read anywhere) and was removed.
