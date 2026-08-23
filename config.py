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
        self.hf_config.use_vectorized_moe = self.use_vectorized_moe
        self.hf_config.use_moe_w8a8 = self.use_moe_w8a8
        self.hf_config.moe_w8a8_weight_group_size = self.moe_w8a8_weight_group_size
