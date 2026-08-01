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
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            for key, value in vars(text_config).items():
                setattr(hf_config, key, value)
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            for key, value in rope_parameters.items():
                setattr(hf_config, key, value)
        self.hf_config = hf_config
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
