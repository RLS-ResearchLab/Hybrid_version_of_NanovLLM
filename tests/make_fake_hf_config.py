"""Generates a fake HF-style checkpoint directory (config.json only, no
weights) so ModelRunner can be constructed against the small Qwen3.5 hybrid
config without a real checkpoint.

Usage:
    python tests/make_fake_hf_config.py
    python tests/test_qwen35_model_runner.py
"""
import json, os

OUT_DIR = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
os.makedirs(OUT_DIR, exist_ok=True)

config = {
    "architectures": ["Qwen35MoEForCausalLM"],
    "model_type": "qwen3_5_moe",
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10_000_000.0,
    "max_position_embeddings": 4096,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "full_attention_interval": 4,
    # Real checkpoint field names (see models/qwen3_5.py's getattr chains in
    # Qwen35DecoderLayer.__init__ -- these are tried first, the old
    # linear_attn_*/conv_kernel_size names are a fallback kept only for
    # configs that bypass AutoConfig entirely, e.g. tests' make_small_config()
    # SimpleNamespace). Going through real AutoConfig.from_pretrained (as
    # ModelRunner does), the real names below are what actually gets read --
    # they must be present or Qwen3_5MoeConfig's own class defaults
    # (16/32/128/4) silently win instead of these values.
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "linear_conv_kernel_dim": 4,
    "intermediate_size": 256,
    "moe_intermediate_size": 256,
    "shared_expert_intermediate_size": 256,
    "num_experts": 32,
    "num_experts_per_tok": 4,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "torch_dtype": "bfloat16",
}

with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"Wrote fake config to {OUT_DIR}/config.json")
print("NOTE: no .safetensors file is written on purpose — the ModelRunner")
print("test harness monkeypatches load_model to a no-op and explicitly")
print("initializes Experts params itself, since Experts has no default init.")