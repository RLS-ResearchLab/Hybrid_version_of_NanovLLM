---
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B
tags:
- qwen3.5
- moe
- text-only
- vllm
---

# Qwen3.5-35B-A3B Text-Only

Text-only weights extracted from [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (VLM, Mixture-of-Experts) for use with vLLM's `Qwen3_5MoeForCausalLM` architecture.

## What this is

Qwen3.5 MoE models are natively multimodal (VLM). Their HuggingFace checkpoints use `Qwen3_5MoeForConditionalGeneration` with weights prefixed as `model.language_model.*`. This repo provides the **language model backbone only**, with:

- `architectures: ["Qwen3_5MoeForCausalLM"]`
- `model_type: "qwen3_5_moe_text"`
- Weight keys at `model.layers.*` (standard causal LM format, no `language_model.` prefix)
- Vision encoder and MTP weights removed

## Model structure

- **Architecture**: Hybrid GatedDeltaNet + Full Attention, Sparse Mixture-of-Experts
- **Total parameters**: ~35B (3B active per token)
- **Dtype**: bfloat16

## How to use with vLLM

```python
from vllm import LLM
llm = LLM(model="codecho/Qwen3.5-35B-A3B-text-only", trust_remote_code=True, tensor_parallel_size=2)
```
