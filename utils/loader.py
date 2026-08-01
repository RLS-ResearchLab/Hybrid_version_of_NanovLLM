import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# Real checkpoints (e.g. Qwen3.5-MoE VLM) prefix language-model weights with
# "model.language_model." instead of the bare "model." that Qwen35ForCausalLM's
# module tree uses (self.model = Qwen35Model(...), no "language_model" submodule).
# lm_head.* already matches as-is. Translate the checkpoint's prefix into the
# internal parameter namespace before any packed_modules_mapping matching happens.
_CHECKPOINT_LM_PREFIX = "model.language_model."
_INTERNAL_LM_PREFIX = "model."
_LM_HEAD_PREFIX = "lm_head."


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def translate_weight_name(weight_name: str) -> str | None:
    """Map a raw checkpoint key to the internal parameter-name namespace.

    Returns None if weight_name isn't under one of the text-model prefixes
    (e.g. a vision-tower or MTP-head key from a VLM checkpoint).
    """
    if weight_name.startswith(_CHECKPOINT_LM_PREFIX):
        return _INTERNAL_LM_PREFIX + weight_name[len(_CHECKPOINT_LM_PREFIX):]
    if weight_name.startswith(_LM_HEAD_PREFIX):
        return weight_name
    return None


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param_name = translate_weight_name(weight_name)
                if param_name is None:
                    continue
                for k in packed_modules_mapping:
                    if k in param_name:
                        v, shard_id = packed_modules_mapping[k]
                        mapped_name = param_name.replace(k, v)
                        param = model.get_parameter(mapped_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
