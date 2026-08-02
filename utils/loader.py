import os
from glob import glob
import torch
import torch.distributed as dist
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


def expert_local_slot(global_expert_id: int, rank: int, tp_size: int) -> int | None:
    """Round-robin expert-parallel assignment: expert e belongs to rank e % tp_size,
    at local slot e // tp_size within that rank's (num_experts // tp_size, ...) shard.

    Returns None if global_expert_id doesn't belong to this rank.
    """
    if global_expert_id % tp_size != rank:
        return None
    return global_expert_id // tp_size


def shard_experts_tensor(full_tensor: torch.Tensor, rank: int, tp_size: int) -> torch.Tensor:
    """Slice a full (num_experts, ...) batched Experts tensor (gate_up_proj or
    down_proj, as read whole from a single checkpoint key -- real checkpoints
    store all experts for a layer as one batched tensor, not one key per
    expert) down to this rank's local (num_experts // tp_size, ...) shard,
    using round-robin assignment (expert_local_slot).

    Round-robin means each rank's owned experts are non-contiguous (e.g. rank 0
    owns {0, 2, 4, ...}), so unlike contiguous sharding this can't be read as a
    single contiguous slice straight from the safetensors file -- the full
    tensor must be materialized in memory first, then indexed. At real
    256-expert scale this is the same class of memory pressure as tonight's
    GPU OOM, one level down the stack (CPU RAM instead of GPU VRAM) -- worth a
    real check against available system RAM before this runs against the real
    checkpoint, not discovered by hitting it.
    """
    num_experts = full_tensor.shape[0]
    assert num_experts % tp_size == 0
    local_ids = [e for e in range(num_experts) if e % tp_size == rank]
    # Explicit device, matching full_tensor -- not torch's ambient default
    # device. ModelRunner.__init__ calls torch.set_default_device("cuda")
    # before load_model() runs, so an unpinned torch.tensor(...) here would
    # silently land on CUDA while full_tensor (read via safe_open(..., "cpu"))
    # is on CPU -- index_select then fails on the device mismatch. None of
    # tonight's CPU-only tests exercised this, since none of them ran with
    # that default-device override active.
    idx = torch.tensor(local_ids, dtype=torch.long, device=full_tensor.device)
    return full_tensor.index_select(0, idx)


def _is_experts_weight(param_name: str) -> bool:
    """True for exactly the two parameters Qwen35MoE.__init__ shrinks under
    round-robin EP sharding -- Experts.gate_up_proj and Experts.down_proj.
    Narrow suffix match (not a broad substring) so nothing else can
    accidentally match."""
    return param_name.endswith(".experts.gate_up_proj") or param_name.endswith(".experts.down_proj")


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
                    loaded_weight = f.get_tensor(weight_name)
                    # Detected by shape mismatch, not a config flag that could
                    # drift out of sync with the model's actual ep_size: if
                    # this is one of the two Experts tensors and the
                    # checkpoint's expert count doesn't match the local
                    # parameter's (Qwen35MoE.__init__ already shrank it under
                    # EP), shard via the same round-robin assignment used
                    # everywhere else (utils.loader.expert_local_slot). At
                    # ep_size==1 the shapes always match (Qwen35MoE.__init__
                    # allocates the full count), so this block never runs --
                    # loaded_weight passes through unchanged, byte-identical
                    # to the pre-EP load_model() path.
                    if _is_experts_weight(param_name) and loaded_weight.shape[0] != param.shape[0]:
                        loaded_weight = shard_experts_tensor(
                            loaded_weight, dist.get_rank(), dist.get_world_size()
                        )
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
