import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context

# dequantize_weight_int8_grouped lives under tests/ (moe_int8_quantize.py), not on
# any production sys.path -- same footgun models/qwen3_5.py already guards against
# explicitly (see that file's own comment on this exact import). Mirrored here
# rather than assumed already-on-path, since this module can be imported without
# models/qwen3_5.py ever having run first.
_TESTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from moe_int8_quantize import dequantize_weight_int8_grouped


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        if hasattr(self, "weight_int8"):
            # lm_head_int8_integration.py's quantized path. CAPACITY win
            # (~485MiB less resident VRAM at real checkpoint dims), NOT a
            # confirmed throughput win -- this dequantizes the FULL weight
            # matrix fresh every call (unlike the MoE experts' gathered
            # subset), which is a plausible bandwidth REGRESSION until a
            # fused kernel exists. See that file's module docstring before
            # treating use_lm_head_int8=True as a speed flag.
            weight = dequantize_weight_int8_grouped(
                self.weight_int8, self.weight_scale, self.lm_head_int8_group_size, x.dtype
            )
        else:
            weight = self.weight
        logits = F.linear(x, weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
