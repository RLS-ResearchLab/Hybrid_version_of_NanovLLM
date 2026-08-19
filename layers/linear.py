import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def local_num_kv_heads(total_num_kv_heads: int, tp_size: int) -> int:
    """Per-rank kv-head count for GQA-aware tensor parallelism. Single
    source of truth for this computation -- used by both
    models/qwen3_5.py's Qwen35FullAttention (module construction) and
    engine/model_runner.py's allocate_kv_cache (KV-cache tensor sizing), so
    the two can't independently drift the way in_proj_qkv/conv1d once did
    (see Qwen35LinearAttention's __init__ comments for that incident).

    Two regimes:
      - total_num_kv_heads >= tp_size: shard normally, every rank owns a
        distinct, evenly-divided slice (unchanged behavior from before this
        function existed).
      - total_num_kv_heads < tp_size: too few physical kv heads to give
        every rank a whole one -- replicate a full kv head onto multiple
        ranks instead of splitting one across them (splitting would hand a
        rank a fraction of a head's channels, which is not a meaningful
        attention head). Requires tp_size % total_num_kv_heads == 0 (e.g. 2
        kv heads over 4 ranks: each head replicated to 2 ranks) -- a
        combination with no clean mapping (e.g. 3 kv heads over 4 ranks)
        raises rather than silently producing a wrong shard.

    Returns the per-rank kv-head count -- always >= 1, unlike the plain
    `total_num_kv_heads // tp_size` this replaces, which silently returned 0
    whenever tp_size > total_num_kv_heads (a zero-sized KV-cache dimension
    at the model_runner.py call site, with nothing to catch it)."""
    if total_num_kv_heads >= tp_size:
        assert total_num_kv_heads % tp_size == 0, (
            f"num_key_value_heads={total_num_kv_heads} is not evenly divisible by "
            f"tensor_parallel_size={tp_size}, and is not smaller than it either -- "
            f"no clean sharding OR replication mapping exists for this combination."
        )
        return total_num_kv_heads // tp_size
    assert tp_size % total_num_kv_heads == 0, (
        f"tensor_parallel_size={tp_size} is not evenly divisible by "
        f"num_key_value_heads={total_num_kv_heads} -- replicating a kv head across a "
        f"fractional number of ranks has no clean mapping (e.g. num_key_value_heads=3 "
        f"at tensor_parallel_size=4 would need 1.33 ranks per head). Pick a "
        f"tensor_parallel_size that is a multiple of num_key_value_heads, or a "
        f"divisor of it."
    )
    return 1


def kv_head_replica_source(total_num_kv_heads: int, tp_size: int, tp_rank: int) -> int:
    """Which of the total_num_kv_heads physical kv heads this rank replicates,
    only meaningful when total_num_kv_heads < tp_size (see local_num_kv_heads
    above -- at total_num_kv_heads >= tp_size, shard normally instead, there
    is nothing to replicate). Groups of (tp_size // total_num_kv_heads)
    consecutive ranks share the same source head -- this lines up exactly
    with HF's own contiguous query-head-to-kv-head grouping (kv head h
    serves query heads [h * group_size : (h+1) * group_size]) and
    ColumnParallelLinear's contiguous per-rank query-head slice
    ([r * per_rank : (r+1) * per_rank]), since tp_size % total_num_kv_heads
    == 0 is already required by local_num_kv_heads before this is called."""
    assert total_num_kv_heads < tp_size, (
        f"kv_head_replica_source is only meaningful in the replication regime "
        f"(total_num_kv_heads={total_num_kv_heads} < tp_size={tp_size}); at "
        f"total_num_kv_heads >= tp_size, shard normally instead -- every rank "
        f"already owns a distinct slice, there is nothing to replicate."
    )
    replicas_per_head = tp_size // total_num_kv_heads
    return tp_rank // replicas_per_head


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
