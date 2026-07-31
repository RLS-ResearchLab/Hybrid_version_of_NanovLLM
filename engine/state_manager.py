from collections import deque
import torch


class StateManager:
    """Fixed-size recurrent state + conv state pool, one slot per
    concurrently-running sequence, for every linear-attention layer.

    Unlike BlockManager's blocks (count scales with sequence length),
    slot count is fixed at max_num_seqs — GDR state is a compressed
    summary, not raw per-token KV, so it doesn't grow with context length.
    """

    def __init__(
        self,
        max_num_seqs: int,
        num_linear_layers: int,
        lvh: int,
        lhd: int,
        qkv_dim: int,
        conv_kernel_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.max_num_seqs = max_num_seqs
        self.num_linear_layers = num_linear_layers
        self.ck = conv_kernel_size

        # Recurrent state accumulates in float32 (matches the GDR scan's
        # own precision — see src/model.py's comment on why bf16 state
        # compounds rounding error into degenerate repetition loops).
        self.states = torch.zeros(
            num_linear_layers, max_num_seqs, lvh, lhd, lhd,
            dtype=torch.float32, device=device,
        )
        # Conv history stays in model dtype (matches conv1d's input dtype).
        self.conv_states = torch.zeros(
            num_linear_layers, max_num_seqs, qkv_dim, conv_kernel_size - 1,
            dtype=dtype, device=device,
        )

        self.free_slot_ids: deque[int] = deque(range(max_num_seqs))
        self.used_slot_ids: set[int] = set()

    def allocate(self, seq) -> int:
        assert seq.state_slot is None
        slot_id = self.free_slot_ids.popleft()
        self.used_slot_ids.add(slot_id)
        # Zero on allocate: guards a fresh sequence against reading
        # whatever the previous tenant of this slot left behind.
        self.states[:, slot_id].zero_()
        self.conv_states[:, slot_id].zero_()
        # Fail loudly rather than silently serving a contaminated slot --
        # a leftover nonzero value here means a new sequence would read
        # a previous tenant's recurrent/conv state with no crash or NaN,
        # just silently wrong output.
        assert not self.states[:, slot_id].any(), (
            f"StateManager slot {slot_id} not fully zeroed after allocate() -- "
            f"recurrent state contamination risk for the incoming sequence"
        )
        assert not self.conv_states[:, slot_id].any(), (
            f"StateManager slot {slot_id} not fully zeroed after allocate() -- "
            f"conv-state contamination risk for the incoming sequence"
        )
        seq.state_slot = slot_id
        return slot_id

    def free(self, seq) -> None:
        slot_id = seq.state_slot
        if slot_id is None:
            return
        assert slot_id in self.used_slot_ids
        self.used_slot_ids.remove(slot_id)
        self.free_slot_ids.append(slot_id)
        self.states[:, slot_id].zero_()
        self.conv_states[:, slot_id].zero_()
        seq.state_slot = None

    def get(self, layer_idx: int, slot_ids: torch.Tensor):
        """(state, conv_state) for one linear layer, one scheduled batch."""
        state = self.states[layer_idx].index_select(0, slot_ids)
        conv_state = self.conv_states[layer_idx].index_select(0, slot_ids)
        return state, conv_state

    def set(self, layer_idx: int, slot_ids: torch.Tensor, state: torch.Tensor, conv_state: torch.Tensor) -> None:
        """Write updated (state, conv_state) back after a forward pass."""
        self.states[layer_idx].index_copy_(0, slot_ids, state.to(self.states.dtype))
        self.conv_states[layer_idx].index_copy_(0, slot_ids, conv_state.to(self.conv_states.dtype))

    def get_all(self, slot_ids: torch.Tensor, num_total_layers: int, linear_layer_indices: list[int]):
        states = [None] * num_total_layers
        conv_states = [None] * num_total_layers
        for compact_idx, full_idx in enumerate(linear_layer_indices):
            states[full_idx] = self.states[compact_idx].index_select(0, slot_ids)
            conv_states[full_idx] = self.conv_states[compact_idx].index_select(0, slot_ids)
        return states, conv_states

    def set_all(self, slot_ids: torch.Tensor, states: list, conv_states: list, linear_layer_indices: list[int]) -> None:
        for compact_idx, full_idx in enumerate(linear_layer_indices):
            assert states[full_idx] is not None, f"missing state for linear layer at full index {full_idx}"
            self.set(compact_idx, slot_ids, states[full_idx], conv_states[full_idx])

    def memory_bytes(self) -> int:
        """Total bytes consumed — feeds Phase 3's memory budget calc."""
        return (self.states.numel() * self.states.element_size()
                + self.conv_states.numel() * self.conv_states.element_size())