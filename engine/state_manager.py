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
        check_slot_zeroed: bool = False,
    ):
        self.max_num_seqs = max_num_seqs
        self.num_linear_layers = num_linear_layers
        self.ck = conv_kernel_size
        # allocate() zeros a slot's recurrent/conv state unconditionally
        # (correct -- see there). This flag additionally ASSERTS the zero
        # took, via a `.any()` reduction + host sync on ~63 MB per admitted
        # request -- a real cost on the scheduler's admission path (holds
        # BatchedEngine's request-admission lock) for a check that can only
        # fail on a torch/CUDA bug in `.zero_()` itself. Off by default; turn
        # on only for a state-contamination investigation (config.py's
        # debug_check_state_slot_zeroed). The functional zeroing is NOT
        # gated -- only the redundant post-check.
        self.check_slot_zeroed = check_slot_zeroed

        # Slot `max_num_seqs` is a reserved SCRATCH slot, one past the real
        # range and never handed out by allocate() (free_slot_ids only ever
        # contains range(max_num_seqs)). It exists as a safe write target
        # for CUDA-graph-captured decode steps: a captured graph's batch
        # size is fixed, so a real request count smaller than that pads the
        # remainder with whatever slot ids happen to be left in the graph's
        # static buffer. Writing that padding's computed state back
        # in-graph without a dedicated sink could alias and silently
        # corrupt a DIFFERENT, currently-in-use sequence's real state — see
        # model_runner.py's capture_cudagraph()/run() for how padding rows
        # get routed here instead.
        self.scratch_slot_id = max_num_seqs

        # Recurrent state accumulates in float32 (matches the GDR scan's
        # own precision — see src/model.py's comment on why bf16 state
        # compounds rounding error into degenerate repetition loops).
        self.states = torch.zeros(
            num_linear_layers, max_num_seqs + 1, lvh, lhd, lhd,
            dtype=torch.float32, device=device,
        )
        # Conv history stays in model dtype (matches conv1d's input dtype).
        self.conv_states = torch.zeros(
            num_linear_layers, max_num_seqs + 1, qkv_dim, conv_kernel_size - 1,
            dtype=dtype, device=device,
        )

        self.free_slot_ids: deque[int] = deque(range(max_num_seqs))
        self.used_slot_ids: set[int] = set()

    def allocate(self, seq) -> int:
        assert seq.state_slot is None
        slot_id = self.free_slot_ids.popleft()
        self.used_slot_ids.add(slot_id)
        # Zero on allocate: guards a fresh sequence against reading whatever
        # the previous tenant of this slot left behind (or, for a slot never
        # yet handed out, warmup_model()'s residue). Unconditional.
        self.states[:, slot_id].zero_()
        self.conv_states[:, slot_id].zero_()
        if self.check_slot_zeroed:
            # Debug-only -- see __init__. `.any()` + host sync on ~63 MB per
            # admitted request; the zeroing above is the real guarantee, this
            # only catches a `.zero_()` that silently didn't take.
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
        # Single batched index_select across ALL linear layers at once (dim=1
        # selects slots, dim=0 spans layers), instead of one index_select
        # kernel launch per layer. This runs INSIDE the captured CUDA graph
        # every decode step (model_runner.py's capture_cudagraph()'s _step())
        # -- added 2026-08-28 after nsys profiling flagged state I/O as a
        # real per-step cost (~15% of the batched-path step). Reads the exact
        # same bytes as the old per-layer loop (no extra copy introduced --
        # unlike the write side, which would need a torch.stack() of
        # per-layer tensors first since those come from 30 independently-
        # computed forward() outputs, not a shared buffer; that stack would
        # add a full extra copy and isn't clearly a net win without GPU
        # profiling, so set_all()/the mirroring inline loop in _step() is
        # deliberately left untouched here). Slicing the batched result per
        # layer below is a view into already-materialized data, not a copy.
        combined_states = self.states.index_select(1, slot_ids)
        combined_conv = self.conv_states.index_select(1, slot_ids)
        states = [None] * num_total_layers
        conv_states = [None] * num_total_layers
        for compact_idx, full_idx in enumerate(linear_layer_indices):
            states[full_idx] = combined_states[compact_idx]
            conv_states[full_idx] = combined_conv[compact_idx]
        return states, conv_states

    def set_all(self, slot_ids: torch.Tensor, states: list, conv_states: list, linear_layer_indices: list[int]) -> None:
        for compact_idx, full_idx in enumerate(linear_layer_indices):
            assert states[full_idx] is not None, f"missing state for linear layer at full index {full_idx}"
            self.set(compact_idx, slot_ids, states[full_idx], conv_states[full_idx])

    def memory_bytes(self) -> int:
        """Total bytes consumed — feeds Phase 3's memory budget calc."""
        return (self.states.numel() * self.states.element_size()
                + self.conv_states.numel() * self.conv_states.element_size())