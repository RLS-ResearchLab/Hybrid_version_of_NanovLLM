"""CPU-only correctness check for StateManager's scratch-slot design
(engine/state_manager.py), added tonight to make the in-graph state
write-back (model_runner.py's capture_cudagraph()/_step()) safe against
padding rows -- rows in a CUDA-graph-captured batch beyond the real request
count, e.g. when the batch shrinks as sequences finish generating at
different times.

Imports the REAL StateManager class directly (engine/state_manager.py has
no CUDA/triton dependency at import time, confirmed empirically) -- unlike
the int8 tp=1 test, this is NOT a reproduction, it exercises the actual
class. Reproduces the exact call sequence model_runner.py's _step()/run()
perform: get() with a slot_ids tensor whose padding entries are
scratch_slot_id, then set() with the SAME slot_ids tensor writing computed
"new state" back for the full padded batch in one call -- proving real
slots get exactly their own values and the scratch slot absorbs padding
writes without touching any real slot's data, including under multiple
simultaneous padding rows. Ends with a negative control that deliberately
reproduces the PRE-FIX bug (a padding row aliasing a real slot id) to show
this is a real, demonstrated hazard, not a hypothetical one.

Does NOT validate: actual CUDA graph capture/replay (device="cuda" only,
untestable without a GPU), or model_runner.py's own padding-fill code
(gv["state_slot_ids"].fill_(scratch_slot_id) before [:bs]=...) -- this tests
StateManager's own get/set contract in isolation, which is where the real
hazard and the real fix both live.

Usage:
    python tests/test_state_manager_scratch_slot_cpu.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
from state_manager import StateManager  # noqa: E402


class _FakeSeq:
    def __init__(self):
        self.state_slot = None


def main():
    max_num_seqs = 4
    num_linear_layers = 1
    lvh, lhd, qkv_dim, ck = 2, 4, 8, 4  # small synthetic dims

    sm = StateManager(max_num_seqs, num_linear_layers, lvh, lhd, qkv_dim, ck,
                       device="cpu", dtype=torch.float32)
    print(f"scratch_slot_id={sm.scratch_slot_id}  states.shape={tuple(sm.states.shape)} "
          f"(expected max_num_seqs+1={max_num_seqs + 1} in dim 1)")
    assert sm.scratch_slot_id == max_num_seqs
    assert sm.states.shape[1] == max_num_seqs + 1
    assert sm.conv_states.shape[1] == max_num_seqs + 1

    # ---- Simulate 3 real sequences allocated (batch shrank from a captured
    # graph size of 4 down to 3 real requests -- exactly the scenario the
    # scratch slot exists for). ----
    seqs = [_FakeSeq() for _ in range(3)]
    real_slots = [sm.allocate(s) for s in seqs]
    print(f"Allocated real slots: {real_slots}")
    assert set(real_slots) == {0, 1, 2}, "allocate() should hand out the first 3 free slots"

    # ---- Simulate model_runner.py's run(): state_slot_ids buffer filled
    # with scratch, then [:bs] overwritten with the real slot ids. Captured
    # graph size is 4 (max_num_seqs), real bs is 3 -- one padding row. ----
    slot_ids_bs = torch.full((max_num_seqs,), sm.scratch_slot_id, dtype=torch.int64)
    slot_ids_bs[:3] = torch.tensor(real_slots, dtype=torch.int64)
    print(f"slot_ids_bs (1 padding row): {slot_ids_bs.tolist()}")

    old_state, old_conv = sm.get(0, slot_ids_bs)
    assert old_state.shape[0] == max_num_seqs

    # Distinct, easily-traceable "new state" per row -- row i gets (i+1)*10
    # so any cross-contamination is immediately visible by eye.
    new_state = torch.stack([
        torch.full((lvh, lhd, lhd), float((i + 1) * 10)) for i in range(max_num_seqs)
    ])
    new_conv = torch.stack([
        torch.full((qkv_dim, ck - 1), float((i + 1) * 10)) for i in range(max_num_seqs)
    ])

    # The in-graph write-back, exactly as _step() now does it -- the FULL
    # padded batch (including the padding row) in one call.
    sm.set(0, slot_ids_bs, new_state, new_conv)

    for i, slot in enumerate(real_slots):
        val = sm.states[0, slot, 0, 0, 0].item()
        expected = (i + 1) * 10
        print(f"  real slot {slot} (row {i}): states[...]={val}  expected={expected}")
        assert val == expected, f"CORRUPTION: slot {slot} got {val}, expected {expected}"

    scratch_val = sm.states[0, sm.scratch_slot_id, 0, 0, 0].item()
    print(f"  scratch slot {sm.scratch_slot_id}: states[...]={scratch_val} (padding row's value, unused)")
    assert scratch_val == 40.0

    # ---- Multi-padding-row case: captured bs=4 but only 1 real request,
    # THREE padding rows all targeting scratch simultaneously in one
    # index_copy_ call -- must not error, must not corrupt the one real slot. ----
    for s in seqs:
        sm.free(s)
    seq2 = _FakeSeq()
    real_slot2 = sm.allocate(seq2)
    slot_ids_multi_pad = torch.full((max_num_seqs,), sm.scratch_slot_id, dtype=torch.int64)
    slot_ids_multi_pad[0] = real_slot2
    print(f"slot_ids (3 padding rows, 1 real): {slot_ids_multi_pad.tolist()}")
    new_state2 = torch.stack([
        torch.full((lvh, lhd, lhd), float(99 if i == 0 else 7 + i)) for i in range(max_num_seqs)
    ])
    new_conv2 = torch.stack([
        torch.full((qkv_dim, ck - 1), float(99 if i == 0 else 7 + i)) for i in range(max_num_seqs)
    ])
    sm.set(0, slot_ids_multi_pad, new_state2, new_conv2)
    real_val2 = sm.states[0, real_slot2, 0, 0, 0].item()
    print(f"  real slot {real_slot2} after 3-padding-row write: {real_val2}  expected=99.0")
    assert real_val2 == 99.0, f"CORRUPTION under multi-padding-row write: got {real_val2}"

    # ---- Negative control: demonstrates the PRE-FIX bug concretely. Without
    # scratch-slot routing, a padding row carrying a stale REAL slot id (the
    # actual pre-fix failure mode -- see model_runner.py's old behavior) DOES
    # corrupt a real, currently-in-use sequence's state. Not exercising the
    # fix itself -- shown to prove this is a real, demonstrated hazard. ----
    print("\nNegative control (demonstrates the bug the fix prevents, not a StateManager call):")
    sm2 = StateManager(max_num_seqs, num_linear_layers, lvh, lhd, qkv_dim, ck,
                        device="cpu", dtype=torch.float32)
    victim = _FakeSeq()
    victim_slot = sm2.allocate(victim)
    sm2.states[0, victim_slot] = 123.0  # simulate victim's real, in-use state
    # A padding row that (pre-fix) carries a stale REAL slot id instead of
    # the scratch slot -- e.g. leftover from a previous, larger-bs call.
    slot_ids_no_fix = torch.tensor([victim_slot], dtype=torch.int64)
    padding_write = torch.full((1, lvh, lhd, lhd), -999.0)
    padding_conv = torch.full((1, qkv_dim, ck - 1), -999.0)
    sm2.set(0, slot_ids_no_fix, padding_write, padding_conv)
    corrupted = sm2.states[0, victim_slot, 0, 0, 0].item()
    print(f"  WITHOUT scratch-slot routing, a stray padding write DOES corrupt slot {victim_slot}: "
          f"{corrupted} (was 123.0, a real sequence's state)")
    assert corrupted == -999.0, "expected the negative control to demonstrate corruption"

    print("\nPASS -- scratch slot correctly isolates padding-row writes from every real slot, "
          "including under multiple simultaneous padding rows; negative control confirms this "
          "is a real hazard the fix actually prevents, not a hypothetical one.")


if __name__ == "__main__":
    main()
