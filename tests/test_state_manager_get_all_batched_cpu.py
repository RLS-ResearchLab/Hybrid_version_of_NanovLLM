"""CPU-only equivalence check for StateManager.get_all()'s 2026-08-28 rewrite.

engine/state_manager.py's get_all() used to do ONE index_select PER linear
layer (30 kernel launches/step on the real checkpoint) to gather each
sequence's recurrent/conv state by slot id. This runs INSIDE the captured
CUDA graph every decode step (model_runner.py's capture_cudagraph()'s
_step()) -- nsys profiling (SESSION_HANDOFF_2026-08-28.md) flagged state I/O
as a real, measurable per-step cost. Rewritten to do ONE batched
index_select across all layers at once (same total bytes read, no extra
copy introduced -- see the method's own docstring for why the write side
was deliberately left alone).

This test proves the rewrite is bitwise-identical to the original
per-layer-loop implementation (reproduced here standalone, not by reverting
the source) across several slot_id patterns, including the padding/scratch-
slot pattern model_runner.py actually uses (real slots first, scratch slot
repeated for padding rows) and a partial/scrambled ordering.

No CUDA/triton dependency -- engine/state_manager.py imports cleanly on any
machine. device="cpu" throughout.

Usage:
    python tests/test_state_manager_get_all_batched_cpu.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
from state_manager import StateManager  # noqa: E402


def _reference_get_all(sm: StateManager, slot_ids, num_total_layers, linear_layer_indices):
    """The ORIGINAL per-layer-loop implementation, reproduced here (not
    imported) so this test still catches a regression even if someone edits
    get_all() again later without touching this file."""
    states = [None] * num_total_layers
    conv_states = [None] * num_total_layers
    for compact_idx, full_idx in enumerate(linear_layer_indices):
        states[full_idx] = sm.states[compact_idx].index_select(0, slot_ids)
        conv_states[full_idx] = sm.conv_states[compact_idx].index_select(0, slot_ids)
    return states, conv_states


def main():
    torch.manual_seed(0)
    num_total_layers = 8
    linear_layer_indices = [0, 1, 2, 4, 5, 6]   # 2 "full-attention" layers interspersed (3, 7)
    num_linear_layers = len(linear_layer_indices)
    max_num_seqs = 10
    lvh, lhd, qkv_dim, ck = 4, 6, 20, 4

    sm = StateManager(
        max_num_seqs=max_num_seqs, num_linear_layers=num_linear_layers,
        lvh=lvh, lhd=lhd, qkv_dim=qkv_dim, conv_kernel_size=ck,
        device="cpu", dtype=torch.bfloat16,
    )
    # Fill with distinguishable, non-zero data (real StateManager starts
    # zeroed; this simulates several steps of real occupancy).
    sm.states.copy_(torch.randn_like(sm.states))
    sm.conv_states.copy_(torch.randn(sm.conv_states.shape).to(sm.conv_states.dtype))

    scratch = sm.scratch_slot_id
    patterns = {
        "sequential":      torch.arange(0, 6, dtype=torch.int64),
        "scrambled":       torch.tensor([7, 2, 9, 0, 4], dtype=torch.int64),
        "with_padding":    torch.tensor([3, 1, scratch, scratch, scratch], dtype=torch.int64),
        "single":          torch.tensor([5], dtype=torch.int64),
        "all_scratch":     torch.tensor([scratch, scratch, scratch], dtype=torch.int64),
    }

    ok = True
    print("=" * 70)
    print("StateManager.get_all() batched rewrite -- equivalence check")
    print("=" * 70)
    for name, slot_ids in patterns.items():
        ref_states, ref_conv = _reference_get_all(sm, slot_ids, num_total_layers, linear_layer_indices)
        new_states, new_conv = sm.get_all(slot_ids, num_total_layers, linear_layer_indices)

        states_match = all(
            (a is None and b is None) or torch.equal(a, b)
            for a, b in zip(ref_states, new_states)
        )
        conv_match = all(
            (a is None and b is None) or torch.equal(a, b)
            for a, b in zip(ref_conv, new_conv)
        )
        # None-slots must land on the exact same full-model layer indices
        # both ways (i.e. the full-attention layers stay None).
        none_slots_match = [x is None for x in ref_states] == [x is None for x in new_states]

        match = states_match and conv_match and none_slots_match
        ok &= match
        print(f"{name:14s} (n={slot_ids.numel():2d}): states={states_match}  "
              f"conv={conv_match}  none_slots={none_slots_match}")

    # Contiguity check -- downstream code (models/qwen3_5.py) does plain
    # elementwise ops on these, but confirm the sliced-view result isn't
    # secretly non-contiguous in a way that would silently change strides
    # a caller might rely on.
    states, _ = sm.get_all(torch.arange(0, 4, dtype=torch.int64), num_total_layers, linear_layer_indices)
    contig_ok = all(t is None or t.is_contiguous() for t in states)
    print(f"\ncontiguity of returned per-layer views: {contig_ok}")
    ok &= contig_ok

    print("\n" + "-" * 70)
    print("PASS" if ok else "FAIL")
    print("  Batched get_all() is bitwise-identical to the per-layer-loop original,")
    print("  across sequential/scrambled/padded/single/all-scratch slot patterns.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
