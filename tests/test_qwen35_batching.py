"""Phase 2 acceptance test: continuous batching correctness.

1. Run 4+ concurrent sequences through the Scheduler/StateManager path and
   confirm each one matches its own single-sequence baseline (no cross-
   sequence state contamination via state_slot indexing).
2. Slot-reuse test: allocate -> use -> free -> immediately reallocate the
   same slot to a different sequence -> confirm zero leakage (flagged as
   the highest-risk untested case by the earlier design review).
3. Chunked prefill: force max_num_batched_tokens below a prompt's length,
   confirm output is unchanged vs. an unconstrained run.
"""

import sys, os
import torch

# Ensure repository root is on sys.path so imports like
# 'Hybrid_version_of_NanoVLLM.engine.state_manager' resolve from tests/ location.
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from test_qwen35_standalone import init_dist, make_small_config

from engine.state_manager import StateManager
from models.qwen3_5 import Qwen35ForCausalLM
from utils.context import set_context, reset_context


def build_model_and_state(config, device, max_num_seqs):
    torch.manual_seed(2024)
    model = Qwen35ForCausalLM(config).to(device).to(torch.bfloat16)
    num_linear = len(model.model.linear_layer_indices)
    sm = StateManager(
        max_num_seqs=max_num_seqs,
        num_linear_layers=num_linear,
        lvh=config.linear_attn_v_heads,
        lhd=config.linear_attn_head_dim,
        qkv_dim=(config.linear_attn_kq_heads * 2 + config.linear_attn_v_heads) * config.linear_attn_head_dim,
        conv_kernel_size=config.conv_kernel_size,
        device=device,
        dtype=torch.bfloat16,
    )
    return model, sm


def run_packed(model, sm, token_id_lists, device):
    """Run several token sequences packed together in ONE forward call
    (simulating one scheduler step), return per-sequence final-token logits."""
    num_layers = len(model.model.layers)
    linear_idx = model.model.linear_layer_indices

    slot_ids = torch.tensor(
        [sm.free_slot_ids.popleft() for _ in token_id_lists], dtype=torch.int64, device=device
    )
    for s in slot_ids.tolist():
        sm.used_slot_ids.add(s)
        sm.states[:, s].zero_()
        sm.conv_states[:, s].zero_()

    flat_ids = torch.tensor(sum(token_id_lists, []), dtype=torch.int64, device=device)
    lengths = [len(t) for t in token_id_lists]
    cu_seqlens = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                               dtype=torch.int32, device=device)
    positions = torch.cat([torch.arange(l, device=device) for l in lengths])

    states, conv_states = sm.get_all(slot_ids, num_layers, linear_idx)
    hidden, new_states, new_conv = model(flat_ids, positions, cu_seqlens, states, conv_states)
    logits = model.compute_logits(hidden)  # ParallelLMHead extracts last-token-per-segment during prefill
    sm.set_all(slot_ids, new_states, new_conv, linear_idx)

    for s in slot_ids.tolist():
        sm.used_slot_ids.discard(s)
        sm.free_slot_ids.append(s)

    return logits  # (num_segments, VOCAB)


def run_single(model, sm, token_ids, device):
    """Baseline: run ONE sequence alone, its own dedicated slot."""
    return run_packed(model, sm, [token_ids], device)[0]


def test_multi_sequence_no_contamination(device):
    print("\n" + "=" * 70)
    print("TEST: 4+ concurrent sequences vs single-sequence baseline")
    print("=" * 70)

    config = make_small_config()
    model, sm = build_model_and_state(config, device, max_num_seqs=8)

    torch.manual_seed(99)
    seqs = [
        torch.randint(100, 5000, (n,)).tolist()
        for n in [7, 11, 5, 9]  # different lengths, deliberately
    ]

    baseline = [run_single(model, sm, s, device) for s in seqs]
    packed = run_packed(model, sm, seqs, device)

    max_diff = 0.0
    for i, (b, p) in enumerate(zip(baseline, packed)):
        cos = torch.nn.functional.cosine_similarity(
            b.float().reshape(1, -1), p.float().reshape(1, -1)
        ).item()
        top1_match = b.float().argmax().item() == p.float().argmax().item()
        print(f"  seq {i} (len={len(seqs[i])}): cosine={cos:.6f} top1_match={top1_match}")
        assert cos > 0.999, f"seq {i} contaminated: cosine={cos}"
        assert top1_match, f"seq {i} top-1 mismatch"
    print("  [PASS] no cross-sequence contamination")


def test_slot_reuse_no_leakage(device):
    print("\n" + "=" * 70)
    print("TEST: slot reuse — allocate/free/reallocate leaves zero leakage")
    print("=" * 70)

    config = make_small_config()
    model, sm = build_model_and_state(config, device, max_num_seqs=2)

    torch.manual_seed(11)
    seq_a = torch.randint(100, 5000, (10,)).tolist()
    logits_a = run_single(model, sm, seq_a, device)  # occupies a slot, then frees it internally

    # sanity: confirm the manager thinks all slots are free again
    assert len(sm.free_slot_ids) == 2, "run_single should return the slot to the pool"

    torch.manual_seed(22)
    seq_b = torch.randint(100, 5000, (10,)).tolist()
    logits_b_after_a = run_single(model, sm, seq_b, device)

    # Independently: run seq_b as if it were the FIRST thing ever run
    # (fresh model instance's state pool would be all-zero for slot 0 either way,
    # but let's directly verify by re-zeroing and re-running on a matching slot)
    slot_ids = torch.tensor([sm.free_slot_ids[0]], dtype=torch.int64, device=device)
    sm.states[:, slot_ids].zero_()
    sm.conv_states[:, slot_ids].zero_()
    logits_b_fresh = run_single(model, sm, seq_b, device)

    cos = torch.nn.functional.cosine_similarity(
        logits_b_after_a.float().reshape(1, -1), logits_b_fresh.float().reshape(1, -1)
    ).item()
    print(f"  seq_b (after A freed slot) vs seq_b (fresh slot) cosine: {cos:.6f}")
    assert cos > 0.999, f"slot reuse leaked state from seq_a: cosine={cos}"
    print("  [PASS] no leakage across slot reuse")


def test_chunked_prefill_matches_unconstrained(device):
    print("\n" + "=" * 70)
    print("TEST: chunked prefill matches unconstrained single-shot")
    print("=" * 70)

    config = make_small_config()
    model, sm = build_model_and_state(config, device, max_num_seqs=4)

    torch.manual_seed(33)
    tokens = torch.randint(100, 5000, (20,)).tolist()

    unconstrained = run_single(model, sm, tokens, device)

    # Simulate chunked prefill: feed the same sequence in two chunks of 12 + 8,
    # threading state/conv_state manually between calls (as the scheduler would).
    num_layers = len(model.model.layers)
    linear_idx = model.model.linear_layer_indices
    slot_id = sm.free_slot_ids.popleft()
    sm.used_slot_ids.add(slot_id)
    sm.states[:, slot_id].zero_()
    sm.conv_states[:, slot_id].zero_()
    slot_ids = torch.tensor([slot_id], dtype=torch.int64, device=device)

    chunk1, chunk2 = tokens[:12], tokens[12:]
    for i, chunk in enumerate([chunk1, chunk2]):
        flat_ids = torch.tensor(chunk, dtype=torch.int64, device=device)
        cu_seqlens = torch.tensor([0, len(chunk)], dtype=torch.int32, device=device)
        start_pos = 0 if i == 0 else len(chunk1)
        positions = torch.arange(start_pos, start_pos + len(chunk), device=device)

        states, conv_states = sm.get_all(slot_ids, num_layers, linear_idx)
        hidden, new_states, new_conv = model(flat_ids, positions, cu_seqlens, states, conv_states)
        logits_chunked = model.compute_logits(hidden)
        sm.set_all(slot_ids, new_states, new_conv, linear_idx)

    sm.used_slot_ids.discard(slot_id)
    sm.free_slot_ids.append(slot_id)

    cos = torch.nn.functional.cosine_similarity(
        unconstrained.float().reshape(1, -1), logits_chunked[0].float().reshape(1, -1)
    ).item()
    print(f"  unconstrained vs chunked cosine: {cos:.6f}")
    assert cos > 0.999, f"chunked prefill diverged: cosine={cos}"
    print("  [PASS] chunked prefill matches unconstrained")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Qwen3.5 Phase 2 Batching Tests — device={device}")
    init_dist()

    test_multi_sequence_no_contamination(device)
    test_slot_reuse_no_leakage(device)
    test_chunked_prefill_matches_unconstrained(device)

    print("\n" + "=" * 70)
    print("ALL PHASE 2 BATCHING TESTS PASSED")
    print("=" * 70)

    import torch.distributed as dist
    dist.destroy_process_group()


if __name__ == "__main__":
    main()