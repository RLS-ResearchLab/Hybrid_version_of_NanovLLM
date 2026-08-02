"""Q4 / Checkpoint 2: round-robin expert-parallel loader correctness.

Small synthetic config: num_experts=4, tp_size=2. Verifies utils/loader.py's
expert_local_slot() + shard_experts_tensor() assign each expert to exactly the
right rank and exactly the right local slot, bitwise, with an explicit
negative check for cross-rank contamination.

Note on "reference weights": Fix 2 (tonight's earlier session) never had real
numeric weight *values* to check against -- the real qwen35_checkpoint only
ever had config.json + model.safetensors.index.json locally (keys/shapes via
a meta-device model), never actual tensor data. So "known reference weights"
here means synthetic-but-uniquely-tagged values (each expert's slice filled
with a distinct constant), saved under the exact real checkpoint key naming
convention validated in Fix 2/3 (model.language_model.layers.N.mlp.experts.*),
not a reuse of literal numbers from Fix 2 -- there were none to reuse.

Usage: python tests/test_expert_round_robin_loader.py
"""
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file, safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import types as _types
if "nanovllm" not in sys.modules:
    _pkg = _types.ModuleType("nanovllm")
    _pkg.__path__ = [ROOT]
    _pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg

from nanovllm.utils.loader import expert_local_slot, shard_experts_tensor

NUM_EXPERTS = 4
TP_SIZE = 2
INTERMEDIATE = 2
HIDDEN = 2
LAYER_KEY_PREFIX = "model.language_model.layers.0.mlp.experts"

GATE_UP_TAG_BASE = 100  # expert e's gate_up_proj slice filled with GATE_UP_TAG_BASE + e
DOWN_TAG_BASE = 200     # expert e's down_proj slice filled with DOWN_TAG_BASE + e


def build_reference_checkpoint(path: str):
    gate_up = torch.empty(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.float32)
    down = torch.empty(NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.float32)
    for e in range(NUM_EXPERTS):
        gate_up[e] = float(GATE_UP_TAG_BASE + e)
        down[e] = float(DOWN_TAG_BASE + e)
    save_file(
        {
            f"{LAYER_KEY_PREFIX}.gate_up_proj": gate_up,
            f"{LAYER_KEY_PREFIX}.down_proj": down,
        },
        path,
    )
    return gate_up, down


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.safetensors")
        ref_gate_up, ref_down = build_reference_checkpoint(ckpt_path)

        # Simulate what load_model() does today: read the full batched tensor
        # for this layer's experts straight off disk (there is only ever one
        # key per layer -- real checkpoints don't store per-expert files).
        with safe_open(ckpt_path, "pt", "cpu") as f:
            full_gate_up = f.get_tensor(f"{LAYER_KEY_PREFIX}.gate_up_proj")
            full_down = f.get_tensor(f"{LAYER_KEY_PREFIX}.down_proj")

        assert torch.equal(full_gate_up, ref_gate_up)
        assert torch.equal(full_down, ref_down)
        print(f"loaded full checkpoint tensors: gate_up_proj {tuple(full_gate_up.shape)}, "
              f"down_proj {tuple(full_down.shape)}")

        # ---- expert_local_slot() direct checks ----
        expected_slot = {
            (0, 0): 0, (0, 1): None,   # expert 0 -> rank 0, local slot 0
            (1, 0): None, (1, 1): 0,   # expert 1 -> rank 1, local slot 0
            (2, 0): 1, (2, 1): None,   # expert 2 -> rank 0, local slot 1
            (3, 0): None, (3, 1): 1,   # expert 3 -> rank 1, local slot 1
        }
        for (expert_id, rank), expected in expected_slot.items():
            got = expert_local_slot(expert_id, rank, TP_SIZE)
            assert got == expected, f"expert_local_slot({expert_id}, {rank}, {TP_SIZE}) = {got}, expected {expected}"
        print("expert_local_slot(): all 8 (expert_id, rank) assignments correct")

        # ---- shard_experts_tensor() per rank ----
        shards = {}
        for rank in range(TP_SIZE):
            gu_shard = shard_experts_tensor(full_gate_up, rank, TP_SIZE)
            dn_shard = shard_experts_tensor(full_down, rank, TP_SIZE)
            shards[rank] = (gu_shard, dn_shard)
            print(f"[rank{rank}] gate_up_proj shard shape={tuple(gu_shard.shape)} "
                  f"values-per-slot={[gu_shard[i, 0, 0].item() for i in range(gu_shard.shape[0])]}")
            print(f"[rank{rank}] down_proj shard shape={tuple(dn_shard.shape)} "
                  f"values-per-slot={[dn_shard[i, 0, 0].item() for i in range(dn_shard.shape[0])]}")

        # rank 0 must contain exactly experts {0, 2} in local slots {0, 1}
        expected_gu_r0 = ref_gate_up[[0, 2]]
        expected_dn_r0 = ref_down[[0, 2]]
        assert torch.equal(shards[0][0], expected_gu_r0), "rank0 gate_up_proj shard mismatch"
        assert torch.equal(shards[0][1], expected_dn_r0), "rank0 down_proj shard mismatch"
        print("[rank0] POSITIVE check OK -- exact bitwise match: experts {0,2} in slots {0,1}")

        # rank 1 must contain exactly experts {1, 3} in local slots {0, 1}
        expected_gu_r1 = ref_gate_up[[1, 3]]
        expected_dn_r1 = ref_down[[1, 3]]
        assert torch.equal(shards[1][0], expected_gu_r1), "rank1 gate_up_proj shard mismatch"
        assert torch.equal(shards[1][1], expected_dn_r1), "rank1 down_proj shard mismatch"
        print("[rank1] POSITIVE check OK -- exact bitwise match: experts {1,3} in slots {0,1}")

        # ---- NEGATIVE checks: no cross-rank contamination anywhere ----
        gu_r0, dn_r0 = shards[0]
        for forbidden_expert in (1, 3):
            forbidden_gu_val = float(GATE_UP_TAG_BASE + forbidden_expert)
            forbidden_dn_val = float(DOWN_TAG_BASE + forbidden_expert)
            assert not (gu_r0 == forbidden_gu_val).any(), \
                f"rank0 gate_up_proj shard contains expert {forbidden_expert}'s data"
            assert not (dn_r0 == forbidden_dn_val).any(), \
                f"rank0 down_proj shard contains expert {forbidden_expert}'s data"
        print("[rank0] NEGATIVE check OK -- no trace of expert 1's or 3's data anywhere")

        gu_r1, dn_r1 = shards[1]
        for forbidden_expert in (0, 2):
            forbidden_gu_val = float(GATE_UP_TAG_BASE + forbidden_expert)
            forbidden_dn_val = float(DOWN_TAG_BASE + forbidden_expert)
            assert not (gu_r1 == forbidden_gu_val).any(), \
                f"rank1 gate_up_proj shard contains expert {forbidden_expert}'s data"
            assert not (dn_r1 == forbidden_dn_val).any(), \
                f"rank1 down_proj shard contains expert {forbidden_expert}'s data"
        print("[rank1] NEGATIVE check OK -- no trace of expert 0's or 2's data anywhere")

        # ---- Round-trip: scatter both ranks' shards back and recover the full tensor exactly ----
        reconstructed_gu = torch.empty_like(ref_gate_up)
        reconstructed_dn = torch.empty_like(ref_down)
        for rank in range(TP_SIZE):
            gu_shard, dn_shard = shards[rank]
            for e in range(NUM_EXPERTS):
                slot = expert_local_slot(e, rank, TP_SIZE)
                if slot is None:
                    continue
                reconstructed_gu[e] = gu_shard[slot]
                reconstructed_dn[e] = dn_shard[slot]
        assert torch.equal(reconstructed_gu, ref_gate_up), "round-trip gate_up_proj mismatch"
        assert torch.equal(reconstructed_dn, ref_down), "round-trip down_proj mismatch"
        print("ROUND-TRIP OK -- scattering both ranks' shards back recovers the exact original tensors")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
