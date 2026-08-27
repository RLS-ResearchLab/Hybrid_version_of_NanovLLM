"""Wiring check for utils/loader.py's EP-aware load_model(): does the real
load_model() function correctly route Experts.gate_up_proj/down_proj through
expert_local_slot/shard_experts_tensor when EP sharding applies, and leave
everything else (including ep_size==1) completely untouched.

Three checks, CPU-only, no GPU, no real checkpoint weights:

  Test A: construct the REAL Qwen35ForCausalLM (real qwen35_checkpoint config
  -- hidden_size=2048, num_hidden_layers=40, num_experts=256,
  num_experts_per_tok=8, ...) on the meta device at ep_size=2. Confirms
  Experts.gate_up_proj/down_proj are (128, ...), matching what
  Qwen35MoE.__init__ already produces (num_experts // ep_size).

  Test B: a small-but-structurally-real model (num_experts=256, top_k=8 --
  the values that matter for testing sharding scale -- but small
  hidden_size/intermediate_size/num_hidden_layers=1, so real tensor data
  stays tractable; matches Checkpoint 2's precedent of using real expert
  count with small per-expert dims). A synthetic checkpoint file, with each
  of the 256 experts' slice tagged with a distinct constant (matching
  Checkpoint 2's tagging pattern), is loaded through the REAL load_model()
  function (not a hand-rolled substitute) at ep_size=2. Confirms rank 0's
  loaded Experts tensor contains exactly experts {0,2,...,254} in the
  correct local slots, rank 1 contains exactly {1,3,...,255} -- bitwise
  positive match plus explicit negative-contamination check, same standard
  as Checkpoint 2, now through the real load_model() end-to-end.

  Test B also re-runs the identical small checkpoint through the real
  load_model() at ep_size==1 (single rank) and confirms the loaded Experts
  tensor is the full, unsharded (256, ...) reference, bitwise -- direct
  proof the new shape-mismatch branch never fires when shapes already match.

Usage: python tests/test_load_model_ep.py
"""
import os
import sys
import tempfile
import types
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    _pkg = types.ModuleType("nanovllm")
    _pkg.__path__ = [ROOT]
    _pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = _pkg

sys.path.insert(0, os.path.dirname(__file__))

CKPT_DIR = os.path.join(ROOT, "qwen35_checkpoint")

# Test B's small-but-real-expert-count config.
B_HIDDEN = 8
B_INTERMEDIATE = 3
B_NUM_EXPERTS = 256
B_TOP_K = 8
LAYER_KEY_PREFIX = "model.language_model.layers.0.mlp.experts"
GATE_UP_TAG_BASE = 1000.0
DOWN_TAG_BASE = 2000.0


def _stub_attention_module():
    """Same rationale as Fix 2's test: layers/attention.py imports triton,
    not installed on this CPU dev box. Attention holds no parameters of its
    own -- safe to stub for a structure/weight-loading check."""
    import torch.nn as nn

    stub = types.ModuleType("nanovllm.layers.attention")

    class Attention(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, *a, **k):
            raise NotImplementedError

    stub.Attention = Attention
    sys.modules["nanovllm.layers.attention"] = stub


def _small_real_expert_count_config():
    """SimpleNamespace fake config: real num_experts/top_k (what matters for
    sharding-scale correctness), small everything else (tractable real
    tensor data), single linear-attention layer (avoids needing the
    triton-backed full-attention path at all)."""
    return SimpleNamespace(
        hidden_size=B_HIDDEN,
        num_hidden_layers=1,
        full_attention_interval=1000,  # > num_hidden_layers -> layer 0 is linear_attention
        linear_attn_kq_heads=2,
        linear_attn_v_heads=4,
        linear_attn_head_dim=2,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
        vocab_size=100,
        moe_intermediate_size=B_INTERMEDIATE,
        shared_expert_intermediate_size=4,
        num_experts=B_NUM_EXPERTS,
        num_experts_per_tok=B_TOP_K,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
    )


def test_a(rank: int):
    """Meta-device, real full config, ep_size=2 -- confirms Experts shapes."""
    from nanovllm.config import Config
    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM

    config = Config(CKPT_DIR)
    # No guardrail here (rule 4, per task instructions): torch.device("meta")
    # construction allocates no real tensor storage -- there is nothing for
    # assert_all_parameters_initialized to check, and calling it would just
    # error on meta tensors, not catch a real bug.
    with torch.device("meta"):
        model = Qwen35ForCausalLM(config.hf_config)

    experts = model.model.layers[0].mlp.experts
    expected_local = config.hf_config.num_experts // 2  # ep_size=2
    print(f"[rank{rank}] Test A: layer0 experts.gate_up_proj shape={tuple(experts.gate_up_proj.shape)} "
          f"down_proj shape={tuple(experts.down_proj.shape)} (expect local dim {expected_local})")
    assert experts.gate_up_proj.shape[0] == expected_local, \
        f"expected {expected_local}, got {experts.gate_up_proj.shape[0]}"
    assert experts.down_proj.shape[0] == expected_local, \
        f"expected {expected_local}, got {experts.down_proj.shape[0]}"
    assert experts.gate_up_proj.shape == (
        expected_local, 2 * config.hf_config.moe_intermediate_size, config.hf_config.hidden_size
    )
    assert experts.down_proj.shape == (
        expected_local, config.hf_config.hidden_size, config.hf_config.moe_intermediate_size
    )
    print(f"[rank{rank}] Test A OK -- shapes match num_experts//ep_size == 256//2 == 128, at real dims")


def _build_reference_and_checkpoint(tmp_path: str):
    gate_up = torch.empty(B_NUM_EXPERTS, 2 * B_INTERMEDIATE, B_HIDDEN, dtype=torch.float32)
    down = torch.empty(B_NUM_EXPERTS, B_HIDDEN, B_INTERMEDIATE, dtype=torch.float32)
    for e in range(B_NUM_EXPERTS):
        gate_up[e] = float(GATE_UP_TAG_BASE + e)
        down[e] = float(DOWN_TAG_BASE + e)
    save_file(
        {f"{LAYER_KEY_PREFIX}.gate_up_proj": gate_up, f"{LAYER_KEY_PREFIX}.down_proj": down},
        tmp_path,
    )
    return gate_up, down


def test_b_ep(rank: int, ckpt_dir: str, ref_gate_up: torch.Tensor, ref_down: torch.Tensor):
    """ep_size=2: real load_model() against the small-but-256-expert checkpoint."""
    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM
    from nanovllm.utils.loader import load_model, shard_experts_tensor

    config = _small_real_expert_count_config()
    model = Qwen35ForCausalLM(config)
    load_model(model, ckpt_dir)

    # NOTE: assert_all_parameters_initialized() is deliberately NOT called --
    # this is an EP-sharding unit test whose fixture contains ONLY the layer-0
    # experts tensors, by design. A full-model init guardrail would fire on
    # embed_tokens / norm / lm_head / linear_attn / layernorms (legitimately
    # absent, not a routing bug). The bitwise POSITIVE + NEGATIVE checks below
    # (own shard exact-match vs shard_experts_tensor oracle; zero trace of the
    # other rank's experts) are the real, stronger coverage.

    loaded_gu = model.model.layers[0].mlp.experts.gate_up_proj.data
    loaded_dn = model.model.layers[0].mlp.experts.down_proj.data
    print(f"[rank{rank}] Test B (ep_size=2): loaded gate_up_proj shape={tuple(loaded_gu.shape)} "
          f"values-per-slot(sample)={[loaded_gu[i, 0, 0].item() for i in range(min(4, loaded_gu.shape[0]))]}...")

    expected_gu = shard_experts_tensor(ref_gate_up, rank, 2)
    expected_dn = shard_experts_tensor(ref_down, rank, 2)
    assert torch.equal(loaded_gu, expected_gu), \
        f"[rank{rank}] gate_up_proj MISMATCH vs shard_experts_tensor oracle"
    assert torch.equal(loaded_dn, expected_dn), \
        f"[rank{rank}] down_proj MISMATCH vs shard_experts_tensor oracle"
    print(f"[rank{rank}] Test B POSITIVE check OK -- exact bitwise match vs shard_experts_tensor oracle "
          f"(experts {{{rank},{rank+2},{rank+4},...}} in local slots 0..127)")

    forbidden_ranks = [r for r in range(2) if r != rank]
    for other_rank in forbidden_ranks:
        forbidden_experts = list(range(other_rank, B_NUM_EXPERTS, 2))
        for e in forbidden_experts:
            gu_val = float(GATE_UP_TAG_BASE + e)
            dn_val = float(DOWN_TAG_BASE + e)
            assert not (loaded_gu == gu_val).any(), \
                f"[rank{rank}] gate_up_proj contains expert {e}'s data (owned by rank {other_rank})"
            assert not (loaded_dn == dn_val).any(), \
                f"[rank{rank}] down_proj contains expert {e}'s data (owned by rank {other_rank})"
    print(f"[rank{rank}] Test B NEGATIVE check OK -- no trace of the other rank's "
          f"{len(forbidden_experts)} experts' data anywhere")


def worker(rank: int, ckpt_dir: str, ref_gate_up: torch.Tensor, ref_down: torch.Tensor):
    dist.init_process_group("gloo", init_method="tcp://localhost:29701", rank=rank, world_size=2)
    _stub_attention_module()
    test_a(rank)
    test_b_ep(rank, ckpt_dir, ref_gate_up, ref_down)
    dist.destroy_process_group()


def test_b_ep1(ckpt_dir: str, ref_gate_up: torch.Tensor, ref_down: torch.Tensor):
    """ep_size=1 (single process): real load_model() must load the FULL,
    unsharded (256, ...) tensor -- direct proof the new shape-mismatch
    branch never fires when shapes already match."""
    dist.init_process_group("gloo", init_method="tcp://localhost:29702", rank=0, world_size=1)

    from nanovllm.models.qwen3_5 import Qwen35ForCausalLM
    from nanovllm.utils.loader import load_model

    config = _small_real_expert_count_config()
    model = Qwen35ForCausalLM(config)
    load_model(model, ckpt_dir)

    # (No full-model init guardrail -- experts-only fixture by design; see the
    # ep_size=2 worker above. The bitwise full-tensor equality checks below
    # are the real coverage.)

    loaded_gu = model.model.layers[0].mlp.experts.gate_up_proj.data
    loaded_dn = model.model.layers[0].mlp.experts.down_proj.data
    print(f"[ep_size=1] loaded gate_up_proj shape={tuple(loaded_gu.shape)} (expect full {B_NUM_EXPERTS})")
    assert loaded_gu.shape[0] == B_NUM_EXPERTS
    assert torch.equal(loaded_gu, ref_gate_up), "[ep_size=1] gate_up_proj MISMATCH vs full unsharded reference"
    assert torch.equal(loaded_dn, ref_down), "[ep_size=1] down_proj MISMATCH vs full unsharded reference"
    print(f"[ep_size=1] OK -- full unsharded ({B_NUM_EXPERTS}, ...) tensor loaded bitwise-exact, "
          f"new EP branch did not fire")

    dist.destroy_process_group()


def main():
    _stub_attention_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.safetensors")
        ref_gate_up, ref_down = _build_reference_and_checkpoint(ckpt_path)

        # ep_size=1 first, single process, in this same process.
        test_b_ep1(tmpdir, ref_gate_up, ref_down)

        # ep_size=2, real 2-process spawn.
        mp.spawn(worker, args=(tmpdir, ref_gate_up, ref_down), nprocs=2, join=True)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
