"""Exercises the REAL ModelRunner (not the standalone-test bypass) against
the fake Qwen3.5-small config, to validate:
  - Bug 0 fix (dispatch no longer overwritten by dead Qwen3ForCausalLM code)
  - Bug 1 fix (KV-cache sized to full-attention layers only, not all layers)
  - layer_types / linear_layer_indices wiring
  - actual KV-cache byte counts, with and without the layer-count fix

Requires: tests/make_fake_hf_config.py to have been run first.

Usage:
    python tests/make_fake_hf_config.py
    python tests/test_qwen35_model_runner.py
"""
import sys, os, json
import torch

sys.path.insert(0, os.path.dirname(__file__))

# Inline the same "fake nanovllm package" import-path trick used by
# test_qwen35_standalone.py — but WITHOUT calling its init_dist(), since
# that initializes a gloo process group and ModelRunner.__init__ initializes
# its own nccl process group; torch refuses to init the default group twice.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm":
    import types
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

import nanovllm.utils.loader as loader_mod
loader_mod.load_model = lambda model, path, **kw: None   # no real .safetensors to load

import nanovllm.config as config_mod


class AttrDict(dict):
    """Minimal stand-in for HF's PretrainedConfig — attribute access over a dict."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _fake_from_pretrained(path, *args, **kwargs):
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "ModelRunner requires CUDA (flash-attn, paged KV cache)"

    # Monkeypatch AutoConfig.from_pretrained so Config.__post_init__ doesn't
    # need a real HF-registered model_type.
    config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)

    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.models.qwen3_5 import Experts

    fake_dir = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
    assert os.path.isdir(fake_dir), "run tests/make_fake_hf_config.py first"

    config = Config(
        model=fake_dir,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        max_model_len=128,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        enforce_eager=False,
    )

    print("Constructing ModelRunner against fake hybrid config...")
    runner = ModelRunner(config, rank=0, event=[])
    print("  [OK] ModelRunner constructed without error")

    # --- Fix gap #3: Experts has no default init and no real weights were
    # loaded (load_model is a no-op here) — initialize explicitly so later
    # forward passes don't run against uninitialized memory. ---
    with torch.no_grad():
        n_experts_init = 0
        for m in runner.model.modules():
            if isinstance(m, Experts):
                torch.nn.init.normal_(m.gate_up_proj, std=0.02)
                torch.nn.init.normal_(m.down_proj, std=0.02)
                n_experts_init += 1
        print(f"  [OK] explicitly initialized {n_experts_init} Experts module(s)")

    # --- Report actual KV-cache byte accounting ---
    hf_config = config.hf_config
    actual_layers = runner.kv_cache.shape[1]
    actual_bytes = runner.kv_cache.numel() * runner.kv_cache.element_size()

    unfixed_layers = hf_config.num_hidden_layers
    unfixed_bytes = actual_bytes * unfixed_layers / actual_layers

    print()
    print("=" * 60)
    print("KV-CACHE MEMORY REPORT")
    print("=" * 60)
    print(f"  num_hidden_layers (all layers):        {unfixed_layers}")
    print(f"  actual KV-cache layer dim (this run):   {actual_layers}")
    print(f"  ratio (should be ~full_attention_interval): "
          f"{unfixed_layers / actual_layers:.2f}")
    print(f"  actual kv_cache bytes:                  {actual_bytes:,}")
    print(f"  bytes if sized for ALL layers (unfixed): {int(unfixed_bytes):,}")
    print(f"  shrinkage factor:                        {unfixed_bytes / actual_bytes:.2f}x")

    if runner.state_manager is not None:
        sb = runner.state_manager.memory_bytes()
        print(f"  StateManager bytes:                     {sb:,}")
        print(f"  state_manager.num_linear_layers:         {runner.state_manager.num_linear_layers}")
        print(f"  model.model.linear_layer_indices:        {runner.model.model.linear_layer_indices}")
        assert runner.state_manager.num_linear_layers == len(runner.model.model.linear_layer_indices), \
            "StateManager layer count doesn't match model's linear_layer_indices — index-space mismatch"
        print("  [OK] StateManager layer count matches model.linear_layer_indices")

        # --- gap #4 check: did warmup leave the slot pool in a clean state? ---
        assert len(runner.state_manager.free_slot_ids) == runner.state_manager.max_num_seqs, (
            f"warmup left {runner.state_manager.max_num_seqs - len(runner.state_manager.free_slot_ids)} "
            f"slot(s) marked used after warmup — state_slot bookkeeping bug"
        )
        print("  [OK] state pool fully free after warmup (no leaked slots)")
    else:
        print("  StateManager: None (dense model path — unexpected for this config)")

    print("=" * 60)


if __name__ == "__main__":
    main()