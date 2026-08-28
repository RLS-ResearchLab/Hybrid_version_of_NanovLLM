"""Phase 4 acceptance test: eager vs. graph-replay decode consistency,
across multiple batch sizes (not just whatever one batch size happened to
fall out of a fixed prompt list).

Usage:
    python tests/make_fake_hf_config.py   # if not already run
    python tests/cuda_graph_consistency_test.py
"""

import sys, os, json
import torch

sys.path.insert(0, os.path.dirname(__file__))

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
loader_mod.load_model = lambda model, path, **kw: None  # no real .safetensors to load

import nanovllm.config as config_mod


class AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _fake_from_pretrained(path, *args, **kwargs):
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d


def _decode_logits_no_sample(runner, seqs, force_eager: bool):
    """Mirrors ModelRunner.run()'s hybrid decode branch, but returns raw
    logits instead of sampling -- avoids RNG as a confound when comparing
    graph vs. eager numerically."""
    from nanovllm.utils.context import get_context, reset_context

    input_ids, positions = runner.prepare_decode(seqs)
    context = get_context()

    use_graph = (not force_eager) and (not runner.enforce_eager) \
        and hasattr(runner, "graphs") and input_ids.size(0) <= 512

    if use_graph:
        with torch.inference_mode():
            bs = input_ids.size(0)
            graph = runner.graphs[next(x for x in runner.graph_bs if x >= bs)]
            gv = runner.graph_vars
            gv["input_ids"][:bs] = input_ids
            gv["positions"][:bs] = positions
            gv["slot_mapping"].fill_(-1)
            gv["slot_mapping"][:bs] = context.slot_mapping
            gv["context_lens"].zero_()
            gv["context_lens"][:bs] = context.context_lens
            gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            gv["state_slot_ids"][:bs] = context.state_slot_ids

            graph.replay()

            # State write-back happens INSIDE the captured graph (the
            # in-graph index_copy_ in capture_cudagraph()'s _step()) --
            # runner.state_manager's real buffers are already correct by
            # the time replay() returns, matching production ModelRunner.run().
            hidden = gv["outputs"][:bs]
            logits = runner.model.compute_logits(hidden)
    else:
        num_layers = len(runner.model.model.layers)
        linear_idx = runner.model.model.linear_layer_indices
        states, conv_states = runner.state_manager.get_all(context.state_slot_ids, num_layers, linear_idx)
        logits, new_states, new_conv_states = runner.model(
            input_ids, positions, context.cu_seqlens_q, states, conv_states
        )
        runner.state_manager.set_all(context.state_slot_ids, new_states, new_conv_states, linear_idx)
        logits = runner.model.compute_logits(logits)

    reset_context()
    return logits


def run_one_batch_size(runner, scheduler, Sequence, bs, seed):
    """Prefill `bs` fresh sequences, then compare graph vs eager for one
    decode step at this batch size. Cleans up (frees blocks/state slots)
    before returning, so the next batch size starts from a clean slate."""
    torch.manual_seed(seed)
    prompts = [torch.randint(100, 5000, (torch.randint(4, 8, (1,)).item(),)).tolist() for _ in range(bs)]
    for p in prompts:
        scheduler.add(Sequence(p))

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill, f"[bs={bs}] expected prefill on first step"
    token_ids = runner.run(seqs, is_prefill)
    scheduler.postprocess(seqs, token_ids, is_prefill)

    assert len(scheduler.running) == bs, (
        f"[bs={bs}] expected all {bs} sequences to finish prefill in one step, "
        f"got {len(scheduler.running)} running -- check chunking didn't trigger "
        f"(reduce prompt lengths or raise max_num_batched_tokens if this fires)"
    )
    decode_seqs = list(scheduler.running)
    for s in decode_seqs:
        assert s.state_slot is not None, f"[bs={bs}] sequence missing state_slot after prefill"
        s.num_scheduled_tokens = 1

    sm = runner.state_manager
    states_snapshot = sm.states.clone()
    conv_snapshot = sm.conv_states.clone()
    kv_snapshot = runner.kv_cache.clone()

    logits_graph = _decode_logits_no_sample(runner, decode_seqs, force_eager=False)

    sm.states.copy_(states_snapshot)
    sm.conv_states.copy_(conv_snapshot)
    runner.kv_cache.copy_(kv_snapshot)
    logits_eager = _decode_logits_no_sample(runner, decode_seqs, force_eager=True)

    cos = torch.nn.functional.cosine_similarity(
        logits_graph.float().reshape(1, -1), logits_eager.float().reshape(1, -1)
    ).item()
    top1_graph = logits_graph.float().argmax(dim=-1)
    top1_eager = logits_eager.float().argmax(dim=-1)
    top1_match = bool((top1_graph == top1_eager).all().item())

    # Clean up: free every sequence from this round so the next batch size
    # starts with a full pool of blocks and state slots.
    for s in decode_seqs:
        scheduler.block_manager.deallocate(s)
        sm.free(s)
        scheduler.running.remove(s)

    return cos, top1_match, top1_graph.tolist(), top1_eager.tolist()


def main():
    assert torch.cuda.is_available(), "requires CUDA (flash-attn, paged KV cache, CUDA graphs)"

    config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)

    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence

    fake_dir = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
    assert os.path.isdir(fake_dir), "run tests/make_fake_hf_config.py first"

    # max_num_seqs raised to 8 (from the original 4) so StateManager has
    # enough slots to actually test batch size 8, one of the sizes
    # capture_cudagraph always captures regardless of max_num_seqs.
    config = Config(
        model=fake_dir,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=128,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
        enforce_eager=False,  # graphs must be built for this test
    )

    print("Constructing ModelRunner (graphs enabled)...")
    runner = ModelRunner(config, rank=0, event=[])
    print("  [OK] constructed, graphs captured for batch sizes:", runner.graph_bs)

    with torch.no_grad():
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        for name, param in runner.model.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                torch.nn.init.zeros_(param)

    scheduler = Scheduler(config)
    scheduler.state_manager = runner.state_manager

    batch_sizes_to_test = [1, 4, 8]
    results = {}

    print("\n" + "=" * 70)
    print("PHASE 4: CUDA GRAPH vs EAGER DECODE CONSISTENCY — MULTI BATCH SIZE")
    print("=" * 70)

    for i, bs in enumerate(batch_sizes_to_test):
        print(f"\n--- batch_size={bs} ---")
        cos, top1_match, t1g, t1e = run_one_batch_size(
            runner, scheduler, Sequence, bs, seed=100 + i
        )
        print(f"  cosine(graph, eager): {cos:.6f}")
        print(f"  top-1 match (all):    {top1_match}")
        print(f"  top-1 graph:          {t1g}")
        print(f"  top-1 eager:          {t1e}")
        status = "PASS" if (cos > 0.999 and top1_match) else "FAIL"
        print(f"  [{status}]")
        results[bs] = (cos, top1_match)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for bs, (cos, top1_match) in results.items():
        ok = cos > 0.999 and top1_match
        all_pass = all_pass and ok
        print(f"  batch_size={bs}: cosine={cos:.6f} top1_match={top1_match} -> {'PASS' if ok else 'FAIL'}")

    assert all_pass, "One or more batch sizes failed graph-vs-eager consistency"
    print("\n[PASS] graph-replay decode matches eager decode across all tested batch sizes")


if __name__ == "__main__":
    main()