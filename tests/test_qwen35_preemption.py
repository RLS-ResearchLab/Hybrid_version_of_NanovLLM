"""Phase 3 acceptance test: preemption-forced run must produce
byte-identical output vs. an unconstrained run.

Forces Scheduler.preempt() to actually fire by constructing a ModelRunner
with a deliberately small num_kvcache_blocks (via low gpu_memory_utilization),
running several concurrent sequences long enough that they can't all fit
their KV blocks simultaneously, and comparing the resulting tokens against
the same sequences run with no memory pressure.

Sampling is monkeypatched to plain argmax (greedy) for both runs -- the
stock Sampler's Gumbel-max trick draws from torch's RNG stream once per
call, and preemption changes how many calls happen and in what order, so a
"byte-identical" comparison against stochastic sampling is meaningless by
construction, not just imprecise. Greedy removes that confound entirely.

Usage:
    python tests/make_fake_hf_config.py
    python tests/test_qwen35_preemption.py
"""
import sys, os, json, gc
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
loader_mod.load_model = lambda model, path: None

import nanovllm.config as config_mod
sys.path.insert(0, os.path.dirname(__file__))
from test_utils import init_model_weights_, assert_not_degenerate

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


def _build_engine_pieces(gpu_memory_utilization, max_num_seqs=4):
    """Construct a fresh ModelRunner + Scheduler pair with a given memory
    budget. Returns (runner, scheduler). Re-seeds weight init identically
    every call so different memory budgets are still comparing the SAME
    model weights."""
    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.models.qwen3_5 import Experts
    

    fake_dir = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
    config = Config(
        model=fake_dir,
        max_num_batched_tokens=256,
        max_num_seqs=max_num_seqs,
        max_model_len=128,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=True,   # keep this test focused on preemption, not graphs
    )
    # ModelRunner.__init__ unconditionally calls dist.init_process_group(),
    # which throws if the default process group is already initialized --
    # and this helper gets called more than once per process (once for the
    # unconstrained run, once or more inside the memory-budget search loop).
    # Tear down any prior group before building the next ModelRunner.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    runner = ModelRunner(config, rank=0, event=[])

    init_model_weights_(runner.model, seed=2024)

    scheduler = Scheduler(config)
    scheduler.state_manager = runner.state_manager
    return runner, scheduler


def _run_to_completion(runner, scheduler, prompts, max_tokens, greedy_sampler):
    """Drive the real Scheduler/ModelRunner loop to completion for a batch
    of prompts, returning {index_in_prompts: generated_token_ids}."""
    from nanovllm.engine.sequence import Sequence, SequenceStatus
    from nanovllm.sampling_params import SamplingParams

    seqs = []
    for p in prompts:
        seq = Sequence(p, SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=1.0))
        scheduler.add(seq)
        seqs.append(seq)

    orig_sampler_forward = runner.sampler.forward
    runner.sampler.forward = greedy_sampler

    preemption_fired = False
    orig_preempt = scheduler.preempt
    def _tracking_preempt(seq):
        nonlocal preemption_fired
        preemption_fired = True
        return orig_preempt(seq)
    scheduler.preempt = _tracking_preempt

    steps = 0
    first_step = True
    while not scheduler.is_finished():
        batch, is_prefill = scheduler.schedule()
        token_ids = runner.run(batch, is_prefill)
        if first_step:
            # token_ids here are already sampled ints, not logits -- cheapest
            # real check at this point is just: are they suspiciously uniform?
            if len(set(token_ids)) == 1:
                print(f"  [WARNING] all sampled tokens identical on first step: {token_ids[0]} "
                      f"-- possible degenerate/uninitialized model, verify before trusting this run")
            first_step = False
        scheduler.postprocess(batch, token_ids, is_prefill)
        steps += 1

    runner.sampler.forward = orig_sampler_forward
    scheduler.preempt = orig_preempt

    return {i: seqs[i].completion_token_ids for i in range(len(seqs))}, preemption_fired, steps


def _force_small_kv_cache(runner, scheduler, num_blocks):
    """Directly truncate the block dimension of an already-constructed
    ModelRunner's kv_cache and re-wire k_cache/v_cache pointers, instead of
    hunting for a gpu_memory_utilization value that happens to land on a
    small block count. Deterministic and exact, rather than a coarse search
    over an indirect knob.

    CRITICAL: scheduler.block_manager was already constructed with the
    ORIGINAL (large) block count in _build_engine_pieces, before this ever
    runs. If not rebuilt here too, BlockManager will keep handing out block
    ids beyond the physical kv_cache tensor's new (smaller) size -- an
    out-of-bounds write in the Triton store_kvcache kernel, which does NOT
    bounds-check, silently corrupting adjacent GPU memory instead of
    raising an error. Must rebuild BlockManager in lockstep with kv_cache,
    not just truncate the tensor."""
    assert num_blocks < runner.config.num_kvcache_blocks
    runner.config.num_kvcache_blocks = num_blocks
    runner.kv_cache = runner.kv_cache[:, :, :num_blocks].contiguous()
    layer_id = 0
    for module in runner.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = runner.kv_cache[0, layer_id]
            module.v_cache = runner.kv_cache[1, layer_id]
            layer_id += 1
    assert layer_id == runner.kv_cache.shape[1]

    from nanovllm.engine.block_manager import BlockManager
    scheduler.block_manager = BlockManager(num_blocks, runner.block_size)


def main():
    assert torch.cuda.is_available(), "requires CUDA"
    config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)

    fake_dir = os.path.join(os.path.dirname(__file__), "fake_qwen35_small")
    assert os.path.isdir(fake_dir), "run tests/make_fake_hf_config.py first"

    def greedy_sampler(logits, temperatures):
        return logits.float().argmax(dim=-1)

    torch.manual_seed(500)
    # Prompts + generation must exceed block_size (256) for at least one
    # sequence, or Scheduler.preempt() can never fire at all -- it only
    # triggers when a RUNNING sequence needs a SECOND block (crossing a
    # block-size boundary), never on prefill-time admission contention.
    # Short sequences (prev attempt: 12-18 tokens + 20 generated) can only
    # ever exercise admission-stalling, a different, non-preemption code
    # path -- confirmed by that attempt's step count changing (20->40
    # steps) while preempted_c stayed False the whole time.
    prompts = [torch.randint(100, 5000, (n,)).tolist() for n in [10, 12, 8, 15]]
    max_tokens = 260   # prompt + generated > 256 for every sequence here

    # ---- unconstrained baseline: generous memory budget ----
    print("Building UNCONSTRAINED engine (generous memory budget)...")
    runner_u, sched_u = _build_engine_pieces(gpu_memory_utilization=0.5)
    baseline, preempted_u, steps_u = _run_to_completion(runner_u, sched_u, prompts, max_tokens, greedy_sampler)
    print(f"  done in {steps_u} steps, preemption fired: {preempted_u} (expected False)")
    assert not preempted_u, "unconstrained run unexpectedly preempted -- budget not generous enough, raise it"
    del runner_u, sched_u
    gc.collect()
    torch.cuda.empty_cache()

    # ---- memory-constrained: build once, then directly truncate kv_cache
    # to a small fixed block count that's guaranteed to force contention.
    # Each of the 4 prompts (max ~38 tokens including generation) needs
    # exactly 1 block at block_size=256, so num_blocks < 4 guarantees at
    # least one sequence can't get a block when all 4 are concurrently
    # scheduled. ----
    print("\nBuilding CONSTRAINED engine (will truncate kv_cache directly)...")
    runner_c, sched_c = _build_engine_pieces(gpu_memory_utilization=0.15, max_num_seqs=4)
    print(f"  built with {runner_c.config.num_kvcache_blocks} blocks (unconstrained-for-this-config)")
    _force_small_kv_cache(runner_c, sched_c, num_blocks=len(prompts))
    print(f"  truncated to {runner_c.config.num_kvcache_blocks} blocks "
          f"(== {len(prompts)} concurrent sequences -> 0 free once all admitted; "
          f"first sequence to cross a block-size boundary must preempt another)")

    constrained, preempted_c, steps_c = _run_to_completion(runner_c, sched_c, prompts, max_tokens, greedy_sampler)
    print(f"  done in {steps_c} steps, preemption fired: {preempted_c}")
    assert preempted_c, (
        "still no preemption even with only 2 blocks for 4 concurrent sequences -- "
        "check BlockManager.can_allocate()/Scheduler.schedule()'s decode-branch "
        "preemption trigger, this should have been unavoidable"
    )

    print(f"\nPreemption forced. Comparing outputs...")
    print("=" * 60)
    all_match = True
    for i in range(len(prompts)):
        match = baseline[i] == constrained[i]
        all_match &= match
        print(f"  seq {i}: {'MATCH' if match else 'MISMATCH'}"
              f"{'' if match else f' -- baseline={baseline[i]} constrained={constrained[i]}'}")
    print("=" * 60)
    assert all_match, "preempted run diverged from unconstrained run -- NOT byte-identical"
    print("[PASS] preemption-forced run is byte-identical to unconstrained run")


if __name__ == "__main__":
    main()