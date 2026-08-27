"""C2 regression: engine/scheduler.py's decode branch used to end with a bare
`assert scheduled_seqs`. Under real KV-cache pressure a decode round CAN
preempt every sequence it tries to run (each needs a fresh block, none free,
nobody left to evict) -- that assert then crashed the whole server instead of
letting the next round re-admit the preempted sequences via prefill.

Also covers the sibling case: a prompt that needs more KV blocks than the
cache physically holds must fail with a clear RuntimeError, not an
AssertionError and not an infinite empty-round spin.

Pure CPU -- Scheduler / BlockManager never touch a GPU or the model (same
rationale as tests/diag_scheduler_starvation_cpu.py, which this borrows the
_FakeConfig pattern from).

Usage:
    python tests/test_scheduler_empty_decode_round_cpu.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "nanovllm" not in sys.modules:
    pkg = types.ModuleType("nanovllm")
    pkg.__path__ = [ROOT]
    pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = pkg

from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402


class _FakeConfig:
    def __init__(self, max_num_seqs, max_num_batched_tokens, num_kvcache_blocks, block_size):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_kvcache_blocks = num_kvcache_blocks
        self.kvcache_block_size = block_size
        self.eos = -1
        self.debug_print_preemptions = False


def _drive(scheduler, seqs, max_rounds):
    """Fake LLMEngine.step() loop: schedule, fake-generate one token each,
    postprocess. Mirrors LLMEngine.step()'s new empty-batch guard."""
    rounds = empty_rounds = 0
    while not scheduler.is_finished() and rounds < max_rounds:
        rounds += 1
        scheduled, is_prefill = scheduler.schedule()
        if not scheduled:
            empty_rounds += 1
            if empty_rounds > 50:
                raise AssertionError("livelock: >50 consecutive-ish empty schedule() rounds")
            continue
        empty_rounds = 0
        scheduler.postprocess(scheduled, [42] * len(scheduled), is_prefill)
    return rounds


def test_heavy_preemption_no_assert():
    """3 sequences, each individually fits the cache (3 of 4 blocks at full
    length) but they cannot all fit at once (9 > 4) -- forces decode rounds
    that preempt the entire running set. The old `assert scheduled_seqs`
    turned that into a crash; now the engine must drain all 3 without ever
    raising AssertionError."""
    block_size = Sequence.block_size
    cfg = _FakeConfig(max_num_seqs=3, max_num_batched_tokens=1 << 20,
                      num_kvcache_blocks=4, block_size=block_size)
    sched = Scheduler(cfg, state_manager=None, model_runner=None, tokenizer=None)
    # prompt (~1 block) + 300 generated ≈ 554 tokens ≈ 3 blocks -- fits in 4,
    # three of them do not.
    sp = SamplingParams(temperature=0.0, max_tokens=300, ignore_eos=True)
    seqs = [Sequence([1] * (block_size - 2), sp) for _ in range(3)]
    for s in seqs:
        sched.add(s)

    try:
        rounds = _drive(sched, seqs, max_rounds=200000)
    except AssertionError as e:
        raise AssertionError(f"schedule() hit the old bare-assert failure mode: {e!r}")
    finished = sum(1 for s in seqs if s.is_finished)
    print(f"  heavy-preemption case: {rounds} rounds, {finished}/3 finished, no AssertionError")
    assert finished == 3, f"expected all 3 sequences to complete, got {finished}"
    print("  [PASS] heavy preemption drains cleanly, no assert crash")


def test_lone_sequence_fills_cache():
    """One sequence, a cache too small for its full length: schedule() must
    return ([], False) or raise a clear RuntimeError once the sequence
    outgrows the cache -- never a bare AssertionError."""
    block_size = Sequence.block_size
    cfg = _FakeConfig(max_num_seqs=1, max_num_batched_tokens=1 << 20,
                      num_kvcache_blocks=2, block_size=block_size)
    sched = Scheduler(cfg, state_manager=None, model_runner=None, tokenizer=None)
    sp = SamplingParams(temperature=0.0, max_tokens=block_size * 4, ignore_eos=True)
    seq = Sequence([1] * (block_size - 2), sp)
    sched.add(seq)

    saw_empty = False
    for _ in range(5000):
        if sched.is_finished():
            break
        try:
            scheduled, is_prefill = sched.schedule()
        except RuntimeError as e:
            assert "blocks" in str(e).lower()
            print(f"  [PASS] lone-seq case -> clear RuntimeError once it outgrows cache: {str(e)[:70]}...")
            return
        except AssertionError as e:
            raise AssertionError(f"schedule() hit the old bare assert: {e!r}")
        if not scheduled:
            saw_empty = True
            # engine would re-prefill next round; emulate that by continuing
            continue
        sched.postprocess(scheduled, [42] * len(scheduled), is_prefill)
    assert saw_empty or seq.is_finished, "expected an empty round or completion, not a silent stall"
    print("  [PASS] lone-seq case handled without AssertionError")


def test_prompt_larger_than_cache():
    """A prompt needing more blocks than the cache has must raise a clear
    RuntimeError, not AssertionError and not an infinite empty-round loop."""
    block_size = Sequence.block_size
    cfg = _FakeConfig(max_num_seqs=4, max_num_batched_tokens=1 << 20,
                      num_kvcache_blocks=2, block_size=block_size)
    sched = Scheduler(cfg, state_manager=None, model_runner=None, tokenizer=None)
    sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    big = Sequence([1] * (block_size * 5), sp)  # needs 5 blocks, cache has 2
    sched.add(big)

    raised = None
    try:
        sched.schedule()
    except RuntimeError as e:
        raised = e
    except AssertionError as e:  # the old failure mode -- must NOT happen
        raise AssertionError(f"schedule() hit the old bare assert instead of a clear error: {e!r}")
    assert raised is not None, "expected a RuntimeError for an unservable prompt"
    assert "blocks" in str(raised).lower()
    print(f"  [PASS] oversized prompt -> clear RuntimeError: {str(raised)[:80]}...")


def main():
    print("=" * 70)
    print("C2 -- scheduler empty-decode-round / oversized-prompt handling")
    print("=" * 70)
    test_heavy_preemption_no_assert()
    test_lone_sequence_fills_cache()
    test_prompt_larger_than_cache()
    print("\nALL C2 CHECKS PASSED")


if __name__ == "__main__":
    main()
