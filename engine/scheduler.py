from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# Bounds the cost of _check_stop_string to a fixed-size decode+scan per step,
# regardless of how long the sequence has grown -- NOT a full
# redecode-and-rescan of the whole completion every step (that would be
# O(completion length) per step, O(length^2) over a sequence). 32 tokens is
# generous headroom for a marker phrase ("the answer is " / "#### ") plus a
# multi-digit/decimal number plus the trailing delimiter the stop patterns
# require -- see gsm8k_extract.py's GSM8K_STOP_PATTERNS docstring.
_STOP_CHECK_WINDOW_TOKENS = 32


class Scheduler:

    def __init__(self, config: Config, state_manager=None, model_runner=None, tokenizer=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        # Only needed for stop-string checking (_check_stop_string decodes a
        # trailing window of completion tokens back to text). Optional, like
        # state_manager/model_runner below -- callers that never pass `stop`
        # in a SamplingParams never need it, and _check_stop_string's
        # `if not seq.stop_patterns: return False` guard means this is never
        # dereferenced when unused.
        self.tokenizer = tokenizer
        # disable_prefix_cache=True whenever a StateManager is active -- see
        # BlockManager.__init__'s docstring comment for why KV-cache reuse
        # and always-reset-to-zero recurrent state are incompatible.
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size,
            disable_prefix_cache=(state_manager is not None),
        )
        self.state_manager = state_manager
        # model_runner is optional -- when given (the real LLMEngine always
        # gives one), allocate/free are dispatched to EVERY tensor-parallel
        # rank via model_runner.call(...), not just applied to this
        # process's local state_manager. Without it (only the two
        # tensor_parallel_size=1 standalone tests in tests/ construct
        # Scheduler this way, setting .state_manager directly afterward),
        # falls back to calling state_manager directly -- correct there
        # since a single rank has no "other ranks" to leave un-zeroed.
        self.model_runner = model_runner
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def _allocate_state(self, seq: Sequence) -> None:
        """See engine/model_runner.py's allocate_state_slot() docstring for
        why this must reach every rank's own StateManager, not just
        whichever process happens to run this Scheduler."""
        if self.model_runner is not None:
            self.model_runner.call("allocate_state_slot", seq)
        elif self.state_manager is not None:
            self.state_manager.allocate(seq)

    def _free_state(self, seq: Sequence) -> None:
        """Free-side counterpart of _allocate_state -- see
        engine/model_runner.py's free_state_slot()."""
        if self.model_runner is not None:
            self.model_runner.call("free_state_slot", seq)
        elif self.state_manager is not None:
            self.state_manager.free(seq)

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                if self.state_manager is not None:
                    self._allocate_state(seq)
                    print(f"[REPREFILL DEBUG] seq_id={seq.seq_id} num_tokens={seq.num_tokens} "
                    f"num_cached_tokens={seq.num_cached_tokens} state_slot={seq.state_slot}")
                seq.num_cached_tokens = num_cached_blocks * self.block_size
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        if self.state_manager is not None:
            self._free_state(seq)
        seq.num_cached_tokens = 0

        print(f"[PREEMPT DEBUG] seq_id={seq.seq_id} num_tokens={seq.num_tokens} "
          f"num_prompt_tokens={seq.num_prompt_tokens} token_ids_len={len(seq.token_ids)} "
          f"state_slot={seq.state_slot}")

        self.waiting.appendleft(seq)

    def _check_stop_string(self, seq: Sequence) -> bool:
        """Regex match against a bounded trailing window of DECODED
        COMPLETION text -- see _STOP_CHECK_WINDOW_TOKENS above for why this
        is a fixed-size window, not the full completion. Deliberately scans
        seq.completion_token_ids (tokens generated AFTER the prompt), never
        seq.token_ids (which includes the prompt): every GSM8K prompt this
        engine is actually run against (gsm8k_prompt.py's 8-shot exemplars)
        contains "The answer is N." literally in the PROMPT text itself --
        scanning the prompt would false-trigger before generation even
        starts. Requires seq.stop_patterns to already be compiled (Sequence
        does this once at construction, not per-call) and self.tokenizer to
        be set (only true when the caller passed one -- see __init__)."""
        if not seq.stop_patterns:
            return False
        window_ids = seq.completion_token_ids[-_STOP_CHECK_WINDOW_TOKENS:]
        window_text = self.tokenizer.decode(window_ids)
        for p in seq.stop_patterns:
            m = p.search(window_text)
            if m:
                # Diagnostic for exactly the "stopped later than expected"
                # question -- shows what ACTUALLY matched, not what we
                # assumed would. Cheap: only runs once, at the single step a
                # match fires, not every step.
                print(f"[STOP-STRING DEBUG] seq_id={seq.seq_id} stopped at "
                      f"completion_token={seq.num_completion_tokens} "
                      f"matched_pattern={p.pattern!r} matched_text={m.group(0)!r} "
                      f"full_window={window_text!r}")
                return True
        return False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # Stop-string match is checked identically to EOS/max_tokens --
            # same single finish path below, not a separate parallel
            # mechanism. Short-circuits via `or` so _check_stop_string's
            # tokenizer.decode() call never runs for sequences with no
            # stop_patterns (the common case) or once EOS has already fired.
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or self._check_stop_string(seq)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if self.state_manager is not None:
                   self._free_state(seq)
                self.running.remove(seq)
