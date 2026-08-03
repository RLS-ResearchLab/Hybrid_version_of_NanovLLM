from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config, state_manager=None, model_runner=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
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

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if self.state_manager is not None:
                   self._free_state(seq)
                self.running.remove(seq)
