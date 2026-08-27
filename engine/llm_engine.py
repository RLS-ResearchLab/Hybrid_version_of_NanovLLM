import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self.ack_events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            ack_event = ctx.Event()
            ack_event.set()  # no message sent yet -- rank0's first write_shm() must not block waiting for an ack
            process = ctx.Process(target=ModelRunner, args=(config, i, event, ack_event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
            self.ack_events.append(ack_event)
        # Registered BEFORE constructing rank0's own ModelRunner (below), not
        # after: rank>0 processes are already alive at this point (started
        # above), and rank0's construction can itself raise (e.g. OOM) or
        # hang (e.g. waiting on a collective from a rank>0 peer that already
        # died) before ever completing. Registering only after a successful
        # construction left self.ps permanently orphaned -- still holding
        # their GPU memory -- on any rank0-side failure or Ctrl-C during
        # this window, since exit() never even got registered to run.
        atexit.register(self.exit)
        self.model_runner = ModelRunner(config, 0, self.events, self.ack_events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config, self.model_runner.state_manager, self.model_runner, self.tokenizer)

    def exit(self):
        # Do NOT bail out early just because rank0's own ModelRunner never
        # finished constructing (or failed) -- self.ps (rank>0 processes) are
        # started earlier and independently, and are exactly the processes
        # most likely to be orphaned, alive, and holding GPU memory in that
        # scenario. They must still be cleaned up below.
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model_runner.call("exit")
            del self.model_runner
        for p in self.ps:
            # terminate() (SIGTERM) before join(): a rank>0 peer can be stuck
            # blocked inside a collective (e.g. dist.barrier()/init_process_group)
            # that will never complete on its own if rank0 died first -- plain
            # join() alone would hang this cleanup forever in that case.
            if p.is_alive():
                p.terminate()
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        # Returned so a caller driving step() from a single dedicated thread
        # (see src/server.py's BatchedEngine) can correlate a finished
        # output back to the request that submitted it, without needing to
        # also drive generate()'s own while-loop -- purely additive, every
        # existing caller already ignores this return value.
        return seq.seq_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            # schedule() placed nobody this tick -- e.g. a decode round that
            # had to preempt the entire running set for lack of a free KV
            # block (see Scheduler.schedule). The preempted sequences are back
            # in self.waiting and the next tick re-admits them via prefill.
            # Nothing to run; don't drive the model on an empty batch.
            return [], 0
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
