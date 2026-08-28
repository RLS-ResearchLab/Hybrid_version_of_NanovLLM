import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                all_greedy: bool | None = None):
        # Whole-batch all-greedy fast path -- the common eval/benchmark case
        # (temperature=0 for every request). _sample() below computes a
        # vocab-wide softmax + a Gumbel `exponential_()` RNG draw + a
        # division, then discards all of it via torch.where whenever a row is
        # greedy. At vocab~248k, batch 64 that is ~300 MB of pointless
        # memory traffic per decode step, so skipping the stochastic path
        # entirely when nothing is sampled is a clear win.
        #
        # all_greedy: caller-supplied shortcut. The temperatures tensor's
        # values are already known on the host in plain Python BEFORE it's
        # ever built (engine/model_runner.py's prepare_sample() constructs it
        # straight from seq.temperature) -- so model_runner passes the
        # all-greedy decision through here precomputed, and the GPU-side
        # `bool((temperatures == 0).all())` fallback below (still used by
        # callers that only have the tensor, e.g. diagnostics/tests) never
        # needs to run in the hot path. That fallback forces a GPU->host
        # sync every decode step to re-derive a fact the caller already had
        # for free -- a real, avoidable synchronization point, not just a
        # few wasted cycles: nsys profiling (2026-08-28) found an
        # unidentified, high-variance ~2ms/step cost in this exact region,
        # and a data-dependent host sync sitting between two kernel launches
        # is a textbook cause of exactly that variance signature. Kept OUT
        # of the @torch.compile'd body so it never forces a graph break, and
        # it means _sample() is never even compiled for batch shapes that
        # only ever run all-greedy.
        if all_greedy is None:
            all_greedy = bool((temperatures == 0).all())
        if all_greedy:
            # No .float() before argmax: bf16/fp16 -> fp32 is an exact,
            # order-preserving widening (every low-precision value maps to a
            # unique, correctly-ordered fp32 value), so casting first cannot
            # change argmax's result for ANY input dtype -- this is a strict
            # mathematical equivalence, not an approximation (see
            # tests/test_sampler_greedy_fastpath_cpu.py's bf16 case). Removing
            # it also removes real, unnecessary traffic: at vocab~248k,
            # batch 64, the cast alone was writing an extra ~64 MB/step for
            # no benefit.
            return logits.argmax(dim=-1)
        return self._sample(logits, temperatures)

    @torch.compile
    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        # temperature==0 means greedy (deterministic argmax on the raw
        # logits, no randomness at all) -- NOT an edge case of the
        # stochastic path below. Dividing by temperature==0 directly would
        # produce inf/nan; a small positive stand-in (e.g. 0.01) sharpens
        # the softmax close to one-hot but the Gumbel-max trick below still
        # isn't guaranteed to pick the true argmax every call, which breaks
        # exact-match eval reproducibility. Batches CAN mix greedy and
        # sampled requests (different seqs, different temperatures), so this
        # computes both paths and selects per-row -- forward()'s fast path
        # above only fires when the WHOLE batch is greedy.
        greedy_mask = temperatures == 0
        safe_temperatures = temperatures.masked_fill(greedy_mask, 1.0)
        scaled_logits = logits.div(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled_tokens = probs.div(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)
