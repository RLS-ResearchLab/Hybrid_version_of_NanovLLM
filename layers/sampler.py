import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Whole-batch all-greedy fast path -- the common eval/benchmark case
        # (temperature=0 for every request). _sample() below computes a
        # vocab-wide softmax + a Gumbel `exponential_()` RNG draw + a
        # division, then discards all of it via torch.where whenever a row is
        # greedy. At vocab~248k, batch 64 that is ~300 MB of pointless
        # memory traffic per decode step. `.all()` costs one ~microsecond
        # D2H sync (dwarfed by the `.tolist()` the caller already does on
        # this function's output), so skipping the stochastic path entirely
        # when nothing is sampled is a clear win. Kept OUT of the
        # @torch.compile'd body so it never forces a graph break, and it
        # means _sample() is never even compiled for batch shapes that only
        # ever run all-greedy.
        if bool((temperatures == 0).all()):
            return logits.float().argmax(dim=-1)
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
