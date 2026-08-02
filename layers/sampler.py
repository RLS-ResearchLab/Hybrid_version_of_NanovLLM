import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        # temperature==0 means greedy (deterministic argmax on the raw
        # logits, no randomness at all) -- NOT an edge case of the
        # stochastic path below. Dividing by temperature==0 directly would
        # produce inf/nan; a small positive stand-in (e.g. 0.01) sharpens
        # the softmax close to one-hot but the Gumbel-max trick below still
        # isn't guaranteed to pick the true argmax every call, which breaks
        # exact-match eval reproducibility. Batches can mix greedy and
        # sampled requests (different seqs, different temperatures), so
        # this computes both paths and selects per-row -- not a global
        # branch on the whole batch.
        greedy_mask = temperatures == 0
        safe_temperatures = temperatures.masked_fill(greedy_mask, 1.0)
        scaled_logits = logits.div(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled_tokens = probs.div(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)
