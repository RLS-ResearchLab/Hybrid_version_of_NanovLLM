"""Shared test utilities for the Qwen3.5 hybrid engine test suite.

init_model_weights_ REPLACES every ad-hoc, per-test weight-init block in
this project. Use it, don't re-derive it -- this exact bug (isinstance()
checks missing nano-vLLM's custom ColumnParallelLinear / RowParallelLinear /
ReplicatedLinear / MergedColumnParallelLinear classes, none of which are
plain nn.Linear, leaving torch.empty()-garbage weights that are frequently
all zeros on a fresh CUDA allocation) has independently recurred in Phase 1,
Phase 2, Phase 4, and Phase 3's preemption test. The fingerprint is always
the same: cosine == 0.0 or nan, every argmax landing on the same index, or
-- the more dangerous variant -- a test that "passes" because the model is
producing degenerate zero output regardless of what's being tested, not
because the thing under test is actually correct.

Usage:
    from test_utils import init_model_weights_
    init_model_weights_(runner.model, seed=42)
"""

import torch


def init_model_weights_(model: torch.nn.Module, seed: int = 42) -> None:
    """In-place, shape-based initialization covering EVERY parameter,
    regardless of which nano-vLLM linear/embedding subclass owns it.

    Rule: parameters with ndim >= 2 (weight matrices) get small random
    values; parameters with ndim < 2 (biases, 1D norm weights, A_log,
    dt_bias) get zero-initialized. This does not attempt to reproduce the
    reference model's exact init semantics for norm weights (e.g.
    Qwen35RMSNormGated's weight is meant to start at ones, not zeros) --
    if a specific test cares about that distinction, initialize those
    modules explicitly by type AFTER calling this function, not instead of
    it. For pure "does this code path produce real, non-degenerate output"
    checks (contamination tests, preemption tests, graph-vs-eager
    consistency), zero vs. one on a norm weight doesn't matter; only
    catching true torch.empty() garbage does.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                torch.nn.init.zeros_(param)


def assert_not_degenerate(logits: torch.Tensor, context: str = "") -> None:
    """Cheap guard to catch the all-zero-output fingerprint immediately,
    rather than downstream as a confusing 'test passed for the wrong
    reason' or 'cosine == nan' surprise several steps later.

    Call this right after the first real forward pass in any new test,
    before trusting anything the test goes on to measure.
    """
    abs_sum = logits.float().abs().sum().item()
    assert abs_sum > 0.0, (
        f"Degenerate (all-zero) logits detected{f' ({context})' if context else ''} -- "
        f"this almost always means some parameter upstream is still "
        f"torch.empty()-uninitialized. Use init_model_weights_(model) on "
        f"every parameter (shape-based, not isinstance-based) before running "
        f"anything through the model."
    )