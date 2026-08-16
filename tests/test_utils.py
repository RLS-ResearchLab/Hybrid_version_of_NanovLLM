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

assert_all_parameters_initialized REPLACES manually eyeballing a model for
torch.empty() garbage after construction/loading. This exact bug
(uninitialized/mis-initialized parameters silently producing degenerate or
misleading results) has now recurred FOUR times under four different
mechanisms -- see that function's docstring for the full incident list.
Vigilance was not a sufficient control; call this guardrail instead.

Usage:
    from test_utils import (
        init_model_weights_,              # shape-based only (dim>=2 -> normal, else zero)
        init_model_weights_with_norms_,    # + correct-by-type norm-weight override (preferred default)
        init_reference_moe_experts_,       # src/model_small_qwen3.5.py-only: fixes Experts, leaves the rest
        known_zero_initialized_param_names,
        assert_all_parameters_initialized,
    )
    init_model_weights_with_norms_(model, seed=42)
    assert_all_parameters_initialized(
        model, whitelist_zero=known_zero_initialized_param_names(model)
    )
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


def init_model_weights_with_norms_(model: torch.nn.Module, seed: int = 42) -> None:
    """init_model_weights_, plus the correct-by-type norm-weight override on
    top -- centralizes the isinstance() block that fixed the Phase 2 incident
    (a blanket "dim>=2 -> normal, else zero" rule left every
    Qwen35RMSNormGated.weight at zero, silencing the GDR path's output; the
    reference initializes that gain to ONES). That exact isinstance() block
    previously lived inline in test_qwen35_batching.build_model_and_state --
    this is that same logic, moved here so every harness that needs a
    numerically-plausible full model calls ONE shared function instead of
    re-deriving it a fifth time.

    Qwen35RMSNormGated.weight -> ones (reference's gain init).
    Qwen35RMSNorm.weight (incl. q_norm/k_norm, which are Qwen35RMSNorm
      instances too) -> zeros (reference's (1+w) variant init).
    Everything else -> init_model_weights_'s shape-based rule.

    Lazy nanovllm import: matches this file's existing zero-dependency-at-
    import-time property (nanovllm is a fake package registered at runtime
    by test_qwen35_standalone.init_dist()/make_small_config(); importing it
    at module load time here would impose an import-order requirement on
    every caller of this file).
    """
    from nanovllm.layers.layernorm import Qwen35RMSNorm, Qwen35RMSNormGated

    init_model_weights_(model, seed=seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, Qwen35RMSNormGated):
                torch.nn.init.ones_(module.weight)     # gain, ref inits to ones
            elif isinstance(module, Qwen35RMSNorm):
                torch.nn.init.zeros_(module.weight)    # (1+w) variant, ref inits to zeros


def init_reference_moe_experts_(model: torch.nn.Module, seed: int = 42, std: float = 0.02) -> None:
    """Initializes src/model_small_qwen3.5.py's Experts.gate_up_proj /
    down_proj -- the ONE thing that module tree does not self-initialize
    (RMSNorm/RMSNormGated construct with torch.zeros/torch.ones, nn.Linear
    self-inits by default; Experts alone uses raw
    nn.Parameter(torch.empty(...)) with no init call, see that class's
    docstring). src/model_small_qwen3.5.py itself is the project's numerical
    ground truth and is off-limits to edit, so this has to happen in the
    caller after construction, not in the class.

    This is a generalization of the fix tests/test_qwen35_vectorized_moe.py
    already found and applied locally (its test_vs_reference does
    `ref_moe.experts.gate_up_proj.normal_(0, 0.02)` /
    `.down_proj.normal_(0, 0.02)` by hand) -- pulled out here so
    tests/test_qwen35_full_model.py, tests/test_qwen35_standalone.py, and
    tests/make_fake_checkpoint.py can all call the same fix instead of each
    re-deriving it. Deliberately NOT init_model_weights_with_norms_: that
    function's norm-weight override isinstance-checks against
    nanovllm.layers.layernorm's classes, which src/model_small_qwen3.5.py's
    own RMSNorm/RMSNormGated are NOT instances of (separate module, loaded
    via importlib) -- calling it on a reference-module instance would fall
    through to the shape-based rule for every param, zeroing
    RMSNormGated.weight (1D) instead of leaving it at the reference's own
    correct ones-init. That is exactly the Phase 2 incident this project
    already had once; this function only touches the two tensors that are
    actually broken; everything else in the reference model is left as the
    reference constructed it.

    Duck-typed rather than isinstance-checked against Experts: the
    reference module is loaded per-call via importlib under a synthetic
    module name (test_qwen35_standalone.load_reference_module()), so two
    calls in the same process produce two distinct, non-identical Experts
    classes -- isinstance would silently fail to match a model built from a
    different load_reference_module() call. Matching on "this module owns
    parameters literally named gate_up_proj and down_proj" is exactly
    Experts' shape, and nothing else in this file has both.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for module in model.modules():
            own = dict(module.named_parameters(recurse=False))
            if "gate_up_proj" in own and "down_proj" in own:
                own["gate_up_proj"].normal_(0, std)
                own["down_proj"].normal_(0, std)


def known_zero_initialized_param_names(model: torch.nn.Module) -> set[str]:
    """Dotted names of parameters that are legitimately all-zero BY DESIGN in
    this codebase, for use as assert_all_parameters_initialized's
    whitelist_zero. Two families, both self-initializing (not torch.empty()):

      - Qwen35RMSNorm.weight (incl. q_norm/k_norm submodules): the (1+w)
        variant, reference initializes w to zero -- see layers/layernorm.py.
      - Qwen35LinearAttention.A_log / .dt_bias: constructed via
        nn.Parameter(torch.zeros(...)) directly in models/qwen3_5.py, not
        torch.empty() -- see the comment at that call site.

    Deliberately NOT a guess made inside the guardrail itself -- the caller
    must call this (or build their own set) and pass it explicitly, per the
    Phase 2 lesson: a helper that silently decides which zeros are "fine" by
    type-sniffing is exactly the kind of blanket rule that caused that
    incident in the first place.
    """
    from nanovllm.layers.layernorm import Qwen35RMSNorm

    names: set[str] = set()
    for mod_name, module in model.named_modules():
        prefix = f"{mod_name}." if mod_name else ""
        if isinstance(module, Qwen35RMSNorm):
            names.add(f"{prefix}weight")
        for local_name, _ in module.named_parameters(recurse=False):
            if local_name in ("A_log", "dt_bias"):
                names.add(f"{prefix}{local_name}")
    return names


def assert_all_parameters_initialized(
    model: torch.nn.Module,
    whitelist_zero: "set[str] | frozenset[str]" = frozenset(),
    max_abs: float = 1000.0,
) -> None:
    """Precondition guardrail: walks EVERY parameter and buffer and fails
    loudly, naming the specific tensor, if it still looks like torch.empty()
    garbage rather than something that was actually initialized or loaded.
    Call this right after building/loading a model and BEFORE running
    anything through it -- it is a precondition check, not a replacement for
    a test's real assertions.

    This exists because "someone remembers to init every parameter" has
    independently failed FOUR times in this project (Phase 1's cosine==0.0,
    Phase 2's blanket-zeroed RMSNormGated.weight, the MoE port's raw
    torch.empty() Experts params, and the EP histogram's un-initialized
    ReplicatedLinear gate.weight producing a false "247/256 experts
    starved" result) -- four different mechanisms, which means vigilance
    isn't the right control. A guardrail that runs every time is.

    Three independent checks, since torch.empty() garbage can look like any
    of them depending on what bytes happened to be sitting in that memory:

      1. Non-finite (NaN/Inf anywhere in the tensor). Never legitimate,
         no whitelist accepted.
      2. All-zero. Legitimate for SOME tensors by design (norm gains that
         start at zero, A_log/dt_bias) -- those must be passed in
         whitelist_zero by dotted name (see known_zero_initialized_param_names
         for the common case). Everything else that's all-zero fails: a
         fresh CUDA allocation reads as all-zero disproportionately often,
         which is exactly the Phase 1 fingerprint.
      3. Suspiciously large in magnitude (> max_abs). Heuristic backstop for
         torch.empty() garbage that happens to be finite and non-zero:
         reinterpreting arbitrary leftover bytes as floats spreads the
         result across nearly the full representable range, so almost none
         of that probability mass lands in a "plausible weight" band. Every
         value this codebase intentionally produces stays far under 100 in
         magnitude -- normal_(std=0.02) matrices practically never sample
         above ~1, norm gains are exactly 0 or 1, A_log/dt_bias start at 0,
         and even a real bf16-checkpoint outlier weight is orders of
         magnitude under this. max_abs=1000.0 leaves a wide margin above all
         of that (so it will not false-positive on any value this project
         actually produces) while still being tight enough to catch garbage
         draws that land outside a normal float range. This is a heuristic,
         not a proof -- it is a backstop alongside checks 1 and 2, not a
         replacement for them.
    """
    import itertools

    for name, tensor in itertools.chain(
        model.named_parameters(), model.named_buffers()
    ):
        if tensor.numel() == 0:
            continue
        t = tensor.detach().float()

        if not torch.isfinite(t).all():
            raise AssertionError(
                f"assert_all_parameters_initialized: '{name}' contains "
                f"NaN/Inf -- this is the torch.empty()-garbage fingerprint, "
                f"not a legitimate initialized or loaded value. Initialize "
                f"it (init_model_weights_with_norms_ or the real checkpoint "
                f"loader) before running anything through this model."
            )

        if name not in whitelist_zero and torch.count_nonzero(t).item() == 0:
            raise AssertionError(
                f"assert_all_parameters_initialized: '{name}' is all-zero "
                f"and not in whitelist_zero -- this is the Phase 1/EP-"
                f"histogram fingerprint (uninitialized torch.empty() memory, "
                f"or a fresh CUDA allocation, both frequently read back as "
                f"all-zero). If this parameter is LEGITIMATELY zero-"
                f"initialized by design, add its exact name to whitelist_zero "
                f"explicitly (see known_zero_initialized_param_names) -- "
                f"don't silence this by guessing."
            )

        max_val = t.abs().max().item()
        if max_val > max_abs:
            raise AssertionError(
                f"assert_all_parameters_initialized: '{name}' has max|.| = "
                f"{max_val:.3g}, exceeding the max_abs={max_abs:g} heuristic "
                f"threshold for plausible initialized/loaded values -- this "
                f"looks like torch.empty() garbage (arbitrary leftover bytes "
                f"reinterpreted as floats), not a real weight."
            )


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