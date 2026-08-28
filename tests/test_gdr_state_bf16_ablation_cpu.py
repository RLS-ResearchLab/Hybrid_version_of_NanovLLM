"""bf16 GDR recurrent-state ablation -- does storing the state in bf16
instead of fp32 cause the "compounds rounding error into degenerate
repetition loops" failure engine/state_manager.py's own comment warns about?

WHY THIS TEST EXISTS: state_manager.py cites "src/model.py's comment" as the
source of that warning. `src/model.py` does not exist anywhere in this
repo's git history (confirmed via `git log --all --diff-filter=D`) -- it is
almost certainly a reference to nano-vLLM's separate "basic engine" sibling
codebase mentioned throughout src/server.py's docstrings, not something
this repo can trace or verify. This test re-derives the answer empirically,
independent of that unverifiable inherited claim, using this project's own
real recurrence math and real checkpoint dimensions.

METHOD: _forward_decode_batched (models/qwen3_5.py) always computes the
delta-rule recurrence in fp32 internally regardless of what dtype the
incoming `states` tensor is (`S = states.float()`) and always returns fp32
(`new_states = S.detach()`, documented float32 in the method's own
docstring). So the real question a bf16 StateManager would introduce isn't
"bf16 arithmetic" -- it's a STORAGE round-trip: state gets downcast to bf16
between decode steps, then upcast back to fp32 each time it's read. This
test simulates exactly that round-trip via `.bfloat16().float()` between
steps, with NO changes to the production forward path -- a pure test-
harness-level simulation of what StateManager would do if `self.states`
were bf16 instead of fp32.

Compares, run 200 consecutive decode steps at N=1 (the worst case for
compounding -- no batching to dilute a single sequence's own drift):
  - fp32 baseline: state carried at full fp32 precision, no round-trip.
  - bf16-roundtrip: state round-tripped through bf16 after every step.
Tracks, per step: cosine similarity of the output vs fp32 baseline, max abs
diff, state norm (both paths, to catch divergence/collapse), and whether
the actual argmax (would-be sampled channel) agrees.

Looks specifically for the FAILURE MODE the inherited warning describes:
does divergence GROW over the run (real compounding, unsafe) or plateau
(bounded rounding noise, safe)? A flat, bounded error is expected and fine
for ANY reduced-precision storage -- what would actually validate the
warning is a clearly growing trend or a state-norm blowup/collapse.

Also runs a second, shorter pass at the REAL checkpoint's dimensions
(hidden_size=2048, linear_num_key_heads=16, linear_num_value_heads=32,
linear_*_head_dim=128, linear_conv_kernel_dim=4 -- from qwen35_checkpoint/
config.json) to confirm the small-test-dims result generalizes, not just an
artifact of the toy size test_qwen35_gdr_decode_batched.py uses for speed.

Pure CPU. Run with TORCHDYNAMO_DISABLE=1.

Usage:
    TORCHDYNAMO_DISABLE=1 python tests/test_gdr_state_bf16_ablation_cpu.py
"""
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import types  # noqa: E402
if "nanovllm" not in sys.modules:
    pkg = types.ModuleType("nanovllm")
    pkg.__path__ = [ROOT]
    pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = pkg

from test_utils import init_model_weights_with_norms_  # noqa: E402


def _init_dist():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29534")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")


def _build(H, LKH, LVH, LHD, CK, EPS=1e-6, seed=0):
    from nanovllm.models.qwen3_5 import Qwen35LinearAttention
    torch.manual_seed(seed)
    m = Qwen35LinearAttention(
        hidden_size=H, linear_attn_kq_heads=LKH, linear_attn_v_heads=LVH,
        linear_attn_head_dim=LHD, conv_kernel_size=CK, rms_norm_eps=EPS,
        use_batched_gdr_decode=True,
    )
    init_model_weights_with_norms_(m, seed=seed + 1)
    return m.to(torch.float32).eval()


def _decode(m, hs, states, conv_states):
    N = hs.shape[0]
    cu = torch.arange(0, N + 1, dtype=torch.int32)
    with torch.no_grad():
        return m(hs, cu, states=states, conv_states=conv_states)


def run_ablation(H, LKH, LVH, LHD, CK, num_steps, seed, label):
    print(f"\n{'=' * 78}\n{label}  (H={H} LKH={LKH} LVH={LVH} LHD={LHD} CK={CK}, "
          f"{num_steps} steps, N=1)\n{'=' * 78}")
    m = _build(H, LKH, LVH, LHD, CK, seed=seed)
    qkv_dim = m.qkv_dim

    st_fp32 = torch.zeros(1, LVH, LHD, LHD)
    cv_fp32 = torch.zeros(1, qkv_dim, CK - 1)
    st_bf16 = st_fp32.clone()
    cv_bf16 = cv_fp32.clone()

    cosines, diffs, argmax_matches = [], [], []
    norm_fp32_hist, norm_bf16_hist = [], []

    for step in range(num_steps):
        torch.manual_seed(9000 + seed * 100000 + step)
        hs = torch.randn(1, H)

        y_fp32, st_fp32, cv_fp32 = _decode(m, hs, st_fp32, cv_fp32)
        y_bf16, st_bf16, cv_bf16 = _decode(m, hs, st_bf16, cv_bf16)

        # Simulate a bf16 StateManager: round-trip the state (not the conv
        # state -- that's already model-dtype in production, unaffected by
        # this specific question) through bf16 storage before next step.
        st_bf16 = st_bf16.to(torch.bfloat16).to(torch.float32)

        cos = F.cosine_similarity(y_fp32.float().flatten().unsqueeze(0),
                                  y_bf16.float().flatten().unsqueeze(0), dim=-1).item()
        d = (y_fp32.float() - y_bf16.float()).abs().max().item()
        am_match = torch.equal(y_fp32.argmax(dim=-1), y_bf16.argmax(dim=-1))

        cosines.append(cos)
        diffs.append(d)
        argmax_matches.append(am_match)
        norm_fp32_hist.append(st_fp32.float().norm().item())
        norm_bf16_hist.append(st_bf16.float().norm().item())

        if step < 5 or step % max(1, num_steps // 10) == 0 or step == num_steps - 1:
            print(f"    step {step:4d}: cosine={cos:.8f}  max_abs_diff={d:.3e}  "
                  f"argmax_match={am_match}  |state|_fp32={norm_fp32_hist[-1]:.4f}  "
                  f"|state|_bf16={norm_bf16_hist[-1]:.4f}")

    # Trend check: compare mean cosine of the first vs last quarter of the
    # run. Real compounding should show a clear downward trend across that
    # split; bounded rounding noise should not.
    q = max(1, num_steps // 4)
    first_q_cos = sum(cosines[:q]) / q
    last_q_cos = sum(cosines[-q:]) / q
    first_q_norm_ratio = sum(b / f for b, f in zip(norm_bf16_hist[:q], norm_fp32_hist[:q])) / q
    last_q_norm_ratio = sum(b / f for b, f in zip(norm_bf16_hist[-q:], norm_fp32_hist[-q:])) / q

    match_rate = sum(argmax_matches) / len(argmax_matches)
    min_cos = min(cosines)

    print(f"\n  Summary over {num_steps} steps:")
    print(f"    cosine: min={min_cos:.8f}  first-quarter mean={first_q_cos:.8f}  "
          f"last-quarter mean={last_q_cos:.8f}  (trend={'DOWN' if last_q_cos < first_q_cos - 1e-4 else 'flat/up'})")
    print(f"    |state_bf16|/|state_fp32| ratio: first-quarter={first_q_norm_ratio:.6f}  "
          f"last-quarter={last_q_norm_ratio:.6f}  (should stay ~1.0 if no divergence/collapse)")
    print(f"    argmax match rate: {match_rate:.4f} ({sum(argmax_matches)}/{len(argmax_matches)})")

    return {
        "min_cos": min_cos,
        "first_q_cos": first_q_cos,
        "last_q_cos": last_q_cos,
        "match_rate": match_rate,
        "norm_ratio_drift": abs(last_q_norm_ratio - first_q_norm_ratio),
    }


def main():
    _init_dist()

    results = []
    # Small test dims (matches test_qwen35_gdr_decode_batched.py's convention),
    # long run, 3 seeds -- fast enough to actually push step count high.
    for seed in (0, 1, 2):
        results.append(run_ablation(
            H=256, LKH=4, LVH=8, LHD=32, CK=4, num_steps=200, seed=seed,
            label=f"Small test dims, seed={seed}",
        ))

    # Real checkpoint dims (qwen35_checkpoint/config.json), shorter run --
    # confirms the small-dims result isn't a toy-size artifact.
    results.append(run_ablation(
        H=2048, LKH=16, LVH=32, LHD=128, CK=4, num_steps=64, seed=42,
        label="REAL checkpoint dims",
    ))

    print(f"\n{'=' * 78}\nOVERALL VERDICT\n{'=' * 78}")
    ok = True
    for i, r in enumerate(results):
        # Compounding-error red flags: cosine trending down beyond noise,
        # or the bf16/fp32 state-norm ratio drifting away from 1.0 over
        # the run (divergence or collapse).
        trend_bad = r["last_q_cos"] < r["first_q_cos"] - 1e-3
        norm_bad = r["norm_ratio_drift"] > 0.05
        run_ok = not trend_bad and not norm_bad
        ok &= run_ok
        print(f"  run {i}: min_cos={r['min_cos']:.6f}  argmax_match_rate={r['match_rate']:.4f}  "
              f"trend_bad={trend_bad}  norm_ratio_drift={r['norm_ratio_drift']:.4f}  "
              f"norm_bad={norm_bad}  -> {'OK' if run_ok else 'FLAG'}")

    print("\n" + "-" * 78)
    if ok:
        print("PASS -- no compounding-error signature found across any run.")
        print("  bf16 state round-tripping shows bounded, non-growing rounding noise,")
        print("  not the 'compounds into degenerate repetition' failure mode the")
        print("  (unverifiable, untraceable in this repo) inherited warning describes.")
        print("  This does NOT mean bf16 state is production-safe on its own -- it means")
        print("  the specific failure mode described in the warning was not reproduced")
        print("  here. A real GSM8K run over a full generation length is still the gate")
        print("  before shipping this, same discipline as every other precision change")
        print("  in this project.")
    else:
        print("FLAGGED -- at least one run showed a growing-divergence or state-norm")
        print("  drift signature. This is evidence the inherited warning may be real --")
        print("  do NOT proceed with bf16 GDR state without investigating the flagged")
        print("  run(s) further.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
