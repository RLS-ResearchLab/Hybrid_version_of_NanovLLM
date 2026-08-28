# Picked commits for 4xH100 milestone profiling

Selection criteria: a descriptive, non-"debugging" commit message AND either
(a) an explicit correctness/perf verification recorded in the commit message
itself, or (b) a session handoff doc that independently confirms the state
worked on real hardware (or, for the newest one, is CPU-proven and is what
this profiling run exists to validate). Commits with messages like
"debugging"/"kerenl u win this time" between two verified points were
skipped on purpose — they're mid-flight, not safe reference states.

| # | commit | date | why it's trustworthy | tp | notes |
|---|---|---|---|---|---|
| 1 | `db148fb` | 2026-08-21 | Commit message itself documents 3 independent correctness checks (kernel cosine, eager GSM8K, real generated text) before locking in 172.7 tok/s — the fused INT8 Triton MoE kernel beating bf16. | 2 | tp=1 INT8 crashes before `eec3fe0` (see below) — use tp=2 for this one. |
| 2 | `eec3fe0` | 2026-08-23 | Unblocks tp=1 INT8, adds Hopper FP8 W8A8 + lm_head INT8. Commit message is explicit about what's proven vs not ("CPU-validated, GPU-unconfirmed" for the tp=1 fix) — profiling this now is exactly how to close that gap. | 2 | Defaulted to tp=2 here too, conservatively, since the tp=1 claim was GPU-unconfirmed at commit time. Override to tp=1 in the run table if you want to test that claim directly. |
| 3 | `57e4373` | 2026-08-27 | End of the project's first real H100 GPU window: `SESSION_HANDOFF_2026-08-27.md` documents 4 independent bugs found+fixed+confirmed on real hardware, a kernel retune, **206.2 tok/s**, and GSM8K 95.91% PASS (1265/1319) via the exact harness this commit adds. | 1 | The single best-evidenced commit in the whole history. |
| 4 | `b891eae` | 2026-08-27 | End of the CPU window that followed: `SESSION_HANDOFF_2026-08-27_cpu_window.md` confirms a clean `git status` at this exact SHA, with `--batched-gdr-decode` and `--fused-gdr-decode-kernel` built and CPU-verified bitwise-identical to the sequential scan. Explicitly called "GPU-ready" — the very next GPU window validated it at 3.3x/5.3x. | 1 | Profiled in **2 variants** (batched vs. fused/fla), matching how it was actually validated. |
| 5 | `1124c2d` | 2026-08-28 | Current tip of `main`. Includes the sampler host-sync removal, batched `StateManager.get_all()`, and the torch.compile experiments (reverted then partially re-added) on top of #4. This is "where the project actually is" right now. | 1 | Also profiled in **2 variants**. Uncommitted work in your current checkout (lm_head-in-graph) is NOT part of this commit — the worktree-based script never touches it. |

## What was deliberately left out

- Everything with a "debugging"/joke commit message between two of the
  picks above — by definition mid-fix, not a state worth trusting for a
  performance number.
- `SESSION_HANDOFF_2026-08-28.md`'s own exact commit (the one that measured
  908/1467 tok/s) couldn't be pinned to one SHA with confidence — that
  session's commits mix code fixes with unrelated cleanup, and the handoff
  itself says several patches were "applied on the box, commit status
  unverified." `b891eae` (its documented starting point) and `1124c2d`
  (today's tip) bracket it instead; the fresh profiling run in
  `profile_milestones.sh` is what re-establishes trustworthy numbers rather
  than trying to archaeologically match the old ones.
- Anything requiring tp=4 specifically (GQA sharding, multi-rank EP
  dispatch) — per your note, that's not required for these picks, and per
  [[project_qllm_4xh100_20k_target]], tp=4 numerics have never been run on
  real hardware yet. Keep that as a separate, later step once these 5
  baselines are in hand — don't conflate "does the code evolution show a
  trend" (this list) with "does tp=4 actually work" (untested, higher risk).
