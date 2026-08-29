#!/usr/bin/env bash
# Generate nsys traces for a fixed set of "known-good" commits, so throughput
# can be compared across the project's evolution on the SAME 4xH100 box.
#
# Run this ON the GPU box (never over an assistant-driven SSH session -- see
# this project's own "give commands, don't connect directly" convention).
# It never touches your current checkout: every commit is built in its own
# `git worktree`, so uncommitted work-in-progress in the main checkout
# (e.g. the lm_head-in-graph edits) is never at risk.
#
# Picked commits (see PICKED_MILESTONES.md for the full rationale for each):
#   db148fb  "Lock in the verified fused-kernel result (172.7 tok/s)"      -- tp=2 (pre-dates the tp=1 INT8 unblock)
#   eec3fe0  "Unblock tp=1 INT8, add True W8A8 (Hopper FP8) + lm_head INT8" -- tp=2 (conservative; see note below)
#   57e4373  first real H100 window closed out: 4 bugs fixed + kernel retune, 206.2 tok/s, GSM8K 95.91% PASS -- tp=1
#   b891eae  CPU window close-out: batched-gdr-decode + fused-gdr-decode-kernel built, CPU-verified, "GPU-ready" -- tp=1, 2 variants
#   1124c2d  current HEAD (main) -- tp=1, 2 variants
#
# Usage:
#   chmod +x profile_milestones.sh
#   ./profile_milestones.sh                 # run everything
#   KEEP_WORKTREES=1 ./profile_milestones.sh  # keep worktrees after each run (debugging)
#   RUNS_FILTER=1124c2d ./profile_milestones.sh  # only run entries whose label contains this substring

set -uo pipefail   # NOT -e: one commit failing to boot must not abort the whole batch

REPO_DIR="$HOME/qLLM"
CKPT_DIR="$REPO_DIR/qwen35_checkpoint"
WORKTREE_ROOT="$HOME/qLLM_profile_worktrees"
OUT_ROOT="$HOME/qLLM_profile_traces/$(date +%Y-%m-%d)"
PORT=8000
# 30 min, not 10: SESSION_HANDOFF_2026-08-28.md documents multi-bucket graph
# capture already taking "minutes for 10-12 buckets" on the FIXED decode
# path. db148fb/eec3fe0 (2 of these 7 runs) still have the UNFIXED O(batch)
# per-segment GDR loop that the rest of this project's history exists to
# fix -- the CPU window measured that loop unrolling into ~30,000 near-empty
# kernel launches per decode step at concurrency 64 once captured. A
# too-short timeout here would kill a server that's still legitimately
# capturing and report a false FAILED, wasting the boot for nothing.
HEALTH_TIMEOUT_S=1800
BENCH_LEVELS=64           # production never scales past concurrency 64 on this project -- see memory
KEEP_WORKTREES="${KEEP_WORKTREES:-0}"
RUNS_FILTER="${RUNS_FILTER:-}"

mkdir -p "$WORKTREE_ROOT" "$OUT_ROOT"
cd "$REPO_DIR" || { echo "FATAL: $REPO_DIR not found -- run setup.sh first" >&2; exit 1; }

# Clears any worktree registrations left dangling by an earlier interrupted
# run (e.g. this script got killed mid-run, or a worktree directory was
# removed by hand without `git worktree remove`) -- confirmed by testing:
# without this, `git worktree add` on a path git still half-remembers fails
# with "is a missing but already registered worktree", and that failure
# would otherwise cost a full manual `git worktree prune` + re-run cycle in
# the middle of a session where that turnaround is expensive.
git worktree prune -v

if ! command -v nsys >/dev/null 2>&1; then
    echo "FATAL: nsys (Nsight Systems CLI) not found on PATH." >&2
    echo "  Usually ships with the CUDA toolkit at /usr/local/cuda-*/bin/nsys, or install with:" >&2
    echo "  sudo apt-get install -y nsight-systems-cli   (package name varies by CUDA version)" >&2
    exit 1
fi
if [ ! -d "$CKPT_DIR" ]; then
    echo "FATAL: checkpoint not found at $CKPT_DIR (run setup.sh's download step first)" >&2
    exit 1
fi
source "$REPO_DIR/cuda_env.sh" 2>/dev/null || true   # CUDA_HOME/PATH/LD_LIBRARY_PATH, if the box has it (see project_qllm_box_setup_flash_attn_cuda13 memory)
if [ ! -f "$REPO_DIR/.venv/bin/activate" ]; then
    echo "FATAL: $REPO_DIR/.venv/bin/activate not found -- run setup.sh first." >&2
    exit 1
fi
source "$REPO_DIR/.venv/bin/activate"

# ── run table: sha|label|tp|gpu_ids|gmu|extra_env|extra_server_flags ──────
# One shared venv, one shared checkpoint (symlinked into each worktree, NOT
# copied -- 5 commits x 67GB would blow most rental boxes' root disk, exactly
# the failure mode setup.sh's own checkpoint step already guards against).
#
# extra_env matters for db148fb/eec3fe0 specifically: at those commits there
# is NO --fused-moe-kernel CLI flag yet (confirmed via `git show <sha>:src/
# server.py | grep add_argument` -- it doesn't exist before 57e4373). The
# fused Triton MoE kernel is instead gated by `_USE_FUSED_MOE_KERNEL =
# os.environ.get("NANOVLLM_USE_FUSED_MOE_KERNEL", "0") == "1"`, a MODULE-
# IMPORT-TIME statement in models/qwen3_5.py -- it MUST be in the process
# environment before `python src/server.py` starts, or db148fb (whose whole
# point is "lock in the verified FUSED-kernel result") would silently run
# the slow gather+dequant+einsum path instead, quietly measuring the wrong
# thing rather than crashing. Confirmed identical gating in eec3fe0 too.
RUNS=(
"db148fb|db148fb_fused_int8_kernel|2|0,1|0.85|NANOVLLM_USE_FUSED_MOE_KERNEL=1|--moe-w8a8"
"eec3fe0|eec3fe0_w8a8_hopper_lmhead_int8|2|0,1|0.85|NANOVLLM_USE_FUSED_MOE_KERNEL=1|--moe-w8a8"
"57e4373|57e4373_h100_4bugs_kernel_retune|1|0|0.70||--moe-w8a8 --fused-moe-kernel --vectorized-moe --fused-gdr-kernel"
"b891eae|b891eae_batched_gdr_decode|1|0|0.70||--moe-w8a8 --fused-moe-kernel --vectorized-moe --fused-gdr-kernel --batched-gdr-decode"
"b891eae|b891eae_fused_gdr_decode_kernel|1|0|0.70||--moe-w8a8 --fused-moe-kernel --vectorized-moe --fused-gdr-kernel --fused-gdr-decode-kernel"
"1124c2d|1124c2d_HEAD_batched_gdr_decode|1|0|0.70||--moe-w8a8 --fused-moe-kernel --vectorized-moe --fused-gdr-kernel --batched-gdr-decode"
"1124c2d|1124c2d_HEAD_fused_gdr_decode_kernel|1|0|0.70||--moe-w8a8 --fused-moe-kernel --vectorized-moe --fused-gdr-kernel --fused-gdr-decode-kernel"
)
# Note on eec3fe0 at tp=2: its own commit message says the tp=1 INT8 fix was
# "CPU-validated, GPU-unconfirmed" at the time it landed. tp=2 is the
# conservative choice here; if you want to test the tp=1 claim itself,
# override the "2|0,1" field to "1|0" for that one line.

wait_for_health() {
    local port="$1" timeout_s="$2" waited=0
    while [ "$waited" -lt "$timeout_s" ]; do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    return 1
}

smoke_test() {
    local port="$1"
    curl -sf -X POST "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":16,"temperature":0.0}' \
        >/dev/null 2>&1
}

stop_server() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        # SIGINT, not SIGTERM: nsys's own documented graceful-stop path is
        # Ctrl-C (SIGINT) -- that's what makes it stop the target AND
        # finalize the .nsys-rep. SIGTERM's behavior against the nsys
        # wrapper process itself isn't documented the same way, and this is
        # exactly the spot where guessing wrong is expensive: losing the
        # trace after the full multi-minute bench has already run, not just
        # a quick retry. 120s budget before escalating -- finalizing a trace
        # with thousands of captured kernels can take real, non-trivial time.
        kill -INT "$pid" 2>/dev/null
        local waited=0
        while [ "$waited" -lt 120 ]; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 2
            waited=$((waited + 2))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "  [WARN] nsys/server still alive ${waited}s after SIGINT -- escalating to SIGTERM then SIGKILL (trace may be incomplete/corrupt if it comes to this)"
            kill -TERM "$pid" 2>/dev/null
            sleep 10
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    pkill -9 -f "src/server.py" 2>/dev/null || true
    sleep 3   # port 2333 (NCCL init) release -- documented gotcha, EADDRINUSE otherwise

    # Verify the port is ACTUALLY free before returning -- if something is
    # still bound to it, the NEXT run's wait_for_health would burn its full
    # timeout guessing why a brand-new server never comes up, instead of
    # surfacing the real cause immediately.
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "  [WARN] port ${PORT} still answering after stop_server -- something didn't die. Check 'ps aux | grep server.py' before the next run."
    fi
}

run_one() {
    local sha="$1" label="$2" tp="$3" gpu_ids="$4" gmu="$5" extra_env="$6" flags="$7"
    local wt_dir="$WORKTREE_ROOT/$label"
    local out_dir="$OUT_ROOT/$label"
    mkdir -p "$out_dir"

    echo "=================================================================="
    echo "[$label] commit=$sha  tp=$tp  gpus=$gpu_ids  gmu=$gmu"
    echo "  flags: $flags"
    echo "=================================================================="

    if [ ! -d "$wt_dir" ]; then
        if ! git worktree add "$wt_dir" "$sha"; then
            # Self-heal once: a prior run earlier in THIS SAME script
            # execution (not just a previous invocation -- the startup
            # prune only catches staleness from before the script started)
            # can leave a stale registration if `git worktree remove`
            # failed silently for the same label. Retry once after pruning
            # before giving up, rather than losing the whole run to
            # something a single extra command fixes.
            echo "[$label] git worktree add failed, retrying once after 'git worktree prune'..."
            git worktree prune -v
            git worktree add "$wt_dir" "$sha" || { echo "[$label] FAILED: git worktree add (after prune retry)"; return 1; }
        fi
    else
        # Reusing a directory left over from an earlier interrupted run (e.g.
        # `git worktree remove` failed silently because a killed GPU process
        # still held an open file handle). Verify it's actually checked out
        # at the commit we think it is before trusting it -- silently running
        # the wrong commit's code would be a much worse failure mode than a
        # loud one here, since nothing else in this script would catch it.
        local want_sha have_sha
        want_sha=$(git rev-parse "$sha") || { echo "[$label] FAILED: cannot resolve $sha"; return 1; }
        have_sha=$(git -C "$wt_dir" rev-parse HEAD 2>/dev/null) || have_sha=""
        if [ "$want_sha" != "$have_sha" ]; then
            echo "[$label] FAILED: existing worktree at $wt_dir is checked out at ${have_sha:-<unknown>}, expected $want_sha -- remove it manually (git worktree remove --force '$wt_dir') and re-run."
            return 1
        fi
        echo "[$label] reusing existing worktree (verified at $want_sha)"
    fi
    # rm -rf FIRST, unconditionally, before symlinking the checkpoint in.
    # Confirmed by testing this exact sequence locally: db148fb (unlike the
    # other 4 picks) still git-tracks 13 small files under qwen35_checkpoint/
    # (tokenizer.json, config.json, etc. -- only *.safetensors was gitignored
    # that early; the whole directory wasn't gitignored until eec3fe0). `git
    # worktree add` materializes those into a REAL, non-empty directory at
    # that path, and `ln -sfn` onto an existing real (non-symlink) directory
    # does NOT replace it -- it silently no-ops or nests the link inside it,
    # depending on the ln implementation -- either way the server would then
    # load a checkpoint dir with only tokenizer/config files and no weights.
    rm -rf "$wt_dir/qwen35_checkpoint"
    ln -sfn "$CKPT_DIR" "$wt_dir/qwen35_checkpoint"

    local srv_log="$out_dir/server.log"
    local rep_base="$out_dir/${label}"   # nsys appends .nsys-rep

    (
        cd "$wt_dir" || exit 1
        # No --kill none here (unlike the interactive sessions in the session
        # handoffs, which used it to let a server outlive a --duration-boxed
        # capture for further un-traced bench trials): this script signals
        # nsys itself once the bench finishes (see stop_server), and WANTS
        # nsys's default behavior of propagating that to the server so it
        # exits and nsys can finalize the .nsys-rep cleanly.
        # `env` is NOT decorative here -- confirmed by testing directly: bash
        # only recognizes a leading `NAME=value` TOKEN as an env-assignment
        # prefix; a variable EXPANSION in that position (`$extra_env ...`,
        # even when non-empty) does not qualify, and once one word in a
        # simple command isn't an assignment, nothing after it is treated as
        # one either -- `$extra_env CUDA_VISIBLE_DEVICES=... nsys ...` would
        # make "CUDA_VISIBLE_DEVICES=..." itself the (nonexistent) command
        # name whenever $extra_env is empty, i.e. for 5 of these 7 runs.
        # `env` sidesteps this: it's a real command that takes NAME=value
        # pairs as plain arguments, so $extra_env can safely be empty or not.
        env $extra_env CUDA_VISIBLE_DEVICES="$gpu_ids" PYTHONUNBUFFERED=1 \
        nsys profile -o "$rep_base" --force-overwrite=true --cuda-graph-trace=node \
            python src/server.py \
                --model qwen35_checkpoint \
                --port "$PORT" \
                --tensor-parallel-size "$tp" \
                --cuda-graphs \
                --concurrency-mode batched \
                --max-num-seqs 64 \
                --gpu-memory-utilization "$gmu" \
                $flags \
                > "$srv_log" 2>&1 &
        echo $! > "$out_dir/server.pid"
    )
    local srv_pid
    srv_pid=$(cat "$out_dir/server.pid" 2>/dev/null)
    if [ -z "$srv_pid" ]; then
        echo "[$label] FAILED: server did not start (see $srv_log)"
        return 1
    fi

    if ! wait_for_health "$PORT" "$HEALTH_TIMEOUT_S"; then
        echo "[$label] FAILED: /health never came up within ${HEALTH_TIMEOUT_S}s -- tail of $srv_log:"
        tail -40 "$srv_log"
        stop_server "$srv_pid"
        return 1
    fi

    if ! smoke_test "$PORT"; then
        echo "[$label] FAILED: smoke chat-completion request failed -- tail of $srv_log:"
        tail -40 "$srv_log"
        stop_server "$srv_pid"
        return 1
    fi
    echo "[$label] server up, smoke test passed -- running bench + nsys capture"

    (
        cd "$wt_dir" || exit 1
        python bench_http_concurrency.py \
            --base-url "http://localhost:${PORT}" \
            --tokenizer-dir qwen35_checkpoint \
            --levels "$BENCH_LEVELS" \
            --prompt-tokens 128 --max-tokens 1024 \
            --trials 3 --warmup-trials 1 --ignore-eos \
            --out "$out_dir/bench.csv" \
            > "$out_dir/bench.log" 2>&1
    )
    local bench_rc=$?
    if [ "$bench_rc" -ne 0 ]; then
        echo "[$label] WARNING: bench_http_concurrency.py exited $bench_rc -- see $out_dir/bench.log (trace may still be usable up to the failure point)"
    fi

    stop_server "$srv_pid"

    if [ -f "${rep_base}.nsys-rep" ]; then
        # .nsys-rep files are large and don't reliably survive being copied off
        # a rental box (documented in SESSION_HANDOFF_2026-08-28.md) -- the
        # text stats are the portable record; keep the .nsys-rep too for
        # interactive nsys-ui inspection while still on this box.
        nsys stats --report cuda_gpu_kern_sum "${rep_base}.nsys-rep" > "$out_dir/${label}_kernel_summary.txt" 2>&1
        nsys stats --report cuda_api_sum "${rep_base}.nsys-rep" > "$out_dir/${label}_api_summary.txt" 2>&1
        echo "[$label] trace: ${rep_base}.nsys-rep  (+ kernel/api summary .txt in $out_dir)"
    else
        echo "[$label] WARNING: no .nsys-rep produced -- check $srv_log and nsys's own stderr above"
    fi

    if [ "$KEEP_WORKTREES" != "1" ]; then
        git worktree remove --force "$wt_dir" 2>/dev/null || true
    fi
    return 0
}

# ── main loop ────────────────────────────────────────────────────────────
FAILED=()
for entry in "${RUNS[@]}"; do
    IFS='|' read -r sha label tp gpu_ids gmu extra_env flags <<< "$entry"
    if [ -n "$RUNS_FILTER" ] && [[ "$label" != *"$RUNS_FILTER"* ]]; then
        continue
    fi
    if ! run_one "$sha" "$label" "$tp" "$gpu_ids" "$gmu" "$extra_env" "$flags"; then
        FAILED+=("$label")
    fi
done

# ── summary table (tok/s at concurrency 64, from each run's bench.csv) ────
SUMMARY="$OUT_ROOT/summary.md"
{
    echo "# Milestone profiling summary — $(date +%Y-%m-%d)"
    echo
    echo "| label | commit | tok/s @ c${BENCH_LEVELS} | trace |"
    echo "|---|---|---|---|"
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r sha label tp gpu_ids gmu extra_env flags <<< "$entry"
        blog="$OUT_ROOT/$label/bench.log"
        toks="n/a"
        if [ -f "$blog" ]; then
            # bench_http_concurrency.py prints "concurrency=N: mean tok/s over
            # K trials = X.X" once the level completes -- pull X.X from there
            # rather than the CSV (whose per-trial rows would need averaging,
            # and whose last COLUMN is prompt_tokens_target, not tok_s).
            toks=$(grep -oE "mean tok/s over [0-9]+ trials = [0-9.]+" "$blog" | tail -1 | grep -oE "[0-9.]+$")
            [ -z "$toks" ] && toks="n/a (see bench.log)"
        fi
        echo "| $label | $sha | $toks | $label/${label}.nsys-rep |"
    done
} > "$SUMMARY"

echo
echo "=================================================================="
cat "$SUMMARY"
echo "=================================================================="
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED runs (see server.log / bench.log under $OUT_ROOT/<label>/): ${FAILED[*]}"
fi
echo "All artifacts under: $OUT_ROOT"
