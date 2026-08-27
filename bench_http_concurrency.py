#!/usr/bin/env python
"""Item 1 (today's A6000 validation session) -- concurrency sweep against
src/server.py's HTTP endpoint under REAL continuous batching
(--concurrency-mode batched), unlike bench_throughput.py which talks to
LLMEngine directly and never goes through Engine._gen_lock / BatchedEngine
at all. This script exists because no committed harness does concurrent
HTTP load-testing against the server -- the earlier flat ~36.7 tok/s FCFS
baseline was measured via ad hoc curl against the default --concurrency-mode
fcfs server, not via any script in this repo.

For each concurrency level N in --levels, fires N requests at once (one
"wave"), waits for all to finish, and records aggregate tok/s = sum(reported
completion_tokens across successful responses) / wall_clock for that wave.
Repeats for --trials waves per level (plus --warmup-trials discarded waves
first, matching bench_throughput.py's own warm-up rationale: first-shape
compile/graph-capture cost shouldn't land in a timed trial).

Also does the reported-vs-re-tokenized completion-token spot check: for
--retok-sample responses per level, independently re-encodes the response's
decoded text with the same tokenizer and compares to the server-reported
completion_tokens count.

Prompts are built by tokenizing filler text and trimming to exactly
--prompt-tokens tokens, then decoding back to text -- so the *content* field
is close to the target length, but the server's actual reported
prompt_tokens (chat-template overhead included) is what gets logged, not
assumed.

Usage:
    # 1) start the server in another terminal first:
    python src/server.py --model qwen35_checkpoint --tensor-parallel-size 2 \\
        --gpu-memory-utilization 0.9 --max-num-seqs 8 --concurrency-mode batched

    # 2) run the sweep:
    python bench_http_concurrency.py --tokenizer-dir qwen35_checkpoint \\
        --levels 1 2 4 8 --trials 2 --warmup-trials 1 --ignore-eos \\
        --prompt-tokens 1024 --max-tokens 1024 --out throughput_batched_http.csv

Note on --ignore-eos: without it, the server lets each completion stop at
EOS, so tok/s is measured over whatever (possibly short) length the model
happened to generate -- not a real 1024-out number. Pass --ignore-eos to
force every completion to run the full --max-tokens for an apples-to-apples
throughput measurement.

Stop condition (per today's task): if tok/s does NOT increase with
concurrency, STOP and report -- do not keep sweeping past whatever level
first shows this, and do not proceed to Items 2/3.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_prompt_text(tokenizer, target_tokens: int) -> str:
    base = (
        "The history of scientific discovery is filled with unexpected "
        "turns, careful measurement, and revised assumptions. "
    )
    text = base * (target_tokens // 8 + 20)
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


def post_chat_completion(base_url: str, prompt_text: str, max_tokens: int, timeout: float,
                          ignore_eos: bool = False, wave_t0: float = 0.0):
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "ignore_eos": ignore_eos,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    start_offset_s = t0 - wave_t0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t0
        return {"ok": True, "latency_s": dt, "body": body,
                "start_offset_s": start_offset_s, "end_offset_s": start_offset_s + dt}
    except Exception as e:  # noqa: BLE001 -- record failure, don't crash the wave
        dt = time.perf_counter() - t0
        return {"ok": False, "latency_s": dt, "error": repr(e),
                "start_offset_s": start_offset_s, "end_offset_s": start_offset_s + dt}


def run_wave(base_url, prompt_text, max_tokens, concurrency, timeout, ignore_eos=False):
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(post_chat_completion, base_url, prompt_text, max_tokens, timeout, ignore_eos, t0)
            for _ in range(concurrency)
        ]
        for f in as_completed(futs):
            results.append(f.result())
    wall_s = time.perf_counter() - t0

    # Diagnostic added mid-session 2026-08-27: prints each request's own
    # [start_offset, end_offset] within the wave, in submission order -- the
    # direct way to tell "genuinely overlapping" from "back-to-back
    # sequential" apart, rather than inferring it from aggregate wall_s
    # alone (wall_s ~= N * single-request-latency is consistent with BOTH
    # true serialization AND, less obviously, a batched engine that simply
    # isn't speeding up under this workload -- only the per-request
    # overlap picture actually distinguishes them).
    for i, r in enumerate(sorted(results, key=lambda r: r["start_offset_s"])):
        print(f"    req[{i}]: start={r['start_offset_s']:6.2f}s  end={r['end_offset_s']:6.2f}s  "
              f"latency={r['latency_s']:6.2f}s  ok={r['ok']}")

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    total_completion_tokens = sum(r["body"]["usage"]["completion_tokens"] for r in successes)
    tok_s = total_completion_tokens / wall_s if wall_s > 0 else float("nan")
    return {
        "wall_s": wall_s,
        "n_requests": concurrency,
        "n_success": len(successes),
        "n_failure": len(failures),
        "failures": [r["error"] for r in failures],
        "total_completion_tokens": total_completion_tokens,
        "tok_s": tok_s,
        "successes": successes,
    }


def retok_spot_check(tokenizer, successes, sample_size):
    rows = []
    for r in successes[:sample_size]:
        body = r["body"]
        text = body["choices"][0]["message"]["content"]
        reported = body["usage"]["completion_tokens"]
        retokenized = len(tokenizer.encode(text, add_special_tokens=False))
        rows.append({
            "reported_completion_tokens": reported,
            "retokenized_count": retokenized,
            "diff": retokenized - reported,
            "pct_diff": (100.0 * (retokenized - reported) / reported) if reported else float("nan"),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--tokenizer-dir", required=True, help="Path to the checkpoint dir (for tokenizer + re-tok spot check).")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--prompt-tokens", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--trials", type=int, default=2, help="Timed waves per level.")
    p.add_argument("--warmup-trials", type=int, default=1, help="Untimed waves per level, discarded.")
    p.add_argument("--retok-sample", type=int, default=3, help="Responses per level to spot-check reported-vs-retokenized token count.")
    p.add_argument("--timeout", type=float, default=600.0, help="Per-request HTTP timeout, seconds.")
    p.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=False,
                    help="Force every completion to run to --max-tokens instead of stopping at "
                         "EOS. Without this, tok/s reflects whatever (possibly much shorter than "
                         "--max-tokens) length the model happened to stop at, not a real "
                         "--max-tokens-out throughput number -- pass this for throughput sweeps.")
    p.add_argument("--out", default="throughput_batched_http.csv")
    args = p.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.tokenizer_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    print(f"Building ~{args.prompt_tokens}-token prompt text ...")
    prompt_text = build_prompt_text(tokenizer, args.prompt_tokens)
    actual_prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    print(f"Prompt content is {actual_prompt_tokens} tokens before chat-template overhead "
          f"(server's reported usage.prompt_tokens will include that overhead).")

    # Health check before burning any wave on a server that isn't up.
    try:
        with urllib.request.urlopen(f"{args.base_url}/health", timeout=10) as resp:
            health_body = resp.read().decode("utf-8")
            print(f"Server health: {health_body}")
    except Exception as e:
        print(f"[FATAL] server not reachable at {args.base_url}: {e!r}")
        sys.exit(1)

    # Added 2026-08-26, after a real concurrency=8/16/32 sweep produced a
    # near-flat tok/s curve (88.1/93.1/117.7) and a separate concurrency=64
    # run showed 12/64 requests never progressing -- CPU-only simulation of
    # the real Scheduler (tests/diag_scheduler_starvation_cpu.py) could NOT
    # reproduce that via eviction/thrashing, but reproduced the EXACT same
    # shape (a clean tail of later-arriving requests queued, not a bug) from
    # an undersized --max-num-seqs relative to the swept concurrency. This
    # script has no way to see that cap on its own (it's a server-launch
    # flag, not something the concurrency level implies) -- src/server.py's
    # /health now reports it, so check it here instead of silently sweeping
    # past a ceiling this script can't otherwise know about.
    try:
        health = json.loads(health_body)
        server_cfg = health.get("config")
    except (ValueError, NameError):
        server_cfg = None
    if server_cfg is not None:
        max_num_seqs = server_cfg.get("max_num_seqs")
        concurrency_mode = server_cfg.get("concurrency_mode")
        if concurrency_mode == "fcfs":
            print(f"[WARNING] server is running --concurrency-mode fcfs -- every request is fully "
                  f"serialized regardless of --levels here (see src/server.py's Engine docstring). "
                  f"tok/s will NOT scale with concurrency at all. Restart with --concurrency-mode "
                  f"batched to measure real continuous-batching throughput.")
        if max_num_seqs is not None and max(args.levels) > max_num_seqs:
            print(f"[WARNING] server's --max-num-seqs={max_num_seqs} is BELOW the highest "
                  f"--levels value requested here ({max(args.levels)}). Requests beyond "
                  f"{max_num_seqs} in a wave will queue instead of batching -- this produces a "
                  f"flatter-than-real tok/s curve at levels above {max_num_seqs}, and at high "
                  f"enough concurrency can leave a tail of requests still queued when --timeout "
                  f"fires, which looks like scheduler starvation but is really just this cap. "
                  f"Restart the server with --max-num-seqs >= {max(args.levels)} for a result "
                  f"that reflects the engine's real batching capability.")

    fieldnames = [
        "level", "trial_index", "n_requests", "n_success", "n_failure",
        "wall_s", "total_completion_tokens", "tok_s", "prompt_tokens_target",
    ]
    write_header = not os.path.exists(args.out)
    csv_file = open(args.out, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    level_means = {}
    stop_flagged = False

    for level in args.levels:
        print(f"\n=== concurrency={level} ===")

        for w in range(args.warmup_trials):
            res = run_wave(args.base_url, prompt_text, args.max_tokens, level, args.timeout, args.ignore_eos)
            print(f"  warmup {w + 1}/{args.warmup_trials}: wall_s={res['wall_s']:.2f} "
                  f"tok/s={res['tok_s']:.1f} success={res['n_success']}/{res['n_requests']} (discarded)")
            if res["failures"]:
                print(f"    failures: {res['failures']}")

        tok_s_values = []
        for t in range(args.trials):
            res = run_wave(args.base_url, prompt_text, args.max_tokens, level, args.timeout, args.ignore_eos)
            tok_s_values.append(res["tok_s"])
            print(f"  trial {t + 1}/{args.trials}: wall_s={res['wall_s']:.2f} "
                  f"total_tokens={res['total_completion_tokens']} tok/s={res['tok_s']:.1f} "
                  f"success={res['n_success']}/{res['n_requests']}")
            if res["failures"]:
                print(f"    [ANOMALY] {len(res['failures'])} failed request(s): {res['failures']}")

            if res["successes"] and args.retok_sample > 0:
                spot = retok_spot_check(tokenizer, res["successes"], args.retok_sample)
                for i, row in enumerate(spot):
                    print(f"    retok-check[{i}]: reported={row['reported_completion_tokens']} "
                          f"retokenized={row['retokenized_count']} diff={row['diff']:+d} "
                          f"({row['pct_diff']:+.1f}%)")

            writer.writerow({
                "level": level, "trial_index": t, "n_requests": res["n_requests"],
                "n_success": res["n_success"], "n_failure": res["n_failure"],
                "wall_s": res["wall_s"], "total_completion_tokens": res["total_completion_tokens"],
                "tok_s": res["tok_s"], "prompt_tokens_target": args.prompt_tokens,
            })
            csv_file.flush()

        mean_tok_s = sum(tok_s_values) / len(tok_s_values)
        level_means[level] = mean_tok_s
        print(f"  concurrency={level}: mean tok/s over {args.trials} trials = {mean_tok_s:.1f}")

        prior_levels = [l for l in level_means if l < level]
        if prior_levels:
            prev_level = max(prior_levels)
            if level_means[level] <= level_means[prev_level]:
                print(f"\n[STOP CONDITION HIT] tok/s at concurrency={level} "
                      f"({level_means[level]:.1f}) did not increase over concurrency={prev_level} "
                      f"({level_means[prev_level]:.1f}). Per today's task rules: stop here, "
                      f"do not proceed to Items 2/3, report immediately -- this indicates a "
                      f"config/scheduler issue, not just a hardware ceiling.")
                stop_flagged = True
                break

    csv_file.close()
    print(f"\nResults written to {args.out}")
    print(f"\nSummary: {level_means}")
    if stop_flagged:
        print("\n[STOP CONDITION WAS HIT -- see above. Do not proceed to Items 2/3 without diagnosing this first.]")
        sys.exit(2)


if __name__ == "__main__":
    main()
