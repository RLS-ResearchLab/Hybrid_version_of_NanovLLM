"""GSM8K correctness run against a LIVE src/server.py, over HTTP -- the only
harness in this repo that scores GSM8K through the ACTUAL shipping decode
path (whatever flags the running server was launched with:
--vectorized-moe's INT8 dispatch branch, --fused-moe-kernel's retuned
Triton config, --cuda-graphs, --concurrency-mode batched, and
--fused-gdr-kernel if enabled).

Every other GSM8K script here (gsm8k_full_run.py, gsm8k_smoke_test.py, the
cluster_*_gsm8k.py pair) builds its own in-process LLM(...) and never sets
use_vectorized_moe or NANOVLLM_USE_FUSED_MOE_KERNEL, so none of them touch
_forward_dispatch_vectorized's INT8 branch or fused_moe_int8._DEFAULT_CONFIG
-- exactly the code paths changed this session. This script closes that gap
by talking to the server that IS running those paths.

WHAT IT REUSES, NOT REINVENTS
  - gsm8k_prompt.build_chat_messages(): the single-user-turn, 8-shot CoT
    prompt. The server itself applies apply_chat_template(...,
    enable_thinking=False) + add_generation_prompt (src/server.py's
    chat_completions), so this reproduces the exact 'chat-no-think' format
    that produced the 95.30% GSM8K result -- matched to
    cluster_q6_moe_w8a8_gsm8k.py's _run_arm_chat_no_think, not a new format.
  - gsm8k_extract.extract_answer_detailed() + GSM8K_STOP_PATTERNS: identical
    extraction and (regex) stop-string set as every other scored run here.
  - Record shape: identical to gsm8k_full_run.py's -- so
    `python tests/gsm8k_score.py --results <this file> --expected-total N`
    scores it directly with no adapter.

CHECKPOINTED + RESUMABLE, same as gsm8k_full_run.py: each result is appended
to the jsonl and flushed immediately; re-running skips any idx already
present. Safe to Ctrl-C and resume.

SAMPLING: temperature=0 (deterministic argmax server-side), max_tokens=512
(gsm8k_full_run.py's justified default -- gsm8k_answer_position_check.py
found truncation risk, not headroom, below that). Stop strings ON by default
(the 95.30% run's config); --no-stop-strings to disable.

CONCURRENCY: --concurrency N fires N requests at once against the server's
continuous-batching engine. Correctness is temperature=0 argmax and must not
depend on batch composition -- if a --concurrency 1 run and a --concurrency
32 run disagree on any example, that is itself a finding (decode
cross-sequence contamination), not noise. Keep --concurrency <= the server's
--max-num-seqs (reported by /health, checked below) or the tail just queues.

Usage:
    # server already running in another terminal, e.g.:
    #   python src/server.py --model qwen35_checkpoint --tensor-parallel-size 1 \
    #     --moe-w8a8 --fused-moe-kernel --vectorized-moe --cuda-graphs \
    #     --max-num-seqs 64 --gpu-memory-utilization 0.70 --concurrency-mode batched

    # quick sanity subset (fixed seed -> same 40 every time):
    python tests/gsm8k_http_run.py --num-examples 40 --concurrency 8
    python tests/gsm8k_score.py \
        --results tests/_gsm8k_cache/http_n40_c8_stop.jsonl --expected-total 40

    # full test set:
    python tests/gsm8k_http_run.py --concurrency 32
    python tests/gsm8k_score.py \
        --results tests/_gsm8k_cache/http_full_c32_stop.jsonl --expected-total 1319
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm8k_prompt import build_chat_messages  # noqa: E402
from gsm8k_extract import extract_answer_detailed, GSM8K_STOP_PATTERNS  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "tests", "_gsm8k_cache")
EXPECTED_TOTAL = 1319
SUBSAMPLE_SEED = 42  # matches gsm8k_full_run.py / cluster_a4 convention -- same subset every run


def _select_indices(total: int, num_examples) -> list[int]:
    if num_examples is None:
        return list(range(total))
    assert 0 < num_examples <= total, f"--num-examples must be in (0, {total}], got {num_examples}"
    rng = random.Random(SUBSAMPLE_SEED)
    all_indices = list(range(total))
    rng.shuffle(all_indices)
    return sorted(all_indices[:num_examples])


def _results_path(num_examples, concurrency: int, max_tokens: int, stop_strings: bool) -> str:
    base = "http_full" if num_examples is None else f"http_n{num_examples}"
    suffix = f"_c{concurrency}"
    if max_tokens != 512:
        suffix += f"_mt{max_tokens}"
    if stop_strings:
        suffix += "_stop"
    return os.path.join(CACHE_DIR, f"{base}{suffix}.jsonl")


def _load_completed(path: str) -> dict:
    completed = {}
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: skipping malformed line {line_no} in {path} "
                      f"(likely a crash mid-write) -- {line[:80]!r}...")
                continue
            completed[record["idx"]] = record
    return completed


def _health_check(base_url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] server not reachable at {base_url}: {e!r}")
        sys.exit(1)
    print(f"Server health: {json.dumps(body)}")
    return body.get("config")


def _post_one(base_url: str, question: str, max_tokens: int, stop, timeout: float, retries: int = 3):
    payload = json.dumps({
        "messages": [{"role": m["role"], "content": m["content"]}
                     for m in build_chat_messages(question)],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": False,
        "stop": stop,
    }).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"], body.get("usage", {})
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} attempts: {last_err!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--num-examples", type=int, default=None,
                    help="Fixed-seed random subsample of this many examples (seed 42 -> same "
                         "subset every run). Omit for the full 1319-example test set.")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Requests in flight at once. Keep <= server --max-num-seqs (checked "
                         "against /health below).")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="Default 512 -- gsm8k_full_run.py's justified value, do not lower "
                         "without re-running gsm8k_answer_position_check.py.")
    ap.add_argument("--stop-strings", action=argparse.BooleanOptionalAction, default=True,
                    help="Default ON (the 95.30%% run's config). --no-stop-strings disables.")
    ap.add_argument("--timeout", type=float, default=600.0, help="Per-request HTTP timeout, seconds.")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    results_path = _results_path(args.num_examples, args.concurrency, args.max_tokens, args.stop_strings)

    server_cfg = _health_check(args.base_url)
    if server_cfg is not None:
        print(f"Server config: {json.dumps(server_cfg)}")
        mns = server_cfg.get("max_num_seqs")
        if server_cfg.get("concurrency_mode") == "fcfs":
            print("[WARNING] server is --concurrency-mode fcfs -- requests fully serialize; this "
                  "still scores correctly but will be slow and won't exercise batched decode.")
        if mns is not None and args.concurrency > mns:
            print(f"[WARNING] --concurrency {args.concurrency} > server --max-num-seqs {mns}: the "
                  f"tail will queue rather than batch. Not a correctness problem, just slower.")

    from datasets import load_dataset
    print("Loading openai/gsm8k (main, test split) ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if len(ds) != EXPECTED_TOTAL:
        print(f"STOP: expected {EXPECTED_TOTAL} test examples, got {len(ds)}. Not proceeding.")
        sys.exit(1)

    selected = _select_indices(len(ds), args.num_examples)
    if args.num_examples is not None:
        print(f"Subsampled {len(selected)}/{len(ds)} examples (seed={SUBSAMPLE_SEED}): {selected}")

    completed = _load_completed(results_path)
    n_correct_total = sum(1 for r in completed.values() if r.get("correct"))
    print(f"Resuming: {len(completed)}/{len(selected)} already completed in {results_path}")
    remaining = [i for i in selected if i not in completed]
    if not remaining:
        print("All examples already completed -- run tests/gsm8k_score.py to see results.")
        return

    stop = GSM8K_STOP_PATTERNS if args.stop_strings else None
    write_lock = threading.Lock()
    done = 0
    n_err = 0
    t0 = perf_counter()

    def work(idx: int) -> dict:
        ex = ds[idx]
        gold = extract_answer_detailed(ex["answer"])
        assert gold.method == "hash", f"example {idx}: gold answer lacks '#### N': {ex['answer']!r}"
        text, _usage = _post_one(args.base_url, ex["question"], args.max_tokens, stop, args.timeout)
        model_result = extract_answer_detailed(text)
        correct = (model_result.value is not None
                   and abs(model_result.value - gold.value) < 1e-6)
        return {
            "idx": idx,
            "question": ex["question"],
            "gold_answer": gold.value,
            "model_output": text,
            "extracted_answer": model_result.value,
            "extraction_method": model_result.method,
            "correct": correct,
        }

    with open(results_path, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(work, i): i for i in remaining}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                record = fut.result()
            except Exception as e:  # noqa: BLE001 -- one bad request must not abandon a long run
                n_err += 1
                print(f"  [ERROR] idx={idx}: {e!r} -- left unscored, re-run this script to retry it")
                continue
            with write_lock:
                f.write(json.dumps(record) + "\n")
                f.flush()
            done += 1
            if record["correct"]:
                n_correct_total += 1
            total_done = len(completed) + done
            elapsed = perf_counter() - t0
            eta = (elapsed / done) * (len(selected) - total_done)
            status = "PASS" if record["correct"] else "FAIL"
            print(f"  [{status}] idx={idx} gold={record['gold_answer']} "
                  f"extracted={record['extracted_answer']} (method={record['extraction_method']})  "
                  f"| {total_done}/{len(selected)} running {n_correct_total}/{total_done} "
                  f"({100 * n_correct_total / total_done:.1f}%) | ETA {eta / 60:.1f}m")

    print("\nRUN COMPLETE" + (f" -- {n_err} request(s) errored and were left unscored; "
                              f"re-run to retry them" if n_err else ""))
    print(f"Results in {results_path}")
    print(f"Score it:\n  python tests/gsm8k_score.py --results {results_path} "
          f"--expected-total {len(selected)}")


if __name__ == "__main__":
    main()
