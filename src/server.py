"""
Minimal OpenAI-compatible server for Qwen3.5-35B-A3B -- hybrid-nano variant.

Implements the same API surface as the basic engine's server.py, so both
can be pointed at by the same throughput harness:
  GET  /health
  POST /v1/chat/completions

Route definitions, request/response schemas, and token-count accounting
below are copied verbatim from the basic engine's src/server.py. Only the
model-invocation layer differs: this server drives nanovllm's `LLM`
(LLM.generate() / SamplingParams, tensor_parallel_size=2) instead of the
basic engine's hand-rolled per-token loop over a raw nn.Module.

Concurrency: THE BASIC ENGINE NEVER BATCHES CONCURRENT REQUESTS.
------------------------------------------------------------------
Read straight from the basic engine's server.py before writing this file:
each request runs in its own thread (`loop.run_in_executor`), and a single
`threading.Lock` (`Engine._gpu_lock`) is acquired and released once PER
TOKEN -- for exactly one forward pass on exactly one sequence -- inside
both the prefill step and every decode-loop iteration. Two concurrent
requests never share a forward pass; the lock just makes them take turns
at token granularity (request A's token N, request B's token M, request
A's token N+1, ...). So the basic engine's strategy is "handle each
request independently, round-robining at the token level, batch size 1
per forward pass" -- not batching, and not a continuous-batching queue
either.

nanovllm's LLMEngine is architected the opposite way. Its
Scheduler.schedule() (engine/scheduler.py) is continuous batching by
design: every decode step gathers ALL currently-`RUNNING` sequences (up
to max_num_seqs) into ONE forward pass --
`while self.running and len(scheduled_seqs) < self.max_num_seqs: ...` --
that's the entire point of the engine, and it's exactly the code path
bench_throughput.py measures when it submits a multi-prompt list to
`engine.generate()`. There is no supported way to call this engine such
that two independently-submitted requests interleave token-by-token
without the scheduler batching them into the same forward pass the
moment both are concurrently `RUNNING`. (Forcing `max_num_seqs=1` doesn't
recover the basic engine's behavior either -- it would make the scheduler
run one sequence to full completion before admitting the next, i.e.
strict FCFS with no interleaving at all, not the basic engine's
round-robin.)

Given that, this server matches the basic engine's actual invariant --
"a forward pass never contains more than one request's tokens" -- rather
than its finer-grained token-level round-robin, which nanovllm's API has
no hook for. A single lock (`Engine._gen_lock`) serialises whole
`LLM.generate()` calls, one prompt at a time, so concurrent HTTP requests
never share a batched forward pass. The cost: they run strictly FCFS
(request B doesn't get a single decode step until request A fully
finishes), not round-robin.

*** METHODOLOGICAL CAVEAT FOR THE THROUGHPUT COMPARISON ***
At concurrency == 1 the two servers are equivalent (one sequence, one
forward pass at a time, either way) and the comparison isolates raw
per-token compute. At concurrency > 1 they are NOT equivalent: the basic
engine's server structurally cannot batch (by construction of its
hand-written loop), while this server *chooses* not to batch only to
match it -- nanovllm's engine is designed to batch, and disabling that
(via the request-level lock instead of letting the scheduler interleave
concurrently-added sequences) is deliberately underselling what this
engine can actually do in production. If the goal is "how fast is the
hybrid architecture's raw decode step," this server is the fair
comparison. If the goal is "how fast can each engine serve real
concurrent traffic," this server's forced serialization is NOT
representative of nanovllm's real capability, and that gap should be
called out rather than left implicit whenever concurrency > 1 numbers
from this server are quoted.

Also note: `ChatRequest.top_p` is kept in the schema for API-surface
parity but is NOT applied here -- nanovllm's `SamplingParams`
(sampling_params.py) and `Sampler` (layers/sampler.py) only support
`temperature` (0 = greedy, otherwise a Gumbel-max equivalent of
temperature sampling). There is no top_p / nucleus-sampling knob in this
engine to wire it to.
"""

import argparse
import asyncio
import atexit
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Bootstrap the "nanovllm" package alias -- identical to the pattern used
# throughout this repo (bench_throughput.py, tests/run_small_model_smoke_test.py)
# since this project's own directory isn't named "nanovllm" but every
# internal module does `from nanovllm.xxx import yyy`.
ROOT = str(Path(__file__).resolve().parent.parent)
_PARENT = str(Path(ROOT).parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if os.path.basename(ROOT) != "nanovllm":
    import types
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

from nanovllm.sampling_params import SamplingParams
# `from nanovllm.llm import LLM` is deliberately NOT imported at module level
# here -- LLM transitively imports models/qwen3_5.py, which reads
# NANOVLLM_USE_FUSED_MOE_KERNEL from os.environ at MODULE IMPORT time (a
# one-shot global, not re-read per call). Importing LLM before main() has
# parsed --fused-moe-kernel and set that env var would freeze the flag off
# regardless of what's passed on the command line. See main()'s local
# import of LLM, right after the env var is set, for the fix.

# ── Fake-config-loader shim (small fixture smoke-testing only) ────────────────
#
# tests/make_fake_hf_config.py writes a FLAT config.json (every field --
# hidden_size, num_hidden_layers, layer_types, etc. -- at the top level).
# That fixture was written back when model_type "qwen3_5_moe" wasn't
# registered with transformers' AutoConfig at all (see bench_throughput.py's
# module docstring / comments), so `_fake_from_pretrained` below (same
# shim bench_throughput.py and the tests/ scripts already use) was needed
# just to load it.
#
# That assumption is now stale: this environment's transformers ships a
# real, registered `Qwen3_5MoeConfig`, which nests text-model fields under
# `text_config` and derives `layer_types` from Qwen3_5MoeTextConfig()'s own
# DEFAULT num_hidden_layers (40) whenever the source JSON has no
# `text_config` key -- exactly the fake fixture's flat shape. The result:
# real `AutoConfig.from_pretrained` silently loads without error, but
# `hf_config.layer_types` ends up a 40-entry list while `num_hidden_layers`
# stays 8 (nanovllm/config.py's flatten step only fills top-level attrs
# that are None, and the flat JSON's top-level `num_hidden_layers=8` is
# already non-None) -- tripping `_get_layer_types`'s
# `assert len(layers_block_type) == num_layers` in models/qwen3_5.py.
# Confirmed by direct repro against `nanovllm.config.Config` alone (no
# CUDA needed): https://github.com -- N/A, reproduced locally in this repo.
#
# This is a config-fixture/transformers-version mismatch, not a server.py
# bug, and not expected to affect a real checkpoint's own (properly
# nested) config.json. `--fake-config-loader` (opt-in, default off) routes
# around it the same way bench_throughput.py does: bypass real AutoConfig
# entirely and read config.json into a plain attribute-accessible dict.
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

class _AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v

def _fake_from_pretrained(path, *args, **kwargs):
    with open(os.path.join(path, "config.json")) as f:
        d = json.load(f)
    d = _AttrDict(d)
    d.dtype = _DTYPE_MAP[d.pop("torch_dtype")]
    return d

# ── Request / response schemas (copied verbatim from the basic engine) ────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "Qwen/Qwen3.5-35B-A3B"
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0  # accepted, not applied -- see module docstring
    ignore_eos: bool = False  # passthrough to SamplingParams -- see sampling_params.py
    stop: list[str] | None = None  # passthrough to SamplingParams -- see sampling_params.py

# ── Engine ────────────────────────────────────────────────────────────────────

class Engine:
    """
    Wraps nanovllm's `LLM` (LLM.generate() / SamplingParams,
    tensor_parallel_size=2). `_gen_lock` serialises whole `generate()`
    calls so concurrent requests never share a batched forward pass --
    the closest match to the basic engine's "never batch concurrent
    requests" invariant that nanovllm's API allows. See the module
    docstring's "Concurrency" section for the full analysis and the
    resulting methodological caveat (FCFS here vs. token-level
    round-robin in the basic engine).
    """
    def __init__(self, llm: "LLM"):  # noqa: F821 -- LLM is imported locally in main(), not at module level; see that import's comment.
        self.llm = llm
        self.tok = llm.tokenizer
        self.eos_id = self.tok.eos_token_id
        self._gen_lock = threading.Lock()

    def generate(self, prompt_token_ids: list[int], max_tokens: int,
                 temperature: float, ignore_eos: bool = False,
                 stop: list[str] | None = None) -> list[int]:
        """Run full generation for one request. Blocks caller thread."""
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                             ignore_eos=ignore_eos, stop=stop)
        with self._gen_lock:
            outputs = self.llm.generate([prompt_token_ids], sp, use_tqdm=False)
        return outputs[0]["token_ids"]


class BatchedEngine:
    """Real continuous batching across concurrent requests. This is the
    default --concurrency-mode as of 2026-08-26 (see main()'s argparse
    setup) -- FCFS (the `Engine` class above) stays available as an
    explicit opt-in, e.g. for a controlled comparison against a simpler
    engine that structurally cannot batch, without reverting any code.

    IMPORTANT -- this is NOT "Engine with the lock deleted". Naively
    removing `_gen_lock` and letting multiple threads each call
    `self.llm.generate(...)` concurrently would be a genuine correctness
    bug: LLMEngine.generate() runs its own `while not self.is_finished():
    self.step()` loop, and `step()` mutates Scheduler.waiting/running
    (plain deques, not thread-safe) and drives the model forward pass --
    two threads calling it concurrently race on that shared state with no
    synchronization at all. The fix isn't a smaller lock around the same
    per-thread-loop shape, it's a different shape: exactly ONE thread
    (the background loop below) ever calls step()/is_finished(), matching
    how single-writer async engines normally dispatch. HTTP-handler
    threads only ever call add_request() (also lock-guarded, since it
    mutates the same waiting deque the loop thread reads) and then block
    on a per-request threading.Event that the loop thread sets once that
    request's seq_id shows up in a step()'s finished-outputs list. This
    is what actually lets nanovllm's Scheduler batch concurrently-submitted
    requests into the same forward pass, which is the entire point of
    relaxing FCFS.

    Heads-up not enforced automatically here (deliberately -- this class
    doesn't reach into LLM construction args): src/server.py's own
    --max-num-seqs default (4) is sized for Engine's "at most one sequence
    ever in flight" invariant (see that flag's help text). Pass a higher
    --max-num-seqs explicitly when using --concurrency-mode batched, or
    concurrent requests beyond that count will simply queue in
    Scheduler.waiting rather than batch, same as today.
    """
    def __init__(self, llm: "LLM"):  # noqa: F821 -- LLM is imported locally in main(), not at module level; see that import's comment.
        self.llm = llm
        self.tok = llm.tokenizer
        self.eos_id = self.tok.eos_token_id
        # Guards add_request()/step()/is_finished() only -- never held for
        # a whole request's duration, unlike Engine._gen_lock. This is what
        # lets two requests submitted moments apart both be RUNNING and get
        # swept into the same step() call.
        self._lock = threading.Lock()
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, list[int]] = {}
        self._stop = threading.Event()
        self._loop_thread = threading.Thread(target=self._loop, daemon=True)
        self._loop_thread.start()

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                idle = self.llm.is_finished()
                if not idle:
                    outputs, _ = self.llm.step()
                    for seq_id, token_ids in outputs:
                        self._results[seq_id] = token_ids
                        ev = self._pending.pop(seq_id, None)
                        if ev is not None:
                            ev.set()
            if idle:
                # Nothing scheduled -- no in-flight or waiting requests.
                # Short poll interval, not a Condition variable woken by
                # add_request(): kept simple deliberately for a same-day
                # diagnostic flag; revisit if this ever needs to be the
                # permanent server path.
                time.sleep(0.001)
            else:
                # GIL yield, found 2026-08-27: with no sleep at all on this
                # branch, this thread calls step() back-to-back in a tight
                # loop for the entire span of an active generation. Every
                # decode step's Python-side bookkeeping (Scheduler.schedule(),
                # postprocess(), tensor indexing) holds the GIL, and with no
                # cooperative yield between iterations, this starves the
                # asyncio event loop thread -- which has to actually read/
                # parse an incoming HTTP connection before it can even call
                # add_request() -- for the full duration of an active
                # generation. Confirmed via [STEP DEBUG] server-side logging:
                # a second request fired concurrently with a first did not
                # reach add_request() until ~17-19s later, right when the
                # first request's generation was finishing -- not explainable
                # by network/tokenization latency (millisecond-scale), but
                # exactly the signature of this thread starving the request
                # from ever being read. A brief sleep here doesn't meaningfully
                # cost decode throughput (a fraction of a percent against a
                # ~15-20ms decode step) but gives the event loop thread a real
                # chance to run between steps.
                time.sleep(0.0001)

    def generate(self, prompt_token_ids: list[int], max_tokens: int,
                 temperature: float, ignore_eos: bool = False,
                 stop: list[str] | None = None) -> list[int]:
        """Run full generation for one request. Blocks caller thread, but
        (unlike Engine.generate) does NOT block other requests from being
        admitted into the scheduler and batched alongside this one."""
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                             ignore_eos=ignore_eos, stop=stop)
        ev = threading.Event()
        with self._lock:
            seq_id = self.llm.add_request(prompt_token_ids, sp)
            self._pending[seq_id] = ev
        ev.wait()
        with self._lock:
            return self._results.pop(seq_id)

    def shutdown(self):
        self._stop.set()
        self._loop_thread.join(timeout=5)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app  = FastAPI()
_engine: Optional[Engine] = None
# Found 2026-08-27: asyncio's DEFAULT executor (what loop.run_in_executor(None, ...)
# uses) is sized min(32, os.cpu_count()+4) -- on this box (28 cores) that's
# exactly 32. Every concurrent request blocks one of those threads for its
# ENTIRE generation (minutes, via Engine._gen_lock or BatchedEngine.generate's
# ev.wait()), so the 33rd+ concurrent request doesn't even reach
# add_request() until one of the first 32 finishes -- a hard wall completely
# independent of max_num_seqs/max_num_batched_tokens/KV-cache capacity,
# confirmed via [STEP DEBUG] server-side logging showing admission stuck at
# exactly 32 regardless of prompt length. Explicit, correctly-sized executor
# instead of the default -- sized in main() once args.max_num_seqs is known.
_executor: Optional[ThreadPoolExecutor] = None
# Populated in main() right after argparse, read-only afterward -- exists so
# /health can report the actual admission-control config a running server
# was launched with. Added 2026-08-26: a real concurrency=64 sweep session
# found what looked like scheduler "starvation" (a specific subset of
# requests never progressing) that CPU-only simulation of the real
# Scheduler/BlockManager logic (tests/diag_scheduler_starvation_cpu.py)
# could NOT reproduce via eviction/thrashing, but COULD reproduce cleanly
# via an undersized --max-num-seqs simply queuing the tail of a wide wave
# behind a much smaller real batch cap -- i.e. not a bug, just a client
# sweeping past a cap it had no way to see. Exposing the cap here lets a
# client (bench_http_concurrency.py) catch that mismatch BEFORE spending a
# GPU window on a misleading run, instead of discovering it after the fact.
_server_config: Optional[dict] = None

@app.get("/health")
def health():
    body = {"status": "ok"}
    if _server_config is not None:
        body["config"] = _server_config
    return JSONResponse(body)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    global _engine
    if _engine is None:
        raise HTTPException(503, "Model not loaded")

    # Apply chat template (thinking disabled via enable_thinking=False)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        text = _engine.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        text = _engine.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    input_ids = _engine.tok.encode(text, add_special_tokens=False)
    prompt_tokens = len(input_ids)

    # Run generation in a thread so the event loop stays unblocked.
    # The engine's generation lock serialises whole requests across all threads.
    loop = asyncio.get_event_loop()
    output_ids = await loop.run_in_executor(
        _executor,
        _engine.generate,
        input_ids,
        req.max_tokens,
        req.temperature,
        req.ignore_eos,
        req.stop,
    )

    output_text = _engine.tok.decode(output_ids, skip_special_tokens=True)
    completion_tokens = len(output_ids)
    finish_reason = "stop" if (output_ids and output_ids[-1] == _engine.eos_id) else "length"

    return JSONResponse({
        "id":      f"chatcmpl-{uuid.uuid4()}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "Qwen/Qwen3.5-35B-A3B",
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": output_text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    })

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _engine, _server_config, _executor

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to the Qwen3.5-35B-A3B HF checkpoint dir")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        dest="tensor_parallel_size",
                        help="The path validated in Phase 5 / Phases 4-7 (see README.md "
                             "for current validation status).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        dest="gpu_memory_utilization")
    parser.add_argument("--max-model-len", type=int, default=4096, dest="max_model_len")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384,
                        dest="max_num_batched_tokens")
    parser.add_argument("--enforce-eager", dest="enforce_eager",
                        action="store_true", default=True,
                        help="Default. Eager decode, no CUDA graph capture. Matters "
                             "beyond just CUDA graphs: engine/model_runner.py only sets "
                             "torch._dynamo.config.disable = enforce_eager, which is what "
                             "actually keeps the several @torch.compile decorators in "
                             "layers/ (layernorm.py, sampler.py, ...) from tracing at all "
                             "-- leaving this False re-enables torch.compile/inductor, a "
                             "less-tested path this project's own bench_throughput.py "
                             "avoids by defaulting to eager too.")
    parser.add_argument("--cuda-graphs", dest="enforce_eager", action="store_false",
                        help="Use CUDA-graph decode instead of eager mode (validated in "
                             "Phase 4, see README.md -- opt in explicitly).")
    parser.add_argument("--fake-config-loader", dest="fake_config_loader",
                        action="store_true", default=False,
                        help="Bypass transformers' real AutoConfig and read config.json "
                             "directly instead. Only for tests/fake_qwen35_small-style "
                             "fixtures with a flat (non-nested) config.json -- see the "
                             "shim's docstring above for why real AutoConfig now "
                             "mis-derives layer_types for that fixture shape. Do not use "
                             "this against a real checkpoint's own config.json.")
    parser.add_argument("--max-num-seqs", type=int, default=4, dest="max_num_seqs",
                        help="Default 4, NOT nanovllm's library default of 512. "
                             "StateManager (engine/state_manager.py) pre-allocates a "
                             "fixed-size recurrent-state slot pool sized at max_num_seqs "
                             "-- memory scales linearly with it regardless of actual "
                             "load, so this stays conservative rather than defaulting to "
                             "512 (confirmed: 512 needed 15GB here, on a checkpoint whose "
                             "weights already consumed most of a 48GB A6000 under TP=2, "
                             "and OOM'd by ~180MB). IMPORTANT with the default "
                             "--concurrency-mode batched: this caps how many requests can "
                             "actually batch together -- anything beyond it queues instead "
                             "of batching, which produces a flatter-than-real tok/s curve "
                             "at higher concurrency (see SESSION_HANDOFF_2026-08-26.md's "
                             "scheduler-starvation finding). Raise this to at least your "
                             "highest tested concurrency level before a throughput sweep; "
                             "the 4 default only made sense back when --concurrency-mode "
                             "fcfs (at most one sequence ever in flight) was the default.")
    parser.add_argument("--concurrency-mode", dest="concurrency_mode",
                        choices=["fcfs", "batched"], default="batched",
                        help="Default 'batched' (flipped 2026-08-26 -- was 'fcfs'; see "
                             "SESSION_HANDOFF_2026-08-26.md): BatchedEngine (see its "
                             "docstring), real continuous batching across concurrent "
                             "requests via nanovllm's Scheduler -- the mode every "
                             "concurrency-sweep throughput number in this project's docs "
                             "(204.1 tok/s etc.) is meant to reflect. 'fcfs': Engine._gen_lock "
                             "serialises whole requests (see Engine's docstring) -- a forward "
                             "pass never contains more than one request's tokens, so tok/s "
                             "will NOT scale with concurrency at all. Kept as an explicit opt-in "
                             "for the specific case this was built for -- an apples-to-apples "
                             "comparison against a simpler engine that structurally cannot batch "
                             "(see the module docstring's Concurrency section) -- not as a "
                             "default anyone should hit by omitting this flag. Raise "
                             "--max-num-seqs when using 'batched' -- see that flag's help text.")
    parser.add_argument("--moe-w8a8", dest="use_moe_w8a8", action="store_true", default=False,
                        help="Quantize MoE expert weights to INT8 after loading (Q4/Q6 -- "
                             "correctness- and accuracy-validated, 40/40 GSM8K non-regression "
                             "under chat-no-think; see moe_quantization_memo.md). Off by "
                             "default, matching every other entry point in this project. "
                             "Under CUDA graphs (--cuda-graphs), needs more headroom than "
                             "bf16 -- allocate_kv_cache() sizes the KV cache before "
                             "capture_cudagraph() claims its private pool, so pass a lower "
                             "--gpu-memory-utilization than you would for bf16 (0.75 was the "
                             "smallest that worked at concurrency=16 on a 48GB A6000; the "
                             "margin needed grows with --max-num-seqs, since that sets the "
                             "largest captured graph bucket).")
    parser.add_argument("--moe-w8a8-group-size", type=int, default=128,
                        dest="moe_w8a8_weight_group_size",
                        help="Group size for INT8 grouped quantization. 128 divides both "
                             "hidden_size=2048 and moe_intermediate_size=512 exactly on the "
                             "real checkpoint (Q0). Only used when --moe-w8a8 is set.")
    parser.add_argument("--fused-moe-kernel", dest="fused_moe_kernel", action="store_true", default=False,
                        help="Sets NANOVLLM_USE_FUSED_MOE_KERNEL=1 before the model is built. "
                             "This is the flag that made the validated 204.1 tok/s A6000 result "
                             "possible -- without it, use_moe_w8a8=True still runs, but through "
                             "the much slower plain gather+dequant+einsum path, not the fused "
                             "Triton kernel. Only takes effect together with --moe-w8a8. "
                             "Previously only settable by exporting the env var by hand before "
                             "launching this script -- easy to forget, and forgetting it silently "
                             "produces a real but much slower number with no warning.")
    parser.add_argument("--vectorized-moe", dest="vectorized_moe", action="store_true", default=False,
                        help="Enables Qwen35MoE's grouped-GEMM prefill dispatch "
                             "(_forward_dispatch_vectorized, torch._grouped_mm) instead of the "
                             "default per-expert Python loop (_forward_dispatch) -- the loop "
                             "does an .item()-based host sync for every one of num_experts "
                             "(256 on the real checkpoint), every prefill call. Only takes "
                             "effect at --tensor-parallel-size 1 (ep_size>1 always uses the "
                             "separate EP dispatch path, which this flag does not touch -- see "
                             "Qwen35MoE.forward()'s branch order). See "
                             "tests/test_qwen35_vectorized_moe.py / tests/bench_vectorized_moe.py.")
    args = parser.parse_args()

    # See _executor's module-level comment -- sized to max_num_seqs (no
    # point exceeding what the engine can ever usefully run concurrently
    # anyway) instead of asyncio's default min(32, cpu_count()+4), which
    # silently caps real concurrency at 32 on this box regardless of what
    # --max-num-seqs is actually set to.
    _executor = ThreadPoolExecutor(max_workers=args.max_num_seqs)

    # Must happen before the LLM import a few lines down -- see that
    # import's own comment for why NANOVLLM_USE_FUSED_MOE_KERNEL has to be
    # set before models/qwen3_5.py is first imported, not just before the
    # model is constructed.
    if args.fused_moe_kernel:
        os.environ["NANOVLLM_USE_FUSED_MOE_KERNEL"] = "1"
    from nanovllm.llm import LLM

    _server_config = {
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "concurrency_mode": args.concurrency_mode,
        "tensor_parallel_size": args.tensor_parallel_size,
        "fused_moe_kernel": args.fused_moe_kernel,
        "vectorized_moe": args.vectorized_moe,
    }

    # Loud, launch-time version of the same check bench_http_concurrency.py
    # does against /health -- catch a likely-flat-throughput misconfig
    # before anyone spends a GPU window sweeping concurrency against it,
    # not just after the fact from a client script.
    if args.concurrency_mode == "batched" and args.max_num_seqs <= 4:
        print(f"[WARNING] --concurrency-mode batched with --max-num-seqs={args.max_num_seqs} -- "
              f"requests beyond {args.max_num_seqs} in a wave will queue instead of batching. "
              f"Raise --max-num-seqs to at least your highest planned concurrency level for a "
              f"real throughput sweep (see that flag's help text).")
    if args.use_moe_w8a8 and not args.fused_moe_kernel:
        print(f"[WARNING] --moe-w8a8 without --fused-moe-kernel -- running the slower plain "
              f"gather+dequant+einsum path, not the fused Triton kernel. This is NOT the "
              f"204.1 tok/s A6000 configuration. Pass --fused-moe-kernel too unless the "
              f"dequant-only path is deliberately what's being measured.")

    if args.fake_config_loader:
        import nanovllm.config as config_mod
        config_mod.AutoConfig.from_pretrained = staticmethod(_fake_from_pretrained)

    print(f"Building nanovllm engine for model={args.model} "
          f"(tensor_parallel_size={args.tensor_parallel_size})...")
    llm = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        use_moe_w8a8=args.use_moe_w8a8,
        moe_w8a8_weight_group_size=args.moe_w8a8_weight_group_size,
        use_vectorized_moe=args.vectorized_moe,
    )

    # Added 2026-08-27: real KV-cache block count is only known after
    # allocate_kv_cache() runs inside LLM(...) construction, unlike the
    # other _server_config fields (known straight from args). Exposed here
    # so a client can directly check real concurrent-request capacity
    # (num_kvcache_blocks * kvcache_block_size // tokens_per_request)
    # instead of inferring it after the fact from a suspicious "N requests
    # ran, the rest queued behind them" pattern in a real sweep -- see
    # BlockManager.can_allocate() returning -1 mid-prefill-admission as the
    # actual mechanism this caps: Scheduler.schedule()'s prefill loop stops
    # admitting new sequences the moment free blocks run out, even if
    # max_num_seqs/max_num_batched_tokens would otherwise allow more.
    _server_config["num_kvcache_blocks"] = llm.model_runner.config.num_kvcache_blocks
    _server_config["kvcache_block_size"] = llm.model_runner.config.kvcache_block_size

    print(f"Starting engine (concurrency_mode={args.concurrency_mode})...")
    if args.concurrency_mode == "batched":
        _engine = BatchedEngine(llm)
        atexit.register(_engine.shutdown)
    else:
        _engine = Engine(llm)

    print(f"Server ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
