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
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

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

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

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
    def __init__(self, llm: LLM):
        self.llm = llm
        self.tok = llm.tokenizer
        self.eos_id = self.tok.eos_token_id
        self._gen_lock = threading.Lock()

    def generate(self, prompt_token_ids: list[int], max_tokens: int,
                 temperature: float) -> list[int]:
        """Run full generation for one request. Blocks caller thread."""
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        with self._gen_lock:
            outputs = self.llm.generate([prompt_token_ids], sp, use_tqdm=False)
        return outputs[0]["token_ids"]

# ── FastAPI app ───────────────────────────────────────────────────────────────

app  = FastAPI()
_engine: Optional[Engine] = None

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

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
        None,
        _engine.generate,
        input_ids,
        req.max_tokens,
        req.temperature,
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
    global _engine

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
                        action="store_true", default=False,
                        help="Eager decode, no CUDA graph capture.")
    args = parser.parse_args()

    print(f"Building nanovllm engine for model={args.model} "
          f"(tensor_parallel_size={args.tensor_parallel_size})...")
    llm = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
    )

    print("Starting engine...")
    _engine = Engine(llm)

    print(f"Server ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
