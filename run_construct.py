"""One-off: attempt LLM(...) construction AND a single-token generate()
against the real qwen35_checkpoint. Run from the repo root:
python3 run_construct.py

max_tokens=1 is deliberate: engine/scheduler.py's postprocess() marks a
sequence finished as soon as num_completion_tokens == max_tokens, in the
same call that processes the token -- so with max_tokens=1 the first
(prefill) step's token immediately satisfies completion and the decode
path (Qwen35MoE._forward_gathered, still NotImplementedError at ep_size>1)
is never reached. This exercises construction, EP/TP-aware load_model(),
and one real prefill forward pass -- not decode.
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_WS_NAME = os.path.basename(ROOT)
if _WS_NAME != "nanovllm" and "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

# Import the submodule directly rather than `from nanovllm import LLM` --
# the latter goes through nanovllm/__init__.py's __getattr__ lazy-import
# shim, and CPython's IMPORT_FROM opcode swallows whatever the real
# underlying exception is if that shim's own import chain raises
# AttributeError anywhere, replacing it with a generic
# "cannot import name 'LLM'" message with no traceback. Importing the
# submodule directly surfaces the real error.
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


def main():
    llm = LLM(
        "qwen35_checkpoint",
        enforce_eager=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_num_seqs=1,
        max_model_len=2048,
    )
    print("LLM construction succeeded:", type(llm))

    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=1),
    )
    print("GENERATION SUCCEEDED")
    print(out)


if __name__ == "__main__":
    # Required: engine/llm_engine.py spawns rank>0 ModelRunner processes via
    # multiprocessing's "spawn" context whenever tensor_parallel_size > 1.
    # Under spawn, child processes re-import this file as __main__ -- without
    # this guard they'd re-run LLM(...) too, recursing before the parent even
    # finishes starting the first child.
    main()
