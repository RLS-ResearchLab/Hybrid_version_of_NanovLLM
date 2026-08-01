"""One-off: attempt LLM(...) construction against the real qwen35_checkpoint.
Construction only -- no generate(). Run from the repo root: python3 run_construct.py
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

llm = LLM(
    "qwen35_checkpoint",
    enforce_eager=True,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    max_num_seqs=1,
    max_model_len=2048,
)
print("LLM construction succeeded:", type(llm))
