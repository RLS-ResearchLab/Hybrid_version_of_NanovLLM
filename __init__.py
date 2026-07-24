import os
import sys
import types

if "nanovllm" not in sys.modules:
	nanovllm_pkg = types.ModuleType("nanovllm")
	nanovllm_pkg.__path__ = [os.path.dirname(__file__)]
	nanovllm_pkg.__file__ = os.path.join(os.path.dirname(__file__), "__init__.py")
	sys.modules["nanovllm"] = nanovllm_pkg

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
	if name == "LLM":
		from .llm import LLM

		return LLM
	if name == "SamplingParams":
		from .sampling_params import SamplingParams

		return SamplingParams
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
