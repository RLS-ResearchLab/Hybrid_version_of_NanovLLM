# tests/debug_warmup_state.py
import os
import sys
import types
#### importation
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if __package__ in (None, "") and "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg
###################################
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "2333")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.config import Config

cfg = Config(
    model="fake_qwen35_small",   # adjust path as needed
    tensor_parallel_size=1,
    enforce_eager=True,
    max_num_seqs=4,
    max_num_batched_tokens=512,
)
print(">>> constructing ModelRunner (watch for crash inside warmup_model)")
mr = ModelRunner(cfg, 0, [])
print(">>> survived warmup + full init")