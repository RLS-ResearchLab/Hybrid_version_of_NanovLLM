# tests/debug_conv1d.py
import torch
import torch.nn.functional as F

import os
import sys
import types

import torch.distributed as dist

# Mirror the standalone test's package redirect so local imports resolve.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

if "nanovllm" not in sys.modules:
    nanovllm_pkg = types.ModuleType("nanovllm")
    nanovllm_pkg.__path__ = [ROOT]
    nanovllm_pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["nanovllm"] = nanovllm_pkg

from nanovllm.models.qwen3_5 import Qwen35LinearAttention

# Reuse the same config helper from the existing test file.
from test_qwen35_standalone import make_small_config

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
if not dist.is_initialized():
    dist.init_process_group("gloo")

config = make_small_config()
device = "cuda" if torch.cuda.is_available() else "cpu"

la = Qwen35LinearAttention(
    hidden_size=config.hidden_size,
    linear_attn_kq_heads=config.linear_attn_kq_heads,
    linear_attn_v_heads=config.linear_attn_v_heads,
    linear_attn_head_dim=config.linear_attn_head_dim,
    conv_kernel_size=config.conv_kernel_size,
    rms_norm_eps=config.rms_norm_eps,
).to(device).to(torch.bfloat16)

x = torch.randn(1, 10, config.hidden_size, device=device, dtype=torch.bfloat16)

with torch.no_grad():
    y_full, state_full, conv_full = la(x)
    print("y_full[:, 0:1, :] abs sum:", y_full[:, 0:1, :].abs().sum().item())

    y0, state0, conv0 = la(x[:, 0:1, :], state=None, conv_state=None)
    print("y0 abs sum:", y0.abs().sum().item())
    print("state0 abs sum:", state0.abs().sum().item())
    print("conv0:", conv0)

    print("y0 vs y_full[:,0:1,:] cosine:",
          F.cosine_similarity(y0.float().reshape(-1).unsqueeze(0),
                               y_full[:, 0:1, :].float().reshape(-1).unsqueeze(0)))

dist.destroy_process_group()