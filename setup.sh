#!/usr/bin/env bash
# One-shot GPU-box bootstrap for qLLM debugging windows.
#
# Written after burning real time re-deriving this by hand across two dead
# boxes in one window (H100 -> box died -> H200). Auto-detects the driver's
# CUDA ceiling from nvidia-smi instead of hardcoding a toolkit/torch version,
# so the same script works whether the next box is H100, H200, or anything
# else Hopper-or-later. Idempotent: safe to re-run on the same box (skips
# steps already done) or a brand new one (does everything from scratch).
#
# Usage on a fresh box:
#   wget https://raw.githubusercontent.com/RLS-ResearchLab/qLLM/main/setup.sh -O setup.sh
#   chmod +x setup.sh && ./setup.sh
# (or scp/paste this file directly if the box has no internet to GitHub yet)

set -euo pipefail

REPO_URL="https://github.com/RLS-ResearchLab/qLLM.git"
REPO_DIR="$HOME/qLLM"

echo "=== [1/8] GPU / driver detection ==="
nvidia-smi
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
if [ -z "$CUDA_VER" ]; then
    echo "ERROR: could not parse driver's CUDA ceiling from nvidia-smi output. Inspect manually." >&2
    exit 1
fi
CUDA_TAG=$(echo "$CUDA_VER" | tr -d '.')       # 12.8 -> 128, 13.0 -> 130 (pytorch.org index tag)
CUDA_PKG=$(echo "$CUDA_VER" | tr '.' '-')      # 12.8 -> 12-8, 13.0 -> 13-0 (apt package suffix)
echo "Driver CUDA ceiling: $CUDA_VER  (torch index cu${CUDA_TAG}, apt pkg cuda-toolkit-${CUDA_PKG})"

echo "=== [2/8] Repo clone/pull ==="
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull

echo "=== [3/8] Python venv ==="
if [ ! -d .venv ]; then
    python3 -m venv .venv 2>/dev/null || {
        sudo apt-get update && sudo apt-get install -y python3.10-venv
        python3 -m venv .venv
    }
fi
source .venv/bin/activate
echo "python3 -> $(which python3)"

echo "=== [4/8] CUDA-independent pip deps + background checkpoint download ==="
pip install --upgrade pip -q
pip install -q transformers xxhash numpy tqdm safetensors fastapi "uvicorn[standard]" pydantic \
    huggingface_hub accelerate "jinja2>=3.1.0" aiohttp tabulate datasets ninja packaging
# (deliberately skipping lm-eval and flash-linear-attention from requirements.txt --
# neither is needed for the actual test scripts, both are slow to install)

if [ ! -f qwen35_checkpoint/.download_complete ]; then
    nohup bash -c "huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir ./qwen35_checkpoint && touch qwen35_checkpoint/.download_complete" \
        > checkpoint_download.log 2>&1 &
    disown
    echo "Checkpoint download started in background (PID $!) -> checkpoint_download.log"
else
    echo "Checkpoint already present (qwen35_checkpoint/.download_complete found), skipping."
fi

echo "=== [5/8] CUDA toolkit (matching driver ceiling $CUDA_VER) ==="
CURRENT_NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "")
if [ "$CURRENT_NVCC_VER" != "$CUDA_VER" ]; then
    UBUNTU_TAG="ubuntu$(. /etc/os-release && echo "$VERSION_ID" | tr -d '.')"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_TAG}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update
    sudo apt-get -y install "cuda-toolkit-${CUDA_PKG}"

    if ! grep -q "cuda-${CUDA_VER}/bin" ~/.bashrc 2>/dev/null; then
        echo "export PATH=/usr/local/cuda-${CUDA_VER}/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VER}/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    fi
    export PATH="/usr/local/cuda-${CUDA_VER}/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VER}/lib64:${LD_LIBRARY_PATH:-}"
else
    echo "nvcc $CURRENT_NVCC_VER already matches driver ceiling, skipping toolkit install."
fi
nvcc --version

echo "=== [6/8] torch (cu${CUDA_TAG} index) ==="
NEED_TORCH=1
if python3 -c "import torch" 2>/dev/null; then
    if python3 -c "
import torch, sys
drv = tuple(int(x) for x in '${CUDA_VER}'.split('.'))
got = tuple(int(x) for x in torch.version.cuda.split('.')[:2])
sys.exit(0 if got <= drv and torch.cuda.is_available() else 1)
" 2>/dev/null; then
        NEED_TORCH=0
        echo "torch already installed and compatible, skipping."
    fi
fi
if [ "$NEED_TORCH" = "1" ]; then
    pip install torch --index-url "https://download.pytorch.org/whl/cu${CUDA_TAG}" --no-cache-dir -q
fi
python3 -c "
import torch, sys
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
drv = tuple(int(x) for x in '${CUDA_VER}'.split('.'))
got = tuple(int(x) for x in torch.version.cuda.split('.')[:2])
# torch built for a HIGHER CUDA than the driver supports will fail at runtime
# (this is exactly what happened on the H100 box: driver 12.8, torch cu130 ->
# 'CUDA driver too old' on the very first torch.cuda call). Lower is fine
# (backward compatible); higher is not -- fail loudly here instead of an
# hour later mid-test.
if got > drv:
    print(f'FATAL: torch built for CUDA {torch.version.cuda} > driver ceiling ${CUDA_VER}. Re-run with a lower --index-url (e.g. cu${CUDA_TAG} explicitly, or pin an exact older torch version).')
    sys.exit(1)
if not torch.cuda.is_available():
    print('FATAL: torch.cuda.is_available() is False. Investigate directly, do not proceed.')
    sys.exit(1)
"

echo "=== [7/8] flash-attn + triton ==="
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
NPROC=$(nproc)
FA_MAX_JOBS=$(( RAM_GB / 4 ))
[ "$FA_MAX_JOBS" -lt 1 ] && FA_MAX_JOBS=1
[ "$FA_MAX_JOBS" -gt "$NPROC" ] && FA_MAX_JOBS=$NPROC
echo "RAM=${RAM_GB}GB nproc=${NPROC} -> MAX_JOBS=${FA_MAX_JOBS} (caps parallel nvcc compiles to avoid the OOM hit on the H100 box)"
if ! python3 -c "import flash_attn" 2>/dev/null; then
    # NOTE: unresolved as of 2026-08-23 -- flash-attn's bundled cutlass had
    # real compile errors against CUDA 13.0 on the H200 box (not just the
    # deprecation-warning noise), root cause not yet found. If this fails,
    # capture full output (not just the tail) and read the actual "error:"
    # lines, not the surrounding warnings:
    #   MAX_JOBS=N pip install flash-attn --no-build-isolation > flash_attn_build.log 2>&1
    #   grep -B5 "error:" flash_attn_build.log
    MAX_JOBS=$FA_MAX_JOBS pip install flash-attn --no-build-isolation
else
    echo "flash-attn already importable, skipping."
fi
python3 -c "import triton; print('triton', triton.__version__)"

echo "=== [8/8] Done ==="
echo "Checkpoint download progress: tail -f $REPO_DIR/checkpoint_download.log"
echo ""
echo "Next steps, in priority order (see H200_test_day_checklist.md / SESSION_HANDOFF*.md):"
echo "  1. python layers/smoke_test_moe_w8a8_hopper.py"
echo "     (Hopper W8A8 wgmma/TMA kernel compile + smoke test -- the main event)"
echo "  2. python tests/cluster_q6_moe_w8a8_gsm8k.py --tp 1 --dry-run --gpu-memory-utilization 0.85"
echo "     (tp=1 INT8 fix validation -- also try with NANOVLLM_USE_FUSED_MOE_KERNEL=1 prefixed)"
