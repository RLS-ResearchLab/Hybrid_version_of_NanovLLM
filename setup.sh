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
    huggingface_hub accelerate "jinja2>=3.1.0" aiohttp tabulate datasets ninja packaging \
    wheel setuptools psutil
# (deliberately skipping lm-eval and flash-linear-attention from requirements.txt --
# neither is needed for the actual test scripts, both are slow to install)

if [ ! -f qwen35_checkpoint/.download_complete ]; then
    # The checkpoint is ~67GB. GPU rental boxes on this provider have been seen
    # with a small (~100GB) root disk that's already 90%+ full from the CUDA
    # toolkit/venv/build artifacts, plus a much larger, empty /ephemeral data
    # volume -- prefer that if it exists, rather than filling the root disk and
    # failing mid-download with "No space left on device".
    CKPT_DIR="$REPO_DIR/qwen35_checkpoint"
    if [ -d /ephemeral ] && [ "$(df --output=avail /ephemeral 2>/dev/null | tail -1)" -gt "$(df --output=avail "$REPO_DIR" | tail -1)" ]; then
        mkdir -p /ephemeral/qwen35_checkpoint
        ln -sfn /ephemeral/qwen35_checkpoint "$CKPT_DIR"
        echo "Using /ephemeral for the checkpoint (more free space than the root disk)."
    fi
    # `huggingface-cli download` is deprecated in newer huggingface_hub releases
    # and silently no-ops (prints a warning, downloads nothing) instead of
    # erroring -- use `hf download` (the replacement) so this doesn't look like
    # it started when it didn't.
    nohup bash -c "hf download Qwen/Qwen3.5-35B-A3B --local-dir '$CKPT_DIR' && touch '$CKPT_DIR'/.download_complete" \
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

# Every session handoff in this repo assumes `source ~/qLLM/cuda_env.sh`
# before launching the server (nvcc-on-PATH gotcha above, plus
# PYTHONUNBUFFERED=1 -- without it, server startup logs are block-buffered
# and don't appear until the process exits, making `tail -f`/`grep` on a
# live server misleading). Generate it here so a from-scratch setup.sh run
# actually produces the file every later doc/script expects, instead of
# relying on it being hand-written mid-session.
cat > "$REPO_DIR/cuda_env.sh" <<EOF
export CUDA_HOME=/usr/local/cuda-${CUDA_VER}
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
EOF
echo "Wrote $REPO_DIR/cuda_env.sh -- source it before every server launch."

echo "=== [6/8] torch (cu${CUDA_TAG} index) ==="

# torch built for a HIGHER CUDA than the driver supports will fail at runtime
# ('CUDA driver too old' on the first torch.cuda call). Lower is fine
# (backward compatible); higher is not. Checked via static attributes only
# (torch.__version__/torch.version.cuda) -- NOT torch.cuda.is_available() or
# get_device_name(), which both trigger cuInit() and crash with an ugly raw
# traceback exactly when this check would otherwise catch the problem.
check_torch() {
    python3 -c "
import torch, sys
drv = tuple(int(x) for x in '${CUDA_VER}'.split('.'))
got = tuple(int(x) for x in torch.version.cuda.split('.')[:2])
sys.exit(0 if got <= drv else 1)
" 2>/dev/null
}

NEED_TORCH=1
if python3 -c "import torch" 2>/dev/null && check_torch; then
    NEED_TORCH=0
    echo "torch already installed and compatible, skipping."
fi

if [ "$NEED_TORCH" = "1" ]; then
    echo "Installing torch from cu${CUDA_TAG} index (unpinned)..."
    pip install torch --index-url "https://download.pytorch.org/whl/cu${CUDA_TAG}" --no-cache-dir -q

    if ! check_torch; then
        # Confirmed twice now (H100 box: cu124 index -> resolved 2.13.0+cu130;
        # this box: cu128 index -> also resolved 2.13.0+cu130) -- unpinned
        # "latest" resolution against these older CUDA-tagged pytorch.org
        # indices is not reliable. Self-heal by pinning to whatever version
        # is ACTUALLY hosted under this tag instead of trusting resolution.
        echo "WARNING: unpinned torch resolved to an incompatible CUDA build (seen before)." >&2
        echo "Retrying pinned to whatever version pip index versions actually lists under cu${CUDA_TAG}..." >&2
        PINNED_VER=$(pip index versions torch --index-url "https://download.pytorch.org/whl/cu${CUDA_TAG}" 2>/dev/null \
            | grep -oP '^torch \(\K[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if [ -z "$PINNED_VER" ]; then
            echo "FATAL: could not determine any torch version hosted under cu${CUDA_TAG}. Manual intervention needed." >&2
            exit 1
        fi
        echo "Pinning torch==${PINNED_VER} (cu${CUDA_TAG})"
        pip uninstall -y torch -q
        pip cache remove torch -q 2>/dev/null || true
        pip install "torch==${PINNED_VER}" --index-url "https://download.pytorch.org/whl/cu${CUDA_TAG}" --no-cache-dir -q
    fi
fi

python3 -c "
import torch, sys
print('torch', torch.__version__, 'compiled for cuda', torch.version.cuda)
drv = tuple(int(x) for x in '${CUDA_VER}'.split('.'))
got = tuple(int(x) for x in torch.version.cuda.split('.')[:2])
if got > drv:
    print(f'FATAL: torch built for CUDA {torch.version.cuda} > driver ceiling ${CUDA_VER} even after the pinned retry. Manual intervention needed.')
    sys.exit(1)
if not torch.cuda.is_available():
    print('FATAL: torch.cuda.is_available() is False. Investigate directly, do not proceed.')
    sys.exit(1)
print('torch/CUDA compatibility confirmed:', torch.cuda.get_device_name(0))
"

echo "=== [7/8] flash-attn + triton ==="
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
NPROC=$(nproc)
# RAM_GB/4 was tried and OOM-killed TWICE (once on ~50GB-class box, once here
# with 100GB+ RAM giving MAX_JOBS=28) -- flash-attn's largest backward-pass
# kernels (hdim256 etc.) spike well past 4GB/job regardless of total system
# RAM, so a RAM-proportional formula isn't safe for this specific package.
# Cap hard at 4 unless RAM is enormous (500GB+), rather than scaling with it.
if [ "$RAM_GB" -ge 500 ]; then
    FA_MAX_JOBS=8
else
    FA_MAX_JOBS=4
fi
[ "$FA_MAX_JOBS" -gt "$NPROC" ] && FA_MAX_JOBS=$NPROC
echo "RAM=${RAM_GB}GB nproc=${NPROC} -> MAX_JOBS=${FA_MAX_JOBS} (flat cap, not RAM-proportional -- see comment above)"

# RESOLVED 2026-08-27 (was "unresolved, root cause not found" as of 2026-08-23):
# the "cutlass compile errors" were never a cutlass/CUDA-13 incompatibility --
# two mundane, sequential blockers that the ninja stack trace hides behind
# generic-looking noise:
#   1. nvcc not actually on PATH even after the toolkit install above
#      (nvidia-smi's "CUDA Version" is the DRIVER's ceiling, not a usable
#      toolkit link) -- already handled by this script's own PATH/
#      LD_LIBRARY_PATH exports in step 5, but g++ is not, see next point.
#   2. g++ missing -- apt's cuda-toolkit package pulls in gcc but NOT g++,
#      and nvcc's host compiler (cc1plus) ships with g++, not gcc alone.
#      Confirmed error: "gcc: fatal error: cannot execute 'cc1plus'".
# If a build still fails after the g++ install below, the ninja trace still
# hides the real line -- always: grep -A25 "^FAILED:" flash_attn_build.log
if ! command -v g++ >/dev/null 2>&1 || ! gcc -print-prog-name=cc1plus | grep -q '/'; then
    echo "g++ missing (nvcc needs cc1plus, which ships with g++, not gcc alone) -- installing..."
    sudo apt-get update -qq
    sudo apt-get install -y g++
fi

if ! python3 -c "import flash_attn" 2>/dev/null; then
    # FLASH_ATTN_CUDA_ARCHS=90 -- flash-attn ignores TORCH_CUDA_ARCH_LIST and
    # defaults to its OWN "80;90;100;110;120" (5 archs) if left unset; on a
    # real box this alone turned an unconstrained build (found 2026-08-28,
    # A40 box) into one compiling backward-pass kernels for 4 architectures
    # nothing here targets. 90 = Hopper (H100/H200) -- the only target this
    # project runs on. FLASH_ATTENTION_DISABLE_BACKWARD=TRUE roughly halves
    # the compile units (inference-only, no training). Confirmed working
    # recipe, ~15-25 min on a 20-core/125GB box:
    FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=$FA_MAX_JOBS FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
        FLASH_ATTENTION_FORCE_BUILD=TRUE \
        pip install flash-attn --no-build-isolation \
        2>&1 | tee flash_attn_build.log \
        || { echo "flash-attn build FAILED -- ninja's own error is usually buried; try: grep -A25 '^FAILED:' flash_attn_build.log" >&2; exit 1; }
else
    echo "flash-attn already importable, skipping."
fi
python3 -c "import triton; print('triton', triton.__version__)"

if ! python3 -c "import fla" 2>/dev/null; then
    # Pinned, not latest -- --fused-gdr-decode-kernel was written against
    # this exact fused_recurrent_gated_delta_rule signature. Triton-based,
    # no compile step -- trivial compared to flash-attn above.
    pip install -q "flash-linear-attention==0.5.2"
fi

echo "=== [8/8] Done ==="
echo "Checkpoint download progress: tail -f $REPO_DIR/checkpoint_download.log"
echo ""
echo "Next steps, in priority order (see H200_test_day_checklist.md / SESSION_HANDOFF*.md):"
echo "  1. python layers/smoke_test_moe_w8a8_hopper.py"
echo "     (Hopper W8A8 wgmma/TMA kernel compile + smoke test -- the main event)"
echo "  2. python tests/cluster_q6_moe_w8a8_gsm8k.py --tp 1 --dry-run --gpu-memory-utilization 0.85"
echo "     (tp=1 INT8 fix validation -- also try with NANOVLLM_USE_FUSED_MOE_KERNEL=1 prefixed)"
