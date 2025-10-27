#!/usr/bin/env bash
# Bootstrap script for the 4×L4 DeepSeek-OCR environment on an AWS g6 instance.
# Usage: bash setup_multi_gpu_env.sh

set -euo pipefail

# --- Configurable paths ---
CONDA_ENV_NAME="deepseek-ocr"
HF_CACHE_DIR="/opt/huggingface"
MODEL_CACHE="/opt/models"

# --- Sanity checks ---
if [[ $EUID -ne 0 ]]; then
  echo "[!] Please run this script with sudo (it needs apt + /opt writes)." >&2
  exit 1
fi

# Record start time
START_TS=$(date +%s)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Updating apt package index..."
apt-get update -y
apt-get upgrade -y

log "Installing system prerequisites..."
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential git curl wget jq unzip pkg-config ninja-build \
  libgl1 libglib2.0-0 libtiff-dev zlib1g-dev \
  htop nvtop tmux

# Ensure conda exists (Deep Learning AMI ships it under /opt/conda)
if [[ ! -x /opt/conda/bin/conda ]]; then
  log "Conda not found in /opt/conda; installing Miniconda..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
fi

source /opt/conda/etc/profile.d/conda.sh

# Accept Anaconda TOS in advance (avoids interactive prompts)
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

if conda info --envs | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  log "Conda env ${CONDA_ENV_NAME} already exists; skipping creation."
else
  log "Creating conda env ${CONDA_ENV_NAME} with Python 3.12..."
  conda create -y -n "${CONDA_ENV_NAME}" python=3.12
fi

log "Activating conda env ${CONDA_ENV_NAME}..."
conda activate "${CONDA_ENV_NAME}"

log "Ensuring uv package manager is installed..."
if ! command -v uv >/dev/null 2>&1; then
  export UV_INSTALL_DIR=/usr/local/bin
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

log "Installing PyTorch nightly (cu124) stack..."
uv pip install --python "$(which python)" --pre --index-url https://download.pytorch.org/whl/nightly/cu124 torch torchvision torchaudio

log "Installing vLLM nightly (cu124)..."
uv pip install --python "$(which python)" --pre --extra-index-url https://wheels.vllm.ai/nightly vllm

log "Installing Python dependencies (PyMuPDF, Pillow, numpy, transformers, etc.)..."
uv pip install --python "$(which python)" PyMuPDF pillow numpy rich tqdm huggingface-hub safetensors "transformers>=4.45.0"
uv pip install --python "$(which python)" 'huggingface_hub[hf_transfer]'

log "Persisting environment exports..."
mkdir -p /etc/profile.d
cat >/etc/profile.d/deepseek-ocr.sh <<ENV
export HF_HOME=${HF_CACHE_DIR}
export HUGGINGFACE_HUB_CACHE=${HF_CACHE_DIR}
export VLLM_WORKER_MULTIPROC=1
export PATH=/usr/local/bin:\$PATH
ENV
chmod 0644 /etc/profile.d/deepseek-ocr.sh

mkdir -p "${HF_CACHE_DIR}" "${MODEL_CACHE}"
chown -R $(logname):$(logname) "${HF_CACHE_DIR}" "${MODEL_CACHE}"

log "(Optional) Run 'huggingface-cli login' with HF_TOKEN after script completes."
log "Total setup time: $(( $(date +%s) - START_TS )) seconds"
