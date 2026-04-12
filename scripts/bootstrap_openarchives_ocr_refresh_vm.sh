#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <repo-root> <venv-path>" >&2
  exit 1
fi

REPO_ROOT="$1"
VENV_PATH="$2"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y \
  git \
  rsync \
  zstd \
  build-essential \
  pkg-config \
  libssl-dev \
  python3.11 \
  python3.11-venv \
  python3-dev \
  curl \
  patchelf

if [[ ! -d "$HOME/.cargo" ]]; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
source "$HOME/.cargo/env"
rustup toolchain install stable
rustup default stable

"$PYTHON_BIN" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install --upgrade pip wheel setuptools maturin
pip install -e "$REPO_ROOT"

cd "$REPO_ROOT"
maturin develop -m rust/glossapi_rs_noise/Cargo.toml

echo "bootstrap complete"
echo "repo_root=$REPO_ROOT"
echo "venv_path=$VENV_PATH"
