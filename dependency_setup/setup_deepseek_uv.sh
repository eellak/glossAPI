#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/deepseek_uv"

PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${GLOSSAPI_DEEPSEEK_VENV:-${REPO_ROOT}/dependency_setup/.venvs/deepseek}"
MODEL_ROOT="${DEEPSEEK_ROOT:-${REPO_ROOT}/deepseek-ocr-2-model}"
DOWNLOAD_MODEL=0
RUN_SMOKE=0
RUN_TESTS=0

info()  { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[err]\033[0m %s\n" "$*" >&2; exit 1; }

SYNC_ARGS=(--no-dev)

usage() {
  cat <<'EOF'
Usage: setup_deepseek_uv.sh [options]

Options:
  --venv PATH            Target virtual environment path
  --python PATH          Python executable to use for uv venv
  --model-root PATH      Destination root for the DeepSeek-OCR-2 model
  --download-model       Download DeepSeek-OCR-2 via huggingface_hub
  --run-tests            Run the DeepSeek pytest subset after installation
  --smoke-test           Run dependency_setup/deepseek_gpu_smoke.py
  --help                 Show this help message
EOF
}

prepend_path_if_dir() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    case ":${PATH}:" in
      *":${dir}:"*) ;;
      *) export PATH="${dir}:${PATH}" ;;
    esac
  fi
}

ensure_stable_python() {
  local python_bin="$1"
  local release_level
  release_level="$("${python_bin}" - <<'PY'
import sys
print(sys.version_info.releaselevel)
PY
)"
  if [[ "${release_level}" != "final" ]]; then
    error "Python interpreter ${python_bin} is not a stable final release (releaselevel=${release_level}). Install a stable CPython (for example via 'uv python install 3.11.11') and rerun with --python."
  fi
}

check_rust_toolchain() {
  if ! command -v cargo >/dev/null 2>&1; then
    error "cargo is required to build the Rust extensions. Install Rust (for example via rustup) and ensure cargo is on PATH."
  fi
  if ! cargo metadata --format-version 1 --manifest-path "${REPO_ROOT}/rust/glossapi_rs_cleaner/Cargo.toml" >/dev/null 2>&1; then
    error "Current cargo cannot parse the repo Rust metadata/Cargo.lock. Upgrade Rust (for example 'rustup toolchain install stable && rustup default stable') and rerun setup."
  fi
}

while (( "$#" )); do
  case "$1" in
    --venv)
      shift || { echo "--venv requires a path" >&2; exit 1; }
      VENV_PATH="${1:-}"
      ;;
    --python)
      shift || { echo "--python requires a path" >&2; exit 1; }
      PYTHON_BIN="${1:-}"
      ;;
    --model-root|--weights-dir)
      shift || { echo "--model-root requires a path" >&2; exit 1; }
      MODEL_ROOT="${1:-}"
      ;;
    --download-model|--download-deepseek)
      DOWNLOAD_MODEL=1
      ;;
    --run-tests)
      RUN_TESTS=1
      ;;
    --smoke-test)
      RUN_SMOKE=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift || true
done

prepend_path_if_dir "${HOME}/.local/bin"
prepend_path_if_dir "${HOME}/.cargo/bin"

command -v uv >/dev/null 2>&1 || error "uv is required. Install it first, e.g. 'python3 -m pip install --user uv'."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || error "Python executable not found: ${PYTHON_BIN}"
ensure_stable_python "${PYTHON_BIN}"
check_rust_toolchain

MODEL_DIR="${MODEL_ROOT}/DeepSeek-OCR-2"

if [[ -x "${VENV_PATH}/bin/python" ]]; then
  info "Reusing uv environment at ${VENV_PATH}"
else
  info "Creating uv environment at ${VENV_PATH}"
  uv venv --python "${PYTHON_BIN}" "${VENV_PATH}"
fi

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  SYNC_ARGS+=(--group test)
fi

info "Syncing DeepSeek runtime from ${PROJECT_DIR}"
UV_PROJECT_ENVIRONMENT="${VENV_PATH}" uv sync --project "${PROJECT_DIR}" --python "${VENV_PATH}/bin/python" "${SYNC_ARGS[@]}"

info "Installing Rust extensions in editable mode"
uv pip install --python "${VENV_PATH}/bin/python" -e "${REPO_ROOT}/rust/glossapi_rs_cleaner"
uv pip install --python "${VENV_PATH}/bin/python" -e "${REPO_ROOT}/rust/glossapi_rs_noise"

if [[ "${DOWNLOAD_MODEL}" -eq 1 ]]; then
  info "Downloading DeepSeek-OCR-2 model to ${MODEL_DIR}"
  HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}}" \
  "${VENV_PATH}/bin/python" - <<PY
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-OCR-2",
    repo_type="model",
    token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    local_dir=r"${MODEL_DIR}",
)
PY
else
  if [[ ! -d "${MODEL_DIR}" ]]; then
    warn "Model directory ${MODEL_DIR} is absent. Use --download-model or populate it manually."
  fi
fi

export GLOSSAPI_DEEPSEEK_PYTHON="${VENV_PATH}/bin/python"
export GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT="${REPO_ROOT}/src/glossapi/ocr/deepseek/run_pdf_ocr_transformers.py"
export GLOSSAPI_DEEPSEEK_MODEL_DIR="${MODEL_DIR}"

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  info "Running DeepSeek pytest subset"
  "${VENV_PATH}/bin/python" -m pytest -q -m "deepseek" tests
fi

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  info "Running DeepSeek smoke test"
  "${VENV_PATH}/bin/python" "${SCRIPT_DIR}/deepseek_gpu_smoke.py"
fi

cat <<EOF

DeepSeek uv environment ready.
Activate with:
  source "${VENV_PATH}/bin/activate"

Runtime exports:
  export GLOSSAPI_DEEPSEEK_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT="${REPO_ROOT}/src/glossapi/ocr/deepseek/run_pdf_ocr_transformers.py"
  export GLOSSAPI_DEEPSEEK_MODEL_DIR="${MODEL_DIR}"
EOF
