#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="docling"
PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${GLOSSAPI_VENV:-}"
DOWNLOAD_DEEPSEEK=0
DEEPSEEK_ROOT="${DEEPSEEK_ROOT:-${REPO_ROOT}/deepseek-ocr-2-model}"
RUN_TESTS=0
RUN_SMOKE=0

info()  { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[err]\033[0m %s\n" "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: setup_glossapi.sh [options]

Options:
  --mode MODE            Environment profile: docling or deepseek (default: docling)
  --venv PATH            Target virtual environment path
  --python PATH          Python executable to use when creating the venv
  --download-deepseek    Fetch DeepSeek-OCR-2 weights (DeepSeek mode only)
  --weights-dir PATH     Destination directory root for DeepSeek weights (default: $REPO_ROOT/deepseek-ocr-2-model)
  --run-tests            Run pytest -q after installation
  --smoke-test           Run dependency_setup/deepseek_gpu_smoke.py (deepseek mode only)
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
    error "Python interpreter ${python_bin} is not a stable final release (releaselevel=${release_level}). Install a stable CPython and rerun with --python."
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
    --mode)
      shift || { echo "--mode requires a value" >&2; exit 1; }
      MODE="${1:-}"
      ;;
    --venv)
      shift || { echo "--venv requires a path" >&2; exit 1; }
      VENV_PATH="${1:-}"
      ;;
    --python)
      shift || { echo "--python requires a path" >&2; exit 1; }
      PYTHON_BIN="${1:-}"
      ;;
    --download-deepseek)
      DOWNLOAD_DEEPSEEK=1
      ;;
    --weights-dir)
      shift || { echo "--weights-dir requires a path" >&2; exit 1; }
      DEEPSEEK_ROOT="${1:-}"
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
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || error "Python executable not found: ${PYTHON_BIN}"
ensure_stable_python "${PYTHON_BIN}"
check_rust_toolchain

case "${MODE}" in
  vanilla)
    warn "Mode 'vanilla' is deprecated; using 'docling' instead."
    MODE="docling"
    ;;
  docling|deepseek) ;;
  *)
    echo "Invalid mode '${MODE}'. Expected docling or deepseek." >&2
    exit 1
    ;;
esac

if [[ "${MODE}" == "deepseek" ]]; then
  exec "${SCRIPT_DIR}/setup_deepseek_uv.sh" \
    --python "${PYTHON_BIN}" \
    --venv "${VENV_PATH:-${REPO_ROOT}/dependency_setup/.venvs/deepseek}" \
    --model-root "${DEEPSEEK_ROOT}" \
    $([[ "${DOWNLOAD_DEEPSEEK}" -eq 1 ]] && printf '%s' "--download-model") \
    $([[ "${RUN_TESTS}" -eq 1 ]] && printf '%s' "--run-tests") \
    $([[ "${RUN_SMOKE}" -eq 1 ]] && printf '%s' "--smoke-test")
fi

if [[ -z "${VENV_PATH}" ]]; then
  VENV_PATH="${REPO_ROOT}/.venv_glossapi_${MODE}"
fi

REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-glossapi-${MODE}.txt"
if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Requirements file not found for mode ${MODE}: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

ensure_venv() {
  if [[ ! -d "${VENV_PATH}" ]]; then
    info "Creating virtual environment at ${VENV_PATH}"
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
  else
    info "Reusing existing virtual environment at ${VENV_PATH}"
  fi
}

pip_run() {
  "${VENV_PATH}/bin/pip" "$@"
}

python_run() {
  "${VENV_PATH}/bin/python" "$@"
}

ensure_venv
info "Upgrading pip tooling"
pip_run install --upgrade pip wheel setuptools

info "Installing ${MODE} requirements from $(basename "${REQUIREMENTS_FILE}")"
pip_run install -r "${REQUIREMENTS_FILE}"

info "Installing glossapi in editable mode"
pip_run install -e "${REPO_ROOT}" --no-deps

info "Building Rust extensions via editable installs"
pip_run install -e "${REPO_ROOT}/rust/glossapi_rs_cleaner"
pip_run install -e "${REPO_ROOT}/rust/glossapi_rs_noise"

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  pytest_args=("-q")
  case "${MODE}" in
    docling)
      pytest_args+=("-m" "not deepseek")
      ;;
  esac

  info "Running pytest ${pytest_args[*]} tests"
  python_run -m pytest "${pytest_args[@]}" tests
fi

cat <<EOF

Environment ready (${MODE}).
Activate with:
  source "${VENV_PATH}/bin/activate"

EOF
