#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="vanilla"
PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${GLOSSAPI_VENV:-}"
DOWNLOAD_DEEPSEEK=0
DEEPSEEK_ROOT="${DEEPSEEK_ROOT:-${REPO_ROOT}/deepseek-ocr}"
RUN_TESTS=0
RUN_SMOKE=0

usage() {
  cat <<'EOF'
Usage: setup_glossapi.sh [options]

Options:
  --mode MODE            Environment profile: vanilla, rapidocr, deepseek (default: vanilla)
  --venv PATH            Target virtual environment path
  --python PATH          Python executable to use when creating the venv
  --download-deepseek    Fetch DeepSeek-OCR weights (only meaningful for --mode deepseek)
  --weights-dir PATH     Destination directory for DeepSeek weights (default: $REPO_ROOT/deepseek-ocr)
  --run-tests            Run pytest -q after installation
  --smoke-test           Run dependency_setup/deepseek_gpu_smoke.py (deepseek mode only)
  --help                 Show this help message
EOF
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

case "${MODE}" in
  vanilla|rapidocr|deepseek) ;;
  *)
    echo "Invalid mode '${MODE}'. Expected vanilla, rapidocr, or deepseek." >&2
    exit 1
    ;;
esac

if [[ -z "${VENV_PATH}" ]]; then
  VENV_PATH="${REPO_ROOT}/.venv_glossapi_${MODE}"
fi

REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-glossapi-${MODE}.txt"
if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Requirements file not found for mode ${MODE}: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

info()  { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[err]\033[0m %s\n" "$*" >&2; exit 1; }

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

download_deepseek_weights() {
  local root="$1"
  local target="${root}/DeepSeek-OCR"

  if [[ -d "${target}" ]]; then
    info "DeepSeek-OCR weights already present at ${target}"
    return 0
  fi

  mkdir -p "${root}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    info "Downloading DeepSeek weights with huggingface-cli (this may take a while)"
    huggingface-cli download deepseek-ai/DeepSeek-OCR \
      --repo-type model \
      --include "DeepSeek-OCR/*" \
      --local-dir "${target}" \
      --local-dir-use-symlinks False || warn "huggingface-cli download failed; falling back to git-lfs"
  fi

  if [[ ! -d "${target}" ]]; then
    if command -v git >/dev/null 2>&1; then
      if ! command -v git-lfs >/dev/null 2>&1; then
        warn "git-lfs not available; install git-lfs to clone DeepSeek weights via git."
      else
        info "Cloning DeepSeek weights via git-lfs"
        git lfs install --skip-repo >/dev/null 2>&1 || true
        git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR "${target}"
      fi
    else
      warn "Neither huggingface-cli nor git found; skipping DeepSeek weight download."
    fi
  fi

  if [[ ! -d "${target}" ]]; then
    warn "DeepSeek weights were not downloaded. Set DEEPSEEK_ROOT manually once acquired."
  fi
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

if [[ "${MODE}" == "deepseek" ]]; then
  export GLOSSAPI_DEEPSEEK_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_DEEPSEEK_VLLM_SCRIPT="${DEEPSEEK_ROOT}/run_pdf_ocr_vllm.py"
  export GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH="${DEEPSEEK_ROOT}/libjpeg-turbo/lib"
  export GLOSSAPI_DEEPSEEK_ALLOW_STUB=0
  export LD_LIBRARY_PATH="${GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

  if [[ "${DOWNLOAD_DEEPSEEK}" -eq 1 ]]; then
    download_deepseek_weights "${DEEPSEEK_ROOT}"
  else
    warn "DeepSeek weights not downloaded (use --download-deepseek to fetch automatically)."
  fi
fi

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  pytest_args=("-q")
  case "${MODE}" in
    vanilla)
      pytest_args+=("-m" "not rapidocr and not deepseek")
      ;;
    rapidocr)
      pytest_args+=("-m" "not deepseek")
      ;;
    deepseek)
      pytest_args+=("-m" "not rapidocr")
      ;;
  esac

  info "Running pytest ${pytest_args[*]} tests"
  python_run -m pytest "${pytest_args[@]}" tests
fi

if [[ "${MODE}" == "deepseek" && "${RUN_SMOKE}" -eq 1 ]]; then
  info "Running DeepSeek smoke test"
  python_run "${SCRIPT_DIR}/deepseek_gpu_smoke.py"
fi

cat <<EOF

Environment ready (${MODE}).
Activate with:
  source "${VENV_PATH}/bin/activate"

EOF

if [[ "${MODE}" == "deepseek" ]]; then
  cat <<EOF
DeepSeek-specific exports (add to your shell before running glossapi):
  export GLOSSAPI_DEEPSEEK_PYTHON="${VENV_PATH}/bin/python"
  export GLOSSAPI_DEEPSEEK_VLLM_SCRIPT="${DEEPSEEK_ROOT}/run_pdf_ocr_vllm.py"
  export GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH="${DEEPSEEK_ROOT}/libjpeg-turbo/lib"
  export GLOSSAPI_DEEPSEEK_ALLOW_STUB=0
  export LD_LIBRARY_PATH="\$GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH:\${LD_LIBRARY_PATH:-}"
EOF
fi
