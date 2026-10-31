#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${GLOSSAPI_VENV:-${REPO_ROOT}/.venv_glossapi_core}"
DRY_RUN=0
RUN_TESTS=0
INSTALL_ARGS=()
usage() {
  cat <<'EOF'
Usage: setup_glossapi.sh [core|download|browser|layout|annotate ...] [options]
Default: core only. Layout defaults to cpu.
  --packages NAMES       Comma-separated optional packages
  --device DEVICE        cpu | cuda126 | cuda132 | mps (layout only)
  --pypdfium2-version V   Pin a qualified >=4.30,<5 release
  --venv PATH            Target virtual environment
  --python PATH          Python 3.11–3.14 executable used to create it
  --editable             Install local source in editable mode
  --run-tests            Install test tools and run the complete suite
  --dry-run              Print every exact command, without creating/installing
  --mode core|docling    Compatibility alias; docling selects layout
EOF
}
while (( "$#" )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --venv|--python|--mode|--packages|--device|--pypdfium2-version)
      option="$1"; shift
      [[ $# -gt 0 && -n "$1" ]] || { echo "${option} requires a value" >&2; exit 2; }
      case "${option}" in
        --venv) VENV_PATH="$1" ;;
        --python) PYTHON_BIN="$1" ;;
        --mode)
          case "$1" in
            core) INSTALL_ARGS+=(core) ;;
            docling) INSTALL_ARGS+=(layout) ;;
            *) echo 'Mode must be core or docling; select named packages for other features.' >&2; exit 2 ;;
          esac ;;
        *) INSTALL_ARGS+=("${option}" "$1") ;;
      esac ;;
    --run-tests) RUN_TESTS=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) INSTALL_ARGS+=("$1") ;;
  esac
  shift
done
# Resolve relative paths before printing commands (the installer uses absolute paths).
VENV_PATH="$("${PYTHON_BIN}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${VENV_PATH}")"
VENV_PYTHON="${VENV_PATH}/bin/python"
run() {
  printf '%q ' "$@"; printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then "$@"; fi
}
# Validate package/device choices before any mutation, including venv creation.
"${PYTHON_BIN}" "${REPO_ROOT}/install_glossapi.py" "${INSTALL_ARGS[@]}" --dry-run >/dev/null
if [[ ! -x "${VENV_PYTHON}" ]]; then
  run "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  # Render commands for the future interpreter without needing a venv to exist.
  "${PYTHON_BIN}" - "${REPO_ROOT}" "${VENV_PYTHON}" "${INSTALL_ARGS[@]}" <<'PY'
import sys
from pathlib import Path
repo, executable, *args = sys.argv[1:]
sys.path.insert(0, str(Path(repo) / 'src'))
from glossapi.scripts.install_glossapi import main
sys.executable = executable
raise SystemExit(main([*args, '--dry-run']))
PY
else
  "${VENV_PYTHON}" "${REPO_ROOT}/install_glossapi.py" "${INSTALL_ARGS[@]}"
fi
if [[ "${RUN_TESTS}" -eq 1 ]]; then
  run "${VENV_PYTHON}" -m pip install --index-url https://pypi.org/simple -r "${SCRIPT_DIR}/requirements-tooling.txt"
  run "${VENV_PATH}/bin/uv" pip install --python "${VENV_PYTHON}" --index-url https://pypi.org/simple --group "${REPO_ROOT}/pyproject.toml:test"
  TEST_TEMP="${VENV_PATH}/pytest-basetemp"
  # The test group includes the pinned `reference` oracles, so a missing oracle fails rather than skips.
  (cd "${REPO_ROOT}"; run env GLOSSAPI_REFERENCE_REQUIRED=1 "${VENV_PYTHON}" -m pytest -q "--basetemp=${TEST_TEMP}")
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${VENV_PYTHON}" -c 'import shutil,sys; shutil.rmtree(sys.argv[1])' "${TEST_TEMP}"
  fi
fi
