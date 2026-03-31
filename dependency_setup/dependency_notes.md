# GlossAPI Dependency Profiles & Test Notes

## Environment Profiles
- **Docling** – main GlossAPI environment for extraction, cleaning, sectioning, annotation, and math/code enrichment. Uses `requirements-glossapi-docling.txt`.
- **DeepSeek** – dedicated OCR runtime managed with `uv`. Pins the tested Torch/Transformers stack in `dependency_setup/deepseek_uv/pyproject.toml` and intentionally excludes the Docling layout stack.

Recommended installation commands:
```bash
./dependency_setup/setup_glossapi.sh --mode docling --venv dependency_setup/.venvs/docling --run-tests
./dependency_setup/setup_deepseek_uv.sh --venv dependency_setup/.venvs/deepseek --run-tests
```

Key flags:
- `--download-model` optionally fetches DeepSeek weights (set `--model-root` if they live elsewhere).
- `--smoke-test` (DeepSeek only) runs `dependency_setup/deepseek_gpu_smoke.py`.

## Test Segmentation
Pytest markers were added so suites can be run per profile:
- `deepseek` – DeepSeek execution paths.
- Unmarked tests cover the Docling/core footprint.

Suggested commands:

| Profile   | Command |
|-----------|---------|
| Docling   | `pytest -q -m "not deepseek" tests` |
| DeepSeek  | `pytest -q -m "deepseek" tests` |

## Validation Runs (2026-03-08)
- `./dependency_setup/setup_glossapi.sh --mode docling --venv dependency_setup/.venvs/docling --run-tests`
- `./dependency_setup/setup_deepseek_uv.sh --venv dependency_setup/.venvs/deepseek --run-tests`
- `./dependency_setup/setup_deepseek_uv.sh --venv dependency_setup/.venvs/deepseek --smoke-test`

These completed successfully after the following adjustments:
1. **Rust extensions** – use editable installs for `rust/glossapi_rs_{cleaner,noise}` so local changes are picked up immediately.
2. **DeepSeek stack** – moved to a uv-managed runtime pinned to the `transformers`-based OCR-2 path.
3. **Attention fallback** – the DeepSeek runner falls back to `eager` attention if `flash-attn` is unavailable.

## Known Follow-ups
- **DeepSeek weights** – installer warns if weights are absent. Set `--download-model` or populate `${MODEL_ROOT}/DeepSeek-OCR-2` before running the real CLI tests (`GLOSSAPI_RUN_DEEPSEEK_CLI=1`).
- **flash-attn** – optional. Reintroduce into the pinned flow once wheel availability is stable across target hosts.
- **Patchelf warnings** – maturin emits rpath hints if `patchelf` is missing; they are benign but install `patchelf` if cleaner logs are desired.
- **Deprecation noise** – Docling and Transformers emit some warnings on current pins; currently harmless but worth tracking for future upgrades.

## Quick Reference
- Activate an environment: `source dependency_setup/.venvs/<profile>/bin/activate`
- Re-run tests manually:
  - Docling: `pytest -m "not deepseek" tests`
  - DeepSeek: `pytest -m "deepseek" tests`
- DeepSeek runtime exports:
  ```bash
  export GLOSSAPI_DEEPSEEK_PYTHON="dependency_setup/.venvs/deepseek/bin/python"
  export GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT="/mnt/data/glossAPI/src/glossapi/ocr/deepseek/run_pdf_ocr_transformers.py"
  export GLOSSAPI_DEEPSEEK_MODEL_DIR="/mnt/data/glossAPI/deepseek-ocr-2-model/DeepSeek-OCR-2"
  ```

These notes capture the current dependency state, the rationale behind constraint changes, and the validation steps used to exercise each profile.
