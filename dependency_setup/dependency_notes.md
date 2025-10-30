# GlossAPI Dependency Profiles & Test Notes

## Environment Profiles
- **Vanilla** – core GlossAPI pipeline without GPU OCR add-ons. Uses `requirements-glossapi-vanilla.txt`.
- **RapidOCR** – Docling + RapidOCR GPU stack. Builds on vanilla requirements and adds ONNX runtime (`requirements-glossapi-rapidocr.txt`).
- **DeepSeek** – GPU OCR via DeepSeek/vLLM. Extends vanilla requirements with torch/cu128, nightly vLLM and supporting CUDA libs (`requirements-glossapi-deepseek.txt`). `xformers` was dropped because the published wheels still pin Torch 2.8; the rest of the stack now installs cleanly on Torch 2.9.

Each profile is installed through `dependency_setup/setup_glossapi.sh`:
```bash
# Examples (venv path optional)
./dependency_setup/setup_glossapi.sh --mode vanilla  --venv dependency_setup/.venvs/vanilla  --run-tests
./dependency_setup/setup_glossapi.sh --mode rapidocr --venv dependency_setup/.venvs/rapidocr --run-tests
./dependency_setup/setup_glossapi.sh --mode deepseek --venv dependency_setup/.venvs/deepseek --run-tests
```

Key flags:
- `--download-deepseek` optionally fetches DeepSeek weights (skipped by default; set `--weights-dir` if they live elsewhere).
- `--smoke-test` (DeepSeek only) runs `dependency_setup/deepseek_gpu_smoke.py`.

## Test Segmentation
Pytest markers were added so suites can be run per profile:
- `rapidocr` – GPU Docling/RapidOCR integration tests.
- `deepseek` – DeepSeek execution paths.
- Unmarked tests cover the vanilla footprint.

`setup_glossapi.sh` now chooses marker expressions automatically:

| Mode      | Command run by script                                   |
|-----------|---------------------------------------------------------|
| vanilla   | `pytest -q -m "not rapidocr and not deepseek" tests`    |
| rapidocr  | `pytest -q -m "not deepseek" tests`                     |
| deepseek  | `pytest -q -m "not rapidocr" tests`                     |

Heavy GPU tests in `tests/test_pipeline_smoke.py` were guarded with `pytest.importorskip("onnxruntime")` so vanilla installs skip them cleanly. Helper PDFs now embed DejaVuSans with Unicode support and insert spacing to keep OCR-friendly glyphs.

## Validation Runs (2025-10-30)
- `./dependency_setup/setup_glossapi.sh --mode vanilla  --venv dependency_setup/.venvs/vanilla  --run-tests`
- `./dependency_setup/setup_glossapi.sh --mode rapidocr --venv dependency_setup/.venvs/rapidocr --run-tests`
- `./dependency_setup/setup_glossapi.sh --mode deepseek --venv dependency_setup/.venvs/deepseek --run-tests`

All three completed successfully after the following adjustments:
1. **Rust extensions** – switched to `pip install -e rust/glossapi_rs_{cleaner,noise}` because `maturin develop` left the wheel unregistered.
2. **Parquet locking** – `_parquet_lock` now creates parent directories before attempting the file lock (fixes `FileNotFoundError` in concurrent metadata tests).
3. **RapidOCR pipeline** – fixed `GlossExtract.create_extractor()` to build the Docling converter regardless of import path and added UTF-8 PDF generation improvements; smoke tests now pass on CUDA.
4. **DeepSeek stack** – updated nightly vLLM pin (`0.11.1rc5.dev58+g60f76baa6.cu129`) and removed `xformers` to resolve Torch 2.9 dependency conflicts.

## Known Follow-ups
- **DeepSeek weights** – installer warns if weights are absent. Set `--download-deepseek` or populate `${DEEPSEEK_ROOT}/DeepSeek-OCR` before running the real CLI tests (`GLOSSAPI_RUN_DEEPSEEK_CLI=1`).
- **xformers kernels** – removed pending compatible Torch 2.9 wheels. Reintroduce once upstream publishes matching builds.
- **Patchelf warnings** – maturin emits rpath hints if `patchelf` is missing; they are benign but install `patchelf` if cleaner logs are desired.
- **Deprecation noise** – Docling emits future warnings (Pydantic) and RapidOCR font deprecation notices; currently harmless but worth tracking for future upgrades.

## Quick Reference
- Activate an environment: `source dependency_setup/.venvs/<mode>/bin/activate`
- Re-run tests manually:
  - Vanilla: `pytest -m "not rapidocr and not deepseek" tests`
  - RapidOCR: `pytest -m "not deepseek" tests`
  - DeepSeek: `pytest -m "not rapidocr" tests`
- DeepSeek runtime exports:
  ```bash
  export GLOSSAPI_DEEPSEEK_PYTHON="dependency_setup/.venvs/deepseek/bin/python"
  export GLOSSAPI_DEEPSEEK_VLLM_SCRIPT="/mnt/data/glossAPI/deepseek-ocr/run_pdf_ocr_vllm.py"
  export GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH="/mnt/data/glossAPI/deepseek-ocr/libjpeg-turbo/lib"
  export LD_LIBRARY_PATH="$GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
  ```

These notes capture the current dependency state, the rationale behind constraint changes, and the validation steps used to exercise each profile.
