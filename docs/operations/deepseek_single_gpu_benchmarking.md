# DeepSeek Single-GPU Benchmarking

This runbook defines the benchmark shape to use before and after DeepSeek OCR
changes. The goal is to keep throughput comparisons honest and exactness checks
easy to reproduce.

## Standard Benchmark Shape

Use the OpenArchives lane runner shape, even for a single GPU:

- entrypoint: `python -m glossapi.scripts.openarchives_ocr_run_node`
- runtime: `vllm`
- scheduler: `exact_fill`
- target batch pages: `96`
- workers per GPU: `1`
- OCR profile: `markdown_grounded`
- render DPI: `144`
- GPU memory utilization: `0.9`
- max new tokens: `2048`
- repair mode: `auto`

These values are now shared in code through:

- `src/glossapi/ocr/deepseek/defaults.py`
- `src/glossapi/ocr/deepseek/launcher.py`
- `src/glossapi/ocr/deepseek/runner.py`

If you need the ownership map for those files, read:

- `docs/architecture/corpus_ocr_stack.md`
- `docs/architecture/deepseek_runner_stack.md`
- `docs/operations/deepseek_runtime_contract.md`

## Environment Requirements

Use the dedicated DeepSeek uv runtime:

```bash
./dependency_setup/setup_deepseek_uv.sh \
  --venv dependency_setup/.venvs/deepseek \
  --model-root /path/to/deepseek-ocr-2-model
```

Important runtime details:

- the DeepSeek launcher must expose both `torch/lib` and wheel-managed CUDA
  libraries on `LD_LIBRARY_PATH`
- the uv runtime now installs `nvidia-cuda-runtime-cu12` so the vLLM extension
  has a user-space `libcudart.so.12` even on boxes without a system CUDA toolkit

## Work-Root Contract

If you pass `--skip-download`, the `work_root` must already contain:

- `downloads/` with the PDF payloads
- the shard parquet used for row selection

The simplest pattern is:

```bash
mkdir -p "$RUN_ROOT"
cp smoke.parquet "$RUN_ROOT"/smoke.parquet
ln -s /path/to/downloads "$RUN_ROOT"/downloads
```

## Launch Command

```bash
export GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT="$PWD/src/glossapi/ocr/deepseek/run_pdf_ocr_vllm.py"

python -m glossapi.scripts.openarchives_ocr_run_node \
  --shard-parquet "$RUN_ROOT/smoke.parquet" \
  --work-root "$RUN_ROOT" \
  --skip-download \
  --runtime-backend vllm \
  --ocr-profile markdown_grounded \
  --repair-mode auto \
  --scheduler exact_fill \
  --target-batch-pages 96 \
  --workers-per-gpu 1 \
  --render-dpi 144 \
  --max-new-tokens 2048 \
  --gpu-memory-utilization 0.9
```

## Throughput Calculation

Benchmark from runtime artifacts, not from ad hoc wall-clock estimates.

Use:

- `sidecars/ocr_runtime/runtime_summary.json`
- `sidecars/ocr_runtime/*.runtime.json`
- `json/metrics/*.metrics.json`

Recommended formulas:

- `overall_sec_per_page = steady_state_window_seconds / total_pages`
- `sec_per_page_per_gpu = steady_state_window_seconds * workers / total_pages`

Where:

- `steady_state_window_seconds = last_batch_finished_at - first_batch_started_at`
- `total_pages` comes from the OCR metrics or direct PDF page counts for the
  exact shard input

Cold-start model load and compile time should be tracked separately from the
steady-state OCR window.

## Exactness Check

Run the same shard twice into fresh work roots and compare canonical markdown:

```bash
find "$RUN_A/markdown" -type f -name '*.md' -print0 | sort -z | xargs -0 sha256sum > run_a.sha256
find "$RUN_B/markdown" -type f -name '*.md' -print0 | sort -z | xargs -0 sha256sum > run_b.sha256
diff -u run_a.sha256 run_b.sha256
```

Record the sha256 manifest diff alongside the throughput numbers.

Observed result on April 11, 2026:

- baseline run root: `oa_single_gpu_smoke_ultra_base2`
- patched run root: `oa_single_gpu_smoke_ultra_patch1`
- repeated patched probe: `fqa_vllm_patch2`
- hardware: `glossapi-a2-ultra` in `europe-west4-a` with `1x A100`
- benchmark set: `IBK_476` (`591` pages), `FQA_524` (`576` pages),
  `ZKI_504` (`566` pages)
- baseline overall throughput: `0.4847 sec/page`
- patched overall throughput: `0.4784 sec/page`
- `FQA_524` patched rerun throughput: `0.4699 sec/page`
- `FQA_524` sha256 values differed across all three runs

That means the single-GPU vLLM path met the throughput target on April 11, 2026,
but did not reproduce byte-identical markdown across repeated reruns. Treat the
hash diff as a required benchmark artifact, not as a guaranteed passing gate,
until the underlying byte instability is explained.
