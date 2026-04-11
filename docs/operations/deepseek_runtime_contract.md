# DeepSeek Runtime Contract

This note documents the supported DeepSeek OCR runtime shape after the OCR
readability and maintainability refactor.

## Supported Execution Path

The supported OpenArchives-style OCR path is:

1. `python -m glossapi.scripts.openarchives_ocr_run_node`
2. `glossapi.ocr.deepseek.runner.run_for_files()`
3. `src/glossapi/ocr/deepseek/run_pdf_ocr_vllm.py`

That path is the production contract for throughput checks, regression
benchmarking, and future OCR maintenance work.

## Ownership Boundary

Keep the ownership split strict:

- `src/glossapi/corpus/ocr/`
  - corpus-facing OCR policy
  - parquet selection and result updates
  - OCR-vs-math mode handling
- `src/glossapi/ocr/deepseek/`
  - runtime/backend execution
  - vLLM worker launch, queueing, and shard reassembly
  - runtime defaults shared across scripts and callers

If a change needs parquet knowledge, it probably belongs in the corpus package.
If it needs worker-process or GPU knowledge, it probably belongs in the
DeepSeek runtime package.

## Shared Defaults

The canonical OCR runtime defaults now live in:

- `src/glossapi/ocr/deepseek/defaults.py`

The current validated defaults are:

- `runtime_backend='vllm'`
- `ocr_profile='markdown_grounded'`
- `render_dpi=144`
- `target_batch_pages=96`
- `workers_per_gpu=1`
- `gpu_memory_utilization=0.9`
- `repair_mode='auto'`
- `max_new_tokens=2048`

These defaults are consumed by:

- `src/glossapi/corpus/ocr/config.py`
- `src/glossapi/ocr/deepseek/launcher.py`
- `src/glossapi/ocr/deepseek/runner.py`
- `src/glossapi/scripts/openarchives_ocr_run_node.py`

Do not introduce a new default in only one of those paths.

## Runtime Environment

Use the dedicated DeepSeek uv environment created by:

```bash
./dependency_setup/setup_deepseek_uv.sh
```

Operational expectations:

- Python 3.11 environment
- vLLM-based OCR runner
- user-space CUDA runtime available from the environment, not only from the
  system image
- the DeepSeek model directory is supplied through
  `GLOSSAPI_DEEPSEEK_MODEL_DIR` or an equivalent explicit path

If you change the runtime dependency shape, update:

- the setup script
- this contract note
- the benchmark runbook
- at least one install/runtime smoke test

## Safe Change Checklist

Before merging an OCR runtime change:

1. Run the focused OCR/unit test slice.
2. Run the 3-PDF single-GPU benchmark from
   `docs/operations/deepseek_single_gpu_benchmarking.md`.
3. Record throughput and markdown sha256 values.
4. Sample the markdown outputs for legibility, not only hash drift.
5. Update docs if the ownership boundary, defaults, or operational entrypoint
   changed.

## Tests That Should Move With Runtime Changes

- `tests/test_deepseek_runner_contract.py`
- `tests/test_deepseek_multi_gpu_runtime.py`
- `tests/test_openarchives_ocr_run_node.py`
- `tests/test_openarchives_single_gpu_benchmark.py`

## Tests That Should Move With Corpus OCR Changes

- `tests/test_corpus_flow.py`
- `tests/test_ocr_dispatch_backends.py`
- `tests/test_math_policy.py`
- `tests/test_corpus_ocr_modules.py`
