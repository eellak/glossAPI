# DeepSeek runtime contract

Status: vLLM is the supported runtime backend. The transformers backend is
preserved as best-effort and is currently broken with the upstream
DeepSeek-OCR-2 bundled modeling code (see "Known issues" below).

## Supported runtime

`runtime_backend = "vllm"` is the default and only supported value.

The DeepSeek-OCR-2 model is served via the script
`src/glossapi/ocr/deepseek/run_pdf_ocr_vllm.py`, which loads the model with
vLLM's own model loader and exposes the model through vLLM's batched
inference API. The runner subprocesses this script per lane and reads back
markdown plus per-page metrics from disk.

## Page-level efficiency contract

The vLLM runtime ships with a blank-page skip guard:

- `_is_effectively_empty_page(image_stats, repair_mode)` runs on the
  pre-rendered image stats (overall and per-region dark ratios). When
  `repair_mode == "auto"` (default) and the page falls below the configured
  brightness thresholds, the runtime emits a synthetic page metric
  (`repair_strategy="skip_empty"`, `empty_page_skipped=True`, `infer_sec=0.0`)
  and counts it in the aggregate `empty_pages_skipped`. No model forward pass
  happens for skipped pages.

This guard does NOT exist on the transformers script. It is one of the
reasons the supported backend is vLLM.

## Backend choice — why vLLM

- vLLM ships the blank-page skip guard described above.
- vLLM uses its own model loader and does not exercise the transformers
  dynamic-module path that breaks on DeepSeek-OCR-2 with current upstream
  transformers.
- vLLM enables batched inference across multi-GPU lanes via the `exact_fill`
  scheduler in `src/glossapi/ocr/deepseek/scheduling.py`.

Reference benchmark on 2× A100 SXM4 40GB (`a2-highgpu-2g`, us-west1-b):
- 10 OpenArchives PDFs, 683 pages, scheduler `exact_fill`, target 160 pages
  per batch.
- vLLM wall time: 276.16 s (4 min 36 s); 0.65–0.76 s/page per GPU.
- Auto-repair flagged 86 pages and successfully repaired 85 of them.

## Replacing or extending the backend

The runner is a subprocess-per-script architecture. To add a new backend:

1. Add a new `run_pdf_ocr_<backend>.py` script in
   `src/glossapi/ocr/deepseek/`.
2. Wire its CLI surface into `runner.py`'s
   `_build_cli_command` (and `defaults.py` if the backend introduces
   defaults).
3. Add a runtime choice in `defaults.DEFAULT_RUNTIME_BACKEND` and the
   acceptance check at `runner.py:run_for_files`.
4. Document the contract here.

The `scheduling.py` page router and `work_queue.py` durable batch queue are
backend-agnostic and consume the same `WorkSlice` / `(doc_id, page_number)`
abstractions regardless of which inference backend runs.

## Known issues

- **transformers backend is broken** with the version pulled transitively
  by `vllm==0.18.0`. The DeepSeek-OCR-2 bundled `modeling_deepseekv2.py`
  imports `LlamaFlashAttention2` from `transformers.models.llama.modeling_llama`,
  which was removed upstream in transformers ≥ 4.46. The transformers script
  also requires `matplotlib` at first import, which is not declared in the
  `deepseek` extra. We do not fix these here; the supported backend is vLLM.

## Testing

- `tests/test_deepseek_runner_contract.py` — runner contract tests.
- `tests/test_ocr_dispatch_backends.py` — dispatch tests.
- `tests/test_deepseek_scheduling.py` — scheduling tests.
- `dependency_setup/deepseek_gpu_smoke.py` — minimal real-GPU smoke test.
- `src/glossapi/scripts/deepseek_pipeline_benchmark.py` — full pipeline
  benchmark with per-GPU and per-lane metrics; supports
  `--scheduler {whole_doc, fixed_shard, exact_fill}` and
  `--runtime-backend {vllm, transformers}`.
