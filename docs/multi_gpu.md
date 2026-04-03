# Multi‑GPU & Benchmarking

GlossAPI can scale across multiple visible GPUs. Faster GPUs drain more work from a shared queue of **absolute
file paths**, so no worker rescans directories.

## Extract (Phase‑1) on Multiple GPUs

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', use_gpus='multi', phase1_backend='docling', workers_per_device=2)
```

- Workers are bound using `CUDA_VISIBLE_DEVICES=<id>` and run Docling on `cuda:0` relative to each worker.
- `workers_per_device` defaults to `1`; raise it only when benchmarking a strong GPU such as an A100.
- Threads auto‑tune when `num_threads=None` (roughly `min(cpu_count, 2 * #GPUs)`). Override explicitly if needed.
- The controller persists extraction progress in `download_results/download_results.parquet` after each reported
  batch, so interrupted runs can resume cleanly without ad-hoc checkpoint files.
- Worker batches requeue failed files and report `problematic` lists back to the controller, preventing silent loss.
- Periodic summaries log processed/problematic counts and queue size every ~30s for easier monitoring.

## Phase‑2 (Math) on Multiple GPUs

```python
from glossapi import Corpus
c = Corpus('OUT', 'OUT')
c.ocr(use_gpus='multi', math_batch_size=12)
```

- Spawns math workers that bind to their GPU via `CUDA_VISIBLE_DEVICES`. Formula decoding runs on `cuda:0` relative to each worker.
- Each worker writes a marker file (`logs/math_workers/gpu<N>.current`) containing the stems it is processing and keeps an append-only log in `logs/math_workers/` (override with `GLOSSAPI_WORKER_LOG_DIR`).
- Crashed workers are respawned automatically; control the retry budget per GPU with `GLOSSAPI_MATH_RESPAWN_CAP` (default `5`). Use `GLOSSAPI_WORKER_LOG_VERBOSE=0` to silence the banner that prints the binding info.
- When a device exceeds the respawn cap, remaining stems are added to the fatal skip-list and their artifacts are quarantined under `downloads/problematic_math/` and `json/problematic_math/` for follow-up.

## DeepSeek OCR on Multiple GPUs

```python
from glossapi import Corpus
c = Corpus("OUT", "OUT")
c.ocr(
    use_gpus="multi",
    runtime_backend="vllm",
    workers_per_gpu=1,
    scheduler="exact_fill",
    target_batch_pages=96,
)
```

- `scheduler="exact_fill"` is the preferred multi-GPU vLLM scheduler when PDFs vary widely in length. It shards large documents into page ranges and keeps GPU lanes filled more evenly.
- Internal shard runs now preserve the public `Corpus.ocr()` contract. Canonical outputs are reassembled back into `markdown/<stem>.md` and `json/metrics/<stem>.metrics.json` for each source PDF.
- When OCR starts from canonical corpus rows, the preferred stage handoff is also a canonical parquet where corrected `text` is embedded back into the same row identity. Markdown and metrics remain sidecars for inspection and audit.
- Shard markdown and shard metrics are retained for debugging under `sidecars/ocr_shards/` instead of remaining in the canonical handoff directories.
- The vLLM path now renders pages into memory and feeds a bounded queue directly into inference, which removes the temporary PNG round-trip and overlaps rendering with generation.
- Empty-page detection still happens before inference, and repair retries reuse the in-memory page image instead of reopening a file from disk.
- Final OCR markdown now tags each page split with `<!-- page:N -->` so page images, markdown, and metrics stay aligned during inspection.
- If a repair retry hits the garbage cutoff again, the page is blanked rather than keeping the failed first-pass garbage.
- Multi-GPU vLLM workers now pull from a durable shared batch queue in `sidecars/ocr_runtime/work_queue.sqlite`, so finished batches survive worker crashes and respawned workers can continue without rescanning completed work.
- Repair work now runs as a second global queue phase. First-pass batches finish and persist shard outputs first; then any worker can claim the queued repair shards. This keeps repair tails balanced across GPUs without mixing worker-local repair state into the controller.
- Workers may pack multiple pending repair items into one larger execution batch. Queue durability stays item-granular, but the runtime no longer has to execute the repair tail as one tiny origin-shard retry at a time.
- Each worker writes `sidecars/ocr_runtime/worker_*.runtime.json` with heartbeat state and steady-state timing markers. The runner also emits `gpu_preflight.json`, `gpu_telemetry.jsonl`, and `runtime_summary.json`.
- The runner checks GPU persistence mode before launch by default. Control it with `GLOSSAPI_DEEPSEEK_GPU_PREFLIGHT=off|warn|ensure`. The default is `ensure`, which will try `sudo -n nvidia-smi -pm 1` and record the result in `gpu_preflight.json`.
- When the DeepSeek runtime is built from wheel-managed CUDA packages, the runner now auto-discovers the venv's `site-packages/nvidia/*/lib` directories and prepends them to `LD_LIBRARY_PATH`. `GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH` still works as a manual override or supplement.
- Worker reliability knobs are environment-driven: `GLOSSAPI_DEEPSEEK_WORKER_RESPAWN_CAP`, `GLOSSAPI_DEEPSEEK_WORK_ITEM_MAX_ATTEMPTS`, `GLOSSAPI_DEEPSEEK_WORK_STALE_AFTER_SEC`, `GLOSSAPI_DEEPSEEK_WORK_HEARTBEAT_SEC`, and `GLOSSAPI_DEEPSEEK_TELEMETRY_INTERVAL_SEC`.
- The default `GLOSSAPI_DEEPSEEK_WORK_ITEM_MAX_ATTEMPTS=2` means one retry after the first failed claim, then the batch is marked failed instead of retrying forever.
- `workers_per_gpu=1` remains the safe default on A100 40GB nodes. Prefer increasing `target_batch_pages` before adding more workers per device.
- For fresh GCP A100 nodes, run `python -m glossapi.scripts.deepseek_runtime_report --repo-root <repo-root>` before applying ad hoc fixes. Treat that report as the baseline comparison against a known-good node. See [operations/deepseek_gcp_a100_setup.md](operations/deepseek_gcp_a100_setup.md).

## Provider & Device Checks

- ONNXRuntime providers must include `CUDAExecutionProvider`.
- Torch must see CUDA devices (`torch.cuda.is_available()` and device count).

## Benchmarking Tips

- Use `benchmark_mode=True` for `extract(...)` to skip per‑doc/page metrics (reduces I/O and profiling overhead).
- Pin `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to avoid CPU oversubscription on multi‑GPU nodes.
- Keep caches (`HF_HOME`, `XDG_CACHE_HOME`, `DOCLING_CACHE_DIR`) on fast disks.
