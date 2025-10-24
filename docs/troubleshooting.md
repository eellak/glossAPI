# Troubleshooting

## OCR runs on CPU

- Verify ONNXRuntime GPU: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` — must include `CUDAExecutionProvider`.
- Ensure CPU ORT wheel is not installed: `pip uninstall -y onnxruntime`.
- Make sure you pass `accel_type='CUDA'` (or `use_gpus='multi'`).

## Torch doesn’t see the GPU

- Check `nvidia-smi` and driver installation.
- Match Torch CUDA build to your driver; see getting_started.md for the recommended wheel.

## RapidOCR font download failure

- The first OCR call might download a visualization font. Ensure egress is allowed; the file is cached afterwards.

## Out of memory

- Lower Phase‑2 `batch_size` (e.g., 8) and reduce inline `GLOSSAPI_FORMULA_BATCH`.
- Reduce `GLOSSAPI_IMAGES_SCALE` (e.g., 1.1–1.2).
- Split large batches or files.

## Worker respawn limit reached

- When a GPU crashes repeatedly, the controller stops respawning it after `GLOSSAPI_MATH_RESPAWN_CAP` attempts. Any pending stems are added to the skip‑list and their inputs are copied to `downloads/problematic_math/` (PDFs) and `json/problematic_math/` (Docling artifacts); inspect those folders, address the issue, then rerun `Corpus.ocr(..., reprocess_completed=True)` or move the quarantined files back into `downloads/`.
- Check the corresponding worker log under `logs/math_workers/` (or the directory set via `GLOSSAPI_WORKER_LOG_DIR`) for stack traces and the active stem list stored in `gpu<N>.current`.

## Where are my files?

- Enriched Markdown overwrites `markdown/<stem>.md`.
- JSON/indices/latex maps: `json/`. Metrics: `json/metrics/`.
