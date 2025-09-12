# Multi‑GPU & Benchmarking

GlossAPI can scale across multiple visible GPUs. Faster GPUs will drain more work using a shared queue.

## Extract (Phase‑1) on Multiple GPUs

```python
from glossapi import Corpus
c = Corpus('IN', 'OUT')
c.extract(input_format='pdf', use_gpus='multi', force_ocr=True)
```

- Workers are bound using `CUDA_VISIBLE_DEVICES=<id>` and run Docling on `cuda:0` relative to each worker.
- Threads auto‑tune when `num_threads=None` (roughly CPU / NGPUs). You can override explicitly.

## Phase‑2 (Math) on Multiple GPUs

```python
from glossapi import Corpus
c = Corpus('OUT', 'OUT')
c.ocr(force=True, use_gpus='multi', math_enhance=True, math_batch_size=12)
```

- Spawns math workers that bind to their GPU via `CUDA_VISIBLE_DEVICES`. Formula decoding runs on `cuda:0` relative to each worker.

## Provider & Device Checks

- ONNXRuntime providers must include `CUDAExecutionProvider`.
- Torch must see CUDA devices (`torch.cuda.is_available()` and device count).

## Benchmarking Tips

- Use `benchmark_mode=True` for `extract(...)` to skip per‑doc/page metrics (reduces I/O and profiling overhead).
- Pin `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to avoid CPU oversubscription on multi‑GPU nodes.
- Keep caches (`HF_HOME`, `XDG_CACHE_HOME`, `DOCLING_CACHE_DIR`) on fast disks.
