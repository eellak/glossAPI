# DeepSeek-OCR vLLM Runner Notes

## Repo snapshot (2025-10-26 export)
- Runner scripts: `run_pdf_ocr_vllm.py` (high-throughput vLLM path) and `run_pdf_ocr.py` (single-process baseline).
- Model scaffold: `DeepSeek-OCR-empty/` mirrors the checkpoint layout; download `deepseek-ai/DeepSeek-OCR` and unpack it here so `model-00001-of-000001.safetensors` replaces the 135-byte placeholder.
- AWS ops files live under `aws/` (bootstrap script, deployment checklist, multi-GPU plan, sample PDF manifest).
- Notes (this file) carry the operational history, tuning tips, and install timings. Update here if you change dependencies or infra recipes.

## Quickstart (fresh clone)
1. Fetch weights: `huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir DeepSeek-OCR --local-dir-use-symlinks False`. (Leave `DeepSeek-OCR-empty/` in place or remove it after you confirm the real weights are present.)
2. Create a Python 3.12 environment (`conda create -n deepseek-ocr python=3.12` or reuse the AWS bootstrap script). Activate it and run:
   ```bash
   pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu124 torch torchvision torchaudio
   pip install --pre --extra-index-url https://wheels.vllm.ai/nightly vllm
   pip install PyMuPDF pillow numpy rich tqdm huggingface-hub safetensors "transformers>=4.55"
   pip install 'huggingface_hub[hf_transfer]'
   ```
3. (Optional) Build `libjpeg-turbo` 3.0.3 + NASM 2.16 if you need Pillow-SIMD speedups; export `LD_LIBRARY_PATH=/path/to/libjpeg-turbo/lib:$LD_LIBRARY_PATH` before invoking the runners.
4. For multi-GPU AWS runs, copy `aws/setup_multi_gpu_env.sh` to the instance, run with sudo, then follow the checklist (`aws/deployment_checklist.md`). Sample PDFs and metadata are listed in `aws/sample_pdfs/greek_pdf_samples.json`.
5. Run a smoke test (replace `/path/to/pdfs` with a directory containing the sample PDFs from the manifest):
   ```bash
   python run_pdf_ocr_vllm.py --input-dir /path/to/pdfs --output-dir ./outputs --mode clean --max-pages 2
   ```

## Environment quirks
- Pillow-SIMD needs system JPEG headers. Build `libjpeg-turbo` 3.0.3 from source (with NASM 2.16.01) and export `LD_LIBRARY_PATH=/path/to/libjpeg-turbo/lib:$LD_LIBRARY_PATH` before running anything that touches Pillow.
- vLLM nightly currently requires the CUDA 12.9 stack (`torch/torchvision/torchaudio==2.9.0`). Leaving the checklist item for CUDA 11.8 wheels unchecked until an upstream build supports them.
- `flash-attn==2.7.3` installs cleanly once the CUDA 12.9 toolchain is active; keep the `--no-build-isolation` flag and reuse the custom `LD_LIBRARY_PATH`.
- Latest flash-attn build (SSM command `a3f3845c-1e2e-405c-a90a-1ff096a09672`) ran `pip install flash-attn==2.7.4.post1 --no-build-isolation` via `AWS-RunShellScript` on `deepseek-ocr-g6` and completed in roughly 43 minutes, so plan on ~45 minutes for future compiles.

## Runner behaviour
- `run_pdf_ocr_vllm.py` now handles PDF rendering, blank-page detection, batching, FP8 KV cache defaults, and logs per-run throughput. Check `/tmp/run_vllm_full.log` (or custom log path) after runs for detailed timing.
- After clearing stale GPU processes and enabling blank-page short-circuiting, the latest smoke run (JSM_564 + PXJ_747, 5 pages) finished in 12.6 s (~0.79 pages/s; see `tmp_pdf_runs/test_run3.log`). Use this as the healthy baseline for short bursts; a full corpus pass on 2025-10-25 processed 724 pages in 1661 s (~0.44 pages/s, log `tmp_pdf_runs/run_bf16_full_20251025_152925.log`).
- Increase `--batch-pages` or enable `--enable-fp8-weights` to explore higher throughput once quality is validated; keep `--gpu-memory-utilization` near 0.95 for the L4.
- Combined Markdown files live directly under the output directory, named `<pdf_stem>.md` (e.g. `AAB_001.md`). Page-level `.md` dumps are no longer written; optional assets (page PNGs or ROI crops) are stored under `<pdf_stem>_assets/`.
- Multi-GPU prep: `--num-shards N` and `--shard-index i` fan out PDF batches across workers/GPUs; combine with per-worker `CUDA_VISIBLE_DEVICES`, `--tensor-parallel-size` (if sharing a process across multiple devices), and optional `--mm-encoder-tp-mode data` to exercise vLLM’s multimodal encoder sharding.
- All dev smoke runs now wrap the CLI in `timeout 300 …` so we fail fast if decoding stalls.

## Implemented enhancements
- Split prompts/modes: grounded runs use `<|grounding|>` and keep `<|ref|>/<|det|>` blocks, while clean runs strip meta tokens and emit Markdown-only text.
- Added tokenizer whitelist for `<table>/<tr>/<th>/<td>` tags so the DeepSeek n-gram guard no longer suppresses short HTML tokens.
- Upgraded `clean_output` to remove prompt echoes, `<image>` tokens, image captions, redundant bounding-box prefixes, spammy LaTeX/TikZ blocks, and tokenizer-declared special/meta tokens (tracked via a dynamic regex cache).
- Placeholder-safe tables: regex post-processing strips `None`/`N/A`/dash filler cells, collapses whitespace-only `<td>`s, and drops fully empty tables. (Logit bias guardrails are temporarily disabled as of 2025-10-26.)
- Combined outputs per PDF (`combined.md` / `combined_grounded.md`) with `----- Page N -----` headings starting from page 2 so Markdown previews render correctly.
- Canonical markdown pass trims multi-blank lines, de-hyphenates wrapped words, drops empty tables, and remaps superscript citations (`<sup>71</sup>`) to GitHub footnotes (`[^71]`).
- Fast blank-page detector (`is_mostly_blank_pix`) skips inference on empty renders and writes a `[[Blank page]]` stub, preventing hallucinated content on blank scans.
- Optional ROI second pass: grounded runs can crop table/title/paragraph/figure regions and re-infer them with the clean prompt for higher fidelity.
- Optional Large-mode retry: pages missing required labels (default `table`) re-run in the 1280 “Large” vision mode to recover objects missed by the standard tiler; the Large LLM is now reused across PDFs and capped via `--retry-large-cap` to avoid runaway retries.
- Per-page logging now records token counts/preview text and appends lightweight debug stats to `/tmp/debug.txt` during development.
- Decoding guardrails: sampler uses a shared stop list for `<|...|>` fragments and logs a post-run sanity summary (`placeholder_cells_pruned=…`, `tables_dropped=…`, etc.); placeholder logit bias is currently disabled pending a safer design.
- Supports BF16 baseline with FP8 KV cache by default, plus an `--enable-fp8-weights` switch for experimentation (quality needs manual review).
- Sharding helpers keep runner scripts multi-GPU ready without orchestration: each worker sets `CUDA_VISIBLE_DEVICES`, passes its shard index/count, and can increase `--tensor-parallel-size` when sharing an engine across local GPUs.

## Follow-ups
- Replace the temporary `LD_LIBRARY_PATH` step with a wrapper script or env file to avoid manual exports.
- Revisit the CUDA 11.8 wheel requirement when a matching vLLM build becomes available; that will let us tick the outstanding checklist item.
- Validate the multi-GPU sharding flow on a 4× L4 block (one worker per GPU) and capture throughput deltas vs. single-device runs.
- Rework prompt-level placeholder instructions/logit bias; both were reverted on 2025-10-26 after causing “Here is the text…” hallucinations. Current runs use the minimal base prompt with no placeholder bias.
- Document 2025-10-26 AWS bootstrap adjustments: instance now uses IAM role `deepseek-ocr-ssm-role` + instance profile for SSM, CloudWatch, and S3 logging so long installs (e.g., flash-attn builds) can run via `aws ssm send-command`.
- FlashAttention build remains a pain point: no prebuilt wheel for Python 3.12 + CUDA 12.4, so `pip install flash-attn==2.7.4.post1 --no-build-isolation` compiles from source (~45–60 min). SSM command `a3f3845c-…` kicked off the build; monitor with `aws ssm get-command-invocation …` or CloudWatch `/deepseek-ocr/ssm` once the stream appears.
