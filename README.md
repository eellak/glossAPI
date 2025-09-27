# GlossAPI

[![PyPI Status](https://img.shields.io/pypi/v/glossapi?logo=pypi)](https://pypi.org/project/glossapi/)

A library for processing texts in Greek and other languages, developed by [Open Technologies Alliance(GFOSS)](https://gfoss.eu/).

## Features

- **Document Processing**: Extract text content from academic PDFs, DOCX, HTML, and other formats with structure preservation
- **Document Downloading**: Download documents from URLs with automatic handling of various formats
- **Quality Control**: Assess extraction quality using fast Rust-based noise metrics and automatically filter low-quality documents
- **Section Extraction**: Identify and extract academic sections from documents
- **Section Classification**: Classify sections using machine learning models
- **Greek Language Support**: Specialized processing for Greek academic texts
- **Metadata Handling**: Process academic texts with accompanying metadata
- **Customizable Annotation**: Map section titles to standardized categories
- **Flexible Pipeline**: Start the processing from any stage in the pipeline, now with selectable extraction profiles (PyPDFium for safety, Docling for throughput)

## Installation

```bash
# Optional: block ~/.local packages from leaking into the environment
export PYTHONNOUSERSITE=1
pip install glossapi
```

### Local Environment Setup (from repo, recommended)

The steps below set up a clean venv, install GlossAPI from source, and ensure GPU OCR works.

Prerequisites
- Python 3.8+
- NVIDIA driver with CUDA 12.x recommended (check with `nvidia-smi`)
- Rust toolchain (required for Rust extensions: noise + cleaner) and maturin

Install Rust + maturin (once per machine):
```bash
# Rust toolchain
curl https://sh.rustup.rs -sSf | sh -s -- -y
. "$HOME/.cargo/env"
# Build tools (Linux; optional if already present)
sudo apt-get update -y && sudo apt-get install -y build-essential pkg-config || true
# Maturin for building Python/Rust extensions
pip install -U pip
pip install "maturin>=1.5,<2.0"
```

1) Create venv with uv and install GlossAPI (editable)

```bash
cd /path/to/glossAPI
uv venv .venv
. .venv/bin/activate

# Fast path: install project with dependencies (builds Rust extensions via maturin)
uv pip install -e .

# Alternative (stricter sync):
# uv pip sync requirements.txt
# uv pip install -e . --no-deps
```

2) Ensure ONNXRuntime GPU only and verify CUDA provider

```bash
pip uninstall -y onnxruntime || true
python - <<'PY'
import onnxruntime as ort
p = ort.get_available_providers()
print(p)
assert 'CUDAExecutionProvider' in p, f'CUDAExecutionProvider missing: {p}'
PY
```

3) Optional: install Torch CUDA (GPU layout + formula/code enrichment)

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1
 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

4) Install enrichment helpers (for JSON + math/code enrichment)

```bash
pip install pypdfium2 zstandard
```

5) Patch Docling keys mapping (once per venv)

```bash
bash repro_rapidocr_onnx/scripts/repatch_docling.sh
```

6) Quick smoke test (uses packaged ONNX models and keys)

```bash
python -m glossapi.docling_rapidocr_pipeline /path/to/pdfs /path/to/out --device cuda:0
# Outputs *.md, *.json, and metrics in the out folder
```

### AWS SageMaker / Amazon Linux (conda-based)

On SageMaker Notebook Instances or Amazon Linux environments where `conda` is
the primary environment manager, you can bootstrap GlossAPI with the helper
script:

```bash
cd /path/to/glossAPI
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh

# Later, when you need the environment:
conda activate glossapi
```

The script creates a Python 3.10 conda environment named `glossapi`, installs
Rust via `rustup`, ensures `maturin` is present, runs `pip install -e .`, and
builds the required Rust extensions (`glossapi_rs_cleaner` and
`glossapi_rs_noise`), then reapplies the Docling RapidOCR patch so the packaged
ONNX keys are wired correctly. Run the optional GPU OCR / Torch steps from above
inside the activated conda environment as usual.

Rust extensions (required for `Corpus.clean()` and noise metrics)
- Build them once per environment right after installing dependencies:
  - `python -m pip install "maturin>=1.5,<2.0"` (already done above)
  - `python -m maturin develop --release --manifest-path rust/glossapi_rs_cleaner/Cargo.toml`
  - `python -m maturin develop --release --manifest-path rust/glossapi_rs_noise/Cargo.toml`

> **Tip:** If you previously installed the Rust wheels globally (e.g. via
> `pip install --user glossapi_rs_cleaner`), uninstall them to avoid stale
> modules taking precedence on `sys.path`:
> `pip uninstall -y glossapi_rs_cleaner glossapi_rs_noise`

## Usage

The recommended way to use GlossAPI is through the `Corpus` class, which provides a complete pipeline for processing academic documents. You can use the same directory for both input and output. By default, extraction runs without OCR; you can optionally force GPU OCR only for files detected as badly extracted by the Rust cleaner.

```python
from glossapi import Corpus
import logging

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)

# Set the directory path (use the same for input and output)
folder = "/path/to/corpus"  # Use abstract path names

// Initialize Corpus with input and output directories
corpus = Corpus(
    input_dir=folder,
    output_dir=folder,
)

# The pipeline can start from any of these steps:

# Step 1: Download documents (if URLs are provided)
corpus.download(url_column='a_column_name')  # Specify column with URLs, default column name is 'url'

# Step 2: Extract documents (no OCR by default)
# Single‑GPU (default)
corpus.extract(input_format="pdf", use_gpus="single")

# Or Multi‑GPU: auto‑detect all visible GPUs and distribute work
# corpus.extract(input_format="pdf", use_gpus="multi")

# Optional: Clean and OCR only on badly extracted files
corpus.clean()      # computes quality metrics and produces cleaned markdown
corpus.ocr()        # re-extracts only the flagged bad files with GPU OCR

# Step 3: Extract sections from filtered documents
corpus.section()

# Step 4: Classify and annotate sections
corpus.annotate()  # or corpus.annotate(annotation_type="chapter") For texts without TOC or bibliography
```

### Controlling Phase‑1 extraction

GlossAPI offers two extraction profiles:

- **Safe (default)** — PyPDFium backend with size‑1 batching. Recommended when you prioritise stability.
- **Docling/native** — batches documents through `docling_parse` for maximum throughput. Use only when you are comfortable trading some stability for speed.

Pick one via environment variables (or programmatically through `GlossExtract.configure_batch_policy(...)`):

```bash
# Safe mode
export GLOSSAPI_BATCH_POLICY=safe
export GLOSSAPI_BATCH_MAX=1

# Higher throughput (use with caution)
export GLOSSAPI_BATCH_POLICY=docling
export GLOSSAPI_BATCH_MAX=5
```

Regardless of policy, the extractor now clamps OMP/OpenBLAS/MKL pools to a
single thread per worker so multi‑GPU runs stay well behaved.

## Documentation

- Getting started, quickstart, and API reference live under `docs/`.
  - Start here: docs/index.md
  - Quickstart: docs/quickstart.md
  - OCR + Math enrichment: docs/ocr_and_math_enhancement.md
  - Corpus API: docs/api/corpus.md

## Folder Structure

After running the pipeline, the following folder structure will be created:

```
corpus/  # Your specified folder
├── downloads/                     # Downloaded source files
├── download_results/              # Parquet(s) with URL + processing metadata
├── markdown/                      # Canonical Markdown outputs
│   └── <stem>.md                  # Enriched Markdown (overwrites plain MD if Phase‑2 ran)
├── json/                          # Docling + scheduling artifacts
│   ├── <stem>.docling.json(.zst)  # DoclingDocument payload for Phase‑2
│   ├── <stem>.formula_index.jsonl # FORMULA/CODE index for scheduling
│   ├── <stem>.latex_map.jsonl     # Phase‑2 LaTeX/code recognition results
│   └── metrics/                   # Timing & page metrics
│       ├── <stem>.metrics.json
│       └── <stem>.per_page.metrics.json
├── sections/
│   └── sections_for_annotation.parquet
├── classified_sections.parquet    # Intermediate processing form
└── fully_annotated_sections.parquet  # Final processing form with section predictions
```

- Markdown is the canonical output and, when Phase‑2 is used, the enriched MD replaces the plain MD.
- JSON artifacts (Docling payload, indices, and LaTeX maps) live under `json/`. Metrics are under `json/metrics/` to keep Markdown clean.

The `fully_annotated_sections.parquet` file contains the final processing form. The `predicted_sections` column shows the type of section: 'π' (table of contents), 'β' (bibliography), 'ε.σ.' (introductory note), 'κ' (main text), or 'a' (appendix). For files without table of contents or bibliography, the annotation will be "άλλο" (other).

## Note on Starting Points

**Option 1: Start with Document Download**
Create a corpus folder and add a parquet file with URLs for downloading:
```
corpus/
└── metadata.parquet (with a column containing document URLs)
```
Then use `corpus.download(url_column='column_name')` with the URL column name from your parquet file.

**Option 2: Start with Document Extraction**
Alternatively, place documents directly in the corpus folder and skip download:
```
corpus/
└── document1.pdf, document2.docx, etc.
```
GlossAPI will automatically create a metadata folder in downloads if starting from extract.

## License

This project is licensed under the [European Union Public Licence 1.2 (EUPL 1.2)](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12).

## GPU OCR with Docling + RapidOCR

The project includes a GPU-capable OCR pipeline using Docling for layout and RapidOCR (ONNXRuntime) for OCR. By default, Corpus.extract runs without OCR; use `Corpus.clean()` then `Corpus.ocr()` to re-extract only the pages/files that the Rust cleaner flags as badly extracted. These steps are portable across machines:

- Create a fresh venv and install packages
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -U pip`
  - `pip install docling==2.48.0 rapidocr onnxruntime-gpu==1.18.1`
  - Ensure only GPU ORT is present: `pip uninstall -y onnxruntime || true`
- Install Torch CUDA for GPU layout and enrichment (choose a build matching your driver):
    - `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1`
- Provide ONNX models and Greek keys
  - Package files under `glossapi/models/rapidocr/{onnx,keys}` or set `GLOSSAPI_RAPIDOCR_ONNX_DIR` to a directory containing:
    - `det/inference.onnx`, `rec/inference.onnx`, `cls/ch_ppocr_mobile_v2.0_cls_infer.onnx`, and `greek_ppocrv5_keys.txt`
  - Generate keys from Paddle `inference.yml` using `repro_rapidocr_onnx/scripts/extract_keys.py`
  - Patch Docling to pass the keys path to RapidOCR
    - File: `<venv>/lib/python3.10/site-packages/docling/models/rapid_ocr_model.py`
    - Replace `"Rec.keys_path"` with `"Rec.rec_keys_path"` (or run `repro_rapidocr_onnx/scripts/repatch_docling.sh`)

## Multi‑GPU Tips and Benchmarking

- Dynamic scheduling: `Corpus.extract(..., use_gpus="multi")` now uses a shared work queue so faster GPUs pull more files and the tail finishes sooner.
- Auto threads: pass `num_threads=None` to auto‑set threads based on CPU count and number of GPUs (roughly CPU/NGPUs). You can also set `OMP_NUM_THREADS`/`MKL_NUM_THREADS` externally.
- Benchmark mode: add `benchmark_mode=True` to `Corpus.extract(...)` to skip per‑document/per‑page metrics JSON and Docling timing collection, reducing I/O and profiling overhead during scaling tests. Defaults remain unchanged when omitted.
- Verify providers
  - `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` → should include `CUDAExecutionProvider`
- Quick system check
  - `python scripts/check_system.py` → writes `system_check_report.md`; confirm GPUs and CUDAExecutionProvider are OK.
- Run with Corpus API (few‑line usage)
  - Python
    - Single GPU: `Corpus.extract(input_format="pdf", use_gpus="single")`
    - Multi GPU: `Corpus.extract(input_format="pdf", use_gpus="multi")` (auto‑uses all visible GPUs)
  - CLI‑style example
    - `python -c "from glossapi import Corpus; c=Corpus('IN_DIR','OUT_DIR'); c.extract(input_format='pdf', use_gpus='multi')"`

### Defaults and tuning

- Accurate table parsing: kept enabled (TableFormer ACCURATE).
- Orientation classifier: disabled by default (`use_cls=False`) for digital PDFs; override with `use_cls=True` if needed.
- OCR thresholds: `text_score=0.45`; image scale hint `images_scale=1.25`.

### Environment variables (tuning & placement)

- `CUDA_VISIBLE_DEVICES`: restrict/assign visible GPUs, e.g. `export CUDA_VISIBLE_DEVICES=0,1,2,3`.
- `GLOSSAPI_DOCLING_DEVICE`: preferred device for Docling inside a worker/CLI, e.g. `export GLOSSAPI_DOCLING_DEVICE=cuda:0`.
- `GLOSSAPI_IMAGES_SCALE`: image scale hint for parsing/OCR (default ~`1.1`–`1.25`).
- `GLOSSAPI_FORMULA_BATCH`: batch size for CodeFormula math enrichment (default `16`).
- `GLOSSAPI_RAPIDOCR_ONNX_DIR`: override path containing RapidOCR ONNX models and Greek keys.
- Math early-stop (optional; default enabled when supported):
  - `GLOSSAPI_LATEX_EARLYSTOP`=`1|0` enable/disable early-stop wrapper.
  - `GLOSSAPI_LATEX_MAX_CHARS` (default `3000`) cap generated LaTeX length.
  - `GLOSSAPI_LATEX_MAX_REPEAT` (default `50`) stop on last-token repeat run.
  - `GLOSSAPI_LATEX_MAX_NEW_TOKENS` (default unset) cap new tokens at the decoder.
  - `GLOSSAPI_LATEX_LEN_STRIDE` (default `16`) decode stride for char-length checks.
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS`: cap CPU threads to avoid oversubscription on mixed CPU/GPU nodes.
- Caches: set `HF_HOME`, `XDG_CACHE_HOME`, `DOCLING_CACHE_DIR` to fast storage (e.g., NVMe) for best throughput.
- Multi‑GPU networking (if needed): `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` can quiet warnings on some hosts.

### CUDA/ORT compatibility

- Keep your ORT + CUDA stack compatible:
  - Check driver/toolkit: `nvidia-smi` (look for “CUDA Version”) and optionally `nvcc --version`.
  - Use `onnxruntime-gpu==1.18.1` for CUDA 12.x drivers (e.g., L4 instances). If you must run on CUDA 11.x drivers, select an ORT GPU build compatible with CUDA 11.x (e.g., 1.16.x).
  - Never have `onnxruntime` CPU installed alongside `onnxruntime-gpu` (uninstall CPU ORT).
  - Verify GPU providers: `python repro_rapidocr_onnx/scripts/check_ort.py` → must include `CUDAExecutionProvider`.
  - Keep `numpy<2` for best wheel compatibility.

### Downloader checkpoints and resume

- The downloader writes periodic checkpoints to `output_dir/download_results/download_results_<input>.partial.parquet` while running (default every ~1000 completions or 60s).
- If a run is interrupted, rerunning the same script resumes by loading the partial results and skipping already successful URLs.
- When a run finishes, `Corpus.download()` writes the final results parquet to `output_dir/download_results/download_results_<input>.parquet`.
- In per-domain mode, if all active domains are drained and only parked/down domains remain, the run terminates after `down_wait_max_seconds` (default 300s). Tune via `c.download(..., down_wait_max_seconds=120)`.

### Alternate: Minimal repro runner

If you prefer a self‑contained repro with explicit model paths, follow `repro_rapidocr_onnx/RUN.md` or:

```bash
python3 -m venv .venv_docling
source .venv_docling/bin/activate
pip install -U pip
pip install -r repro_rapidocr_onnx/requirements.txt
pip uninstall -y onnxruntime || true
bash repro_rapidocr_onnx/scripts/repatch_docling.sh
python repro_rapidocr_onnx/scripts/check_ort.py  # verify CUDAExecutionProvider
bash repro_rapidocr_onnx/scripts/run_onnx.sh --det DET.onnx --rec REC.onnx \
  --keys greek_keys.txt --in INPUT_PDFS --out OUTPUT_DIR --device cuda:0
```

Further docs
- Math enrichment runtime (early‑stop + post‑processing + targeted runs): see `docs/math_enrichment_runtime.md`.

Automating Torch selection
- Use `scripts/install_torch_auto.sh` to pick a suitable Torch build automatically:
  - `bash scripts/install_torch_auto.sh` (uses CUDA 12.1 if available, falls back to 11.8, else installs CPU)

Notes
- Avoid installing the CPU `onnxruntime` wheel. Some packages (e.g., `rapidocr_onnxruntime` or `docling[rapidocr]`) declare `onnxruntime` as a dependency and can auto‑pull the CPU wheel. Prefer `rapidocr` + `onnxruntime-gpu` instead, and use `pip install -e . --no-deps` for editable installs.
- OCR runs on ORT GPU when `onnxruntime-gpu` is installed; layout/enrichment use Torch CUDA.
- If you encounter NCCL warnings on multi-GPU systems, set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`.

## Math Enrichment & JSON Intermediates

JSON is emitted by default during extraction (written to `json/<stem>.docling.json(.zst)`). Keep it if you plan to enrich math/code; set `export_doc_json=False` if you truly want to skip it.

- Phase‑1 (layout, no OCR) — emit JSON only if enriching later:

```python
from glossapi import Corpus
c = Corpus('IN','OUT')
c.extract(
    input_format='pdf',
    use_gpus='multi',           # or 'single'
    emit_formula_index=True,    # request json/{stem}.formula_index.jsonl alongside the default JSON
)
```

- Triage math density and route candidates:

```python
c.triage_math()  # writes per-doc summary + recommendation into parquet
```

- Phase‑2 (GPU) — enrich from JSON and re‑export Markdown:

```python
# Requires pypdfium2 (rasterization) and zstandard (compressed JSON)
c.formula_enrich_from_json(device='cuda', batch_size=12)
```

Multi‑GPU: `extract(..., use_gpus='multi')` uses a shared queue across all visible GPUs. OCR and math enrichment run on GPU; OCR currently binds a single GPU per process. Roadmap: multi‑GPU OCR scheduling.
