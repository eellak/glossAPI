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
- **Flexible Pipeline**: Start the processing from any stage in the pipeline

## Installation

```bash
pip install glossapi
```

### Install via uv (recommended)

uv is a fast, reliable Python package manager. It works with normal `requirements.txt` files and speeds up installs.

- Create a venv and install dependencies from this repo:
  - `uv venv /mnt/data/venv`
  - `uv pip install -r requirements.txt`
- Ensure GPU ORT only (avoid CPU ORT collisions):
  - `uv pip uninstall -y onnxruntime`
  - `uv pip install --no-cache --force-reinstall onnxruntime-gpu==1.18.1`
- Optional: install Torch CUDA for layout/formula/code enrichment:
  - `uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1`
- Patch Docling so RapidOCR receives the Greek keys (one-time per venv):
  - `uv run --python /mnt/data/venv/bin/python bash repro_rapidocr_onnx/scripts/repatch_docling.sh`
- Verify providers include CUDA:
  - `uv run --python /mnt/data/venv/bin/python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY`

Notes
- No changes to `requirements.txt` are needed for uv; the file works unchanged. uv accepts the same syntax and constraints, and you may also use `uv pip sync requirements.txt` if you want a strictly synced environment.

## Usage

The recommended way to use GlossAPI is through the `Corpus` class, which provides a complete pipeline for processing academic documents. You can use the same directory for both input and output:

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

# Step 2: Extract documents (GPU OCR by default)
# Single‑GPU (default)
corpus.extract(input_format="pdf", use_gpus="single")

# Or Multi‑GPU: auto‑detect all visible GPUs and distribute work
# corpus.extract(input_format="pdf", use_gpus="multi")

# Step 3: Extract sections from filtered documents
corpus.section()

# Step 4: Classify and annotate sections
corpus.annotate()  # or corpus.annotate(annotation_type="chapter") For texts without TOC or bibliography
```

## Folder Structure

After running the pipeline, the following folder structure will be created:

```
corpus/  # Your specified folder
├── download_results # stores metadata file with annotation from previous processing steps
├── downloads/  # Downloaded documents (if download() is used)
├── markdown/    # Extracted text files in markdown format 
├── sections/    # Contains the processed sections in parquet format
│   ├── sections_for_annotation.parquet
├── classified_sections.parquet    # Intermediate processing form
├── fully_annotated_sections.parquet  # Final processing form with section predictions
```

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

The project includes a GPU-first OCR pipeline using Docling for layout and RapidOCR (ONNXRuntime) for OCR. These steps are portable across machines:

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

### CUDA/ORT compatibility

- Keep your ORT + CUDA stack compatible:
  - Check driver/toolkit: `nvidia-smi` (look for “CUDA Version”) and optionally `nvcc --version`.
  - Use `onnxruntime-gpu==1.18.1` for CUDA 12.x drivers (e.g., L4 instances). If you must run on CUDA 11.x drivers, select an ORT GPU build compatible with CUDA 11.x (e.g., 1.16.x).
  - Never have `onnxruntime` CPU installed alongside `onnxruntime-gpu` (uninstall CPU ORT).
  - Verify GPU providers: `python repro_rapidocr_onnx/scripts/check_ort.py` → must include `CUDAExecutionProvider`.
  - Keep `numpy<2` for best wheel compatibility.

Automating Torch selection
- Use `scripts/install_torch_auto.sh` to pick a suitable Torch build automatically:
  - `bash scripts/install_torch_auto.sh` (uses CUDA 12.1 if available, falls back to 11.8, else installs CPU)

Notes
- Avoid installing the CPU `onnxruntime` wheel. Some packages (e.g., `rapidocr_onnxruntime` or `docling[rapidocr]`) declare `onnxruntime` as a dependency and can auto‑pull the CPU wheel. Prefer `rapidocr` + `onnxruntime-gpu` instead, and use `pip install -e . --no-deps` for editable installs.
- OCR runs on ORT GPU when `onnxruntime-gpu` is installed; layout/enrichment use Torch CUDA.
- If you encounter NCCL warnings on multi-GPU systems, set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`.
