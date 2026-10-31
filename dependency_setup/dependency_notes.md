# Dependency setup

`./dependency_setup/setup_glossapi.sh` creates a **core-only** environment. The
root `requirements.txt` delegates to the same `pyproject.toml` runtime metadata;
there are no parallel vanilla/Docling requirements lists. Python 3.11–3.14 is
supported. A Rust toolchain is needed when building from source; the root wheel
bundles the cleaner, noise and PDF font-repair extensions.

Select named packages explicitly:

```sh
./dependency_setup/setup_glossapi.sh download --venv /path/to/download-env
./dependency_setup/setup_glossapi.sh layout --device cpu --venv /path/to/layout-env
./dependency_setup/setup_glossapi.sh download browser annotate
python install_glossapi.py layout --device cuda126 --dry-run
python install_glossapi.py layout --device cuda132
python install_glossapi.py layout --device mps
python install_glossapi.py --pypdfium2-version 4.30.0 --dry-run
```

The Python entry point installs in the active interpreter. Both entry points
accept `--dry-run` and print the exact commands without creating environments or
installing packages. The shell also accepts `--python`, `--venv`, `--editable`
and `--run-tests`; only `--run-tests` installs build/test tools in the target
runtime. The old `--mode docling` spelling selects `layout`; vanilla is removed.

Upgrading an environment containing standalone `glossapi_rs_cleaner`,
`glossapi_rs_noise` or `glossapi_rs_pdfrepair` distributions is refused before
installation. Follow the printed uninstall command in that interpreter, then
rerun the installer. After installation a fresh process verifies native module
origins and root-wheel RECORD hashes (editable installs verify source origins).

- `download`: aiohttp and YAML support (retries and file writes use the standard library).
- `browser`: download plus Playwright and Pillow. Browser engines are a separate
  explicit `python -m playwright install chromium` download.
- `layout`: Docling **2.81.0**, including math/code enrichment. The installer first
  installs **torch 2.14.1 / torchvision 0.29.1**, then installs GlossAPI/Docling
  from PyPI with constraints preserving that torch build. Linux devices are
  `cpu`, `cuda126` and `cuda132`, using their dedicated PyTorch indexes. Apple
  Silicon `cpu` and `mps` use normal PyPI wheels. These choices install packages;
  they do not select the device of a later extraction run. CUDA needs compatible
  NVIDIA hardware and a driver supporting the chosen runtime.
- `annotate`: no extra packages. The packaged section classifier is a neutral export
  (`glossapi/models/section_classifier/`: JSON manifest, vocabulary, float64 weights, hash-checked) scored by
  glossAPI's own numpy inference (`glossapi.section_model`); scikit-learn and joblib are not imported. The extra
  name is kept so existing `glossapi[annotate]` installs keep resolving.
- `annotate-legacy`: **scikit-learn 1.6.1** and joblib, needed only to load an explicitly supplied `*.joblib`
  model (`section_classifier_model_path=...joblib`) or to train (`GlossSectionClassifier.train_from_csv`).
  Without it those paths fail with an error naming `glossapi[annotate-legacy]`. The neutral model is produced
  once from the shipped pickle by `tools/export_section_classifier.py` in the environment of
  `tools/export_section_classifier.requirements.txt`.

Core installs contain no torch or Docling. PDFium remains `>=4.30,<5`; pass
`--pypdfium2-version 4.30.0` to reproduce the qualified PDFium baseline. Direct
`pip install '.[layout]'` cannot choose a torch index; use the installer for a
device-specific layout environment. External OCR engines remain separately
installed plugins; see [OCR plugin migration](../docs/ocr_plugin_migration.md).

## PDF libraries and retired compatibility paths

PDFium (`pypdfium2`) is the only PDF library on the runtime path: text, page
counts, page ranges and rendering. Font repair runs in the bundled Rust
extension (`glossapi_rs_pdfrepair`); the Python reference repair (`pypdf`,
`fontTools`) is used only when that extension cannot be imported, or when
requested explicitly, and needs the `font-repair` extra.

Retired, with no replacement fallback (accepted install-path behaviour change):

- **PyPDF2** is no longer used anywhere. The legacy extract page-count fallback
  (`pypdf`, then `PyPDF2`) and the benchmark scripts' PyPDF2/PyMuPDF/pypdf
  counters are gone. When PDFium cannot open a PDF (malformed, password-protected
  or zero-page), the extraction work estimator counts it as one page and the
  strict counters (checkpoint, Docling cost and batched benchmarks, OCR
  reassembly) raise PDFium's error.
- **PyMuPDF** (`fitz`) is no longer used at runtime (it went with the DeepSeek
  backend) or in the tests, whose fixture PDFs are now built with PDFium
  (`tests/pdf_fixtures.py`).

## Test environments

The `test` dependency group (and `dev`, which includes it) installs the
`reference` group: pinned oracles for the parity tests (tenacity 9.2.1,
aiofiles 25.1.0, dask 2026.8.0, pypdf 6.21.3, fontTools 4.68.0). With
`GLOSSAPI_REFERENCE_REQUIRED=1` (set by `--run-tests`) a missing or different
oracle fails its test instead of skipping it.
`requirements-docs.lock` locks the docs profile (mkdocs 1.6.1,
mkdocs-material 9.8.0) with hashes.

`requirements-tooling.txt` pins the surveyed build/test tools, including pip. Standalone Rust
builds use maturin >=1.15; root builds use setuptools >=84.2.1. The serialized
classifier does not support Python 3.14; the declared interpreter range describes
core. The original qualification exercised core on Python 3.11; Python 3.13
had not been run in that qualification. See the cycle-1 dead-weight reply for
the subsequent Python 3.13 run and its exact test scope.
