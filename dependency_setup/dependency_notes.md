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

- `download`: aiohttp, aiofiles, Tenacity and YAML support.
- `browser`: download plus Playwright and Pillow. Browser engines are a separate
  explicit `python -m playwright install chromium` download.
- `layout`: Docling **2.81.0**, including math/code enrichment. The installer first
  installs **torch 2.14.1 / torchvision 0.29.1**, then installs GlossAPI/Docling
  from PyPI with constraints preserving that torch build. Linux devices are
  `cpu`, `cuda126` and `cuda132`, using their dedicated PyTorch indexes. Apple
  Silicon `cpu` and `mps` use normal PyPI wheels. These choices install packages;
  they do not select the device of a later extraction run. CUDA needs compatible
  NVIDIA hardware and a driver supporting the chosen runtime.
- `annotate`: the existing classifier, retaining **scikit-learn 1.6.1**.

Core installs contain no torch or Docling. PDFium remains `>=4.30,<5`; pass
`--pypdfium2-version 4.30.0` to reproduce the qualified PDFium baseline. Direct
`pip install '.[layout]'` cannot choose a torch index; use the installer for a
device-specific layout environment. External OCR engines remain separately
installed plugins; see [OCR plugin migration](../docs/ocr_plugin_migration.md).

`requirements-tooling.txt` pins the surveyed build/test tools. Standalone Rust
builds use maturin >=1.15; root builds use setuptools >=84.2.1. The serialized
classifier does not support Python 3.14; the declared interpreter range describes
core. The original qualification exercised core on Python 3.11; Python 3.13
had not been run in that qualification. See the cycle-1 dead-weight reply for
the subsequent Python 3.13 run and its exact test scope.
