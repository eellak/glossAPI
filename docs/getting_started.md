# Onboarding Guide

This guide gets a new GlossAPI contributor from clone → first extraction with minimal detours. Use it alongside the [Quickstart recipes](quickstart.md) once you're ready to explore specialised flows.

## Checklist

- Python 3.11–3.14 (`requires-python = ">=3.11,<3.15"`)
- Recent `pip` (or `uv`)
- A Rust toolchain when you build from a checkout: the root wheel compiles and bundles the cleaner, noise and PDF
  font-repair extensions (`rust-toolchain.toml` pins the qualified toolchain)
- Optional: an NVIDIA GPU with a CUDA driver, or Apple Silicon, for Docling layout and OCR

On fresh Linux hosts, make these assumptions explicit instead of relying on shell startup files:

- prefer a stable final CPython, not a prerelease distro build
- keep `~/.local/bin` on `PATH` if `uv` was installed with `pip install --user uv`
- keep `~/.cargo/bin` on `PATH` if Rust was installed with `rustup`

## Install GlossAPI

### Recommended setup

```bash
git clone https://github.com/eellak/glossAPI.git
cd glossAPI
./dependency_setup/setup_glossapi.sh                       # core only, in .venv_glossapi_core
./dependency_setup/setup_glossapi.sh layout --device cpu   # adds Docling layout (cpu, cuda126, cuda132 or mps)
```

Add `--venv PATH` to choose the environment, `--editable` for an editable install, `--run-tests` to install the test
tools and run the suite, and `--dry-run` to print every command without running it. Install the default Greek OCR
package with `python install_glossapi.py ocr --device cpu` (or `mps` / `cuda126` / `cuda132`); see
[OCR setup](ocr_plugin_migration.md). The full package list is in
[Dependency setup](https://github.com/eellak/glossAPI/blob/main/dependency_setup/dependency_notes.md).

### Manual editable install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

This builds the Rust extensions needed for `Corpus.clean()` and the noise metrics. Re-run `pip install -e .` (or
`python install_glossapi.py core --editable`) after pulling changes that touch `rust/`. `glossapi capabilities`
reports a cleaner or noise extension built from older source as missing, with the detail
`stale native module ...; rebuild core`, and `glossapi doctor` reports it as an error.

### Adding packages later

In an environment that already has glossAPI, `glossapi install --device cpu --package download` prints the exact
install commands for the named packages without running them; add `--yes` to run them. Each command's output is
shown on stderr while it runs.

## GPU prerequisites (optional)

`setup_glossapi.sh layout --device cuda126` (or `cuda132`, `mps`) installs the matching torch build before Docling.
The device name selects the wheel source only; it does not choose the device of a later run. To check that torch sees
the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## First run (lightweight corpus)

```bash
python - <<'PY'
from pathlib import Path
from glossapi import Corpus

input_dir = Path("samples/lightweight_pdf_corpus/pdfs")
output_dir = Path("artifacts/lightweight_pdf_run")
output_dir.mkdir(parents=True, exist_ok=True)

corpus = Corpus(input_dir, output_dir)
corpus.extract(input_format="pdf")
PY
```

- Inspect `artifacts/lightweight_pdf_run/markdown/` and compare with `samples/lightweight_pdf_corpus/expected_outputs.json`.
- Run `pytest tests/test_corpus_phases_lightweight.py` for a reproducible regression check on the same two PDFs
  (extract, clean and section; no GPU or Docling).

## Next steps

- Jump into [Quickstart recipes](quickstart.md) for GPU OCR, Docling, and enrichment commands.
- Explore [Pipeline overview](pipeline.md) to understand each processing stage and emitted artifact.
- Read [Resumability, recovery and retention](architecture/resumability_recovery_and_retention.md) before running
  large corpora with `glossapi run` or `Corpus.run()`.
