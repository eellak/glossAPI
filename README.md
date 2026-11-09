# GlossAPI

GlossAPI is GFOSS's document-processing pipeline for turning academic PDFs into structured Markdown, cleaned text, sections and metadata.

## Greek OCR: code, models and data

**GlossAPI 0.2.0 uses the Greek OCR toolkit as its only supported OCR engine.**

- **[OCR toolkit on GitHub](https://github.com/fffoivos/greek-ocr-toolkit)** — the implementation integrated with GlossAPI.
- **[Greek OCR models on Hugging Face](https://huggingface.co/fffoivos/greek-ocr)** — the model repository used by the default OCR configuration.
- **[Greek OCR Lines dataset on Hugging Face](https://huggingface.co/datasets/fffoivos/greek-ocr-lines)** — the toolkit's upstream line-data project.

The default OCR configuration uses the toolkit's R8 segmenter and x1 recognizer, with source and model revisions pinned by GlossAPI. The dataset link provides upstream data context; inference uses model assets. See [OCR installation, pinned revisions and supported inputs](docs/ocr_plugin_migration.md).

Core PDF extraction uses PDFium and bundled Rust extensions to produce `layout_v2` Markdown without Docling or a GPU. Set `GLOSSAPI_MARKDOWN_LAYOUT=plain` before a run for plain page text. OCR is an optional install and explicit processing step; Docling layout and math/code enrichment are optional capabilities.

## Why GlossAPI

- Download public documents, extract text, clean it, split sections, annotate and export through the `Corpus` API or CLI.
- Inspect per-page OCR routing and provenance through the corpus ledger.
- Resume streaming runs with frozen settings and recorded artifact identities.
- Use the bundled neutral section classifier without scikit-learn; install legacy model support only when needed.

See [0.2.0 changes and limitations](CHANGELOG.md). The installation commands below build from a source checkout.

## Quickstart (local repo)

```bash
git clone https://github.com/eellak/glossAPI.git
cd glossAPI
python -m venv .venv && source .venv/bin/activate   # Python 3.11–3.14
pip install -e .                                    # builds the bundled Rust extensions (needs a Rust toolchain)

# Run the lightweight PDF corpus (no GPU/Docling required)
python - <<'PY'
from pathlib import Path
from glossapi import Corpus

input_dir = Path("samples/lightweight_pdf_corpus/pdfs")
output_dir = Path("artifacts/lightweight_pdf_run")
output_dir.mkdir(parents=True, exist_ok=True)

corpus = Corpus(input_dir, output_dir)
corpus.extract(input_format="pdf")  # Safe PyPDFium backend by default
PY
```

- Compare the generated Markdown in `artifacts/lightweight_pdf_run/markdown/`
  with `samples/lightweight_pdf_corpus/expected_outputs.json` for a fast smoke check.
- `pytest tests/test_corpus_phases_lightweight.py` runs the same two PDFs through extract, clean and section without
  GPU or Docling.

### Corpus usage contract
`Corpus` is the organizing surface: keep contributions wired through `Corpus.run()` and the phase methods (`download()`, `extract()`, `clean()`, `ocr()`, `section()`, `annotate()`, `jsonl()`, `jsonl_sharded()`). The intended use is a short script chaining those calls, or the `glossapi plan`/`glossapi run` commands; avoid bespoke monkeypatches or side channels so resumability and artifact layout stay consistent.

`Corpus.run()` and `glossapi run` record every stage in the corpus ledger (`corpus.sqlite3`) and can resume an interrupted run (`glossapi run --resume RUN_ID`). Their stage workers are separate processes started with `spawn`, which import your script again; phase methods such as `extract()` also run their per-document work in `spawn` workers. A script file that calls `Corpus.run()` or a phase method must therefore guard those calls with `if __name__ == "__main__":` (the `python - <<'PY'` examples below run from standard input and need no guard). See [Resumability, recovery and retention](docs/architecture/resumability_recovery_and_retention.md) and the [CLI JSON schema](docs/cli_json_schema.md).

`Corpus.extract(batched=None)` defaults to layout batch 8 with 8 rasteriser workers for single-device, layout-only Docling on CUDA; safe PyPDFium and multi-GPU runs stay sequential by default.

## Optional packages

For 0.2.0 changes, API migration and limitations, see [CHANGELOG.md](CHANGELOG.md).
The default Markdown mode is `layout_v2`; set `GLOSSAPI_MARKDOWN_LAYOUT=plain` before a run for plain page text.
Neither core mode requires Docling. From this checkout, `pip install .`, `pip install '.[ocr]'` and
`pip install '.[layout]'` install core, Greek OCR and optional Docling layout respectively. Source installs need
Rust and a C toolchain; see [installation](docs/getting_started.md).

The default install is **core**: extraction with PDFium, cleaning, sectioning, annotation and export. Add named
packages when you need them:

Run these in the environment that has glossAPI (activate it first):

- `python install_glossapi.py layout --device cpu` (or `cuda126`, `cuda132`, `mps`): Docling layout
  extraction and math/code enrichment.
- `python install_glossapi.py ocr --device cpu` (or `mps`, `cuda126`, `cuda132`): the pinned external Greek OCR
  plugin. Then call `corpus.ocr(mode='ocr_bad')`, or `glossapi run --stages ocr,export`, which use that plugin by
  default.
- `python install_glossapi.py download` (HTTP downloads) or `python install_glossapi.py browser` (browser-assisted
  downloads; then run `python -m playwright install chromium` once, which glossAPI never runs for you).
- In an environment that already has glossAPI, `glossapi install --device cpu --package download` prints the exact
  commands for the named packages; add `--yes` to run them.

See [Dependency setup](dependency_setup/dependency_notes.md) and
[OCR setup, development override and qualification](docs/ocr_plugin_migration.md).

## Browser-Gated Download Mode

Downloads need the `download` package; the browser routes need the `browser` package plus the Chromium download
described above.

All download routes support public targets only. URLs containing credentials, signed/token query fields, cookies
or credential settings are refused before any request. Browser recovery handles public interstitials, not login.

`Corpus.download(...)` now supports three high-level routes for file acquisition:

- `download_mode="standard"`: direct HTTP downloader only
- `download_mode="auto"`: direct HTTP first, then browser-assisted recovery when the response is a recoverable browser-gated interstitial
- `download_mode="browser"`: go straight to browser-assisted acquisition for known browser-gated file endpoints

Use `browser_mode=True` as a legacy alias for `download_mode="browser"`.

### Policy-driven routing

If you know which domains require browser bootstrap, route them with a policy file instead of probing every URL:

```yaml
default:
  downloader: standard

rules:
  - match:
      domains: [eur-lex.europa.eu]
    downloader: browser

  - match:
      url_regex: "https://example.org/protected/.*"
    downloader: auto
```

```python
from glossapi import Corpus

corpus = Corpus(input_dir="out", output_dir="out")
corpus.download(
    input_parquet="input_urls.parquet",
    download_policy_file="download_policy.yml",
)
```

### Operational notes

- Browser mode is for browser-gated file endpoints, not viewer-only sources.
- Browser sessions are cached per domain so a successful bootstrap can be reused across multiple files.
- Successful downloads still land in `downloads/`; extraction continues to consume only real files from that directory.
- Viewer-style sources still fail cleanly in `download_results/*.parquet` and do not create fake files.

### Regression strategy

The checked-in browser download tests use mocked browser/session flows and fake PDF bytes rather than hard-coded live URLs.

For manual smoke checks against live browser-gated sources, build an ad hoc parquet locally and run it outside the committed test suite.

## Choose Your Install Path

| Scenario | Commands | Notes |
| --- | --- | --- |
| Core from a checkout | `./dependency_setup/setup_glossapi.sh`, then `source .venv_glossapi_core/bin/activate` | Creates a core-only virtual environment (Python 3.11–3.14; `--venv PATH` chooses another, then activate `PATH/bin/activate`). The root wheel bundles the cleaner, noise and PDF font-repair Rust extensions. |
| Named optional packages | `./dependency_setup/setup_glossapi.sh layout --device cpu`, or in the activated environment `python install_glossapi.py ocr --device cuda126` | `download`, `browser`, `layout`, `ocr`, `annotate`, `annotate-legacy`. `layout` and `ocr` install a device-specific torch build first. Add `--dry-run` to print the commands only. |
| Manual editable install | `pip install -e .` after cloning | Core only. Needs a Rust toolchain to build the extensions; rerun it after pulling changes to `rust/`. |
| Add packages to an existing environment | `glossapi install --device DEV --package NAME` | A dry run by default; `--yes` runs the printed commands. |

See [Dependency setup](dependency_setup/dependency_notes.md) for the package contents, the supported devices and the
native-extension checks, and the docs index (`docs/index.md`) for troubleshooting tips.

## Repo Landmarks
- `docs/code_map.md`: fast map from pipeline ideas to implementing classes and files.
- `docs/pipeline.md`: stage contracts, key parameters, and artifact outputs.
- `docs/cli_json_schema.md`: the `glossapi` commands, their JSON output, exit codes and typed errors.
- `samples/lightweight_pdf_corpus/`: two one-page PDFs (`pdfs/alpha.pdf`, `pdfs/beta.pdf`) with their expected
  Markdown lines in `expected_outputs.json`.
- `src/glossapi/`: Corpus pipeline, ledger, stages, cleaners, and orchestration logic.
- `tests/test_corpus_phases_lightweight.py`: a fast regression check on the lightweight corpus (no GPU or Docling).
- `docs/`: MkDocs site with onboarding, pipeline recipes, and configuration guides.

## Pipeline map

Use this as the shortest path from a documentation concept to the public call that implements it.

| Stage | Main call | Important parameters | Writes |
| --- | --- | --- | --- |
| Download | `Corpus.download(...)` | `input_parquet`, `links_column`, `parallelize_by`, `download_mode="standard"|"auto"|"browser"`, `download_policy_file`, downloader kwargs | `downloads/`, `download_results/*.parquet` |
| Extract (Phase-1) | `Corpus.extract(...)` | `input_format`, `phase1_backend`, `use_gpus`, `workers_per_device`, `export_doc_json`, `emit_formula_index` | `markdown/<stem>.md`, `json/metrics/*.json`; `json/<stem>.docling.json(.zst)` with `phase1_backend='docling'` only |
| Clean | `Corpus.clean(...)` | `drop_bad`, `empty_char_threshold`, `empty_min_pages` (`threshold` is deprecated and has no effect) | `clean_markdown/<stem>.md`, updated parquet metrics/flags |
| OCR / math follow-up | `Corpus.ocr(...)` | `mode`, `fix_bad`, `page_routing` (default `"all"`), `math_enhance`, `use_gpus`, `devices` | refreshed `markdown/<stem>.md`, optional `json/<stem>.latex_map.jsonl` |
| Section | `Corpus.section()` | uses cleaner/parquet outputs to choose inputs | `sections/sections_for_annotation.parquet` |
| Annotate | `Corpus.annotate(...)` | `annotation_type`, `fully_annotate` | `classified_sections.parquet`, `fully_annotated_sections.parquet` |
| Triage math density | `Corpus.triage_math()` | no required args | updated `download_results/*.parquet` routing columns |
| JSONL export | `Corpus.jsonl(...)`, `Corpus.jsonl_sharded(...)` | `output_path` / output directory | merged training/export JSONL |
| Whole run with a ledger | `Corpus.run(...)`, `glossapi run` | `stages`, `settings` (`debug`, `memory_saving`), `resume`, `retry_failed` | the stage products above plus `corpus.sqlite3` and `export.jsonl` |

## Contributing
- Run `pytest tests/test_corpus_phases_lightweight.py` for a fast end-to-end check on CPU.
- `tests/test_pipeline_smoke.py` needs torch, Docling and a CUDA GPU; on other machines its tests skip
  (`GLOSSAPI_REQUIRE_GPU=1` makes a missing GPU fail instead).
- Prefer `uv` or `pip` editable installs so Rust extensions rebuild locally. After a change to `rust/`, rebuild with
  `python install_glossapi.py core --editable` (or `pip install -e .`): `glossapi capabilities` reports a cleaner or
  noise build that does not carry this checkout's native API level as missing (`stale native module ...; rebuild core`).

Open an issue or PR if you spot drift between expected outputs and the pipeline.

## License

This project is licensed under the [EUPL 1.2](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12).

Bundled dictionary and fixture terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
