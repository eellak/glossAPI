# Changelog

## 0.2.0 — 2026-11-09

### What changed

- The owner's [Greek OCR toolkit](https://github.com/fffoivos/greek-ocr-toolkit) is the only supported OCR engine. DeepSeek and RapidOCR engines, install modes and bundled RapidOCR models have been removed. The `ocr` extra pins the public toolkit source and inference dependencies; runtime admission verifies that source.
- Core extraction now produces layout-aware Markdown (`layout_v2`) from PDFium text order and Rust layout labels, without Docling. Headings, lists, captions, footnotes and page furniture have hash-bound page/strip sidecars. Set `GLOSSAPI_MARKDOWN_LAYOUT=plain` before planning/running to retain plain page text. Docling layout, JSON and math/code enrichment remain optional through `layout` (alias `docling`) and explicit `phase1_backend='docling'`.
- Per-page OCR routing defaults to v4, with an offline Greek Hunspell dictionary for image-dominant pages, measured vocabulary/junk checks, contamination checks and the short-caption scan gate. Validity is a routing signal, not recognition accuracy. Experimental English v5 and multilingual Latin v6 validity remain opt-in; v4 is unchanged for ordinary non-image pages.
- PDF font repair runs in the bundled Rust extension before extraction. Glyph-only layers route to OCR. Extraction and page records distinguish decoding failure, no text layer, blank/whitespace text and supported text; undecodable selected pages fail explicitly.
- Cleaner metrics operate on markup-free text using verified strip records. Partial OCR replaces selected physical pages, preserves retained native text and blank-page numbering, and records replacement provenance. Missing or stale page offsets use a reported supplement fallback.
- `glossapi plan`, `run`, `status`, `report`, `capabilities`, `doctor`, `install` and `guide` expose a versioned JSON envelope. The corpus ledger records frozen settings, attempts, input/output hashes, page routing, retained/released artifacts and publication membership; interrupted runs can resume with `--resume RUN_ID`.
- Core installs bundle the cleaner, noise and PDF-repair Rust extensions and support extraction, cleaning, sectioning, annotation and export without torch or Docling. Default annotation uses the bundled neutral model with NumPy; legacy joblib models/training require `annotate-legacy`.
- Downloads are **public-only**, including browser recovery. Credential URLs, signed/token query fields, cookies and credential settings are refused before planning, recording or requesting them.
- CUDA device hints use contiguous logical ordinals, including containers that expose a nonzero host device node; explicit visibility masks and disabled GPUs are respected.
- Source distributions now include the Rust reader self-test and fixture assets needed to rebuild wheels. macOS wheel tags match the native architecture, and native binaries use portable compiler paths.
- Upstream documentation and contributing guidance are retained. Missing pandas state cells (`pd.NA`, NaN, NaT, None) are handled without truth-value errors; new state rows retain null unknown URLs.

### Install from source

Use Python 3.11–3.14, a Rust toolchain and a C linker (Xcode Command Line Tools on macOS; `build-essential` on Debian/Ubuntu). With rustup, `rust-toolchain.toml` selects the qualified toolchain; the crates' minimum Rust version is 1.85.

```sh
git clone https://github.com/eellak/glossAPI.git
cd glossAPI
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .                 # core; builds all three Rust extensions
python -m pip install '.[ocr]'          # optional Greek OCR toolkit and inference stack
python -m pip install '.[layout]'       # optional Docling layout/math stack
glossapi --help
glossapi capabilities --json
glossapi doctor
```

To choose a specific torch device build, use `python install_glossapi.py ocr --device cpu` (or `mps`, `cuda126`, `cuda132`); `./dependency_setup/setup_glossapi.sh layout --device cpu` installs the optional layout stack. Device names choose wheel sources; actual inference devices are selected in the run. Model assets download on first use; prewarm both toolkit and Docling caches for offline OCR. A source/wheel distribution can be built with `python -m pip install build` followed by `python -m build`.

```sh
glossapi run --corpus output --files book.pdf \
  --stages extract,clean,sections,annotate,export --json
# OCR is an explicit additional stage:
glossapi run --corpus scanned-output --files scan.pdf \
  --stages extract,clean,ocr,sections,annotate,export --device ocr=cpu --json
```

### API migration

The `Corpus` phase methods remain available; consult [their signatures](docs/api/corpus.md) and the [ledger/resume contract](docs/architecture/resumability_recovery_and_retention.md).

- Removed DeepSeek/RapidOCR options and setup modes must be replaced with the Greek toolkit `ocr` package. `force_ocr` in extraction is deprecated and ignored; call `Corpus.ocr(mode='ocr_bad')` or select the run's `ocr` stage.
- `Corpus.clean(threshold=...)` is deprecated and has no effect; cleaning uses the documented fixed quality gates. `page_routing` defaults to `all` for OCR modes (page-quality selection across documents); choose `flagged` explicitly for cleaner-flagged documents only.
- The layout-aware Markdown default changes serialized text and sidecars. The plain escape hatch is part of execution identity; choose it before starting a new run rather than changing a saved run.
- Metadata has one null rule: unknown values remain null, including newly discovered URLs; callers must not assume empty strings or truth-test pandas missing values. Cleaning measures original text without generated markup.
- A `Corpus.run()` script must use `if __name__ == '__main__':` because stage workers use multiprocessing `spawn`. Resume restores frozen settings and refuses conflicting planning arguments. Explicit `--retry-failed` retries exhausted/non-transient failures.
- `GLOSSAPI_STAGE_PROFILE=legacy` is a comparison control. The default `cutover` profile uses the ledger for run/resume/status/retention. `sequential-adapters` is an earlier qualification profile.

### Known limitations

- Core layout labels preserve PDFium reading order; they do not reconstruct complex tables, reorder columns or provide Docling JSON. Use optional Docling layout when those outputs are needed. `math` is not executable in the streaming run yet; use the documented phase API with Docling JSON.
- Modern Greek dictionary entries are not affix-expanded and under-cover polytonic, historical and specialist text. Good text may be routed to OCR; wrong text composed of valid words may pass. v5/v6 Latin dictionaries have incomplete/coarse surface coverage and are experimental, explicitly opt-in. No general recognition-accuracy claim follows from routing or execution parity.
- OCR recognition is sequential per page; the detector uses CPU (CoreML optionally on macOS), not a CUDA detector. Accelerator/device combinations must be qualified on the target machine. CPU OCR is suitable for small checks, not a throughput promise. Toolkit model and layout assets require cache space and initial network access.
- Annotation currently materializes whole-corpus frames, so memory grows with section text (about 983 MB peak for the documented 53 MB sections parquet cohort). Scale accordingly; see the recovery/retention guide.
- `memory_saving=extreme` requires Linux with two independently size-limited mount-point filesystems and explicit disk budgets. Ordinary shared volumes and macOS are refused. Supplied originals are never deleted by that policy.
- After a killed debug-off run, resume/close it before writing to the same corpus with the legacy profile. macOS cannot fence worker descendants that detach into a different process group. WAL commits protect against process death; the documented fsync boundaries and residual same-clock-tick edit ambiguity apply to power loss.
- The native layout labeller classifies characters with Python 3.11 / Unicode 14.0 rules. On Python 3.12+ (Unicode 15.0) about 4,400 code points added in Unicode 15 (rare scripts; none Greek or Latin) count as letters to the interpreter but not to the labeller, so the pure-Python fallback can label such text differently.
- GPU OCR with default batch settings used about 12 GiB of GPU memory per recognizer worker on an RTX 5070 Ti (glossAPI X2 test documents). Plan GPU sharing accordingly.
- Authenticated downloads are unsupported. Existing user edits are protected by explicit post-close integrity refusals; a saved run's source/environment hashes must still match on resume.
