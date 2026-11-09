# glossapi_rs_pdfrepair

Conservative native port of `glossapi.pdf_repair.tounicode`. Font and PDF parsing,
CMap decisions, reports, incremental serialization and the eight-page
text-presence sample run in Rust (K5a: the sample is decided from content-stream
text operators; the former Rust→Python→PDFium callback is gone). Analysis
returns an error-bearing `PdfFontReport`; repair returns an error-bearing
`RepairResult`, matching the reference dataclasses. Neither function raises for
ordinary document/font failures. An OS-level I/O failure (an `io::Error` with an
errno, such as ENOSPC on a full disk, EDQUOT, EIO or ENOENT) raises `OSError`
with that errno instead (Python picks the subclass, e.g. `FileNotFoundError`), as
the reference does, so a full disk is never read as a broken PDF. The module
exports `__glossapi_native_api__` (`GLOSSAPI_NATIVE_API:glossapi_rs_pdfrepair:1;`,
level 1 = this contract); glossAPI's repair selector does not use a build
without it.

## Build

The root package bundles this extension: `pip install .` (or `pip install -e .`,
or `python install_glossapi.py core`) from the repository root builds it with
setuptools-rust together with the cleaner and noise extensions. That is the
supported install. The glossAPI installer refuses an environment that also holds
a standalone `glossapi_rs_pdfrepair` distribution.

For development of this crate alone, from the repository root, in any virtual
environment with maturin >=1.15 (`$VENV` below):

```sh
export PYTHONDONTWRITEBYTECODE=1
cargo test --offline --locked --manifest-path rust/glossapi_rs_pdfrepair/Cargo.toml
"$VENV/bin/maturin" build --release --offline --locked \
  --manifest-path rust/glossapi_rs_pdfrepair/Cargo.toml \
  --out rust/glossapi_rs_pdfrepair/target/wheels --interpreter "$VENV/bin/python"
cargo build --release --offline --locked \
  --manifest-path rust/glossapi_rs_pdfrepair/Cargo.toml \
  --no-default-features --features cli
```

`"$VENV/bin/python" rust/glossapi_rs_pdfrepair/fixtures/unpack_wheel.py` unpacks
the newest wheel from `target/wheels` into `target/python`; the tests put that
directory first on `sys.path`. Unpacking replaces each file atomically (do not
truncate an extension loaded by another process on macOS). Do not install the
standalone wheel into a glossAPI environment.

`cargo clippy --all-targets -- -D warnings` passes for this crate, and for the
cleaner, noise and common crates.

## API

```python
import glossapi_rs_pdfrepair
report = glossapi_rs_pdfrepair.analyze_pdf("source.pdf")
result = glossapi_rs_pdfrepair.repair_pdf("source.pdf", "/tmp/repaired.pdf")
```

Both return plain JSON-safe dictionaries. The shim additionally accepts `Path`
arguments and `protected_sources=`. The extension supports an optional third
`protected_sources` list. The existing Python backend is not automatically changed.

Analyse once, repair from the analysis (the extraction pre-pass does this):

```python
report, analysis = glossapi_rs_pdfrepair.analyze_for_repair("source.pdf")
if analysis is not None:  # None exactly when report["error"] is set
    result = glossapi_rs_pdfrepair.repair_pdf("source.pdf", "/tmp/repaired.pdf", [], analysis)
```

The repair reuses the analysis's parsed fonts and text sample, so it neither
re-inspects fonts nor re-samples; it refuses (as a result error) another source
path or a source whose bytes changed since the analysis. The output equals the
standalone `repair_pdf` byte for byte.

Text presence: up to eight evenly spaced pages (PDFium page order and count), a
page has text when a text-showing operator (`Tj`, `TJ`, `'`, `"`), directly or in
a drawn Form XObject, shows a character that is not whitespace after ToUnicode
(else the code itself, as PDFium falls back). Invisible text counts, inline images
are skipped, a damaged FlateDecode content stream keeps its inflated prefix.
Declared difference from the PDFium sample: text drawn wholly outside the page
box counts here, while PDFium's bounded text excludes it. `_native_sample_json`
and `_native_page_flags_json` are diagnostics for the equality gates;
`glossapi.pdf_repair.rust_backend._sample_json` stays as the PDFium reference.

Rust callers use `analyze(path)` / `repair(src, dst)` (native sample), the
analyse-once pair `analyze_for_repair` / `repair_with_analysis`, or the
`analyze_with_sample` / `repair_with_sample` variants with their own renderer.
No Rust API launches Python.

## CLI

```sh
export GLOSSAPI_PDFREPAIR_PYTHON="$VENV/bin/python"   # default: python3
export PYTHONPATH="$PWD/rust/glossapi_rs_pdfrepair/target/python:$PWD/src"
rust/glossapi_rs_pdfrepair/target/release/pdfrepair classify \
  --pdfs /path/to/pdfs --out /tmp/classification.jsonl
rust/glossapi_rs_pdfrepair/target/release/pdfrepair repair \
  --pdfs /path/to/pdfs --out-dir /tmp/repaired --report /tmp/repair.jsonl
```

The binary is a Rust launcher for the shim's native-backend worker. It reuses the
reference CLI's bounded multiprocessing scheduler, JSONL schema, warning capture,
first-20-page text metrics, artifact protection and atomic publication. Its font
analysis and PDF repair functions are replaced with this extension. Python and
PDFium remain runtime requirements for the CLI. `--workers`, `--timeout` and
`--wordlist` therefore retain the audited reference semantics. Warning counts
reflect actual libraries used; Rust parsing does not issue fontTools warnings.

## Decisions and preservation

Type0: encoding → CID → GID → embedded Unicode. Simple TrueType: symbol cmap or
explicit Differences → GID → embedded Unicode. Invert all Unicode cmap subtables
by GID, retaining many-to-one mappings. Single candidates win; otherwise real
post/CFF names decoded through AGL win when compatible, then Greek-block letters
and the lowest code point. Synthesized format-3 names never become a Unicode
oracle. Preserve original ToUnicode entries lacking an embedded candidate.

Reports retain all reference fields, including reason/error/object ID. Document
categories first prioritize the absence of text on the up-to-eight evenly spaced
sample pages, then Greek repairability. Remaining homogeneous font categories are
`consistent`, `not_embedded`, or `unsupported`; different categories give `mixed`.
At document level `no_tounicode`, non-Greek `repairable` and `unsupported_direct`
count as unsupported. `no_text_layer` remains a sampled verdict.

Repairs retain every original byte as a prefix. Append only new ToUnicode streams
and updated indirect font dictionaries, followed by a classic xref/trailer (no
extra xref object). Unknown mappings, font programs, content streams, resources
and all other original objects remain intact. No-op outputs are byte-identical
copies. Reject encrypted inputs and source pathname/symlink/hardlink aliases.
Publish using a reserved temporary file and atomic rename; recheck aliases before
publication. Type3, Type1, direct dictionaries requiring changes, inherited CMaps
and unsupported encodings are refused as in the reference.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  rust/glossapi_rs_pdfrepair/tests/test_backend.py
```

This is the supported entry point. It runs the reference's generated PDF
fixtures (`tests/test_pdf_font_repair.py`) through the native analysis and
repair, and checks extracted text on the first 20 pages, the original objects and
the new object IDs with the helpers in `tests/pdf_invariants.py`. It needs the
extension (the root install or `target/python`) and `pypdf` (the `font-repair`
extra or the `test` dependency group). The generated AGL/CFF/post/Greek lookup
tables are standard oracle data, not precomputed sample decisions. Tiny synthetic
font fixtures have no dependency on copyrighted system-font binaries.

The full equivalence driver and the measurement notes of the original port
(`docs/experiments/glossapi_improvement_20261006/` on the
`codex/compute-reduction-20261006` branch) are historical and are not part of
this repository.
