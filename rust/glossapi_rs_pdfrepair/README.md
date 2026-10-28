# glossapi_rs_pdfrepair

Conservative native port of `glossapi.pdf_repair.tounicode`. Font and PDF parsing,
CMap decisions, reports, and incremental serialization run in Rust. The optional
Python shim supplies PDFium's identical eight-page text-presence sample. Analysis
returns an error-bearing `PdfFontReport`; repair returns an error-bearing
`RepairResult`, matching the reference dataclasses. Neither function raises for
ordinary document/font failures.

## Build offline

From the repository root, using the supplied experiment environment:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
cargo test --offline --locked --manifest-path rust/glossapi_rs_pdfrepair/Cargo.toml
.venv-exp/bin/maturin build --release --offline --locked \
  --manifest-path rust/glossapi_rs_pdfrepair/Cargo.toml \
  --out rust/glossapi_rs_pdfrepair/target/wheels --interpreter .venv-exp/bin/python
cargo build --release --offline --locked \
  --manifest-path rust/glossapi_rs_pdfrepair/Cargo.toml \
  --no-default-features --features cli
```

Install the wheel in a user-selected environment, or run
`.venv-exp/bin/python rust/glossapi_rs_pdfrepair/fixtures/unpack_wheel.py`
and add `target/python` to `PYTHONPATH`. During development, replace
extracted files atomically (do not truncate an extension loaded by another process
on macOS). This task's validation used local extraction and did not install into
or change `.venv-exp`.

## API

```python
import glossapi_rs_pdfrepair
report = glossapi_rs_pdfrepair.analyze_pdf("source.pdf")
result = glossapi_rs_pdfrepair.repair_pdf("source.pdf", "/tmp/repaired.pdf")
```

Both return plain JSON-safe dictionaries. The shim additionally accepts `Path`
arguments and `protected_sources=`. The extension supports an optional third
`protected_sources` list. The existing Python backend is not automatically changed.

Rust callers use `analyze(path)` / `repair(src, dst)` or the
`analyze_with_sample` / `repair_with_sample` variants with their own renderer.
The convenience Rust API invokes the shim using `GLOSSAPI_PDFREPAIR_PYTHON`
(default: `python3`); the callback variants do not launch Python. The Python
extension calls the shim directly, without a subprocess.

## CLI

```sh
export GLOSSAPI_PDFREPAIR_PYTHON="$PWD/.venv-exp/bin/python"
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
PYTHONDONTWRITEBYTECODE=1 .venv-exp/bin/pytest -q \
  rust/glossapi_rs_pdfrepair/tests/test_backend.py -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv-exp/bin/python \
  docs/experiments/glossapi_improvement_20261006/equivalence_pdfrepair.py
```

The equivalence script compares every report field in order, repair results,
repaired text on the first 20 pages, after-repair reports, all original objects,
all new object IDs, original-byte prefixes and source SHA-256 hashes. It writes
artifacts under `/tmp` and exits nonzero on any mismatch. `--limit` and
`--skip-object-check` are diagnostic options, not full acceptance runs.

See `docs/experiments/glossapi_improvement_20261006/T1_1_NOTES.md` for actual
measurements and limitations. The generated AGL/CFF/post/Greek lookup tables are
standard oracle data, not precomputed sample decisions. Tiny synthetic font
fixtures have no dependency on copyrighted system-font binaries.
