# Extract Stage

## Purpose

The extraction stage converts source files into Markdown and optional structured byproducts such as Docling JSON.

## Main responsibilities

- locate source files for processing
- choose extraction backend behavior
- produce Markdown
- optionally emit structured JSON artifacts
- balance throughput against stability

## Main inputs

- files in `downloads/` or explicitly provided file paths
- extraction configuration
- backend selection and accelerator settings

## Main outputs

- repaired PDF sidecars in `repaired_pdfs/<stem>.pdf` when eligible
- font-repair metadata in the canonical download-results parquet
- Markdown in `markdown/`
- optional JSON artifacts in `json/`
- stage logs and timeout/problematic-file traces

## Key design point

This stage is where the Docling throughput strategy meets the pipeline's stability constraints.

It should be understood as both:

- a text extraction stage
- an operational risk management stage

## Backend semantics

The extraction logic distinguishes between safer extraction behavior and Docling-focused behavior.

This matters because Docling is powerful, but not equally stable under all batching and corpus conditions.

## Layout-aware Markdown (layout v2, core install)

The safe backend (the default `phase1_backend='auto'`/`'safe'`) writes layout-aware Markdown without Docling: the
PDFium plain text in plain order, cut into blocks and labelled by layout v2 (`glossapi.markdown_layout`, Rust in
`glossapi_rs_pdfrepair`). Labels never change the text and nothing is re-ordered. It needs only the core install
(pypdfium2 and the `glossapi_rs_pdfrepair` extension at native API level 2); the `layout` extra and its `docling`
alias are not involved.

`GLOSSAPI_MARKDOWN_LAYOUT` selects the safe backend's Markdown: `layout_v2` (default, extract stage contract 2) or
`plain` (the plain pages, byte for byte as before; contract 1). The value is part of the extract stage identity.

Rendering rules (`glossapi-layout-md/1`, `markdown_layout/render.py`):

| block | Markdown |
| --- | --- |
| page separator | one blank line between non-blank pages, as before (page offsets are in the metadata) |
| between blocks of a page | one blank line (replaces the block's original whitespace separator) |
| `title` | `# text` |
| `section_header` level n | `#` x min(6, n + 1), a space, the text (newlines inside the heading become spaces) |
| `list_item` | `- text`, unless the text already starts with `-`, `*`, `+`, `1.` or `1)` and a space |
| `caption`, `footnote`, `table`, `picture`, `formula` | a `<!-- caption -->` (etc.) line, then the text |
| `page_header`, `page_footer` | a `<!-- page-header -->` / `<!-- page-footer -->` line, then the text (kept) |
| `paragraph`, unlabelled page | the text |

Text is never escaped. Deleting the added markup and restoring each block's original separator gives the page text
byte for byte. A page is emitted unlabelled (its text unchanged) when it has no char table, no text layer, a page box
or font size outside the input contract (`layout_input_rejected`) or the labeller fails on it.

The extract sidecar (`sidecars/extract/<stem>.json`) carries `markdown_layout`: schema, rules, reader (`native` or
`python`), status (`labelled`, `partial`, `unlabelled`, `unavailable`), per-page and per-block code-point offsets into
the Markdown with labels and levels, label counts, labeller warnings, rejected pages and the document context.

Reading: each page is loaded once. `layout_v2_read_document` opens the document with Rust-owned PDFium handles and
returns each page's text (exactly `get_text_range()`) and char table from the same text page. If the native reader is
unavailable (`LayoutV2ReaderUnavailable`) the text comes from pypdfium2 as before and the char tables from the Python
reference reader (same bytes, slower); a document or page the native reader cannot open is read the pypdfium2 way,
with exactly pypdfium2's exceptions. A missing or stale extension gives the plain Markdown.

Reader threat model (K5 policy P7): Documents and public argument values are untrusted; the Python runtime, installed
dependencies and executable caller code are trusted. The wrapper validates arguments and preserves native handle
ownership and lifetime. Dependency replacement, forged ctypes state and uncoordinated library teardown are
unsupported. All PDFium users, initialization and destruction require process-wide coordination; the reader's mutex
and GIL do not serialize external ctypes calls. Identity checks and self-tests detect accidental incompatibility, not
malicious substitution. This boundary does not sandbox PDFium or guarantee unlimited resource availability.

Coordination: every glossAPI module that opens a PDF with pypdfium2 does so under `glossapi.pdfium_guard` (one
re-entrant lock per process, reset in a forked child), and every native read takes the same lock, so no two PDFium
users overlap in one process. glossAPI never initializes or destroys PDFium: pypdfium2 does at import and at exit.

Docling stays available through `phase1_backend='docling'` and the `layout` extra (alias `docling`) for what layout
v2 does not replace:

| capability | core (layout v2) | `layout` extra (Docling 2.81) |
| --- | --- | --- |
| headings, lists, captions, footnotes, furniture labels | yes (labels on the plain order) | yes (Docling layout model) |
| text source | PDFium plain text | docling-parse text (transliterates TeX-encoded Greek) |
| table structure (rows, cells, Markdown tables) | no (table text marked `<!-- table -->`) | yes |
| math/code enrichment | no | yes (`formula_enrichment`, `code_enrichment`) |
| Docling JSON (`json/<stem>.docling.json`) | no | yes (`export_doc_json`) |
| non-PDF conversion | no | yes |
| GPU | not needed | optional |

## Font repair pre-pass

`Corpus.extract(..., repair_fonts=True)` enables CPU ToUnicode repair by default
for both the safe and Docling backends. Set `repair_fonts=False` to extract the
original text layer. Each file's extraction worker analyzes its PDF immediately
before conversion; a `repairable` or `mixed` document with at least one eligible
font receives a repaired copy in `<output_dir>/repaired_pdfs/<stem>.pdf`. The
extractor opens that copy. Original PDFs in `downloads/` stay unchanged, and
Markdown names, resume keys and metadata retain the original filename/stem.
Non-PDF inputs bypass repair.

An existing `repaired_pdfs/<stem>.pdf` is never reused: a newer modification
time does not prove that the copy was made from the current source bytes, so each
extraction analyses the source and repairs it again (the repair is deterministic,
so the same source gives the same bytes). The ledger paths (`Corpus.run()`,
`glossapi run`) reuse work by the source hashes they record instead. Ordinary
`skip_existing=True` resume still skips completed Markdown entirely;
`skip_existing=False` re-extracts and repairs again.

The pre-pass uses at most half of the existing per-file extraction timeout,
reserving the rest for extraction. Its elapsed time is deducted from the shared
budget, including chunked conversion and retries. Analysis/repair failures and
pre-pass timeouts fall back to the original PDF and record `error` plus the
message in the extract sidecar and parquet's existing `extract_error` column.
The stage continues. Docling retains its existing native timeout behavior;
converters without native timeout support cannot enforce the extraction portion
of the deadline. POSIX main-thread signal timers bound the pre-pass without an
extra process pool. Unsupported timer contexts fall back with `error`; native C
calls may delay signal delivery until they return to Python.

Workers write `sidecars/extract/<stem>.json`; the controller merges these four
columns into the canonical metadata parquet after extraction:

| Column | Type / default | Meaning |
| --- | --- | --- |
| `font_repair_status` | string / `skipped` | `repaired`, `not_needed` (consistent), `unsupported`, `no_text_layer`, `error`, or `skipped` (disabled/unprocessed). |
| `font_repair_fonts` | int / `0` | Number of fonts repaired. |
| `font_repair_valid_rate_before` | float / null | Dictionary validity over the original PDF's first 20 pages. |
| `font_repair_valid_rate_after` | float / null | Dictionary validity over the repaired PDF's first 20 pages, or the unchanged original. |

Configure `GLOSSAPI_GREEK_WORDLIST` with a local UTF-8 word list to compute rates
with `extract_text_stats`. Missing/unreadable dictionaries or metric failures
leave rates null. Each repaired PDF gets one INFO line with its stem, font count
and available before→after rates. The final summary counts statuses for the stage's selected PDFs, including
previously completed files on resume.

## Failure concerns

Typical issues include:

- Docling crashes
- hangs or timeouts
- worker poisoning after a bad file
- partial progress during corpus-scale processing

## Fatal skiplist

The fatal skiplist holds files a worker failed on; a later direct
`Corpus.extract` call skips them (a `Corpus.run()` decides from the ledger and
does not read the skiplist). When every multi-GPU worker has exited, the files still queued were
never attempted: they are recorded as `workers_exhausted_unattempted` and are
not added to the skiplist, so the next `Corpus.extract` processes them.

## Recovery concerns

Recovery should preserve:

- successful Markdown already produced
- visibility into failed documents
- enough information to avoid repeating the same crash pattern blindly

## Contributor note

Any change to batch policy, backend selection, or output naming should be treated as an architectural change, not just a local refactor.
