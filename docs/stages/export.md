# Export Stage

## Purpose

The export stage reshapes processed corpus data into downstream-friendly formats such as JSONL and aggregated metadata views.

## Main responsibilities

- read processed metadata and stage outputs
- aggregate chunked and unchunked document information
- emit export-oriented records for downstream consumers

## Main inputs

- metadata parquet
- section or text outputs depending on export mode
- chunk-aware naming and aggregation logic

## Main outputs

- exported JSONL or related downstream formats

In a streaming run (`Corpus.run()`, `glossapi run`) the export stage writes `export.jsonl` in the corpus folder and
records, for every selected document version, whether it was included or omitted and why. `glossapi report` (the
census) and `glossapi report --why` read those decisions, and they count a version as included only when the export
generation wrote its record. After OCR this still holds: the cleaned text produced after OCR is bound to the planned
document version, so the census, `--why` and `export.jsonl` agree.

## Page, formula and code counters

The exported `page_count`, `formula_total` and `code_total` carry no measurement method. They come from the first
readable metrics file of the document (`json/metrics/<stem>.metrics.json`, else
`json/metrics/<stem>.per_page.metrics.json`); `formula_total` and `code_total` add up the per-page counts in that
file, and a page without a count adds 0. Without a metrics file the metadata row's existing value is kept, and
`formula_total`/`code_total` become 0 when that is missing too. So 0 can mean "not measured". The ledger records which
file was used and labels the method `unknown_if_not_reported_by_metrics`.

## Why this stage matters

Export is where many earlier assumptions become visible:

- whether document identity is stable
- whether chunked outputs can be reconstructed correctly
- whether metadata remains coherent after reruns and recovery

## Contributor note

Changes to export logic often reveal hidden assumptions in the rest of the pipeline. Document those assumptions explicitly rather than leaving them embedded in naming or aggregation code.
