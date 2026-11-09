# Download Stage

## Purpose

The download stage acquires source documents from parquet-based URL metadata and creates the initial local corpus artifact set.

Prerequisites: the core install does not include the downloader. Install the `download` package
(`python install_glossapi.py download`) for HTTP downloads. The browser routes (`download_mode="auto"` escalation and
`download_mode="browser"`) need the `browser` package (`python install_glossapi.py browser`) and a one-time
`python -m playwright install chromium`, which glossAPI never runs for you.

## Main responsibilities

- read URL-bearing parquet input
- download files concurrently
- route known browser-gated sources through browser-assisted acquisition when configured
- retain source metadata context
- avoid refetching previously successful downloads
- assign stable-enough local filenames for downstream processing

## Main inputs

- input parquet containing URLs
- downloader configuration
- optional prior download result parquet files

## Main outputs

- downloaded files under `downloads/`
- parquet outputs under `download_results/`
- log files under `logs/`

## Important metadata role

This stage seeds the metadata that later stages build on top of.

That is why original scraping metadata should be preserved rather than discarded after download.

## Resumability behavior

The downloader supports resume-like behavior by consulting existing result parquet files and skipping already successful URLs.

A rerun of `Corpus.download` over the same input skips the URLs that already succeeded. The URLs still pending go to
the downloader with every column of the input (for example `title`, or a preassigned `filename_base`) and the
provenance columns (`source_row`, `url_index`, `collection_slug`, `source_id`, `resource_identity`), exactly as on
the first run. A row resumed after a failure therefore keeps its source fields and its preassigned file name.

A skipped duplicate URL and a failed download have an empty `filename`. They are kept as separate rows, each with its
own URL, error and source fields (see
[Metadata, artifacts and run diagnostics](../architecture/metadata_artifacts_and_run_diagnostics.md)).

`glossapi run` downloads through the same downloader but decides what to fetch from the ledger, not from the result
parquet files.

This is a major operational feature for large corpora.

## Failure concerns

Typical issues include:

- transient network failures
- rate limiting
- browser-gated file endpoints that return HTML challenge/interstitial pages
- viewer-only sources that should fail cleanly instead of being recorded as successful downloads
- duplicate URLs
- filename collisions
- partially completed corpus fetches

## Public targets only

A URL with userinfo (`user:pass@`) or a credential-named query or fragment field, and any credential setting or
header, is refused with `credentials_unsupported` before a request is built. Hand over protected files as local
paths instead.

## Browser-gated sources

The downloader now distinguishes between:

- direct file endpoints
- browser-gated file endpoints
- viewer-only/document-reader sources

For browser-gated file endpoints:

- `download_mode="auto"` probes with direct HTTP and escalates to a browser session when it detects a recoverable interstitial
- `download_mode="browser"` goes directly to the browser-assisted path
- `download_policy_file=...` can route known domains or URL patterns to the correct path without probing every file

Browser-assisted mode is designed for retrievable file endpoints, not for sources that only expose page images, tiles, HTML/SVG re-rendering, or DRM-wrapped readers.

## Session reuse

Browser-assisted mode reuses cached browser session state per domain so multiple files from the same protected source do not need a fresh browser bootstrap every time.

This keeps the browser as a session-bootstrap resource rather than the main downloader.

## Contributor note

Any change to filename assignment or result parquet structure can have downstream impact on:

- extraction discovery
- metadata joins
- rerun behavior
