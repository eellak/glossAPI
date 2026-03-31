# Download Stage

## Purpose

The download stage acquires source documents from parquet-based URL metadata and creates the initial local corpus artifact set.

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
