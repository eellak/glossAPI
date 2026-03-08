# Download Stage

## Purpose

The download stage acquires source documents from parquet-based URL metadata and creates the initial local corpus artifact set.

## Main responsibilities

- read URL-bearing parquet input
- download files concurrently
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
- duplicate URLs
- filename collisions
- partially completed corpus fetches

## Contributor note

Any change to filename assignment or result parquet structure can have downstream impact on:

- extraction discovery
- metadata joins
- rerun behavior
