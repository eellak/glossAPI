# OpenArchives OCR Clean Refresh

## Contract

This refresh targets the canonical raw dataset:

- `glossAPI/openarchives.gr`

The update set is:

- every row where `pipeline_metadata.ocr_success == true`

The join key is:

- OpenArchives `doc_id`
- OCR manifest `source_doc_id`

The refresh operation is:

1. extract the exact OCR-success population from OpenArchives
2. match every target row to a premerged OCR markdown document
3. clean those texts with the shared OCR clean/debug renderer
4. recompute deterministic clean-owned metadata
5. patch the OpenArchives rows in place by `doc_id`
6. validate exact updated-row coverage and text hashes
7. upload the refreshed dataset

This is intentionally not a fuzzy or heuristic merge.

## Source Of Truth

There is no remaining source-of-truth ambiguity for the text payload:

- the target dataset is `glossAPI/openarchives.gr`
- the OCR manifest is `/home/foivos/data/glossapi_work/analysis/openarchives_ocr_completion/release_refresh_20260403T180204Z/ocr_manifest.parquet`
- the premerged OCR markdown corpus is `/home/foivos/data/glossapi_work/analysis/openarchives_ocr_completion/release_refresh_20260403T180204Z/merged_markdown`

The premerged OCR corpus already resolves split documents. The split merge rule is the exact one previously used for the OpenArchives OCR refresh:

- split parts are ordered by page range
- contiguity is required
- neighboring parts are joined with:
  - `\n<--- Page Split --->\n`

So the refresh driver should clean the premerged corpus, not rebuild the split merge from lane shards again.

## Update Rules

For each matched `doc_id`:

- replace top-level `text` with the cleaned text
- patch `pipeline_metadata` fields owned by OCR cleaning and publish-side reevaluation

Fields updated in `pipeline_metadata`:

- `filter`
- `needs_ocr`
- `ocr_success`
- `percentage_greek`
- `latin_percentage`
- `polytonic_ratio`
- `greek_badness_score`
- `char_count_no_comments`
- `is_empty`
- `page_count`
- `pages_total`
- `quality_method`
- `reevaluated_at`
- `ocr_noise_suspect`
- `ocr_noise_flags`
- `ocr_repeat_phrase_run_max`
- `ocr_repeat_line_run_max`
- `ocr_repeat_suspicious_line_count`
- `ocr_repeat_suspicious_line_ratio`

Fields preserved because the fast refresh path does not currently expose an exact scorer for them:

- `mojibake_badness_score`
- `mojibake_latin_percentage`
- download-status fields
- duplicate-routing fields
- formula and math-enrichment counters
- processing-stage history
- source URLs and provenance

This preservation is intentional. The refresh must not invent values for fields that are not recomputed by the current execution path.

## Failure Policy

This run should treat the OpenArchives OCR-success set as authoritative and complete.

That means the following are hard failures, not silent drops:

- an OpenArchives `ocr_success == true` row without a matching OCR manifest row
- a matched manifest row without a premerged OCR markdown file
- a matched target row without a cleaned output file
- duplicate `doc_id` rows in any stage
- row-count drift after patching
- text hash mismatch between cleaned outputs and patched dataset rows

## Runtime Shape

Do not use the full `Corpus.clean_ocr()` parquet-update path for the remote refresh.

Reason:

- it bundles a slow second-pass directory rescoring path
- it is much slower than the shared clean/debug renderer
- it is not the right primitive for a large remote dataset refresh

Use instead:

1. shared fast OCR render loop for clean + debug
2. publish-side reevaluation over the cleaned outputs
3. patch raw OpenArchives JSONL shards in place

## Remote Execution Shape

Recommended remote machine:

- CPU-heavy Compute Engine VM
- tested working shape: `n2-standard-32` with `500GB` boot disk
- Ubuntu LTS
- enough SSD for:
  - OpenArchives snapshot
  - 41,675 merged OCR docs
  - clean outputs
  - debug outputs
  - staged upload tree

Recommended bootstrap requirements:

- `git`
- `rsync`
- `zstd`
- `build-essential`
- `pkg-config`
- `libssl-dev`
- `python3.11`
- `python3.11-venv`
- `python3-dev`
- `curl`
- `patchelf`
- Rust via `rustup`
- `maturin`

Rust initialization requirement:

- prebuild `glossapi_rs_noise` explicitly before the large run

This avoids first-use build surprises inside the long-running clean job.

## Artifacts

Each refresh run should produce:

- target manifest
- clean outputs
- debug outputs
- `match_index.jsonl`
- page-metrics summary
- reevaluated metadata parquet/jsonl
- patched staged dataset tree
- validation summary

The debug tree should be copied back to `home` after the remote run completes successfully.

## Helper Scripts

The repo now includes executable helpers for the remote path:

- `scripts/create_openarchives_ocr_refresh_gce.sh`
  - create the GCE worker VM
- `scripts/stage_openarchives_ocr_refresh_to_gce.sh`
  - copy the repo, OCR manifest, and premerged OCR markdown corpus to the VM
- `scripts/bootstrap_openarchives_ocr_refresh_vm.sh`
  - install system packages, Rust, maturin, and the project venv on the VM
- `scripts/run_openarchives_ocr_refresh_remote.sh`
  - execute the end-to-end refresh driver on the VM

The refresh driver also accepts an explicit relocated merged corpus path:

- `--merged-markdown-root /path/to/merged_markdown`

This is required for remote execution because the historical OCR manifest stores absolute `merged_path` values from `home`.

## Local Mirror Warning

The current local raw mirror at:

- `/home/foivos/data/glossapi_raw/hf/openarchives.gr`

is stale relative to the previously refreshed OpenArchives OCR snapshot. At least some rows that are present in the 41,675-doc OCR manifest still appear there as:

- `pipeline_metadata.ocr_success = false`
- `pipeline_metadata.needs_ocr = true`

So local smoke tests should use the already refreshed staged snapshot under:

- `/home/foivos/data/glossapi_work/analysis/openarchives_ocr_completion/release_refresh_20260403T180204Z/staged_hf_repo`

For the real refresh run, the VM should download a fresh snapshot from Hugging Face instead of assuming the local raw mirror is current.
