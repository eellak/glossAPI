# Resumability, Recovery, and Retention

GlossAPI is designed to process corpora that are too large and too messy to assume a perfect one-pass run. This page
describes how a run resumes, how it recovers from failures, and what it keeps on disk.

There are two ways to run the pipeline:

- **A streaming run**: `glossapi run` or `Corpus.run()`. Every stage attempt is recorded in the corpus ledger
  `corpus.sqlite3`, and resume, retry, retention, `glossapi status` and `glossapi report` decide from that ledger.
- **The phase methods** called one by one (`Corpus.extract()`, `Corpus.clean()`, ...). They resume from the metadata
  parquet and the files on disk, as before the ledger existed (see "Phase methods" below).

## Stage profiles

The default stage profile (`cutover`) writes the provenance ledger (`corpus.sqlite3`) and decides from it: plan, run,
resume, retention and status/report. `GLOSSAPI_STAGE_PROFILE=legacy` runs the pre-ledger stage code as a comparison
control; its wrappers only record observations. `sequential-adapters` is the earlier qualification profile (CPU, one
device, no batching, no OCR plugin).

## Resuming a run

- `glossapi run --corpus CORPUS --resume RUN_ID` (or `Corpus.run(resume=RUN_ID)`) restores the run's frozen plan and
  settings. Any other planning argument is refused (`invalid_request`).
- A completed run is not executed again: resume checks the frozen execution environment and the retained recovery
  files by size and SHA-256, and returns the saved result.
- Interrupted or lost attempts (`attempt_interrupted`, `worker_lost`, `worker_process_crashed`, `extract_timeout`)
  are retried automatically, within the run and on resume, at most **3 attempts per task**. Every interrupted attempt
  counts, so a task whose worker is killed three times stays failed.
- Other failures, and tasks past that budget, stay failed. `glossapi run --resume RUN_ID --retry-failed` is the
  explicit, operator-authorised retry of the failed tasks of that run.
- A stage worker that misses its deadline is stopped together with its descendants before its replacement starts. On
  Linux this includes descendants that moved to another process group (found through `/proc`). On macOS only the
  worker's own process group is signalled, so a descendant that detached into its own group (for example a browser)
  is not fenced there.

## Recovery

One failed document does not stop the others. `glossapi status` lists each task's state and exits 1 when a listed task
failed, with one typed error per failed task. `glossapi report` is the census (every requested source exactly once:
produced, excluded, unavailable, failed or pending), and `glossapi report --why DOCUMENT` explains one document from the
ledger alone, also after its intermediates were released.

## Retention: what a run keeps

Two settings, fixed when the run starts:

| setting | contract |
| --- | --- |
| `debug=off` (default) | An intermediate is released only after every planned reader has succeeded on its exact hash. Supplied files, selected final products, evidence and the ledger are kept. |
| `debug=on` | Keep every stage's intermediates and attempts, including bounded inner HTTP request bodies. |
| `memory_saving=normal` (default) | Never delete a supplied original or a downloaded file. |
| `memory_saving=extreme` | Delete only downloads this run owns, after the last planned reader succeeds and outputs are durable. Never a supplied original, another run's file, or a path reached through a link. Requires a disk budget. |

- Private scratch (under `.glossapi/artifacts/`) is released in groups: one summary per stage records the released
  files, their hashes and the planned readers. With `debug=off` the per-file release steps of private scratch are not
  kept in the ledger; producer hashes, every stage outcome, every deletion of a user-visible or source file and the run
  summary are.
- The cleaned texts (`clean_markdown/`) and the per-document extraction sidecars (`sidecars/extract/`) are kept in
  every setting.

## Durability

- The ledger runs in WAL mode with `synchronous=NORMAL`: an ordinary commit survives the death of the process
  (SIGKILL), not a power loss.
- Before every unlink that a user or a later stage could need, and at run close, the run makes exactly the needed bytes
  durable: it fsyncs the affected files, then their parent directories inside the corpus, then commits the ledger with
  `synchronous=FULL`. It never calls `syncfs`.
- Policy deletions run in batches of up to 128 files. Each batch records its intents, unlinks the files, fsyncs every
  affected parent directory once, and only then commits the deletion records. A crash in between leaves recoverable
  intents; the next start finishes or re-checks them. Private scratch released inside a run is not made durable at
  all: after a crash its release is recovered and the stage runs again from kept inputs.
- Corpus copies the run close leaves unflushed (for example `export.jsonl` and the metadata parquets, whose managed
  source is durable) are verified at the next start. A copy whose bytes no longer match is restored from its managed
  source only when its bytes are what a power loss can leave: a torn image of the managed bytes (a prefix, or blocks
  that read as zeros) or the copy this ledger registered there before. Timestamps never prove a copy unchanged. Any
  other content is treated as your edit: it is kept, and every start refuses (`PostCloseEditError`, message
  `post_close_edit: ...`; the CLI reports it with the generic code `unsupported_schema`, exit 2) until you move or
  delete it, after which it is republished from the managed source. Between a run close and the next start, a reader of such a copy can see a torn file after a power loss.

Declared residuals: an edit that only truncates a copy to a prefix of the managed bytes, within the same clock tick as
the run close, is indistinguishable from a torn copy and is restored. Two process caches compare file identity (device,
inode, size, mtime, ctime) and can miss a same-size rewrite that keeps mtime and ctime within one clock tick: the
ledger's integrity trust and the registry of glossAPI's own file hashes. Destructive steps re-read the bytes they rely
on. The original-hash cache keeps an original only when its mtime and ctime were already more than one second old when
it was read, so a later write always invalidates it.

## memory_saving=extreme setup

`memory_saving=extreme` needs positive `disk_budget_bytes`, `max_source_bytes` and `max_attempt_bytes`
(`disk_budget_required` otherwise). The guard reads the filesystems and never mounts anything. It requires:

- Linux;
- the corpus folder and `CORPUS/.glossapi/staging` each to be a mount point of its own size-limited filesystem, on
  different devices;
- the two capacities together to be at most `disk_budget_bytes`, and the staging capacity at most `max_attempt_bytes`;
- the corpus capacity to hold one attempt reservation (2 × `max_attempt_bytes`) plus the emergency reserve
  (max(1 GiB, 5% of the budget) + 64 MiB).

Anything else, including an ordinary shared volume, is refused with `disk_guard_unavailable` (exit 3). Admission then
runs one attempt at a time.

Before the first publication, and again before each publication, the run checks that the corpus filesystem has room for
what the remaining publications add: their input snapshots, products, the export's generation and published copy,
re-published metadata tables and parquet footprints, minus the private scratch the run releases between publications
(only files fully released, after their last reader). After the clean publication only the documents selected for
sections are charged to sections and annotation. Measured on a 242-PDF cohort, the charge was 1.01–1.12× the real need
at every check made after the selection is known and 1.04–1.08× for document-export-only runs; before the selection
exists, the sections and annotation charge counts every document (7.2× there). A run that does not fit is refused with
`disk_budget_exceeded`. If a publication still fills the disk, the run stops with a typed `disk_full` error, an
interruption record, and can be resumed.

## Known limitations

- **Linked products after a killed debug-off run.** If a `debug=off` run is killed before its private release, the
  corpus products may still be hard links to the run's private copies until the next run closes. Running a
  legacy-profile writer (`GLOSSAPI_STAGE_PROFILE=legacy`) on such a corpus in that window can write through a link into
  the private copy. The default profile is not affected: it publishes by copying and replaces a linked destination
  first. Resume or close the killed run before using the legacy profile on the same corpus.
- **Annotation memory.** Annotation (`Corpus.annotate`, and the annotate publication of `glossapi run`) reads the whole
  classified output back into memory twice: the classifier's prediction parts are concatenated into one frame for the
  index/threshold post-processing, and the full-annotation step reads `classified_sections.parquet` whole. Memory
  therefore grows with the total section text of the corpus. Measured on a 242-PDF cohort (409 sections, a 53.2 MB
  `sections_for_annotation.parquet`): peak RSS 983 MB (125 MB when the call started, +859 MB, 5.4 s); each whole read
  turns about 52.6 MB of parquet into a 122.7 MB frame. The peak is about 16 times the size of the sections parquet;
  assuming linear growth (not measured beyond that cohort), a corpus with a 1 GB sections parquet needs about 16 GB.

## Phase methods

Called one by one, the phase methods resume from the metadata parquet (`download_results/download_results.parquet`)
and the files on disk: extraction skips files the parquet records as processed (`skip_existing=True`), and the
download skips URLs that already succeeded. No pickle state file is written or read. They keep everything they write:

- source files, intermediate Markdown, cleaned Markdown, Docling JSON and metrics JSON;
- section and annotation parquets;
- `problematic_files/` and `timeout_files/` from extraction, and logs.

Retention deletes nothing outside a streaming run.
