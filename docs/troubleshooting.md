# Troubleshooting

## OCR runs on CPU

- Verify Torch CUDA: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`.

## Torch doesn’t see the GPU

- Check `nvidia-smi` and driver installation.
- Match Torch CUDA build to your driver; see getting_started.md for the recommended wheel.

## Out of memory

- Lower Phase‑2 `batch_size` (e.g., 8) and reduce inline `GLOSSAPI_FORMULA_BATCH`.
- Reduce `GLOSSAPI_IMAGES_SCALE` (e.g., 1.1–1.2).
- Split large batches or files.

## Worker respawn limit reached

- When a GPU crashes repeatedly, the controller stops respawning it after `GLOSSAPI_MATH_RESPAWN_CAP` attempts. Any pending stems are added to the skip‑list and their inputs are copied to `downloads/problematic_math/` (PDFs) and `json/problematic_math/` (Docling artifacts); inspect those folders, address the issue, then rerun `Corpus.ocr(..., reprocess_completed=True)` or move the quarantined files back into `downloads/`.
- Check the corresponding worker log under `logs/math_workers/` (or the directory set via `GLOSSAPI_WORKER_LOG_DIR`) for stack traces and the active stem list stored in `gpu<N>.current`.

## A script that calls `Corpus.run()` fails at start-up

The stage workers of a streaming run are started with `spawn`, so each one imports your script again. Put the call
under `if __name__ == "__main__":`.

## `clean` is unavailable after pulling new code

`glossapi capabilities` shows the cleaner or noise module as `missing` with `stale native module ...; rebuild core`:
the installed extension was built from older Rust source. Rebuild with `python install_glossapi.py core` (add
`--editable` for a checkout) or `pip install -e .`.

## A run refuses to start with `post_close_edit`

A corpus copy that the previous run left unflushed (for example `export.jsonl`) was changed after that run closed.
glossAPI keeps your edit and refuses every start until you move or delete the file; it is then republished from the
managed source. The CLI reports the typed code `post_close_edit` (exit 3) with the edited paths in `details.paths`. See [Resumability, recovery and retention](architecture/resumability_recovery_and_retention.md).

## A run refuses with `unresolved_recovery`

Every product glossAPI manages has one expected state, decided from the ledger alone: the bytes of its newest committed
generation, nothing after a committed deletion of that generation itself, or a refusal when no committed generation can
be named. A registration is a committed generation only when its producing attempt ended `ok` and a positive record
binds it to that attempt: a publication record of this path and these bytes, or the attempt's own record of producing
these bytes (its output manifest). A deletion of any other registration of the path (a step's temporary file at the
same name, another run's) never removes a product's obligation. A file the legacy compatibility steps only observed is
not a managed product only for the allowlisted diagnostic roles (logs, the temporary cleaning report); any other file
only a compatibility step registered is refused (`unlisted_observation_role`). A rollback that undoes a failed attempt
records a permanent invalidation of that attempt's registrations; if it cannot restore a product (its only source was
released), it records a pending republication instead, which only a newer committed generation of that product
discharges.

The refusal comes from one of three places, each with the typed code `unresolved_recovery` (exit 3) and its
`details.boundary`:

- **a run start** (`start`) or **a repeated publication** (`publication`) found a corpus file it must repair (a copy the
  previous run left unflushed, or bytes a failed attempt put in place) and could not bring it to its expected state: the
  file is kept as it is and the run stops there;
- **the completion gate** (`completion`): once per run, after the run's final flush and under the corpus lock, every
  managed product is hashed against its expected state. The run's work has been done, but the run does not report
  `completed`: a product that differs (`completion_gate_mismatch`), is missing (`completion_gate_missing`), is present
  where it must be absent (`completion_gate_extra`), or has no committed generation (`completion_gate_refused`, for
  example a pending republication) is named.

Every refusal is recorded as a verdict. `glossapi status` and `glossapi report` (and a streaming run's status) show a run
`completed` only when its own completion gate passed AND that pass is still current: the ledger's latest verdict is a
pass and no product was written, deleted, rolled back or invalidated after it. After a refusal they show
`unresolved_recovery` (status exits 3); when the gate never ran, or later writes have not been verified yet, `unverified`.
A later run whose gate passes makes the corpus current again. The gate's cost (products, bytes read, passes, elapsed)
is part of its verdict event.

`details.conditions` names each path and its reason:

- `hard_linked`: remove the extra hard link.
- `conflicting_locks` or `lock_mismatch`: move or delete the file; it is then republished.
- `path_replaced`: the file was replaced by a symlink; remove the symlink.
- `source_changed` or `no_intact_source`: the managed source under `.glossapi/artifacts` changed or is missing;
  restore it or use another corpus.
- `contradictory_registrations`: the ledger still lists another generation of the file as present; it needs a
  maintainer (report it).
- `present_after_deletion`: the ledger deleted this file by policy but a file is there again; move it away.
- `missing_at_flush`, `missing`, `changed_after_flush`, `changed_after_restore`, `changed_during_check`, `flush_failed`,
  `recovery_error`:
  the file vanished, changed or could not be read or flushed while recovery checked it (the condition names the
  error). Check the disk and anything else writing to the corpus, then resume; the check is repeated from the start.

After the cause is removed, resume: the file is repaired and the run goes on. For a gate refusal, resume after removing
the cause (or rerun the publication that must replace a product with a pending republication); the gate runs again at
the end of that run.

A leftover hidden temporary beside a product (`.<name>.<id>.tmp`) whose bytes recovery cannot prove to be its own
abandoned copy is never deleted and never blocks a run, once the product itself is resolved. It is residue: delete it
yourself if you do not need it.

## `conversion.log` appears in the working directory

Standalone scripts that construct the extractor write `conversion.log` in the working directory unless
`GLOSSAPI_CONVERSION_LOG_FILE` names another path. Point it outside the checkout for long runs.

## Where are my files?

- Enriched Markdown overwrites `markdown/<stem>.md`.
- JSON/indices/latex maps: `json/`. Metrics: `json/metrics/`.
- A streaming run's ledger: `corpus.sqlite3`; its document export: `export.jsonl`; its private stage outputs:
  `.glossapi/artifacts/`.
