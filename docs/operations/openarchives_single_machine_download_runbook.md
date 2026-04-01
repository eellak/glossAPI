# OpenArchives Single-Machine Download And Greek Supplement Runbook

This runbook documents how to reproduce the current OpenArchives download corpus on a single remote machine, using:

- the normal GlossAPI downloader for the source-side pull
- a deterministic NTUA retry pass
- a single combined high-priority list for Greek-box supplementation

This document is repo-safe. It does not store the Greek-box password in git. A private companion note with the password and direct copy commands is stored outside the repo.

## Goal

Rebuild the currently available OpenArchives PDF corpus on one machine so it can later be:

- frozen as a local PDF corpus
- backed up
- sharded for OCR

The intended end state is:

1. source-download pass from the original OA URLs
2. targeted retry pass for the NTUA rows that the first pass stranded
3. supplementation from the Greek backup box for the high-priority unreachable set

## Source Of Truth

The OA routing state used here comes from the enriched OpenArchives parquet:

- current cleaner-box path:
  - `/home/ubuntu/glossapi/work/needs_ocr_enriched.parquet`

This enriched parquet is produced from:

- canonical OA document-level parquet after fill/clean
- raw HF OpenArchives dataset snapshot

The script that creates it is:

- [openarchives_ocr_enrich.py](/Users/foivoskarounos-zamparloukos/Projects/glossapi-development/src/glossapi/scripts/openarchives_ocr_enrich.py)

CLI contract:

```bash
PYTHONPATH=src python -m glossapi.scripts.openarchives_ocr_enrich \
  --parquet /data/openarchives/filled_document_level.parquet \
  --raw-repo-root /data/openarchives_hf \
  --output-parquet /data/openarchives/needs_ocr_enriched.parquet
```

If `needs_ocr_enriched.parquet` is already available, use it directly.

## Required Inputs

You need:

1. GlossAPI checkout at `development`
2. the sample OA download policy file:
   - [openarchives_download_policy.yml](/Users/foivoskarounos-zamparloukos/Projects/glossapi-development/samples/openarchives_download_policy.yml)
3. a local snapshot of the HF dataset `glossAPI/openarchives.gr`, at least:
   - `data/openarchives/**`
4. the enriched routing parquet:
   - `needs_ocr_enriched.parquet`
5. for Greek supplementation:
   - the combined high-priority list

Current combined high-priority list:

- transfer box:
  - `/home/ubuntu/openarchives_stage/unreachable_from_source_20260331/priority_high_combined_20260401.csv`
  - `/home/ubuntu/openarchives_stage/unreachable_from_source_20260331/priority_high_combined_20260401.txt`
- local copy:
  - [priority_high_combined_20260401.csv](/Users/foivoskarounos-zamparloukos/Projects/glossapi-ocr-local/reports/artifacts-20260401-source-backup/priority_high_combined_20260401.csv)
  - [priority_high_combined_20260401.txt](/Users/foivoskarounos-zamparloukos/Projects/glossapi-ocr-local/reports/artifacts-20260401-source-backup/priority_high_combined_20260401.txt)

That combined list is the union of:

- `unreachable_from_source_20260331.csv`
- `hard_fail_from_source_20260401.csv`

Current combined count:

- `14,532` filenames

## HF Dataset Snapshot

Recommended pull method:

```bash
export HF_TOKEN=...
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="glossAPI/openarchives.gr",
    repo_type="dataset",
    local_dir="/data/openarchives_hf",
    allow_patterns=[
        "README.md",
        "data/openarchives/**",
    ],
    token=True,
)
PY
```

If you already have a cleaner-produced enriched parquet, keep it outside the HF snapshot, for example:

- `/data/openarchives_work/needs_ocr_enriched.parquet`

## Collections Targeted

This workflow targets every OA row with `needs_ocr=True` in the enriched parquet.

Current collection breakdown from the enriched parquet:

```text
IKEE_AUT  19966
ntua      8118
Pandemos  3441
Dione     3170
hellanicus 2121
elocus    1647
estia     1024
dias      850
helios    701
edulll    643
KEIED     549
elstat    542
nemertes  427
cetpe     427
deltion   374
aua       322
apothesis 299
kallipos  254
ekke      143
JHVMS     126
anaktisis 121
ktisis    105
KEEF      103
ariadne   73
geosociety 1
```

## Host Split Used In Practice

The current download plan was not collection-first. It was host-first.

Phase 1 bulk-good-host source download:

- include all `needs_ocr=True` rows except:
  - `ikee.lib.auth.gr`
  - `olympias.lib.uoi.gr`

Targeted NTUA retry pass:

- rows from phase 1 with:
  - `download_success == false`
  - `download_error == ""`
  - `host == "dspace.lib.ntua.gr"`

Greek-box supplementation:

- union of:
  - original source-unreachable set
  - explicit hard source failures

## Exact Download Policy

Use:

- [openarchives_download_policy.yml](/Users/foivoskarounos-zamparloukos/Projects/glossapi-development/samples/openarchives_download_policy.yml)

Current important host rules:

- `ikee.lib.auth.gr`
  - `request_timeout: 180`
  - `per_domain_concurrency: 1`
  - `sleep: 1.5`
- `dspace.lib.ntua.gr`
  - `request_timeout: 120`
  - `per_domain_concurrency: 1`
  - `sleep: 1.0`
- `olympias.lib.uoi.gr`
  - `request_timeout: 180`
  - `ssl_verify: false`
  - `per_domain_concurrency: 1`
  - `sleep: 1.0`
- `ktisis.cut.ac.cy`
  - `ssl_verify: false`
- `repository.academyofathens.gr`
  - high concurrency, short timeout

## GlossAPI Downloader Entry Point

Use:

- [openarchives_download_freeze.py](/Users/foivoskarounos-zamparloukos/Projects/glossapi-development/src/glossapi/scripts/openarchives_download_freeze.py)

The script:

1. reads one manifest parquet
2. downloads PDFs into `downloads/`
3. writes canonical `download_results/download_results.parquet`
4. stops without OCR

Important arguments:

- `--input-parquet`
- `--work-root`
- `--download-concurrency`
- `--download-timeout`
- `--download-scheduler-mode`
- `--download-group-by`
- `--download-policy-file`

Current proven values for the source bulk pass:

- `download_concurrency = 24`
- `download_timeout = 60`
- `download_scheduler_mode = per_domain`
- `download_group_by = base_domain`
- `download_policy_file = samples/openarchives_download_policy.yml`

## Single-Machine Directory Layout

Suggested layout:

```text
/data/openarchives_hf
/data/openarchives_work/needs_ocr_enriched.parquet
/data/openarchives_work/openarchives_download_phases_20260401/
/data/openarchives_work/phase1_bulk_good_hosts/
/data/openarchives_work/retry_ntua_zeroerror_all/
/data/openarchives_work/greek_supplement/
```

## Step 1: Build The Bulk-Good-Hosts Manifest

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

src = Path("/data/openarchives_work/needs_ocr_enriched.parquet")
outdir = Path("/data/openarchives_work/openarchives_download_phases_20260401")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(src)
target = df[df["needs_ocr"].fillna(False)].copy()
host = target["host"].fillna("").astype(str)
bulk = target[~host.isin(["ikee.lib.auth.gr", "olympias.lib.uoi.gr"])].copy()

bulk.to_parquet(outdir / "phase1_bulk_good_hosts.parquet", index=False)
host.value_counts().rename_axis("host").reset_index(name="docs").to_parquet(
    outdir / "phase1_bulk_good_hosts_host_stats.parquet", index=False
)
print({
    "bulk_docs": int(len(bulk)),
    "bulk_pages": float(pd.to_numeric(bulk.get("pages_total_source", 0), errors="coerce").fillna(0).sum()),
})
PY
```

## Step 2: Run The Bulk Source Download

```bash
cd /path/to/glossapi-development
PYTHONPATH=src python -m glossapi.scripts.openarchives_download_freeze \
  --input-parquet /data/openarchives_work/openarchives_download_phases_20260401/phase1_bulk_good_hosts.parquet \
  --work-root /data/openarchives_work/phase1_bulk_good_hosts \
  --download-concurrency 24 \
  --download-timeout 60 \
  --download-scheduler-mode per_domain \
  --download-group-by base_domain \
  --download-policy-file samples/openarchives_download_policy.yml
```

Outputs:

- `/data/openarchives_work/phase1_bulk_good_hosts/downloads/*.pdf`
- `/data/openarchives_work/phase1_bulk_good_hosts/download_results/download_results.parquet`
- `/data/openarchives_work/phase1_bulk_good_hosts/download_results/download_results_download_input.parquet`

## Step 3: Build The NTUA Retry Manifest

This is the critical correction from the first run.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

src = Path("/data/openarchives_work/phase1_bulk_good_hosts/download_results/download_results_download_input.parquet")
outdir = Path("/data/openarchives_work/openarchives_retry_20260401")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(src)
err = df["download_error"].fillna("").astype(str)
host = df["host"].fillna("").astype(str)
succ = df["download_success"].fillna(False)

ntua = df[(~succ) & err.eq("") & host.eq("dspace.lib.ntua.gr")].copy()
ntua.to_parquet(outdir / "ntua_zeroerror_all.parquet", index=False)
ntua.to_csv(outdir / "ntua_zeroerror_all.csv", index=False)
print({"ntua_retry_rows": int(len(ntua))})
PY
```

## Step 4: Run The NTUA Retry

Use the exact settings that recovered `6879/6879` in the current run:

```bash
cd /path/to/glossapi-development
PYTHONPATH=src python -m glossapi.scripts.openarchives_download_freeze \
  --input-parquet /data/openarchives_work/openarchives_retry_20260401/ntua_zeroerror_all.parquet \
  --work-root /data/openarchives_work/retry_ntua_zeroerror_all \
  --download-concurrency 6 \
  --download-timeout 90 \
  --download-scheduler-mode global
```

Do not reuse the phase-1 per-domain setup for this retry. The global retry pattern is what recovered the full missed NTUA bucket.

## Step 5: Greek-Box Supplementation

You need:

- the single combined priority list
- SSH access to the Greek box
- the Greek raw path

Repo-safe details:

- host: `83.212.80.170`
- user: `debian`
- raw path:
  - `/glossapi/1000/s3-backup/open-archive-data/raw`

For the password and direct commands, see the private companion note outside the repo.

### 5A. Copy The Combined Priority List To The Remote Machine

Use:

- `priority_high_combined_20260401.txt`

### 5B. Reduce It To The Still-Missing Local Files

On the single remote machine:

```bash
find /data/openarchives_work/phase1_bulk_good_hosts/downloads -maxdepth 1 -type f -name '*.pdf' -printf '%f\n' | sort -u > /tmp/source_phase1_have.txt
find /data/openarchives_work/retry_ntua_zeroerror_all/downloads -maxdepth 1 -type f -name '*.pdf' -printf '%f\n' | sort -u > /tmp/source_ntua_have.txt
cat /tmp/source_phase1_have.txt /tmp/source_ntua_have.txt | sort -u > /tmp/source_have_all.txt
comm -23 priority_high_combined_20260401.txt /tmp/source_have_all.txt > /tmp/priority_still_missing.txt
```

### 5C. Build The Matched Greek Relative-Path Manifest

On the Greek box, generate the list of relative PDF paths whose basenames are in `priority_still_missing.txt`.

Because the Greek raw tree is nested under `.part_*`, the supplement step must be path-aware. The simplest reproducible pattern is:

```bash
python3 - <<'PY'
from pathlib import Path
wanted = set(Path('/tmp/priority_still_missing.txt').read_text().splitlines())
raw_root = Path('/glossapi/1000/s3-backup/open-archive-data/raw')
out = Path('/tmp/greek_matched_relative_paths.txt')
with out.open('w') as f:
    for p in raw_root.rglob('*.pdf'):
        if p.name in wanted:
            f.write(str(p.relative_to(raw_root)) + '\n')
print(out)
PY
```

### 5D. Pull The Missing Files From The Greek Box

From the single remote machine:

```bash
mkdir -p /data/openarchives_work/greek_supplement/downloads
rsync -av --files-from=/tmp/greek_matched_relative_paths.txt \
  debian@83.212.80.170:/glossapi/1000/s3-backup/open-archive-data/raw/ \
  /data/openarchives_work/greek_supplement/raw/
find /data/openarchives_work/greek_supplement/raw -type f -name '*.pdf' -exec cp -n {} /data/openarchives_work/greek_supplement/downloads/ \\;
```

That produces a flat supplement directory of Greek-recovered PDFs.

## Step 6: Freeze The Final Available Corpus

Final available set is the union of:

- `/data/openarchives_work/phase1_bulk_good_hosts/downloads`
- `/data/openarchives_work/retry_ntua_zeroerror_all/downloads`
- `/data/openarchives_work/greek_supplement/downloads`

At that point:

- rebuild the cutoff inventory
- shard for OCR
- or archive the source-only-new subset if desired

## Notes

- The Greek priority queue was built as one combined list so the supplement step is reproducible on a single remote instance.
- The first bulk run missed many NTUA files for execution reasons, not because the URLs were dead. The dedicated NTUA retry is therefore mandatory.
- The Greek box should be treated as a supplement path, not the default source path.
