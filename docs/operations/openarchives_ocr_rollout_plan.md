# OpenArchives OCR Rollout Plan

This document records the concrete execution plan for running DeepSeek OCR over the OpenArchives subset with `needs_ocr=True`, including how to recover or regenerate the routing state, how to shard work across AWS nodes, and how to merge results back into the canonical GlossAPI corpus.

## Current validated baseline

- Validated OCR node type: `g7e.48xlarge`
- Validated AMI: `ami-052266c3e21dff7db`
- AMI name: `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20260320`
- Validated runtime stack on the OCR node:
  - `torch 2.10.0+cu130`
  - `vllm 0.18.0`
  - `transformers 4.57.6`
- Standard DeepSeek settings:
  - `runtime_backend='vllm'`
  - `ocr_profile='markdown_grounded'`
  - `max_new_tokens=2048`
  - `repair_mode='auto'`
  - `render_dpi=144`
  - `gpu_memory_utilization=0.9`
- Restored clean benchmark on the stopped OCR box:
  - `7,624` pages in about `541s`
  - about `0.0710 sec/page` overall on one `8`-GPU node
  - about `0.3927` to `0.5000 sec/page/GPU`
- Derived per-node throughput:
  - about `14.08 pages/sec`
  - about `50,700 pages/hour`

## Current AWS capacity

`us-east-1` service quotas currently allow:

- `Running On-Demand G and VT instances = 768`
- `Running On-Demand Standard instances = 640`

For the validated OCR node:

- `g7e.48xlarge = 192 vCPU, 8 GPUs`

So the current maximum concurrent validated OCR fleet is:

- `floor(768 / 192) = 4` nodes
- total rollout capacity: `32 GPUs`

## Phase 1: Recover or regenerate the canonical OCR routing state

Goal:

- produce one canonical `download_results/download_results.parquet` for the OpenArchives corpus root
- ensure it contains, at minimum:
  - `filename`
  - `needs_ocr`
  - `greek_badness_score`
  - `mojibake_badness_score`
  - `ocr_success`
  - `page_count` or `pages_total`

Decision order:

1. Check the stopped GPU OCR instance first.
2. If the full corpus parquet is not there, run a dedicated CPU cleaning pass.

### 1A. Check the stopped OCR instance first

Reason:

- the NVMe volume persists across stop/start
- if the full OpenArchives cleaning pass was already run there, this is the fastest path

Concrete steps:

1. Start instance `i-0504a326a1fee541f`.
2. SSH in and search for the full OpenArchives corpus root and canonical parquet:
   - `find /opt /data /home -name download_results.parquet`
   - verify row count is the full OpenArchives set, not the `43`-document benchmark subset
3. Validate that the parquet has the required OCR routing columns listed above.
4. If found:
   - copy the canonical parquet and any supporting cleaner outputs back to stable storage
   - stage a copy on `home`
   - upload the parquet artifact to the Hugging Face dataset repo as routing metadata

Acceptance check:

- row count matches the full OpenArchives working set
- `needs_ocr=True` count is available directly from the parquet
- page totals are available

### 1B. Fallback: regenerate the routing state on a CPU instance

If the OCR box does not contain the full canonical parquet:

- launch a dedicated CPU node for the cleaner pass
- recommended instance family: `c7i` or `r7i`
- recommended first choice: `c7i.8xlarge` with sufficient gp3 storage for the OpenArchives markdown/output root

Reason:

- `Corpus.clean()` is CPU-bound and does not need GPUs
- we only need one clean, reproducible routing pass

Concrete steps:

1. Launch one Ubuntu 24.04 CPU instance.
2. Clone `glossapi-development` at `development`.
3. Bootstrap the standard GlossAPI environment.
4. Mount or sync the full OpenArchives corpus root.
5. Run `Corpus.clean()` over the full markdown corpus.
6. Verify that `download_results/download_results.parquet` now exists and includes the required OCR routing columns.
7. Store the resulting parquet:
   - on the corpus root
   - on `home`
   - in the Hugging Face dataset repo as routing metadata

## Phase 2: Quantify the actual OCR workload

Once the canonical parquet exists:

1. Filter `needs_ocr == True`
2. Count:
   - total documents
   - total pages from `pages_total` or `page_count`
3. Also record:
   - `greek_badness_score > 60`
   - `mojibake_badness_score > 0.1`
   - overlap between those conditions and `needs_ocr`

This step defines the real production workload and the true ETA.

## Phase 3: Shard across nodes

Shard across nodes by document, not by page range.

Reason:

- cross-node merge stays trivial
- node-local GPU scheduling already exists in GlossAPI
- splitting one document across nodes adds complexity without clear benefit

### Coordinator manifest

Build one coordinator manifest from the canonical parquet with:

- `filename`
- stable OpenArchives document id or canonical filename
- `pages_total`
- `needs_ocr`

Then:

1. keep only `needs_ocr=True`
2. greedily bin-pack documents across `N=4` nodes by page count
3. write one shard manifest parquet per node

Each shard manifest should contain:

- `filename`
- `pages_total`
- `node_id`
- `shard_id`
- original metadata keys needed for rejoin

### Node-local execution

Each node:

1. loads only its shard manifest
2. runs GlossAPI OCR over that subset
3. keeps standard GlossAPI outputs only:
   - `markdown/<stem>.md`
   - `json/metrics/*.json`
   - shard-local `download_results.parquet`

Inside each node:

- use the existing GlossAPI DeepSeek path
- let node-local scheduling handle GPU balance
- do not invent a separate OCR metadata format

## Phase 4: Merge back into the canonical corpus

Merge rules:

1. Markdown:
   - copy updated `markdown/<stem>.md` into the canonical corpus root
2. Metrics:
   - copy `json/metrics/*.json` into the canonical corpus root
3. Metadata parquet:
   - concatenate shard metadata
   - upsert by canonical document id / filename into the master parquet
   - preserve the standard GlossAPI contract:
     - `needs_ocr`
     - `ocr_success`
     - `processing_stage`
     - page and quality fields

Recommended additional execution metadata:

- `ocr_node_id`
- `ocr_shard_id`
- `ocr_started_at`
- `ocr_finished_at`
- `ocr_attempt_count`

These fields are operational and should not replace the existing GlossAPI routing fields.

## Phase 5: Standardize all OCR nodes

All OCR nodes should use the exact same:

- AMI
- bootstrap script
- DeepSeek venv setup
- model path
- runtime defaults

Standard production recipe:

- AMI: `ami-052266c3e21dff7db`
- instance type: `g7e.48xlarge`
- DeepSeek venv created by `dependency_setup/setup_deepseek_uv.sh`
- defaults:
  - `runtime_backend='vllm'`
  - `ocr_profile='markdown_grounded'`
  - `max_new_tokens=2048`
  - `repair_mode='auto'`
  - `render_dpi=144`
  - `gpu_memory_utilization=0.9`

Do not allow per-node env drift during the rollout.

## Phase 6: ETA

Validated throughput on one node:

- about `50,700 pages/hour`

With `4` nodes:

- about `202,800 pages/hour`

Exact ETA formula:

- `ETA_hours = total_needs_ocr_pages / 202800`

Reference scenarios:

- `400,000` pages: about `1.97h`
- `600,000` pages: about `2.96h`
- `800,000` pages: about `3.95h`
- `1,000,000` pages: about `4.93h`

Equivalent document scenarios for `40,000` documents:

- average `10` pages/doc: about `1.97h`
- average `15` pages/doc: about `2.96h`
- average `20` pages/doc: about `3.95h`
- average `25` pages/doc: about `4.93h`

The exact ETA should be recalculated once the canonical parquet gives the real total page count for `needs_ocr=True`.

## Phase 7: Deployment and monitoring

### Deployment

1. Produce canonical parquet
2. Compute shard manifests
3. Stage manifests and source data
4. Launch `4` OCR nodes
5. Bootstrap the same OCR environment on all nodes
6. Run one shard per node
7. Collect outputs
8. Merge back into the canonical corpus

### Monitoring

Each node should write a heartbeat JSON at a fixed interval with:

- `node_id`
- `docs_done`
- `pages_done`
- current file
- GPU utilization snapshot
- VRAM usage snapshot
- last successful write time
- error count

The coordinator should watch:

- stale heartbeat
- zero progress
- failed OCR process
- low GPU utilization for a sustained period

### Recovery

- rerun only failed shard manifests
- keep shard manifests immutable
- merge is idempotent by canonical document id / filename

## Immediate next actions

1. Start the stopped OCR instance and search for the full OpenArchives canonical parquet.
2. If found, validate and upload the routing parquet to stable storage and Hugging Face.
3. If not found, launch one CPU instance and run the full `Corpus.clean()` pass.
4. Compute exact `needs_ocr` doc/page totals from the canonical parquet.
5. Generate the `4` node shard manifests.
6. Launch the `4` OCR nodes and execute the distributed run.
