# DeepSeek Runner Stack

This note explains the current structure of the DeepSeek OCR runner after the
`runner.py` decomposition.

## Public Entry Point

`src/glossapi/ocr/deepseek/runner.py` is still the public shim used by
`Corpus.ocr()` and by the runner contract tests.

It intentionally keeps:

- `run_for_files()`
- the public constants used by callers/tests
- thin wrappers for monkeypatch-sensitive helpers such as `_run_multi_cli()`,
  `_ensure_gpu_preflight()`, and `_launch_worker_process()`

That shim exists so internal refactors do not force every caller or test to
learn new module paths.

## Internal Modules

The runner internals now split across focused helpers:

- `runner_planning.py`
  - GPU/device discovery
  - page counting
  - lane planning
  - exact-fill and shard batch planning
  - automatic vLLM batch-size derivation
- `runner_runtime_support.py`
  - runtime-summary generation
  - GPU preflight and persistence-mode checks
  - telemetry helpers
  - XID fault collection
  - worker process launch and process-group shutdown helpers
- `runner_reassembly.py`
  - shard-stem parsing
  - canonical markdown/metrics reassembly
  - archival of shard artifacts under `sidecars/ocr_shards`
- `launcher.py`
  - CLI command construction
  - runtime environment construction
- `work_queue.py`
  - durable queue state for vLLM worker batches and repair batches
- `scheduling.py`
  - core slice/batch/lane planning primitives

## Why The Shim Still Matters

Some runner tests patch private helpers directly on `runner.py`.

Examples:

- patching `_run_cli()` or `_run_multi_cli()` during `run_for_files()` tests
- patching `_page_count()` during planning tests
- patching `_query_persistence_mode()` and `runner.subprocess.run` during GPU preflight tests

Because of that, `runner.py` should stay a stable compatibility layer even when
the implementation moves behind it.

## Safe Change Rules

- If you are changing public OCR orchestration, start in `runner.py`.
- If you are changing batch planning logic, start in `runner_planning.py`.
- If you are changing runtime telemetry or worker lifecycle plumbing, start in
  `runner_runtime_support.py`.
- If you are changing shard-to-canonical artifact behavior, start in
  `runner_reassembly.py`.
- Keep `run_for_files()` readable. Push policy or low-level mechanics downward
  unless the public orchestration flow itself changes.

## Tests That Protect This Stack

- `tests/test_deepseek_scheduling.py`
- `tests/test_deepseek_runner_contract.py`
- `tests/test_deepseek_multi_gpu_runtime.py`
- `tests/test_openarchives_single_gpu_benchmark.py`
