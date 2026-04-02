# DELETE ME: DeepSeek Reliability Pending Work

This note is temporary. Delete it after the first production soak confirms the
merged reliability path is stable and the follow-up items below are either done
or explicitly discarded.

## What shipped in this merge

- durable multi-GPU DeepSeek work queue with separate main and repair phases
- worker respawn with process-group teardown so orphaned `VLLM::EngineCore`
  processes do not pin VRAM after a crash
- GPU preflight and telemetry sidecars under `sidecars/ocr_runtime/`
- steady-state timing in the runtime summary
- default work-item retry ceiling of two total attempts
  - first failure: retry once
  - second failure: mark the batch failed and stop retrying it

## Pending follow-up

1. Capture and archive one clean fault-injection receipt on the merged
   `development` branch.
   - Goal: preserve one explicit production-like run where a worker is killed
     mid-run, the supervisor respawns it, the in-flight batch is retried once,
     and the run still completes.

2. Add operator-facing handling for terminally failed batches.
   - The durable queue already marks them `failed`.
   - The remaining work is a cleaner operator handoff, for example a dedicated
     quarantine/export path or a documented replay workflow.

3. Replace the current image-content stats implementation in
   `run_pdf_ocr_vllm.py`.
   - It still uses a CPU-heavy PIL pixel scan and currently emits a Pillow
     deprecation warning.

4. Run a longer unattended soak after merge.
   - The current validation covers targeted tests, full end-to-end runs, and
     reliability-path implementation, but production confidence still benefits
     from a longer multi-hour burn-in on the merged branch.
