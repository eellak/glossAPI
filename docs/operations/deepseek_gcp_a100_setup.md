# DeepSeek GCP A100 Setup

This note captures the current known-good baseline for bringing up GlossAPI
DeepSeek OCR on fresh GCP A100 nodes and the required diagnosis workflow when a
fresh node does not behave like the already-converged fleet.

## Goal

Treat a fresh OCR node as a reproducible setup target, not as a one-off machine
that is repaired interactively until it happens to work.

The target is a clean path from:

1. create instance
2. bootstrap machine
3. prepare GlossAPI runtime
4. run a normal GlossAPI OCR workflow

## Known-good baseline

This rollout has validated the following stack on working OCR fleet nodes:

- Ubuntu `22.04.5`
- NVIDIA driver `590.48.01`
- `A100 40GB` GPUs
- host Python `3.10`
- DeepSeek venv Python `3.11`
- `torch 2.10.0+cu130`
- `vllm 0.18.0`
- `transformers 4.57.6`
- `workers_per_gpu=1`

The runner also expects GPU persistence mode to be enabled and will record the
preflight result under `sidecars/ocr_runtime/gpu_preflight.json`.

## First command on a fresh node

Run the checked-in runtime report before changing code or applying ad hoc fixes:

```bash
python -m glossapi.scripts.deepseek_runtime_report --repo-root /opt/glossapi/repo
```

The report prints:

- OS and hostname
- repo revision
- GPU model, driver, and memory
- selected Python executable and venv root
- `torch` / `vllm` / `transformers` import details
- wheel-managed NVIDIA library directories
- a focused `pip freeze` subset
- selected runtime environment variables

Prefer comparing this output against a known-good OCR node before modifying
GlossAPI itself.

## Fresh-node diagnosis rule

If a fresh node fails, classify the problem before patching code:

1. instance creation choice
   - wrong image
   - wrong driver path
   - wrong machine family or GPU shape
2. bootstrap incompleteness
   - missing system packages
   - missing wheel-managed CUDA libraries
   - model / cache / filesystem layout mismatch
   - missing env wiring
3. actual GlossAPI runtime assumption
   - hidden dependency on a particular venv layout
   - hidden dependency on a specific CUDA wheel layout
   - hidden runner / vLLM startup assumption

Write down which class the current failure belongs to before making broad code
changes.

## Current benchmark-node findings

The fresh `a2-highgpu-2g` benchmark node used during the April 3, 2026 work
surfaced two setup classes:

- early missing shared-library failure:
  - `ImportError: libcudart.so.12: cannot open shared object file`
- later engine startup failure after bootstrap fixes:
  - `RuntimeError: Engine core initialization failed. Failed core proc(s): {}`

This means instance creation itself worked, but bootstrap/runtime reproducibility
was incomplete.

## Current runner expectation

`glossapi.ocr.deepseek.runner._build_env()` now auto-discovers
`site-packages/nvidia/*/lib` directories under the selected DeepSeek virtualenv
and prepends them to `LD_LIBRARY_PATH`.

This is the right place to normalize wheel-managed CUDA library discovery. Do
not rely on manual shell-session exports as the primary contract.

## Practical bring-up checklist

1. confirm the node matches the OS / driver baseline
2. run `deepseek_runtime_report`
3. compare report output to a known-good node
4. fix bootstrap mismatches first
5. rerun the report
6. only then run a small OCR validation workload
7. if OCR still fails, inspect worker logs and decide whether the remaining gap
   belongs in GlossAPI runtime code or external bootstrap
