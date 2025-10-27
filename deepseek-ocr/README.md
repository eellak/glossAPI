# DeepSeek-OCR Runner Snapshot

This folder is a trimmed export of the DeepSeek-OCR automation work so it can be pushed to a remote repo without the 6 GB checkpoint or local build artifacts.

- `run_pdf_ocr_vllm.py` – high-throughput vLLM runner with batching, blank-page skips, ROI retries, and multi-GPU sharding flags.
- `run_pdf_ocr.py` – minimal single-GPU baseline that uses the model’s native `.infer` API.
- `DeepSeek-OCR-empty/` – directory structure the runners expect; download the real weights from Hugging Face and overwrite the placeholder files.
- `aws/` – environment bootstrap script, deployment checklist, scaling plan, and the 10-sample PDF manifest used for regression runs.
- `NOTES.md` – operational logbook with build timings, throughput baselines, tuning guidance, and a quickstart guide for reproducing the setup.

For fresh environments, follow the quickstart section in `NOTES.md`, then run the vLLM runner on your PDF corpus. The AWS scripts assume a g6.12xlarge (4× L4) instance but can be adapted to other CUDA 12.4+ stacks with minor tweaks.
