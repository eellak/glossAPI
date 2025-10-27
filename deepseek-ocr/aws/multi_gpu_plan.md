# Multi-GPU DeepSeek-OCR Deployment Plan (AWS g6 4×L4)

## Objectives
- Process large PDF corpora with DeepSeek-OCR using our vLLM runner on 4× NVIDIA L4 GPUs.
- Replicate the current bf16 + FP8 KV configuration used locally, with optional FP8 weight toggles.
- Provide a reproducible environment bootstrap script and operational checklist before launching the instance.

## Target AWS Resources
- **Region:** `us-east-1` (availability of g6 family and proximity to S3/HF endpoints).
- **Instance type:** `g6.12xlarge` → 4× NVIDIA L4 (24 GB each), 24 vCPUs, 96 GB RAM. Alternative: `g6.16xlarge` if more CPU/RAM needed.
- **AMI:** *Deep Learning Base GPU AMI (Ubuntu 22.04) 2024.x* (includes NVIDIA drivers, CUDA 12.4 runtime, conda/mamba). Fallback: latest Ubuntu 22.04 + AWS-provided NVIDIA driver installer.
- **Storage:** 500 GB gp3 EBS (burstable to 1 TB if dataset expansion expected). Throughput 500 MB/s, IOPS ≥ 6k.
- **Networking:** Associate Elastic IP if remote access required; enable placement group optional for multi-instance scale.

## System Packages (via `apt`)
Install after `sudo apt-get update`:
- `git`, `curl`, `wget`, `jq`, `unzip`, `tmux`, `htop`, `nvtop`, `build-essential`, `pkg-config`, `ninja-build`
- Optional utilities for S3 sync / diagnostics: `s5cmd`, `iftop` (if available in repos).
- No additional image libraries or libjpeg builds needed; PyMuPDF and Pillow wheels ship their own binaries.

## Python & Environment Strategy
- Use the AMI’s preinstalled **conda/mamba** (`/opt/conda`) to create an isolated env:
  ```bash
  source /opt/conda/etc/profile.d/conda.sh
  conda create -y -n deepseek-ocr python=3.12
  conda activate deepseek-ocr
  ```
- Install [uv](https://github.com/astral-sh/uv) for fast wheel resolution (optional but recommended):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  ```
- Install **matching CUDA nightly builds** of PyTorch and vLLM (DeepSeek-OCR support landed Oct 23 2025 and requires nightly until v0.11.1):
  ```bash
  uv pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu124 torch torchvision torchaudio
  uv pip install --pre --extra-index-url https://wheels.vllm.ai/nightly vllm
  ```
- Install runner dependencies (no Pillow-SIMD or flash-attn pin required):
  ```bash
  uv pip install PyMuPDF pillow numpy rich tqdm huggingface-hub safetensors "transformers>=4.45.0"
  uv pip install huggingface_hub[hf_transfer]  # optional: accelerates HF downloads
  ```
- Export helpful env vars in `.bashrc` / profile:
  ```bash
  echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc
  echo 'export HF_HUB_DISABLE_TELEMETRY=1' >> ~/.bashrc
  ```

## Model / Data Access
- **Hugging Face token** (`HF_TOKEN`) required to download `deepseek-ai/DeepSeek-OCR` if gated; export and run `huggingface-cli login`.
- Preload checkpoints to `/opt/models/DeepSeek-OCR` (or EBS/local NVMe path) to avoid repeated downloads.
- Upload or sync our repository (e.g., `git clone` from internal repo or `scp` / `aws s3 sync`).

## Multi-GPU Execution Plan
- Use our runner with sharding: launch one process per GPU or a single process with tensor parallelism.
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 python run_pdf_ocr_vllm.py \
    --input-dir /data/pdfs \
    --output-dir /data/outputs/deepseek_vllm_outputs_clean_combined \
    --mode clean \
    --tensor-parallel-size 4 \
    --mm-encoder-tp-mode data \
    --num-shards 4 --shard-index $SHARD_ID \
    --batch-pages 16 --gpu-memory-utilization 0.93 \
    --log-level INFO \
    --timeout-seconds 300  # apply via wrapper script during smoke tests
  ```
- For data parallelism, launch 4 separate tmux panes with `--num-shards 4 --shard-index {0..3}`. Use a shared NFS/EBS path for outputs.
- Keep `timeout 300` wrappers during smoke tests; remove for production runs once stable.

## Monitoring & Housekeeping
- Monitor GPU usage with `watch -n 5 nvidia-smi` and `nvtop`.
- Enable CloudWatch logs/metrics if long jobs.
- Snapshot EBS volume post-run for reproducibility.
- If the instance exposes **local NVMe (instance store)**, mount it (e.g., `/local`) for intermediate PNG caches to reduce EBS churn.

## Credentials & Secrets
- Store `HF_TOKEN`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional Git tokens in AWS Secrets Manager or SSM Parameter Store. Pass them via IAM instance profile or `aws ssm get-parameter` during setup.
- Avoid hardcoding tokens in scripts; use environment exports or `.env` files with restricted permissions.

## Next Actions Before Launch
1. Prepare environment bootstrap script (`aws/setup_multi_gpu_env.sh`).
2. Stage sample PDFs / URLs (10 Greek-language files) for validation.
3. Confirm S3 bucket or EFS path for long-term storage if outputs exceed instance lifespan.
4. Decide on orchestration (manual tmux vs. SLURM/PM2).

## Preflight Checklist on Instance
1. `nvidia-smi` → confirm 4× L4 GPUs and driver R550+.
2. `python - <<'PY'` sanity script to report Torch/vLLM versions and CUDA availability.
3. Run single-PDF smoke test with `timeout 300` and `--batch-pages 1` to confirm decoding emits text (no blank outputs).
4. Record baseline throughput (pages/s) before full run; adjust `--batch-pages` while keeping ~1–2 GB VRAM headroom.
