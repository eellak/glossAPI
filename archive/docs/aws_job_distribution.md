# AWS Job Distribution Playbook

Last updated: 2025-10-07

## Goal
Build a four-node GPU farm that can run the glossAPI pipeline on the OpenArchives corpus (or any similar large S3-hosted PDF collection) in parallel, using balanced page shards and reproducible environment setup.

## Instance Plan
- Region: `us-east-1` (quota recently increased to 256 vCPUs).
- Target topology: 4 × `g5.12xlarge` (4× NVIDIA A10G 24 GB, 48 vCPUs, 192 GiB RAM each).
- IAM instance profile: `AccesOpenArchivesBucket` (needs S3 read/write and secrets access).
- Security group: default VPC group is sufficient; open SSH only if needed (Session Manager preferred).
- AMI: Amazon Linux 2023 (glibc 2.34, CUDA 12.4-compatible kernel).

## Storage Layout
- Increase root EBS volume to **at least 200 GiB** at launch (`gp3` works well). The default 8 GiB fills quickly with dnf caches, conda, CUDA, and logs, which causes journald failures and SSM commands to break.
- Attach one 3.8 TB NVMe (`/dev/nvme1n1`) provided by the g5.12xlarge and mount at `/mnt/data`.
- Format steps (run once per instance):
  ```bash
  sudo parted /dev/nvme1n1 --script mklabel gpt mkpart primary xfs 0% 100%
  sudo mkfs.xfs /dev/nvme1n1p1
  sudo mkdir -p /mnt/data
  echo '/dev/nvme1n1p1 /mnt/data xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab
  sudo mount -a
  ```
- Move noisy caches off the root volume:
  ```bash
  sudo mkdir -p /mnt/data/{var-cache,dnf,tmp}
  sudo rsync -a /var/cache/dnf/ /mnt/data/var-cache/
  echo 'export TMPDIR=/mnt/data/tmp' | sudo tee /etc/profile.d/tmpdir.sh
  sudo ln -sfn /mnt/data/var-cache /var/cache/dnf
  sudo sed -i 's/#SystemMaxUse=/SystemMaxUse=1G/' /etc/systemd/journald.conf
  sudo systemctl restart systemd-journald
  ```

## Bootstrap Script (user data)
Store the following user data script in S3 and reference it when launching instances (make sure HF token is already injected through Secrets Manager or SSM Parameter Store):

```bash
#!/bin/bash
set -euxo pipefail

# system packages
sudo dnf update -y
sudo dnf install -y git awscli jq htop tmux kernel-modules-extra gcc-c++ make
sudo modprobe nvidia || true

# mount the NVMe scratch disk (see Storage Layout if running manually)
if ! mountpoint -q /mnt/data; then
  mkdir -p /mnt/data
  mkfs.xfs /dev/nvme1n1p1 || true
  mount /dev/nvme1n1p1 /mnt/data
fi

# install Miniforge under /mnt/data to keep root volume lean
MINIFORGE=/mnt/data/miniforge3
if [ ! -x "$MINIFORGE/bin/conda" ]; then
  curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$MINIFORGE"
fi

cat <<'EOF_TOKEN' > /home/ec2-user/.bash_profile
export PATH=/mnt/data/miniforge3/bin:$PATH
export TMPDIR=/mnt/data/tmp
EOF_TOKEN

source /mnt/data/miniforge3/bin/activate || true
conda init bash

# clone glossAPI once and reuse it
cd /mnt/data
if [ ! -d glossAPI ]; then
  git clone https://github.com/eellak/glossAPI.git
fi

aws s3 cp s3://open-archive-data/scripts/conda_setup.sh /mnt/data/conda_setup.sh
chmod +x /mnt/data/conda_setup.sh
/mnt/data/conda_setup.sh --skip-clone --workdir /mnt/data --repo /mnt/data/glossAPI --skip-codex

source /mnt/data/miniforge3/bin/activate glossapi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
pip install pytest
pytest tests -k smoke --maxfail=1 || touch /mnt/data/logs/glossapi_pytest_failed
```

## Hugging Face Token Handling
- Store `HF_TOKEN` in AWS Secrets Manager or SSM Parameter Store.
- During Session Manager shell, export with `export HF_TOKEN=$(aws secretsmanager get-secret-value ...)` before running docling downloads.
- Avoid embedding the token in shell history; use `.env` or `~/.secrets/hf_token` with permission `600` and source it when activating the env.

## Shared Scripts (S3)
- `s3://open-archive-data-use1/scripts/conda_setup.sh`
- `s3://open-archive-data-use1/scripts/open_archive_page_counts.py`
- `s3://open-archive-data-use1/scripts/split_manifest_by_pages.py`

Ensure each node syncs these on boot: `aws s3 sync s3://open-archive-data-use1/scripts /mnt/data/scripts`.

## Page Counting Pipeline
1. Activate the env: `source /mnt/data/miniforge3/bin/activate glossapi`.
2. Run the counter (no local staging):
   ```bash
   cd /mnt/data
   python open_archive_page_counts.py open-archive-data \
     --prefix raw/ \
     --max-workers 64 \
     --chunk-size 2000 \
     --include-size \
     --s3-output s3://open-archive-data-use1/manifests/page_counts \
     --output-file /mnt/data/manifests/page_counts_open_archive.csv \
     --log-level INFO
   ```
3. Monitor progress via `tail -f /mnt/data/logs/page_counts_open_archive.log`.
4. If the process is interrupted, rerun with `--resume-from s3://open-archive-data-use1/manifests/page_counts/page_counts_master.csv` once we stitch the chunks.

## Sharding Strategy
Once `page_counts_open_archive.csv` is complete:
```bash
python split_manifest_by_pages.py s3://open-archive-data-use1/manifests/page_counts/page_counts_master.csv \
  --shards 4 \
  --skip-errors \
  --min-pages 1 \
  --output-prefix s3://open-archive-data-use1/manifests/shards \
  --summary-name shard_summary.json
```
- The script balances on page counts so each node handles roughly 25 % of total pages.
- Output per shard: `shard_1.txt` … `shard_4.txt` stored in S3 and a JSON summary with page totals.

## Work Allocation
- Node naming: tag instances `OpenArchive-Shard-1` … `OpenArchive-Shard-4`.
- On each node, fetch only its manifest: `aws s3 cp s3://.../shard_1.txt /mnt/data/manifests/`.
- Populate local workdir: `/mnt/data/aws_bundle` mirroring the didaktorika layout (downloads, markdown, json, etc.).
- Run the pipeline per file:
  ```bash
  export HF_TOKEN=$(cat ~/.secrets/hf_token)
  cd /mnt/data/glossAPI
  while read -r uri; do
    python run_resume_pipeline.py --input "$uri" --workdir /mnt/data/aws_bundle \
      --output-s3 s3://open-archive-results/shard-1 --math --ocr
  done < /mnt/data/manifests/shard_1.txt
  ```
- Consider adapting `run_shard_template.sh` to read directly from shard manifests and manage success/failure markers under `s3://open-archive-results/status/`.

## Testing Before Full Runs
- Smoke test: `python tests/run_glossapi_test.py --clean --sample parquet_samples/greek_pdf_urls.parquet` (requires sample parquet in `/mnt/data`).
- GPU sanity: `python -c "import torch; print(torch.cuda.get_device_name(0))"`.
- Storage check: `df -h / /mnt/data` should show ample free space (>50 % free on root).

## Monitoring and Logs
- Use `nvidia-smi dmon` and `htop` for GPU/CPU monitoring.
- Write pipeline logs to `/mnt/data/logs/<shard>.log` and ship to CloudWatch if desired.
- Keep track of processed docs via `aws s3 ls s3://open-archive-results/status/ --recursive | grep .done | wc -l`.

## Known Issues & Lessons Learned
- **Root volume exhaustion**: The default 8 GiB root disk fills, causing `systemd-journald` to spam “No space left on device” and SSM to refuse commands. Mitigation: grow the volume before boot or immediately after, move caches to `/mnt/data`, and set journald retention limits.
- **Missing kernel modules**: Install `kernel-modules-extra` before loading the NVIDIA driver (`modprobe nvidia`). Without it, CUDA stays offline.
- **HF token exposure**: Avoid embedding the token in user data; fetch it per session or use dedicated secrets.
- **SSM shell failures**: When `/` is full, `aws ssm send-command` silently fails. Always confirm disk space via `df -h` in bootstrap scripts.

## Next Steps
- Finish page counting job and upload consolidated `page_counts_master.csv`.
- Generate the four shard manifests and double-check balance (`jq` on `shard_summary.json`).
- Validate pipeline on a tiny slice for each shard before full-scale runs.
- Automate shard assignment with SSM commands or a runbook script stored in S3.
