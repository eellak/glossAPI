from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DOCS = ("IBK_476", "FQA_524", "ZKI_504")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_single_gpu_benchmark",
        description="Run the standardized 3-PDF OpenArchives single-GPU OCR benchmark.",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--downloads-dir", required=True)
    parser.add_argument("--source-shard-parquet", required=True)
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--runner-script", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--keep-work-root", action="store_true")
    return parser.parse_args()


def _resolve_python_bin(repo_root: Path, explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    candidate = repo_root / "dependency_setup" / ".venvs" / "deepseek" / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def _summary_for_run(run_root: Path) -> Dict[str, Any]:
    runtime_summary = json.loads(
        (run_root / "sidecars" / "ocr_runtime" / "runtime_summary.json").read_text(encoding="utf-8")
    )
    window = float(runtime_summary["steady_state"]["first_batch_to_last_batch_window_sec"])
    docs: Dict[str, Any] = {}
    total_pages = 0
    for stem in DEFAULT_DOCS:
        metrics = json.loads((run_root / "json" / "metrics" / f"{stem}.metrics.json").read_text(encoding="utf-8"))
        markdown_path = run_root / "markdown" / f"{stem}.md"
        pages = int(metrics["page_count"])
        total_pages += pages
        docs[stem] = {
            "markdown_path": str(markdown_path),
            "markdown_bytes": int(markdown_path.stat().st_size),
            "markdown_sha256": hashlib.sha256(markdown_path.read_bytes()).hexdigest(),
            "pages": pages,
            "wall_time_sec": float(metrics.get("wall_time_sec", 0.0)),
            "wall_sec_per_page": float(metrics.get("wall_time_sec", 0.0)) / pages if pages else None,
            "infer_sec_total": float(metrics.get("infer_sec_total", 0.0)),
            "infer_sec_per_page": float(metrics.get("infer_sec_total", 0.0)) / pages if pages else None,
        }
    return {
        "run_root": str(run_root),
        "steady_state_window_sec": window,
        "total_pages": total_pages,
        "overall_sec_per_page": window / total_pages if total_pages else None,
        "sec_per_page_per_gpu": window / total_pages if total_pages else None,
        "documents": docs,
    }


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    work_root = Path(args.work_root).resolve()
    downloads_dir = Path(args.downloads_dir).resolve()
    source_shard_parquet = Path(args.source_shard_parquet).resolve()
    python_bin = _resolve_python_bin(repo_root, args.python_bin)
    runner_script = (
        Path(args.runner_script).resolve()
        if args.runner_script
        else repo_root / "src" / "glossapi" / "ocr" / "deepseek" / "run_pdf_ocr_vllm.py"
    )

    if work_root.exists() and not args.keep_work_root:
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    shard_parquet = work_root / source_shard_parquet.name
    shutil.copy2(source_shard_parquet, shard_parquet)
    downloads_link = work_root / "downloads"
    if downloads_link.exists() or downloads_link.is_symlink():
        if downloads_link.is_dir() and not downloads_link.is_symlink():
            shutil.rmtree(downloads_link)
        else:
            downloads_link.unlink()
    os.symlink(downloads_dir, downloads_link)

    runtime_report_path = work_root / "runtime_report.json"
    runtime_report = subprocess.run(
        [
            str(python_bin),
            "-m",
            "glossapi.scripts.deepseek_runtime_report",
            "--repo-root",
            str(repo_root),
            "--python-bin",
            str(python_bin),
            "--json",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(repo_root),
    )
    runtime_report_path.write_text(runtime_report.stdout, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    env["GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT"] = str(runner_script)

    cmd = [
        str(python_bin),
        "-m",
        "glossapi.scripts.openarchives_ocr_run_node",
        "--shard-parquet",
        str(shard_parquet),
        "--work-root",
        str(work_root),
        "--skip-download",
        "--runtime-backend",
        "vllm",
        "--ocr-profile",
        "markdown_grounded",
        "--repair-mode",
        "auto",
        "--scheduler",
        "exact_fill",
        "--target-batch-pages",
        "96",
        "--workers-per-gpu",
        "1",
        "--render-dpi",
        "144",
        "--max-new-tokens",
        "2048",
        "--gpu-memory-utilization",
        "0.9",
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)

    summary = _summary_for_run(work_root)
    summary["repo_head"] = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    summary["label"] = args.label
    summary["runtime_report_path"] = str(runtime_report_path)

    summary_path = work_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
