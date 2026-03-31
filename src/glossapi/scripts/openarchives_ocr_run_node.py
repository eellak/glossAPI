from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from glossapi import Corpus
from glossapi.parquet_schema import ParquetSchema


DEFAULT_DOWNLOAD_CONCURRENCY = 24
DEFAULT_DOWNLOAD_TIMEOUT = 60
DEFAULT_HEARTBEAT_INTERVAL = 60


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_ocr_run_node",
        description=(
            "Materialize one OpenArchives OCR shard into a normal GlossAPI corpus root, "
            "download its PDFs, and run DeepSeek OCR with the standardized settings."
        ),
    )
    p.add_argument("--shard-parquet", required=True)
    p.add_argument("--work-root", required=True)
    p.add_argument("--python-log-level", default="INFO")
    p.add_argument("--download-concurrency", type=int, default=DEFAULT_DOWNLOAD_CONCURRENCY)
    p.add_argument("--download-timeout", type=int, default=DEFAULT_DOWNLOAD_TIMEOUT)
    p.add_argument("--download-group-by", default="repository_collection")
    p.add_argument("--heartbeat-path")
    p.add_argument("--heartbeat-interval", type=int, default=DEFAULT_HEARTBEAT_INTERVAL)
    p.add_argument("--instance-id", default="")
    p.add_argument("--node-id", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--scheduler", default="whole_doc")
    p.add_argument("--target-batch-pages", type=int, default=160)
    p.add_argument("--shard-pages", type=int, default=0)
    p.add_argument("--shard-threshold-pages", type=int, default=0)
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--runtime-backend", default="vllm")
    p.add_argument("--ocr-profile", default="markdown_grounded")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--render-dpi", type=int, default=144)
    p.add_argument("--repair-mode", default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return p.parse_args(argv)


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return ""


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _prepare_download_input(df: pd.DataFrame) -> pd.DataFrame:
    required = {"filename", "pdf_url"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Shard parquet missing required column(s): {', '.join(missing)}")
    out = df.copy()
    out["url"] = out["pdf_url"].astype(str)
    out["filename_base"] = out["filename"].astype(str).map(lambda s: Path(s).stem)
    return out


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


def _normalize_download_results(
    *,
    shard_df: pd.DataFrame,
    download_results_df: pd.DataFrame,
    url_column: str = "url",
) -> pd.DataFrame:
    shard = shard_df.copy()
    if "filename_base" not in shard.columns:
        shard["filename_base"] = shard["filename"].astype(str).map(lambda s: Path(s).stem)

    dl = download_results_df.copy()
    if "filename_base" not in dl.columns:
        dl["filename_base"] = dl["filename"].astype(str).map(lambda s: Path(s).stem)

    merged = dl.merge(
        shard,
        on="filename_base",
        how="left",
        suffixes=("", "_shard"),
    )
    if "filename_shard" in merged.columns:
        merged["filename"] = merged["filename_shard"].fillna(merged["filename"])
        merged = merged.drop(columns=["filename_shard"])
    if "pdf_url" in merged.columns and url_column in merged.columns:
        merged[url_column] = merged["pdf_url"].fillna(merged[url_column])
    elif "pdf_url" in merged.columns and url_column not in merged.columns:
        merged[url_column] = merged["pdf_url"]
    if "download_success" not in merged.columns:
        merged["download_success"] = False
    if "download_error" not in merged.columns:
        merged["download_error"] = ""
    if "ocr_success" not in merged.columns:
        merged["ocr_success"] = False
    if "needs_ocr" not in merged.columns:
        merged["needs_ocr"] = True
    return merged


def _write_canonical_metadata(work_root: Path, df: pd.DataFrame) -> Path:
    schema = ParquetSchema({"url_column": "url"})
    canonical = work_root / "download_results" / "download_results.parquet"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    normalized = schema.normalize_metadata_frame(df)
    schema.write_metadata_parquet(normalized, canonical)
    return canonical


def _read_progress(parquet_path: Path, page_col: str = "page_count_source") -> Dict[str, Any]:
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        return {"parquet_error": str(exc)}
    total_docs = int(len(df))
    docs_done = int(df.get("ocr_success", pd.Series(dtype=bool)).fillna(False).sum()) if "ocr_success" in df.columns else 0
    total_pages = 0
    pages_done = 0
    if page_col in df.columns:
        page_values = pd.to_numeric(df[page_col], errors="coerce").fillna(0)
        total_pages = int(page_values.sum())
        if "ocr_success" in df.columns:
            pages_done = int(page_values[df["ocr_success"].fillna(False)].sum())
    return {
        "docs_total": total_docs,
        "docs_done": docs_done,
        "pages_total": total_pages,
        "pages_done": pages_done,
    }


class _HeartbeatThread(threading.Thread):
    def __init__(
        self,
        *,
        heartbeat_path: Path,
        interval: int,
        parquet_path: Path,
        context: Dict[str, Any],
    ) -> None:
        super().__init__(daemon=True)
        self.heartbeat_path = heartbeat_path
        self.interval = max(10, int(interval))
        self.parquet_path = parquet_path
        self.context = dict(context)
        self.stage = "init"
        self.error = ""
        self.stop_event = threading.Event()
        self.started_at = time.time()

    def set_stage(self, stage: str) -> None:
        self.stage = str(stage)

    def set_error(self, error: str) -> None:
        self.error = str(error)

    def stop(self) -> None:
        self.stop_event.set()

    def _payload(self) -> Dict[str, Any]:
        payload = dict(self.context)
        payload.update(
            {
                "timestamp": int(time.time()),
                "hostname": _hostname(),
                "stage": self.stage,
                "error": self.error,
                "uptime_sec": round(time.time() - self.started_at, 1),
                "parquet_path": str(self.parquet_path),
            }
        )
        payload.update(_read_progress(self.parquet_path))
        return payload

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                _atomic_write_json(self.heartbeat_path, self._payload())
            except Exception:
                pass
            self.stop_event.wait(self.interval)
        try:
            _atomic_write_json(self.heartbeat_path, self._payload())
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    shard_path = Path(args.shard_parquet).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = work_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    shard_df = _prepare_download_input(_load_frame(shard_path))
    download_input = manifests_dir / "download_input.parquet"
    shard_df.to_parquet(download_input, index=False)

    metadata_path = work_root / "download_results" / "download_results.parquet"
    if not metadata_path.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _write_canonical_metadata(work_root, shard_df)

    heartbeat: Optional[_HeartbeatThread] = None
    if args.heartbeat_path:
        heartbeat = _HeartbeatThread(
            heartbeat_path=Path(args.heartbeat_path).expanduser().resolve(),
            interval=int(args.heartbeat_interval),
            parquet_path=metadata_path,
            context={
                "instance_id": str(args.instance_id or ""),
                "node_id": str(args.node_id or ""),
                "shard_parquet": str(shard_path),
                "work_root": str(work_root),
            },
        )
        heartbeat.start()

    try:
        if args.dry_run:
            if heartbeat:
                heartbeat.set_stage("dry_run")
            return 0

        corpus = Corpus(
            input_dir=work_root / "downloads",
            output_dir=work_root,
            metadata_path=metadata_path,
            log_level=getattr(logging, str(args.python_log_level).upper(), logging.INFO),
            verbose=False,
        )

        if heartbeat:
            heartbeat.set_stage("download")
        dl_df = corpus.download(
            input_parquet=download_input,
            links_column="url",
            parallelize_by=str(args.download_group_by),
            concurrency=int(args.download_concurrency),
            request_timeout=int(args.download_timeout),
        )
        canonical_df = _normalize_download_results(shard_df=shard_df, download_results_df=dl_df, url_column="url")
        metadata_path = _write_canonical_metadata(work_root, canonical_df)
        if heartbeat:
            heartbeat.parquet_path = metadata_path
            heartbeat.set_stage("ocr")

        corpus = Corpus(
            input_dir=work_root / "downloads",
            output_dir=work_root,
            metadata_path=metadata_path,
            log_level=getattr(logging, str(args.python_log_level).upper(), logging.INFO),
            verbose=False,
        )
        corpus.ocr(
            fix_bad=True,
            mode="ocr_bad",
            backend="deepseek",
            runtime_backend=str(args.runtime_backend),
            ocr_profile=str(args.ocr_profile),
            use_gpus="multi",
            workers_per_gpu=int(args.workers_per_gpu),
            render_dpi=int(args.render_dpi),
            max_new_tokens=int(args.max_new_tokens),
            repair_mode=str(args.repair_mode),
            scheduler=str(args.scheduler),
            target_batch_pages=int(args.target_batch_pages),
            shard_pages=int(args.shard_pages),
            shard_threshold_pages=int(args.shard_threshold_pages),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            math_enhance=False,
        )
        if heartbeat:
            heartbeat.set_stage("done")
        return 0
    except Exception as exc:
        if heartbeat:
            heartbeat.set_stage("failed")
            heartbeat.set_error(str(exc))
        raise
    finally:
        if heartbeat:
            heartbeat.stop()
            heartbeat.join(timeout=5)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
