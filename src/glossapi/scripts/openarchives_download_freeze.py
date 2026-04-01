from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from glossapi import Corpus
from glossapi.scripts.openarchives_ocr_run_node import (
    DEFAULT_DOWNLOAD_CONCURRENCY,
    DEFAULT_DOWNLOAD_TIMEOUT,
    _load_frame,
    _normalize_download_results,
    _prepare_download_input,
    _write_canonical_metadata,
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_download_freeze",
        description=(
            "Materialize one OpenArchives manifest into a canonical GlossAPI downloads root "
            "without starting OCR. This is the reproducible PDF-freeze entrypoint."
        ),
    )
    p.add_argument("--input-parquet", required=True)
    p.add_argument("--work-root", required=True)
    p.add_argument("--python-log-level", default="INFO")
    p.add_argument("--download-concurrency", type=int, default=DEFAULT_DOWNLOAD_CONCURRENCY)
    p.add_argument("--download-timeout", type=int, default=DEFAULT_DOWNLOAD_TIMEOUT)
    p.add_argument("--download-mode", default="auto")
    p.add_argument("--download-scheduler-mode", default="per_domain")
    p.add_argument("--download-group-by", default="base_domain")
    p.add_argument("--download-policy-file", default="")
    p.add_argument("--supported-formats", default="pdf")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input_parquet).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = work_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = _prepare_download_input(_load_frame(input_path))
    download_input = manifests_dir / "download_input.parquet"
    manifest_df.to_parquet(download_input, index=False)

    metadata_path = work_root / "download_results" / "download_results.parquet"
    if not metadata_path.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _write_canonical_metadata(work_root, manifest_df)

    if args.dry_run:
        return 0

    corpus = Corpus(
        input_dir=work_root / "downloads",
        output_dir=work_root,
        metadata_path=metadata_path,
        log_level=getattr(logging, str(args.python_log_level).upper(), logging.INFO),
        verbose=False,
    )
    dl_df = corpus.download(
        input_parquet=download_input,
        links_column="url",
        download_mode=str(args.download_mode),
        parallelize_by=str(args.download_group_by),
        concurrency=int(args.download_concurrency),
        request_timeout=int(args.download_timeout),
        scheduler_mode=str(args.download_scheduler_mode),
        supported_formats=[part.strip() for part in str(args.supported_formats).split(",") if part.strip()],
        download_policy_file=(str(args.download_policy_file) if str(args.download_policy_file or "").strip() else None),
    )
    canonical_df = _normalize_download_results(shard_df=manifest_df, download_results_df=dl_df, url_column="url")
    _write_canonical_metadata(work_root, canonical_df)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
