from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import pandas as pd

from glossapi import Corpus


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_download_probe",
        description=(
            "Sample OpenArchives OCR-target PDFs by host, run a controlled download probe, "
            "and write per-host success summaries."
        ),
    )
    p.add_argument("--parquet", required=True, help="needs_ocr_enriched parquet with pdf_url and filename columns")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--policy-file", default="")
    p.add_argument("--samples-per-host", type=int, default=12)
    p.add_argument("--max-hosts", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--request-timeout", type=int, default=60)
    p.add_argument("--download-group-by", default="base_domain")
    p.add_argument("--hosts", nargs="*", default=None, help="Optional explicit host allowlist")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _host_from_url(url: str) -> str:
    try:
        return (urlparse(str(url)).hostname or "").lower()
    except Exception:
        return ""


def _prepare_probe_frame(
    df: pd.DataFrame,
    *,
    samples_per_host: int,
    max_hosts: int,
    seed: int,
    hosts: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    frame = df.copy()
    if "pdf_url" not in frame.columns or "filename" not in frame.columns:
        raise SystemExit("Probe parquet must include at least 'pdf_url' and 'filename' columns")
    frame["host"] = frame["pdf_url"].astype(str).map(_host_from_url)
    frame = frame[frame["host"].astype(bool)].copy()
    if hosts:
        allowed = {str(h).strip().lower() for h in hosts if str(h).strip()}
        frame = frame[frame["host"].isin(allowed)].copy()
    ranked_hosts = (
        frame.groupby("host", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(max(1, int(max_hosts)))
        .index.tolist()
    )
    probe = frame[frame["host"].isin(ranked_hosts)].copy()
    sampled = (
        probe.groupby("host", group_keys=True)
        .apply(
            lambda grp: grp.sample(n=min(len(grp), int(samples_per_host)), random_state=int(seed)),
            include_groups=False,
        )
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    sampled["url"] = sampled["pdf_url"].astype(str)
    sampled["base_domain"] = sampled["pdf_url"].astype(str).map(
        lambda s: f"{urlparse(str(s)).scheme or 'https'}://{(urlparse(str(s)).netloc or '').lower()}".rstrip("/")
        if _host_from_url(str(s))
        else ""
    )
    return sampled


def _summary_payload(df: pd.DataFrame, *, source_rows: int) -> dict:
    out = df.copy()
    if "download_success" not in out.columns:
        out["download_success"] = False
    grouped = (
        out.groupby("host", dropna=False)
        .agg(
            docs=("host", "size"),
            successes=("download_success", lambda s: int(pd.Series(s).fillna(False).sum())),
            failures=("download_success", lambda s: int((~pd.Series(s).fillna(False)).sum())),
        )
        .reset_index()
        .sort_values(["docs", "successes"], ascending=[False, False])
    )
    return {
        "source_rows": int(source_rows),
        "probe_rows": int(len(out)),
        "hosts": grouped.to_dict(orient="records"),
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    parquet_path = Path(args.parquet).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_parquet(parquet_path)
    probe_df = _prepare_probe_frame(
        source_df,
        samples_per_host=int(args.samples_per_host),
        max_hosts=int(args.max_hosts),
        seed=int(args.seed),
        hosts=args.hosts,
    )
    probe_input = output_dir / "probe_input.parquet"
    probe_df.to_parquet(probe_input, index=False)

    if args.dry_run:
        summary = _summary_payload(probe_df, source_rows=len(source_df))
        (output_dir / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    corpus = Corpus(
        input_dir=output_dir / "downloads",
        output_dir=output_dir,
        log_level="INFO",
        verbose=False,
    )
    results = corpus.download(
        input_parquet=probe_input,
        links_column="url",
        parallelize_by=str(args.download_group_by),
        concurrency=int(args.concurrency),
        request_timeout=int(args.request_timeout),
        download_policy_file=(str(args.policy_file) if str(args.policy_file or "").strip() else None),
    )
    merged = results.merge(
        probe_df[["url", "host", "filename"]],
        on="url",
        how="left",
        suffixes=("", "_probe"),
    )
    merged_path = output_dir / "probe_results.parquet"
    merged.to_parquet(merged_path, index=False)
    summary = _summary_payload(merged, source_rows=len(source_df))
    (output_dir / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
