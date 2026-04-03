from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


_MARKDOWN_SHARD_RE = re.compile(r"^(?P<stem>.+)__p(?P<start>\d+)-(?P<end>\d+)\.md$")


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_ocr_merge",
        description="Merge shard-level OCR metadata back into a canonical GlossAPI download_results parquet.",
    )
    p.add_argument("--master-parquet", required=True)
    p.add_argument("--shard-parquets", nargs="+", required=True)
    p.add_argument("--output-parquet", required=True)
    p.add_argument("--key-column", default="filename")
    p.add_argument("--preserve-master-columns", default="")
    p.add_argument("--artifact-work-roots", nargs="*", default=[])
    p.add_argument("--artifact-output-root", default="")
    return p.parse_args(argv)


def _normalize_key(df: pd.DataFrame, key: str) -> pd.Series:
    if key not in df.columns:
        raise SystemExit(f"Key column '{key}' not present in dataframe.")
    return df[key].astype(str).str.strip()


def _merge_markdown_parts(parts: List[str]) -> str:
    merged: List[str] = []
    for part in parts:
        if not part:
            continue
        if merged and not merged[-1].endswith("\n"):
            merged[-1] = merged[-1] + "\n"
        merged.append(part)
    return "".join(merged)


def _copy_once(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def _resolve_markdown_payload(
    *,
    stem: str,
    md_name: str,
    work_roots: List[Path],
    output_root: Optional[Path],
) -> tuple[Optional[str], Optional[str]]:
    markdown_out = output_root / "markdown" if output_root is not None else None
    shard_out = output_root / "sidecars" / "ocr_shards" / "markdown" if output_root is not None else None

    for root in work_roots:
        canonical_src = root / "markdown" / f"{stem}.md"
        if canonical_src.exists():
            payload = canonical_src.read_text(encoding="utf-8")
            if markdown_out is not None:
                _copy_once(canonical_src, markdown_out / md_name)
                return payload, str(Path("markdown") / md_name)
            return payload, None

        shard_sources = []
        for candidate in sorted((root / "markdown").glob(f"{stem}__p*.md")):
            match = _MARKDOWN_SHARD_RE.match(candidate.name)
            if not match or match.group("stem") != stem:
                continue
            shard_sources.append((int(match.group("start")), candidate))
        if not shard_sources:
            continue

        shard_sources.sort(key=lambda item: item[0])
        payload = _merge_markdown_parts([path.read_text(encoding="utf-8") for _, path in shard_sources])
        if markdown_out is not None:
            destination = markdown_out / md_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
            if shard_out is not None:
                for _, shard_path in shard_sources:
                    _copy_once(shard_path, shard_out / shard_path.name)
            return payload, str(Path("markdown") / md_name)
        return payload, None
    return None, None


def _collect_artifact_updates(
    *,
    shard_rows: pd.DataFrame,
    work_roots: List[Path],
    output_root: Optional[Path],
) -> tuple[int, pd.DataFrame]:
    copied = 0
    markdown_out = output_root / "markdown" if output_root is not None else None
    metrics_out = output_root / "json" / "metrics" if output_root is not None else None
    if metrics_out is not None:
        metrics_out.mkdir(parents=True, exist_ok=True)
    updates: List[Dict[str, object]] = []
    for row in shard_rows.to_dict(orient="records"):
        merge_key = str(row.get("_merge_key") or "").strip()
        stem = str(row.get("filename_base") or Path(str(row.get("filename") or "")).stem).strip()
        if not stem:
            continue
        md_name = str(row.get("md_filename") or f"{stem}.md")
        md_payload, md_relpath = _resolve_markdown_payload(
            stem=stem,
            md_name=md_name,
            work_roots=work_roots,
            output_root=output_root,
        )
        if md_payload is not None and markdown_out is not None:
            copied += 1
        metrics_relpath = None
        for suffix in (".metrics.json", ".per_page.metrics.json"):
            for root in work_roots:
                src = root / "json" / "metrics" / f"{stem}{suffix}"
                if src.exists():
                    if metrics_out is not None:
                        _copy_once(src, metrics_out / src.name)
                        copied += 1
                        metrics_relpath = str(Path("json") / "metrics" / src.name)
                    break
            if metrics_relpath is not None:
                break
        updates.append(
            {
                "_merge_key": merge_key,
                "text": md_payload,
                "ocr_markdown_relpath": md_relpath,
                "ocr_metrics_relpath": metrics_relpath,
                "ocr_text_sha256": (
                    hashlib.sha256(md_payload.encode("utf-8")).hexdigest()
                    if isinstance(md_payload, str)
                    else None
                ),
            }
        )
    return copied, pd.DataFrame(updates)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    master_path = Path(args.master_parquet).expanduser().resolve()
    out_path = Path(args.output_parquet).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preserve_master_columns = [c.strip() for c in str(args.preserve_master_columns or "").split(",") if c.strip()]
    master = pd.read_parquet(master_path).copy()
    master["_merge_key"] = _normalize_key(master, str(args.key_column))

    shard_frames: List[pd.DataFrame] = []
    for shard in args.shard_parquets:
        shard_df = pd.read_parquet(Path(shard).expanduser().resolve()).copy()
        shard_df["_merge_key"] = _normalize_key(shard_df, str(args.key_column))
        shard_frames.append(shard_df)
    shards = pd.concat(shard_frames, ignore_index=True)
    shards = shards.drop_duplicates(subset=["_merge_key"], keep="last")

    master = master.set_index("_merge_key", drop=False)
    shards = shards.set_index("_merge_key", drop=False)

    for column in shards.columns:
        if column == "_merge_key":
            continue
        if column in preserve_master_columns:
            continue
        master.loc[shards.index, column] = shards[column]

    copied = 0
    if args.artifact_work_roots:
        roots = [Path(p).expanduser().resolve() for p in args.artifact_work_roots]
        artifact_output_root = (
            Path(args.artifact_output_root).expanduser().resolve()
            if str(args.artifact_output_root or "").strip()
            else None
        )
        copied, artifact_updates = _collect_artifact_updates(
            shard_rows=shards.reset_index(drop=True),
            work_roots=roots,
            output_root=artifact_output_root,
        )
        if not artifact_updates.empty:
            artifact_updates = artifact_updates.drop_duplicates(subset=["_merge_key"], keep="last").set_index("_merge_key")
            for column in artifact_updates.columns:
                if column in preserve_master_columns:
                    continue
                if column not in master.columns:
                    master[column] = None
                mask = artifact_updates[column].notna()
                if bool(mask.any()):
                    master.loc[artifact_updates.index[mask], column] = artifact_updates.loc[mask, column]
    master = master.reset_index(drop=True).drop(columns=["_merge_key"], errors="ignore")
    master.to_parquet(out_path, index=False)
    print(f"Merged {len(shards)} shard row(s) into {master_path} -> {out_path}; copied {copied} artifact file(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
