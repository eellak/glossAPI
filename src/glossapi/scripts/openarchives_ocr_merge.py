from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

import pandas as pd


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


def _copy_artifacts(
    *,
    shard_rows: pd.DataFrame,
    work_roots: List[Path],
    output_root: Path,
) -> int:
    copied = 0
    markdown_out = output_root / "markdown"
    metrics_out = output_root / "json" / "metrics"
    markdown_out.mkdir(parents=True, exist_ok=True)
    metrics_out.mkdir(parents=True, exist_ok=True)
    for row in shard_rows.to_dict(orient="records"):
        stem = str(row.get("filename_base") or Path(str(row.get("filename") or "")).stem).strip()
        if not stem:
            continue
        md_name = str(row.get("md_filename") or f"{stem}.md")
        for root in work_roots:
            md_src = root / "markdown" / f"{stem}.md"
            if md_src.exists():
                shutil.copy2(md_src, markdown_out / md_name)
                copied += 1
                break
        for suffix in (".metrics.json", ".per_page.metrics.json"):
            for root in work_roots:
                src = root / "json" / "metrics" / f"{stem}{suffix}"
                if src.exists():
                    shutil.copy2(src, metrics_out / src.name)
                    copied += 1
                    break
    return copied


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

    master = master.reset_index(drop=True).drop(columns=["_merge_key"], errors="ignore")
    master.to_parquet(out_path, index=False)
    copied = 0
    if args.artifact_work_roots and str(args.artifact_output_root or "").strip():
        roots = [Path(p).expanduser().resolve() for p in args.artifact_work_roots]
        copied = _copy_artifacts(
            shard_rows=shards.reset_index(drop=True),
            work_roots=roots,
            output_root=Path(args.artifact_output_root).expanduser().resolve(),
        )
    print(f"Merged {len(shards)} shard row(s) into {master_path} -> {out_path}; copied {copied} artifact file(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
