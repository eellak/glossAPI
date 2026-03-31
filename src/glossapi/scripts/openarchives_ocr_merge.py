from __future__ import annotations

import argparse
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
    return p.parse_args(argv)


def _normalize_key(df: pd.DataFrame, key: str) -> pd.Series:
    if key not in df.columns:
        raise SystemExit(f"Key column '{key}' not present in dataframe.")
    return df[key].astype(str).str.strip()


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    master_path = Path(args.master_parquet).expanduser().resolve()
    out_path = Path(args.output_parquet).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
        master.loc[shards.index, column] = shards[column]

    master = master.reset_index(drop=True).drop(columns=["_merge_key"], errors="ignore")
    master.to_parquet(out_path, index=False)
    print(
        f"Merged {len(shards)} shard row(s) into {master_path} -> {out_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
