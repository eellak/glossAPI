from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from glossapi._naming import canonical_stem
from glossapi.parquet_schema import ParquetSchema


def summarize_math_density_from_metrics(per_page_path: Path) -> dict[str, Any]:
    data = json.loads(Path(per_page_path).read_text(encoding="utf-8"))
    pages = data.get("pages", [])
    if not pages:
        return {"formula_total": 0, "formula_avg_pp": 0.0, "formula_p90_pp": 0.0, "pages_with_formula": 0, "pages_total": 0}
    counts = [int(p.get("formula_count", 0)) for p in pages]
    total = int(sum(counts))
    avg = float(np.mean(counts)) if counts else 0.0
    p90 = float(np.quantile(counts, 0.90)) if counts else 0.0
    pwf = int(sum(1 for c in counts if c > 0))
    return {
        "formula_total": total,
        "formula_avg_pp": avg,
        "formula_p90_pp": p90,
        "pages_with_formula": pwf,
        "pages_total": len(counts),
    }


def recommend_phase(summary: dict[str, Any], *, short_doc_total_min: int = 10) -> str:
    """Return '2A' to run Phase‑2 or 'stop' to skip.

    Default policy: run Phase‑2 on any document with at least one formula
    (i.e., only skip true no‑math docs). To enable the older heuristic
    thresholds, set env GLOSSAPI_TRIAGE_HEURISTIC=1.
    """
    total = int(summary.get("formula_total", 0))
    if os.getenv("GLOSSAPI_TRIAGE_HEURISTIC", "0").strip().lower() not in {"1", "true", "yes"}:
        return "2A" if total > 0 else "stop"
    # Legacy heuristic mode
    pages = max(1, int(summary.get("pages_total", 0)))
    pwf = int(summary.get("pages_with_formula", 0))
    frac = pwf / pages if pages else 0.0
    p90 = float(summary.get("formula_p90_pp", 0.0))
    maxp = float(summary.get("formula_max_pp", summary.get("formula_p90_pp", 0.0)))
    if frac >= 0.15 or p90 >= 2 or maxp >= 4 or total >= short_doc_total_min:
        return "2A"
    return "stop"


def update_download_results_parquet(root_dir: Path, filename_stem: str, summary: dict[str, Any], recommendation: str, url_column: str = "url") -> Optional[Path]:
    """Record math summary for a document into sidecar and parquet.

    Always writes sidecar JSON under sidecars/triage/{stem}.json. Additionally,
    updates or creates download_results/download_results.parquet and ensures a row
    exists for {stem}. Updates the following fields at minimum:
      - formula_total (int)
      - pages_total, pages_with_formula, formula_avg_pp, formula_p90_pp (if present)
      - phase_recommended ("2A"|"stop")
    """
    root_dir = Path(root_dir)
    # 1) Sidecar write (best-effort, never raises)
    try:
        sc_dir = root_dir / "sidecars" / "triage"
        sc_dir.mkdir(parents=True, exist_ok=True)
        path = sc_dir / f"{filename_stem}.json"
        data = dict(summary)
        data["phase_recommended"] = recommendation
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    # 2) Parquet update (create if missing)
    try:
        dl_dir = root_dir / "download_results"
        dl_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = dl_dir / "download_results.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.DataFrame()
        # Ensure filename column and find/create row
        schema = ParquetSchema()
        if "filename" not in df.columns:
            df["filename"] = pd.Series(dtype=str)
        target_stem = canonical_stem(filename_stem)
        stem_series = df["filename"].astype(str).map(canonical_stem)
        mask = stem_series == target_stem
        if not mask.any():
            # append new row
            df.loc[len(df)] = {"filename": f"{target_stem}.pdf"}
            stem_series = df["filename"].astype(str).map(canonical_stem)
            mask = stem_series == target_stem
        # Rounded numeric fields for readability/consistency
        ints = ["formula_total", "pages_total", "pages_with_formula"]
        floats = ["formula_avg_pp", "formula_p90_pp"]
        for col in ints:
            if col in summary and summary.get(col) is not None:
                try:
                    df.loc[mask, col] = int(summary.get(col))
                except Exception:
                    df.loc[mask, col] = summary.get(col)
        for col in floats:
            if col in summary and summary.get(col) is not None:
                try:
                    df.loc[mask, col] = round(float(summary.get(col)), 3)
                except Exception:
                    df.loc[mask, col] = summary.get(col)
        df.loc[mask, "phase_recommended"] = recommendation
        schema.write_metadata_parquet(df, parquet_path)
        return parquet_path
    except Exception:
        return None


__all__ = [
    "summarize_math_density_from_metrics",
    "recommend_phase",
    "update_download_results_parquet",
    "update_math_enrich_results",
]

def update_math_enrich_results(parquet_path: Path, stem: str, *, items: int, accepted: int, time_sec: float) -> None:
    """Record math enrichment results for a document.

    Default: write sidecar under sidecars/math/{stem}.json. If GLOSSAPI_PARQUET_COMPACTOR=0,
    update consolidated parquet in place (legacy behavior).
    """
    import os as _os
    use_sidecars = _os.getenv("GLOSSAPI_PARQUET_COMPACTOR", "1").strip() not in {"0", "false", "no"}
    root = Path(parquet_path).parent.parent if parquet_path else Path.cwd()
    if use_sidecars:
        sc_dir = root / "sidecars" / "math"
        sc_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "items": int(items),
            "accepted": int(accepted),
            "time_sec": float(time_sec),
        }
        (sc_dir / f"{stem}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    # Always try to update consolidated parquet as well
    if not Path(parquet_path).exists():
        return
    df = pd.read_parquet(parquet_path)
    if "filename" not in df.columns:
        return
    schema = ParquetSchema()
    target_stem = canonical_stem(stem)
    stem_series = df["filename"].astype(str).map(canonical_stem)
    mask = stem_series == target_stem
    if not mask.any():
        return
    df.loc[mask, "enriched_math"] = True
    df.loc[mask, "math_items"] = int(items)
    df.loc[mask, "math_accept_rate"] = (float(accepted) / float(items)) if items else 0.0
    df.loc[mask, "math_time_sec"] = float(time_sec)
    if "math_enriched" not in df.columns:
        df["math_enriched"] = df.get("enriched_math", False)
    if mask.any():
        df.loc[mask, "math_enriched"] = True
    schema.write_metadata_parquet(df, parquet_path)
