from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import numpy as np


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
    pages = max(1, int(summary.get("pages_total", 0)))
    pwf = int(summary.get("pages_with_formula", 0))
    frac = pwf / pages if pages else 0.0
    p90 = float(summary.get("formula_p90_pp", 0.0))
    maxp = float(summary.get("formula_max_pp", summary.get("formula_p90_pp", 0.0)))
    total = int(summary.get("formula_total", 0))
    # Heuristics per plan
    if frac >= 0.15 or p90 >= 2 or maxp >= 4 or total >= short_doc_total_min:
        return "2A"
    return "stop"


def update_download_results_parquet(root_dir: Path, filename_stem: str, summary: dict[str, Any], recommendation: str, url_column: str = "url") -> Optional[Path]:
    """Record math summary for a document.

    By default, writes a sidecar JSON under sidecars/triage/{stem}.json to avoid
    concurrent writes to the consolidated parquet. If env GLOSSAPI_PARQUET_COMPACTOR=0,
    falls back to in-place parquet update (legacy behavior).
    """
    root_dir = Path(root_dir)
    use_sidecars = (str(Path.cwd()) is not None)  # dummy always-true construct for mypy
    import os as _os
    use_sidecars = _os.getenv("GLOSSAPI_PARQUET_COMPACTOR", "1").strip() not in {"0", "false", "no"}
    if use_sidecars:
        sc_dir = root_dir / "sidecars" / "triage"
        sc_dir.mkdir(parents=True, exist_ok=True)
        path = sc_dir / f"{filename_stem}.json"
        data = dict(summary)
        data["phase_recommended"] = recommendation
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return None
    # Legacy path: mutate parquet in-place
    candidates = [root_dir / "download_results" / "download_results.parquet"]
    parquet_path = next((p for p in candidates if p.exists()), None)
    if parquet_path is None:
        return None
    df = pd.read_parquet(parquet_path)
    if "filename" not in df.columns:
        return parquet_path
    mask = df["filename"].astype(str).str.replace(r"\.pdf$", "", regex=True) == filename_stem
    if not mask.any():
        return parquet_path
    for k, v in summary.items():
        df.loc[mask, k] = v
    df.loc[mask, "phase_recommended"] = recommendation
    df.to_parquet(parquet_path, index=False)
    return parquet_path


__all__ = [
    "summarize_math_density_from_metrics",
    "recommend_phase",
    "update_download_results_parquet",
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
        return
    # Legacy path
    if not Path(parquet_path).exists():
        return
    df = pd.read_parquet(parquet_path)
    if "filename" not in df.columns:
        return
    mask = df["filename"].astype(str).str.replace(r"\.pdf$", "", regex=True) == stem
    if not mask.any():
        return
    df.loc[mask, "enriched_math"] = True
    df.loc[mask, "math_items"] = int(items)
    df.loc[mask, "math_accept_rate"] = (float(accepted) / float(items)) if items else 0.0
    df.loc[mask, "math_time_sec"] = float(time_sec)
    df.to_parquet(parquet_path, index=False)
