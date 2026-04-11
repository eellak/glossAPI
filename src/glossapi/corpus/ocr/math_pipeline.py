"""Phase-2 math enrichment helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .context import CorpusOcrContext
from .math_targets import ensure_math_placeholder_sidecars, filter_stems_by_parquet_math_signals


def formula_enrich_from_json(
    context: CorpusOcrContext,
    files: Optional[List[str]] = None,
    *,
    device: str = "cuda",
    batch_size: int = 8,
    dpi_base: int = 220,
    targets_by_stem: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> None:
    """Phase-2: enrich math/code from Docling JSON without re-running layout."""

    from ...ocr import math as _math_pkg  # type: ignore

    try:
        enrich_from_docling_json = getattr(_math_pkg, "enrich_from_docling_json")
    except AttributeError as exc:
        raise RuntimeError("Math enrichment backend unavailable") from exc
    if not callable(enrich_from_docling_json):
        raise RuntimeError("Math enrichment backend missing 'enrich_from_docling_json'")

    json_dir = context.output_dir / "json"
    dl_dir = context.output_dir / "downloads"
    stems: List[str] = []
    if files:
        stems = list(files)
    else:
        candidates: List[Path] = []
        if json_dir.exists():
            candidates += list(json_dir.glob("*.docling.json"))
            candidates += list(json_dir.glob("*.docling.json.zst"))
        stems = [path.name.replace(".docling.json.zst", "").replace(".docling.json", "") for path in candidates]
    if not stems:
        context.logger.info("No Docling JSON files found for Phase-2 enrichment")
        return

    context.logger.info("Phase-2: enriching %d document(s) from JSON", len(stems))
    context.logger.info("Phase-2: placeholder sidecars for stems: %s", ",".join(stems))
    ensure_math_placeholder_sidecars(context, stems=stems)
    stems = filter_stems_by_parquet_math_signals(context, stems=stems)
    ensure_math_placeholder_sidecars(context, stems=stems)

    for stem in stems:
        try:
            context.logger.info("Phase-2: processing stem=%s", stem)
        except Exception:
            pass
        try:
            if (json_dir / f"{stem}.docling.json.zst").exists():
                json_path = json_dir / f"{stem}.docling.json.zst"
            elif (json_dir / f"{stem}.docling.json").exists():
                json_path = json_dir / f"{stem}.docling.json"
            else:
                context.logger.warning("JSON not found for %s", stem)
                continue

            pdf_path = None
            if (dl_dir / f"{stem}.pdf").exists():
                pdf_path = dl_dir / f"{stem}.pdf"
            else:
                try:
                    from ...ocr.utils.json_io import load_docling_json  # type: ignore

                    doc = load_docling_json(json_path)
                    meta = getattr(doc, "meta", {}) or {}
                    relpath = meta.get("source_pdf_relpath") or ""
                    if relpath:
                        candidate = Path(relpath)
                        if not candidate.is_absolute():
                            candidate = context.output_dir / relpath
                        if candidate.exists():
                            pdf_path = candidate
                except Exception:
                    pass
            if pdf_path is None:
                context.logger.warning("PDF not found for %s; skipping", stem)
                continue

            out_md = context.markdown_dir / f"{stem}.md"
            out_map = json_dir / f"{stem}.latex_map.jsonl"
            out_md.parent.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)

            picks = None
            try:
                if targets_by_stem and stem in targets_by_stem:
                    picks = [(int(page), int(index)) for (page, index) in targets_by_stem.get(stem, [])]
            except Exception:
                picks = None

            try:
                from ...ocr.utils.triage import update_math_enrich_results  # type: ignore

                pq_path = context._get_cached_metadata_parquet()
                if pq_path is None:
                    from ...parquet_schema import ParquetSchema as _ParquetSchema

                    pq_schema = _ParquetSchema({"url_column": context.url_column})
                    pq_path = context._resolve_metadata_parquet(pq_schema, ensure=True, search_input=True)
                if pq_path is None:
                    pq_path = context.output_dir / "download_results" / "download_results.parquet"
                context._cache_metadata_parquet(pq_path)
                update_math_enrich_results(pq_path, stem, items=0, accepted=0, time_sec=0.0)
            except Exception:
                pass

            stats = enrich_from_docling_json(
                json_path,
                pdf_path,
                out_md,
                out_map,
                device=device,
                batch_size=int(batch_size),
                dpi_base=int(dpi_base),
                targets=picks,
            )
            context.logger.info(
                "Phase-2: %s -> items=%s accepted=%s time=%.2fs",
                stem,
                stats.get("items"),
                stats.get("accepted"),
                stats.get("time_sec"),
            )
            try:
                from ...ocr.utils.triage import update_math_enrich_results  # type: ignore

                pq_path = context._get_cached_metadata_parquet()
                if pq_path is None:
                    from ...parquet_schema import ParquetSchema as _ParquetSchema

                    pq_schema = _ParquetSchema({"url_column": context.url_column})
                    pq_path = context._resolve_metadata_parquet(pq_schema, ensure=True, search_input=True)
                if pq_path is None:
                    pq_path = context.output_dir / "download_results" / "download_results.parquet"
                context._cache_metadata_parquet(pq_path)
                update_math_enrich_results(
                    pq_path,
                    stem,
                    items=int(stats.get("items", 0)),
                    accepted=int(stats.get("accepted", 0)),
                    time_sec=float(stats.get("time_sec", 0.0)),
                )
            except Exception as exc:
                context.logger.warning("Parquet math-enrich update failed for %s: %s", stem, exc)
        except Exception as exc:
            context.logger.warning("Phase-2 failed for %s: %s", stem, exc)
            try:
                from ...ocr.utils.triage import update_math_enrich_results  # type: ignore

                pq_path = context._get_cached_metadata_parquet()
                if pq_path is None:
                    from ...parquet_schema import ParquetSchema as _ParquetSchema

                    pq_schema = _ParquetSchema({"url_column": context.url_column})
                    pq_path = context._resolve_metadata_parquet(pq_schema, ensure=True, search_input=True)
                if pq_path is None:
                    pq_path = context.output_dir / "download_results" / "download_results.parquet"
                context._cache_metadata_parquet(pq_path)
                update_math_enrich_results(pq_path, stem, items=0, accepted=0, time_sec=0.0)
            except Exception:
                pass


def triage_math(context: CorpusOcrContext) -> None:
    """Summarize per-page formula density and update routing recommendation in parquet."""

    try:
        from ...ocr.utils.triage import (
            summarize_math_density_from_metrics,
            recommend_phase,
            update_download_results_parquet,
        )
    except Exception as exc:
        context.logger.warning("Triage utilities unavailable: %s", exc)
        return

    markdown_dir = Path(context.markdown_dir)
    if not markdown_dir.exists():
        context.logger.warning("markdown_dir %s not found for triage", markdown_dir)
        return

    metrics_files_set = set()
    json_metrics = context.output_dir / "json" / "metrics"
    if json_metrics.exists():
        metrics_files_set |= set(json_metrics.glob("*.per_page.metrics.json"))
    metrics_files_set |= set(markdown_dir.rglob("*.per_page.metrics.json"))
    metrics_files = sorted(metrics_files_set)
    if not metrics_files:
        context.logger.info("No per-page metrics JSON found under %s", markdown_dir)
        return

    for metrics_path in metrics_files:
        stem = metrics_path.name.replace(".per_page.metrics.json", "")
        try:
            summary = summarize_math_density_from_metrics(metrics_path)
            summary["formula_max_pp"] = float(summary.get("formula_p90_pp", 0.0))
            recommendation = recommend_phase(summary)
            update_download_results_parquet(
                context.output_dir,
                stem,
                summary,
                recommendation,
                url_column=context.url_column,
            )
            context.logger.info("Triage: %s -> %s", stem, recommendation)
        except Exception as exc:
            context.logger.warning("Triage failed for %s: %s", stem, exc)
