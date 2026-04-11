"""Canonical output reassembly helpers for the DeepSeek runner shim."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SHARD_STEM_RE = re.compile(r"^(?P<source_stem>.+)__p(?P<start>\d{5})-(?P<end>\d{5})$")
REASSEMBLED_CONFIG_KEYS = (
    "ocr_profile",
    "attn_backend",
    "runtime_backend",
    "base_size",
    "image_size",
    "crop_mode",
    "render_dpi",
    "max_new_tokens",
    "batch_size",
    "gpu_memory_utilization",
    "disable_fp8_kv",
    "repair_mode",
)


def parse_shard_stem(stem: str) -> Optional[Dict[str, Any]]:
    match = SHARD_STEM_RE.match(str(stem))
    if match is None:
        return None
    return {
        "source_stem": str(match.group("source_stem")),
        "start_page": int(match.group("start")),
        "end_page": int(match.group("end")),
    }


def split_markdown_pages(
    markdown_text: str,
    *,
    expected_pages: int,
    split_page_outputs_fn: Callable[[str], List[str]],
) -> List[str]:
    pages = split_page_outputs_fn(markdown_text)
    if len(pages) < int(expected_pages):
        pages.extend([""] * (int(expected_pages) - len(pages)))
    elif len(pages) > int(expected_pages):
        pages = pages[: int(expected_pages)]
    return pages


def archive_shard_artifact(*, out_root: Path, source_path: Path, relative_path: Path) -> None:
    archive_path = out_root / "sidecars" / "ocr_shards" / relative_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    source_path.replace(archive_path)


def reassemble_canonical_output_for_source(
    *,
    out_root: Path,
    pdf_path: Path,
    source_name: str,
    page_count_fn: Callable[[Path], int],
    split_page_outputs_fn: Callable[[str], List[str]],
    join_page_outputs_fn: Callable[[List[str]], str],
    write_outputs_fn: Callable[..., Any],
) -> bool:
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    source_stem = Path(source_name).stem
    canonical_md = md_dir / f"{source_stem}.md"
    canonical_metrics = metrics_dir / f"{source_stem}.metrics.json"
    if canonical_md.exists() and canonical_metrics.exists():
        return True

    shard_records: List[Dict[str, Any]] = []
    for metrics_path in sorted(metrics_dir.glob(f"{source_stem}__p*.metrics.json")):
        shard_stem = metrics_path.name.removesuffix(".metrics.json")
        shard_md = md_dir / f"{shard_stem}.md"
        if not shard_md.exists():
            continue
        shard_meta = parse_shard_stem(shard_stem)
        if shard_meta is None:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        start_page = int(metrics.get("source_start_page", shard_meta["start_page"]))
        end_page = int(metrics.get("source_end_page", shard_meta["end_page"]))
        shard_records.append(
            {
                "stem": shard_stem,
                "md_path": shard_md,
                "metrics_path": metrics_path,
                "metrics": metrics,
                "start_page": start_page,
                "end_page": end_page,
            }
        )

    if not shard_records:
        return False

    shard_records.sort(key=lambda item: (int(item["start_page"]), int(item["end_page"]), str(item["stem"])))
    page_count = max(int(page_count_fn(pdf_path)), max(int(item["end_page"]) for item in shard_records))
    merged_pages = [""] * int(page_count)
    merged_page_metrics: List[Optional[Dict[str, Any]]] = [None] * int(page_count)
    merged_extra_metrics: Dict[str, Any] = {}
    repair_totals: Dict[str, int] = {}
    render_sec_total = 0.0
    infer_sec_total = 0.0
    wall_time_sec_total = 0.0
    reassembled_ranges: List[Dict[str, int]] = []

    for shard in shard_records:
        metrics = dict(shard["metrics"])
        start_page = int(shard["start_page"])
        end_page = int(shard["end_page"])
        expected_pages = max(0, end_page - start_page + 1)
        reassembled_ranges.append({"start_page": start_page, "end_page": end_page})

        shard_pages = split_markdown_pages(
            shard["md_path"].read_text(encoding="utf-8"),
            expected_pages=expected_pages,
            split_page_outputs_fn=split_page_outputs_fn,
        )
        for offset, page_text in enumerate(shard_pages):
            merged_pages[start_page - 1 + offset] = page_text

        for idx, page_metric in enumerate(list(metrics.get("page_metrics") or []), start=1):
            absolute_page = start_page + int(page_metric.get("page_number", idx)) - 1
            if absolute_page <= 0 or absolute_page > int(page_count):
                continue
            merged_metric = dict(page_metric)
            merged_metric["page_number"] = int(absolute_page)
            merged_page_metrics[absolute_page - 1] = merged_metric

        render_sec_total += float(metrics.get("render_sec", 0.0))
        infer_sec_total += float(metrics.get("infer_sec_total", 0.0))
        wall_time_sec_total += float(metrics.get("wall_time_sec", 0.0))
        for key, value in dict(metrics.get("repair_summary") or {}).items():
            if key == "repair_mode":
                continue
            repair_totals[key] = int(repair_totals.get(key, 0)) + int(value)
        for key in REASSEMBLED_CONFIG_KEYS:
            if key in metrics and key not in merged_extra_metrics:
                merged_extra_metrics[key] = metrics[key]

    merged_extra_metrics.update(
        {
            "source_file": str(source_name),
            "source_stem": str(source_stem),
            "source_start_page": 1,
            "source_end_page": int(page_count),
            "reassembled_from_shards": True,
            "reassembled_shard_count": len(shard_records),
            "reassembled_source_ranges": reassembled_ranges,
            "render_sec": float(render_sec_total),
            "infer_sec_total": float(infer_sec_total),
            "wall_time_sec": float(wall_time_sec_total),
            "wall_time_sec_semantics": "sum_of_shard_wall_times",
            "page_metrics": [item for item in merged_page_metrics if item is not None],
        }
    )
    if repair_totals:
        merged_extra_metrics["repair_summary"] = {
            "repair_mode": str(merged_extra_metrics.get("repair_mode", "unknown")),
            **{key: int(value) for key, value in repair_totals.items()},
        }

    merged_markdown = join_page_outputs_fn(merged_pages) if merged_pages else "[[Blank page]]"
    write_outputs_fn(
        output_dir=out_root,
        stem=source_stem,
        markdown=merged_markdown,
        page_count=int(page_count),
        extra_metrics=merged_extra_metrics,
    )
    for shard in shard_records:
        archive_shard_artifact(
            out_root=out_root,
            source_path=Path(shard["md_path"]),
            relative_path=Path("markdown") / Path(shard["md_path"]).name,
        )
        archive_shard_artifact(
            out_root=out_root,
            source_path=Path(shard["metrics_path"]),
            relative_path=Path("json") / "metrics" / Path(shard["metrics_path"]).name,
        )
    return True


def ensure_canonical_outputs(
    *,
    out_root: Path,
    pdf_root: Path,
    file_list: List[str],
    page_count_fn: Callable[[Path], int],
    split_page_outputs_fn: Callable[[str], List[str]],
    join_page_outputs_fn: Callable[[List[str]], str],
    write_outputs_fn: Callable[..., Any],
) -> None:
    for name in file_list:
        pdf_path = (pdf_root / name).resolve()
        if reassemble_canonical_output_for_source(
            out_root=out_root,
            pdf_path=pdf_path,
            source_name=name,
            page_count_fn=page_count_fn,
            split_page_outputs_fn=split_page_outputs_fn,
            join_page_outputs_fn=join_page_outputs_fn,
            write_outputs_fn=write_outputs_fn,
        ):
            continue
