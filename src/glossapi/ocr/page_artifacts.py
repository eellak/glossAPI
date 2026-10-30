"""Backend-neutral OCR page artifacts and lossless partial-page reassembly.

Moved from the retired backend so page routing and retained native Markdown
remain available to external Docling OCR engines.
"""
from __future__ import annotations
import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)
SHARD_STEM_RE = re.compile(r"^(?P<source_stem>.+)__p(?P<start>\d{5})-(?P<end>\d{5})$")
PAGE_SPLIT_RE = re.compile(r"(?:^|\n)(?:<!-- page:\d+ -->\n)?<--- Page Split --->\n?")
REASSEMBLED_CONFIG_KEYS = ("backend", "ocr_engine", "device")
# Lines that compose an OCR supplement. They are structure, not document text.
SUPPLEMENT_MARKER_RE = re.compile(
    r"(?m)^<!-- (?:ocr-pages: [^\n]*?|/ocr-pages|ocr-supplement-(?:begin|end): [^\n]*?) -->[ \t]*(?:\n|$)")
# Option fields that are file locations, not OCR configuration: this call's working
# files, model/codec/seed locations (model identity is model_revision plus
# model_manifest_sha256 / checkpoint_sha256 / config_sha256) and the diagnostic
# outputs block, which selects debug artifacts and never changes the OCR text.
# Paths name the corpus location; they must not enter the assembly identity.
NON_IDENTITY_OPTION_KEYS = ("routing_sidecar", "call_log", "checkpoint", "codec", "release_dir",
                            "ornament_seeds", "hole_seeds", "outputs")
ASSEMBLY_ID_SCHEMA = "glossapi.ocr_supplement_assembly.v1"


def strip_supplement_markers(markdown: str) -> str:
    """Remove the OCR supplement's composition lines; text metrics must not count them."""
    return SUPPLEMENT_MARKER_RE.sub("", str(markdown or ""))


def _canonical_option(value: Any) -> Any:
    """Process-independent form of an option value (FG-G02 node finding).

    Lists are sorted by their canonical JSON: plugins may build a list from a
    set, whose order follows the per-process hash seed. Absolute paths are
    dropped wherever they occur. Order-sensitive meaning is not lost from the
    identity, because the assembly id also hashes every produced page's text.
    """
    if isinstance(value, dict):
        return {str(key): _canonical_option(item) for key, item in value.items()
                if not _is_absolute_path(item)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical_option(item) for item in value if not _is_absolute_path(item)]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return value


def _is_absolute_path(value: Any) -> bool:
    if isinstance(value, Path):
        return value.is_absolute()
    return isinstance(value, str) and (value.startswith("/") or bool(re.match(r"^[A-Za-z]:[\\/]", value))
                                       or value.startswith("file:"))


def ocr_configuration(options: Optional[Dict[str, Any]], device: Optional[str]) -> Dict[str, Any]:
    """The OCR configuration that identifies an assembly: no paths, no hash-order-dependent lists."""
    values = {key: value for key, value in dict(options or {}).items() if key not in NON_IDENTITY_OPTION_KEYS}
    values["device"] = device or values.get("device")
    return _canonical_option(values)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assembly_id(*, pdf_path: Path, pages: List[int], configuration: Dict[str, Any],
                 page_texts: Dict[int, str]) -> str:
    """Content-derived identity of one OCR assembly (FG-G02).

    The same source bytes, routed pages, OCR configuration and OCR page text
    always give the same 32-hex id, so a rerun or a resume after a kill writes
    byte-identical Markdown. A per-call random id would make every rerun differ.
    """
    payload = {
        "schema": ASSEMBLY_ID_SCHEMA,
        "source_sha256": _file_sha256(pdf_path),
        "pages": sorted(int(page) for page in pages),
        "configuration": configuration,
        "page_text_sha256": {str(page): hashlib.sha256(page_texts[page].encode("utf-8")).hexdigest()
                             for page in sorted(page_texts)},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _page_count(pdf_path: Path) -> int:
    import pypdfium2
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def _page_split_comment(page_number: int) -> str:
    return f"\n<!-- page:{int(page_number)} -->\n<--- Page Split --->\n"


def _join_page_outputs(page_outputs: List[str]) -> str:
    if not page_outputs:
        return ""
    first_page = str(page_outputs[0])
    parts = [first_page]
    emitted = bool(first_page)
    for page_number, page_text in enumerate(page_outputs[1:], start=2):
        separator = _page_split_comment(page_number)
        if not emitted:
            separator = separator.lstrip("\n")
        parts.append(separator)
        emitted = True
        parts.append(str(page_text))
    return "".join(parts)


def _split_page_outputs(markdown_text: str) -> List[str]:
    content = str(markdown_text or "").rstrip("\n")
    if not content:
        return []
    return PAGE_SPLIT_RE.split(content)


def _serialize_markdown(markdown: str) -> str:
    return str(markdown or "").rstrip("\n") + "\n"


def _write_outputs(
    output_dir: Path,
    stem: str,
    markdown: str,
    page_count: int,
    extra_metrics: dict | None = None,
) -> None:
    md_dir = output_dir / "markdown"
    metrics_dir = output_dir / "json" / "metrics"
    progress_dir = output_dir / "sidecars" / "ocr_progress"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    (md_dir / f"{stem}.md").write_text(_serialize_markdown(markdown), encoding="utf-8")
    metrics = {
        "page_count": page_count,
        "backend": "docling",
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    (metrics_dir / f"{stem}.metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    partial_path = progress_dir / f"{stem}.partial.md"
    if partial_path.exists():
        partial_path.unlink()


def _parse_shard_stem(stem: str) -> Optional[Dict[str, Any]]:
    match = SHARD_STEM_RE.match(str(stem))
    if match is None:
        return None
    return {
        "source_stem": str(match.group("source_stem")),
        "start_page": int(match.group("start")),
        "end_page": int(match.group("end")),
    }


def _split_markdown_pages(markdown_text: str, *, expected_pages: int) -> List[str]:
    pages = _split_page_outputs(markdown_text)
    if len(pages) < int(expected_pages):
        pages.extend([""] * (int(expected_pages) - len(pages)))
    elif len(pages) > int(expected_pages):
        pages = pages[: int(expected_pages)]
    return pages


def _archive_shard_artifact(*, out_root: Path, source_path: Path, relative_path: Path) -> None:
    archive_path = out_root / "sidecars" / "ocr_shards" / relative_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    source_path.replace(archive_path)


def _clear_previous_page_shards(*, out_root: Path, file_list: List[str]) -> None:
    """Remove only target stems' live shards before dispatch; keep canonical files."""
    stems = {Path(name).stem for name in file_list}
    for directory, suffix in ((out_root / "markdown", ".md"),
                              (out_root / "json" / "metrics", ".metrics.json")):
        for path in directory.glob(f"*__p*{suffix}"):
            meta = _parse_shard_stem(path.name.removesuffix(suffix))
            if meta is not None and meta["source_stem"] in stems:
                path.unlink()


def _docling_item_markdown_span(
    item: Dict[str, Any], markdown: str, cursor: int,
) -> tuple[int, int]:
    """Locate an item's existing Markdown without re-exporting its formatting."""
    if "data" in item and "table_cells" in item["data"]:
        cells = [str(cell.get("text", "")).strip() for cell in item["data"]["table_cells"]]
        # A table is one indivisible block, including its Markdown header rule.
        for match in re.finditer(r"(?m)^[ \t]*\|[^\n]*\|[ \t]*(?:\n[ \t]*\|[^\n]*\|[ \t]*)*", markdown):
            if match.start() < cursor:
                continue
            rendered_cells = []
            for line in match.group().splitlines():
                row = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-+:?", cell) for cell in row):
                    continue
                rendered_cells.extend(row)
            if any(cells) and rendered_cells == cells:
                return match.start(), match.end()
        raise ValueError("Docling table cannot be matched to existing Markdown")
    text = str(item.get("text", ""))
    if not text:
        raise ValueError("Docling item has no matchable Markdown text")
    offset = markdown.find(text, cursor)
    if offset < 0:
        raise ValueError("Docling text cannot be matched to existing Markdown")
    # Retain the whole line, including heading/list/emphasis syntax.
    start = markdown.rfind("\n", 0, offset) + 1
    end = markdown.find("\n", offset + len(text))
    end = len(markdown) if end < 0 else end
    prefix, suffix = markdown[start:offset], markdown[offset + len(text):end]
    if (not re.fullmatch(r"[ \t]*(?:(?:#{1,6}|[-+*]|\d+[.)])[ \t]+)?[*_`]*", prefix)
            or not re.fullmatch(r"[*_` \t]*", suffix)):
        raise ValueError("Unmapped content on a Docling text item's Markdown line")
    return start, end


def _docling_markdown_pages(
    *, out_root: Path, source_stem: str, markdown: str, page_count: int,
) -> Optional[List[str]]:
    """Split existing Markdown only when ordered JSON provenance maps all content.

    Ambiguous, crossing-page or unmatched items return None so the caller retains
    the original whole. No Docling dependency or Markdown re-export is required.
    """
    for suffix in (".docling.json", ".docling.json.zst"):
        path = out_root / "json" / f"{source_stem}{suffix}"
        if not path.exists():
            continue
        try:
            data = path.read_bytes()
            if suffix.endswith(".zst"):
                import zstandard

                with zstandard.ZstdDecompressor().stream_reader(data) as reader:
                    data = reader.read()
            document = json.loads(data)
            items: List[Dict[str, Any]] = []
            visited: set[str] = set()

            def visit(node: Dict[str, Any]) -> None:
                if "$ref" in node:
                    ref = str(node["$ref"])
                    if ref in visited:
                        raise ValueError("Repeated Docling item reference")
                    visited.add(ref)
                    resolved = document
                    for part in ref.removeprefix("#/").split("/"):
                        resolved = resolved[int(part)] if isinstance(resolved, list) else resolved[part]
                    node = resolved
                if node.get("text") or "data" in node or node.get("label") == "picture":
                    items.append(node)
                for child in node.get("children", []):
                    visit(child)

            visit(document["body"])
            starts: Dict[int, int] = {}
            cursor = 0
            previous_page = 0
            for item in items:
                pages = {int(prov["page_no"]) for prov in item.get("prov", [])}
                if len(pages) != 1:
                    raise ValueError("Missing or crossing-page Docling provenance")
                page = pages.pop()
                if not previous_page <= page <= page_count or page < 1:
                    raise ValueError("Docling provenance is not in page order")
                start, end = _docling_item_markdown_span(item, markdown, cursor)
                if start < cursor or markdown[cursor:start].strip():
                    raise ValueError("Unmapped Markdown between Docling items")
                starts.setdefault(page, cursor)
                cursor, previous_page = end, page
            if not starts or markdown[cursor:].strip():
                raise ValueError("Unmapped Markdown after Docling items")
            result = [""] * page_count
            ordered = list(starts.items())
            for index, (page, start) in enumerate(ordered):
                end = ordered[index + 1][1] if index + 1 < len(ordered) else len(markdown)
                result[page - 1] = markdown[start:end]
            return result
        except Exception as exc:
            LOGGER.warning("Cannot derive Markdown page boundaries from %s: %s", path, exc)
    return None


def _existing_markdown_pages(markdown: str, page_count: int) -> Optional[List[str]]:
    """Accept complete ordered page markers, never an appended OCR supplement."""
    if "<!-- ocr-pages:" in markdown:
        return None
    markers = list(re.finditer(r"(?m)^[ \t]*<!-- page:(\d+) -->[ \t]*\n(?:<--- Page Split --->\n)?", markdown))
    if markers:
        numbers = [int(marker.group(1)) for marker in markers]
        if numbers == list(range(2, page_count + 1)):
            pages = [markdown[:markers[0].start()].rstrip("\n")]
        elif numbers == list(range(1, page_count + 1)) and not markdown[:markers[0].start()].strip():
            pages = []
        else:
            return None
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(markdown)
            pages.append(markdown[marker.end():end].rstrip("\n"))
        return pages
    pages = _split_page_outputs(markdown)
    return pages if len(pages) == page_count else None


def _retained_native_markdown(*, out_root: Path, source_stem: str) -> str:
    """Save pre-OCR Markdown once; generated canonical text is never the baseline.

    An older supplement without its native snapshot cannot safely establish a
    pre-OCR baseline. Refuse it rather than retaining obsolete generated text.
    Independent OCR runs must not share an output directory.
    """
    native_path = out_root / "sidecars" / "ocr_native" / f"{source_stem}.md"
    canonical_path = out_root / "markdown" / f"{source_stem}.md"
    if native_path.exists():
        original = native_path.read_text(encoding="utf-8")
    else:
        if not canonical_path.exists():
            raise FileNotFoundError(f"Partial OCR requires retained native Markdown for {source_stem}")
        payload = canonical_path.read_bytes()
        original = payload.decode("utf-8")
        if "<!-- ocr-pages:" in original:
            raise FileNotFoundError(f"Partial OCR requires retained native Markdown for {source_stem}; "
                                    "canonical Markdown already contains an OCR supplement")
        if original.strip():
            native_path.parent.mkdir(parents=True, exist_ok=True)
            with native_path.open("xb") as handle:
                handle.write(payload)
    if not original.strip():
        raise FileNotFoundError(f"Partial OCR requires nonempty retained native Markdown for {source_stem}")
    return original


def _previous_ocr_pages(*, state_path: Path, page_count: int) -> Dict[int, Dict[str, str]]:
    """Keep generated page replacements separate from immutable native Markdown."""
    if not state_path.exists():
        return {}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["page_count"] != page_count:
        raise RuntimeError(f"OCR page-state count mismatch: {state_path}")
    pages = {int(page): entry for page, entry in state["pages"].items()}
    if any(not 1 <= page <= page_count or not isinstance(entry.get("text"), str)
           or not re.fullmatch(r"[0-9a-f]{32}", entry.get("run_id", ""))
           for page, entry in pages.items()):
        raise RuntimeError(f"Invalid OCR page state: {state_path}")
    return pages


def _reassemble_canonical_output_for_source(
    *,
    out_root: Path,
    pdf_path: Path,
    source_name: str,
    selected_ranges: Optional[List[tuple[int, int]]] = None,
    ocr_configuration: Optional[Dict[str, Any]] = None,
) -> bool:
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    source_stem = Path(source_name).stem
    canonical_md = md_dir / f"{source_stem}.md"
    canonical_metrics = metrics_dir / f"{source_stem}.metrics.json"
    if not selected_ranges and canonical_md.exists() and canonical_metrics.exists():
        return True

    shard_records: List[Dict[str, Any]] = []
    for metrics_path in sorted(metrics_dir.glob(f"{source_stem}__p*.metrics.json")):
        shard_stem = metrics_path.name.removesuffix(".metrics.json")
        shard_md = md_dir / f"{shard_stem}.md"
        if not shard_md.exists():
            continue
        shard_meta = _parse_shard_stem(shard_stem)
        if shard_meta is None or shard_meta["source_stem"] != source_stem:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        start_page = int(metrics.get("source_start_page", shard_meta["start_page"]))
        end_page = int(metrics.get("source_end_page", shard_meta["end_page"]))
        if selected_ranges and not any(first <= start_page <= end_page <= last for first, last in selected_ranges):
            continue
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
        # Multi-worker planning represents an all-pages selection as a full
        # document, which writes canonical artifacts directly instead of shards.
        if (selected_ranges == [(1, int(_page_count(pdf_path)))]
                and canonical_md.exists() and canonical_metrics.exists()):
            return True
        return False

    produced: set[int] = set()
    for shard in shard_records:
        pages = set(range(int(shard["start_page"]), int(shard["end_page"]) + 1))
        if produced & pages:
            raise RuntimeError(f"Overlapping OCR shards for {source_name}")
        produced.update(pages)
    if selected_ranges:
        expected = {p for first, last in selected_ranges for p in range(first, last + 1)}
        if produced != expected:
            raise RuntimeError(f"Incomplete selected OCR pages for {source_name}")

    shard_records.sort(key=lambda item: (int(item["start_page"]), int(item["end_page"]), str(item["stem"])))
    page_count = max(int(_page_count(pdf_path)), max(int(item["end_page"]) for item in shard_records))
    merged_pages = [""] * int(page_count)
    retained_whole: Optional[str] = None
    kept_markdown_source = "all_pages_ocr"
    partial = bool(selected_ranges) and produced != set(range(1, page_count + 1))
    state_path = out_root / "sidecars" / "ocr_pages" / f"{source_stem}.json"
    previous_pages: Dict[int, Dict[str, str]] = {}
    if partial:
        original = _retained_native_markdown(out_root=out_root, source_stem=source_stem)
        previous_pages = _previous_ocr_pages(state_path=state_path, page_count=page_count)
        original_pages = _existing_markdown_pages(original, page_count)
        if original_pages is not None:
            merged_pages = original_pages
            kept_markdown_source = "page_markers"
        else:
            docling_pages = _docling_markdown_pages(
                out_root=out_root, source_stem=source_stem, markdown=original, page_count=page_count,
            )
            if docling_pages is not None:
                merged_pages = docling_pages
                kept_markdown_source = "docling_json_provenance"
            else:
                retained_whole = original
                kept_markdown_source = "existing_markdown_whole_plus_ocr"
        for page, entry in previous_pages.items():
            merged_pages[page - 1] = entry["text"]
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

        shard_pages = _split_markdown_pages(
            shard["md_path"].read_text(encoding="utf-8"),
            expected_pages=expected_pages,
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
            "kept_markdown_source": kept_markdown_source,
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

    if partial:
        configuration = (ocr_configuration if ocr_configuration is not None else
                         {key: merged_extra_metrics.get(key) for key in REASSEMBLED_CONFIG_KEYS})
        run_id = _assembly_id(pdf_path=pdf_path, pages=sorted(produced), configuration=configuration,
                              page_texts={page: merged_pages[page - 1] for page in produced})
        previous_pages.update({page: {"text": merged_pages[page - 1], "run_id": run_id}
                               for page in produced})
        merged_extra_metrics["ocr_run_id"] = run_id
        merged_extra_metrics["ocr_run_id_basis"] = ASSEMBLY_ID_SCHEMA
        merged_extra_metrics["native_markdown_path"] = str(
            Path("sidecars") / "ocr_native" / f"{source_stem}.md")
    merged_markdown = _join_page_outputs(merged_pages) if merged_pages else "[[Blank page]]"
    if retained_whole is not None:
        from glossapi.page_quality import compress_page_ranges

        # Rebuild one supplement in page order from current replacements. The
        # marker identifies this assembly by content (source, pages, configuration,
        # OCR text), never by a random per-call id; page state retains each
        # producing assembly's id. The ledger attempt records the call itself.
        page_set = compress_page_ranges(previous_pages)
        sections = [f"<!-- page:{page} -->\n{previous_pages[page]['text']}" for page in sorted(previous_pages)]
        marker = f"pages={page_set} run_id={run_id}"
        merged_markdown = (retained_whole + f"\n\n<!-- ocr-pages: {page_set} -->\n"
                           + f"<!-- ocr-supplement-begin: {marker} -->\n"
                           + "\n\n".join(sections)
                           + f"\n<!-- ocr-supplement-end: {marker} -->\n<!-- /ocr-pages -->\n")
    _write_outputs(
        output_dir=out_root,
        stem=source_stem,
        markdown=merged_markdown,
        page_count=int(page_count),
        extra_metrics=merged_extra_metrics,
    )
    if partial:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps({"page_count": page_count, "pages": previous_pages}), encoding="utf-8")
            temporary.replace(state_path)
        finally:
            temporary.unlink(missing_ok=True)
    for shard in shard_records:
        _archive_shard_artifact(
            out_root=out_root,
            source_path=Path(shard["md_path"]),
            relative_path=Path("markdown") / Path(shard["md_path"]).name,
        )
        _archive_shard_artifact(
            out_root=out_root,
            source_path=Path(shard["metrics_path"]),
            relative_path=Path("json") / "metrics" / Path(shard["metrics_path"]).name,
        )
    return True


def _ensure_canonical_outputs(*, out_root: Path, pdf_root: Path, file_list: List[str],
                              page_ranges: Optional[List[str]] = None,
                              source_paths: Optional[Dict[str, Path]] = None,
                              ocr_configuration: Optional[Dict[str, Any]] = None) -> None:
    """Reassemble nonoverlapping current shards while retaining kept Markdown."""
    selected: Dict[str, List[tuple[int, int]]] = {}
    for spec in page_ranges or []:
        name, first, last = spec.rsplit(":", 2)
        selected.setdefault(name, []).append((int(first), int(last)))
    for name in file_list:
        pdf_path = (source_paths or {}).get(name, pdf_root / name).resolve()
        assembled = _reassemble_canonical_output_for_source(
            out_root=out_root, pdf_path=pdf_path, source_name=name,
            selected_ranges=selected.get(name), ocr_configuration=ocr_configuration,
        )
        if name in selected and not assembled:
            raise FileNotFoundError(f"No selected OCR shards produced for {name}")


