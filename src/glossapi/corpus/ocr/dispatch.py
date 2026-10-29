"""OCR remediation through Docling's external OCR engine entry points."""
from __future__ import annotations

from ...ledger.journal import observe

from glossapi.optional_dependencies import optional_dependencies

from importlib.metadata import entry_points

from ...ocr import page_artifacts
from .config import OcrRequest
from .context import CorpusOcrContext


def require_ocr_options(options):
    """Require explicit engine configuration; never fall back to bundled OCR."""
    if options is not None:
        return options
    plugins = [point for point in entry_points(group="docling") if point.name != "docling"]
    if not plugins:
        raise RuntimeError(
            "Install an OCR plugin (for example the toolkit's greekocr package) "
            "registered in the Docling 'docling' entry-point group, then pass its "
            "configured options as ocr_options to Corpus.ocr()."
        )
    raise ValueError(
        "Configure an installed OCR plugin and pass its options as ocr_options to Corpus.ocr()."
    )


def _build_converter(request):
    try:
        with optional_dependencies("layout"):
            from docling.datamodel.base_models import InputFormat
        with optional_dependencies("layout"):
            from docling.document_converter import DocumentConverter, PdfFormatOption
        from ...ocr.docling.pipeline import _apply_common_pdf_options, _resolve_accelerator
    except ImportError as exc:
        raise RuntimeError("Install glossapi[docling] and an OCR plugin to use Corpus.ocr().") from exc
    accelerator, _ = _resolve_accelerator(request.device or "cpu")
    options = _apply_common_pdf_options(
        acc=accelerator, images_scale=1.25, formula_enrichment=False,
        code_enrichment=False, ocr_options=request.ocr_options,
    )
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def run_plugin_ocr(
    context: CorpusOcrContext, filenames: list[str], *, request: OcrRequest,
    page_ranges: list[str] | None = None,
) -> None:
    import os
    if os.environ.get("GLOSSAPI_STAGE_PROFILE") == "sequential-adapters" and not getattr(context, "_inside_ocr_stage", False):
        from ...stages.compatibility import ocr_compatibility_documents
        return ocr_compatibility_documents(context, filenames, request=request, page_ranges=page_ranges)
    require_ocr_options(request.ocr_options)
    if request.max_pages is not None and request.max_pages < 1:
        raise ValueError("max_pages must be positive")
    selected = {}
    for spec in page_ranges or []:
        name, first, last = spec.rsplit(":", 2)
        selected.setdefault(name, []).append((int(first), int(last)))
    if set(selected) - set(filenames):
        raise ValueError("Page ranges refer to files outside the OCR targets")
    # Validate every partial target before loading models or touching live shards.
    jobs = []
    for name in filenames:
        source = next((directory / name for directory in
                       (context.output_dir / "downloads", context.input_dir / "downloads", context.input_dir)
                       if (directory / name).is_file()), context.input_dir / name)
        count = page_artifacts._page_count(source)
        ranges = selected.get(name, [(1, count)])
        if any(not 1 <= first <= last <= count for first, last in ranges):
            raise ValueError(f"Invalid OCR page range for {name}")
        if request.max_pages is not None:
            ranges = [(first, min(last, request.max_pages)) for first, last in ranges
                      if first <= request.max_pages]
        if not ranges:
            raise ValueError(f"No selected OCR pages within max_pages for {name}")
        if {page for first, last in ranges for page in range(first, last + 1)} != set(range(1, count + 1)):
            page_artifacts._retained_native_markdown(out_root=context.output_dir, source_stem=source.stem)
        jobs.append((name, source, ranges))
    converter = _build_converter(request)
    page_artifacts._clear_previous_page_shards(out_root=context.output_dir, file_list=filenames)
    completed_ranges = []
    for name, source, ranges in jobs:
        for first, last in ranges:
            result = converter.convert(source, page_range=(first, last))
            status = getattr(result.status, "value", result.status)
            if status != "success":
                observe("ocr_plugin_conversion_failed", path=source, status="failed", backend_status=str(status), page_first=first, page_last=last)
                raise RuntimeError(f"OCR plugin failed for {name}:{first}:{last}: {result.status}")
            # Keep the source PDF intact: engines see original physical page numbers.
            page_artifacts._write_outputs(
                output_dir=context.output_dir, stem=f"{source.stem}__p{first:05d}-{last:05d}",
                markdown=page_artifacts._join_page_outputs([
                    result.document.export_to_markdown(page_no=page) for page in range(first, last+1)
                ]), page_count=last-first+1,
                extra_metrics={"source_start_page": first, "source_end_page": last,
                               "ocr_engine": request.ocr_options.kind, "device": request.device or "cpu"},
            )
            observe("ocr_completed", path=source, status="ok", page_first=first, page_last=last, plugin_kind=request.ocr_options.kind, page_drop_reason="unknown_not_reported")
            completed_ranges.append(f"{name}:{first}:{last}")
    page_artifacts._ensure_canonical_outputs(
        out_root=context.output_dir, pdf_root=context.input_dir,
        file_list=filenames, page_ranges=completed_ranges,
        source_paths={name: source for name, source, _ in jobs},
    )
