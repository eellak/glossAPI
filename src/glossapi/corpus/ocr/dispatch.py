"""OCR remediation through Docling's external OCR engine entry points."""
from __future__ import annotations

from ...stages.profile import adapters_enabled

from ...ledger.journal import observe

from glossapi.optional_dependencies import optional_dependencies

from importlib.metadata import entry_points
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from ...ocr import page_artifacts
from .config import OcrRequest
from .context import CorpusOcrContext


def require_ocr_options(options):
    """Explicit options win; the sole default engine is the Greek OCR plugin."""
    if options is not None:
        return options
    from ...ocr.defaults import INSTALL_MESSAGE
    if not any(point.name == 'glossapi' for point in entry_points(group='docling')):
        raise RuntimeError(INSTALL_MESSAGE)
    return None  # Options need the run's source-bound routing sidecar first.


def _build_converter(request):
    import os
    try:
        with optional_dependencies("layout"):
            from docling.datamodel.base_models import InputFormat
        with optional_dependencies("layout"):
            from docling.document_converter import DocumentConverter, PdfFormatOption
        from ...ocr.docling.pipeline import _apply_common_pdf_options, _resolve_accelerator
    except ImportError as exc:
        from ...ocr.defaults import INSTALL_MESSAGE
        raise RuntimeError(INSTALL_MESSAGE) from exc
    accelerator, _ = _resolve_accelerator(request.device or "cpu")
    accelerator.num_threads = request.num_threads
    import torch
    torch.set_num_threads(request.num_threads)
    options = _apply_common_pdf_options(
        acc=accelerator, images_scale=getattr(request.ocr_options,'crop_scale',1.25), formula_enrichment=False,
        code_enrichment=False, ocr_options=request.ocr_options,
    )
    if os.environ.get('DOCLING_ARTIFACTS_PATH'):
        options.artifacts_path = Path(os.environ['DOCLING_ARTIFACTS_PATH'])
    format_options = {'pipeline_options':options}
    if request.ocr_options.kind == 'glossapi':
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        format_options['backend'] = PyPdfiumDocumentBackend
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(**format_options)})


def run_plugin_ocr(
    context: CorpusOcrContext, filenames: list[str], *, request: OcrRequest,
    page_ranges: list[str] | None = None,
) -> None:
    from ...ocr.defaults import check_requested_device
    # Reject an explicit device that disagrees with ocr_options.device before any
    # pool selection, adapter delegation or converter construction.
    check_requested_device(request.device, request.ocr_options)
    require_ocr_options(request.ocr_options)
    if adapters_enabled() and not getattr(context, "_inside_ocr_stage", False):
        from ...stages.compatibility import ocr_compatibility_documents
        return ocr_compatibility_documents(context, filenames, request=request, page_ranges=page_ranges)
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
    from ...ocr.defaults import cache_root
    # The plugin reads its cache root only from GREEKOCR_CACHE_DIR, at model
    # acquisition time. Scope the override to this call (OCRDEF-C2-G03).
    with _scoped_environment('GREEKOCR_CACHE_DIR', str(cache_root(getattr(context, '_ocr_cache_root', None)))) as cache_directory:
        _convert_jobs(context, filenames, jobs, request=request, cache_directory=cache_directory)


@contextmanager
def _scoped_environment(name, value):
    """Set one environment variable for the block, then restore the caller's value or absence."""
    import os
    missing = object()
    previous = os.environ.get(name, missing)
    os.environ[name] = value
    try:
        yield value
    finally:
        if previous is missing:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _convert_jobs(context, filenames, jobs, *, request, cache_directory):
    from ...ocr.defaults import greek_options, model_provenance, select_device, serialize_options, write_routes, package_requirements
    requested_device = request.device
    device = (getattr(request.ocr_options, 'device', None) or select_device(request.device))
    options = request.ocr_options
    if options is None or getattr(options, 'kind', None) == 'glossapi':
        sidecars = context.output_dir/'sidecars/ocr_runtime'
        sidecars.mkdir(parents=True, exist_ok=True)
        routing = sidecars/'routing.jsonl'
        write_routes(routing, jobs)
        values = serialize_options(options) if options is not None else {}
        values.pop('kind', None)
        # Corpus owns routing, including when the caller supplied replace_policy=all.
        # Restrict conversion AND engine replacement to its selected physical pages.
        values['replace_policy'] = 'routed_pages'
        options = greek_options(values, device=device, routing_sidecar=routing,
                                call_log=sidecars/'calls.jsonl')
        from ...ocr.defaults import toolkit_source_status
        toolkit_source = toolkit_source_status()
        if not toolkit_source['admitted']:
            raise RuntimeError(toolkit_source['reason'])
    request = replace(request, ocr_options=options, device=device)
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
            if options.kind == 'glossapi':
                import json
                calls=[json.loads(line) for line in options.call_log.read_text().splitlines()]
                actual={call['page']:call for call in calls if call['filename']==source.name}
                if any(page not in actual or actual[page]['status']!='replaced' for page in range(first,last+1)):
                    raise RuntimeError('Greek OCR did not replace every needs_ocr page; inspect plugin call log')
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
    if options.kind == 'glossapi':
        import json
        provenance = {'options':serialize_options(options), 'requested_device':requested_device, 'device':device,
            'detector_device':options.detector_device, 'models':model_provenance(converter),
            'cache_root':cache_directory}
        provenance['model_source']=('local_release_override' if options.release_dir else
            'local_checkpoint_override' if options.checkpoint else
            'verified_hf_fetch' if provenance['models'] else 'not_loaded_no_lines')
        from ...ocr.defaults import detector_provenance, toolkit_source_status
        source=toolkit_source_status(fingerprint=True)
        provenance['toolkit']={'version':source.get('version'),
            'requested_source':next(row for row in package_requirements() if row.startswith('greek-ocr-toolkit')),
            'installed_source':source.get('installed_direct_url'),
            'admission':source.get('status'),'override_declared':source.get('override_declared'),
            'installed_code_fingerprint':source.get('installed_code_fingerprint')}
        detector_calls=[json.loads(line).get('detector') for line in options.call_log.read_text().splitlines()]
        detector_observations=list({json.dumps(row,sort_keys=True):row for row in (
            {key:call.get(key) for key in ('path','sha256','requested_device','providers')}
            for call in detector_calls if call)}.values())
        # Override status and the expected identity are recorded even when no line reached the detector.
        provenance['detector']=detector_provenance(detector_observations,
            segmenter_detector=getattr(options,'segmenter_detector','small'))
        (context.output_dir/'sidecars/ocr_runtime/provenance.json').write_text(json.dumps(provenance), encoding='utf-8')
        observe('ocr_completed', status='ok', **provenance)
