"""OCR helpers for GlossAPI using Docling + RapidOCR (ONNXRuntime).

GPU-first OCR that auto-discovers packaged ONNX models and Greek keys within
the installed `glossapi` package. Designed as a drop-in for Corpus.ocr().
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

_PIPELINE_CACHE: dict[str, Tuple[object, object]] = {}


def _build_pipeline(device: Optional[str] = None):
    # Lazy imports to keep module import light
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        LayoutOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    try:
        from docling.models.rapid_ocr_model import RapidOcrModel  # type: ignore
        try:
            from docling.pipelines.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
        except Exception:
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
    except Exception as e:
        raise RuntimeError("Docling RapidOCR modules not available; install docling[rapidocr].") from e

    from ._rapidocr_paths import resolve_packaged_onnx_and_keys

    # Device selection (GPU preferred)
    dev = device or "cuda:0"
    acc = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if str(dev).lower().startswith("cuda") else AcceleratorDevice.CPU
    )

    # Resolve packaged model paths
    r = resolve_packaged_onnx_and_keys()
    if not (r.det and r.rec and r.cls):
        raise FileNotFoundError(
            "Packaged RapidOCR ONNX models not found in glossapi.models. "
            "Add det/rec/cls under models/rapidocr/onnx and keys under models/rapidocr/keys."
        )

    ocr_opts = RapidOcrOptions(
        backend="onnxruntime",
        lang=["el", "en"],
        force_full_page_ocr=False,
        use_det=True,
        use_cls=True,
        use_rec=True,
        text_score=0.45,
        det_model_path=r.det,
        rec_model_path=r.rec,
        cls_model_path=r.cls,
        print_verbose=False,
    )
    if r.keys:
        ocr_opts.rec_keys_path = r.keys

    table_opts = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    opts = PdfPipelineOptions(
        accelerator_options=acc,
        ocr_options=ocr_opts,
        layout_options=LayoutOptions(),
        do_ocr=True,
        do_table_structure=True,
        force_backend_text=False,
        generate_parsed_pages=False,
        table_structure_options=table_opts,
        allow_external_plugins=True,
    )

    # Prefer explicit injection path when supported; otherwise fall back to factory
    try:
        import inspect as _inspect
        sig = _inspect.signature(StandardPdfPipeline.__init__)
        if "ocr_model" in sig.parameters:
            # Docling 2.48.0 RapidOcrModel signature: (enabled, artifacts_path, options, accelerator_options)
            ocr_model = RapidOcrModel(True, None, ocr_opts, acc)  # type: ignore[arg-type]
            pipeline = StandardPdfPipeline(opts, ocr_model=ocr_model)  # type: ignore
            return pipeline, r
    except Exception:
        pass

    # Fallback: construct a DocumentConverter using the factory path
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return converter, r


def run_rapidocr_onnx(
    pdf_path: Path | str,
    *,
    device: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Docling + RapidOCR (ONNX) OCR on a PDF and return markdown text.

    Returns
    -------
    dict with keys:
      - markdown_text: str
      - duration_s: float
      - pages: int
      - models: dict with file names of det/rec/cls/keys
    """
    from time import perf_counter
    pdf_p = Path(pdf_path)
    if not pdf_p.exists():
        raise FileNotFoundError(pdf_p)

    key = str(device or "cuda:0").lower()
    cached = _PIPELINE_CACHE.get(key)
    if cached is None:
        pipe, r = _build_pipeline(device=device)
        _PIPELINE_CACHE[key] = (pipe, r)
    else:
        pipe, r = cached  # type: ignore[misc]

    t0 = perf_counter()
    conv = pipe.convert(source=str(pdf_p))  # type: ignore[attr-defined]
    doc = conv.document
    md_text = doc.export_to_markdown()
    duration = perf_counter() - t0

    # Attempt to get page count from conv/document
    pages = 0
    try:
        if hasattr(doc, "pages"):
            pages = len(doc.pages)  # type: ignore[attr-defined]
    except Exception:
        pages = 0

    # Return model identifiers as file names only (no full paths)
    import os as _os
    models = {
        "det": _os.path.basename(r.det) if r.det else None,
        "rec": _os.path.basename(r.rec) if r.rec else None,
        "cls": _os.path.basename(r.cls) if r.cls else None,
        "keys": _os.path.basename(r.keys) if r.keys else None,
    }

    return {
        "markdown_text": md_text or "",
        "duration_s": duration,
        "pages": int(pages),
        "models": models,
    }


__all__ = [
    "run_rapidocr_onnx",
]
