"""OCR helpers for GlossAPI using Docling + RapidOCR (ONNXRuntime).

GPU-first OCR that auto-discovers packaged ONNX models and Greek keys within
the installed `glossapi` package. Designed as a drop-in for Corpus.ocr().
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

_PIPELINE_CACHE: dict[str, Tuple[object, object]] = {}


def _build_pipeline(
    device: Optional[str] = None,
    *,
    use_cls: Optional[bool] = None,
    text_score: Optional[float] = None,
    images_scale: Optional[float] = None,
):
    # Delegate to canonical builder to avoid duplication
    from glossapi.ocr.rapidocr.pipeline import build_rapidocr_pipeline

    engine, opts = build_rapidocr_pipeline(
        device=(device or "cuda:0"),
        text_score=(0.45 if text_score is None else float(text_score)),
        images_scale=(1.25 if images_scale is None else float(images_scale)),
        formula_enrichment=False,
        code_enrichment=False,
    )
    # Apply use_cls override if requested
    try:
        if use_cls is not None and hasattr(opts, "ocr_options"):
            setattr(opts.ocr_options, "use_cls", bool(use_cls))  # type: ignore[attr-defined]
    except Exception:
        pass
    return engine, opts


def run_rapidocr_onnx(
    pdf_path: Path | str,
    *,
    device: Optional[str] = None,
    use_cls: Optional[bool] = None,
    text_score: Optional[float] = None,
    images_scale: Optional[float] = None,
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
        pipe, r = _build_pipeline(device=device, use_cls=use_cls, text_score=text_score, images_scale=images_scale)
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
