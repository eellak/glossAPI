from __future__ import annotations

from typing import Iterable, Optional


def run_via_extract(
    corpus,
    files: Iterable[str],
    *,
    export_doc_json: bool = False,
    internal_debug: bool = False,
    content_debug: Optional[bool] = None,
) -> None:
    """Thin adapter that forwards to Corpus.extract for RapidOCR/Docling.

    This exists for symmetry with deepseek_runner and to keep the OCR package
    as the single entry point for OCR backends.
    """
    # Note: internal_debug/content_debug are no-ops for the Docling/RapidOCR path.
    # Docling's output already produces a single concatenated Markdown document.
    corpus.extract(
        input_format="pdf",
        num_threads=1,  # let extract decide; override in tests if needed
        accel_type="CUDA",
        force_ocr=True,
        formula_enrichment=False,
        code_enrichment=False,
        filenames=list(files),
        skip_existing=False,
        export_doc_json=bool(export_doc_json),
        emit_formula_index=bool(export_doc_json),
        phase1_backend="docling",
    )
