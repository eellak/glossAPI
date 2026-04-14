"""High-level OCR orchestration for corpus remediation."""

from __future__ import annotations

from .artifacts import persist_ocr_success, refresh_cleaner_after_ocr
from .config import OcrRequest
from .context import CorpusOcrContext
from .dispatch import run_deepseek_ocr
from .targets import build_ocr_selection


def run_ocr_phase(context: CorpusOcrContext, request: OcrRequest) -> None:
    """Run the OCR-remediation path while preserving the current runtime engine."""

    if request.mode == "math_only":
        raise ValueError("run_ocr_phase handles OCR remediation only")

    selection = build_ocr_selection(
        context,
        mode=request.mode,
        reprocess_completed=request.reprocess_completed,
    )

    if not selection.bad_files:
        context.logger.info("OCR: no bad documents flagged by cleaner; skipping OCR fix")
        return

    run_deepseek_ocr(
        context,
        request=request,
        filenames=selection.bad_files,
    )

    try:
        persist_ocr_success(
            context,
            filenames=selection.bad_files,
            backend_norm=request.backend,
        )
    except Exception as exc:
        context.logger.warning("Failed to update OCR success metadata: %s", exc)

    try:
        refresh_cleaner_after_ocr(context)
    except Exception as exc:
        context.logger.warning("Cleaner refresh after OCR failed: %s", exc)
