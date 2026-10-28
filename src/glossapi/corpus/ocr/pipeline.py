"""High-level OCR orchestration for corpus remediation."""

from __future__ import annotations

from ...ledger.journal import observe

from .artifacts import persist_ocr_success, refresh_cleaner_after_ocr
from .config import OcrRequest
from .context import CorpusOcrContext
from . import dispatch
from .targets import build_ocr_selection


def run_ocr_phase(context: CorpusOcrContext, request: OcrRequest) -> list[str]:
    """Run the OCR-remediation path while preserving the current runtime engine."""

    if request.mode == "math_only":
        raise ValueError("run_ocr_phase handles OCR remediation only")

    selection = build_ocr_selection(
        context,
        mode=request.mode,
        page_routing=request.page_routing,
        reprocess_completed=request.reprocess_completed,
    )

    if not selection.bad_files:
        observe("ocr_not_routed", status="skipped")
        context.logger.info("OCR: no bad documents flagged by cleaner; skipping OCR fix")
        return []

    dispatch.run_plugin_ocr(
        context,
        selection.bad_files,
        request=request,
        **({"page_ranges": selection.page_ranges} if selection.page_ranges else {}),
    )

    try:
        persist_ocr_success(
            context,
            filenames=selection.bad_files,
            backend_norm=request.backend,
        )
    except Exception as exc:
        observe("ocr_metadata_persist_failed", error=exc, status="failed")
        context.logger.warning("Failed to update OCR success metadata: %s", exc)

    try:
        refresh_cleaner_after_ocr(context)
    except Exception as exc:
        observe("ocr_clean_refresh_failed", error=exc, status="failed")
        context.logger.warning("Cleaner refresh after OCR failed: %s", exc)

    return selection.bad_files
