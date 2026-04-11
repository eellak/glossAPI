"""High-level corpus OCR orchestration."""

from __future__ import annotations

from .context import CorpusOcrContext
from .artifacts import persist_ocr_success, refresh_cleaner_after_ocr
from .config import OcrRequest
from .dispatch import run_deepseek_ocr
from .math_runtime import run_math_enrichment
from .math_targets import (
    discover_docling_json_stems,
    ensure_math_placeholder_sidecars,
    filter_math_only_stems,
    select_followup_math_stems,
)
from .targets import build_ocr_selection


def run_ocr_phase(context: CorpusOcrContext, request: OcrRequest) -> None:
    mode = request.mode

    if request.backend == "deepseek" and mode in {"ocr_bad", "ocr_bad_then_math"}:
        context.logger.info(
            "DeepSeek backend: Phase-2 math is not required; equations are included inline via OCR."
        )
        if mode == "ocr_bad_then_math":
            context.logger.info(
                "DeepSeek OCR does not run Phase-2 math; treating mode='ocr_bad_then_math' as 'ocr_bad'."
            )
            mode = "ocr_bad"

    selection = build_ocr_selection(
        context,
        mode=mode,
        reprocess_completed=request.reprocess_completed,
    )

    if mode == "math_only":
        stems = discover_docling_json_stems(context.output_dir)
        stems = filter_math_only_stems(
            stems=stems,
            bad_files=selection.bad_files,
            math_done_stems=selection.math_done_stems,
            reprocess_completed=request.reprocess_completed,
            logger=context.logger,
        )
        run_math_enrichment(
            context,
            stems=stems,
            request=request,
            skip_mgr=selection.skip_mgr,
            skiplist_path=selection.skiplist_path,
        )
        return

    if mode in {"ocr_bad", "ocr_bad_then_math"} and not selection.bad_files:
        context.logger.info("OCR: no bad documents flagged by cleaner; skipping OCR fix")
        if mode == "ocr_bad_then_math":
            stems = discover_docling_json_stems(context.output_dir)
            run_math_enrichment(
                context,
                stems=stems,
                request=request,
                skip_mgr=selection.skip_mgr,
                skiplist_path=selection.skiplist_path,
            )
        return

    reran_ocr = False
    if mode in {"ocr_bad", "ocr_bad_then_math"}:
        if request.backend == "deepseek":
            try:
                run_deepseek_ocr(
                    context,
                    request=request,
                    filenames=selection.bad_files,
                )
            except Exception as exc:
                context.logger.error("DeepSeek OCR runner failed: %s", exc)
                raise
        reran_ocr = True

        try:
            persist_ocr_success(
                context,
                filenames=selection.bad_files,
                backend_norm=request.backend,
            )
        except Exception as exc:
            context.logger.warning("Failed to update OCR success metadata: %s", exc)

    if reran_ocr:
        try:
            refresh_cleaner_after_ocr(context)
        except Exception as exc:
            context.logger.warning("Cleaner refresh after OCR failed: %s", exc)

    if mode == "ocr_bad_then_math":
        try:
            stems = discover_docling_json_stems(context.output_dir)
            stems = select_followup_math_stems(
                stems=stems,
                bad_files=selection.bad_files,
                math_done_stems=selection.math_done_stems,
                reprocess_completed=request.reprocess_completed,
                reran_ocr=reran_ocr,
                logger=context.logger,
            )
            if not stems:
                context.logger.info("Math enrichment: no pending documents after filtering.")
                return
            ensure_math_placeholder_sidecars(
                context,
                stems=stems,
                include_parquet_signals=True,
            )
            context.logger.info("OCR: invoking Phase-2 math for stems: %s", ",".join(stems))
            run_math_enrichment(
                context,
                stems=stems,
                request=request,
                skip_mgr=selection.skip_mgr,
                skiplist_path=selection.skiplist_path,
            )
            context.logger.info("OCR: Phase-2 math completed for stems: %s", ",".join(stems))
        except Exception as exc:
            context.logger.warning("Phase-2 enrichment after OCR failed: %s", exc)
