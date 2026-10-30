"""Request normalization for corpus OCR orchestration."""

from __future__ import annotations

from ...ledger.journal import observe

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(slots=True)
class OcrRequest:
    mode: str
    backend: str
    ocr_options: object | None
    device: Optional[str]
    max_pages: Optional[int]
    math_enhance: bool
    math_targets: Optional[Dict[str, List[Tuple[int, int]]]]
    math_batch_size: int
    math_dpi_base: int
    use_gpus: str
    devices: Optional[List[int]]
    reprocess_completed: bool
    content_debug: bool
    page_routing: str = "flagged"
    num_threads: int = 1


def _resolve_mode(
    *,
    logger,
    mode: Optional[str],
    fix_bad: bool,
    math_enhance: bool,
) -> Optional[str]:
    mode_norm: Optional[str] = None
    if mode:
        candidate = str(mode).strip().lower()
        if candidate in {"ocr_bad", "math_only", "ocr_bad_then_math"}:
            mode_norm = candidate
        else:
            logger.warning("Unknown mode '%s'; falling back to legacy flags", mode)
    if mode_norm is None:
        if fix_bad and math_enhance:
            mode_norm = "ocr_bad_then_math"
        elif fix_bad:
            mode_norm = "ocr_bad"
        elif math_enhance:
            mode_norm = "math_only"
    return mode_norm


def normalize_ocr_request(
    *,
    logger,
    fix_bad: bool,
    mode: Optional[str],
    backend: str,
    ocr_options=None,
    device: Optional[str],
    max_pages: Optional[int],
    page_routing: str = "flagged",
    math_enhance: bool = True,
    math_targets: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    math_batch_size: int = 8,
    math_dpi_base: int = 220,
    use_gpus: str = "single",
    devices: Optional[List[int]] = None,
    force: Optional[bool] = None,
    reprocess_completed: Optional[bool] = None,
    skip_existing: Optional[bool] = None,
    content_debug: bool = False,
    CONTENT_DEBUG: Optional[bool] = None,
    internal_debug: bool = False,
    INTERNAL_DEBUG: Optional[bool] = None,
) -> Optional[OcrRequest]:
    if page_routing not in {"flagged", "all"}:
        raise ValueError("page_routing must be 'flagged' or 'all'")
    backend_norm = str(backend or "docling").strip().lower()
    if backend_norm != "docling":
        raise ValueError("backend must be 'docling'; configure an external OCR plugin with ocr_options")

    if CONTENT_DEBUG is not None:
        content_debug = bool(CONTENT_DEBUG)
    elif INTERNAL_DEBUG is not None:
        content_debug = bool(INTERNAL_DEBUG)
    elif internal_debug:
        content_debug = True

    fix_bad_effective = bool(fix_bad)
    if force is not None:
        logger.warning("Corpus.ocr(force=...) is deprecated; use fix_bad=... instead")
        fix_bad_effective = bool(force)

    mode_norm = _resolve_mode(
        logger=logger,
        mode=mode,
        fix_bad=fix_bad_effective,
        math_enhance=bool(math_enhance),
    )
    if mode_norm is None:
        observe("ocr_not_requested", status="skipped")
        logger.info(
            "OCR: no operation requested (enable fix_bad and/or math_enhance or set mode='ocr_bad'|'math_only'|'ocr_bad_then_math')"
        )
        return None

    reprocess_explicit = reprocess_completed is not None
    reprocess_flag = bool(reprocess_completed) if reprocess_explicit else False
    if skip_existing is not None:
        skip_flag = bool(skip_existing)
        logger.warning(
            "Corpus.ocr(skip_existing=...) is deprecated; use reprocess_completed=... instead."
        )
        desired = not skip_flag
        if reprocess_explicit and desired != reprocess_flag:
            logger.info(
                "Corpus.ocr(): skip_existing=%s overrides reprocess_completed=%s (effective reprocess_completed=%s).",
                skip_flag,
                reprocess_flag,
                desired,
            )
        reprocess_flag = desired

    return OcrRequest(
        mode=mode_norm,
        page_routing=page_routing,
        backend=backend_norm,
        ocr_options=ocr_options,
        device=device,
        max_pages=max_pages,
        math_enhance=bool(math_enhance),
        math_targets=math_targets,
        math_batch_size=int(math_batch_size),
        math_dpi_base=int(math_dpi_base),
        use_gpus=str(use_gpus or "single"),
        devices=devices,
        reprocess_completed=bool(reprocess_flag),
        content_debug=bool(content_debug),
    )
