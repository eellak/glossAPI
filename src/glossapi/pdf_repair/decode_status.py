"""Per-page content-stream decode status of the safe extractor (K5 quality policy P4b, K5-A2b).

``partial``: the native content-stream check (``glossapi_rs_pdfrepair.page_decode_status_json``) observed a decode
failure or a truncated content stream on the page (its own streams or a Form XObject it draws); the page keeps the
readable text PDFium extracts. ``unknown``: the check could not run or could not decide (the native module is
missing or stale, the document cannot be parsed by the native parser - for example an encrypted document - its page
count differs from PDFium's, the page is unavailable, a stream object is unreadable, or a filter the check does not
implement). ``complete``: the check ran and found no decode failure. PDFium's own error state is not a signal (on
the K5P-093 page ``FPDF_GetLastError()`` is 0 before and after extracting 1,583 characters).

The status is recorded in the extract sidecar (``decode_status``) and in the ledger's page coverage; it never routes
a page to OCR by itself (routing follows the normal quality thresholds).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

SCHEMA = "glossapi.decode_status.v1"
CHECK = "glossapi_rs_pdfrepair.page_decode_status_json"
STATUSES = ("complete", "partial", "unknown")


def _unknown(page_count: int, reason: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "check": CHECK, "pages": ["unknown"] * page_count, "reasons": {}, "reason": reason}


def page_decode_status(path: Path, page_count: int) -> dict[str, Any]:
    """The ``decode_status`` metadata of the PDF at ``path`` (the file PDFium read) with ``page_count`` pages:
    ``pages`` (one status per page, PDFium order), ``reasons`` (page -> reason, for pages that are not complete) and,
    when the whole check could not run, ``reason``."""
    from ..markdown_layout import native_module
    native, problem = native_module()
    if native is None:
        return _unknown(page_count, problem or "native module unavailable")
    try:
        rows = json.loads(native.page_decode_status_json(str(path)))
    except (OSError, ValueError) as error:
        return _unknown(page_count, f"native parser cannot read the document: {error}")
    if len(rows) != page_count:
        return _unknown(page_count, f"native page count {len(rows)} differs from PDFium's {page_count}")
    pages: list[str] = []
    reasons: dict[str, str] = {}
    for index, (status, reason) in enumerate(rows, 1):
        if status not in STATUSES:
            status, reason = "unknown", f"unexpected status {status!r}"
        pages.append(status)
        if status != "complete":
            reasons[str(index)] = str(reason or status)
    return {"schema": SCHEMA, "check": CHECK, "pages": pages, "reasons": reasons}


def summary(metadata: Optional[dict[str, Any]]) -> dict[str, int]:
    """Counts per status of a ``decode_status`` metadata value."""
    counts = {status: 0 for status in STATUSES}
    for status in (metadata or {}).get("pages", []):
        counts[status] = counts.get(status, 0) + 1
    return counts
