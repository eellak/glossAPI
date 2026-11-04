"""Bounded OCR attempt budgets, computed before worker dispatch."""
from __future__ import annotations

import json
import math


def attempt_budget(artifacts, settings, default=600):
    """120 s startup + 45 s/routed page, floor 600 s, ceiling 7200 s.

    An explicit timeout_seconds is authoritative. A fixed absolute deadline
    still kills a hung worker; retries do not extend an active attempt. Only
    hash-verified quality/source inputs may determine the automatic budget.
    """
    if 'timeout_seconds' in settings or default != 600:
        seconds = float(settings.get('timeout_seconds', default))
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('timeout_seconds must be finite and positive')
        return seconds, {'policy':'explicit', 'seconds':seconds}
    from ..stages.execution import sha256_file
    count = None
    count_error = None
    quality = next((a for a in artifacts if a.kind == 'page_quality'), None)
    ranges = settings.get('page_ranges')
    maximum = settings.get('max_pages')
    if ranges:
        count = len({page for first, last in ranges for page in range(int(first), int(last)+1)
                     if maximum is None or page <= maximum})
    elif quality is not None and sha256_file(quality.path) == quality.sha256:
        report = json.loads(quality.path.read_text())
        count = sum(p['page_needs_ocr'] is True and (maximum is None or p['page'] <= maximum)
                    for p in report['pages'])
    else:
        source = next((a for a in artifacts if a.kind in ('source_payload','download_payload')), None)
        if source is not None and sha256_file(source.path) == source.sha256:
            from .page_artifacts import _page_count
            try:
                count = _page_count(source.path)
            except Exception as error:
                # Counting is a budget estimate, never the source-open verdict.
                # The stage reports an unreadable PDF through its normal path.
                count_error = f'{type(error).__name__}: {error}'
            if count is not None and maximum is not None:
                count = min(count, maximum)
    seconds = max(float(default), min(7200, 120 + 45 * (count or 0)))
    return seconds, {'policy':'routed_pages_v1', 'pages':count, 'startup_seconds':120,
                     'seconds_per_page':45, 'ceiling_seconds':7200, 'seconds':seconds,
                     **({'count_error':count_error} if count_error is not None else {})}
