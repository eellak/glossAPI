"""Bounded OCR attempt budgets, computed before worker dispatch."""
from __future__ import annotations

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
    count = None
    count_error = None
    from .selection import effective_selection
    try:
        selected, _ = effective_selection(artifacts, settings)
        if selected is not None:
            count = len(selected)
        else:
            source = next((a for a in artifacts if a.kind in ('source_payload','download_payload')), None)
            if source is not None:
                from .page_artifacts import _page_count
                count = _page_count(source.path)
                if settings.get('max_pages') is not None:
                    count = min(count, settings['max_pages'])
    except Exception as error:
        # The worker reports invalid/stale inputs; a budget estimate cannot
        # authorize them or replace the stage's normal source-open verdict.
        count_error = f'{type(error).__name__}: {error}'
    seconds = max(float(default), min(7200, 120 + 45 * (count or 0)))
    return seconds, {'policy':'effective_pages_v2', 'pages':count, 'startup_seconds':120,
                     'seconds_per_page':45, 'ceiling_seconds':7200, 'seconds':seconds,
                     **({'count_error':count_error} if count_error is not None else {})}
