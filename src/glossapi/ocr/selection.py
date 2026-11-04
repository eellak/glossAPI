"""One hash-verified effective page selection for OCR routing and deadlines."""
from __future__ import annotations

import json


def native_text_is_glyph_only(inputs):
    from ..stages.clean import LAYOUT_INPUT_KINDS, layout_metadata_of
    native = next((item for item in inputs if item.kind in ('native_markdown', 'raw_markdown')), None)
    if native is None:
        return False
    text = native.path.read_bytes().decode('utf-8', 'surrogateescape')
    layout_input = next((item for item in inputs if item.kind in LAYOUT_INPUT_KINDS), None)
    if layout_input is not None:
        from ..markdown_layout import markup_free
        stripped = markup_free(text, layout_metadata_of(layout_input.path))
        if stripped is not None:
            text = stripped
    from ..corpus.phase_clean import glyph_only
    return glyph_only(text)


def effective_selection(inputs, settings, *, report=None):
    """Return (selected physical pages or None, glyph override).

    A supplied quality report wins over ranges, as in the stage. None means
    whole-document OCR without a quality report. Every input consulted for
    selection is verified, including the native text and its strip metadata.
    max_pages is a physical-page ceiling, not a count of selected pages.
    """
    from ..stages.execution import sha256_file
    from ..stages.clean import LAYOUT_INPUT_KINDS
    relevant = {'source_payload', 'download_payload', 'native_markdown', 'raw_markdown', 'page_quality', *LAYOUT_INPUT_KINDS}
    for item in inputs:
        if item.kind in relevant and sha256_file(item.path) != item.sha256:
            raise ValueError('input_hash_changed')
    quality = next((item for item in inputs if item.kind == 'page_quality'), None)
    if report is None and quality is not None:
        report = json.loads(quality.path.read_text())
    glyph = False
    if report is not None:
        selected = {int(page['page']) for page in report['pages'] if page['page_needs_ocr'] is True}
        if native_text_is_glyph_only(inputs):
            selected = set(range(1, int(report['pages_total']) + 1))
            glyph = True
    elif settings.get('page_ranges'):
        selected = {page for first, last in settings['page_ranges'] for page in range(int(first), int(last) + 1)}
    else:
        return None, False
    maximum = settings.get('max_pages')
    return sorted(page for page in selected if maximum is None or page <= maximum), glyph
