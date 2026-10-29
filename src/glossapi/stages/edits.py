"""Ordered, codepoint edit receipts and composition back to the initial text."""
from __future__ import annotations

import hashlib


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def changed_span(before, after):
    """Locate a producer-reported transform; this function never assigns rules."""
    prefix = 0
    limit = min(len(before), len(after))
    while prefix < limit and before[prefix] == after[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit-prefix and before[len(before)-suffix-1] == after[len(after)-suffix-1]:
        suffix += 1
    return prefix, len(before)-suffix, after[prefix:len(after)-suffix if suffix else len(after)]


class EditRecorder:
    def __init__(self, text):
        self.initial_hash = text_hash(text)
        self.text = text
        # Interval pieces, rather than one Python object per input character.
        self.pieces = [(len(text), 0, len(text))] if text else []
        self._hint = (0, 0)
        self.edits = []
        self.replacements = {}

    def _split(self, position):
        hint_index, hint_cursor = self._hint
        first_index, cursor = (hint_index, hint_cursor) if position >= hint_cursor else (0, 0)
        for index in range(first_index, len(self.pieces)):
            length, first, last = self.pieces[index]
            if position == cursor:
                return index
            if cursor < position < cursor+length:
                delta = position-cursor
                if last-first == length:
                    left, right = (delta, first, first+delta), (length-delta, first+delta, last)
                else:
                    left, right = (delta, first, last), (length-delta, first, last)
                self.pieces[index:index+1] = [left, right]
                return index+1
            cursor += length
        if cursor == position:
            return len(self.pieces)
        raise ValueError('Edit outside current surface')

    def edit(self, rule_id, start, end, replacement, *, reason=None):
        if not 0 <= start <= end <= len(self.text):
            raise ValueError('Edit outside current surface')
        left = self._split(start)
        right = self._split(end)
        removed = self.pieces[left:right]
        if removed:
            origin = (min(piece[1] for piece in removed), max(piece[2] for piece in removed))
        else:
            edge = self.pieces[left][1] if left < len(self.pieces) else (self.pieces[-1][2] if self.pieces else 0)
            origin = (edge, edge)
        digest = text_hash(replacement)
        self.replacements[digest] = replacement
        self.edits.append({'start': start, 'end': end, 'replacement_length': len(replacement),
            'replacement_sha256': digest, 'rule_id': rule_id, 'reason': reason or rule_id,
            'initial_range': list(origin)})
        self.pieces[left:right] = [(len(replacement), *origin)] if replacement else []
        self._hint = (left, start)
        self.text = self.text[:start]+replacement+self.text[end:]

    def transform(self, rule_id, result, *, reason=None):
        if self.text != result:
            self.edit(rule_id, *changed_span(self.text, result), reason=reason)

    def receipt(self):
        return {'input_hash': self.initial_hash, 'output_hash': text_hash(self.text),
            'offset_unit': 'unicode_codepoint', 'coordinate_basis': 'ordered_current_surface',
            'edits': self.edits, 'composition_map': [list(piece) for piece in self.pieces]}


def replay_edits(text, receipt, replacements):
    if text_hash(text) != receipt['input_hash']:
        raise ValueError('Edit input hash mismatch')
    for edit in receipt['edits']:
        replacement = replacements[edit['replacement_sha256']]
        if text_hash(replacement) != edit['replacement_sha256'] or len(replacement) != edit['replacement_length']:
            raise ValueError('Edit replacement hash mismatch')
        text = text[:edit['start']]+replacement+text[edit['end']:]
    if text_hash(text) != receipt['output_hash']:
        raise ValueError('Edit output hash mismatch')
    return text


from contextlib import contextmanager
from contextvars import ContextVar
_ocr_edit_sink = ContextVar('glossapi_ocr_producer_edit_sink', default=None)


class OcrEditSink:
    def __init__(self, recorder):
        self.recorder = recorder
        self.offset = 0
        self.delta = 0

    def emit(self, rule, start, end, replacement, reason=None):
        before = self.recorder.text[self.offset+start+self.delta:self.offset+end+self.delta]
        if before != replacement:
            self.recorder.edit(rule,self.offset+start+self.delta,self.offset+end+self.delta,replacement,reason=reason)
            self.delta += len(replacement)-(end-start)


@contextmanager
def capture_ocr_edits(recorder):
    sink = OcrEditSink(recorder)
    token = _ocr_edit_sink.set(sink)
    try:
        yield sink
    finally:
        _ocr_edit_sink.reset(token)
