"""K5-C: the exported-word-table tokenizer (Python 3.12+) is exact and as fast as the interpreter's own ``\\w``.

The reference is the K2 implementation it replaces: the whole table as one ``re`` character class, maximal runs,
runs shorter than two characters dropped. On the producer's Unicode (3.11) both also equal the scikit-learn token
pattern ``(?u)\\b\\w\\w+\\b`` itself.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
import unicodedata

import pandas as pd
import pytest

import glossapi
from glossapi import section_model
from glossapi.section_model import NeutralModelError, TableTokenizer

NEUTRAL = Path(glossapi.__file__).resolve().parent / 'models' / 'section_classifier'
MANIFEST = json.loads((NEUTRAL / 'manifest.json').read_text(encoding='utf-8'))
TABLE_FILE = NEUTRAL / MANIFEST['files']['word_ranges']['path']
TABLE = section_model._word_ranges(TABLE_FILE.read_text(encoding='ascii'))
PRODUCER_UNICODE = unicodedata.unidata_version == MANIFEST['producer']['unicodedata']


def reference(ranges):
    """The K2 tokenizer: one character class for the whole table (slow above U+FFFF, but the definition)."""
    runs = re.compile(section_model._char_class(ranges) + '+').findall
    return lambda text: [token for token in runs(text) if len(token) > 1]


def boundary_characters(ranges):
    """Every table range's first and last code point and their outside neighbours (surrogates included)."""
    points = set()
    for start, end in ranges:
        points.update(value for value in (start - 1, start, end, end + 1) if 0 <= value <= 0x10FFFF)
    return sorted(points)


def fuzz_texts(ranges, count=3000, seed=20261104):
    rng = random.Random(seed)
    pool = [chr(c) for c in boundary_characters(ranges)]
    pool += list('αβγδεζηθικλμνξοπρσςτυφχψωάέήίόύώϊΐΆΈ ἡᾠδὴᾳᾼ abcXYZ_0123456789 \t\n.,;:!?-()[]«»·΄΅ͅ')
    pool += ['\U0001d465', '\U0001d7ce', '\U00010000', '\U00020000', '\U0001f600', '\U000e0001', '\U0010ffff',
             '\ud800', '\udfff', '﻿', '￿', ' ', ' ']
    texts = []
    for _ in range(count):
        texts.append(''.join(rng.choice(pool) for _ in range(rng.randrange(0, 64))))
    return texts


def per_code_point_text(start, stop):
    """Each code point doubled and followed by a space: a word character gives exactly one token of two."""
    return ''.join(chr(value) * 2 + ' ' for value in range(start, stop))


def test_table_tokenizer_equals_the_whole_table_class_on_every_code_point():
    fast, slow = TableTokenizer(TABLE), reference(TABLE)
    contiguous = ''.join(chr(value) for value in range(0x110000))
    assert fast(contiguous) == slow(contiguous)
    for start in range(0, 0x110000, 0x22000):  # in slices: the reference is the slow path being replaced
        text = per_code_point_text(start, min(start + 0x22000, 0x110000))
        assert fast(text) == slow(text)
    # The same text without any astral character takes the BMP-only expression.
    bmp = per_code_point_text(0, 0x10000)
    assert fast(bmp) == slow(bmp) and not fast._has_astral(bmp)


def test_table_tokenizer_equals_the_whole_table_class_on_mixed_texts():
    fast, slow = TableTokenizer(TABLE), reference(TABLE)
    texts = fuzz_texts(TABLE)
    assert sum(1 for text in texts if fast._has_astral(text)) > 1000  # both expressions are exercised
    for text in texts:
        assert fast(text) == slow(text), ascii(text)
    for text in ('', ' ', 'α', 'αβ', '\U0001d465', '\U0001d465\U0001d465', 'a\U0001d465', '\U0001d465a b',
                 'x\U0001f600y', 'ab\U0001f600cd', 'ab\ud800cd', 'βιβλιογραφία_αναφορές 𝑥𝑦 😀😀 τέλος'):
        assert fast(text) == slow(text), ascii(text)


@pytest.mark.parametrize('ranges', [
    [(0x61, 0x7A)],                                   # BMP only: the astral class is empty
    [(0x10000, 0x1FFFF)],                             # astral only: the BMP class is empty
    [(0x30, 0x39), (0xFF00, 0x10010), (0x1D400, 0x1D7FF)],  # a range straddling U+FFFF is split
    [(0x0, 0x10FFFF)],                                # every code point
    [(0x5F, 0x5F), (0xFFFF, 0xFFFF), (0x10000, 0x10000), (0x10FFFF, 0x10FFFF)],  # single points at the seams
])
def test_table_tokenizer_handles_any_valid_table(ranges):
    fast, slow = TableTokenizer(ranges), reference(ranges)
    assert all(end <= 0xFFFF for _, end in fast.bmp) and all(start >= 0x10000 for start, _ in fast.astral)
    covered = sorted(fast.bmp + fast.astral)
    assert sum(end - start + 1 for start, end in covered) == sum(end - start + 1 for start, end in ranges)
    for text in fuzz_texts(ranges, count=400, seed=len(ranges)):
        assert fast(text) == slow(text), ascii(text)


def test_split_classes_cover_the_packaged_table_exactly():
    tokenizer = TableTokenizer(TABLE)
    assert all(end <= 0xFFFF for _, end in tokenizer.bmp) and all(start >= 0x10000 for start, _ in tokenizer.astral)
    members = lambda ranges: sum(end - start + 1 for start, end in ranges)
    assert members(tokenizer.bmp) + members(tokenizer.astral) == members(TABLE)
    assert len(tokenizer.astral) > 100  # why the single class was slow: astral ranges scanned on every BMP miss


@pytest.mark.skipif(not PRODUCER_UNICODE, reason='the interpreter Unicode differs from the producer table')
def test_table_tokenizer_equals_the_sklearn_token_pattern_on_the_producer_unicode():
    fast, native = TableTokenizer(TABLE), re.compile(section_model.TOKEN_PATTERN).findall
    for start in range(0, 0x110000, 0x22000):
        text = per_code_point_text(start, min(start + 0x22000, 0x110000))
        assert fast(text) == native(text)
    for text in fuzz_texts(TABLE, count=1000):
        assert fast(text) == native(text), ascii(text)


def test_model_uses_the_table_tokenizer_off_the_producer_unicode(monkeypatch):
    monkeypatch.setattr(section_model.unicodedata, 'unidata_version', '0.0-test')
    model = section_model.load(NEUTRAL)
    assert model.tokenizer == 'exported-word-table' and isinstance(model._findall, TableTokenizer)
    assert model._findall.bmp + model._findall.astral == TableTokenizer(TABLE).bmp + TableTokenizer(TABLE).astral
    assert model.tokens('ΒΙΒΛΙΟΓΡΑΦΙΑ 𝑥𝑦 a_b c') == ['βιβλιογραφια', '𝑥𝑦', 'a_b']


def corrupt_table(tmp_path, name, data):
    target = tmp_path / name
    shutil.copytree(NEUTRAL, target)
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    entry = manifest['files']['word_ranges']
    (target / entry['path']).write_bytes(data)
    entry['sha256'] = hashlib.sha256(data).hexdigest()
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return target


@pytest.mark.parametrize('data, message', [
    ('30 39\nάλφα\n'.encode('utf-8'), 'not ASCII'),
    (b'30 39\n20 2f\n', 'ascend'),
    (b'30 39\n38 41\n', 'ascend'),
    (b'30 110000\n', 'ascend'),
    (b'30 39', 'empty or does not end'),
    (b'', 'empty or does not end'),
    (b'30-39\n', 'expected'),
    (b'30 3G\n', 'expected'),
])
def test_malformed_word_tables_are_refused_on_every_interpreter(tmp_path, data, message):
    model = corrupt_table(tmp_path, 'bad', data)
    with pytest.raises(NeutralModelError, match=message):
        section_model.load(model)


def test_cohort409_table_path_scores_are_bitwise_the_default_path(monkeypatch):
    """On the 409-section cohort the table path gives the default path's labels and the same float64 scores."""
    path = os.environ.get('GLOSSAPI_W2B_COHORT')
    if not path:
        pytest.skip('GLOSSAPI_W2B_COHORT (the rebuilt 409-section cohort parquet) is not set')
    cohort = pd.read_parquet(path)
    X = cohort[['header', 'section', 'has_table', 'has_list']].copy()
    for column in ('header', 'section'):
        X[column] = X[column].fillna('')
    for column in ('has_table', 'has_list'):
        X[column] = X[column].astype(int)
    default = section_model.load(NEUTRAL)
    monkeypatch.setattr(section_model.unicodedata, 'unidata_version', '0.0-test')
    table = section_model.load(NEUTRAL)
    assert table.tokenizer == 'exported-word-table'
    assert table.predict(X).tolist() == default.predict(X).tolist() == cohort.node_model_label.tolist()
    assert table.decision_function(X).tobytes() == default.decision_function(X).tobytes()


def test_table_tokenizer_is_not_slower_than_the_interpreter_word_class():
    """K2B3-G3 regression guard: the single-class table was 4-8x slower than re's own \\w on Greek prose."""
    rng = random.Random(7)
    words = ['βιβλιογραφία', 'Αθήνα,', 'εκδόσεις.', '(2001)', 'σελ.', '12–14;', 'και', '«Ἡ ψυχή»', 'ISBN', '978-960']
    text = ' '.join(rng.choice(words) for _ in range(150_000)).lower()
    fast, native = TableTokenizer(TABLE), re.compile(section_model.TOKEN_PATTERN).findall
    best = {'table': float('inf'), 'native': float('inf')}
    for _ in range(5):
        for name, function in (('table', fast), ('native', native)):
            start = time.perf_counter(); function(text); best[name] = min(best[name], time.perf_counter() - start)
    assert best['table'] <= 2.5 * best['native'], best
