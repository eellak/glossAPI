"""Neutral section-classifier model and its own inference (K2 decision W2b, option iii).

Format ``glossapi.section-classifier.linear-tfidf`` version 1: a directory holding ``manifest.json`` and raw data
files (vocabulary, IDF, coefficients, intercepts and the producer's word-character table), written once from the
shipped scikit-learn 1.6.1 pipeline by ``tools/export_section_classifier.py``. Loading reads JSON, UTF-8 text and
little-endian float64 only: no pickle, scikit-learn, scipy or joblib. Every declared file is hash-checked before use,
and any manifest value outside the specification below is refused.

Inference reproduces the shipped pipeline on the prediction path:
``text = header + ' ' + section`` (nulls become ``''``), Python ``str.lower()``, tokens = maximal runs of at least two
word characters (``(?u)\\b\\w\\w+\\b``, with ``\\w`` as Python ``re`` defined it under the producer's Unicode
version), raw counts times the stored smooth IDF, L2 normalisation over the text block only (an empty row stays zero),
then ``has_table`` and ``has_list`` as raw values; decision ``x . coef[k] + intercept[k]``; label = class with the
largest decision, ties to the lowest class index (as numpy argmax / scikit-learn). The sums run in Python floats in
feature-index order, so scores do not depend on BLAS or the platform.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

FORMAT = 'glossapi.section-classifier.linear-tfidf'
FORMAT_VERSION = 1
MANIFEST = 'manifest.json'
PACKAGED_DIRECTORY = Path(__file__).resolve().parent / 'models' / 'section_classifier'
FILE_ROLES = ('vocabulary', 'idf', 'coef', 'intercept', 'word_ranges')
INPUT_COLUMNS = ('header', 'section', 'has_table', 'has_list')
TOKEN_PATTERN = r'(?u)\b\w\w+\b'
# The specification this reader implements. A manifest that says anything else is refused, never guessed at.
TEXT_SPEC = {'combine': "header + ' ' + section (after null -> '')", 'lowercase': 'python str.lower()',
             'strip_accents': None, 'unicode_normalization': None, 'token_pattern': TOKEN_PATTERN,
             'ngram_range': [1, 1], 'oov': 'ignored'}
TFIDF_SPEC = {'tf': 'raw count', 'idf': 'smooth (stored values)', 'sublinear_tf': False,
              'norm': 'l2 over the text features only, zero row stays zero'}
INPUT_SPEC = {'columns': list(INPUT_COLUMNS), 'null_text': "''",
              'boolean_columns': 'cast to int (False=0, True=1) before scoring'}
CLASSIFIER_SPEC = {'kind': 'linear-ovr', 'decision': 'x . coef[k] + intercept[k]',
                   'label': 'classes[argmax(decision)]; ties -> lowest index (numpy argmax, as sklearn)'}


class NeutralModelError(ValueError):
    """The neutral model directory is missing, tampered with, or outside the version-1 specification."""


def manifest_file(path: str | Path) -> Path | None:
    """The manifest of a neutral model given as its directory or its manifest file; None for anything else."""
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / MANIFEST
        return candidate if candidate.is_file() else None
    return candidate if candidate.suffix == '.json' and candidate.is_file() else None


def is_neutral(path: str | Path) -> bool:
    """Neutral models are directories or ``.json`` manifests; any other file (``*.joblib``) is a legacy pickle."""
    candidate = Path(path)
    return candidate.is_dir() or candidate.suffix == '.json'


def stage_input(path: str | Path) -> Path:
    """The single file that stands for a model in stage inputs: the manifest of a neutral model, else the path."""
    found = manifest_file(path)
    return found if found is not None else Path(path)


def model_format(path: str | Path) -> str:
    return f'neutral-v{FORMAT_VERSION}' if is_neutral(path) else 'joblib'


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(path: Path) -> tuple[dict[str, Any], Path]:
    found = manifest_file(path)
    if found is None:
        raise NeutralModelError(f'no neutral section-classifier manifest at {path}')
    try:
        manifest = json.loads(found.read_bytes().decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NeutralModelError(f'unreadable manifest {found}: {error}') from error
    if not isinstance(manifest, dict) or manifest.get('format') != FORMAT:
        raise NeutralModelError(f'{found}: format {manifest.get("format") if isinstance(manifest, dict) else None!r} '
                                f'is not {FORMAT!r}')
    version = manifest.get('format_version')
    if type(version) is not int or version != FORMAT_VERSION:
        raise NeutralModelError(f'{found}: unsupported format_version {version!r} (this reader implements '
                                f'{FORMAT_VERSION})')
    files = manifest.get('files')
    if not isinstance(files, dict) or sorted(files) != sorted(FILE_ROLES):
        raise NeutralModelError(f'{found}: files must be exactly {sorted(FILE_ROLES)}')
    for role, entry in files.items():
        name = entry.get('path') if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..', MANIFEST):
            raise NeutralModelError(f'{found}: file {role!r} must be a plain file name beside the manifest')
        if not isinstance(entry.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']):
            raise NeutralModelError(f'{found}: file {role!r} has no sha256')
    return manifest, found


def model_files(path: str | Path) -> list[Path]:
    """Every file that defines the model (manifest first, then the declared files); ``[path]`` for a pickle."""
    if not is_neutral(path):
        return [Path(path)]
    found = manifest_file(path)
    if found is None:
        return []
    manifest, found = _manifest(found)
    return [found, *(found.parent / manifest['files'][role]['path'] for role in FILE_ROLES)]


def manifest_summary(path: str | Path) -> dict[str, Any]:
    """Manifest hash and the source model it was exported from (for lineage)."""
    manifest, found = _manifest(Path(path))
    return {'model_format': model_format(path), 'manifest_sha256': _sha256(found.read_bytes()),
            'source_model_sha256': manifest.get('source_model', {}).get('sha256')}


def freeze_beside(source: str | Path, frozen_manifest: str | Path) -> list[Path]:
    """Copy the declared files of ``source`` next to an already frozen copy of its manifest, verifying hashes.

    Stage inputs are frozen file by file; the manifest is the stage input and carries every data file's hash,
    so the data files are frozen against the frozen manifest, not against the mutable source directory."""
    frozen_manifest = Path(frozen_manifest)
    manifest, _ = _manifest(frozen_manifest)
    origin = manifest_file(source)
    if origin is None:
        raise NeutralModelError(f'no neutral section-classifier manifest at {source}')
    copied = []
    for role in FILE_ROLES:
        entry = manifest['files'][role]
        data = (origin.parent / entry['path']).read_bytes()
        if _sha256(data) != entry['sha256']:
            raise NeutralModelError(f'{origin.parent / entry["path"]}: sha256 does not match the frozen manifest')
        target = frozen_manifest.parent / entry['path']
        target.write_bytes(data)
        copied.append(target)
    return copied


def _word_ranges(text: str) -> list[tuple[int, int]]:
    """The exported word table as ascending, non-overlapping inclusive ``(start, end)`` code-point ranges."""
    ranges = []
    previous = -1
    lines = text[:-1].split('\n') if text.endswith('\n') else []
    for number, line in enumerate(lines, start=1):
        fields = line.split(' ')
        if len(fields) != 2 or not all(re.fullmatch(r'[0-9a-f]+', field) for field in fields):
            raise NeutralModelError(f'word_ranges line {number}: expected "<start hex> <end hex>"')
        start, end = int(fields[0], 16), int(fields[1], 16)
        if not previous < start <= end <= 0x10FFFF:
            raise NeutralModelError(f'word_ranges line {number}: ranges must ascend without overlap')
        previous = end
        ranges.append((start, end))
    if not ranges:
        raise NeutralModelError('word_ranges is empty or does not end with a newline')
    return ranges


def _char_class(ranges: Sequence[tuple[int, int]]) -> str:
    if not ranges:
        return '[^\\x00-\\U0010ffff]'  # the empty class: matches no character
    code = lambda value: '\\U%08x' % value
    return '[' + ''.join(code(a) if a == b else code(a) + '-' + code(b) for a, b in ranges) + ']'


ASTRAL = '[\\U00010000-\\U0010ffff]'


class TableTokenizer:
    """Maximal runs of at least two word characters of an exported table: ``(?u)\\b\\w\\w+\\b`` with that table's ``\\w``.

    Python ``re`` tests a character class by a 65,536-bit map for the code points up to U+FFFF and then, on a miss,
    by scanning the class's ranges above U+FFFF one by one. With the whole table as one class, every non-word BMP
    character (spaces, punctuation) paid for the ~300 astral ranges, which made 3.12+ tokenization 4-8x slower than
    the interpreter's own ``\\w`` (K2B3-G3). The table is therefore split at U+FFFF: a text without astral characters
    (found by one range test per character) is tokenized with the BMP class alone, which is the table on such a
    text; any other text with an alternation that consults the astral ranges only for astral characters. Both give
    exactly the runs of the single whole-table class."""

    def __init__(self, ranges: Sequence[tuple[int, int]]):
        self.bmp = [(start, min(end, 0xFFFF)) for start, end in ranges if start <= 0xFFFF]
        self.astral = [(max(start, 0x10000), end) for start, end in ranges if end > 0xFFFF]
        word = _char_class(self.bmp)
        self._bmp_runs = re.compile(word + word + '+').findall
        self._has_astral = re.compile(ASTRAL).search
        self._mixed_runs = re.compile(f'(?:{word}|(?={ASTRAL}){_char_class(self.astral)}){{2,}}').findall

    def __call__(self, text: str) -> list[str]:
        return self._mixed_runs(text) if self._has_astral(text) else self._bmp_runs(text)


def word_ranges(pattern: str = r'\w') -> str:
    """The current interpreter's ``re`` word class as ``start end`` hex lines (the exported table's format)."""
    match = re.compile(pattern).match
    lines, start = [], None
    for value in range(0x110000):
        inside = match(chr(value)) is not None
        if inside and start is None:
            start = value
        elif not inside and start is not None:
            lines.append(f'{start:x} {value - 1:x}\n')
            start = None
    if start is not None:
        lines.append(f'{start:x} 10ffff\n')
    return ''.join(lines)


def _null(value: Any) -> bool:
    # None, float NaN and pandas.NA (whose truth value raises) without importing pandas.
    return value is None or (isinstance(value, float) and math.isnan(value)) or type(value).__name__ == 'NAType'


def _text(value: Any) -> str:
    if _null(value):
        return ''
    if not isinstance(value, str):
        raise TypeError(f'section text must be str or null, not {type(value).__name__}')
    return value


def _flag(value: Any) -> float:
    number = float(value)  # bool -> 0.0/1.0; the product path casts both flags to int beforehand
    if not math.isfinite(number):
        raise ValueError(f'has_table/has_list must be finite, got {value!r}')
    return number


class NeutralSectionModel:
    """A loaded, verified version-1 model. ``predict``/``decision_function`` mirror the scikit-learn pipeline."""

    def __init__(self, path: str | Path):
        manifest, found = _manifest(Path(path))
        self.path = found.parent
        self.manifest = manifest
        self.manifest_sha256 = _sha256(found.read_bytes())
        data = {}
        for role in FILE_ROLES:
            entry = manifest['files'][role]
            target = found.parent / entry['path']
            try:
                content = target.read_bytes()
            except FileNotFoundError as error:
                raise NeutralModelError(f'{target}: declared model file is missing') from error
            if _sha256(content) != entry['sha256']:
                raise NeutralModelError(f'{target}: sha256 does not match the manifest (tampered or corrupt model)')
            data[role] = content
        for key, expected in (('inputs', INPUT_SPEC), ('text', TEXT_SPEC), ('tfidf', TFIDF_SPEC)):
            given = {name: value for name, value in (manifest.get(key) or {}).items() if name != 'token_regex_flavour'}
            if given != expected:
                raise NeutralModelError(f'{found}: {key} specification {given!r} differs from version 1 {expected!r}')
        classifier = manifest.get('classifier') or {}
        if {name: classifier.get(name) for name in CLASSIFIER_SPEC} != CLASSIFIER_SPEC:
            raise NeutralModelError(f'{found}: classifier specification differs from version 1')
        classes = classifier.get('classes')
        if not (isinstance(classes, list) and classes and all(isinstance(c, str) for c in classes)
                and len(set(classes)) == len(classes)):
            raise NeutralModelError(f'{found}: classes must be distinct strings')
        try:
            vocabulary_text = data['vocabulary'].decode('utf-8')
        except UnicodeDecodeError as error:
            raise NeutralModelError('vocabulary.txt is not UTF-8') from error
        if not vocabulary_text.endswith('\n'):
            raise NeutralModelError('vocabulary.txt must end with a newline')
        vocabulary = vocabulary_text[:-1].split('\n')
        features = manifest.get('features') or {}
        n_text = len(vocabulary)
        if (features.get('order') != ['vocabulary[0..n)', 'has_table', 'has_list'] or features.get('n_text') != n_text
                or features.get('n_total') != n_text + 2 or manifest['files']['vocabulary'].get('lines') != n_text
                or len(set(vocabulary)) != n_text or not all(vocabulary)):
            raise NeutralModelError(f'{found}: vocabulary and feature declaration disagree')
        arrays = {}
        for role, shape in (('idf', (n_text,)), ('coef', (len(classes), n_text + 2)), ('intercept', (len(classes),))):
            entry = manifest['files'][role]
            if entry.get('dtype') != 'float64-le' or entry.get('shape') != list(shape) or len(data[role]) != 8 * math.prod(shape):
                raise NeutralModelError(f'{role}: expected float64-le with shape {list(shape)}')
            array = np.frombuffer(data[role], dtype='<f8').reshape(shape)
            if not np.isfinite(array).all():
                raise NeutralModelError(f'{role}: non-finite value')
            arrays[role] = array
        producer = str((manifest.get('producer') or {}).get('unicodedata'))
        word_entry = manifest['files']['word_ranges']
        if word_entry.get('unicodedata') != producer or word_entry.get('pattern') != r'\w':
            raise NeutralModelError(f'{found}: the word table must be the producer\'s Python re \\w')
        try:
            word_table = _word_ranges(data['word_ranges'].decode('ascii'))
        except UnicodeDecodeError as error:
            raise NeutralModelError('word_ranges is not ASCII') from error
        self.classes = list(classes)
        self.classes_ = np.array(self.classes, dtype=object)
        self.vocabulary = {term: index for index, term in enumerate(vocabulary)}
        self.n_text = n_text
        self.idf = arrays['idf']
        self.coef = arrays['coef']
        self.intercept = arrays['intercept']
        self.producer_unicodedata = producer
        # Python's re \w is defined by the interpreter's Unicode database; when it is the producer's, the
        # standard token pattern is the exported table (asserted by the tests). Otherwise use the table itself.
        if unicodedata.unidata_version == producer:
            self.tokenizer = 'python-re-word'
            self._findall = re.compile(TOKEN_PATTERN).findall
        else:
            self.tokenizer = 'exported-word-table'
            self._findall = TableTokenizer(word_table)
        self._idf = self.idf.tolist()
        self._columns = [self.coef[:, j].tolist() for j in range(n_text + 2)]
        self._intercept = self.intercept.tolist()

    def tokens(self, text: str) -> list[str]:
        return self._findall(text.lower())

    def features(self, header: Any, section: Any, has_table: Any, has_list: Any) -> tuple[list[int], list[float]]:
        """Sparse feature row: indices ascending, values as the pipeline computes them."""
        counts: dict[int, int] = {}
        for token in self.tokens(_text(header) + ' ' + _text(section)):
            index = self.vocabulary.get(token)
            if index is not None:
                counts[index] = counts.get(index, 0) + 1
        indices = sorted(counts)
        values = [counts[index] * self._idf[index] for index in indices]
        square = 0.0
        for value in values:
            square += value * value
        if square > 0.0:
            norm = math.sqrt(square)
            values = [value / norm for value in values]
        for index, flag in ((self.n_text, _flag(has_table)), (self.n_text + 1, _flag(has_list))):
            if flag != 0.0:
                indices.append(index)
                values.append(flag)
        return indices, values

    def decision_row(self, indices: Sequence[int], values: Sequence[float]) -> list[float]:
        scores = [0.0] * len(self.classes)
        for index, value in zip(indices, values):
            column = self._columns[index]
            for k in range(len(scores)):
                scores[k] += value * column[k]
        return [score + intercept for score, intercept in zip(scores, self._intercept)]

    def _rows(self, X: Any) -> Iterable[tuple[Any, Any, Any, Any]]:
        try:
            columns = [X[name] for name in INPUT_COLUMNS]
        except (KeyError, TypeError, IndexError):
            return X  # an iterable of (header, section, has_table, has_list)
        return zip(*columns)

    def decision_function(self, X: Any) -> np.ndarray:
        rows = [self.decision_row(*self.features(*row)) for row in self._rows(X)]
        return np.array(rows, dtype=np.float64).reshape(len(rows), len(self.classes))

    def predict(self, X: Any) -> np.ndarray:
        labels = []
        for row in self._rows(X):
            scores = self.decision_row(*self.features(*row))
            best = 0
            for k in range(1, len(scores)):
                if scores[k] > scores[best]:
                    best = k
            labels.append(self.classes[best])
        return np.array(labels, dtype=object)


def load(path: str | Path) -> NeutralSectionModel:
    return NeutralSectionModel(path)
