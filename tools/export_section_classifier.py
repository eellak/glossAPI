"""Trusted one-time export: the shipped scikit-learn 1.6.1 section classifier -> neutral model directory.

Maintenance script, never imported by the runtime. It runs only in the producer environment pinned in
``tools/export_section_classifier.requirements.txt`` (the ``annotate-legacy`` extra: scikit-learn 1.6.1), loads the
pickle only after its sha256 matches the trusted shipped model, refuses any pipeline shape or parameter outside the
``glossapi.section-classifier.linear-tfidf`` version-1 specification, and writes byte-reproducible files:

  manifest.json            format, version, source model hash, producer versions, the preprocessing/tf-idf/classifier
                           specification, feature and class order, and the sha256 of every file below
  vocabulary.txt           terms in feature-index order (UTF-8, one per line)
  idf.f64le                smooth IDF, float64 little-endian
  coef.f64le               classes x (terms + 2) float64 little-endian (has_table, has_list last)
  intercept.f64le          one float64 per class
  word_ranges_u<V>.txt     Python re ``\\w`` under the producer's Unicode V, as ``start end`` hex code-point ranges

Usage (from the repository root, in the producer environment):
  python tools/export_section_classifier.py [--source src/glossapi/models/section_classifier.joblib]
         [--output src/glossapi/models/section_classifier] [--source-sha256 <trusted sha256>]
The runtime reader is ``glossapi.section_model``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'src/glossapi/models/section_classifier.joblib'
OUTPUT = REPO / 'src/glossapi/models/section_classifier'
# The shipped model (127,590 bytes; model_format_probe.json, decision memo W2b section 1).
TRUSTED_SOURCE_SHA256 = '12963ebe47b38855a932649df7c75577ca43882d0a09ce3a63ef93946f490737'
PRODUCER_SKLEARN = '1.6.1'
TFIDF_PARAMETERS = dict(analyzer='word', binary=False, lowercase=True, ngram_range=(1, 1), norm='l2', preprocessor=None,
                        smooth_idf=True, stop_words=None, strip_accents=None, sublinear_tf=False,
                        token_pattern=r'(?u)\b\w\w+\b', tokenizer=None, use_idf=True, input='content')


class ExportRefused(ValueError):
    """The pipeline is not the shape the version-1 neutral format describes."""


def _require(condition, message):
    if not condition:
        raise ExportRefused(message)


def check_pipeline(model):
    """Return (vectorizer, classifier, vocabulary in feature order) or refuse."""
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import FunctionTransformer
    _require(type(model) is Pipeline and [name for name, _ in model.steps] == ['preprocessor', 'classifier'],
             'expected Pipeline[preprocessor, classifier]')
    pre, clf = model.named_steps['preprocessor'], model.named_steps['classifier']
    _require(type(pre) is ColumnTransformer and type(clf) is LinearSVC, 'expected ColumnTransformer + LinearSVC')
    _require([(name, columns) for name, _, columns in pre.transformers] ==
             [('combined_text', ['header', 'section']), ('binary', ['has_table', 'has_list'])],
             'expected transformers combined_text[header, section] and binary[has_table, has_list]')
    _require(pre.transformers[1][1] == 'passthrough', 'the binary columns must be passthrough')
    _require(pre.remainder == 'drop' and pre.transformer_weights is None, 'remainder must be drop, no weights')
    fitted = [(name, columns) for name, _, columns in pre.transformers_ if name != 'remainder']
    _require(fitted == [('combined_text', ['header', 'section']), ('binary', ['has_table', 'has_list'])],
             'fitted transformers differ from the specification')
    text = pre.named_transformers_['combined_text']
    _require(type(text) is Pipeline and [name for name, _ in text.steps] == ['combine', 'tfidf'],
             'expected Pipeline[combine, tfidf]')
    combine, tfidf = text.named_steps['combine'], text.named_steps['tfidf']
    _require(type(combine) is FunctionTransformer and combine.kw_args is None
             and (combine.func.__module__, combine.func.__name__) == ('glossapi.gloss_section_classifier', 'combine_text'),
             'the text combiner must be glossapi.gloss_section_classifier.combine_text without arguments')
    _require(type(tfidf) is TfidfVectorizer, 'expected a plain TfidfVectorizer')
    parameters = tfidf.get_params()
    for name, expected in TFIDF_PARAMETERS.items():
        _require(parameters[name] == expected, f'TfidfVectorizer {name}={parameters[name]!r}, expected {expected!r}')
    _require(parameters['dtype'] is np.float64, 'TfidfVectorizer dtype must be float64')
    width = len(tfidf.vocabulary_)
    vocabulary = [None] * width
    for term, index in tfidf.vocabulary_.items():
        _require(0 <= int(index) < width and vocabulary[int(index)] is None, 'vocabulary indices are not a permutation')
        vocabulary[int(index)] = term
    _require(all(isinstance(term, str) and term and '\n' not in term for term in vocabulary), 'bad vocabulary term')
    _require(tfidf.idf_.shape == (width,), 'idf shape')
    _require(pre.output_indices_['combined_text'] == slice(0, width) and
             pre.output_indices_['binary'] == slice(width, width + 2), 'feature layout is not [terms, has_table, has_list]')
    _require(clf.multi_class == 'ovr' and clf.fit_intercept, 'LinearSVC must be one-vs-rest with an intercept')
    _require(clf.coef_.shape == (len(clf.classes_), width + 2) and clf.intercept_.shape == (len(clf.classes_),),
             'coefficient shape')
    _require(len(clf.classes_) >= 2 and all(isinstance(c, str) for c in clf.classes_), 'classes must be strings')
    for name in ('idf', 'coef', 'intercept'):
        array = tfidf.idf_ if name == 'idf' else getattr(clf, name + '_')
        _require(np.isfinite(array).all(), f'{name} has non-finite values')
    return tfidf, clf, vocabulary


def word_ranges():
    sys.path.insert(0, str(REPO / 'src'))
    from glossapi.section_model import word_ranges as table
    return table(r'\w')


def export_pipeline(model, output, source):
    """Write the neutral directory for an already checked, trusted pipeline; return the manifest."""
    import joblib
    import numpy as np
    import sklearn
    _require(sklearn.__version__ == PRODUCER_SKLEARN, f'producer must run scikit-learn {PRODUCER_SKLEARN}, '
             f'found {sklearn.__version__}')
    tfidf, clf, vocabulary = check_pipeline(model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    version = unicodedata.unidata_version
    blobs = {'vocabulary': ('vocabulary.txt', ''.join(term + '\n' for term in vocabulary).encode('utf-8')),
             'idf': ('idf.f64le', np.ascontiguousarray(tfidf.idf_, dtype='<f8').tobytes()),
             'coef': ('coef.f64le', np.ascontiguousarray(clf.coef_, dtype='<f8').tobytes()),
             'intercept': ('intercept.f64le', np.ascontiguousarray(clf.intercept_, dtype='<f8').tobytes()),
             'word_ranges': (f'word_ranges_u{version}.txt', word_ranges().encode('ascii'))}
    names = {'manifest.json', *(name for name, _ in blobs.values())}
    stale = sorted(path.name for path in output.iterdir() if path.name not in names)
    _require(not stale, f'output directory holds files outside the model: {stale}')
    for name, data in blobs.values():
        (output / name).write_bytes(data)
    digest = lambda role: hashlib.sha256(blobs[role][1]).hexdigest()
    files = {
        'vocabulary': dict(path=blobs['vocabulary'][0], encoding='utf-8', lines=len(vocabulary), order='feature index',
                           sha256=digest('vocabulary')),
        'idf': dict(path=blobs['idf'][0], dtype='float64-le', shape=[len(vocabulary)], sha256=digest('idf')),
        'coef': dict(path=blobs['coef'][0], dtype='float64-le', shape=list(clf.coef_.shape), sha256=digest('coef')),
        'intercept': dict(path=blobs['intercept'][0], dtype='float64-le', shape=list(clf.intercept_.shape),
                          sha256=digest('intercept')),
        'word_ranges': dict(path=blobs['word_ranges'][0], encoding='ascii', pattern=r'\w', unicodedata=version,
                            lines=blobs['word_ranges'][1].count(b'\n'), format='start end (inclusive, lowercase hex)',
                            sha256=digest('word_ranges')),
    }
    manifest = {
        'format': 'glossapi.section-classifier.linear-tfidf', 'format_version': 1,
        'exporter': 'tools/export_section_classifier.py',
        'source_model': source,
        'producer': dict(sklearn=sklearn.__version__, joblib=joblib.__version__, numpy=np.__version__,
                         python=platform.python_version(), unicodedata=version, platform=platform.platform()),
        'inputs': {'columns': ['header', 'section', 'has_table', 'has_list'], 'null_text': "''",
                   'boolean_columns': 'cast to int (False=0, True=1) before scoring'},
        'text': {'combine': "header + ' ' + section (after null -> '')", 'lowercase': 'python str.lower()',
                 'strip_accents': None, 'unicode_normalization': None, 'token_pattern': TFIDF_PARAMETERS['token_pattern'],
                 'token_regex_flavour': f'python re (str), unicodedata {version}; table in files.word_ranges',
                 'ngram_range': [1, 1], 'oov': 'ignored'},
        'tfidf': {'tf': 'raw count', 'idf': 'smooth (stored values)', 'sublinear_tf': False,
                  'norm': 'l2 over the text features only, zero row stays zero'},
        'features': {'order': ['vocabulary[0..n)', 'has_table', 'has_list'], 'n_text': len(vocabulary),
                     'n_total': len(vocabulary) + 2},
        'classifier': {'kind': 'linear-ovr', 'classes': [str(c) for c in clf.classes_],
                       'decision': 'x . coef[k] + intercept[k]',
                       'label': 'classes[argmax(decision)]; ties -> lowest index (numpy argmax, as sklearn)'},
        'files': files,
    }
    (output / 'manifest.json').write_bytes((json.dumps(manifest, ensure_ascii=False, indent=1) + '\n').encode('utf-8'))
    return manifest


def export_file(source=SOURCE, output=OUTPUT, source_sha256=TRUSTED_SOURCE_SHA256):
    data = Path(source).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    # The pickle executes code on load: open only the trusted bytes.
    _require(digest == source_sha256, f'{source} sha256 {digest} is not the trusted {source_sha256}')
    import joblib
    sys.path.insert(0, str(REPO / 'src'))  # pickles reference glossapi.gloss_section_classifier.combine_text
    model = joblib.load(source)
    return export_pipeline(model, output, dict(path='glossapi/models/section_classifier.joblib', sha256=digest,
                                               bytes=len(data)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--source', default=str(SOURCE))
    parser.add_argument('--output', default=str(OUTPUT))
    parser.add_argument('--source-sha256', default=TRUSTED_SOURCE_SHA256)
    args = parser.parse_args(argv)
    manifest = export_file(args.source, args.output, args.source_sha256)
    print(json.dumps({key: manifest[key] for key in ('source_model', 'producer')}, ensure_ascii=False))
    print(json.dumps({name: os.path.getsize(Path(args.output) / name) for name in sorted(os.listdir(args.output))}))


if __name__ == '__main__':
    main()
