"""K2 W2b: the neutral section-classifier model and its own inference, with scikit-learn 1.6.1 as the oracle.

The oracle is the shipped joblib pipeline scored by the pinned scikit-learn of the `reference` group. Without it the
oracle tests skip (or fail under GLOSSAPI_REFERENCE_REQUIRED=1); the model-only tests always run.

The 409-section cohort parity test reads the whole-cohort annotate input rebuilt as in the W2b decision evidence
(`GLOSSAPI_W2B_COHORT=<cohort409.parquet>`, columns header/section/has_table/has_list/node_model_label).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unicodedata

import numpy as np
import pandas as pd
import pytest

import glossapi
from glossapi import section_model
from glossapi.section_model import NeutralModelError, NeutralSectionModel
from reference_oracles import oracle as reference_oracle

REPO = Path(__file__).resolve().parents[1]
PACKAGED = Path(glossapi.__file__).resolve().parent / 'models'
NEUTRAL = PACKAGED / 'section_classifier'
JOBLIB = PACKAGED / 'section_classifier.joblib'
EDGE_CASES = {
    'empty': ('', '', 0, 0), 'null_header': (None, 'Βιβλιογραφία', 0, 0), 'nan_section': ('Βιβλιογραφία', float('nan'), 0, 0),
    'oov_only': ('zzqx', 'qqqzz ψψψψω', 0, 0), 'single_char_tokens': ('a', 'β γ 1 2', 0, 0),
    'table_only': ('', '', 1, 0), 'list_only': ('', '', 0, 1), 'both_flags_no_text': ('', '', 1, 1),
    'upper_final_sigma': ('ΒΙΒΛΙΟΓΡΑΦΙΚΕΣ ΑΝΑΦΟΡΕΣ', 'ΠΗΓΕΣ ΚΑΙ ΣΗΜΕΙΩΣΕΙΣ', 0, 1),
    'polytonic_nfc': ('Βιβλιογραφία', unicodedata.normalize('NFC', 'ἡ ᾠδὴ τῆς ψυχῆς ᾳ ᾼ'), 0, 0),
    'polytonic_nfd_ypogegrammeni': ('Βιβλιογραφία', unicodedata.normalize('NFD', 'ἡ ᾠδὴ τῆς ψυχῆς ᾳ ᾼ'), 0, 0),
    'digits_index': ('Περιεχόμενα', '\n'.join(f'| Κεφάλαιο {i} | {i * 7} |' for i in range(1, 12)), 1, 0),
    'underscore_mixed': ('ref_1', 'doi:10.1000/xyz_ab ISBN 978-960 βιβλιογραφία_αναφορές', 0, 0),
    'repeated_terms': ('Βιβλιογραφία', ' '.join(['βιβλιογραφία', 'αθήνα', 'εκδόσεις'] * 40), 0, 0),
}


def frame(cases):
    rows = [dict(header=h, section=s, has_table=t, has_list=l) for h, s, t, l in cases]
    return pd.DataFrame(rows, columns=['header', 'section', 'has_table', 'has_list'])


def prepared(df):
    """The product path's prepare(): nulls -> '' and flags -> int, before the model sees the rows."""
    df = df.copy()
    df['header'] = df['header'].fillna('')
    df['section'] = df['section'].fillna('')
    df['has_table'] = df['has_table'].astype(int)
    df['has_list'] = df['has_list'].astype(int)
    return df[['header', 'section', 'has_table', 'has_list']]


@pytest.fixture(scope='module')
def neutral():
    return section_model.load(NEUTRAL)


@pytest.fixture(scope='module')
def sklearn_model():
    reference_oracle('sklearn')
    import joblib
    return joblib.load(JOBLIB)


def write_model(directory, *, vocabulary=None, idf=None, coef=None, intercept=None, classes=None, manifest_edit=None):
    """A neutral model directory with chosen weights and correct hashes (built from the packaged manifest)."""
    directory.mkdir(parents=True)
    manifest = json.loads((NEUTRAL / 'manifest.json').read_text(encoding='utf-8'))
    base = section_model.load(NEUTRAL)
    vocabulary = list(base.vocabulary) if vocabulary is None else vocabulary
    classes = list(base.classes) if classes is None else classes
    arrays = {'idf': np.asarray(base.idf if idf is None else idf, dtype='<f8'),
              'coef': np.asarray(base.coef if coef is None else coef, dtype='<f8'),
              'intercept': np.asarray(base.intercept if intercept is None else intercept, dtype='<f8')}
    blobs = {'vocabulary': ''.join(term + '\n' for term in vocabulary).encode('utf-8'),
             'word_ranges': (NEUTRAL / manifest['files']['word_ranges']['path']).read_bytes(),
             **{role: array.tobytes() for role, array in arrays.items()}}
    for role, data in blobs.items():
        entry = manifest['files'][role]
        (directory / entry['path']).write_bytes(data)
        entry['sha256'] = hashlib.sha256(data).hexdigest()
        if role in arrays:
            entry['shape'] = list(arrays[role].shape)
    manifest['files']['vocabulary']['lines'] = len(vocabulary)
    manifest['features'].update(n_text=len(vocabulary), n_total=len(vocabulary) + 2)
    manifest['classifier']['classes'] = classes
    if manifest_edit:
        manifest_edit(manifest)
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return directory


# ---------------------------------------------------------------- format and loading (always run)

def test_packaged_neutral_model_matches_its_manifest_and_source(neutral):
    manifest = json.loads((NEUTRAL / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['format'] == 'glossapi.section-classifier.linear-tfidf' and manifest['format_version'] == 1
    assert manifest['source_model']['sha256'] == hashlib.sha256(JOBLIB.read_bytes()).hexdigest()
    assert manifest['source_model']['bytes'] == JOBLIB.stat().st_size
    assert manifest['producer']['sklearn'] == '1.6.1'
    assert sorted(path.name for path in NEUTRAL.iterdir()) == sorted(
        ['manifest.json', *(entry['path'] for entry in manifest['files'].values())])
    for entry in manifest['files'].values():
        assert hashlib.sha256((NEUTRAL / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    assert neutral.classes == ['άλλο', 'β', 'π']
    assert neutral.n_text == 2000 and neutral.coef.shape == (3, 2002) and neutral.intercept.shape == (3,)
    assert section_model.model_files(NEUTRAL) == [NEUTRAL / 'manifest.json', *(NEUTRAL / manifest['files'][role]['path']
                                                                             for role in section_model.FILE_ROLES)]
    assert section_model.model_files(JOBLIB) == [JOBLIB]
    assert section_model.stage_input(NEUTRAL) == NEUTRAL / 'manifest.json' and section_model.stage_input(JOBLIB) == JOBLIB


def test_word_table_is_python_re_word_under_the_producer_unicode(neutral):
    """Tokenization is defined by the exported table; on the producer's Unicode it is exactly re's \\w."""
    entry = neutral.manifest['files']['word_ranges']
    assert entry['unicodedata'] == neutral.manifest['producer']['unicodedata'] == '14.0.0'
    if unicodedata.unidata_version != entry['unicodedata']:
        pytest.skip(f'interpreter Unicode {unicodedata.unidata_version} differs; the table path is tested separately')
    assert neutral.tokenizer == 'python-re-word'
    assert section_model.word_ranges() == (NEUTRAL / entry['path']).read_text(encoding='ascii')


def test_table_tokenizer_equals_the_fast_path(monkeypatch, neutral):
    """Interpreters with another Unicode (3.12+) tokenize with the exported table; same tokens here."""
    monkeypatch.setattr(section_model.unicodedata, 'unidata_version', '0.0-test')
    table = section_model.load(NEUTRAL)
    assert table.tokenizer == 'exported-word-table'
    samples = [h + ' ' + s for h, s, _, _ in ((section_model._text(h), section_model._text(s), t, l)
                                               for h, s, t, l in EDGE_CASES.values())]
    samples += [''.join(chr(c) for c in range(0x20, 0x3000)), 'a_b cͅd éf ᾳͅ x']
    for text in samples:
        assert table.tokens(text) == neutral.tokens(text)
    rows = list(EDGE_CASES.values())
    assert table.decision_function(rows).tolist() == neutral.decision_function(rows).tolist()


def test_tampered_or_foreign_models_are_refused(tmp_path):
    good = write_model(tmp_path / 'good')
    section_model.load(good)
    # A changed byte in any declared file.
    for role in section_model.FILE_ROLES:
        bad = tmp_path / ('bad-' + role)
        shutil.copytree(good, bad)
        name = json.loads((bad / 'manifest.json').read_text())['files'][role]['path']
        data = bytearray((bad / name).read_bytes()); data[len(data) // 2] ^= 1
        (bad / name).write_bytes(bytes(data))
        with pytest.raises(NeutralModelError, match='sha256 does not match'):
            section_model.load(bad)
    # A missing file, an unknown version, a different specification, path escapes, wrong shapes.
    missing = tmp_path / 'missing'; shutil.copytree(good, missing); (missing / 'idf.f64le').unlink()
    with pytest.raises(NeutralModelError, match='missing'):
        section_model.load(missing)
    edits = {
        'version': (lambda m: m.update(format_version=2), 'format_version'),
        'version_bool': (lambda m: m.update(format_version=True), 'format_version'),
        'format': (lambda m: m.update(format='other'), 'format'),
        'lowercase': (lambda m: m['text'].update(lowercase=None), 'text specification'),
        'token': (lambda m: m['text'].update(token_pattern=r'\w+'), 'text specification'),
        'sublinear': (lambda m: m['tfidf'].update(sublinear_tf=True), 'tfidf specification'),
        'norm': (lambda m: m['tfidf'].update(norm='l1'), 'tfidf specification'),
        'nulls': (lambda m: m['inputs'].update(null_text='NaN'), 'inputs specification'),
        'tie_rule': (lambda m: m['classifier'].update(label='highest'), 'classifier specification'),
        'escape': (lambda m: m['files']['idf'].update(path='../idf.f64le'), 'plain file name'),
        'extra_file': (lambda m: m['files'].update(extra=dict(path='x', sha256='0' * 64)), 'files must be exactly'),
        'shape': (lambda m: m['files']['coef'].update(shape=[2, 2002]), 'coef'),
        'word_unicode': (lambda m: m['files']['word_ranges'].update(unicodedata='15.0.0'), 'word table'),
        'n_text': (lambda m: m['features'].update(n_text=1999), 'vocabulary and feature'),
    }
    for name, (edit, message) in edits.items():
        with pytest.raises(NeutralModelError, match=message):
            section_model.load(write_model(tmp_path / ('edit-' + name), manifest_edit=edit))
    with pytest.raises(NeutralModelError, match='classes'):
        section_model.load(write_model(tmp_path / 'dup-classes', classes=['a', 'a', 'b']))
    with pytest.raises(NeutralModelError):
        section_model.load(tmp_path / 'nowhere')


def test_near_ties_follow_the_lowest_index_rule(tmp_path):
    """Exact ties go to the lowest class index (numpy argmax / sklearn); one ulp decides otherwise."""
    width = 2000
    coef = np.zeros((3, width + 2))
    cases = {(0.5, 0.5, 0.1): 'άλλο', (0.1, 0.5, 0.5): 'β', (0.5, 0.1, 0.5): 'άλλο', (0.3, 0.3, 0.3): 'άλλο',
             (0.5, np.nextafter(0.5, 1), 0.0): 'β', (0.0, np.nextafter(0.5, 0), 0.5): 'π',
             (-0.0, 0.0, -1e-300): 'άλλο'}
    for number, (intercept, expected) in enumerate(cases.items()):
        model = section_model.load(write_model(tmp_path / f'tie{number}', coef=coef, intercept=list(intercept)))
        decision = model.decision_function([('a', 'b', 0, 0)])
        assert model.classes[int(np.argmax(decision[0]))] == expected
        assert model.predict([('a', 'b', 0, 0)]).tolist() == [expected]
    # A tie produced by the features, not the intercepts: two classes with identical coefficient rows.
    model = section_model.load(NEUTRAL)
    coef = model.coef.copy(); coef[2] = coef[1]
    tied = section_model.load(write_model(tmp_path / 'feature-tie', coef=coef, intercept=[-10.0, 0.25, 0.25]))
    assert tied.predict([('Βιβλιογραφία', 'Αθήνα εκδόσεις', 1, 0)]).tolist() == ['β']


def test_features_preprocessing_idf_and_normalisation(neutral):
    header, section = 'ΒΙΒΛΙΟΓΡΑΦΙΑ', 'βιβλιογραφία Βιβλιογραφία αθήνα zzqx α'
    idx, vals = neutral.features(header, section, True, 0)
    tokens = (header + ' ' + section).lower().split()
    assert neutral.tokens(header + ' ' + section) == [t for t in tokens if len(t) > 1]
    counts = {}
    for token in tokens:
        if token in neutral.vocabulary:
            counts[neutral.vocabulary[token]] = counts.get(neutral.vocabulary[token], 0) + 1
    assert counts[neutral.vocabulary['βιβλιογραφία']] == 2
    assert idx[:-1] == sorted(counts) and idx[-1] == neutral.n_text  # has_table appended, has_list zero
    text = vals[:-1]
    assert abs(sum(v * v for v in text) - 1.0) < 1e-15 and vals[-1] == 1.0  # the flag is not normalised
    raw = [counts[i] * neutral.idf[i] for i in sorted(counts)]
    assert np.allclose(text, np.array(raw) / np.linalg.norm(raw), rtol=0, atol=1e-16)
    assert neutral.features('', None, 0, 0) == ([], [])
    assert neutral.features('zzqx', 'q', 0, 1) == ([neutral.n_text + 1], [1.0])
    with pytest.raises(TypeError):
        neutral.features(7, 'x', 0, 0)
    assert neutral.predict(frame([])).tolist() == [] and neutral.decision_function(frame([])).shape == (0, 3)


# ---------------------------------------------------------------- the scikit-learn 1.6.1 oracle

def test_tokens_vocabulary_idf_and_class_order_equal_sklearn(neutral, sklearn_model):
    pre, clf = sklearn_model.named_steps['preprocessor'], sklearn_model.named_steps['classifier']
    tfidf = pre.named_transformers_['combined_text'].named_steps['tfidf']
    analyzer = tfidf.build_analyzer()
    for header, section, _, _ in EDGE_CASES.values():
        text = section_model._text(header) + ' ' + section_model._text(section)
        assert neutral.tokens(text) == analyzer(text)
    assert neutral.vocabulary == {term: int(index) for term, index in tfidf.vocabulary_.items()}
    assert neutral.idf.tobytes() == np.ascontiguousarray(tfidf.idf_, dtype='<f8').tobytes()
    assert neutral.coef.tobytes() == np.ascontiguousarray(clf.coef_, dtype='<f8').tobytes()
    assert neutral.intercept.tobytes() == np.ascontiguousarray(clf.intercept_, dtype='<f8').tobytes()
    assert neutral.classes == [str(c) for c in clf.classes_]
    X = prepared(frame(EDGE_CASES.values()))
    transformed = pre.transform(X)
    transformed = transformed.toarray() if hasattr(transformed, 'toarray') else np.asarray(transformed)
    for row, values in zip(transformed, X.itertuples(index=False)):
        idx, vals = neutral.features(*values)
        assert np.flatnonzero(row).tolist() == idx
        assert np.max(np.abs(row[idx] - np.array(vals)), initial=0.0) <= 1e-15


def test_edge_case_scores_and_labels_equal_sklearn(neutral, sklearn_model):
    X = prepared(frame(EDGE_CASES.values()))
    assert neutral.predict(X).tolist() == sklearn_model.predict(X).tolist()
    delta = np.abs(neutral.decision_function(X) - sklearn_model.decision_function(X))
    assert delta.max() <= 1e-12
    # Unprepared nulls (None, NaN) are text '' for the neutral model, as after prepare().
    raw = frame(EDGE_CASES.values())
    assert neutral.decision_function(raw).tolist() == neutral.decision_function(X).tolist()


def test_cohort409_parity_with_sklearn(neutral, sklearn_model):
    """The 409 whole-cohort sections (W2b evidence): identical labels, |delta| <= 1e-12, margins reported."""
    path = os.environ.get('GLOSSAPI_W2B_COHORT')
    if not path:
        pytest.skip('GLOSSAPI_W2B_COHORT (the rebuilt 409-section cohort parquet) is not set')
    cohort = pd.read_parquet(path)
    assert len(cohort) == 409 and cohort.filename.nunique() == 216
    X = prepared(cohort)
    expected = sklearn_model.predict(X)
    scores = neutral.decision_function(X)
    reference = sklearn_model.decision_function(X)
    assert neutral.predict(X).tolist() == expected.tolist() == cohort.node_model_label.tolist()
    delta = np.abs(scores - reference)
    ordered = np.sort(reference, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    report = dict(rows=len(X), max_delta=float(delta.max()), min_margin=float(margin.min()),
                  near_ties_below_1e9=int((margin < 1e-9).sum()), labels=pd.Series(expected).value_counts().to_dict())
    print('W2B_COHORT_PARITY', json.dumps(report, ensure_ascii=False))
    assert delta.max() <= 1e-12
    assert (margin < 1e-9).sum() == 0, report


def test_exporter_round_trip_is_byte_reproducible(tmp_path, neutral):
    reference_oracle('sklearn')
    sys.path.insert(0, str(REPO))
    from tools import export_section_classifier as exporter
    first, second = tmp_path / 'first', tmp_path / 'second'
    exporter.export_file(JOBLIB, first)
    exporter.export_file(JOBLIB, second)
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    exported = json.loads((first / 'manifest.json').read_text(encoding='utf-8'))
    shipped = json.loads((NEUTRAL / 'manifest.json').read_text(encoding='utf-8'))
    same_unicode = unicodedata.unidata_version == shipped['producer']['unicodedata']
    for role in ('vocabulary', 'idf', 'coef', 'intercept'):  # the model data never depends on the interpreter
        name = shipped['files'][role]['path']
        assert (first / name).read_bytes() == (NEUTRAL / name).read_bytes(), name
    if same_unicode:
        assert names == sorted(p.name for p in NEUTRAL.iterdir())
        word = shipped['files']['word_ranges']['path']
        assert (first / word).read_bytes() == (NEUTRAL / word).read_bytes()
    # Only the producer block may differ from the shipped export (platform, Python patch version), and under
    # another Unicode version the word table that records it.
    drop = lambda m: {k: (v if k not in ('files', 'text') or same_unicode else
                          {n: e for n, e in v.items() if n not in ('word_ranges', 'token_regex_flavour')})
                      for k, v in m.items() if k != 'producer'}
    assert drop(exported) == drop(shipped)
    reloaded = section_model.load(first)
    rows = list(EDGE_CASES.values())
    assert reloaded.decision_function(rows).tolist() == neutral.decision_function(rows).tolist()


def test_exporter_refuses_untrusted_bytes_and_other_pipelines(tmp_path, sklearn_model):
    sys.path.insert(0, str(REPO))
    from tools import export_section_classifier as exporter
    fake = tmp_path / 'other.joblib'; fake.write_bytes(b'not the shipped model')
    with pytest.raises(exporter.ExportRefused, match='trusted'):
        exporter.export_file(fake, tmp_path / 'never')
    assert not (tmp_path / 'never').exists()
    changes = dict(lowercase=False, binary=True, ngram_range=(1, 2), norm='l1', smooth_idf=False, sublinear_tf=True,
                   strip_accents='unicode', token_pattern=r'(?u)\b\w+\b', use_idf=False, analyzer='char', stop_words=['και'])
    for name, value in changes.items():
        model = copy.deepcopy(sklearn_model)
        model.named_steps['preprocessor'].named_transformers_['combined_text'].named_steps['tfidf'].set_params(**{name: value})
        with pytest.raises(exporter.ExportRefused, match=name):
            exporter.check_pipeline(model)
    model = copy.deepcopy(sklearn_model)
    model.named_steps['preprocessor'].remainder = 'passthrough'
    with pytest.raises(exporter.ExportRefused, match='remainder'):
        exporter.check_pipeline(model)
    model = copy.deepcopy(sklearn_model)
    model.named_steps['classifier'].coef_ = model.named_steps['classifier'].coef_[:, :-1]
    with pytest.raises(exporter.ExportRefused, match='coefficient shape'):
        exporter.check_pipeline(model)
    occupied = tmp_path / 'occupied'; occupied.mkdir(); (occupied / 'stray.bin').write_bytes(b'x')
    with pytest.raises(exporter.ExportRefused, match='outside the model'):
        exporter.export_file(JOBLIB, occupied)


# ---------------------------------------------------------------- the product path

TEXTS = ['Εισαγωγή στο αντικείμενο της μελέτης και στους στόχους της εργασίας.',
         'Το κύριο σώμα του κειμένου με μερικές προτάσεις για τον έλεγχο.',
         'Παπαδόπουλος, Α. (2001). Ένα βιβλίο. Αθήνα: Εκδόσεις. Σμίθ, Β. (1999). Άρθρο. Βιβλιογραφία και αναφορές.',
         '\n'.join(f'| Κεφάλαιο {i} | {i * 4 + 1} |' for i in range(1, 12)),
         'Παράρτημα με πίνακες και πρόσθετα στοιχεία.', '']
HEADERS = ['Εισαγωγή', 'Κεφάλαιο 1', 'Βιβλιογραφία', 'Περιεχόμενα', 'Παράρτημα', None]


def sections_file(path, documents=(6, 9, 1, 4)):
    rows, ident = [], 1
    for number, count in enumerate(documents):
        for ordinal in range(count):
            pick = (number * 7 + ordinal * 3) % len(TEXTS)
            rows.append(dict(id=ident, row_id=f'row_{ident}', filename=f'DOC_{number:03d}', has_table=pick == 3,
                             has_list=ordinal % 4 == 1, header=HEADERS[pick],
                             section=TEXTS[pick] if ordinal % 9 != 5 else None,
                             section_length=len(TEXTS[pick]) // 10 + ordinal, section_propo=(ordinal * 37) % 400))
            ident += 1
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def classify(model_path, source, work, monkeypatch):
    import glossapi.gloss_section_classifier as gsc
    seen = []
    monkeypatch.setattr(gsc, 'observe', lambda kind, **details: seen.append((kind, {k: v for k, v in details.items()
                                                                                    if k not in ('model_path', 'model_sha256')})))
    work.mkdir()
    classifier = gsc.GlossSectionClassifier()
    classifier.classify_sections(str(source), str(work / 'classified.parquet'), str(model_path), n_cpus=2)
    classifier.fully_annotate(str(work / 'classified.parquet'), str(work / 'full.parquet'), annotation_type='text')
    parts = sorted((work / 'classified.parquet.tmp').iterdir())
    blobs = {name: (work / name).read_bytes() for name in ('classified.parquet', 'full.parquet')}
    blobs.update({'parts/' + p.name: p.read_bytes() for p in parts})
    return blobs, seen, type(classifier.loaded_model).__name__


def test_product_path_neutral_and_legacy_joblib_are_byte_identical(tmp_path, monkeypatch, sklearn_model):
    """classify_sections + fully_annotate: model labels, index/bibliography adjustments, partitions and the
    observations are identical between the neutral default and the legacy joblib path."""
    source = tmp_path / 'sections.parquet'
    sections_file(source)
    neutral_blobs, neutral_seen, neutral_kind = classify(NEUTRAL, source, tmp_path / 'neutral', monkeypatch)
    legacy_blobs, legacy_seen, legacy_kind = classify(JOBLIB, source, tmp_path / 'legacy', monkeypatch)
    assert (neutral_kind, legacy_kind) == ('NeutralSectionModel', 'Pipeline')
    assert neutral_blobs.keys() == legacy_blobs.keys() and len(neutral_blobs) > 3
    assert {k: v == legacy_blobs[k] for k, v in neutral_blobs.items()} == {k: True for k in neutral_blobs}
    kinds = [kind for kind, _ in neutral_seen]
    assert 'annotation_index_override' in kinds and 'annotation_prediction_fallback' not in kinds
    strip = lambda seen: [(kind, {k: v for k, v in d.items() if k != 'path'}) for kind, d in seen]
    assert strip(neutral_seen) == strip(legacy_seen)
    final = pd.read_parquet(tmp_path / 'neutral' / 'full.parquet')
    assert {'π', 'β'} <= set(final.predicted_section)


def test_prediction_exception_falls_back_with_the_observation(tmp_path, monkeypatch):
    """A neutral-model exception keeps the product behaviour: the partition gets 'άλλο' and one observation."""
    source = tmp_path / 'sections.parquet'
    sections_file(source)
    import glossapi.gloss_section_classifier as gsc
    seen = []
    monkeypatch.setattr(gsc, 'observe', lambda kind, **details: seen.append((kind, details)))
    original = NeutralSectionModel.predict
    def failing(self, X):
        if (X['header'] == 'Κεφάλαιο 1').any():
            raise RuntimeError('injected neutral failure')
        return original(self, X)
    monkeypatch.setattr(NeutralSectionModel, 'predict', failing)
    classifier = gsc.GlossSectionClassifier()
    classifier.classify_sections(str(source), str(tmp_path / 'out.parquet'), str(NEUTRAL), n_cpus=1)
    fallbacks = [d for kind, d in seen if kind == 'annotation_prediction_fallback']
    assert fallbacks and all(d['fallback_label'] == 'άλλο' and d['degraded'] for d in fallbacks)
    assert all('injected neutral failure' in d['reported_error']['message'] for d in fallbacks)
    raw = pd.concat(pd.read_parquet(p) for p in sorted((tmp_path / 'out.parquet.tmp').iterdir()))
    failed_ids = {i for d in fallbacks for i in d['row_ids']}
    assert set(raw.loc[raw.id.isin(failed_ids), 'predicted_section']) == {'άλλο'}


def test_zero_rows_return_before_prediction(tmp_path, monkeypatch):
    import glossapi.gloss_section_classifier as gsc
    source = tmp_path / 'sections.parquet'
    sections_file(source, documents=(1,))
    out = tmp_path / 'classified.parquet'
    classifier = gsc.GlossSectionClassifier()
    classifier.classify_sections(str(source), str(out), str(NEUTRAL), n_cpus=1)
    first = out.read_bytes()
    # A second call finds every row processed and returns before any prediction.
    monkeypatch.setattr(NeutralSectionModel, 'predict', lambda self, X: pytest.fail('predicted on zero rows'))
    classifier.classify_sections(str(source), str(out), str(NEUTRAL), n_cpus=1)
    assert out.read_bytes() == first


def test_stage_identity_lineage_and_packages(tmp_path):
    """The manifest is the stage input (it binds every file); lineage records it and the source model;
    one changed model byte changes the execution identity and the run's semantic files."""
    from glossapi import Corpus
    from glossapi.ledger import open_readonly
    from glossapi.ledger.encoding import unpack_evidence
    from glossapi.stages.annotate import AnnotateStage
    from glossapi.stages.contracts import StageSpecification
    assert AnnotateStage().packages(StageSpecification('annotate.batch', settings={'classifier_model_format': 'neutral-v1'})) == ()
    legacy = AnnotateStage().packages(StageSpecification('annotate.batch', settings={'classifier_model_format': 'joblib'}))
    assert legacy[0].package_group == 'annotate-legacy' and 'scikit-learn==1.6.1' in legacy[0].distributions
    corpus = Corpus(tmp_path / 'input', tmp_path / 'out')
    corpus.sections_parquet.parent.mkdir(parents=True, exist_ok=True)
    sections_file(corpus.sections_parquet)
    corpus.annotate()
    with open_readonly(corpus.output_dir / 'corpus.sqlite3') as connection:
        evidence = [unpack_evidence(json.loads(raw)) for raw, in connection.execute("SELECT json FROM snapshots WHERE kind='stage_evidence'")]
    decision = next(item for item in evidence if isinstance(item, dict) and 'lineage' in item)
    manifest_sha = hashlib.sha256((NEUTRAL / 'manifest.json').read_bytes()).hexdigest()
    assert decision['model_sha256'] == manifest_sha and decision['model_format'] == 'neutral-v1'
    assert decision['model_source_sha256'] == hashlib.sha256(JOBLIB.read_bytes()).hexdigest()
    assert {row['model_sha256'] for row in decision['lineage']} == {manifest_sha}


def test_one_changed_model_byte_changes_identity(tmp_path, monkeypatch):
    from glossapi.stages import scheduler
    identity = scheduler.execution_environment
    package = tmp_path / 'glossapi'
    shutil.copytree(Path(glossapi.__file__).resolve().parent, package, ignore=shutil.ignore_patterns('__pycache__'))
    fake = package / 'stages' / 'scheduler.py'
    monkeypatch.setattr(scheduler, '__file__', str(fake))
    before = identity()['models']
    coef = package / 'models/section_classifier/coef.f64le'
    data = bytearray(coef.read_bytes()); data[100] ^= 1; coef.write_bytes(bytes(data))
    after = identity()['models']
    changed = [a['path'] for a, b in zip(before, after) if a != b]
    assert changed == ['models/section_classifier/coef.f64le']
    assert any(entry['path'] == 'models/section_classifier/manifest.json' for entry in before)
    with pytest.raises(NeutralModelError):
        section_model.load(package / 'models/section_classifier')
