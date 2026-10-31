"""Default Corpus model resolution (T2-G13): packaged models, no injected paths.

Every Corpus here uses the public default constructor. The only exceptions are
the missing-model cases, which must leave an honest skipped record instead of
classified output.
"""
from pathlib import Path

import pandas as pd
import pytest

import glossapi
from glossapi import Corpus
from glossapi.ledger import open_readonly

PACKAGED = Path(glossapi.__file__).resolve().parent / 'models'


def sections_fixture(corpus):
    rows = [dict(filename='sample', id=i, header=header, section=body, section_length=len(body),
                 section_propo=len(body), has_table=False, has_list=False)
            for i, (header, body) in enumerate([
                ('Εισαγωγή', 'Μικρό κείμενο για έλεγχο της προεπιλεγμένης ταξινόμησης.'),
                ('Κεφάλαιο 1', 'Το κύριο σώμα του κειμένου με μερικές προτάσεις.'),
                ('Βιβλιογραφία', 'Παπαδόπουλος, Α. (2001). Ένα βιβλίο. Αθήνα.')], start=1)]
    corpus.sections_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(corpus.sections_parquet, index=False)
    return rows


def annotation_events(corpus):
    with open_readonly(corpus.output_dir / 'corpus.sqlite3') as db:
        return [dict(zip(('status', 'reason_code'), row)) for row in db.execute(
            "SELECT status,reason_code FROM events WHERE reason_code LIKE 'annotation_%'")]


def test_default_constructor_resolves_packaged_models(tmp_path):
    corpus = Corpus(tmp_path / 'input', tmp_path / 'out')
    # K2 W2b: the packaged default is the neutral section-classifier directory (own inference, no sklearn).
    assert corpus.section_classifier_model_path == PACKAGED / 'section_classifier'
    assert (corpus.section_classifier_model_path / 'manifest.json').is_file()
    assert corpus.extraction_model_path == PACKAGED / 'kmeans_weights.joblib'
    assert corpus.extraction_model_path.is_file()


def test_default_constructor_annotates_with_packaged_model(tmp_path):
    corpus = Corpus(tmp_path / 'input', tmp_path / 'out')
    rows = sections_fixture(corpus)
    corpus.annotate()
    assert corpus.classified_parquet.is_file(), annotation_events(corpus)
    classified = pd.read_parquet(corpus.classified_parquet)
    assert sorted(classified.id.tolist()) == [row['id'] for row in rows]
    assert classified.predicted_section.notna().all()
    assert corpus.fully_annotated_parquet.is_file()
    assert {'status': 'skipped', 'reason_code': 'annotation_model_missing'} not in annotation_events(corpus)


def test_default_constructor_run_annotates_and_exports(tmp_path):
    pytest.importorskip('glossapi_rs_cleaner')
    from test_ledger_stages import simple_pdf
    source = simple_pdf(tmp_path / 'source.pdf')
    corpus = Corpus(tmp_path / 'empty', tmp_path / 'output')
    result = corpus.run([source], stages=('extract', 'clean', 'sections', 'annotate', 'export'),
                        settings={'extract': {'repair_fonts': False}})
    assert result['state'] == 'completed'
    assert corpus.classified_parquet.is_file(), annotation_events(corpus)
    assert len(pd.read_parquet(corpus.classified_parquet)) >= 1
    assert (corpus.output_dir / 'export.jsonl').is_file()
    with open_readonly(corpus.output_dir / 'corpus.sqlite3') as db:
        outcomes = dict(db.execute(
            "SELECT stage,status FROM current_task_state WHERE run_id=? AND operation LIKE '%.compatibility_publish'",
            (result['run_id'],)))
    assert outcomes['annotate'] == 'ok'
    assert {'status': 'skipped', 'reason_code': 'annotation_model_missing'} not in annotation_events(corpus)


def test_explicit_missing_model_is_recorded_as_skipped(tmp_path):
    missing = tmp_path / 'no-such-model.joblib'
    corpus = Corpus(tmp_path / 'input', tmp_path / 'out', section_classifier_model_path=missing)
    sections_fixture(corpus)
    assert corpus.annotate() is None
    assert not corpus.classified_parquet.exists()
    assert {'status': 'skipped', 'reason_code': 'annotation_model_missing'} in annotation_events(corpus)


def test_missing_packaged_default_is_recorded_as_skipped(tmp_path, monkeypatch):
    from glossapi.corpus import corpus_orchestrator
    empty = tmp_path / 'packaged-models'
    empty.mkdir()
    monkeypatch.setattr(corpus_orchestrator, 'PACKAGED_MODELS_DIR', empty)
    corpus = Corpus(tmp_path / 'input', tmp_path / 'out')
    assert corpus.section_classifier_model_path == empty / 'section_classifier'
    sections_fixture(corpus)
    assert corpus.annotate() is None
    assert not corpus.classified_parquet.exists()
    assert {'status': 'skipped', 'reason_code': 'annotation_model_missing'} in annotation_events(corpus)
