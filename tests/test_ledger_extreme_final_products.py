"""K4(b) round 1 (K4BI-G7, G8, G10): under memory_saving=extreme the final products are charged to the corpus
filesystem, never to the per-attempt staging cap; a run whose corpus filesystem cannot hold them is refused with a typed
error up front and again, from exact text sizes, before any publication; plan --json shows the estimate."""
import json
from pathlib import Path

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


@pytest.fixture
def stub_guard(monkeypatch):
    from glossapi.stages import disk_budget
    from glossapi.stages.retention import resolved_settings
    def guard(root, settings):
        if resolved_settings(settings)['memory_saving'] == 'extreme':
            (Path(root).resolve()/'.glossapi/staging').mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(disk_budget, 'require_disk_guard', guard)  # Linux mounts only


def sources(tmp_path):
    from test_ledger_fixset_lg import pdf
    return [pdf(tmp_path/'in'/f'{name}.pdf', f'{name.upper()} body text for the export') for name in ('alpha', 'bravo', 'charlie')]


def extreme(**overrides):
    return {'memory_saving': 'extreme', 'disk_budget_bytes': 1 << 30, 'max_source_bytes': 4096, 'max_attempt_bytes': 8192,
            'extract': {'repair_fonts': False}, **overrides}


def records(root):
    return [{key: value for key, value in json.loads(line).items()} for line in (root/'export.jsonl').read_text().splitlines()]


def test_aggregate_publications_are_charged_to_the_corpus_not_the_attempt_cap(tmp_path, stub_guard):
    docs = sources(tmp_path)
    normal = Corpus(tmp_path/'empty', tmp_path/'normal').run(docs, stages=('extract', 'clean', 'sections', 'annotate', 'export'),
                                                             settings={'extract': {'repair_fonts': False}})
    run = Corpus(tmp_path/'empty', tmp_path/'extreme').run(docs, stages=('extract', 'clean', 'sections', 'annotate', 'export'), settings=extreme())
    assert normal['state'] == run['state'] == 'completed', run
    with open_readonly(tmp_path/'extreme/corpus.sqlite3') as connection:
        aggregate = connection.execute("SELECT SUM(f.bytes) FROM attempt_inputs i JOIN files f USING(file_id) JOIN attempts a USING(attempt_id) "
                                       "JOIN tasks t USING(task_id) WHERE t.operation='export.publish'").fetchone()[0]
    assert aggregate > 8192, 'the export aggregate must exceed the per-attempt cap for this test to mean anything'
    assert [r['document'] for r in records(tmp_path/'extreme')] == [r['document'] for r in records(tmp_path/'normal')]
    assert not any((tmp_path/'extreme/.glossapi/staging').iterdir())


def test_final_products_that_cannot_fit_are_refused_up_front(tmp_path, stub_guard, monkeypatch):
    from glossapi.stages import final_products
    from glossapi.stages.disk_budget import DiskBudgetError
    monkeypatch.setattr(final_products, 'corpus_room', lambda root, settings: 10)
    with pytest.raises(DiskBudgetError, match='disk_budget_exceeded.*estimated at run start'):
        Corpus(tmp_path/'empty', tmp_path/'out').run(sources(tmp_path), stages=('extract', 'clean', 'export'), settings=extreme())
    with open_readonly(tmp_path/'out/corpus.sqlite3') as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0, 'nothing may run before the refusal'


def test_the_exact_check_refuses_before_any_publication_and_the_resume_completes(tmp_path, stub_guard, monkeypatch):
    import os
    from glossapi.stages import final_products
    from glossapi.stages.disk_budget import DiskBudgetError
    monkeypatch.setattr(final_products, 'PLANNING_TEXT_RATIO', 0.0)  # the estimate passes; the exact text does not fit
    monkeypatch.setattr(final_products, 'PER_DOCUMENT_BYTES', 0)
    real = final_products.corpus_room
    monkeypatch.setattr(final_products, 'corpus_room', lambda root, settings: 64)
    saved = {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS',
                                                       'GLOSSAPI_IMAGES_SCALE', 'GLOSSAPI_FORMULA_BATCH')}
    with pytest.raises(DiskBudgetError, match='exact, before the publications'):
        Corpus(tmp_path/'empty', tmp_path/'out').run(sources(tmp_path), stages=('extract', 'clean', 'export'), settings=extreme())
    with open_readonly(tmp_path/'out/corpus.sqlite3') as connection:
        statuses = {row[0]: row[1] for row in connection.execute(
            f"SELECT t.operation,(SELECT status FROM {connection.event_table} e WHERE e.task_id=t.task_id AND e.event_kind='lifecycle' "
            "ORDER BY event_id DESC LIMIT 1) FROM tasks t")}
        run_id = connection.execute("SELECT run_id FROM tasks WHERE operation='export.compatibility_publish'").fetchone()[0]
    assert statuses['clean.initial'] == 'ok' and statuses['export.compatibility_publish'] == 'planned'
    assert not (tmp_path/'out/export.jsonl').exists()
    monkeypatch.setattr(final_products, 'corpus_room', real)
    for name, value in saved.items():
        monkeypatch.setenv(name, value) if value is not None else monkeypatch.delenv(name, raising=False)
    assert Corpus(tmp_path/'empty', tmp_path/'out').run(resume=run_id)['state'] == 'completed'
    assert len(records(tmp_path/'out')) == 3


def test_plan_shows_the_final_product_and_kept_text_estimates(tmp_path):
    from glossapi.cli import plan_documents
    plan = plan_documents(tmp_path/'out', files=[str(path) for path in sources(tmp_path)], stages=['extract', 'clean', 'sections', 'export'],
                          settings={'extract': {'repair_fonts': False}})
    estimate = plan['disk']['final_products']
    assert estimate['final_product_bytes'] > estimate['text_bytes'] > 0 and estimate['kept_text_bytes'] > 0
    assert estimate['sources_without_size'] == 0 and 'planning ratio' in estimate['basis']
