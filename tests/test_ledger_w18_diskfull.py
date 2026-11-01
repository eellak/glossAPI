"""W18 M10: an exhausted filesystem is a typed, resumable interruption, told apart from process death (K4(b))."""
import errno
import json

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def test_disk_full_is_typed_reported_before_resume_and_named_on_the_interrupted_attempt(tmp_path, monkeypatch):
    from glossapi.corpus.phase_clean import CleanPhaseMixin
    from glossapi.stages.run import DiskFullError
    from glossapi.ledger.reader import status_view, select_run
    from test_ledger_fixset_lg import pdf
    from test_k3_interface import cli
    source = pdf(tmp_path/'in/doc.pdf', 'Disk full body text')
    root = tmp_path/'out'
    original = CleanPhaseMixin.clean
    def full(self, *args, **kwargs):
        raise OSError(errno.ENOSPC, 'No space left on device', str(root/'clean_markdown/doc.md'))
    monkeypatch.setattr(CleanPhaseMixin, 'clean', full)
    import os
    saved = {name: os.environ.get(name) for name in ('GLOSSAPI_IMAGES_SCALE', 'GLOSSAPI_FORMULA_BATCH', 'DOCLING_ARTIFACTS_PATH', 'OMP_NUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'GLOSSAPI_OCR_TOOLKIT_SOURCE', 'GREEKOCR_SEGMENTER_MODEL_DIR')}
    with pytest.raises(DiskFullError) as raised:
        Corpus(tmp_path/'empty', root).run([source], stages=('extract', 'clean', 'export'), settings={'extract': {'repair_fonts': False}})
    assert raised.value.code == 'disk_full' and raised.value.errno == errno.ENOSPC and raised.value.details['run_id']
    run_id = raised.value.details['run_id']
    with open_readonly(root/'corpus.sqlite3') as connection:
        status = status_view(connection, run=select_run(connection, run_id))
    assert status['state'] == 'interrupted' and status['interruption']['cause'] == 'disk_full'
    code, envelope = cli('status', '--corpus', str(root), '--run', run_id)
    assert code == 0 and envelope['data']['interruption']['errno'] == 28
    code, envelope = cli('report', '--corpus', str(root))
    assert code == 0, envelope
    monkeypatch.setattr(CleanPhaseMixin, 'clean', original)
    # The in-process run left thread-count variables set; a real resume starts a fresh process (same environment).
    for name, value in saved.items():
        monkeypatch.setenv(name, value) if value is not None else monkeypatch.delenv(name, raising=False)
    resumed = Corpus(tmp_path/'empty', root).run(resume=run_id)
    assert resumed['state'] == 'completed', resumed
    with open_readonly(root/'corpus.sqlite3') as connection:
        from glossapi.ledger.reader import evidence
        interrupted = [evidence(connection, {'details_json': row[0]}) for row in connection.execute(
            f"SELECT details_json FROM {connection.event_table} WHERE run_id=? AND reason_code='attempt_interrupted'", (run_id,))]
    assert interrupted and all(item.get('interruption_cause') == 'disk_full' and item.get('errno') == 28 for item in interrupted)


def test_a_ledger_too_full_for_the_record_still_names_the_cause(tmp_path, monkeypatch):
    """K4BI-G2: the disk_full record allocated at run start carries the cause when the ledger cannot take one more page."""
    import os, sqlite3
    from glossapi.corpus.phase_clean import CleanPhaseMixin
    from glossapi.ledger.writer import LedgerWriter
    from glossapi.stages.run import DiskFullError
    from glossapi.ledger.reader import status_view, select_run
    from glossapi.stages.interruption import NAME, SIZE
    from test_ledger_fixset_lg import pdf
    source = pdf(tmp_path/'in/doc.pdf', 'Ledger full body text')
    root = tmp_path/'out'
    original_clean, original_event = CleanPhaseMixin.clean, LedgerWriter.event
    monkeypatch.setattr(CleanPhaseMixin, 'clean', lambda self, *a, **k: (_ for _ in ()).throw(OSError(errno.ENOSPC, 'No space left on device')))
    def event(self, run_id, **fields):
        if fields.get('reason_code') == 'disk_full':
            raise sqlite3.OperationalError('database or disk is full')
        return original_event(self, run_id, **fields)
    monkeypatch.setattr(LedgerWriter, 'event', event)
    saved = {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS',
                                                       'GLOSSAPI_IMAGES_SCALE', 'GLOSSAPI_FORMULA_BATCH')}
    with pytest.raises(DiskFullError) as raised:
        Corpus(tmp_path/'empty', root).run([source], stages=('extract', 'clean', 'export'), settings={'extract': {'repair_fonts': False}})
    run_id = raised.value.details['run_id']
    assert (root/NAME).stat().st_size == SIZE  # overwritten in place, never grown
    with open_readonly(root/'corpus.sqlite3') as connection:
        status = status_view(connection, run=select_run(connection, run_id))
    assert status['state'] == 'interrupted' and status['interruption']['basis'] == 'record allocated at run start'
    monkeypatch.setattr(CleanPhaseMixin, 'clean', original_clean); monkeypatch.setattr(LedgerWriter, 'event', original_event)
    for name, value in saved.items():
        monkeypatch.setenv(name, value) if value is not None else monkeypatch.delenv(name, raising=False)
    resumed = Corpus(tmp_path/'empty', root).run(resume=run_id)
    assert resumed['state'] == 'completed'
    with open_readonly(root/'corpus.sqlite3') as connection:
        from glossapi.ledger.reader import evidence
        causes = [evidence(connection, {'details_json': row[0]}).get('interruption_cause') for row in connection.execute(
            f"SELECT details_json FROM {connection.event_table} WHERE run_id=? AND reason_code='attempt_interrupted'", (run_id,))]
    assert causes and set(causes) == {'disk_full'}
    assert json.loads((root/NAME).read_text())['state'] == 'clear'


def test_a_worker_output_write_on_a_full_disk_is_the_documents_disk_full():
    """W18 M10 second point: ENOSPC under a worker's output write is the document's typed disk_full outcome."""
    from types import SimpleNamespace
    from glossapi.stages.execution import failed_result
    request = SimpleNamespace(task_id='t', attempt_id='a', lease_token='l', work_directory=__import__('pathlib').Path('/nonexistent-k4b'))
    result = failed_result(request, 'extract_exception', OSError(errno.ENOSPC, 'No space left on device'))
    assert result.reason_code == 'disk_full' and result.error.message.startswith('[Errno 28]')
