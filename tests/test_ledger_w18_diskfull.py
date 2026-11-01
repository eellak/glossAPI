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
