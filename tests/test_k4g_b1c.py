"""K4g B1c: corpus publications are current only for the document versions they cover (the B4-N retry-failed finding is
tests/test_k4g_b4n_retry_failed_republish.py), and PostCloseEditError is a typed CLI error."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_k4g_b4n_retry_failed_republish import CHILD, FAULT, PDFS, REPO


def _child(root, mode, *extra, fault=True):
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([str(root / 'fault'), *filter(None, [os.environ.get('PYTHONPATH')]), str(REPO / 'src')])}
    if fault:
        env['GLOSSAPI_TEST_ENOSPC_ONCE'] = str(root / 'failed-once')
    env.pop('GLOSSAPI_STAGE_PROFILE', None)
    return subprocess.run([sys.executable, '-B', str(root / 'child.py'), str(root / 'run'), mode, *extra],
                          capture_output=True, text=True, env=env, cwd=root, timeout=600)


def _started(tmp_path):
    root = tmp_path
    (root / 'fault').mkdir()
    (root / 'fault' / 'sitecustomize.py').write_text(FAULT)
    (root / 'child.py').write_text(CHILD.replace("result = corpus.run(resume=run, retry_failed=True)",
                                                 "result = corpus.run(resume=run, retry_failed=(mode == 'resume'))"))
    done = _child(root, 'start', str(PDFS / 'alpha.pdf'), str(PDFS / 'beta.pdf'))
    assert done.returncode == 0, done.stderr[-3000:]
    start = json.loads(done.stdout.strip().splitlines()[-1])
    assert start['state'] == 'completed_with_failures' and start['records'] == ['alpha']
    return root, start


def test_b1c_failed_run_keeps_the_products_a_retry_republishes_from(tmp_path):
    """A run that ends with failed documents keeps its other documents' committed private products."""
    root, _ = _started(tmp_path)
    kept = [path for path in (root / 'run' / 'output' / '.glossapi' / 'artifacts').rglob('alpha.md') if path.parent.name == 'markdown']
    assert kept, 'the raw markdown a repeated extract publication materializes was released'


def test_b1c_resume_without_retry_failed_leaves_the_publications_as_they_are(tmp_path):
    root, start = _started(tmp_path)
    done = _child(root, 'plain-resume', fault=False)
    assert done.returncode == 0, done.stderr[-3000:]
    resumed = json.loads(done.stdout.strip().splitlines()[-1])
    assert resumed['state'] == 'completed_with_failures' and resumed['records'] == ['alpha']
    assert resumed['barriers'] == start['barriers']  # coverage unchanged: nothing is published again


def test_b1c_republication_with_released_inputs_is_refused_typed(tmp_path):
    """A repeated extract publication that would need a committed product that is gone is refused with a typed error
    before anything is written (a resume re-runs a document whose committed output vanished while its source exists;
    this guards the case it cannot, e.g. a download already retired under memory_saving=extreme)."""
    from types import SimpleNamespace
    from glossapi.ledger import LedgerWriter
    from glossapi.stages import run as RUN
    root, _ = _started(tmp_path)
    output = root / 'run' / 'output'
    receipts = [path for path in (output / '.glossapi' / 'artifacts').rglob('worker-result.json') if '/extract/' in str(path)]
    assert receipts
    replays = {('extract', 'extract.document', ()): [(None, {}, SimpleNamespace(path=path), path.parent) for path in receipts]}
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        corpus = SimpleNamespace(_scheduler_writer=writer, _streaming_ocr_replays=replays)
        assert RUN._republication_inputs_missing(corpus, 'extract') == []
        assert RUN._republication_inputs_missing(corpus, 'export') == []  # corpus-level publications recompute
        for path in (output / '.glossapi' / 'artifacts').rglob('alpha.md'):
            if path.parent.name == 'markdown':
                path.unlink()
        missing = RUN._republication_inputs_missing(corpus, 'extract')
    assert [path.name for path in missing] == ['alpha.md']
    error = RUN.PublicationInputsReleasedError('extract', missing)
    assert error.code == 'republish_inputs_released' and error.details['missing_count'] == 1


def test_b1c_post_close_edit_is_a_typed_cli_error(tmp_path, capsys, monkeypatch):
    """The CLI reported a post-close edit as the generic `unsupported_schema` (exit 2)."""
    import time
    from glossapi import Corpus
    from glossapi.cli import main
    from test_ledger_fixset_lg import pdf
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))
    root = tmp_path/'out'
    first = Corpus(tmp_path/'empty', root).run([pdf(tmp_path/'doc.pdf', 'Typed post close edit text')], stages=('extract', 'clean', 'export'),
                                               settings={'extract': {'repair_fonts': False}})
    time.sleep(0.1)  # outside the run close's clock tick
    (root/'export.jsonl').write_bytes(b'USER EDIT\n')
    assert main(['run', '--corpus', str(root), '--resume', first['run_id'], '--json']) == 3
    error = json.loads(capsys.readouterr().out)['errors'][0]
    assert error['code'] == 'post_close_edit' and error['details']['paths'] == [str(root/'export.jsonl')]
    assert (root/'export.jsonl').read_bytes() == b'USER EDIT\n'
