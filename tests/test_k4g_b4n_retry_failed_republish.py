"""K4F1-G2 follow-up (K4g B4-N node finding): a worker write that fails with ENOSPC makes the run end
``completed_with_failures``; ``run(resume=..., retry_failed=True)`` then re-runs the failed documents to ``ok`` but does
not re-publish the corpus barriers (clean/sections/export compatibility publications stay at their first, partial
commit). The resumed run reports ``completed`` while export.jsonl and clean_markdown/ lack the recovered documents and
the census still omits them.

Node evidence: k4g/b4n kill matrix ``disk_full_worker`` arm (staging tmpfs filled after extract starts): 6-7 failed
extract attempts typed disk_full, resume state ``completed``, 19-21 product differences against the uninterrupted
reference, every ``*.compatibility_publish`` task ``ok`` with 1 attempt.

Mac reproduction: a ``sitecustomize`` in the child (and its spawned stage workers) makes the first extract attempt of
``beta.pdf`` raise OSError(ENOSPC); the worker types it disk_full (stages/execution.failed_result). Strict xfail until the
ledger/run owner (batch B1b) fixes the resume; the test needs no tmpfs and no model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[1]
PDFS = REPO / 'samples' / 'lightweight_pdf_corpus' / 'pdfs'

FAULT = textwrap.dedent('''
    import os
    if os.environ.get('GLOSSAPI_TEST_ENOSPC_ONCE'):
        import errno
        from pathlib import Path
        from glossapi.stages import extract as _extract
        _original = _extract.ExtractStage._run_document
        def _run_document(self, request, *, reporter):
            marker = Path(os.environ['GLOSSAPI_TEST_ENOSPC_ONCE'])
            if any('beta' in str(item) for item in request.inputs) and not marker.exists():
                marker.write_text('failed once')
                raise OSError(errno.ENOSPC, 'No space left on device (injected worker write failure)')
            return _original(self, request, reporter=reporter)
        _extract.ExtractStage._run_document = _run_document
''')

CHILD = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    from glossapi import Corpus
    from glossapi.ledger.writer import open_readonly

    def main():
        root, mode = Path(sys.argv[1]), sys.argv[2]
        corpus = Corpus(root / 'empty', root / 'output')
        if mode == 'start':
            result = corpus.run([Path(p) for p in sys.argv[3:]], stages=('extract', 'clean', 'export'), settings={'debug': 'off'})
        else:
            with open_readonly(root / 'output' / 'corpus.sqlite3') as connection:
                run = connection.execute("SELECT run_id FROM tasks WHERE operation='extract.compatibility_publish' "
                                         "ORDER BY rowid DESC LIMIT 1").fetchone()[0]
            result = corpus.run(resume=run, retry_failed=True)
        export = root / 'output' / 'export.jsonl'
        with open_readonly(root / 'output' / 'corpus.sqlite3') as connection:
            failed = connection.execute(f"SELECT COUNT(*) FROM {connection.event_table} WHERE reason_code='disk_full' "
                                        "AND status='failed' AND attempt_id IS NOT NULL").fetchone()[0]
            barriers = [list(row) for row in connection.execute(
                "SELECT t.operation, s.status, (SELECT COUNT(*) FROM attempts a WHERE a.task_id=t.task_id) FROM tasks t "
                "JOIN current_task_state s USING(task_id) WHERE t.operation LIKE '%.compatibility_publish'")]
        print(json.dumps(dict(state=result['state'], counts=result.get('document_counts'), worker_disk_full=failed, barriers=barriers,
            records=[json.loads(line).get('filename') for line in export.read_text().splitlines()] if export.exists() else [],
            clean=sorted(p.name for p in (root / 'output' / 'clean_markdown').glob('*.md')))))

    if __name__ == '__main__':  # spawned stage workers re-import this module
        main()
''')


@pytest.fixture(scope='module')
def retried(tmp_path_factory):
    root = tmp_path_factory.mktemp('retry-failed')
    (root / 'fault').mkdir()
    (root / 'fault' / 'sitecustomize.py').write_text(FAULT)
    (root / 'child.py').write_text(CHILD)
    env = {**os.environ, 'GLOSSAPI_TEST_ENOSPC_ONCE': str(root / 'failed-once'), 'PYTHONDONTWRITEBYTECODE': '1',
           'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([str(root / 'fault'), *filter(None, [os.environ.get('PYTHONPATH')]), str(REPO / 'src')])}
    env.pop('GLOSSAPI_STAGE_PROFILE', None)
    out = {}
    for mode, extra in (('start', [str(PDFS / 'alpha.pdf'), str(PDFS / 'beta.pdf')]), ('resume', [])):
        done = subprocess.run([sys.executable, '-B', str(root / 'child.py'), str(root / 'run'), mode, *extra],
                              capture_output=True, text=True, env=env, cwd=root, timeout=600)
        assert done.returncode == 0, done.stderr[-4000:]
        out[mode] = json.loads(done.stdout.strip().splitlines()[-1])
    return out


def test_the_injected_worker_enospc_is_a_typed_document_failure(retried):
    start = retried['start']
    assert start['state'] == 'completed_with_failures' and start['worker_disk_full'] == 1
    assert start['records'] == ['alpha'] and start['clean'] == ['alpha.md']
    assert retried['resume']['state'] == 'completed'


@pytest.mark.xfail(strict=True, reason='K4g B4-N finding for B1b (stages/run.py publish_barrier; scheduler retry_failed): '
                   'the resume re-runs the failed document but keeps the first, partial barrier publications')
def test_retry_failed_resume_republishes_the_recovered_document(retried):
    resume = retried['resume']
    assert len(resume['records']) == 2 and len(resume['clean']) == 2
    assert resume['counts'] == {'selected': 2, 'included': 2, 'omitted': 0, 'failed': 0}
