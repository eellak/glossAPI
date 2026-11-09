"""OCR-COLLIDE (K4g B3): the streaming run's OCR stage on two scanned PDFs with the same basename.

A stub engine replaces only the Docling converter (registered as the ``glossapi`` entry point, with external options
so no Greek OCR model loads). It is installed by a ``sitecustomize`` on the child's PYTHONPATH, so the spawned stage
workers run it too. The engine writes each source's own sha256 into its page text, so a record that carried the other
source's OCR text is detectable. Extract, clean, OCR routing, OCR reassembly, the after-OCR clean, the census and
``report --why`` are the real code.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[1]
SCAN = REPO / 'tests' / 'fixtures' / 'greekocr' / 'polytonic_scan.pdf'

STUB = textwrap.dedent('''
    import os
    if os.environ.get('GLOSSAPI_TEST_STUB_OCR') == '1':
        import hashlib
        from pathlib import Path
        from types import SimpleNamespace
        from glossapi.stages import ocr as _stage
        from glossapi.corpus.ocr import dispatch as _dispatch

        class StubOptions(SimpleNamespace):
            def model_dump(self, mode='json'):
                return {'kind': 'stub-plugin', 'device': 'cpu'}

        _init = _stage.OcrStage.__init__
        def _stub_init(self, ocr_options=None):
            _init(self, ocr_options if ocr_options is not None else StubOptions(kind='stub-plugin', device='cpu'))
        _stage.OcrStage.__init__ = _stub_init
        _stage.OcrStage.packages = lambda self, specification: ()
        _dispatch.entry_points = lambda **_: [SimpleNamespace(name='glossapi')]

        def _convert(source, page_range, **_):
            digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()[:12]
            return SimpleNamespace(status='success', document=SimpleNamespace(
                export_to_markdown=lambda *, page_no: f'STUB OCR {digest} page {page_no}'))
        _dispatch._build_converter = lambda request: SimpleNamespace(convert=_convert)
''')

CHILD = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    from glossapi import Corpus
    from glossapi.ledger.writer import open_readonly
    from glossapi.ledger.reader import report_why, select_run

    def main():
        root, sources = Path(sys.argv[1]), [Path(item) for item in sys.argv[2:]]
        corpus = Corpus(root / 'empty', root / 'output')
        result = corpus.run(sources, stages=('extract', 'clean', 'ocr', 'export'), settings={'extract': {'repair_fonts': False}})
        export = root / 'output' / 'export.jsonl'
        rows = [json.loads(line) for line in export.read_text().splitlines()] if export.exists() else []
        with open_readonly(root / 'output' / 'corpus.sqlite3') as connection:
            run = select_run(connection)
            why = {str(path): report_why(connection, str(path), run=run) for path in sources}
            tasks = [list(row) for row in connection.execute(
                "SELECT stage, operation, status, reason_code FROM current_task_state WHERE stage = 'ocr'")]
        print(json.dumps(dict(state=result['state'], counts=result.get('document_counts'), rows=rows, tasks=tasks,
            why={key: dict(inclusion=value['inclusion'], files=[[item['kind'], Path(item['path']).name] for item in value['files']])
                 for key, value in why.items()})))

    if __name__ == '__main__':  # spawned stage workers re-import this module
        main()
''')


@pytest.fixture(scope='module')
def collided(tmp_path_factory):
    return _ocr_run(tmp_path_factory, 'ocr-collide', ('scan.pdf', 'scan.pdf'))


@pytest.fixture(scope='module')
def distinct(tmp_path_factory):
    return _ocr_run(tmp_path_factory, 'ocr-distinct', ('alpha.pdf', 'beta.pdf'))


def _ocr_run(tmp_path_factory, label, names):
    if not SCAN.is_file():
        pytest.skip(f'scanned fixture missing: {SCAN}')
    root = tmp_path_factory.mktemp(label)
    (root / 'stub').mkdir()
    (root / 'stub' / 'sitecustomize.py').write_text(STUB)
    (root / 'child.py').write_text(CHILD)
    sources = []
    for label, filename in zip(('first', 'second'), names):
        path = root / label / filename
        path.parent.mkdir()
        path.write_bytes(SCAN.read_bytes() + f'\n% variant {label}\n'.encode())  # same basename, different bytes
        sources.append(path)
    env = {**os.environ, 'GLOSSAPI_TEST_STUB_OCR': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([str(root / 'stub'), *filter(None, [os.environ.get('PYTHONPATH')]), str(REPO / 'src')])}
    env.pop('GLOSSAPI_STAGE_PROFILE', None)
    done = subprocess.run([sys.executable, '-B', str(root / 'child.py'), str(root / 'run'), *map(str, sources)],
                          capture_output=True, text=True, env=env, cwd=root, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    result['digests'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()[:12] for path in sources}
    return result


def test_same_basename_scans_get_their_own_ocr_text(collided):
    assert collided['state'] == 'completed'
    assert sorted(task[3] for task in collided['tasks'] if task[1] == 'ocr.document') == ['ocr_completed', 'ocr_completed']
    rows = {row['filename']: row['document'] for row in collided['rows']}
    assert len(rows) == 2 and all(name.startswith('scan__') for name in rows)
    for source, digest in collided['digests'].items():
        names = {name[:-3] for kind, name in collided['why'][source]['files'] if kind == 'ocr_markdown'}
        assert len(names) == 1, collided['why'][source]['files']  # one OCR output, under this source's own name
        (name,) = names
        assert f'STUB OCR {digest} page 1' in rows[name] and f'STUB OCR {digest} page 2' in rows[name]
        other = next(value for key, value in collided['digests'].items() if key != source)
        assert other not in rows[name]


def test_ocr_routed_documents_are_included_in_census_and_why(collided):
    assert collided['counts'] == {'selected': 2, 'included': 2, 'omitted': 0, 'failed': 0}
    for source in collided['digests']:
        decisions = [item for item in collided['why'][source]['inclusion'] if item['product'] == 'document_jsonl']
        assert [item['decision'] for item in decisions] == ['include']


def test_ocr_routed_documents_with_distinct_names_are_included_in_census_and_why(distinct):
    """K4g B1b: the same census defect without a name collision (B3 finding); the census and report --why follow the
    records the export published."""
    assert distinct['state'] == 'completed' and len(distinct['rows']) == 2
    assert distinct['counts'] == {'selected': 2, 'included': 2, 'omitted': 0, 'failed': 0}
    for source in distinct['digests']:
        decisions = [item for item in distinct['why'][source]['inclusion'] if item['product'] == 'document_jsonl']
        assert [item['decision'] for item in decisions] == ['include']
