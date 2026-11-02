"""K4g fix round 3: a full disk inside font repair must not become a successful extract.

Node evidence (K4g final gates on a8a3495, kill matrix worker-ENOSPC arm): FOK_514's Rust font repair hit ENOSPC on the
full staging filesystem. The native returned the io error as ``"ValueError: No space left on device (os error 28)"``
text, ``pdf_repair/extract.py`` took the "repair failed, keep the unrepaired PDF" path, and the extract attempt was
recorded ``ok`` with degraded text (clean markdown 310,446 -> 309,256 bytes); the resume never retried it.

Fix: the native raises an OS-level I/O failure as ``OSError`` with its errno (native API level 1; the selector refuses
older builds), the Python reference does the same, and the font-repair caller and the per-document best-effort
handlers let a full disk propagate, so the attempt fails typed ``disk_full`` and ``retry_failed`` re-runs it.

Pipeline seam: a ``sitecustomize`` in the run's child (inherited by its spawned stage workers) wraps the native
``glossapi_rs_pdfrepair.repair_pdf`` call, as the extract pre-pass makes it, for the Greek document only.
"""
from __future__ import annotations

import errno
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from test_pdf_font_repair import greek_fixture  # noqa: F401  (module fixture)

REPO = Path(__file__).resolve().parents[1]
CONTROL = REPO / 'samples' / 'lightweight_pdf_corpus' / 'pdfs' / 'alpha.pdf'
API = 'GLOSSAPI_NATIVE_API:glossapi_rs_pdfrepair:1;'

FAULT = textwrap.dedent('''
    import os
    MODE = os.environ.get('GLOSSAPI_TEST_REPAIR_FAULT')
    if MODE:
        import errno
        from pathlib import Path
        import glossapi_rs_pdfrepair as native
        _repair = native.repair_pdf
        def repair_pdf(src, dst, *rest):
            if 'greek' in Path(src).name:
                marker = Path(os.environ['GLOSSAPI_TEST_REPAIR_MARKER'])
                if MODE == 'enospc' and not marker.exists():
                    marker.write_text('failed once')
                    # What the native raises (API level 1) when its staged write meets a full filesystem.
                    raise OSError(errno.ENOSPC, 'No space left on device (os error 28)')
                if MODE == 'font_error':
                    raise ValueError('ValueError: injected font program failure')
            return _repair(src, dst, *rest)
        native.repair_pdf = repair_pdf
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
            # Seeded metadata, as the node kill matrix seeds it (one row per source before the run).
            import pandas as pd
            from glossapi.parquet_schema import ParquetSchema
            ParquetSchema().write_metadata_parquet(pd.DataFrame([dict(filename=Path(p).name, filename_base=Path(p).stem,
                url=Path(p).as_uri(), download_success=True, file_ext='pdf', processing_stage='', extract_status='')
                for p in sys.argv[3:]]), root / 'output' / 'download_results' / 'download_results.parquet')
            result = corpus.run([Path(p) for p in sys.argv[3:]], stages=('extract', 'clean', 'export'), settings={'debug': 'off'})
        else:
            with open_readonly(root / 'output' / 'corpus.sqlite3') as connection:
                run = connection.execute("SELECT run_id FROM tasks WHERE operation='extract.compatibility_publish' "
                                         "ORDER BY rowid DESC LIMIT 1").fetchone()[0]
            result = corpus.run(resume=run, retry_failed=True)
        out = root / 'output'
        with open_readonly(out / 'corpus.sqlite3') as connection:
            attempts = [list(row) for row in connection.execute(
                f"SELECT t.operation, e.reason_code, e.status FROM {connection.event_table} e JOIN tasks t USING(task_id) "
                "WHERE t.operation='extract.document' AND e.attempt_id IS NOT NULL AND e.status IN ('ok','failed') "
                "ORDER BY e.event_id")]
        sidecar = out / 'sidecars' / 'extract' / 'greek.json'
        print(json.dumps(dict(state=result['state'], counts=result.get('document_counts'), attempts=attempts,
            sidecar=json.loads(sidecar.read_text()) if sidecar.is_file() else None,
            markdown=sorted(p.name for p in (out / 'markdown').glob('*.md')),
            clean=sorted(p.name for p in (out / 'clean_markdown').glob('*.md')),
            records=[json.loads(line).get('filename') for line in (out / 'export.jsonl').read_text().splitlines()]
                    if (out / 'export.jsonl').exists() else [])))

    if __name__ == '__main__':  # spawned stage workers re-import this module
        main()
''')


def _child_env(root: Path, mode: str | None) -> dict[str, str]:
    env = {**os.environ, 'GLOSSAPI_TEST_REPAIR_MARKER': str(root / 'failed-once'), 'PYTHONDONTWRITEBYTECODE': '1',
           'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([str(root / 'fault'), *filter(None, [os.environ.get('PYTHONPATH')]), str(REPO / 'src')])}
    env.pop('GLOSSAPI_STAGE_PROFILE', None)
    env.pop('GLOSSAPI_GREEK_WORDLIST', None)
    env.pop('GLOSSAPI_TEST_REPAIR_FAULT', None)
    if mode:
        env['GLOSSAPI_TEST_REPAIR_FAULT'] = mode
    return env


def _run(root: Path, mode: str | None, steps) -> dict[str, dict]:
    (root / 'fault').mkdir(parents=True)
    (root / 'fault' / 'sitecustomize.py').write_text(FAULT)
    (root / 'child.py').write_text(CHILD)
    out = {}
    for step, extra in steps:
        done = subprocess.run([sys.executable, '-B', str(root / 'child.py'), str(root / 'run'), step, *extra],
                              capture_output=True, text=True, env=_child_env(root, mode), cwd=root, timeout=900)
        assert done.returncode == 0, done.stderr[-4000:]
        out[step] = json.loads(done.stdout.strip().splitlines()[-1])
        out[step]['products'] = {str(path.relative_to(root / 'run' / 'output')): path.read_bytes()
                                 for name in ('markdown', 'clean_markdown') for path in sorted((root / 'run' / 'output' / name).glob('*.md'))}
        export = root / 'run' / 'output' / 'export.jsonl'
        out[step]['export'] = export.read_bytes() if export.exists() else b''
        import pandas as pd
        out[step]['download_results'] = pd.read_parquet(root / 'run' / 'output' / 'download_results' / 'download_results.parquet')
    return out


@pytest.fixture(scope='module')
def runs(tmp_path_factory, greek_fixture):  # noqa: F811
    if importlib.util.find_spec('glossapi_rs_pdfrepair') is None:
        pytest.skip('glossapi_rs_pdfrepair is not installed: the fault seam is the native repair call')
    inputs = tmp_path_factory.mktemp('k4g-fix3-inputs')
    (inputs / 'greek.pdf').write_bytes(greek_fixture['broken'].read_bytes())
    (inputs / 'alpha.pdf').write_bytes(CONTROL.read_bytes())
    sources = [str(inputs / 'alpha.pdf'), str(inputs / 'greek.pdf')]
    return {
        'reference': _run(tmp_path_factory.mktemp('k4g-fix3-reference'), None, [('start', sources)]),
        'disk_full': _run(tmp_path_factory.mktemp('k4g-fix3-enospc'), 'enospc', [('start', sources), ('resume', [])]),
        'font_error': _run(tmp_path_factory.mktemp('k4g-fix3-font-error'), 'font_error', [('start', sources)]),
    }


def test_reference_run_repairs_the_greek_document(runs):
    reference = runs['reference']['start']
    assert reference['state'] == 'completed'
    assert reference['sidecar']['font_repair_status'] == 'repaired' and reference['sidecar']['font_repair_fonts'] >= 1
    assert reference['sidecar']['font_repair_backend'] == 'rust'
    assert reference['clean'] == ['alpha.md', 'greek.md'] and sorted(reference['records']) == ['alpha', 'greek']


def test_enospc_in_font_repair_fails_the_attempt_typed_disk_full(runs):
    start = runs['disk_full']['start']
    assert start['state'] == 'completed_with_failures', start
    assert ['extract.document', 'disk_full', 'failed'] in start['attempts'], start['attempts']
    assert [row for row in start['attempts'] if row[2] == 'failed'] == [['extract.document', 'disk_full', 'failed']]
    # No degraded success: no extract sidecar recording "ok", no published markdown, clean text or export record.
    assert start['sidecar'] is None or start['sidecar'].get('extract_status') != 'ok', start['sidecar']
    assert 'greek.md' not in start['markdown'] and 'greek.md' not in start['clean']
    assert start['records'] == ['alpha']


def test_resume_reruns_the_document_and_equals_the_no_fault_run(runs):
    resume, reference = runs['disk_full']['resume'], runs['reference']['start']
    assert resume['state'] == 'completed'
    assert resume['counts'] == {'selected': 2, 'included': 2, 'omitted': 0, 'failed': 0}
    assert resume['sidecar']['font_repair_status'] == 'repaired'
    assert resume['sidecar'] == reference['sidecar']
    assert resume['products'] == reference['products']  # markdown and clean_markdown bytes
    # Export and metadata equal the no-fault run except two fields that differ after ANY retried extract failure of a
    # document without Latin letters (latin_percentage 0.0 -> NaN, char_count_no_comments int -> float). That is a
    # separate, pre-existing clean-metrics merge difference: reproduced on 5cedabc with the generic worker ENOSPC of
    # test_k4g_b4n_retry_failed_republish (no font repair involved; k4g/fix3/mac/probe/).
    known = {'latin_percentage'}
    def records(raw):
        return [{key: value for key, value in json.loads(line).items() if key not in known} for line in raw.splitlines()]
    assert records(resume['export']) == records(reference['export'])
    import pandas as pd
    pd.testing.assert_frame_equal(resume['download_results'].drop(columns=sorted(known)),
                                  reference['download_results'].drop(columns=sorted(known)), check_dtype=False)


def test_a_non_disk_repair_error_still_continues_with_the_unrepaired_pdf(runs):
    start, reference = runs['font_error']['start'], runs['reference']['start']
    assert start['state'] == 'completed'
    assert start['sidecar']['font_repair_status'] == 'error' and start['sidecar']['font_repair_fonts'] == 0
    assert 'injected font program failure' in start['sidecar']['font_repair_error']
    assert start['sidecar']['extract_status'] == 'ok'
    assert not any(row[2] == 'failed' for row in start['attempts'])
    assert 'greek.md' in start['clean'] and sorted(start['records']) == ['alpha', 'greek']
    # The unrepaired text differs from the repaired reference (the repair matters for this document).
    assert start['products']['clean_markdown/greek.md'] != reference['products']['clean_markdown/greek.md']


# --------------------------------------------------------------------------------------- the native, unit level ---

def _native():
    native = pytest.importorskip('glossapi_rs_pdfrepair')
    extension = getattr(native, 'glossapi_rs_pdfrepair', native)
    if getattr(extension, '__glossapi_native_api__', None) != API:
        pytest.fail(f'glossapi_rs_pdfrepair on the path is not built from this source (no {API}): rebuild the natives')
    return native


def test_native_raises_a_real_os_error_with_its_errno(tmp_path):
    """A real OS-level failure under the native's own write: the output's parent is a regular file (ENOTDIR)."""
    native = _native()
    source = REPO / 'rust/glossapi_rs_pdfrepair/fixtures/review/name_decoding_valid_utf8.pdf'
    blocker = tmp_path / 'blocker'
    blocker.write_bytes(b'file')
    with pytest.raises(NotADirectoryError) as caught:
        native.repair_pdf(str(source), str(blocker / 'sub' / 'repaired.pdf'))
    assert caught.value.errno == errno.ENOTDIR
    report, analysis = native.analyze_for_repair(str(source))
    with pytest.raises(NotADirectoryError):
        native.repair_pdf(str(source), str(blocker / 'sub' / 'repaired.pdf'), [], analysis)
    with pytest.raises(FileNotFoundError) as missing:
        native.analyze_for_repair(str(tmp_path / 'missing.pdf'))
    assert missing.value.errno == errno.ENOENT
    with pytest.raises(FileNotFoundError):
        native.analyze_pdf(str(tmp_path / 'missing.pdf'))


def test_native_keeps_document_errors_in_the_report(tmp_path):
    native = _native()
    bad = tmp_path / 'bad.pdf'
    bad.write_bytes(b'<html>503</html>')
    report, analysis = native.analyze_for_repair(str(bad))
    assert report['error'] and analysis is None
    assert native.analyze_pdf(str(bad))['error'].startswith('ValueError:')
    result = native.repair_pdf(str(bad), str(tmp_path / 'out.pdf'))
    assert result['error'] and not result['written'] and 'os_error' not in result


def test_python_reference_raises_os_errors_too(tmp_path):
    from glossapi.pdf_repair import tounicode
    source = REPO / 'rust/glossapi_rs_pdfrepair/fixtures/review/name_decoding_valid_utf8.pdf'
    blocker = tmp_path / 'blocker'
    blocker.write_bytes(b'file')
    with pytest.raises(NotADirectoryError):
        tounicode.repair_pdf(source, blocker / 'sub' / 'repaired.pdf')
    with pytest.raises(FileNotFoundError):
        tounicode.analyze_pdf(tmp_path / 'missing.pdf')
    bad = tmp_path / 'bad.pdf'
    bad.write_bytes(b'not a PDF')
    assert tounicode.analyze_pdf(bad).error and not tounicode.repair_pdf(bad, tmp_path / 'out.pdf').written


# ------------------------------------------------------------------------------- the font-repair caller, in process ---

def _backend(error):
    from glossapi.pdf_repair import reports
    def analyze_for_repair(path):
        font = reports.FontReport('F', 'Type0', repair_possible=True, greek_codes=1, category='repairable')
        return reports.PdfFontReport(str(path), category='repairable', pages=1, fonts=[font], backend='rust'), object()
    def repair(src, dst, *, protected_sources=(), analysis=None):
        raise error
    return SimpleNamespace(name='rust', analyze_pdf=None, repair_pdf=repair, analyze_for_repair=analyze_for_repair)


@pytest.mark.parametrize('error', [OSError(errno.ENOSPC, 'No space left on device'), OSError(errno.EDQUOT, 'Disk quota exceeded'),
                                   sqlite3.OperationalError('database or disk is full'),
                                   RuntimeError('wrapped'), ValueError('ValueError: broken font program')],
                         ids=['enospc', 'edquot', 'sqlite_full', 'enospc_as_cause', 'font_error'])
def test_repair_for_extract_raises_a_full_disk_and_keeps_other_errors(tmp_path, monkeypatch, error):
    from glossapi.pdf_repair import extract, selector
    import time
    if isinstance(error, RuntimeError):
        try:
            raise RuntimeError('repair staging failed') from OSError(errno.ENOSPC, 'No space left on device')
        except RuntimeError as wrapped:
            error = wrapped
    monkeypatch.setattr(selector, 'get_backend', lambda *a, **k: _backend(error))
    source = tmp_path / 'downloads' / 'doc.pdf'
    source.parent.mkdir()
    source.write_bytes(b'%PDF-1.4\n%%EOF\n')
    if isinstance(error, ValueError):
        selected, data = extract.repair_for_extract(source, tmp_path, enabled=True, deadline=time.monotonic() + 30)
        assert selected == source and data['font_repair_status'] == 'error' and 'broken font program' in data['font_repair_error']
    else:
        with pytest.raises(type(error)):
            extract.repair_for_extract(source, tmp_path, enabled=True, deadline=time.monotonic() + 30)
    assert not list(tmp_path.glob('repaired_pdfs/*.pdf'))  # nothing published from a failed pre-pass


def test_failed_result_types_a_full_disk_in_the_cause_chain(tmp_path):
    from glossapi.stages.execution import failed_result
    request = SimpleNamespace(task_id='t', attempt_id='a', lease_token='l', work_directory=tmp_path)
    try:
        raise ValueError('export write failed') from OSError(errno.ENOSPC, 'No space left on device')
    except ValueError as error:
        assert failed_result(request, 'export_exception', error).reason_code == 'disk_full'
    assert failed_result(request, 'export_exception', ValueError('other')).reason_code == 'export_exception'


def test_extract_sidecar_write_on_a_full_disk_is_raised(tmp_path, monkeypatch):
    """Sibling of the font-repair bug: the extraction sidecar write used to be best effort, so a full disk left the
    attempt ok without its sidecar (the stage reads a missing sidecar as no backend failure)."""
    from glossapi.gloss_extract import GlossExtract
    extractor = GlossExtract.__new__(GlossExtract)
    extractor._font_repair_records, extractor._font_repair_originals = {}, {}
    extractor._log = SimpleNamespace(warning=lambda *a, **k: None)
    real = Path.write_text
    def full(self, *args, **kwargs):
        if self.parent.name == 'extract':
            raise OSError(errno.ENOSPC, 'No space left on device')
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', full)
    (tmp_path / 'markdown').mkdir()
    with pytest.raises(OSError) as caught:
        extractor._update_extraction_metadata(tmp_path / 'markdown', tmp_path / 'doc.pdf', 'ok', 'pypdfium', page_count=1)
    assert caught.value.errno == errno.ENOSPC
    def denied(self, *args, **kwargs):
        raise PermissionError(errno.EACCES, 'Permission denied')
    monkeypatch.setattr(Path, 'write_text', denied)
    extractor._update_extraction_metadata(tmp_path / 'markdown', tmp_path / 'doc.pdf', 'ok', 'pypdfium', page_count=1)  # still best effort


# ---------------------------------------------------------------------------------------- the stale-native guard ---

def test_selector_does_not_use_a_native_without_the_api_level(monkeypatch):
    from glossapi.pdf_repair import selector
    stale = SimpleNamespace(analyze_pdf=lambda path: pytest.fail('a stale native must not be used'), repair_pdf=None)
    monkeypatch.setitem(sys.modules, 'glossapi_rs_pdfrepair', stale)
    assert selector.get_backend().name == 'python'
    with pytest.raises(ImportError, match='stale native module glossapi_rs_pdfrepair'):
        selector.get_backend('rust')
    current = SimpleNamespace(analyze_pdf=lambda path: {}, repair_pdf=None, __glossapi_native_api__=API)
    monkeypatch.setitem(sys.modules, 'glossapi_rs_pdfrepair', current)
    assert selector.get_backend().name == 'rust'
    package = SimpleNamespace(analyze_pdf=lambda path: {}, repair_pdf=None,
                              glossapi_rs_pdfrepair=SimpleNamespace(__glossapi_native_api__=API))  # maturin layout
    monkeypatch.setitem(sys.modules, 'glossapi_rs_pdfrepair', package)
    assert selector.get_backend().name == 'rust'


def test_pdfrepair_level_is_reported_and_matches_the_crate():
    import re
    from glossapi.capabilities import NATIVE_API_LEVELS, native_module_status
    source = (REPO / 'rust/glossapi_rs_pdfrepair/src/lib.rs').read_text(encoding='utf-8')
    assert re.findall(r'NATIVE_API_MARKER: &str = "GLOSSAPI_NATIVE_API:([a-z_]+):(\d+);"', source) == [
        ('glossapi_rs_pdfrepair', str(NATIVE_API_LEVELS['glossapi_rs_pdfrepair']))]
    row = native_module_status('glossapi_rs_pdfrepair', False)
    assert row['required'] is False and row['required_api_level'] == 1
    if importlib.util.find_spec('glossapi_rs_pdfrepair') is not None:
        assert row['status'] == 'present' and row['api_level'] == 1, row
