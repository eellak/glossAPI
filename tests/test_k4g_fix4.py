"""K4g fix round 4: a retried document gives the same products as an uninterrupted run; a full disk in clean is typed.

1. Retry drift (found by fix 3 on 5cedabc). After ANY retried document failure, a ``--retry-failed`` resume wrote
   products that differ from an uninterrupted run:
   - ``latin_percentage`` NaN for every document that failed in the first pass (Latin letters or not: the clean
     finalize renamed this pass's ``mojibake_latin_percentage`` onto the previous ``latin_percentage`` and kept the
     first of the two equal names, the stale one);
   - ``char_count_no_comments`` float64 instead of int64 (the table persisted with a gap keeps float64 when the gap is
     filled);
   - without seeded metadata (a local-file run): ``url`` ``''``, ``download_success`` False and ``extract_error``
     ``''`` for the retried row and str dtypes, where the single pass writes nulls (``_ProcessingStateManager.save``
     invented values for a row it added, against the one metadata rule of ``ensure_metadata_parquet``);
   - with OCR in the plan: the other documents lost their clean text and were omitted (filter ``extract_failed``):
     the clean frontier released public raw markdown before the repeated OCR publication, which re-cleans markdown/.
2. A full disk in the Rust cleaner was typed ``clean_process_failed``: it printed failed stage-one / edits-CSV writes
   and returned other write errors as ``ValueError`` text. Native API level 2 raises ``OSError(errno, ...)``.

Every product under the corpus root (except ``.glossapi/`` and the ledger) is compared byte for byte; the OCR page
sidecars record the producing run's id, which is compared blanked.
"""
from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import uuid

import pytest

from test_pdf_font_repair import greek_fixture  # noqa: F401  (module fixture)

REPO = Path(__file__).resolve().parents[1]
PDFS = REPO / 'samples' / 'lightweight_pdf_corpus' / 'pdfs'
SCAN = REPO / 'tests' / 'fixtures' / 'greekocr' / 'polytonic_scan.pdf'
CLEANER_API = 'GLOSSAPI_NATIVE_API:glossapi_rs_cleaner:2;'

FAULT = textwrap.dedent('''
    import os
    if os.environ.get('GLOSSAPI_TEST_STUB_OCR') == '1':
        from test_ocr_same_basename import STUB
        exec(STUB)
    STAGE = os.environ.get('K4G4_FAULT_STAGE')
    if STAGE in ('extract', 'clean'):
        # One worker failure, typed disk_full by the stage (the B4N seam): ENOSPC raised once for the target document.
        import errno, importlib
        from pathlib import Path
        owner = getattr(importlib.import_module('glossapi.stages.' + STAGE), {'extract': 'ExtractStage', 'clean': 'CleanStage'}[STAGE])
        name = 'run_document' if STAGE == 'clean' else '_run_document'
        original = getattr(owner, name)
        def wrapped(self, request, *args, **kwargs):
            marker = Path(os.environ['K4G4_MARKER'])
            if any(os.environ['K4G4_TARGET'] in Path(str(getattr(item, 'path', item))).name for item in request.inputs) \\
                    and not marker.exists():
                marker.write_text('failed once')
                raise OSError(errno.ENOSPC, 'No space left on device (injected worker write failure)')
            return original(self, request, *args, **kwargs)
        setattr(owner, name, wrapped)
    if STAGE == 'cleaner_full_disk':
        # The native cleaner's own writes meet a really full filesystem (a full RAM disk), once, for the target
        # document: its output, report and evidence directories are moved there. Nothing else changes.
        from pathlib import Path
        import glossapi_rs_cleaner as cleaner
        real = cleaner.run_complete_pipeline
        def run_complete_pipeline(input_dir, output_dir, report, scripts, threads, write_cleaned_files=True,
                                  evidence_directory=None):
            marker = Path(os.environ['K4G4_MARKER'])
            if any(os.environ['K4G4_TARGET'] in name for name in os.listdir(input_dir)) and not marker.exists():
                marker.write_text('failed once')
                full = Path(os.environ['K4G4_FULL_DIR'])
                return real(input_dir, str(full / 'clean_markdown'), str(full / 'report.parquet'), scripts, threads,
                            write_cleaned_files, evidence_directory=str(full / 'evidence'))
            return real(input_dir, output_dir, report, scripts, threads, write_cleaned_files,
                        evidence_directory=evidence_directory)
        cleaner.run_complete_pipeline = run_complete_pipeline
''')

CHILD = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    from glossapi import Corpus
    from glossapi.ledger.writer import open_readonly

    def main():
        root, mode = Path(sys.argv[1]), sys.argv[2]
        corpus = Corpus(root / 'empty', root / 'output')
        if mode == 'start':
            stages = tuple(os.environ['K4G4_STAGES'].split(','))
            settings = {'debug': 'off'}
            if 'ocr' in stages:
                settings['extract'] = {'repair_fonts': False}
            if os.environ.get('K4G4_SEED') == '1':
                # Seeded metadata, as the node kill matrix seeds it (one row per source before the run).
                import pandas as pd
                from glossapi.parquet_schema import ParquetSchema
                ParquetSchema().write_metadata_parquet(pd.DataFrame([dict(filename=Path(p).name, filename_base=Path(p).stem,
                    url=Path(p).as_uri(), download_success=True, file_ext='pdf', processing_stage='', extract_status='')
                    for p in sys.argv[3:]]), root / 'output' / 'download_results' / 'download_results.parquet')
            result = corpus.run([Path(p) for p in sys.argv[3:]], stages=stages, settings=settings)
            (root / 'runid').write_text(result['run_id'])
        else:
            result = corpus.run(resume=(root / 'runid').read_text(), retry_failed=True)
        with open_readonly(root / 'output' / 'corpus.sqlite3') as connection:
            failed = [list(row) for row in connection.execute(
                f"SELECT t.operation, e.reason_code FROM {connection.event_table} e JOIN tasks t USING(task_id) "
                "WHERE e.status='failed' AND e.attempt_id IS NOT NULL AND t.version_id IS NOT NULL ORDER BY e.event_id")]
        print(json.dumps(dict(state=result['state'], counts=result.get('document_counts'), failed=failed)))

    if __name__ == '__main__':  # spawned stage workers re-import this module
        main()
''')


def _blank_run_ids(value):
    if isinstance(value, dict):
        return {key: '<run>' if key == 'run_id' else _blank_run_ids(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_blank_run_ids(item) for item in value]
    return value


def products(output: Path) -> dict[str, str]:
    """sha256 of every product file under the corpus root, except the ledger and .glossapi/ (private state)."""
    found = {}
    for path in sorted(output.rglob('*')):
        relative = str(path.relative_to(output))
        if not path.is_file() or relative.startswith(('.glossapi/', 'corpus.sqlite3')):
            continue
        data = path.read_bytes()
        if relative.startswith('sidecars/ocr_pages/'):
            data = json.dumps(_blank_run_ids(json.loads(data)), sort_keys=True).encode()  # per-run identity only
        found[relative] = hashlib.sha256(data).hexdigest()
    return found


def run_case(root: Path, sources, *, stages, seed=False, fault=None, target='', full_dir=None, steps=('start',)):
    (root / 'fault').mkdir(parents=True)
    (root / 'fault' / 'sitecustomize.py').write_text(FAULT)
    (root / 'child.py').write_text(CHILD)
    env = {**os.environ, 'K4G4_MARKER': str(root / 'failed-once'), 'K4G4_TARGET': target, 'K4G4_STAGES': ','.join(stages),
           'K4G4_SEED': '1' if seed else '', 'PYTHONDONTWRITEBYTECODE': '1', 'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([str(root / 'fault'), *filter(None, [os.environ.get('PYTHONPATH')]),
                                          str(REPO / 'src'), str(REPO / 'tests')])}
    for key in ('GLOSSAPI_STAGE_PROFILE', 'GLOSSAPI_GREEK_WORDLIST', 'K4G4_FAULT_STAGE', 'GLOSSAPI_TEST_STUB_OCR', 'K4G4_FULL_DIR'):
        env.pop(key, None)
    if fault:
        env['K4G4_FAULT_STAGE'] = fault
    if 'ocr' in stages:
        env['GLOSSAPI_TEST_STUB_OCR'] = '1'
    if full_dir:
        env['K4G4_FULL_DIR'] = str(full_dir)
    out = {}
    for step in steps:
        extra = [str(source) for source in sources] if step == 'start' else []
        done = subprocess.run([sys.executable, '-B', str(root / 'child.py'), str(root / 'run'), step, *extra],
                              capture_output=True, text=True, env=env, cwd=root, timeout=900)
        assert done.returncode == 0, done.stderr[-4000:]
        out[step] = json.loads(done.stdout.strip().splitlines()[-1])
    out['products'] = products(root / 'run' / 'output')
    out['output'] = root / 'run' / 'output'
    return out


def assert_same_products(retried, reference):
    import pandas as pd
    a, b = reference['products'], retried['products']
    differing = {name: (a.get(name), b.get(name)) for name in sorted(set(a) | set(b)) if a.get(name) != b.get(name)}
    columns = []
    pa, pb = reference['output'] / 'download_results/download_results.parquet', retried['output'] / 'download_results/download_results.parquet'
    if pa.is_file() and pb.is_file():
        da, db = pd.read_parquet(pa).set_index('filename'), pd.read_parquet(pb).set_index('filename')
        for column in sorted(set(da.columns) | set(db.columns)):
            if column not in da.columns or column not in db.columns:
                columns.append((column, 'present in one only'))
            elif str(da[column].dtype) != str(db[column].dtype):
                columns.append((column, str(da[column].dtype), str(db[column].dtype)))
            else:
                columns += [(column, name, repr(da.at[name, column]), repr(db.at[name, column]))
                            for name in da.index if name in db.index and repr(da.at[name, column]) != repr(db.at[name, column])]
    assert not differing and not columns, dict(differing_files=differing, download_results=columns[:20])


# ------------------------------------------------------------------ item 1: a retried run equals the single pass ---

CASES = {
    # name: (sources key, stages, seed, fault stage, target)
    'extract-greek-seeded': ('greek', ('extract', 'clean', 'export'), True, 'extract', 'greek'),   # no Latin letters
    'extract-alpha-local': ('greek', ('extract', 'clean', 'export'), False, 'extract', 'alpha'),   # Latin letters
    'clean-greek-local': ('greek', ('extract', 'clean', 'export'), False, 'clean', 'greek'),
    'extract-scan-ocr': ('scan', ('extract', 'clean', 'ocr', 'export'), False, 'extract', 'polytonic'),
    'clean-alpha-ocr-seeded': ('scan', ('extract', 'clean', 'ocr', 'export'), True, 'clean', 'alpha'),
}


@pytest.fixture(scope='module')
def retry_runs(tmp_path_factory, greek_fixture):  # noqa: F811
    if importlib.util.find_spec('glossapi_rs_cleaner') is None:
        pytest.skip('glossapi_rs_cleaner is not installed')
    inputs = tmp_path_factory.mktemp('k4g-fix4-inputs')
    shutil.copyfile(greek_fixture['broken'], inputs / 'greek.pdf')
    for name in ('alpha.pdf', 'beta.pdf'):
        shutil.copyfile(PDFS / name, inputs / name)
    shutil.copyfile(SCAN, inputs / 'polytonic_scan.pdf')
    sources = {'greek': [inputs / 'alpha.pdf', inputs / 'beta.pdf', inputs / 'greek.pdf'],
               'scan': [inputs / 'alpha.pdf', inputs / 'polytonic_scan.pdf']}
    references, runs = {}, {}
    for name, (key, stages, seed, fault, target) in CASES.items():
        if (key, stages, seed) not in references:
            references[(key, stages, seed)] = run_case(tmp_path_factory.mktemp('k4g-fix4-reference'), sources[key],
                                                       stages=stages, seed=seed)
        runs[name] = dict(reference=references[(key, stages, seed)],
                          retried=run_case(tmp_path_factory.mktemp('k4g-fix4-' + name), sources[key], stages=stages,
                                           seed=seed, fault=fault, target=target, steps=('start', 'resume')))
    return runs


@pytest.mark.parametrize('case', sorted(CASES))
def test_retried_run_products_equal_the_uninterrupted_run(retry_runs, case):
    run = retry_runs[case]
    reference, retried = run['reference'], run['retried']
    stage = CASES[case][3]
    assert reference['start']['state'] == 'completed' and not reference['start']['failed']
    start, resume = retried['start'], retried['resume']
    assert start['state'] == 'completed_with_failures', start
    assert [row[1] for row in start['failed']] == ['disk_full'] and start['failed'][0][0].startswith(stage + '.'), start
    assert resume['state'] == 'completed' and resume['counts'] == reference['start']['counts'], (resume, reference['start'])
    assert_same_products(retried, reference)


def test_document_without_latin_letters_keeps_latin_percentage_zero(retry_runs):
    """The case fix 3 reported: a retried document with no Latin letters has latin_percentage 0.0 in both products."""
    import pandas as pd
    retried = retry_runs['extract-greek-seeded']['retried']
    frame = pd.read_parquet(retried['output'] / 'download_results/download_results.parquet').set_index('filename')
    assert frame.at['greek.pdf', 'latin_percentage'] == 0.0
    assert str(frame['char_count_no_comments'].dtype) == 'int64'
    records = {json.loads(line)['filename']: json.loads(line) for line in (retried['output'] / 'export.jsonl').read_text().splitlines()}
    assert records['greek']['latin_percentage'] == 0.0


def test_processing_state_adds_a_row_with_nulls_like_the_artifact_scan(tmp_path):
    """The row _ProcessingStateManager.save adds for a document the table lacks follows the one metadata rule: the URL
    and download fields are null unless the file is on disk; extract_error is null unless an error is given."""
    import pandas as pd
    from glossapi.corpus.corpus_state import _ProcessingStateManager
    from glossapi.parquet_schema import ParquetSchema
    root = tmp_path / 'corpus'
    (root / 'markdown').mkdir(parents=True)
    (root / 'downloads').mkdir()
    (root / 'downloads' / 'ondisk.pdf').write_bytes(b'%PDF-1.4\n')
    ParquetSchema().write_metadata_parquet(pd.DataFrame({'filename': ['known.pdf'], 'url': ['https://x/known.pdf'],
        'download_success': [True], 'download_error': [''], 'extract_status': ['success'], 'extract_error': [''],
        'processing_stage': ['download,extract']}), root / 'download_results' / 'download_results.parquet')
    _ProcessingStateManager(root / 'markdown' / '.processing_state.pkl').save({'known', 'new', 'ondisk'}, set())
    frame = pd.read_parquet(root / 'download_results' / 'download_results.parquet').set_index('filename')
    assert frame.at['known.pdf', 'url'] == 'https://x/known.pdf' and frame.at['known.pdf', 'extract_error'] == ''
    assert pd.isna(frame.at['new.pdf', 'url']) and pd.isna(frame.at['new.pdf', 'download_success'])
    assert pd.isna(frame.at['new.pdf', 'download_error']) and pd.isna(frame.at['new.pdf', 'extract_error'])
    assert frame.at['new.pdf', 'extract_status'] == 'success' and frame.at['new.pdf', 'processing_stage'] == 'extract'
    assert bool(frame.at['ondisk.pdf', 'download_success']) and frame.at['ondisk.pdf', 'download_error'] == ''


# ------------------------------------------------------------------- item 2: the cleaner types OS-level failures ---

def _cleaner():
    cleaner = pytest.importorskip('glossapi_rs_cleaner')
    return cleaner


def _corpus_dir(root: Path, name='doc.md', text='Ελληνικό κείμενο. Greek text.\n') -> Path:
    source = root / 'in'
    source.mkdir(parents=True)
    (source / name).write_text(text, encoding='utf-8')
    return source


def test_cleaner_declares_api_level_2():
    cleaner = _cleaner()
    extension = getattr(cleaner, 'glossapi_rs_cleaner', cleaner)
    assert extension.__glossapi_native_api__ == CLEANER_API
    from glossapi.capabilities import NATIVE_API_LEVELS, native_module_status
    assert NATIVE_API_LEVELS['glossapi_rs_cleaner'] == 2
    row = native_module_status('glossapi_rs_cleaner', True)
    assert row['status'] == 'present' and row['api_level'] == 2, row


@pytest.mark.parametrize('where', ['output_parent', 'evidence_parent', 'edits_csv', 'stage3_write', 'report'])
def test_cleaner_raises_real_os_errors_with_their_errno(tmp_path, where):
    """Real kernel errors under the native's own writes (ENOTDIR/EISDIR): before level 2 the edits-CSV failure was
    printed (and the call returned), the stage-3 write failure was printed and swallowed, and the others were
    ValueError text."""
    cleaner = _cleaner()
    source = _corpus_dir(tmp_path)
    blocker = tmp_path / 'a-file'
    blocker.write_text('x')
    output, evidence, report = tmp_path / 'out', tmp_path / 'evidence', tmp_path / 'report.parquet'
    output.mkdir()
    evidence.mkdir()
    expected = errno.ENOTDIR
    if where == 'output_parent':
        output = blocker / 'out'
    elif where == 'evidence_parent':
        evidence = blocker / 'evidence'
    elif where == 'edits_csv':
        (evidence / 'doc.edits.csv').mkdir()  # the evidence file's name is taken by a directory
        expected = errno.EISDIR
    elif where == 'stage3_write':
        (output / 'doc.md').mkdir()  # the cleaned file's name is taken by a directory
        expected = errno.EISDIR
    elif where == 'report':
        report = blocker / 'report.parquet'
    with pytest.raises(OSError) as raised:
        cleaner.run_complete_pipeline(str(source), str(output), str(report), ['greek', 'latin'], 1, True,
                                      evidence_directory=str(evidence))
    assert raised.value.errno == expected, raised.value
    assert not isinstance(raised.value, ValueError)


def test_cleaner_content_problems_are_unchanged(tmp_path):
    """Invalid UTF-8 input is a content problem: the file is skipped and left out of the report, as before."""
    cleaner = _cleaner()
    import pandas as pd
    source = _corpus_dir(tmp_path, 'good.md')
    (source / 'bad.md').write_bytes(b'\xff\xfe not utf-8 \xc3\x28')
    report = tmp_path / 'report.parquet'
    cleaner.run_complete_pipeline(str(source), str(tmp_path / 'out'), str(report), ['greek', 'latin'], 1, True,
                                  evidence_directory=str(tmp_path / 'evidence'))
    assert sorted(pd.read_parquet(report)['file_name']) == ['good.pdf']


def test_clean_stage_types_an_os_error_from_the_native_disk_full(tmp_path, monkeypatch):
    cleaner = _cleaner()
    from glossapi.stages.clean import CleanStage
    from glossapi.stages.contracts import ArtifactReference, StageRequest
    from glossapi.stages.execution import sha256_file
    import time
    source = tmp_path / 'in' / 'doc.md'
    source.parent.mkdir()
    source.write_text('Ελληνικό κείμενο.\n', encoding='utf-8')

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, 'Failed to write cleaned file: No space left on device (os error 28)')
    monkeypatch.setattr(cleaner, 'run_complete_pipeline', full)
    reference = ArtifactReference('f', 'selected_text', source, sha256_file(source), source.stat().st_size)
    request = StageRequest('t', 'a', 'l', 'd', 'v', 'clean.initial', (reference,), {'num_threads': 1}, {}, 'cpu',
                           tmp_path / 'work', time.monotonic_ns() + 60_000_000_000)
    result = CleanStage().run_document(request, reporter=None)
    assert (result.status, result.reason_code) == ('failed', 'disk_full')


# ---------------------------------------------------- a REAL full disk in clean (macOS RAM disk), typed + resumed ---

@pytest.fixture
def full_ram_disk(tmp_path):
    """A 1 MiB HFS+ RAM disk mounted under tmp_path, with work directories made first and then every block filled."""
    if sys.platform != 'darwin' or not shutil.which('hdiutil') or not shutil.which('diskutil') or not shutil.which('newfs_hfs'):
        pytest.skip('a real full filesystem needs macOS hdiutil (no privileges); Linux has no unprivileged equivalent')
    attached = subprocess.run(['hdiutil', 'attach', '-nomount', 'ram://2048'], capture_output=True, text=True, timeout=60)
    if attached.returncode != 0:
        pytest.skip(f'hdiutil attach failed: {attached.stderr.strip()}')
    device = attached.stdout.split()[0]
    mount = tmp_path / 'ram'
    mount.mkdir()
    try:
        subprocess.run(['newfs_hfs', '-v', 'k4gfix4' + uuid.uuid4().hex[:6], device], check=True, capture_output=True, timeout=60)
        subprocess.run(['diskutil', 'mount', '-mountPoint', str(mount), device], check=True, capture_output=True, timeout=60)
        work = mount / 'work'
        for name in ('clean_markdown', 'evidence'):
            (work / name).mkdir(parents=True)
        descriptor = os.open(mount / 'filler', os.O_WRONLY | os.O_CREAT)
        try:
            for size in (1 << 16, 512, 1):
                while True:
                    try:
                        os.write(descriptor, b'\0' * size)
                    except OSError as error:
                        assert error.errno == errno.ENOSPC, error
                        break
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
        yield work
    finally:
        subprocess.run(['hdiutil', 'detach', '-force', device], capture_output=True, timeout=120)


def test_real_full_disk_in_the_native_cleaner_raises_enospc(full_ram_disk, tmp_path):
    cleaner = _cleaner()
    source = _corpus_dir(tmp_path)
    with pytest.raises(OSError) as raised:
        cleaner.run_complete_pipeline(str(source), str(full_ram_disk / 'clean_markdown'), str(full_ram_disk / 'report.parquet'),
                                      ['greek', 'latin'], 1, True, evidence_directory=str(full_ram_disk / 'evidence'))
    assert raised.value.errno == errno.ENOSPC, raised.value


def test_full_disk_in_clean_is_typed_disk_full_and_the_resume_equals_the_single_pass(full_ram_disk, tmp_path, greek_fixture):  # noqa: F811
    _cleaner()
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    shutil.copyfile(greek_fixture['broken'], inputs / 'greek.pdf')
    shutil.copyfile(PDFS / 'alpha.pdf', inputs / 'alpha.pdf')
    sources = [inputs / 'alpha.pdf', inputs / 'greek.pdf']
    stages = ('extract', 'clean', 'export')
    reference = run_case(tmp_path / 'reference', sources, stages=stages, seed=True)
    retried = run_case(tmp_path / 'full', sources, stages=stages, seed=True, fault='cleaner_full_disk', target='greek',
                       full_dir=full_ram_disk, steps=('start', 'resume'))
    start, resume = retried['start'], retried['resume']
    assert start['state'] == 'completed_with_failures', start
    assert start['failed'] == [['clean.initial', 'disk_full']], start
    assert resume['state'] == 'completed' and resume['counts'] == reference['start']['counts'], resume
    assert_same_products(retried, reference)
