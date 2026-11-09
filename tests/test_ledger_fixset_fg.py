"""Fix set FG (final-gate gaps): multi-file local CLI runs with a partial
metadata parquet (FG-G01), deterministic OCR supplement markers (FG-G02) and
same-content filename aliases in report --why (FG-G04)."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from glossapi.ledger import open_readonly

TESTS = Path(__file__).resolve().parent


def cli(*arguments):
    from glossapi.cli import main
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(list(arguments))
    return code, json.loads(buffer.getvalue())


def text_pdf(path, text):
    from test_ledger_stages import simple_pdf
    simple_pdf(path)
    path.write_bytes(path.read_bytes().replace(b'A small text layer', text.encode('ascii')))
    return path


def failed_events(database):
    with open_readonly(database) as connection:
        return [row for row in connection.execute(
            "SELECT t.operation,e.reason_code,e.message FROM events e JOIN tasks t USING(task_id) WHERE e.status='failed'")]


# ---------------------------------------------------------------- FG-G01 ---

def partial_metadata(corpus_root):
    """A pre-existing metadata parquet that knows only a.pdf, with missing (NA) state cells."""
    directory = corpus_root/'download_results'
    directory.mkdir(parents=True)
    frame = pd.DataFrame({
        'url': ['file:///previous/a.pdf'],
        'filename': ['a.pdf'],
        'extract_status': pd.array([pd.NA], dtype='string'),
        'processing_stage': pd.array([pd.NA], dtype='string'),
    })
    frame.to_parquet(directory/'download_results.parquet', index=False)
    assert pd.read_parquet(directory/'download_results.parquet')['processing_stage'].isna().all()
    return directory/'download_results.parquet'


def test_cli_run_with_several_local_pdfs_and_partial_metadata_publishes(tmp_path):
    sources = tmp_path/'sources'; sources.mkdir()
    files = [text_pdf(sources/'a.pdf', 'Alpha first file'), text_pdf(sources/'b.pdf', 'Bravo second fil'),
             text_pdf(sources/'c.pdf', 'Charlie third fi')]
    corpus = tmp_path/'corpus'
    metadata = partial_metadata(corpus)
    code, envelope = cli('run', '--corpus', str(corpus), '--json', '--stages', 'extract,clean,export',
                         '--files', *map(str, files))
    assert failed_events(corpus/'corpus.sqlite3') == []
    assert code == 0 and envelope['data']['state'] == 'completed', envelope
    export = corpus/'export.jsonl'
    assert export.is_file(), 'the multi-file local CLI run published no export'
    records = [json.loads(line) for line in export.read_text().splitlines() if line.strip()]
    assert len(records) == 3
    frame = pd.read_parquet(metadata)
    assert sorted(frame['filename']) == ['a.pdf', 'b.pdf', 'c.pdf']
    # The row the user already had keeps its URL; a new row's unknown URL is null (the one metadata rule, K4BI-G9;
    # K4g fix 4), never NA text.
    assert frame.set_index('filename').loc['a.pdf', 'url'] == 'file:///previous/a.pdf'
    assert all('extract' in str(value).split(',') for value in frame['processing_stage'])


def test_processing_state_tolerates_missing_cells_in_every_reader(tmp_path):
    from glossapi.corpus.corpus_state import _ProcessingStateManager
    root = tmp_path/'corpus'
    metadata = partial_metadata(root)
    (root/'markdown').mkdir()
    manager = _ProcessingStateManager(root/'markdown'/'.processing_state.pkl')
    assert manager.load() == (set(), set())  # NA status/stage cells are "not processed", not an error
    manager.save({'a', 'b'}, {'c'})
    frame = pd.read_parquet(metadata).set_index('filename')
    assert frame.loc['a.pdf', 'extract_status'] == 'success' and frame.loc['a.pdf', 'processing_stage'] == 'extract'
    # A row added here has a null URL, as the artifact scan writes it (K4BI-G9; K4g fix 4: a retried document's row
    # then equals the single-pass row), never NA text.
    assert pd.isna(frame.loc['b.pdf', 'url']) and frame.loc['c.pdf', 'extract_status'] == 'problematic'
    processed, _ = manager.load()  # reading the rows it just wrote, including NA cells, still works
    assert {'a', 'b'} <= processed


# ---------------------------------------------------------------- FG-G02 ---

OCR_CHILD = r'''
import os, signal, sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from glossapi import Corpus
from glossapi.corpus.ocr import dispatch
from glossapi.ocr import page_artifacts

input_dir, output_dir, step = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
kill_page = int(os.environ.get('FG_KILL_AT_PAGE', '0'))

def convert(source, page_range, **_):
    first, last = page_range
    if kill_page and first <= kill_page <= last:
        os.kill(os.getpid(), signal.SIGKILL)  # a real kill inside OCR, after earlier ranges finished
    return SimpleNamespace(status='success', document=SimpleNamespace(
        export_to_markdown=lambda *, page_no: f'OCR TEXT FOR PAGE {page_no} Ελληνικό κείμενο σελίδας {page_no}'))

class FgOptions(SimpleNamespace):
    """Plugin options shaped like the Greek OCR options on the node: a set-derived
    list (order follows PYTHONHASHSEED) and absolute paths under this corpus."""
    def model_dump(self, mode='json'):
        return {'kind': 'fg-plugin', 'lang': ['el', 'en'], 'crop_scale': 3.0,
                'outputs': {'directory': None, 'files': list({'cells', 'lines', 'segments', 'checks', 'structure'})},
                'diagnostic_kinds': list({'alpha', 'beta', 'gamma', 'delta'}),
                'release_dir': str(output_dir/'models'), 'call_log': str(output_dir/'calls.jsonl')}

dispatch.entry_points = lambda **_: [SimpleNamespace(name='fg-plugin')]
dispatch._build_converter = lambda request: SimpleNamespace(convert=convert)
page_artifacts._page_count = lambda _: 4
corpus = Corpus(input_dir, output_dir)
if step == 'prepare':
    # Native text with no page boundaries forces the supplement composition.
    page_artifacts._write_outputs(output_dir, 'book', 'Native whole text without page markers.\n\nΔεύτερη παράγραφος.', 4)
    corpus.clean(drop_bad=False, num_threads=1)
    metadata = corpus._get_cached_metadata_parquet()
    frame = pd.read_parquet(metadata)
    frame['needs_ocr'] = True
    frame['ocr_pages'] = '2,4'
    frame.to_parquet(metadata, index=False)
elif step == 'ocr':
    corpus.ocr(mode='ocr_bad', ocr_options=FgOptions(kind='fg-plugin'))
elif step == 'export':
    corpus.jsonl(output_dir/'export.jsonl')
'''


def ocr_step(tmp_path, input_dir, output_dir, step, **environment):
    script = tmp_path/'fg_ocr_child.py'
    script.write_text(OCR_CHILD)
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(TESTS.parent/'src'), str(TESTS), os.environ.get('PYTHONPATH', '')]),
           'GLOSSAPI_CONVERSION_LOG_FILE': str(tmp_path/'conversion.log'), **environment}
    env.pop('GLOSSAPI_STAGE_PROFILE', None)
    return subprocess.run([sys.executable, str(script), str(input_dir), str(output_dir), step],
                          env=env, cwd=tmp_path, capture_output=True, text=True, timeout=600)


def ocr_corpus(tmp_path, name, *, seed, kill_seed=None):
    # Every arm is a different corpus directory, and every OCR process gets its own
    # PYTHONHASHSEED, so set-derived option lists come out in different orders.
    input_dir = tmp_path/f'{name}-input'; input_dir.mkdir()
    (input_dir/'book.pdf').write_bytes(b'%PDF-1.4 fixed source bytes for the FG-G02 marker\n%%EOF\n')
    output_dir = tmp_path/name
    done = ocr_step(tmp_path, input_dir, output_dir, 'prepare', PYTHONHASHSEED=seed)
    assert done.returncode == 0, done.stderr[-3000:]
    if kill_seed is not None:
        killed = ocr_step(tmp_path, input_dir, output_dir, 'ocr', FG_KILL_AT_PAGE='4', PYTHONHASHSEED=kill_seed)
        assert killed.returncode == -signal.SIGKILL, (killed.returncode, killed.stderr[-3000:])
    for step in ('ocr', 'export'):
        done = ocr_step(tmp_path, input_dir, output_dir, step, PYTHONHASHSEED=seed)
        assert done.returncode == 0, done.stderr[-3000:]
    return output_dir


def set_order(seed):
    return subprocess.run([sys.executable, '-c', "print(list({'cells','lines','segments','checks','structure'}))"],
                          env={**os.environ, 'PYTHONHASHSEED': seed}, capture_output=True, text=True, check=True).stdout


def test_supplement_exports_are_byte_identical_across_runs_and_after_a_kill(tmp_path):
    assert len({set_order(seed) for seed in ('2', '3', '0')}) == 3  # the seeds really reorder the set
    first, second = ocr_corpus(tmp_path, 'first', seed='2'), ocr_corpus(tmp_path, 'second', seed='3')
    killed = ocr_corpus(tmp_path, 'killed', seed='0', kill_seed='4')
    markdown = (first/'markdown/book.md').read_text()
    assert '<!-- ocr-supplement-begin: pages=2,4 ' in markdown and 'OCR TEXT FOR PAGE 4' in markdown
    exports = [(root/'export.jsonl').read_bytes() for root in (first, second, killed)]
    assert exports[0], 'no export'
    assert exports[0] == exports[1], 'two independent runs exported different bytes'
    assert exports[0] == exports[2], 'kill plus rerun exported different bytes'
    for relative in ('markdown/book.md', 'clean_markdown/book.md', 'sidecars/ocr_pages/book.json'):
        assert (first/relative).read_bytes() == (second/relative).read_bytes() == (killed/relative).read_bytes(), relative


def assemble_supplement(root, pdf_bytes, page_text, configuration=None):
    from glossapi.ocr import page_artifacts
    out = root/'out'; (out/'markdown').mkdir(parents=True); (out/'json/metrics').mkdir(parents=True)
    source = root/'case.pdf'; source.write_bytes(pdf_bytes)
    page_artifacts._write_outputs(out, 'case', 'Native body without markers', 3)
    page_artifacts._write_outputs(out, 'case__p00002-00002', page_text, 1,
                                  extra_metrics={'source_start_page': 2, 'source_end_page': 2, 'ocr_engine': 'x', 'device': 'cpu'})
    keywords = {} if configuration is None else {'ocr_configuration': configuration}
    assert page_artifacts._reassemble_canonical_output_for_source(
        out_root=out, pdf_path=source, source_name='case.pdf', selected_ranges=[(2, 2)], **keywords)
    text = (out/'markdown/case.md').read_text()
    return re.search(r'<!-- ocr-supplement-begin: pages=2 run_id=([0-9a-f]{32}) -->', text)[1], out


def test_supplement_marker_is_content_derived(tmp_path, monkeypatch):
    from glossapi.ocr import page_artifacts
    monkeypatch.setattr(page_artifacts, '_page_count', lambda _: 3)
    base, out = assemble_supplement(tmp_path/'a', b'%PDF one', 'page two text')
    assert base == assemble_supplement(tmp_path/'b', b'%PDF one', 'page two text')[0], 'same content, different marker'
    assert base != assemble_supplement(tmp_path/'c', b'%PDF two', 'page two text')[0]  # document hash
    assert base != assemble_supplement(tmp_path/'d', b'%PDF one', 'other page text')[0]  # OCR output
    configured = assemble_supplement(tmp_path/'e', b'%PDF one', 'page two text', {'model_revision': 'r1'})[0]
    assert configured != assemble_supplement(tmp_path/'f', b'%PDF one', 'page two text', {'model_revision': 'r2'})[0]
    metrics = json.loads((out/'json/metrics/case.metrics.json').read_text())
    state = json.loads((out/'sidecars/ocr_pages/case.json').read_text())
    assert metrics['ocr_run_id'] == base == state['pages']['2']['run_id']


def test_assembly_configuration_ignores_paths_and_hash_order():
    from glossapi.ocr.page_artifacts import ocr_configuration
    def options(root, files, kinds):
        return {'kind': 'glossapi', 'lang': ['el', 'en'], 'model_revision': 'r1', 'crop_scale': 3.0,
                'outputs': {'directory': None, 'files': files}, 'extra_kinds': kinds,
                'checkpoint': None, 'codec': None, 'release_dir': f'{root}/release', 'ornament_seeds': [f'{root}/seed.png'],
                'hole_seeds': [], 'routing_sidecar': f'{root}/.glossapi/staging/a/routing.jsonl', 'call_log': f'{root}/calls.jsonl'}
    one = ocr_configuration(options('/runs/a', ['cells', 'segments', 'structure', 'checks', 'lines'], ['x', 'y']), 'cpu')
    two = ocr_configuration(options('/other/b', ['cells', 'lines', 'segments', 'checks', 'structure'], ['y', 'x']), 'cpu')
    assert one == two
    assert not any(key in one for key in ('outputs', 'release_dir', 'ornament_seeds', 'routing_sidecar', 'call_log'))
    assert '/runs/a' not in json.dumps(one)
    assert ocr_configuration({'model_revision': 'r2'}, 'cpu') != ocr_configuration({'model_revision': 'r1'}, 'cpu')


def test_text_metrics_do_not_count_supplement_markers(tmp_path, monkeypatch):
    from glossapi.ocr import page_artifacts
    from glossapi.stages.clean import score_ocr_profile_directory
    import glossapi_rs_noise as noise
    monkeypatch.setattr(page_artifacts, '_page_count', lambda _: 3)
    _, out = assemble_supplement(tmp_path/'a', b'%PDF one', 'page two text')
    marked = (out/'markdown/case.md').read_text()
    plain = tmp_path/'plain'; plain.mkdir()
    (plain/'case.md').write_text(page_artifacts.strip_supplement_markers(marked))
    assert 'ocr-supplement' not in (plain/'case.md').read_text() and 'page two text' in (plain/'case.md').read_text()
    scored = score_ocr_profile_directory(noise, out/'markdown', 1, 6)
    expected = list(noise.score_markdown_directory_ocr_profile(str(plain), 1, 6))
    assert len(scored) == 1 and scored[0]['path'] == str(out/'markdown/case.md')
    assert {k: v for k, v in scored[0].items() if k != 'path'} == {k: v for k, v in expected[0].items() if k != 'path'}
    counted = list(noise.score_markdown_directory_ocr_profile(str(out/'markdown'), 1, 6))[0]
    assert counted['latin_char_count'] > scored[0]['latin_char_count']  # the markers were Latin text before


# ---------------------------------------------------------------- FG-G04 ---

def api_corpus(tmp_path, name, text):
    from glossapi import Corpus
    input_dir = tmp_path/f'{name}-input'; input_dir.mkdir()
    text_pdf(input_dir/'doc.pdf', text)
    corpus = Corpus(input_dir, tmp_path/name)
    corpus.extract(input_format='pdf'); corpus.clean(); corpus.jsonl(tmp_path/name/'export.jsonl')
    return corpus


def test_report_why_filename_resolves_same_content_aliases_to_one_document(tmp_path):
    from glossapi.ledger.reader import report_why, select_run
    corpus = api_corpus(tmp_path, 'api', 'Same bytes alias')
    database = corpus.output_dir/'corpus.sqlite3'
    with open_readonly(database) as connection:
        documents = [row[0] for row in connection.execute('SELECT document_id FROM documents ORDER BY rowid')]
        assert len(documents) == 2  # ...-input/doc.pdf and .../downloads/doc.pdf
        why = report_why(connection, 'doc.pdf', run=select_run(connection))
    document = why['document']
    aliases = document['aliases']
    assert {document['document_id'], *[alias['document_id'] for alias in aliases]} == set(documents)
    assert document['alias_basis'] == 'same_source_sha256'
    uris = {record['source_uri'] for record in document['source_records']} | {f['path'] for f in why['files']}
    assert any('api-input' in uri for uri in uris) and any('/downloads/doc.pdf' in uri for uri in uris)
    assert any(stage['operation'] == 'extract.document' and stage['status'] == 'ok' for stage in why['stages'])
    code, envelope = cli('report', '--corpus', str(corpus.output_dir), '--why', 'doc.pdf')
    assert code == 0, envelope['errors']
    assert envelope['data']['document']['document_id'] == document['document_id']


def test_report_why_filename_stays_ambiguous_for_different_content(tmp_path):
    from glossapi.ledger.writer import LedgerWriter
    from glossapi.ledger.identity import source_key
    database = tmp_path/'corpus.sqlite3'
    with LedgerWriter(database) as writer, writer.transaction():
        run = writer.run({}, {}, {}, code_revision='test')
        for folder, digest in (('one', 'a'*64), ('two', 'b'*64)):
            uri = f'file:///{folder}/doc.pdf'
            document = writer.document(source_key(uri))
            writer.version(document, source_sha256=digest)
            writer.source_record(document, run, uri, {})
    code, envelope = cli('report', '--corpus', str(tmp_path), '--why', 'doc.pdf')
    assert code == 2 and envelope['errors'][0]['code'] == 'ambiguous_file'
    assert len(envelope['errors'][0]['details']['candidate_document_ids']) == 2
