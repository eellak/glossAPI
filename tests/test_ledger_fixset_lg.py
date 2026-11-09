"""Fix set LG: CP-LEDGER cycle-1 findings F1 (collision-free publication and
final census), F3 (missing local input keeps the census), F4 (local-manifest
occurrences) and F7 (bounded OCR marker scoring). The reviewer's probes are
ported unchanged. F2 (URL credentials never recorded) was withdrawn with the
credential channel (owner, 1 Nov 2026): credentialed URLs are refused at entry
(tests/test_k4_credential_refusal.py)."""
import contextlib
import io
import json
import os
from pathlib import Path

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

TESTS = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def pdf(path, text):
    from test_ledger_stages import simple_pdf
    path.parent.mkdir(parents=True, exist_ok=True)
    simple_pdf(path)
    path.write_bytes(path.read_bytes().replace(b'A small text layer', text.encode('ascii')))
    return path


# ------------------------------------------------- reviewer probes (ported) ---

def test_every_requested_input_remains_reportable_when_local_file_is_missing(tmp_path):
    good = pdf(tmp_path/'good.pdf', 'First input exists')
    missing = tmp_path/'missing.pdf'
    corpus = Corpus(tmp_path/'empty', tmp_path/'output')
    error = None
    try:
        result = corpus.run([good, missing], stages=('extract','clean','export'),
            settings={'extract': {'repair_fonts': False}})
    except Exception as exc:
        error = str(exc)
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
        sources = list(connection.execute('SELECT source_uri FROM source_records'))
        tasks = list(connection.execute('SELECT operation FROM tasks'))
        from glossapi.ledger.reader import report_why, select_run, ReadRequestError
        try:
            why = report_why(connection, str(missing), run=select_run(connection))
        except ReadRequestError as exc:
            why = exc.code
    print('MISSING_INPUT', json.dumps(dict(error=error, sources=sources, tasks=tasks, why=why)))
    assert len(sources) == 2, 'The input transaction rolled back every source instead of retaining the requested census'


def test_two_local_pdfs_with_the_same_basename_export_their_own_text(tmp_path):
    first = pdf(tmp_path/'first/same.pdf', 'ALPHA first input.')
    second = pdf(tmp_path/'second/same.pdf', 'BRAVO other input.')
    corpus = Corpus(tmp_path/'empty', tmp_path/'output')
    result = corpus.run([first, second], stages=('extract','clean','export'),
        settings={'extract': {'repair_fonts': False}})
    export = corpus.output_dir/'export.jsonl'
    rows = [json.loads(line) for line in export.read_text().splitlines()] if export.exists() else []
    from glossapi.ledger.reader import report_why, select_run
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
        why = {str(path): report_why(connection, str(path), run=select_run(connection))['inclusion']
               for path in (first, second)}
    print('BASENAME_COLLISION', json.dumps(dict(state=result['state'], counts=result.get('document_counts'), rows=rows, why=why)))
    assert len(rows) == 2, 'Two distinct source identities were collapsed during compatibility publication'
    # Port note: export records carry their text in the ``document`` field; the
    # probe's ``row.get('text','')`` could never match, so the port reads it.
    assert any('ALPHA first input' in row.get('document', row.get('text','')) for row in rows)
    assert any('BRAVO other input' in row.get('document', row.get('text','')) for row in rows)


def test_local_manifest_retains_each_source_occurrence_in_source_records(tmp_path):
    from glossapi.cli import main
    source = pdf(tmp_path/'given.pdf', 'One local document')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'documents':[
        {'path':str(source),'catalog_id':'original-row-A','source_row':'A'},
        {'path':str(source),'catalog_id':'original-row-B','source_row':'B'}]}))
    output = tmp_path/'output'
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(['run','--corpus',str(output),'--manifest',str(manifest),'--stages','extract,clean,export','--json'])
    envelope = json.loads(buffer.getvalue())
    with open_readonly(output/'corpus.sqlite3') as connection:
        records = [dict(zip(('source_uri','source_row','metadata_json'), row)) for row in connection.execute(
            'SELECT source_uri,source_row,metadata_json FROM source_records')]
        snapshot_kinds_with_original_rows = [kind for kind, encoded in connection.execute('SELECT kind,json FROM snapshots')
                                            if 'original-row-A' in encoded and 'original-row-B' in encoded]
    print('MANIFEST_OCCURRENCES', json.dumps(dict(exit=code,counts=envelope.get('data',{}).get('document_counts'),records=records,
        snapshot_kinds_with_original_rows=snapshot_kinds_with_original_rows)))
    assert len([row for row in records if 'original-row-' in row['metadata_json']]) == 2


# ---------------------------------------------------------------------- F1 ---

def cli(*arguments):
    from glossapi.cli import main
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(list(arguments))
    return code, json.loads(buffer.getvalue())


def export_rows(output):
    path = Path(output)/'export.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_cli_same_basename_inputs_get_distinct_records_counts_and_explanations(tmp_path):
    first = pdf(tmp_path/'first/same.pdf', 'ALPHA first input.')
    second = pdf(tmp_path/'second/same.pdf', 'BRAVO other input.')
    unique = pdf(tmp_path/'third/unique.pdf', 'CHARLIE unique input.')
    output = tmp_path/'output'
    code, envelope = cli('run', '--corpus', str(output), '--files', str(first), str(second), str(unique),
                         '--stages', 'extract,clean,export', '--json')
    assert code == 0, envelope['errors']
    data = envelope['data']
    assert data['state'] == 'completed'
    assert data['document_counts'] == {'selected': 3, 'included': 3, 'omitted': 0, 'failed': 0}
    rows = export_rows(output)
    assert len(rows) == 3
    texts = {row['filename']: row['document'] for row in rows}
    # The unique input keeps exactly its current name; colliding inputs get their own.
    assert 'unique' in texts and 'CHARLIE unique input' in texts['unique']
    colliding = sorted(name for name in texts if name != 'unique')
    assert len(colliding) == 2 and all(name.startswith('same__') for name in colliding)
    assert sorted('ALPHA' in texts[name] for name in colliding) == [False, True]
    assert (output/'clean_markdown/unique.md').is_file() and not (output/'clean_markdown/same.md').exists()
    with open_readonly(output/'corpus.sqlite3') as connection:
        from glossapi.ledger.reader import report_why, select_run
        for path, word in ((first, 'ALPHA'), (second, 'BRAVO')):
            why = report_why(connection, str(path), run=select_run(connection))
            decisions = [item for item in why['inclusion'] if item['product'] == 'document_jsonl']
            assert [item['decision'] for item in decisions] == ['include']
            record = next(row for row in rows if word in row['document'])
            assert decisions[0]['record_hash'] is not None
            assert any(Path(f['path']).name == record['filename']+'.md' for f in why['files'] if f['kind'] == 'clean_markdown')


def test_unique_basenames_keep_their_names_and_receive_no_publication_name(tmp_path):
    from glossapi.stages.run import _publication_names
    entries = [dict(key='local_path:v1:file:///a/x.pdf', path=Path('/a/x.pdf')),
               dict(key='local_path:v1:file:///b/y.pdf', path=Path('/b/y.pdf'))]
    assert _publication_names(entries, []) == {}
    colliding = entries + [dict(key='local_path:v1:file:///c/x.pdf', path=Path('/c/x.pdf'))]
    names = _publication_names(colliding, [])
    assert set(names) == {'local_path:v1:file:///a/x.pdf', 'local_path:v1:file:///c/x.pdf'}
    assert len(set(names.values())) == 2 and all(name.startswith('x__') and name.endswith('.pdf') for name in names.values())
    # Deterministic per source identity, independent of order.
    assert _publication_names(list(reversed(colliding)), []) == names


def test_ambiguous_projection_is_rejected_and_census_reconciles_to_export(tmp_path, monkeypatch):
    import glossapi.stages.run as run_module
    # Simulate an unresolved name collision: the guard must refuse the second
    # projection instead of overwriting the first, and the final census must
    # reflect the export, not DAG completion.
    monkeypatch.setattr(run_module, '_publication_names', lambda entries, remote, claimed=None: {})
    first = pdf(tmp_path/'first/same.pdf', 'ALPHA first input.')
    second = pdf(tmp_path/'second/same.pdf', 'BRAVO other input.')
    corpus = Corpus(tmp_path/'empty', tmp_path/'output')
    result = corpus.run([first, second], stages=('extract','clean','export'), settings={'extract': {'repair_fonts': False}})
    assert result['state'] == 'completed_with_failures'
    assert result['document_counts']['included'] == 0 and result['document_counts']['omitted'] == 2
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
        failures = [row[0] for row in connection.execute(
            "SELECT e.message FROM events e JOIN tasks t USING(task_id) WHERE t.operation='extract.compatibility_publish' AND e.status='failed'")]
        assert failures and 'Ambiguous publication' in failures[-1]
        from glossapi.ledger.reader import report_why, select_run
        for path in (first, second):
            why = report_why(connection, str(path), run=select_run(connection))
            assert any(item['decision'] == 'export_canonical_markdown_missing' for item in why['inclusion'])
    assert not export_rows(corpus.output_dir)


def test_claim_publication_refuses_a_second_version_on_one_location(tmp_path):
    from types import SimpleNamespace
    from glossapi.stages.compatibility import claim_publication
    corpus = SimpleNamespace(_publication_claims={})
    target = tmp_path/'markdown/same.md'; target.parent.mkdir()
    record = claim_publication(corpus, target, 'v1'); target.write_text('first'); record()
    claim_publication(corpus, target, 'v1')()  # the same version may republish
    with pytest.raises(ValueError, match='Ambiguous publication'):
        claim_publication(corpus, target, 'v2')
    assert target.read_text() == 'first'
    assert claim_publication(SimpleNamespace(_publication_claims=None), target, 'v2')() is None


# ---------------------------------------------------------------------- F3 ---

def test_missing_and_unreadable_inputs_are_their_own_outcomes_and_others_proceed(tmp_path):
    good = pdf(tmp_path/'good.pdf', 'First input exists')
    missing = tmp_path/'missing.pdf'
    unreadable = pdf(tmp_path/'locked.pdf', 'Locked input')
    unreadable.chmod(0)
    try:
        if os.access(unreadable, os.R_OK):
            pytest.skip('running with privileges that ignore file modes')
        output = tmp_path/'output'
        code, envelope = cli('run', '--corpus', str(output), '--files', str(good), str(missing), str(unreadable),
                             '--stages', 'extract,clean,export', '--json')
    finally:
        unreadable.chmod(0o644)
    data = envelope['data']
    assert code == 1 and data['state'] == 'completed_with_failures', envelope['errors']
    assert data['document_counts'] == {'selected': 3, 'included': 1, 'omitted': 2, 'failed': 0}
    assert data['source_inventory'] == {'unavailable': 2}
    rows = export_rows(output)
    assert len(rows) == 1 and 'First input exists' in rows[0]['document']
    with open_readonly(output/'corpus.sqlite3') as connection:
        from glossapi.ledger.reader import report_why, select_run
        for path, cause in ((missing, 'local_source_absent'), (unreadable, 'local_source_unreadable')):
            why = report_why(connection, str(path), run=select_run(connection))
            reasons = [reason for stage in why['stages'] for reason in stage['reasons']]
            assert {stage['status'] for stage in why['stages'] if stage['scope'] == 'document'} == {'skipped'}
            assert any(reason['reason_code'] == 'source_unavailable' for reason in reasons)
            events = [row for row in connection.execute(
                "SELECT details_json FROM events WHERE reason_code='source_unavailable' AND event_kind='decision'")]
            assert any(cause in row[0] and str(path) in row[0] for row in events)
            assert any(item['decision'] == 'export_canonical_markdown_missing' for item in why['inclusion'])
            assert any(f['presence'] == 'missing' for f in why['files'])
            assert why['occurrences'][0]['disposition']['decision'] == 'export_canonical_markdown_missing'


def test_inventory_is_durable_before_per_file_inspection(tmp_path, monkeypatch):
    """A crash during hashing leaves the requested census, not an empty ledger."""
    import glossapi.stages.run as run_module
    good = pdf(tmp_path/'good.pdf', 'First input exists')
    other = pdf(tmp_path/'other.pdf', 'Second input exists')
    real = run_module.sha256_file
    def crash(path):
        if Path(path).name in ('good.pdf', 'other.pdf'):
            raise KeyboardInterrupt('simulated death during inspection')
        return real(path)
    monkeypatch.setattr(run_module, 'sha256_file', crash)
    corpus = Corpus(tmp_path/'empty', tmp_path/'output')
    with pytest.raises(KeyboardInterrupt):
        corpus.run([good, other], stages=('extract','clean','export'), settings={'extract': {'repair_fonts': False}})
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
        assert sorted(Path(row[0]).name for row in connection.execute('SELECT source_uri FROM source_records')) == ['good.pdf', 'other.pdf']
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] == 2


# ---------------------------------------------------------------------- F4 ---

def test_local_manifest_occurrences_link_rows_to_version_and_final_disposition(tmp_path):
    source = pdf(tmp_path/'given.pdf', 'One local document')
    other = pdf(tmp_path/'other.pdf', 'Another local document')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'documents': [
        {'path': str(source), 'catalog_id': 'original-row-A', 'source_row': 'A', 'year': 1901},
        {'path': str(other), 'catalog_id': 'original-row-C', 'source_row': 'C'},
        {'path': str(source), 'catalog_id': 'original-row-B', 'source_row': 'B', 'year': None}]}))
    output = tmp_path/'output'
    code, envelope = cli('run', '--corpus', str(output), '--manifest', str(manifest), '--stages', 'extract,clean,export', '--json')
    assert code == 0, envelope['errors']
    assert envelope['data']['document_counts'] == {'selected': 2, 'included': 2, 'omitted': 0, 'failed': 0}
    with open_readonly(output/'corpus.sqlite3') as connection:
        from glossapi.ledger.reader import report_why, select_run
        rows = [(row[0], json.loads(row[1])) for row in connection.execute('SELECT source_row,metadata_json FROM source_records')]
        assert sorted(identity for identity, _ in rows) == ['A', 'B', 'C']
        why = report_why(connection, str(source), run=select_run(connection))
        occurrences = why['occurrences']
        assert sorted(item['source_row'] for item in occurrences) == ['A', 'B']
        encoded = json.dumps([item['metadata'] for item in occurrences])
        assert 'original-row-A' in encoded and 'original-row-B' in encoded and '1901' in encoded
        assert {item['version_id'] for item in occurrences} == {why['document']['version_id']}
        assert all(item['disposition']['decision'] == 'include' for item in occurrences)
        second = next(item for item in occurrences if item['source_row'] == 'B')
        assert '"duplicate_of_occurrence"' in json.dumps(second['metadata'])


# ---------------------------------------------------------------------- F7 ---

class _Noise:
    def __init__(self):
        self.calls = []
    def score_markdown_directory_ocr_profile(self, directory, threads, min_repeat_run):
        directory = Path(directory)
        files = sorted(path for path in directory.rglob('*.md'))
        self.calls.append((directory, [str(path.relative_to(directory)) for path in files]))
        return [{'path': str(path), 'latin_percentage': float(path.read_bytes().count(b'ocr'))} for path in files]


def test_marker_scoring_streams_files_and_copies_only_marked_files_into_owned_scratch(tmp_path, monkeypatch):
    from glossapi.stages.clean import score_ocr_profile_directory
    import tempfile
    directory = tmp_path/'markdown'; (directory/'nested').mkdir(parents=True)
    (directory/'plain.md').write_text('plain text')
    (directory/'nested/other.md').write_text('other text')
    noise = _Noise()
    monkeypatch.setattr(tempfile, 'TemporaryDirectory', lambda *a, **k: pytest.fail('unaccounted temporary directory'))
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda *a, **k: pytest.fail('unaccounted temporary directory'))
    owned = tmp_path/'attempt'; owned.mkdir()
    rows = score_ocr_profile_directory(noise, directory, 1, 6, scratch_root=owned)
    assert len(noise.calls) == 1 and noise.calls[0][0] == directory and len(rows) == 2
    (directory/'marked.md').write_text('text\n<!-- ocr-pages: begin run=1 -->\nocr body\n<!-- ocr-pages: end -->\n')
    noise = _Noise()
    rows = score_ocr_profile_directory(noise, directory, 1, 6, scratch_root=owned)
    assert len(noise.calls) == 2
    scratch, copied = noise.calls[1]
    assert scratch.parent == owned and copied == ['marked.md'], 'only the marked file is copied'
    assert not scratch.exists() and list(owned.iterdir()) == []
    by_name = {Path(row['path']).name: row for row in rows}
    assert set(by_name) == {'plain.md', 'other.md', 'marked.md'}
    assert all(row['path'].startswith(str(directory)) for row in rows)
    from glossapi.ocr.page_artifacts import strip_supplement_markers
    assert by_name['marked.md']['latin_percentage'] == float(strip_supplement_markers((directory/'marked.md').read_text()).count('ocr'))


def test_marker_detection_finds_markers_across_read_blocks(tmp_path):
    from glossapi.stages.clean import _has_supplement_marker
    path = tmp_path/'large.md'
    path.write_bytes(b'x'*((1 << 20)-4)+b'<!-- ocr-pages -->')
    assert _has_supplement_marker(path)
    path.write_bytes(b'x'*((1 << 20)+10))
    assert not _has_supplement_marker(path)


# ------------------------------------------------- F1: sections and resume ---

def test_same_basename_inputs_publish_distinct_sections_and_survive_resume(tmp_path, monkeypatch):
    from glossapi.stages.scheduler import StreamingScheduler
    models = Path(__import__('glossapi').__file__).parent/'models'
    first = pdf(tmp_path/'first/same.pdf', 'ALPHA first input.')
    second = pdf(tmp_path/'second/same.pdf', 'BRAVO other input.')
    original = StreamingScheduler.execute
    monkeypatch.setattr(StreamingScheduler, 'execute', lambda self, run, **kw:
        {'run_id':run,'state':'waiting','attempts':{},'counts':{'planned':1,'running':0,'ok':0,'skipped':0,'failed':0}})
    corpus = lambda: Corpus(tmp_path/'empty', tmp_path/'output', section_classifier_model_path=models/'section_classifier.joblib',
                            extraction_model_path=models/'kmeans_weights.joblib')
    frozen = corpus().run([first, second], stages=('extract','clean','sections','export'), settings={'extract': {'repair_fonts': False}})
    monkeypatch.setattr(StreamingScheduler, 'execute', original)
    result = corpus().run(resume=frozen['run_id'])
    assert result['state'] == 'completed', result
    assert result['document_counts'] == {'selected': 2, 'included': 2, 'omitted': 0, 'failed': 0}
    rows = export_rows(tmp_path/'output')
    assert sorted('ALPHA' in row['document'] for row in rows) == [False, True]
    import pandas as pd
    sections = pd.read_parquet(next((tmp_path/'output/sections').glob('*.parquet')))
    assert sorted(sections['filename'].unique()) == sorted(row['filename'] for row in rows)
    aliases = sorted((tmp_path/'output/.glossapi/publication-inputs'/frozen['run_id']).iterdir())
    assert [alias.name for alias in aliases] == sorted(row['filename']+'.pdf' for row in rows)
    assert {os.readlink(alias) for alias in aliases} == {str(first.resolve()), str(second.resolve())}


def test_marker_scoring_copies_only_marked_files_with_the_original_signature(tmp_path):
    """Behavioural F7 probe with the f9a842f call signature: no whole-directory copy."""
    from glossapi.stages.clean import score_ocr_profile_directory
    directory = tmp_path/'markdown'; directory.mkdir()
    for index in range(5):
        (directory/f'plain{index}.md').write_text('plain text '+str(index))
    (directory/'marked.md').write_text('text\n<!-- ocr-pages: begin run=1 -->\nocr body\n<!-- ocr-pages: end -->\n')
    noise = _Noise()
    rows = score_ocr_profile_directory(noise, directory, 1, 6)
    assert len(rows) == 6
    copies = [listing for scored, listing in noise.calls if scored != directory]
    assert copies == [['marked.md']], 'scratch must hold only marked files, not a copy of the whole directory'
