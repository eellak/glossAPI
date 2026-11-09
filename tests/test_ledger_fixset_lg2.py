"""Fix set LG2 (CP-LEDGER cycle 2): per-row mixed-manifest inventory (F3) and per-occurrence version and
disposition bindings (F4), plus the cycle-2 collision probes and the one-source_id-per-URL rule.

The credential parts of LG2 (allowlist persisted URL form, keyed resource identity, exact credential binding,
structural header redaction, one logging filter: F1/F2/F5) were withdrawn with the credential channel (owner,
1 Nov 2026). Credentialed URLs are refused at entry; see tests/test_k4_credential_refusal.py.
"""
import hashlib
import json
from pathlib import Path

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly
from glossapi.ledger.reader import report_why, select_run
from glossapi.stages.scheduler import StreamingScheduler
from test_ledger_fixset_lg import pdf, export_rows, cli

@pytest.fixture(autouse=True)
def log_in_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))

def stop_execution(monkeypatch):
    monkeypatch.setattr(StreamingScheduler, 'execute', lambda self, run, **kw:
        {'run_id':run, 'state':'waiting', 'attempts':{},
         'counts':{'planned':1,'running':0,'ok':0,'skipped':0,'failed':0}})

def test_six_collisions_sections_and_interrupted_publication_resume(tmp_path, monkeypatch):
    import glossapi.stages.run as run_module
    from glossapi.ledger.journal import source_key
    first = pdf(tmp_path/'0/same.pdf', 'ALPHA collision')
    key = source_key(first.resolve().as_uri())
    occupied = 'same__'+hashlib.sha256(key.encode()).hexdigest()[:10]
    inputs = [first] + [pdf(tmp_path/f'{i}/same.pdf', f'COLLISION input {i}') for i in range(1,4)]
    inputs.append(pdf(tmp_path/'4/same.md.pdf', 'Canonical suffix collision'))
    inputs.append(pdf(tmp_path/(occupied+'.pdf'), 'Natural hash-prefix name'))
    originals = {str(p):p.read_bytes() for p in inputs}
    models = Path(__import__('glossapi').__file__).parent/'models'
    def corpus():
        return Corpus(tmp_path/'empty', tmp_path/'output', section_classifier_model_path=models/'section_classifier.joblib',
                      extraction_model_path=models/'kmeans_weights.joblib')
    real = run_module.publish_barrier
    saved = {}
    def interrupt(c, run, stage, callback, **kwargs):
        saved['run'] = run
        if stage == 'clean':
            raise KeyboardInterrupt('independent interruption after extract publication')
        return real(c, run, stage, callback, **kwargs)
    monkeypatch.setattr(run_module, 'publish_barrier', interrupt)
    with pytest.raises(KeyboardInterrupt):
        corpus().run(inputs, stages=('extract','clean','sections','export'), settings={'extract':{'repair_fonts':False}})
    monkeypatch.setattr(run_module, 'publish_barrier', real)
    aliases = tmp_path/'output/.glossapi/publication-inputs'/saved['run']
    for alias in aliases.iterdir():
        alias.unlink()
    resumed = corpus().run(resume=saved['run'])
    rows = export_rows(tmp_path/'output')
    print('SIX_COLLISIONS', json.dumps({'counts':resumed['document_counts'], 'names':[r['filename'] for r in rows]}))
    assert resumed['document_counts'] == {'selected':6,'included':6,'omitted':0,'failed':0}
    assert len({r['filename'] for r in rows}) == len(rows) == 6
    assert occupied in {r['filename'] for r in rows}
    assert {str(p):p.read_bytes() for p in inputs} == originals
    assert len(list(aliases.iterdir())) == 5
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        for path in inputs:
            why = report_why(connection, str(path), run=select_run(connection))
            assert any(item['decision']=='include' for item in why['inclusion'])

def test_equal_content_colliding_sources_are_still_two_documents(tmp_path):
    first = pdf(tmp_path/'one/same.pdf', 'Identical document content')
    second = tmp_path/'two/same.pdf'; second.parent.mkdir(); second.write_bytes(first.read_bytes())
    result = Corpus(tmp_path/'empty', tmp_path/'output').run([first,second], stages=('extract','clean','export'),
        settings={'extract':{'repair_fonts':False}})
    assert result['document_counts'] == {'selected':2,'included':2,'omitted':0,'failed':0}
    assert len(export_rows(tmp_path/'output')) == 2

def test_duplicate_local_rows_including_missing_input_keep_occurrences(tmp_path):
    good = pdf(tmp_path/'good.pdf', 'Good duplicate content')
    missing = tmp_path/'missing.pdf'
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'documents':[
        {'path':str(good),'source_row':'same-row','catalog':'good-A'},
        {'path':str(good),'source_row':'same-row','catalog':'good-B'},
        {'path':str(missing),'source_row':'same-row','catalog':'missing-A'},
        {'path':str(missing),'source_row':'same-row','catalog':'missing-B'}]}))
    code, envelope = cli('run','--corpus',str(tmp_path/'output'),'--manifest',str(manifest),'--stages','extract,clean,export','--json')
    assert code == 1 and envelope['data']['document_counts'] == {'selected':2,'included':1,'omitted':1,'failed':0}
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM source_records').fetchone()[0] == 4
        for path in (good,missing):
            why = report_why(connection, str(path), run=select_run(connection))
            assert len(why['occurrences']) == 2
            assert len({o['version_id'] for o in why['occurrences']}) == 1

def test_mixed_manifest_keeps_each_local_occurrence_reportable(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    source = pdf(tmp_path/'local.pdf','Mixed manifest local file')
    manifest = {'documents':[
        {'path':str(source),'source_row':'L1','catalog':'local-A'},
        {'url':'https://example.invalid/remote.pdf','source_row':'R'},
        {'path':str(source),'source_row':'L2','catalog':'local-B'}]}
    output = tmp_path/'output'
    Corpus(tmp_path/'empty',output).run([source,'https://example.invalid/remote.pdf'],stages=('download',),
        settings={'source_manifest':manifest})
    with open_readonly(output/'corpus.sqlite3') as connection:
        records = list(connection.execute('SELECT source_uri,source_row,metadata_json FROM source_records'))
        print('MIXED_MANIFEST',json.dumps(records))
        why = report_why(connection,str(source),run=select_run(connection))
        assert len(why['occurrences']) == 2

def test_old_run_occurrence_does_not_claim_new_version(tmp_path):
    source = pdf(tmp_path/'source.pdf','Original version')
    original_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path/'output'
    one = Corpus(tmp_path/'empty',out).run([source],stages=('extract','clean','export'),settings={'extract':{'repair_fonts':False}})
    pdf(source,'Changed version')
    two = Corpus(tmp_path/'empty',out).run([source],stages=('extract','clean','export'),settings={'extract':{'repair_fonts':False}})
    with open_readonly(out/'corpus.sqlite3') as connection:
        why = report_why(connection,str(source),run=select_run(connection,two['run_id']))
        occurrences = why['occurrences']
        original_version = connection.execute('SELECT version_id FROM document_versions WHERE stored_sha256=?',(original_digest,)).fetchone()[0]
        print('CROSS_RUN_OCCURRENCES',json.dumps({'first':one['run_id'],'second':two['run_id'],
            'original_version':original_version,'reported_versions':[o['version_id'] for o in occurrences],
            'dispositions':[o['disposition'] for o in occurrences]}))
        assert occurrences[0]['version_id'] == original_version, 'Prior inventory occurrence is relabeled with the latest content version and disposition'

def test_later_run_replacement_keeps_immutable_prior_products(tmp_path):
    first = pdf(tmp_path/'first/same.pdf','Original ALPHA product')
    second = pdf(tmp_path/'second/same.pdf','Replacement BRAVO product')
    out = tmp_path/'output'
    one = Corpus(tmp_path/'empty',out).run([first],stages=('extract','clean','export'),settings={'extract':{'repair_fonts':False}})
    two = Corpus(tmp_path/'empty',out).run([second],stages=('extract','clean','export'),settings={'extract':{'repair_fonts':False}})
    with open_readonly(out/'corpus.sqlite3') as connection:
        why = report_why(connection,str(first),run=select_run(connection,one['run_id']))
        retained = list(connection.execute("SELECT uri,sha256 FROM current_file_state WHERE kind='document_jsonl' AND presence='present' AND uri LIKE '%/.glossapi/artifacts/%'"))
        from glossapi.stages.scheduler import local_path
        assert retained and all(local_path(uri).is_file() and hashlib.sha256(local_path(uri).read_bytes()).hexdigest()==digest for uri,digest in retained)
        copies = [local_path(uri) for uri,_ in retained]
        assert any('Original ALPHA' in p.read_text() for p in copies)
        assert any('Replacement BRAVO' in p.read_text() for p in copies)
        events = [encoded for encoded, in connection.execute('SELECT details_json FROM events') if 'compatibility_generation_superseded' in encoded]
        assert events
        print('CROSS_RUN_REPLACEMENT',json.dumps({'immutable_export_generations':len(retained),'supersession_events':len(events),'new_counts':two['document_counts']}))


# ------------------------------------------------- identity, F3 and F4 ---

def test_one_source_id_for_two_different_urls_is_rejected_before_deduplication(tmp_path):
    manifest = {'documents': [{'url': 'https://example.invalid/a.pdf?t=1', 'source_id': 'same'},
                              {'url': 'https://example.invalid/b.pdf?t=1', 'source_id': 'same'}]}
    with pytest.raises(ValueError, match='ambiguous_source_identity') as raised:
        Corpus(tmp_path/'empty', tmp_path/'output').run([row['url'] for row in manifest['documents']], stages=('download',),
            settings={'source_manifest': manifest})
    assert 'b.pdf?t=1' in str(raised.value)  # public URLs are named as given
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] == 0


def test_mixed_manifest_keeps_available_missing_and_duplicate_local_rows_beside_urls(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    good = pdf(tmp_path/'good.pdf', 'Mixed good local')
    missing = tmp_path/'missing.pdf'
    remote = 'https://example.invalid/remote.pdf?t=7'  # a public query is recorded as given
    manifest = {'documents': [
        {'path': str(good), 'source_row': 'G1', 'catalog': 'good-A'},
        {'url': remote, 'source_row': 'R1'},
        {'path': str(missing), 'source_row': 'M1', 'catalog': 'missing-A'},
        {'path': str(good), 'source_row': 'G2', 'catalog': 'good-B'},
        {'url': 'https://example.invalid/other.pdf', 'source_row': 'R2'},
        {'path': str(missing), 'source_row': 'M2', 'catalog': 'missing-B'}]}
    output = tmp_path/'output'
    Corpus(tmp_path/'empty', output).run([good, missing, remote, 'https://example.invalid/other.pdf'], stages=('download',),
        settings={'source_manifest': manifest})
    with open_readonly(output/'corpus.sqlite3') as connection:
        uris = [row[0] for row in connection.execute('SELECT source_uri FROM source_records')]
        assert not any(uri.endswith('/nan') for uri in uris) and len(uris) == 6 and remote in uris
        run = select_run(connection)
        for path, rows, catalogs in ((good, ['G1', 'G2'], ['good-A', 'good-B']), (missing, ['M1', 'M2'], ['missing-A', 'missing-B'])):
            why = report_why(connection, str(path), run=run)
            assert [o['source_row'] for o in why['occurrences']] == rows
            assert [next(c for c in catalogs if c in json.dumps(o['metadata'])) for o in why['occurrences']] == catalogs
            assert len({o['version_id'] for o in why['occurrences']}) == 1 and why['occurrences'][0]['version_id']
        assert any(item['reason_code'] == 'source_unavailable' for stage in report_why(connection, str(missing), run=run)['stages']
                   for item in stage['reasons']) or connection.execute(
            "SELECT COUNT(*) FROM events WHERE reason_code='source_unavailable'").fetchone()[0] == 1
        remote_why = report_why(connection, 'https://example.invalid/other.pdf', run=run)
        assert [o['source_row'] for o in remote_why['occurrences']] == ['R2']


def test_repeated_runs_report_each_occurrence_with_its_own_version_and_disposition(tmp_path):
    source = pdf(tmp_path/'source.pdf', 'Version one text')
    out = tmp_path/'output'
    settings = {'extract': {'repair_fonts': False}}
    one = Corpus(tmp_path/'empty', out).run([source], stages=('extract','clean','export'), settings=settings)
    source.write_bytes(b'not a PDF at all')
    two = Corpus(tmp_path/'empty', out).run([source], stages=('extract','clean','export'), settings=settings)
    pdf(source, 'Version three text')
    three = Corpus(tmp_path/'empty', out).run([source], stages=('extract','clean','export'), settings=settings)
    assert one['document_counts']['included'] == 1 and two['document_counts']['omitted'] == 1 and three['document_counts']['included'] == 1
    with open_readonly(out/'corpus.sqlite3') as connection:
        versions = [row[0] for row in connection.execute('SELECT version_id FROM document_versions ORDER BY revision')]
        assert len(versions) == 3
        latest = report_why(connection, str(source), run=select_run(connection, three['run_id']))['occurrences']
        assert [o['version_id'] for o in latest] == versions
        assert [o['execution_run_id'] for o in latest] == [one['run_id'], two['run_id'], three['run_id']]
        assert [o['disposition']['decision'] for o in latest] == ['include', 'export_canonical_markdown_missing', 'include']
        middle = report_why(connection, str(source), run=select_run(connection, two['run_id']))['occurrences']
        assert [o['version_id'] for o in middle] == versions[:2]
        assert [o['disposition']['decision'] for o in middle] == ['include', 'export_canonical_markdown_missing']
        first = report_why(connection, str(source), run=select_run(connection, one['run_id']))['occurrences']
        assert [(o['version_id'], o['disposition']['decision']) for o in first] == [(versions[0], 'include')]
