"""Fix set LG2 (CP-LEDGER cycle 2): allowlist persisted URL form, keyed resource
identity, exact credential binding, structural header redaction and one logging
filter (F1/F2/F5); per-row mixed-manifest inventory (F3); per-occurrence
version and disposition bindings (F4).

Part 1 is the reviewer's probe file review_scratch/CP-LEDGERc2/test_cycle2_adversarial.py
ported verbatim, with one documented adaptation (see
test_pending_resume_refuses_before_any_dispatch_and_rotation_matches).
Part 2 holds this fix set's own tests.
"""
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import LedgerWriter, open_readonly
from glossapi.ledger.reader import report_why, select_run
from glossapi.stages.scheduler import StreamingScheduler
from test_ledger_fixset_lg import pdf, export_rows, cli

MARKER = 'CP_LEDGER_C2_SYNTHETIC_SECRET'

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

def test_protocol_relative_redirect_credentials_never_reach_ledger(tmp_path):
    from glossapi.stages.receipts import capture_transport, request_receipt, response_receipt
    start = 'https://origin.invalid/start.pdf'
    first = SimpleNamespace(url=start, status=302, headers={'Location':'//reviewer:'+MARKER+'@next.invalid/a.pdf?sig='+MARKER})
    second = SimpleNamespace(url='https://reviewer:'+MARKER+'@next.invalid/a.pdf?sig='+MARKER,
        status=307, headers={'Location':'/final.pdf?access_token='+MARKER})
    response = SimpleNamespace(url='https://next.invalid/final.pdf?access_token='+MARKER,
        status=200, headers={'Content-Type':'application/pdf'}, history=[first,second])
    with capture_transport('http',{}) as state:
        with request_receipt(start):
            response_receipt(response,start)
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with writer.transaction():
            digest = writer.snapshot('transport_probe', state)
            run = writer.run({}, {}, {}, code_revision='cycle2-test')
            document = writer.document('url:v1:'+start)
            version = writer.version(document)
            writer.source_record(document,run,start,{})
            task = writer.task(run,'download','download.acquire',{},version_id=version)
            attempt = writer.attempt(task,worker_id='probe',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
            writer.finish(run,task,attempt,status='failed',reason_code='download_exception',details={'http':state})
        encoded = writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()[0]
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        why = report_why(connection,start,run=select_run(connection,run))
    hits = [r for r in state['requests'][0]['redirects'] if MARKER in json.dumps(r)]
    print('PROTOCOL_RELATIVE', json.dumps({'leaky_redirects':hits,'persisted':MARKER in encoded,
        'report_why_contains_marker':MARKER in json.dumps(why)}))
    assert MARKER not in encoded, 'Network-path redirect userinfo credentials survive the actual transport and ledger boundaries'

def test_error_receipt_and_downloader_message_redact_standard_fields(tmp_path):
    from glossapi.stages.receipts import capture_transport, request_receipt
    from glossapi.gloss_downloader import _private_logger
    url = 'https://reviewer:'+MARKER+'@h.invalid/f.pdf?access_token='+MARKER
    with capture_transport('http',{}) as state:
        with pytest.raises(ValueError), request_receipt(url):
            raise ValueError('failed request '+url+' relative /x.pdf?sig='+MARKER)
    assert MARKER not in json.dumps(state)
    out = io.StringIO(); logger = _private_logger('c2-log',logging.DEBUG,propagate=False)
    logger.addHandler(logging.StreamHandler(out)); logger.error('request %s failed',url)
    assert MARKER not in out.getvalue()

def test_downloader_exception_log_redacts_credentials(tmp_path):
    from glossapi.gloss_downloader import _private_logger
    out = io.StringIO(); logger = _private_logger('c2-exception-log',logging.DEBUG,propagate=False)
    logger.addHandler(logging.StreamHandler(out))
    try:
        raise ValueError('https://h.invalid/f.pdf?access_token='+MARKER)
    except ValueError:
        logger.error('download failed',exc_info=True)
    print('EXCEPTION_LOG',json.dumps({'persisted_marker':MARKER in out.getvalue()}))
    assert MARKER not in out.getvalue()

def test_actual_http_error_logs_redact_authorization_and_set_cookie(tmp_path):
    from glossapi.gloss_downloader import GlossDownloader, _private_logger
    from aiohttp import ClientResponseError, RequestInfo
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL
    url = 'https://example.invalid/a.pdf'
    info = RequestInfo(URL(url),'GET',CIMultiDictProxy(CIMultiDict({'Authorization':'Bearer '+MARKER})),URL(url))
    error = ClientResponseError(info,(),status=403,message='Forbidden',headers={'Set-Cookie':'sessionid='+MARKER})
    out = io.StringIO(); downloader = GlossDownloader.__new__(GlossDownloader)
    downloader.logger = _private_logger('c2-http-error',logging.DEBUG,propagate=False)
    downloader.logger.addHandler(logging.StreamHandler(out)); downloader.verbose = True
    downloader._build_http_error_result(url,error,0)
    print('HTTP_ERROR_LOG',json.dumps({'persisted_marker':MARKER in out.getvalue(),'output':out.getvalue()}))
    assert MARKER not in out.getvalue(), 'The actual HTTP error handler prints credential-bearing response and request headers'

def test_key_query_credential_is_redacted_in_every_snapshot(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    url = 'https://example.invalid/f.pdf?key='+MARKER
    Corpus(tmp_path/'empty', tmp_path/'output').run([url],stages=('download',))
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        hits = [kind for kind,value in connection.execute('SELECT kind,json FROM snapshots') if MARKER in value]
        sources = [source for source, in connection.execute('SELECT source_uri FROM source_records') if MARKER in source]
    print('KEY_CREDENTIAL',json.dumps({'snapshots':hits,'source_records':sources}))
    assert hits == [] and sources == []

def test_redaction_does_not_merge_distinct_authenticated_resources(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    urls = ['https://alice:'+MARKER+'@example.invalid/document.pdf',
            'https://bob:'+MARKER+'@example.invalid/document.pdf']
    Corpus(tmp_path/'empty', tmp_path/'output').run(urls,stages=('download',))
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        docs = list(connection.execute('SELECT source_key FROM documents'))
        records = list(connection.execute('SELECT source_uri,metadata_json FROM source_records'))
        tasks = list(connection.execute("SELECT operation FROM tasks WHERE version_id IS NOT NULL"))
    print('AUTH_IDENTITY',json.dumps({'requested':2,'documents':docs,'source_records':records,'tasks':tasks}))
    assert len(docs) == 2, 'Different authenticated resources are silently merged before acquisition; reject ambiguous credentials or require source IDs'

def test_pending_resume_refuses_before_any_dispatch_and_rotation_matches(tmp_path, monkeypatch):
    real = StreamingScheduler.execute
    stop_execution(monkeypatch)
    url = 'https://reviewer:'+MARKER+'@example.invalid/f.pdf?access_token='+MARKER+'&id=7'
    corpus = Corpus(tmp_path/'empty', tmp_path/'output')
    result = corpus.run([url],stages=('download',))
    monkeypatch.setattr(StreamingScheduler,'execute',real)
    with pytest.raises(ValueError,match='credentials_not_persisted'):
        Corpus(tmp_path/'empty',tmp_path/'output').run(resume=result['run_id'])
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 0
    from glossapi.stages.run import _transport_urls
    rotated = url.replace(MARKER,'rotated-credential')
    from glossapi.ledger.urls import sanitize_url
    # PORT ADAPTATION (brief principle 3, exact binding): the probe asserted that
    # a rotated URL binds through its matching redacted spelling. That is the
    # first-redacted-key match the redesign forbids. Without a declared
    # source_id a rotated URL is a different resource identity and must NOT bind;
    # the identical URL binds. Rotation with a declared source_id is covered by
    # test_declared_source_id_supports_token_rotation_on_resume below.
    with LedgerWriter(tmp_path/'output/corpus.sqlite3') as writer:
        engine = StreamingScheduler(writer,tmp_path/'output')
        engine.transport_urls = _transport_urls([rotated],{})
        operations,_,_ = engine._load(result['run_id'])
        assert engine.redacted_credentials
        assert operations[0].specification.settings['url'] != rotated
        assert sanitize_url(rotated) == sanitize_url(url)
        engine = StreamingScheduler(writer,tmp_path/'output')
        engine.transport_urls = _transport_urls([url],{})
        operations,_,_ = engine._load(result['run_id'])
        assert not engine.redacted_credentials
        assert operations[0].specification.settings['url'] == url

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


# =================================================== part 2: fix set LG2 ===
import base64
import stat
import subprocess
import sys
import threading

TESTS = Path(__file__).resolve().parent


def marker_hits(output, marker=MARKER):
    """Every decoded ledger text column and every file byte under the corpus."""
    hits = []
    with open_readonly(Path(output)/'corpus.sqlite3') as connection:
        queries = {'snapshots': 'SELECT kind,json FROM snapshots', 'source_records': 'SELECT source_uri,metadata_json FROM source_records',
                   'documents': 'SELECT source_key FROM documents', 'files': 'SELECT uri FROM files',
                   'events': 'SELECT reason_code,message,traceback,details_json FROM events'}
        for table, query in queries.items():
            for row in connection.execute(query):
                if any(isinstance(value, str) and marker in value for value in row):
                    hits.append(table)
    for path in Path(output).rglob('*'):
        if path.is_file() and not path.is_symlink() and marker.encode() in path.read_bytes():
            hits.append(str(path.relative_to(output)))
    return sorted(set(hits))


class UserServer:
    """Loopback server: the same path returns each authenticated user's own PDF."""
    def __init__(self, payloads):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.requests = []
        server = self
        expected = {'Basic '+base64.b64encode((user+':'+MARKER).encode()).decode(): body for user, body in payloads.items()}
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests.append((self.path, self.headers.get('Authorization')))
                body = expected.get(self.headers.get('Authorization'))
                if body is None or not self.path.startswith('/document.pdf'):
                    self.send_error(401); return
                self.send_response(200); self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    def url(self, user, query=''):
        return f'http://{user}:{MARKER}@127.0.0.1:{self.server.server_port}/document.pdf{query}'
    def __enter__(self):
        self.thread.start(); return self
    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


DOWNLOAD = {'download': {'rate_limit': 0, 'max_retries': 1, 'retry_interval': 0}, 'extract': {'repair_fonts': False}}


def payload(tmp_path, name, text):
    return pdf(tmp_path/'payloads'/name, text).read_bytes()


# ------------------------------------------------- F1: persisted URL form ---

def test_persisted_url_is_an_allowlist_for_every_reference_form():
    from glossapi.ledger.urls import persisted_url, scrub_text, url_policy, PLACEHOLDER as P
    assert persisted_url('https://u:'+MARKER+'@Example.org:8443/a/b.pdf?id=7&x=&flag#frag') == \
        'https://Example.org:8443/a/b.pdf?id='+P+'&x=&'+P+'#'+P
    assert persisted_url('//u:'+MARKER+'@next.invalid/a.pdf?sig='+MARKER) == '//next.invalid/a.pdf?sig='+P
    assert persisted_url('/final.pdf?anything='+MARKER+'#k='+MARKER) == '/final.pdf?anything='+P+'#k='+P
    assert persisted_url('final.pdf?anything='+MARKER) == 'final.pdf?anything='+P
    for unchanged in ('https://example.org/a.pdf', 'file:///tmp/a@b.pdf', 'https://example.org/a?', 'plain text'):
        assert persisted_url(unchanged) is unchanged
    once = persisted_url('https://u:p@h.invalid/a?b=1#c')
    assert persisted_url(once) == once
    text = ('GET https://h.invalid/a?unknown_field='+MARKER+' via //u:'+MARKER+'@h.invalid/x and /rel?q='+MARKER
            +' Authorization: Bearer '+MARKER+"\n{'Cookie': 'a="+MARKER+"; b="+MARKER+"'} url=x.pdf?z="+MARKER)
    assert MARKER not in scrub_text(text)
    with url_policy({'public_query_fields': ['id', 'page']}):
        assert persisted_url('https://h.invalid/a?id=7&page=2&token='+MARKER) == 'https://h.invalid/a?id=7&page=2&token='+P
    with pytest.raises(ValueError):
        with url_policy({'public_query_fields': 'id'}):
            pass


def test_unrecognized_query_and_fragment_values_never_persist_from_a_public_run(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    urls = ['https://example.invalid/a.pdf?foo='+MARKER, 'https://example.invalid/b.pdf?'+MARKER,
            'https://example.invalid/c.pdf#'+MARKER, 'https://example.invalid/d.pdf?session='+MARKER+'&id=4']
    Corpus(tmp_path/'empty', tmp_path/'output').run(urls, stages=('download',))
    assert marker_hits(tmp_path/'output') == []
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] == 4
        from glossapi.ledger.urls import source_uri_of_key
        assert {source_uri_of_key(row[0]) for row in connection.execute("SELECT source_key FROM documents")} == {
            'https://example.invalid/a.pdf?foo=%5Bredacted%5D', 'https://example.invalid/b.pdf?%5Bredacted%5D',
            'https://example.invalid/c.pdf', 'https://example.invalid/d.pdf?session=%5Bredacted%5D&id=%5Bredacted%5D'}


def test_declared_public_fields_are_kept_and_frozen_and_still_redact_the_rest(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    url = 'https://example.invalid/a.pdf?id=7&token='+MARKER
    result = Corpus(tmp_path/'empty', tmp_path/'output').run([url], stages=('download',),
        settings={'url_policy': {'public_query_fields': ['id']}})
    assert marker_hits(tmp_path/'output') == []
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        sources = [row[0] for row in connection.execute('SELECT source_uri FROM source_records')]
        assert sources == ['https://example.invalid/a.pdf?id=7&token=%5Bredacted%5D']
        plan = connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.plan_sha256 WHERE r.run_id=?', (result['run_id'],)).fetchone()[0]
        assert 'public_query_fields' in plan and 'id=7' in plan


def test_identity_key_is_owner_only_outside_the_ledger_and_refused_when_exposed(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    from glossapi.ledger.urls import identity_key_path
    Corpus(tmp_path/'empty', tmp_path/'output').run(['https://u:'+MARKER+'@example.invalid/a.pdf'], stages=('download',))
    key = identity_key_path(tmp_path/'output')
    assert stat.S_IMODE(key.stat().st_mode) == 0o600 and stat.S_IMODE(key.parent.stat().st_mode) == 0o700
    secret = key.read_text().strip()
    assert len(secret) == 64 and marker_hits(tmp_path/'output', secret) == ['.glossapi/keys/resource-identity.key']
    key.chmod(0o644)
    with pytest.raises(Exception, match='owner only'):
        Corpus(tmp_path/'empty', tmp_path/'output').run(['https://u:'+MARKER+'@example.invalid/b.pdf'], stages=('download',))


def test_credential_free_urls_keep_plain_identities_and_create_no_key(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    from glossapi.ledger.urls import identity_key_path
    Corpus(tmp_path/'empty', tmp_path/'output').run(['https://example.invalid/a.pdf'], stages=('download',))
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        assert [row[0] for row in connection.execute('SELECT source_key FROM documents')] == ['url:v1:https://example.invalid/a.pdf']
    assert not identity_key_path(tmp_path/'output').exists()


# ------------------------------------------- F1: headers and log handlers ---

def test_http_error_log_redacts_default_and_declared_secret_headers_structurally(tmp_path):
    from glossapi.gloss_downloader import GlossDownloader, _private_logger
    from glossapi.ledger.urls import url_policy
    from aiohttp import ClientResponseError, RequestInfo
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL
    url = 'https://u:'+MARKER+'@example.invalid/a.pdf?x='+MARKER
    headers = CIMultiDictProxy(CIMultiDict([('Authorization', 'Bearer '+MARKER), ('Proxy-Authorization', MARKER),
        ('Cookie', 'a='+MARKER), ('X-Session', MARKER), ('Accept', 'application/pdf')]))
    info = RequestInfo(URL(url), 'GET', headers, URL(url))
    error = ClientResponseError(info, (), status=403, message='Forbidden',
        headers=CIMultiDict([('Set-Cookie', 'sessionid='+MARKER), ('Set-Cookie', 'other='+MARKER), ('Retry-After', '3')]))
    out = io.StringIO(); downloader = GlossDownloader.__new__(GlossDownloader)
    downloader.logger = _private_logger('lg2-http-error', logging.DEBUG, propagate=False)
    downloader.logger.addHandler(logging.StreamHandler(out)); downloader.verbose = True
    with url_policy({'secret_headers': ['X-Session']}):
        result = downloader._build_http_error_result(url, error, 0)
    assert MARKER not in out.getvalue() and 'application/pdf' in out.getvalue() and 'Retry-After' in out.getvalue()
    assert result[0] is False


def test_one_filter_on_glossapi_handlers_scrubs_messages_tracebacks_and_stacks(tmp_path):
    from glossapi.ledger.urls import protect_handler, CredentialLogFilter
    from glossapi.gloss_downloader import GlossDownloader
    downloader = GlossDownloader(output_dir=str(tmp_path/'d'), progress_log_file=str(tmp_path/'p.log'), request_log_file=str(tmp_path/'r.log'))
    for logger in (downloader.logger, downloader.progress_logger):
        assert any(isinstance(item, CredentialLogFilter) for handler in logger.handlers for item in handler.filters)
    out = io.StringIO()
    third_party = logging.getLogger('lg2.third.party')
    third_party.propagate = False
    handler = protect_handler(logging.StreamHandler(out))
    third_party.addHandler(handler)
    try:
        try:
            raise RuntimeError('fetch //u:'+MARKER+'@h.invalid/a?sig='+MARKER+' Authorization: Bearer '+MARKER)
        except RuntimeError:
            third_party.error('failed %s', 'https://h.invalid/x?k='+MARKER, exc_info=True, stack_info=True)
    finally:
        third_party.removeHandler(handler)
    assert 'RuntimeError' in out.getvalue() and 'Stack (most recent call last)' in out.getvalue()
    assert MARKER not in out.getvalue()


def test_worker_conversion_log_handler_is_protected():
    import inspect
    from glossapi.stages import workers
    source = inspect.getsource(workers)
    assert "protect_handler(logging.FileHandler(request.work_directory/'conversion.log'))" in source


# ---------------------------------------------- F2: identity and binding ---

def test_two_authenticated_users_get_their_own_documents_end_to_end(tmp_path):
    pytest.importorskip('aiohttp')
    pytest.importorskip('glossapi_rs_cleaner')
    bodies = {'alice': payload(tmp_path, 'alice.pdf', 'ALICE private document'),
              'bob': payload(tmp_path, 'bob.pdf', 'BOB private document')}
    with UserServer(bodies) as server:
        corpus = Corpus(tmp_path/'empty', tmp_path/'output')
        result = corpus.run([server.url('alice'), server.url('bob')], stages=('extract','clean','export'), settings=DOWNLOAD)
        assert result['state'] == 'completed', result
        assert result['document_counts'] == {'selected':2,'included':2,'omitted':0,'failed':0}
        texts = sorted(row['document'] for row in export_rows(corpus.output_dir))
        assert len(texts) == 2 and 'ALICE private document' in texts[0] and 'BOB private document' in texts[1]
        assert marker_hits(corpus.output_dir) == []
        with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
            assert connection.execute("SELECT COUNT(*) FROM tasks WHERE operation='download.acquire'").fetchone()[0] == 2
            alice = report_why(connection, server.url('alice'), run=select_run(connection))
            bob = report_why(connection, server.url('bob'), run=select_run(connection))
            assert alice['document']['document_id'] != bob['document']['document_id']
            from glossapi.ledger.reader import ReadRequestError
            with pytest.raises(ReadRequestError, match='multiple'):
                report_why(connection, server.url('alice').replace('alice:'+MARKER+'@', ''), run=select_run(connection))
        import pandas as pd
        frame = pd.read_parquet(corpus.output_dir/'download_results'/next(p.name for p in (corpus.output_dir/'download_results').glob('*.parquet') if 'partial' not in p.name))
        assert frame['download_success'].tolist() == [True, True] and not frame['is_duplicate'].any()


def test_resume_binds_each_resupplied_url_to_its_own_resource_only(tmp_path, monkeypatch):
    real = StreamingScheduler.execute
    stop_execution(monkeypatch)
    alice = 'https://alice:'+MARKER+'@example.invalid/document.pdf'
    bob = 'https://bob:'+MARKER+'@example.invalid/document.pdf'
    frozen = Corpus(tmp_path/'empty', tmp_path/'output').run([alice, bob], stages=('download',))
    monkeypatch.setattr(StreamingScheduler, 'execute', real)
    with pytest.raises(ValueError, match='credentials_not_persisted'):
        Corpus(tmp_path/'empty', tmp_path/'output').run(resume=frozen['run_id'], credentials=[bob])
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 0
    from glossapi.stages.run import _transport_urls
    with LedgerWriter(tmp_path/'output/corpus.sqlite3') as writer:
        engine = StreamingScheduler(writer, tmp_path/'output')
        engine.transport_urls = _transport_urls([bob], {})
        operations, _, _ = engine._load(frozen['run_id'])
        downloads = [operation for operation in operations if operation.stage == 'download']
        assert [operation.specification.settings['url'] == bob for operation in downloads] == [False, True]
        assert len(engine.redacted_credentials) == 1
        engine = StreamingScheduler(writer, tmp_path/'output')
        engine.transport_urls = _transport_urls([bob, alice], {})
        operations, _, _ = engine._load(frozen['run_id'])
        assert [operation.specification.settings['url'] for operation in operations if operation.stage == 'download'] == [alice, bob]
        assert not engine.redacted_credentials


def test_declared_source_id_supports_token_rotation_on_resume(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    first = 'https://example.invalid/document.pdf?access_token='+MARKER+'-one'
    rotated = 'https://example.invalid/document.pdf?access_token='+MARKER+'-two'
    manifest = {'documents': [{'url': first, 'source_id': 'catalog-17', 'title': 'Rotating token'}]}
    frozen = Corpus(tmp_path/'empty', tmp_path/'output').run([first], stages=('download',), settings={'source_manifest': manifest})
    assert marker_hits(tmp_path/'output') == [] and marker_hits(tmp_path/'output', 'catalog-17') != []
    from glossapi.stages.run import _transport_urls
    with LedgerWriter(tmp_path/'output/corpus.sqlite3') as writer:
        engine = StreamingScheduler(writer, tmp_path/'output')
        engine.transport_urls = _transport_urls([rotated], {})
        engine._load(frozen['run_id'])
        assert engine.redacted_credentials, 'a rotated URL without its source_id is another resource'
        engine = StreamingScheduler(writer, tmp_path/'output')
        engine.transport_urls = _transport_urls([{'url': rotated, 'source_id': 'catalog-17'}], {})
        operations, _, _ = engine._load(frozen['run_id'])
        assert not engine.redacted_credentials and operations[0].specification.settings['url'] == rotated


def test_one_source_id_for_two_different_urls_is_rejected_before_deduplication(tmp_path):
    manifest = {'documents': [{'url': 'https://example.invalid/a.pdf?t='+MARKER, 'source_id': 'same'},
                              {'url': 'https://example.invalid/b.pdf?t='+MARKER, 'source_id': 'same'}]}
    with pytest.raises(ValueError, match='ambiguous_source_identity') as raised:
        Corpus(tmp_path/'empty', tmp_path/'output').run([row['url'] for row in manifest['documents']], stages=('download',),
            settings={'source_manifest': manifest})
    assert MARKER not in str(raised.value)
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] == 0


def cli_subprocess(*arguments):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(TESTS.parent/'src'), str(TESTS), os.environ.get('PYTHONPATH', '')])}
    return subprocess.run([sys.executable, '-c', 'import sys; from glossapi.cli import main; sys.exit(main(sys.argv[1:]))', *arguments],
        env=env, capture_output=True, text=True, timeout=300)


def test_cli_plan_then_run_binds_credentials_by_identity_in_separate_processes(tmp_path):
    pytest.importorskip('aiohttp')
    pytest.importorskip('glossapi_rs_cleaner')
    bodies = {'alice': payload(tmp_path, 'alice.pdf', 'ALICE cli document'), 'bob': payload(tmp_path, 'bob.pdf', 'BOB cli document')}
    settings = tmp_path/'settings.json'; settings.write_text(json.dumps(DOWNLOAD))
    with UserServer(bodies) as server:
        output = tmp_path/'output'
        plan = cli_subprocess('plan', '--corpus', str(output), '--files', server.url('alice'), server.url('bob'),
            '--stages', 'extract,clean,export', '--settings', str(settings), '--json')
        assert plan.returncode == 0, plan.stderr[-2000:]
        assert MARKER not in plan.stdout
        (tmp_path/'plan.json').write_text(plan.stdout)
        planned = json.loads(plan.stdout)['data']['documents']
        assert len(planned) == 2 and len({document['source_key'] for document in planned}) == 2
        wrong = tmp_path/'wrong.json'; wrong.write_text(json.dumps([server.url('alice').replace(MARKER, 'rotated'), server.url('bob')]))
        refused = cli_subprocess('run', '--corpus', str(output), '--plan', str(tmp_path/'plan.json'), '--credentials', str(wrong), '--json')
        assert refused.returncode == 2 and 'credentials_not_persisted' in refused.stdout and not server.requests
        creds = tmp_path/'creds.json'; creds.write_text(json.dumps({'urls': [server.url('bob'), server.url('alice')]}))
        process = cli_subprocess('run', '--corpus', str(output), '--plan', str(tmp_path/'plan.json'), '--credentials', str(creds), '--json')
        assert process.returncode == 0, process.stdout[-3000:]+process.stderr[-3000:]
        assert MARKER not in process.stdout and MARKER not in process.stderr and MARKER not in plan.stderr+refused.stdout+refused.stderr
        texts = sorted(row['document'] for row in export_rows(output))
        assert len(texts) == 2 and 'ALICE cli document' in texts[0] and 'BOB cli document' in texts[1]
        assert marker_hits(output) == []


# ------------------------------------------------- F3: mixed manifests ---

def test_mixed_manifest_keeps_available_missing_and_duplicate_local_rows_beside_urls(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    good = pdf(tmp_path/'good.pdf', 'Mixed good local')
    missing = tmp_path/'missing.pdf'
    remote = 'https://example.invalid/remote.pdf?t='+MARKER
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
    assert marker_hits(output) == []
    with open_readonly(output/'corpus.sqlite3') as connection:
        uris = [row[0] for row in connection.execute('SELECT source_uri FROM source_records')]
        assert not any(uri.endswith('/nan') for uri in uris) and len(uris) == 6
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


# --------------------------------------- F4: per-occurrence bindings ---

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
