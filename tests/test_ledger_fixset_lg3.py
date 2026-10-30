"""Fix set LG3 (CP-LEDGER cycle 2 follow-up): the three HIGH gaps left by LG2.

1. Legacy ``Corpus.download(parquet)`` records only persisted URLs plus
   ``resource_identity`` (results, partial checkpoint, registered copies);
   resume matches by identity, reads older raw-URL results and rewrites them
   redacted; persisted input rows need resupplied credentials.
2. ``url_policy.path`` = ``'keep'`` (default, documented) or ``'redact'``.
3. A process-local known-secret scrubber removes exact secret components of
   in-memory raw URLs (bare tokens included) at logs and error boundaries.
"""
import base64
import io
import json
import logging
from pathlib import Path
import threading

import pandas as pd
import pytest

from glossapi import Corpus
from glossapi.ledger.writer import LedgerWriter, open_readonly
from glossapi.ledger.reader import report_why, select_run
from glossapi.stages.scheduler import StreamingScheduler
from test_ledger_fixset_lg import pdf, export_rows

MARKER = 'LG3_SYNTHETIC_CREDENTIAL_MARKER'
TOKEN = 'LG3SYNTHETICBARETOKEN'
DOWNLOAD = {'rate_limit': 0, 'max_retries': 1, 'retry_interval': 0}


@pytest.fixture(autouse=True)
def log_in_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def hits(output, marker=MARKER):
    """Every decoded ledger text column and every file byte under the corpus."""
    found = []
    database = Path(output)/'corpus.sqlite3'
    if database.exists():
        with open_readonly(database) as connection:
            queries = {'snapshots': 'SELECT kind,json FROM snapshots', 'source_records': 'SELECT source_uri,metadata_json FROM source_records',
                       'documents': 'SELECT source_key FROM documents', 'files': 'SELECT uri FROM files',
                       'events': 'SELECT reason_code,message,traceback,details_json FROM events'}
            for table, query in queries.items():
                for row in connection.execute(query):
                    if any(isinstance(value, str) and marker in value for value in row):
                        found.append(table)
    for path in Path(output).rglob('*'):
        if path.is_file() and not path.is_symlink() and marker.encode() in path.read_bytes():
            found.append(str(path.relative_to(output)))
    return sorted(set(found))


def stop_execution(monkeypatch):
    monkeypatch.setattr(StreamingScheduler, 'execute', lambda self, run, **kw:
        {'run_id': run, 'state': 'waiting', 'attempts': {},
         'counts': {'planned': 1, 'running': 0, 'ok': 0, 'skipped': 0, 'failed': 0}})


class Server:
    """Loopback server: basic auth from URL userinfo plus a query token; optional failures."""
    def __init__(self, body, *, fail=(), path_token=None):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.requests = []
        self.fail = set(fail)
        server = self
        expected = 'Basic '+base64.b64encode(('user:'+MARKER).encode()).decode()
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests.append(self.path)
                name = self.path.split('?', 1)[0].rsplit('/', 1)[-1]
                if name.startswith('bare'):
                    # Reason phrase carries a relative reference with a bare token.
                    self.send_response(500, 'failed '+name+'?'+TOKEN+' and '+TOKEN); self.end_headers(); return
                if path_token and path_token not in self.path:
                    self.send_error(404); return
                if not path_token and (self.headers.get('Authorization') != expected or 'token='+MARKER not in self.path):
                    self.send_error(401); return
                if name in server.fail:
                    self.send_error(500); return
                self.send_response(200); self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    def url(self, name):
        return f'http://user:{MARKER}@127.0.0.1:{self.server.server_port}/{name}.pdf?id=7&token={MARKER}'
    def files(self):
        """Document requests only (the downloader may also visit the site root)."""
        return [path for path in self.requests if '.pdf' in path]
    def base(self):
        return f'http://127.0.0.1:{self.server.server_port}'
    def __enter__(self):
        self.thread.start(); return self
    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


def body(tmp_path):
    return pdf(tmp_path/'payload.pdf', 'LG3 payload').read_bytes()


def results(output):
    return pd.read_parquet(Path(output)/'download_results/download_results_in.parquet')


# ------------------------------------------- 1. legacy Corpus.download() ---

def test_legacy_download_records_only_persisted_urls_and_resource_identity(tmp_path):
    pytest.importorskip('aiohttp')
    with Server(body(tmp_path), fail={'b.pdf'}) as server:
        source = tmp_path/'in.parquet'
        pd.DataFrame([{'url': server.url('a'), 'title': 'A'}, {'url': server.url('b'), 'title': 'B'}]).to_parquet(source, index=False)
        corpus = Corpus(tmp_path/'empty', tmp_path/'output')
        returned = corpus.download(source, checkpoint_every=1, **DOWNLOAD)
        assert returned['download_success'].tolist() == [True, False]
        assert any(path.startswith('/a.pdf') for path in server.requests)
    assert hits(tmp_path/'output') == []
    frame = results(tmp_path/'output')
    assert frame['url'].str.startswith('http://127.0.0.1').all() and frame['url'].str.contains('token=%5Bredacted%5D').all()
    assert frame['resource_identity'].str.startswith('url:v2:').all() and frame['resource_identity'].nunique() == 2
    partial = tmp_path/'output/download_results/download_results_in.parquet.partial.parquet'
    assert partial.is_file() and 'resource_identity' in pd.read_parquet(partial).columns
    copies = list((tmp_path/'output/.glossapi/artifacts').rglob('metadata.parquet'))
    assert copies and all('resource_identity' in pd.read_parquet(copy).columns for copy in copies)
    assert not (tmp_path/'output/temp_download_input.parquet').exists()


def test_legacy_resume_matches_by_identity_and_rewrites_older_raw_results(tmp_path):
    pytest.importorskip('aiohttp')
    with Server(body(tmp_path)) as server:
        output = tmp_path/'output'
        source = tmp_path/'in.parquet'
        pd.DataFrame([{'url': server.url('a')}, {'url': server.url('b')}]).to_parquet(source, index=False)
        # An older glossAPI wrote raw URLs: A is recorded as downloaded.
        (output/'download_results').mkdir(parents=True); (output/'downloads').mkdir()
        (output/'downloads/AAA_000.pdf').write_bytes(body(tmp_path))
        pd.DataFrame([{'url': server.url('a'), 'filename': 'AAA_000.pdf', 'download_success': True, 'download_error': ''}]).to_parquet(
            output/'download_results/download_results_in.parquet', index=False)
        Corpus(tmp_path/'empty', output).download(source, **DOWNLOAD)
        assert [path.split('?')[0] for path in server.files()] == ['/b.pdf'], 'A was matched by identity and skipped'
        frame = results(output)
        assert len(frame) == 2 and frame['download_success'].all() and frame['resource_identity'].nunique() == 2
        assert hits(output) == [], 'the older raw-URL results were rewritten redacted'
        # A second resume: everything matches by identity from persisted rows.
        before = len(server.files())
        again = Corpus(tmp_path/'empty', output).download(source, **DOWNLOAD)
        assert len(server.files()) == before and len(again) == 2 and hits(output) == []


def test_legacy_resume_after_failure_requests_only_the_failed_resource(tmp_path):
    pytest.importorskip('aiohttp')
    with Server(body(tmp_path), fail={'b.pdf'}) as server:
        source = tmp_path/'in.parquet'
        pd.DataFrame([{'url': server.url('a')}, {'url': server.url('b')}]).to_parquet(source, index=False)
        Corpus(tmp_path/'empty', tmp_path/'output').download(source, checkpoint_every=1, **DOWNLOAD)
        server.fail.clear()
        first = len(server.files())
        Corpus(tmp_path/'empty', tmp_path/'output').download(source, **DOWNLOAD)
        assert [path.split('?')[0] for path in server.files()[first:]] == ['/b.pdf']
        frame = results(tmp_path/'output')
        assert len(frame) == 2 and frame['download_success'].all()
    assert hits(tmp_path/'output') == []


def test_legacy_persisted_input_needs_resupplied_credentials_bound_by_identity(tmp_path):
    pytest.importorskip('aiohttp')
    with Server(body(tmp_path)) as server:
        first = tmp_path/'first'
        source = tmp_path/'in.parquet'
        pd.DataFrame([{'url': server.url('a')}]).to_parquet(source, index=False)
        Corpus(tmp_path/'empty', first).download(source, **DOWNLOAD)
        recorded = results(first)[['url', 'resource_identity']]
        persisted_input = tmp_path/'replay.parquet'
        recorded.to_parquet(persisted_input, index=False)
        # A second corpus sharing the identity key (copied with the corpus):
        # nothing to reuse, so a real reacquisition must use bound credentials.
        import shutil
        second = tmp_path/'second'
        shutil.copytree(first/'.glossapi/keys', second/'.glossapi/keys')
        before = len(server.requests)
        with pytest.raises(ValueError, match='credentials_not_persisted'):
            Corpus(tmp_path/'empty', second).download(persisted_input, **DOWNLOAD)
        assert len(server.requests) == before
        with pytest.raises(ValueError, match='credentials_not_persisted'):
            Corpus(tmp_path/'empty', second).download(persisted_input, credentials=[server.url('a').replace('id=7', 'id=8')], **DOWNLOAD)
        assert len(server.requests) == before
        returned = Corpus(tmp_path/'empty', second).download(persisted_input, credentials=[server.url('a')], **DOWNLOAD)
        assert returned['download_success'].tolist() == [True] and len(server.files()) == 2
    assert hits(first) == [] and hits(second) == []


# ---------------------------------------------------- 2. url_policy.path ---

def test_path_policy_keep_is_the_documented_default_and_redact_hides_the_path():
    from glossapi.ledger import urls
    from glossapi.ledger.urls import persisted_url, url_policy, PLACEHOLDER as P
    url = 'https://user:'+MARKER+'@h.invalid:8443/token/'+MARKER+'/file.pdf?x=1#f'
    assert persisted_url(url) == 'https://h.invalid:8443/token/'+MARKER+'/file.pdf?x='+P+'#'+P
    assert "{'path': 'redact'}" in urls.__doc__ and 'provenance needs' in urls.__doc__
    with url_policy({'path': 'redact'}):
        assert persisted_url(url) == 'https://h.invalid:8443/'+P+'?x='+P+'#'+P
        assert persisted_url('https://h.invalid/') == 'https://h.invalid/'
        assert persisted_url('//h.invalid/token/'+MARKER) == '//h.invalid/'+P
        assert persisted_url('/token/'+MARKER+'/f.pdf?y=2') == '/'+P+'?y='+P
        assert persisted_url(persisted_url(url)) == persisted_url(url)
        local = 'file:///tmp/token/'+MARKER+'/local.pdf'
        assert persisted_url(local) is local, 'local file paths are never redacted'
    with pytest.raises(ValueError):
        with url_policy({'path': 'hide'}):
            pass


def test_path_policy_in_a_public_run_keeps_or_redacts_with_keyed_identity(tmp_path, monkeypatch):
    stop_execution(monkeypatch)
    url = 'https://example.invalid/token/'+MARKER+'/file.pdf'
    Corpus(tmp_path/'empty', tmp_path/'keep').run([url], stages=('download',))
    assert 'source_records' in hits(tmp_path/'keep'), 'default contract: the path is recorded'
    Corpus(tmp_path/'empty', tmp_path/'redact').run([url], stages=('download',), settings={'url_policy': {'path': 'redact'}})
    assert hits(tmp_path/'redact') == []
    with open_readonly(tmp_path/'redact/corpus.sqlite3') as connection:
        assert [row[0] for row in connection.execute('SELECT source_uri FROM source_records')] == ['https://example.invalid/%5Bredacted%5D']
        assert connection.execute('SELECT source_key FROM documents').fetchone()[0].startswith('url:v2:')


def test_path_token_redacted_end_to_end_with_resume_binding(tmp_path, monkeypatch):
    pytest.importorskip('aiohttp')
    pytest.importorskip('glossapi_rs_cleaner')
    real = StreamingScheduler.execute
    with Server(body(tmp_path), path_token=MARKER) as server:
        url = server.base()+'/token/'+MARKER+'/file.pdf'
        settings = {'download': DOWNLOAD, 'extract': {'repair_fonts': False}, 'url_policy': {'path': 'redact'}}
        stop_execution(monkeypatch)
        frozen = Corpus(tmp_path/'empty', tmp_path/'output').run([url], stages=('extract', 'clean', 'export'), settings=settings)
        monkeypatch.setattr(StreamingScheduler, 'execute', real)
        with pytest.raises(ValueError, match='credentials_not_persisted'):
            Corpus(tmp_path/'empty', tmp_path/'output').run(resume=frozen['run_id'])
        assert not server.requests
        resumed = Corpus(tmp_path/'empty', tmp_path/'output').run(resume=frozen['run_id'], credentials=[url])
        assert resumed['state'] == 'completed', resumed
        assert len(export_rows(tmp_path/'output')) == 1
    assert hits(tmp_path/'output') == []


# -------------------------------------------------- 3. known-secret scrub ---

def test_bare_token_in_a_relative_reference_is_scrubbed_from_logs():
    from glossapi.ledger.urls import CredentialLogFilter, register_url, scrub_text
    register_url('https://example.invalid/x.pdf?'+TOKEN)
    text = 'failed x.pdf?'+TOKEN+' then '+TOKEN
    assert TOKEN in scrub_text(text), 'the pattern rules alone do not catch a bare token'
    out = io.StringIO()
    logger = logging.getLogger('lg3.bare'); logger.propagate = False
    handler = logging.StreamHandler(out); handler.addFilter(CredentialLogFilter()); logger.addHandler(handler)
    try:
        try:
            raise ValueError(text)
        except ValueError:
            logger.error('download failed: %s', text, exc_info=True)
    finally:
        logger.removeHandler(handler)
    assert TOKEN not in out.getvalue() and 'ValueError' in out.getvalue()


def test_legacy_download_exception_with_bare_token_reaches_no_ledger_log_or_file(tmp_path, monkeypatch):
    from glossapi.gloss_downloader import GlossDownloader
    async def broken(self, row_index, url, semaphore, rate_limiter, retry_count=0, filename_base=None, referer=None):
        raise ValueError('failed x.pdf?'+TOKEN+' and bare '+TOKEN)
    monkeypatch.setattr(GlossDownloader, 'download_file', broken)
    source = tmp_path/'in.parquet'
    pd.DataFrame([{'url': 'https://example.invalid/x.pdf?'+TOKEN}]).to_parquet(source, index=False)
    returned = Corpus(tmp_path/'empty', tmp_path/'output').download(source, **DOWNLOAD)
    assert returned['download_success'].tolist() == [False]
    assert hits(tmp_path/'output', TOKEN) == []
    errors = results(tmp_path/'output')['download_error'].tolist()
    assert errors and 'failed x.pdf?' in errors[0]
    log = tmp_path/'conversion.log'
    assert not log.exists() or TOKEN not in log.read_text()


def test_run_worker_error_with_bare_token_reaches_no_ledger_or_file(tmp_path):
    pytest.importorskip('aiohttp')
    with Server(body(tmp_path)) as server:
        url = server.base()+'/bare.pdf?'+TOKEN
        result = Corpus(tmp_path/'empty', tmp_path/'output').run([url], stages=('extract', 'clean', 'export'),
            settings={'download': DOWNLOAD, 'extract': {'repair_fonts': False}})
        assert result['state'] == 'completed_with_failures'
        assert any(path.startswith('/bare.pdf') for path in server.requests)
    found = hits(tmp_path/'output', TOKEN)
    assert found == [], found
    with open_readonly(tmp_path/'output/corpus.sqlite3') as connection:
        why = json.dumps(report_why(connection, url, run=select_run(connection)))
        assert 'failed bare.pdf?%5Bredacted%5D and %5Bredacted%5D' in why, 'the error itself stays explained'
