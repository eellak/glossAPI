"""Fix set LG3 (CP-LEDGER cycle 2 follow-up), legacy ``Corpus.download(parquet)`` part, on public URLs.

Results, the partial checkpoint and the registered copies carry ``resource_identity``; a resume matches rows by
identity, rewrites older results without the column, and requests only what failed.

LG3's credential parts (persisted URL form in legacy results, resupplied credentials for persisted input rows,
``url_policy.path`` and the known-secret scrubber) were withdrawn with the credential channel (owner, 1 Nov 2026):
credentialed URLs are refused at entry (tests/test_k4_credential_refusal.py) and public URLs are recorded as given.
"""
from pathlib import Path
import threading

import pandas as pd
import pytest

from glossapi import Corpus
from test_ledger_fixset_lg import pdf

DOWNLOAD = {'rate_limit': 0, 'max_retries': 1, 'retry_interval': 0}


@pytest.fixture(autouse=True)
def log_in_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


class Server:
    """Loopback public server with a query parameter in every URL; optional failures."""
    def __init__(self, body, *, fail=()):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.requests = []
        self.fail = set(fail)
        server = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests.append(self.path)
                name = self.path.split('?', 1)[0].rsplit('/', 1)[-1]
                if '.pdf' in self.path and 'id=7' not in self.path:
                    self.send_error(404); return
                if name in server.fail:
                    self.send_error(500); return
                self.send_response(200); self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    def url(self, name):
        return f'http://127.0.0.1:{self.server.server_port}/{name}.pdf?id=7&sequence=1'
    def files(self):
        """Document requests only (the downloader may also visit the site root)."""
        return [path for path in self.requests if '.pdf' in path]
    def __enter__(self):
        self.thread.start(); return self
    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


def body(tmp_path):
    return pdf(tmp_path/'payload.pdf', 'LG3 payload').read_bytes()


def results(output):
    return pd.read_parquet(Path(output)/'download_results/download_results_in.parquet')


def test_legacy_download_records_urls_as_given_and_resource_identity(tmp_path):
    pytest.importorskip('aiohttp')
    with Server(body(tmp_path), fail={'b.pdf'}) as server:
        source = tmp_path/'in.parquet'
        pd.DataFrame([{'url': server.url('a'), 'title': 'A'}, {'url': server.url('b'), 'title': 'B'}]).to_parquet(source, index=False)
        corpus = Corpus(tmp_path/'empty', tmp_path/'output')
        returned = corpus.download(source, checkpoint_every=1, **DOWNLOAD)
        assert returned['download_success'].tolist() == [True, False]
        assert any(path.startswith('/a.pdf') for path in server.requests)
        expected = [server.url('a'), server.url('b')]
    frame = results(tmp_path/'output')
    assert frame['url'].tolist() == expected, 'public URLs are recorded as given'
    assert frame['resource_identity'].tolist() == ['url:v1:'+url for url in expected]
    partial = tmp_path/'output/download_results/download_results_in.parquet.partial.parquet'
    assert partial.is_file() and 'resource_identity' in pd.read_parquet(partial).columns
    copies = list((tmp_path/'output/.glossapi/artifacts').rglob('metadata.parquet'))
    assert copies and all('resource_identity' in pd.read_parquet(copy).columns for copy in copies)
    assert not (tmp_path/'output/temp_download_input.parquet').exists()


def test_legacy_resume_matches_by_identity_and_rewrites_older_results(tmp_path):
    pytest.importorskip('aiohttp')
    with Server(body(tmp_path)) as server:
        output = tmp_path/'output'
        source = tmp_path/'in.parquet'
        pd.DataFrame([{'url': server.url('a')}, {'url': server.url('b')}]).to_parquet(source, index=False)
        # An older glossAPI wrote results without resource_identity: A is recorded as downloaded.
        (output/'download_results').mkdir(parents=True); (output/'downloads').mkdir()
        (output/'downloads/AAA_000.pdf').write_bytes(body(tmp_path))
        pd.DataFrame([{'url': server.url('a'), 'filename': 'AAA_000.pdf', 'download_success': True, 'download_error': ''}]).to_parquet(
            output/'download_results/download_results_in.parquet', index=False)
        Corpus(tmp_path/'empty', output).download(source, **DOWNLOAD)
        assert [path.split('?')[0] for path in server.files()] == ['/b.pdf'], 'A was matched by identity and skipped'
        frame = results(output)
        assert len(frame) == 2 and frame['download_success'].all() and frame['resource_identity'].nunique() == 2
        # A second resume: everything matches by identity from the recorded rows.
        before = len(server.files())
        again = Corpus(tmp_path/'empty', output).download(source, **DOWNLOAD)
        assert len(server.files()) == before and len(again) == 2


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
