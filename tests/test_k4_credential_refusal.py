"""K4-CRED-RM (owner decision 1 Nov 2026, PLAN §9): glossAPI downloads public targets only.

The user-credential download channel (LG2-LG4, CP-CRED) is removed. One rule stays: a source URL with userinfo or an
obvious credential query/fragment field, or a download setting that would send credentials, is refused at entry with
``credentials_unsupported`` (CLI, Python run, manifest, plan, legacy download, download policy), before anything is
planned, recorded or fetched. No credential parameter is accepted. Every other URL is recorded as given.
"""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import threading

import pandas as pd
import pytest

from glossapi import Corpus
from glossapi.ledger.urls import CredentialsUnsupportedError, credential_reason, refuse_credentials
from test_ledger_fixset_lg import cli, export_rows, pdf

MARKER = 'K4CREDRMSYNTHETICSECRET'
CREDENTIALED = ['https://user:' + MARKER + '@example.invalid/a.pdf',
                'https://example.invalid/a.pdf?id=7&access_token=' + MARKER,
                'https://bucket.example.invalid/a.pdf?X-Amz-Signature=' + MARKER,
                'https://example.invalid/a.pdf#token=' + MARKER,
                'https://example.invalid/a.pdf?api-key=' + MARKER,
                'https://example.invalid/a.pdf?sig=' + MARKER]
DOWNLOAD = {'rate_limit': 0, 'max_retries': 1, 'retry_interval': 0}


@pytest.fixture(autouse=True)
def log_in_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path / 'conversion.log'))


def marker_hits(root):
    """Every file byte under ``root`` and every *logical* ledger value (read through the ledger's decoding adapter, so
    compressed snapshots and evidence are searched decoded; CP-CRM F8) that holds the marker."""
    from glossapi.ledger.writer import open_readonly
    hits = []
    for path in Path(root).rglob('*'):
        if path.is_file() and not path.is_symlink() and MARKER.encode() in path.read_bytes():
            hits.append(str(path.relative_to(root)))
    database = Path(root) / 'corpus.sqlite3'
    if database.exists():
        with open_readonly(database) as connection:
            for kind, text in connection.execute('SELECT kind,json FROM snapshots'):
                if MARKER in json.dumps(text, default=str):
                    hits.append('ledger:snapshots:' + kind)
            names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'")]
            for name in names:
                for row in connection.execute(f'SELECT * FROM "{name}"'):
                    if MARKER in json.dumps([value if isinstance(value, (str, int, float, type(None))) else
                                             value.decode('latin-1') if isinstance(value, bytes) else str(value) for value in row]):
                        hits.append('ledger:' + name)
    return sorted(set(hits))


def ledger_rows(root, table):
    database = Path(root) / 'corpus.sqlite3'
    if not database.exists():
        return 0
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    try:
        return connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    finally:
        connection.close()


class Server:
    """Loopback public server: counts requests; ``/start.pdf`` redirects to a server-signed URL."""
    def __init__(self, body):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.requests = []
        server = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests.append(self.path)
                if self.path.startswith('/start.pdf'):
                    self.send_response(302); self.send_header('Location', '/final.pdf?X-Amz-Signature=server-issued'); self.end_headers()
                    return
                self.send_response(200); self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    def url(self, path):
        return f'http://127.0.0.1:{self.server.server_port}{path}'
    def __enter__(self):
        self.thread.start(); return self
    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


# ------------------------------------------------------------------- the rule ---

@pytest.mark.parametrize('url', CREDENTIALED)
def test_credentialed_urls_are_refused_with_a_typed_error_that_never_repeats_the_secret(url):
    assert credential_reason(url) is not None
    with pytest.raises(CredentialsUnsupportedError) as raised:
        refuse_credentials(urls=[url])
    assert raised.value.code == 'credentials_unsupported' and 'credentialed downloads are not supported' in str(raised.value)
    assert MARKER not in str(raised.value) and MARKER not in json.dumps(raised.value.details)


@pytest.mark.parametrize('url', ['https://example.org/a.pdf', 'https://repo.example.org/bitstream/1/2/a.pdf?sequence=1&isAllowed=y',
                                 'https://example.org/view?id=7&q=greek', 'https://example.org/a.pdf?token=', 'https://example.org/a.pdf#page=3',
                                 'https://example.org/user@host/a.pdf', '/local/path/with@sign.pdf'])
def test_public_urls_and_local_paths_pass(url):
    assert credential_reason(url) is None
    refuse_credentials(urls=[url])


# -------------------------------------------------------------- Python entry ---

@pytest.mark.parametrize('url', CREDENTIALED[:3])
def test_python_run_refuses_a_credentialed_url_before_anything_is_recorded(tmp_path, url):
    with pytest.raises(CredentialsUnsupportedError):
        Corpus(tmp_path / 'empty', tmp_path / 'out').run([url], stages=('extract', 'clean', 'export'), settings={'download': DOWNLOAD})
    assert marker_hits(tmp_path / 'out') == [] and ledger_rows(tmp_path / 'out', 'documents') == 0


def test_python_run_refuses_credentialed_manifest_rows_and_list_cells(tmp_path):
    local = pdf(tmp_path / 'in/local.pdf', 'LOCAL text body')
    for index, row in enumerate([{'url': CREDENTIALED[1]}, {'url': ['https://example.invalid/ok.pdf', CREDENTIALED[0]]}]):
        out = tmp_path / f'out{index}'
        manifest = {'documents': [{'path': str(local)}, row]}
        with pytest.raises(CredentialsUnsupportedError):
            Corpus(tmp_path / 'empty', out).run([local, 'https://example.invalid/ok.pdf'], stages=('extract', 'clean', 'export'),
                                                settings={'source_manifest': manifest, 'download': DOWNLOAD})
        assert marker_hits(out) == [] and ledger_rows(out, 'documents') == 0


def test_python_run_refuses_credential_download_settings(tmp_path):
    for index, (config, settings) in enumerate([({}, {'download': {'domain_cookies': {'example.invalid': {'s': MARKER}}}}),
                                                ({'domain_cookies': {'example.invalid': {'s': MARKER}}}, {})]):
        out = tmp_path / f'out{index}'
        with pytest.raises(CredentialsUnsupportedError, match='domain_cookies'):
            Corpus(tmp_path / 'empty', out, downloader_config=config).run(['https://example.invalid/a.pdf'], stages=('extract',),
                                                                          settings=settings)
        assert marker_hits(out) == []


def test_no_credential_parameter_is_accepted(tmp_path):
    from glossapi.gloss_downloader import GlossDownloader
    with pytest.raises(TypeError):
        Corpus(tmp_path / 'empty', tmp_path / 'out').run(['https://example.invalid/a.pdf'], credentials=['https://example.invalid/a.pdf'])
    with pytest.raises(TypeError):
        GlossDownloader(output_dir=str(tmp_path / 'dl'), domain_cookies={'example.invalid': {'s': MARKER}})
    import inspect
    from glossapi.stages.run import prepare_streaming
    assert 'credentials' not in inspect.signature(prepare_streaming).parameters
    assert 'credentials' not in inspect.signature(Corpus.download).parameters
    assert marker_hits(tmp_path) == []


def test_legacy_download_refuses_credentialed_rows_and_settings_before_any_request_or_record(tmp_path):
    pytest.importorskip('aiohttp')
    with Server(b'%PDF-1.4\n') as server:
        good = server.url('/a.pdf')
        cases = [({'url': [good, 'http://user:' + MARKER + '@127.0.0.1/b.pdf']}, {}),
                 ({'url': [good, server.url('/b.pdf?token=' + MARKER)]}, {}),
                 ({'url': [good]}, {'domain_cookies': {'127.0.0.1': {'s': MARKER}}}),
                 ({'url': [good]}, {'credentials': [good.replace('/a.pdf', '/a.pdf?k=' + MARKER)]})]
        for index, (columns, kwargs) in enumerate(cases):
            source = tmp_path / f'in{index}.parquet'
            pd.DataFrame(columns).to_parquet(source, index=False)
            out = tmp_path / f'out{index}'
            with pytest.raises(CredentialsUnsupportedError):
                Corpus(tmp_path / 'empty', out).download(source, **DOWNLOAD, **kwargs)
            assert marker_hits(out) == [], index
        assert server.requests == []


# ----------------------------------------------------------------- CLI entry ---

def test_cli_plan_and_run_refuse_credentialed_files_and_manifest_rows(tmp_path):
    local = pdf(tmp_path / 'in/local.pdf', 'LOCAL text body')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'manifest_version': 1, 'documents': [{'path': str(local)}, {'url': CREDENTIALED[1]}]}))
    out = tmp_path / 'out'
    for arguments in (['plan', '--files', CREDENTIALED[0]], ['plan', '--manifest', str(manifest)],
                      ['run', '--files', CREDENTIALED[2]], ['run', '--manifest', str(manifest)]):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            from glossapi.cli import main
            code = main([*arguments, '--corpus', str(out), '--stages', 'extract,clean,export', '--json'])
        envelope = json.loads(buffer.getvalue())
        assert code == 2 and envelope['errors'][0]['code'] == 'credentials_unsupported', envelope
        assert MARKER not in buffer.getvalue()
    assert marker_hits(out) == [] and ledger_rows(out, 'documents') == 0


def test_cli_plan_refuses_a_credential_download_setting(tmp_path):
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'download': {'domain_cookies': {'example.invalid': {'s': MARKER}}}}))
    code, envelope = cli('plan', '--corpus', str(tmp_path / 'out'), '--files', 'https://example.invalid/a.pdf', '--settings', str(settings), '--json')
    assert code == 2 and envelope['errors'][0]['code'] == 'credentials_unsupported' and MARKER not in json.dumps(envelope)


def test_cli_has_no_credentials_option(tmp_path):
    from glossapi.cli import parser
    assert '--credentials' not in parser().format_help()
    creds = tmp_path / 'creds.json'
    creds.write_text(json.dumps(['https://example.invalid/a.pdf']))
    code, envelope = cli('run', '--corpus', str(tmp_path / 'out'), '--resume', 'RUN', '--credentials', str(creds), '--json')
    assert code == 2 and envelope['errors'][0]['code'] == 'invalid_request' and '--credentials' in envelope['errors'][0]['message']


def test_the_guide_and_cli_schema_document_the_code():
    from glossapi.llm_guide import guide_view
    row = guide_view()['errors']['credentials_unsupported']
    assert row['exit_code'] == 2 and 'public' in row['meaning']
    assert '`credentials_unsupported`' in (Path(__file__).resolve().parents[1] / 'docs/cli_json_schema.md').read_text()


# ----------------------------------------------------------- recorded as given ---

def test_public_query_url_and_server_signed_redirect_are_recorded_as_given(tmp_path):
    pytest.importorskip('aiohttp')
    pytest.importorskip('glossapi_rs_cleaner')
    body = pdf(tmp_path / 'payload.pdf', 'PUBLIC query document').read_bytes()
    with Server(body) as server:
        url = server.url('/start.pdf?id=7&sequence=1')
        out = tmp_path / 'out'
        result = Corpus(tmp_path / 'empty', out).run([url], stages=('extract', 'clean', 'export'),
                                                     settings={'download': DOWNLOAD, 'extract': {'repair_fonts': False}})
        assert result['state'] == 'completed', result
        assert '/final.pdf?X-Amz-Signature=server-issued' in server.requests
    assert len(export_rows(out)) == 1
    from glossapi.ledger.reader import report_why, select_run
    from glossapi.ledger.writer import open_readonly
    with open_readonly(out / 'corpus.sqlite3') as connection:
        assert [row[0] for row in connection.execute('SELECT source_key FROM documents')] == ['url:v1:' + url]
        assert [row[0] for row in connection.execute('SELECT source_uri FROM source_records')] == [url]
        plan = connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.plan_sha256 WHERE r.run_id=?',
                                  (result['run_id'],)).fetchone()[0]
        assert 'id=7&sequence=1' in plan, 'the frozen download step holds the URL as given'
        why = report_why(connection, url, run=select_run(connection))
        requests = [request for stage in why['stages'] for attempt in stage['attempts'] for decision in attempt['decisions']
                    for request in decision['details'].get('http', {}).get('requests', [])]
    assert any(request['requested_url'] == url and request['redirects'] and
               request['redirects'][0]['location'] == '/final.pdf?X-Amz-Signature=server-issued' for request in requests), requests
    assert not (out / '.glossapi/keys').exists(), 'no identity key: URLs are not keyed by a secret any more'


def test_settings_and_evidence_are_recorded_as_given():
    from glossapi.ledger.journal import safe_settings
    settings = {'download': {'rate_limit': 5, 'referer': 'https://example.org/page?q=1'}, 'lease_token': 'lease-1',
                'path': Path('/x/y'), 'url': 'https://example.org/a.pdf?id=7'}
    assert safe_settings(settings) == {'download': {'rate_limit': 5, 'referer': 'https://example.org/page?q=1'}, 'lease_token': 'lease-1',
                                       'path': '/x/y', 'url': 'https://example.org/a.pdf?id=7'}


def test_an_installed_toolkit_url_never_records_its_userinfo(monkeypatch):
    import glossapi.ocr.defaults as defaults
    from glossapi.stages.scheduler import execution_environment
    real = defaults.toolkit_source_status
    def status(**kwargs):
        return {**real(**kwargs), 'installed_url': 'https://oauth2:' + MARKER + '@github.com/owner/greek-ocr-toolkit'}
    monkeypatch.setattr(defaults, 'toolkit_source_status', status)
    environment = execution_environment()
    assert environment['ocr']['toolkit']['installed_url'] == 'https://github.com/owner/greek-ocr-toolkit'
    assert MARKER not in json.dumps(environment)


def test_a_ledger_with_withdrawn_url_v2_identities_is_refused(tmp_path):
    from glossapi.ledger.writer import LedgerError, LedgerWriter
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        with writer.transaction():
            writer.document('url:v2:0123456789abcdef0123456789abcdef:https://example.org/a.pdf?id=%5Bredacted%5D')
    with pytest.raises(LedgerError, match='url:v2'):
        LedgerWriter(tmp_path / 'corpus.sqlite3')


def test_the_secret_scanner_reads_compressed_ledger_evidence(tmp_path):
    """Known-positive control (CP-CRM F8): the marker only inside a compressed snapshot is invisible to a raw byte scan
    and found by marker_hits through the ledger's decoding adapter."""
    from glossapi.ledger.writer import LedgerWriter
    out = tmp_path / 'out'
    out.mkdir()
    with LedgerWriter(out / 'corpus.sqlite3') as writer:
        with writer.transaction():
            writer.snapshot('probe', {'kind': 'probe', 'value': ('padding ' * 400) + MARKER + (' padding' * 400)})
    assert not any(MARKER.encode() in path.read_bytes() for path in out.rglob('*') if path.is_file()), 'control is not compressed'
    assert 'ledger:snapshots:probe' in marker_hits(out)
