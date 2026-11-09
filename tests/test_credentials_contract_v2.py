"""PLAN §2.1 credentials contract v2 (after CP-CRMc2): glossAPI never sends credentials.

Part 1, the boundary matrix: on every download path (direct ``download_file``, ``download_files`` sync and async, the
browser downloader incl. its preflight, legacy ``Corpus.download``, a streaming run, a resume that would dispatch), a
credentialed URL, a credential header from settings and a policy cookie (dict, object, file) are refused with zero
requests leaving: a mock transport counts every session request, preflight and worker dispatch; sockets are denied. A
public control per path proves the counter sees traffic.
Part 2: the CP-CRMc2 must-fix probes (R3 schemeless userinfo, R4 parser errors, R5 optional YAML, R7 policy object),
ported from review_scratch/CP-CRMc2 (test_review_round2.py, test_additional.py), and its public-path controls.
R1, R2 and R6 are declared by contract item 4 (records are not confidential); their request boundaries are covered by
part 1 (referer, legacy, resume).
"""
import asyncio
import contextlib
import io
import json
import os
import socket
import subprocess
import sys

import pandas as pd
import pytest

from glossapi import Corpus
from glossapi.ledger.urls import CredentialsUnsupportedError

MARK = 'V2SYNTHETICSECRET'
GOOD = 'https://example.invalid/a.pdf?id=7&sequence=1'
URLS = {'query': 'https://example.invalid/a.pdf?access_token=' + MARK,
        'userinfo': 'https://user:' + MARK + '@example.invalid/a.pdf',
        'schemeless': 'user:' + MARK + '@example.invalid/a.pdf',
        'referer': GOOD}  # with a credentialed referer (CP-CRMc2 R1: refused before any request)
PATHS = ['download_file', 'download_files', 'adownload_files', 'browser', 'legacy', 'streaming', 'resume']


class Transport:
    """Counts every request that could leave: session requests, preflights and worker dispatch."""
    def __init__(self, monkeypatch):
        import aiohttp
        from glossapi.gloss_downloader import GlossDownloader
        from glossapi.gloss_browser_downloader import BrowserGlossDownloader
        from glossapi.stages.scheduler import StreamingScheduler
        self.requests = []
        transport = self
        class Session:
            def __init__(self, *args, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def close(self): pass
            def __getattr__(self, method):
                def request(url, *args, **kwargs):
                    transport.requests.append((method, str(url)))
                    raise RuntimeError('mock transport: request counted, not sent')
                return request
        monkeypatch.setattr(aiohttp, 'ClientSession', Session)
        async def preflight(self, **kwargs):
            transport.requests.append(('preflight', kwargs.get('url')))
            return False, '', '', 'mock preflight counted', 0
        monkeypatch.setattr(GlossDownloader, '_preflight_download', preflight)
        monkeypatch.setattr(BrowserGlossDownloader, '_preflight_download', preflight)
        def dispatch(self, run, **kwargs):
            transport.requests.append(('dispatch', run))
            raise RuntimeError('mock dispatch counted')
        monkeypatch.setattr(StreamingScheduler, 'execute', dispatch)
        def deny(*args, **kwargs):
            raise AssertionError('sockets are denied')
        monkeypatch.setattr(socket.socket, 'connect', deny)
        monkeypatch.setattr(socket, 'create_connection', deny)


def credential(kind, tmp_path):
    """(source URL, download settings) for one credential kind."""
    from glossapi.download_policy import DownloadPolicy
    if kind in URLS:
        return URLS[kind], ({'referer': 'https://example.invalid/page?api_key=' + MARK} if kind == 'referer' else {})
    if kind == 'header':
        return GOOD, {'headers': {'Authorization': 'Bearer ' + MARK}}
    if kind == 'policy_dict':
        return GOOD, {'download_policy': {'default': {'Cookie': MARK}}}
    if kind == 'policy_object':
        return GOOD, {'download_policy': DownloadPolicy(default_options={'Cookie': MARK})}
    policy = tmp_path / 'policy.yaml'
    policy.write_text('rules:\n  - match: {domains: [example.invalid]}\n    downloader: standard\n    Cookie: ' + MARK + '\n')
    return GOOD, {'download_policy_file': str(policy)}


def run_path(path, tmp_path, url, settings):
    from glossapi.gloss_downloader import GlossDownloader, RateLimiter
    from glossapi.gloss_browser_downloader import BrowserGlossDownloader
    base = dict(sleep=0, rate_limit=0, max_retries=1, pre_ping_domains=False)
    referer = settings.pop('referer', None)
    frame = pd.DataFrame({'url': [url], **({'page': [referer]} if referer else {})})
    source = tmp_path / 'source.parquet'
    frame.to_parquet(source, index=False)
    out = tmp_path / 'out'
    if path in ('download_file', 'download_files', 'adownload_files', 'browser'):
        cls = BrowserGlossDownloader if path == 'browser' else GlossDownloader
        dl = cls(output_dir=str(out), **base, **settings, **({'referer_column': 'page'} if referer else {}))
        try:
            if path in ('download_file', 'browser'):
                return asyncio.run(dl.download_file(0, url, None, RateLimiter(1000000), referer=referer))
            if path == 'download_files':
                return dl.download_files(str(source), input_frame=frame)
            return asyncio.run(dl.adownload_files(str(source), input_frame=frame))
        finally:
            dl.close_logs()
    if path == 'legacy':
        return Corpus(tmp_path / 'empty', out).download(source, **base, **settings, **({'referer_column': 'page'} if referer else {}))
    if path == 'streaming':
        rows = {'documents': [{'url': url}]}
        download = {**settings, **({'referer': referer} if referer else {})}
        return Corpus(tmp_path / 'empty', out).run([url], stages=('extract',), settings={'download': download, 'source_manifest': rows})
    # resume: a run frozen by an older build (guard bypassed), resumed by this one
    import glossapi.stages.scheduler as scheduler
    from glossapi.ledger.writer import LedgerWriter
    from glossapi.stages.contracts import DocumentSnapshot, StageSpecification
    out.mkdir()
    guard = scheduler._refuse_credentialed
    scheduler._refuse_credentialed = lambda *args: None
    try:
        with LedgerWriter(out / 'corpus.sqlite3') as writer:
            operation = scheduler.ScheduledOperation('download', 'download', DocumentSnapshot('synthetic-doc', None, url),
                StageSpecification('download.acquire', settings={'url': url, 'filename_base': 'AAA_000', **settings,
                                                                 **({'referer': referer} if referer else {})}))
            run = scheduler.StreamingScheduler(writer, out).freeze([operation])
    finally:
        scheduler._refuse_credentialed = guard
    return Corpus(tmp_path / 'empty', out).run(resume=run)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('kind', ['query', 'userinfo', 'schemeless', 'referer', 'header', 'policy_dict', 'policy_object', 'policy_file'])
def test_no_request_leaves_with_a_credential(tmp_path, monkeypatch, path, kind):
    if path == 'resume' and kind == 'policy_object':
        pytest.skip('a policy object cannot be frozen (recorded as unknown_not_recorded), so a resume never holds one')
    transport = Transport(monkeypatch)
    url, settings = credential(kind, tmp_path)
    with pytest.raises((CredentialsUnsupportedError, TypeError)) as raised:
        run_path(path, tmp_path, url, settings)
    assert transport.requests == [], transport.requests
    assert MARK not in str(raised.value) + json.dumps(getattr(raised.value, 'details', {}), default=str)
    if isinstance(raised.value, TypeError):
        # Only a direct constructor given a removed setting (headers=...) fails as a TypeError; still no request.
        assert kind == 'header' and path in ('download_file', 'download_files', 'adownload_files', 'browser'), raised.value


@pytest.mark.parametrize('path', PATHS)
def test_the_counter_sees_a_public_request_on_every_path(tmp_path, monkeypatch, path):
    transport = Transport(monkeypatch)
    try:
        run_path(path, tmp_path, GOOD, {})
    except RuntimeError:
        pass  # the mock dispatch stops the streaming paths
    assert transport.requests, path


# ------------------------------------------------------------ part 2: CP-CRMc2 must-fix probes and controls ---

MARK2 = 'C2S_467829'
GOOD2 = 'https://example.invalid/a.pdf?id=7&sequence=1'


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path / 'conversion.log'))
    def deny(*a, **kw):
        raise AssertionError('external network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    from glossapi.stages.scheduler import StreamingScheduler
    def stop(*a, **kw):
        raise RuntimeError('stopped before worker dispatch')
    monkeypatch.setattr(StreamingScheduler, 'execute', stop)


def test_r3_scheduler_schemeless_source_refused_before_freeze(tmp_path, offline):
    from test_k4_credential_refusal import marker_hits
    import test_k4_credential_refusal as scanner
    from glossapi.stages.scheduler import StreamingScheduler, ScheduledOperation
    from glossapi.ledger.writer import LedgerWriter
    from glossapi.stages.contracts import DocumentSnapshot, StageSpecification
    source = 'user:' + MARK2 + '@example.invalid/a.pdf'
    out = tmp_path / 'out'
    out.mkdir()
    with LedgerWriter(out / 'corpus.sqlite3') as writer:
        operation = ScheduledOperation('download', 'download', DocumentSnapshot('synthetic-doc', None, source),
            StageSpecification('download.acquire', settings={'url': source, 'filename_base': 'AAA_000'}))
        with pytest.raises(CredentialsUnsupportedError):
            StreamingScheduler(writer, out).freeze([operation])
    scanner.MARKER, saved = MARK2, scanner.MARKER
    try:
        assert marker_hits(out) == []
    finally:
        scanner.MARKER = saved


def test_r3_schemeless_selector_error_does_not_echo_secret(tmp_path, offline):
    from glossapi.cli import plan_documents
    with pytest.raises(Exception) as raised:
        plan_documents(tmp_path / 'out', files=['user:' + MARK2 + '@example.invalid/a.pdf'], stages=['extract'])
    assert MARK2 not in str(raised.value) + json.dumps(getattr(raised.value, 'details', {}))


@pytest.mark.parametrize('entry', ['python', 'cli'])
def test_r4_malformed_policy_error_does_not_echo_secret(tmp_path, offline, entry):
    from glossapi.cli import main, plan_documents
    policy = tmp_path / 'policy.yaml'
    policy.write_text('default:\n  domain_cookies: {session: ' + MARK2 + '\n')
    settings = {'download': {'download_policy_file': str(policy)}}
    if entry == 'python':
        with pytest.raises(ValueError) as raised:
            plan_documents(tmp_path / 'out', files=[GOOD2], stages=['extract'], settings=settings)
        assert MARK2 not in str(raised.value) and raised.value.__cause__ is None and raised.value.__suppress_context__
        assert '(line ' in str(raised.value)  # position only
    else:
        settings_file = tmp_path / 'settings.json'
        settings_file.write_text(json.dumps(settings))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(['plan', '--corpus', str(tmp_path / 'out'), '--files', GOOD2, '--settings', str(settings_file), '--json'])
        assert code == 2 and MARK2 not in buffer.getvalue()
        assert json.loads(buffer.getvalue())['errors'][0]['code'] == 'invalid_settings'


@pytest.mark.parametrize('entry', ['local_run', 'local_plan', 'empty_guard'])
def test_r5_public_local_operations_do_not_require_download_extra(tmp_path, entry):
    code = '''
import importlib.abc, sys
class NoYaml(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'yaml':
            raise ModuleNotFoundError('injected absent optional package', name='yaml')
sys.meta_path.insert(0, NoYaml())
from glossapi import Corpus
from glossapi.cli import plan_documents
from glossapi.ledger.urls import refuse_credentials
from glossapi.stages.scheduler import StreamingScheduler
def stopped(*args, **kwargs):
    raise RuntimeError('offline dispatch sentinel')
StreamingScheduler.execute = stopped
'''
    if entry == 'local_run':
        code += (f"\ntry:\n    Corpus({str(tmp_path / 'empty')!r}, {str(tmp_path / 'out')!r}).run([{str(tmp_path / 'missing.pdf')!r}], "
                 "stages=('extract',))\nexcept RuntimeError as error:\n    assert str(error) == 'offline dispatch sentinel'\n")
    elif entry == 'local_plan':
        code += f"\nplan_documents({str(tmp_path / 'out')!r}, files=[{str(tmp_path / 'missing.pdf')!r}], stages=['extract'], settings={{'download': {{}}}})\n"
    else:
        code += "\nrefuse_credentials(settings={})\n"
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
        env=dict(os.environ, GLOSSAPI_CONVERSION_LOG_FILE=str(tmp_path / 'conversion.log')), timeout=60)
    assert result.returncode == 0, result.stderr


def test_r5_a_local_corpus_run_completes_without_pyyaml(tmp_path):
    """A real local-only Corpus.run (extract, clean, export) in a process where PyYAML cannot be imported."""
    pytest.importorskip('glossapi_rs_cleaner')
    from test_ledger_fixset_lg import pdf
    source = pdf(tmp_path / 'in/local.pdf', 'LOCAL text without yaml')
    code = f'''
import importlib.abc, sys
class NoYaml(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'yaml':
            raise ModuleNotFoundError('injected absent optional package', name='yaml')
sys.meta_path.insert(0, NoYaml())
from glossapi import Corpus
result = Corpus({str(tmp_path / 'empty')!r}, {str(tmp_path / 'out')!r}).run([{str(source)!r}], stages=('extract', 'clean', 'export'),
                                                                             settings={{'extract': {{'repair_fonts': False}}}})
assert result['state'] == 'completed', result
assert 'yaml' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
        env=dict(os.environ, GLOSSAPI_CONVERSION_LOG_FILE=str(tmp_path / 'conversion.log')), timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    assert 'LOCAL text without yaml' in (tmp_path / 'out/export.jsonl').read_text()


def test_r7_direct_browser_policy_object_refused_before_route_receipt(tmp_path, offline, monkeypatch):
    from glossapi.download_policy import DownloadPolicy
    from glossapi.gloss_browser_downloader import BrowserGlossDownloader
    from glossapi.gloss_downloader import RateLimiter
    from glossapi.stages.receipts import capture_transport
    async def stop(*a, **kw):
        return False, '', '.pdf', 'stopped before HTTP', 0
    monkeypatch.setattr(BrowserGlossDownloader, '_preflight_download', stop)
    dl = BrowserGlossDownloader(output_dir=str(tmp_path / 'out'), sleep=0, rate_limit=0,
        download_policy=DownloadPolicy(default_options={'Cookie': MARK2}))
    try:
        with capture_transport('standard', {}) as receipt:
            with pytest.raises(CredentialsUnsupportedError):
                asyncio.run(dl.download_file(0, GOOD2, None, RateLimiter(1000000)))
        assert MARK2 not in json.dumps(receipt, default=str) and receipt['requests'] == []
    finally:
        dl.close_logs()


@pytest.mark.parametrize('url', [GOOD2, 'https://example.org/a.pdf?session=1&keyword=greek&page=2'])
def test_regular_per_file_download_path_accepts_public_queries(tmp_path, offline, monkeypatch, url):
    from glossapi.gloss_downloader import GlossDownloader, RateLimiter
    seen = []
    async def preflight(self, **kwargs):
        seen.append(kwargs['url'])
        return False, '', '.pdf', 'offline public path control', 0
    monkeypatch.setattr(GlossDownloader, '_preflight_download', preflight)
    dl = GlossDownloader(output_dir=str(tmp_path / 'out'), sleep=0, rate_limit=0)
    try:
        result = asyncio.run(dl.download_file(0, url, None, RateLimiter(1000000)))
        assert seen == [url] and result[3] == 'offline public path control'
    finally:
        dl.close_logs()


@pytest.mark.parametrize('entry', ['download_file', 'download_files', 'adownload_files', 'legacy'])
def test_offline_public_query_download_writes_exact_payload(tmp_path, offline, monkeypatch, entry):
    import glossapi.gloss_downloader as module
    from glossapi.gloss_downloader import GlossDownloader, RateLimiter
    from test_k2_thin_download_io import _Session
    payload = b'%PDF-1.4\npublic query test\n%%EOF'
    calls = []
    class Session(_Session):
        def __init__(self, **kwargs):
            super().__init__(GOOD2, [([payload],)])
            calls.append(self)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    async def bootstrap(self, session, url, headers, **kwargs):
        return headers
    monkeypatch.setattr(module.aiohttp, 'ClientSession', Session)
    monkeypatch.setattr(GlossDownloader, '_build_session_connector', lambda *a, **kw: None)
    monkeypatch.setattr(GlossDownloader, '_bootstrap_download_session', bootstrap)
    source = tmp_path / 'source.parquet'
    frame = pd.DataFrame({'url': [GOOD2], 'page': ['https://example.invalid/page?session=4&keyword=greek']})
    frame.to_parquet(source, index=False)
    out = tmp_path / 'out'
    config = dict(sleep=0, rate_limit=0, max_retries=1, pre_ping_domains=False, scheduler_mode='global', referer_column='page')
    if entry == 'legacy':
        result = Corpus(tmp_path / 'empty', out).download(source, **config)
        assert result.download_success.tolist() == [True] and result.url.tolist() == [GOOD2]
    else:
        dl = GlossDownloader(output_dir=str(out), **config)
        try:
            if entry == 'download_file':
                assert asyncio.run(dl.download_file(0, GOOD2, None, RateLimiter(1000000)))[0] is True
            elif entry == 'download_files':
                assert dl.download_files(str(source), input_frame=frame).download_success.tolist() == [True]
            else:
                assert asyncio.run(dl.adownload_files(str(source), input_frame=frame)).download_success.tolist() == [True]
        finally:
            dl.close_logs()
    assert [call for session in calls for call in session.calls] == [('get', GOOD2)]
    assert any(path.read_bytes() == payload for path in (out / 'downloads').glob('*.pdf'))
