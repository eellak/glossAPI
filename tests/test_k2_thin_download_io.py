"""K2 tenacity/aiofiles replacements (glossapi.download_io): fake-clock retry tests, tenacity and
aiofiles differential oracles (when installed), and the downloader's two retry loops end to end."""
import asyncio
import errno
import functools
import logging
import os
import threading
import time

import pytest

from glossapi import download_io
from glossapi.download_io import open_binary, retry_async, retry_wait
from reference_oracles import oracle as reference_oracle


class Retryable(Exception):
    pass


class Fatal(Exception):
    pass


class Clock:
    def __init__(self):
        self.sleeps = []

    async def sleep(self, seconds):
        self.sleeps.append(seconds)


def script(*outcomes):
    """An operation failing or returning per attempt; records the retry indexes it saw."""
    seen, raised = [], []
    async def operation(retry_index):
        seen.append(retry_index)
        outcome = outcomes[min(retry_index, len(outcomes) - 1)]
        if isinstance(outcome, type) and issubclass(outcome, BaseException):
            error = outcome(f'attempt {retry_index}')
            raised.append(error)
            raise error
        return outcome
    return operation, seen, raised


async def tenacity_retry(operation, *, attempts, retry_on, logger, level=logging.INFO, sleep=None):
    """The replaced tenacity configuration, verbatim (oracle)."""
    from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
    first, *rest = retry_on
    retry = retry_if_exception_type(first)
    for kind in rest:
        retry = retry | retry_if_exception_type(kind)
    async for attempt in AsyncRetrying(stop=stop_after_attempt(max(1, int(attempts))), wait=wait_exponential(multiplier=1, min=1, max=10),
                                       retry=retry, before_sleep=before_sleep_log(logger, level), reraise=True,
                                       **({'sleep': sleep} if sleep else {})):
        with attempt:
            return await operation(attempt.retry_state.attempt_number - 1)


def drive(implementation, outcomes, attempts, caplog):
    clock, logger = Clock(), logging.getLogger('k2.retry')
    operation, seen, raised = script(*outcomes)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger='k2.retry'):
        try:
            result = asyncio.run(implementation(operation, attempts=attempts, retry_on=(Retryable, asyncio.TimeoutError),
                                                logger=logger, sleep=clock.sleep))
            error = None
        except BaseException as caught:
            result, error = None, caught
    records = [(r.name, r.levelno, r.getMessage()) for r in caplog.records if r.name == 'k2.retry']
    return dict(result=result, seen=seen, sleeps=clock.sleeps, records=records,
                error=None if error is None else (type(error), str(error), error is (raised[-1] if raised else None)))


SCENARIOS = [
    (['ok'], 3), ([Retryable, 'ok'], 3), ([Retryable, Retryable, 'ok'], 3), ([Retryable] * 9, 3),
    ([Retryable] * 9, 7), ([Retryable] * 9, 1), ([Retryable] * 9, 0), ([asyncio.TimeoutError, 'ok'], 2),
    ([Retryable, Fatal], 5), ([Fatal], 5), ([Retryable, asyncio.CancelledError], 4), ([Retryable] * 4 + ['late'], 5),
]


@pytest.mark.parametrize(('outcomes', 'attempts'), SCENARIOS)
def test_retry_matches_tenacity_oracle(outcomes, attempts, caplog):
    reference_oracle('tenacity')
    own = drive(retry_async, outcomes, attempts, caplog)
    oracle = drive(tenacity_retry, outcomes, attempts, caplog)
    assert own == oracle


def test_attempts_waits_and_final_exception_identity(caplog):
    run = drive(retry_async, [Retryable] * 9, 7, caplog)
    assert run['seen'] == [0, 1, 2, 3, 4, 5, 6]
    assert run['sleeps'] == [1, 2, 4, 8, 10, 10]
    assert run['error'] == (Retryable, 'attempt 6', True)  # the last attempt's own exception object
    assert [message for _, level, message in run['records'] if level == logging.INFO] == [
        f'Retrying <unknown> in {wait} seconds as it raised Retryable: attempt {index}.'
        for index, wait in enumerate([1, 2, 4, 8, 10, 10])]
    assert [retry_wait(index) for index in range(8)] == [0, 1, 2, 4, 8, 10, 10, 10]


def test_non_retryable_and_cancellation_propagate_without_waiting(caplog):
    for outcome in (Fatal, asyncio.CancelledError, KeyboardInterrupt):
        run = drive(retry_async, [outcome], 5, caplog)
        assert run['seen'] == [0] and run['sleeps'] == [] and run['records'] == []
        assert run['error'][0] is outcome and run['error'][2]


def test_cancellation_during_backoff_stops_retrying():
    calls = []
    async def operation(retry_index):
        calls.append(retry_index)
        raise Retryable('x')
    async def main():
        task = asyncio.ensure_future(retry_async(operation, attempts=5, retry_on=(Retryable,), logger=logging.getLogger('k2')))
        await asyncio.sleep(0.05)  # inside the real 1 s backoff
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(main())
    assert calls == [0]


def test_receipt_backoff_matches_the_retry_waits():
    from glossapi.stages import receipts
    seen = []
    class Session:
        def get(self, url, **kwargs):
            class Response:
                async def __aenter__(self): return self
                async def __aexit__(self, *exc): return False
            return Response()
    original = receipts.request_receipt
    def recording(url, **kwargs):
        seen.append(kwargs['backoff_ms'])
        return original(url, **kwargs)
    receipts.request_receipt = recording
    try:
        async def main():
            for index in range(8):
                async with receipts.recorded_request(Session(), 'get', 'https://example.invalid/x', retry_index=index):
                    pass
        asyncio.run(main())
    finally:
        receipts.request_receipt = original
    assert seen == [retry_wait(index) * 1000 for index in range(8)]


# ------------------------------------------------------------------ binary writes

class FakeHandle:
    def __init__(self, *, write_delay=0.0, write_error=None, close_error=None):
        self.data, self.events, self.closed = bytearray(), [], False
        self.write_delay, self.write_error, self.close_error = write_delay, write_error, close_error
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def write(self, chunk):
        with self.lock:
            self.active += 1; self.peak = max(self.peak, self.active)
        self.events.append(('write-start', len(chunk)))
        time.sleep(self.write_delay)
        try:
            if self.write_error:
                raise self.write_error
            self.data.extend(chunk)
            return len(chunk)
        finally:
            self.events.append(('write-end', len(chunk)))
            with self.lock:
                self.active -= 1

    def close(self):
        self.events.append(('close',))
        self.closed = True
        if self.close_error:
            raise self.close_error


@pytest.fixture
def fake_open(monkeypatch):
    handles = []
    def install(**options):
        def opener(path, mode):
            handles.append(FakeHandle(**options))
            return handles[-1]
        monkeypatch.setattr(download_io, 'open', opener, raising=False)
        return handles
    return install


def test_chunks_are_written_in_order_byte_identical(tmp_path):
    chunks = [os.urandom(size) for size in [1, 65536, 3, 70000, 0, 4096] * 20]
    async def main(path):
        async with open_binary(path, 'wb') as handle:
            for chunk in chunks:
                await handle.write(chunk)
    asyncio.run(main(tmp_path/'own.bin'))
    assert (tmp_path/'own.bin').read_bytes() == b''.join(chunks)
    aiofiles = reference_oracle('aiofiles')
    async def reference(path):
        async with aiofiles.open(path, 'wb') as handle:
            for chunk in chunks:
                await handle.write(chunk)
    asyncio.run(reference(tmp_path/'aiofiles.bin'))
    assert (tmp_path/'own.bin').read_bytes() == (tmp_path/'aiofiles.bin').read_bytes()


def test_slow_disk_keeps_the_event_loop_responsive(fake_open):
    handles = fake_open(write_delay=0.3)
    ticks = []
    async def ticker(stop):
        while not stop.is_set():
            ticks.append(time.monotonic()); await asyncio.sleep(0.01)
    async def main():
        stop = asyncio.Event()
        task = asyncio.ensure_future(ticker(stop))
        async with open_binary('slow', 'wb') as handle:
            await handle.write(b'x' * 10)
        stop.set(); await task
    asyncio.run(main())
    assert len(ticks) >= 10 and handles[0].closed


def test_cancel_mid_write_closes_after_the_write_without_leak(fake_open):
    handles = fake_open(write_delay=0.3)
    async def writer():
        async with open_binary('cancel', 'wb') as handle:
            await handle.write(b'a' * 5)
            await handle.write(b'b' * 7)
    async def main():
        task = asyncio.ensure_future(writer())
        await asyncio.sleep(0.45)  # during the second write
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(main())
    handle = handles[0]
    assert handle.closed and handle.events[-1] == ('close',)
    assert handle.events[:-1] == [('write-start', 5), ('write-end', 5), ('write-start', 7), ('write-end', 7)]
    assert handle.peak == 1  # no close race with the in-flight write


def test_disk_full_error_propagates_and_closes(fake_open):
    full = OSError(errno.ENOSPC, 'No space left on device')
    handles = fake_open(write_error=full)
    async def main():
        async with open_binary('full', 'wb') as handle:
            await handle.write(b'x')
    with pytest.raises(OSError) as caught:
        asyncio.run(main())
    assert caught.value is full and handles[0].closed


def test_close_failure_propagates(fake_open):
    failure = OSError(errno.EIO, 'flush failed')
    handles = fake_open(close_error=failure)
    async def main():
        async with open_binary('close', 'wb') as handle:
            await handle.write(b'x')
    with pytest.raises(OSError) as caught:
        asyncio.run(main())
    assert caught.value is failure and handles[0].closed


def test_one_write_in_flight_bounds_buffering(fake_open):
    handles = fake_open(write_delay=0.05)
    async def main():
        async with open_binary('bounded', 'wb') as handle:
            await asyncio.gather(*(handle.write(bytes([i]) * 3) for i in range(6)))
    asyncio.run(main())
    assert handles[0].peak == 1 and len(handles[0].data) == 18


def test_no_leaked_descriptor_on_real_files(tmp_path):
    descriptors = lambda: len(os.listdir('/dev/fd'))
    before = descriptors()
    async def main():
        for index in range(30):
            try:
                async with open_binary(tmp_path/f'{index}.bin', 'wb') as handle:
                    await handle.write(b'x' * 100)
                    if index % 3 == 0:
                        raise Fatal('body failure')
            except Fatal:
                pass
    asyncio.run(main())
    assert descriptors() == before


def test_cancel_during_open_closes_the_late_handle(monkeypatch):
    handles = []
    def slow_open(path, mode):
        time.sleep(0.2); handles.append(FakeHandle()); return handles[-1]
    monkeypatch.setattr(download_io, 'open', slow_open, raising=False)
    async def main():
        async def opener():
            async with open_binary('late', 'wb'):
                pass
        task = asyncio.ensure_future(opener())
        await asyncio.sleep(0.05); task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.3)
    asyncio.run(main())
    assert handles and handles[0].closed


def _cancelled_open_with_late_handle(monkeypatch, close):
    """Cancel a caller while its open waits; release the open; return the max event-loop tick gap and the loop's
    unhandled-exception reports (CP-K2a F3, the reviewer's probe)."""
    opened = threading.Event()
    class Handle:
        def close(self):
            close()
    def slow_open(*args):
        opened.wait(timeout=2)
        return Handle()
    monkeypatch.setattr(download_io, 'open', slow_open, raising=False)
    unhandled = []
    async def run():
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))
        ticks = []
        async def ticker():
            for _ in range(70):
                ticks.append(time.monotonic())
                await asyncio.sleep(.005)
        async def acquire():
            async with open_binary('unused'):
                pass
        ticker_task = asyncio.create_task(ticker())
        task = asyncio.create_task(acquire())
        await asyncio.sleep(.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        opened.set()
        await ticker_task
        await asyncio.sleep(.05)
        return max(b - a for a, b in zip(ticks, ticks[1:]))
    return asyncio.run(run()), unhandled


def test_cancelled_open_closes_the_late_handle_off_the_event_loop(monkeypatch):
    finished = threading.Event()
    def slow_close():
        time.sleep(.2); finished.set()
    gap, unhandled = _cancelled_open_with_late_handle(monkeypatch, slow_close)
    assert finished.is_set()
    assert gap < .1, gap  # 9d5d372: ~0.2 s, the close ran on the event loop thread
    assert not unhandled


def test_cancelled_open_late_close_error_is_logged_not_unhandled(monkeypatch, caplog):
    def failing_close():
        raise OSError(errno.EIO, 'late close failed')
    with caplog.at_level(logging.WARNING, logger='glossapi.download_io'):
        gap, unhandled = _cancelled_open_with_late_handle(monkeypatch, failing_close)
    assert not unhandled  # 9d5d372: the error escaped into the loop's exception handler
    assert any('cancelled open failed' in record.getMessage() and 'late close failed' in record.getMessage()
               for record in caplog.records)
    assert not download_io.__dict__.get('_LATE_CLOSES', set())  # cleanup task released when done


def test_cancelled_open_whose_open_fails_leaves_nothing_unhandled(monkeypatch):
    released = threading.Event()
    def failing_open(*args):
        released.wait(timeout=2)
        raise PermissionError('denied')
    monkeypatch.setattr(download_io, 'open', failing_open, raising=False)
    unhandled = []
    async def run():
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))
        async def acquire():
            async with open_binary('unused'):
                pass
        task = asyncio.create_task(acquire())
        await asyncio.sleep(.02); task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        released.set()
        await asyncio.sleep(.1)
    asyncio.run(run())
    import gc; gc.collect()
    assert not unhandled


def test_text_modes_are_refused():
    with pytest.raises(ValueError):
        open_binary('x', 'w')


# ------------------------------------------------------------------ downloader loops

class _Content:
    def __init__(self, chunks, error):
        self.chunks, self.error = chunks, error
    async def iter_chunked(self, size):
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


class _Response:
    def __init__(self, url, chunks, error=None, status=200, body_error=None):
        self.url, self.status, self.history = url, status, ()
        self.headers = {'Content-Type': 'application/pdf'}
        self.content, self._body, self._status_error = _Content(chunks, error), b''.join(chunks), body_error
    def raise_for_status(self):
        if self._status_error:
            raise self._status_error
    async def read(self):
        return self._body
    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, url, attempts):
        self.url, self.attempts, self.calls = url, list(attempts), []
    def _next(self, method, url, options):
        self.calls.append((method, url))
        return _Response(url, *self.attempts.pop(0))
    def get(self, url, **options):
        return self._next('get', url, options)
    def post(self, url, **options):
        return self._next('post', url, options)


def downloader_run(tmp_path, monkeypatch, implementation, attempts, *, mode='stream', max_retries=3, transport=False, requester='get'):
    aiohttp = pytest.importorskip('aiohttp')
    from glossapi import gloss_downloader
    from glossapi.gloss_downloader import GlossDownloader
    from glossapi.stages.receipts import capture_transport
    clock, captured = Clock(), {}
    monkeypatch.setattr(gloss_downloader, 'retry_async', functools.partial(implementation, sleep=clock.sleep))
    root = tmp_path/implementation.__name__
    downloader = GlossDownloader(output_dir=root, pre_ping_domains=False, max_retries=max_retries, sleep=0)
    original = downloader._finalize_download_result
    async def finalize(**kwargs):
        captured['head'] = kwargs['content']
        if kwargs.get('tmp_path') is not None:
            captured['temp'] = kwargs['tmp_path'].read_bytes()
        return await original(**kwargs)
    downloader._finalize_download_result = finalize
    url = 'https://example.invalid/source.pdf'
    session = _Session(url, attempts)
    async def acquire():
        if mode == 'stream':
            return await downloader._download_via_streaming_get(session=session, row_index=3, url=url, headers={},
                timeout=None, retry_count=0, filename_base='ABC_123', referer=None)
        return await downloader.make_request(session, requester, url, {}, None)
    import contextlib
    context = capture_transport('standard', {}) if transport else contextlib.nullcontext({})
    with context as receipts:
        try:
            result, error = asyncio.run(acquire()), None
        except Exception as caught:
            result, error = None, (type(caught).__name__, str(caught))
    downloader.close_logs()
    files = {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob('*'))
             if path.is_file() and path.parent.name != 'logs'}
    requests = [{k: item.get(k) for k in ('retry_index', 'backoff_ms', 'partial_body', 'retained_body')}
                for item in (receipts.get('requests') or [])]
    return dict(result=result, error=error, sleeps=clock.sleeps, calls=session.calls, files=files, requests=requests,
                head=captured.get('head'), temp=captured.get('temp'))


def _attempts(aiohttp):
    final = b'%PDF-1.4\n' + b'F' * 70000 + b'%%EOF'
    first = b'%PDF-1.4\n' + b'P' * 80000  # longer than the 64 KiB head
    import yarl
    where = yarl.URL('https://example.invalid/source.pdf')
    info = aiohttp.RequestInfo(url=where, method='GET', headers={}, real_url=where)
    not_found = aiohttp.ClientResponseError(info, (), status=404, message='Not Found')
    return {
        'cut_then_ok': ([([first[:70000], first[70000:]], aiohttp.ClientPayloadError('stream cut')), ([final[:65536], final[65536:]],)], final),
        'timeout_then_ok': ([([first[:100]], asyncio.TimeoutError()), ([final],)], final),
        'all_cut': ([([first], aiohttp.ClientPayloadError('cut 1')), ([first[:10]], aiohttp.ClientPayloadError('cut 2')),
                     ([first[:5]], aiohttp.ClientPayloadError('cut 3'))], None),
        'http_404_retried': ([([], None, 404, not_found)] * 3, None),
        'non_retryable': ([([first[:100]], ValueError('decoder bug')), ([final],)], None),
    }


@pytest.mark.parametrize('transport', [False, True])
@pytest.mark.parametrize('case', ['cut_then_ok', 'timeout_then_ok', 'all_cut', 'http_404_retried', 'non_retryable'])
def test_streaming_loop_matches_tenacity_including_temp_and_head(tmp_path, monkeypatch, case, transport):
    aiohttp = pytest.importorskip('aiohttp')
    reference_oracle('tenacity')
    attempts, final = _attempts(aiohttp)[case]
    own = downloader_run(tmp_path, monkeypatch, retry_async, attempts, transport=transport)
    oracle = downloader_run(tmp_path, monkeypatch, tenacity_retry, attempts, transport=transport)
    assert own == oracle
    if final is not None:
        assert own['result'][0] is True and own['files']['downloads/ABC_123.pdf'] == final
        assert own['head'] == final[:65536] and own['temp'] == final  # a fresh head buffer and temp file per attempt
        assert not any(name.startswith('downloads/.part_') for name in own['files'])
    expected = {'cut_then_ok': [1], 'timeout_then_ok': [1], 'all_cut': [1, 2], 'http_404_retried': [1, 2], 'non_retryable': []}[case]
    assert own['sleeps'] == expected and len(own['calls']) == len(expected) + 1
    if transport and case == 'all_cut':
        assert [item['retained_body']['relative_path'] for item in own['requests'][:-1]] == ['request_bodies/3.request0.part', 'request_bodies/3.request1.part']


@pytest.mark.parametrize('requester', ['get', 'post'])
def test_buffered_request_loop_matches_tenacity(tmp_path, monkeypatch, requester):
    aiohttp = pytest.importorskip('aiohttp')
    reference_oracle('tenacity')
    attempts = [([b'x'], None, 200, aiohttp.ServerDisconnectedError()), ([b'%PDF-ok'],)]
    own = downloader_run(tmp_path, monkeypatch, retry_async, attempts, mode='buffered', requester=requester, transport=True)
    oracle = downloader_run(tmp_path, monkeypatch, tenacity_retry, attempts, mode='buffered', requester=requester, transport=True)
    assert own == oracle
    assert own['result'][0] == b'%PDF-ok' and [call[0] for call in own['calls']] == [requester] * 2
    assert [item['retry_index'] for item in own['requests']] == [0, 1] and own['requests'][1]['backoff_ms'] == 1000


def test_downloader_modules_no_longer_import_tenacity_or_aiofiles():
    import subprocess, sys
    from pathlib import Path
    code = '''
import importlib.abc, sys
class Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tenacity', 'aiofiles'}:
            raise ModuleNotFoundError('absent', name=fullname)
sys.meta_path.insert(0, Absent())
import glossapi.gloss_downloader, glossapi.gloss_browser_downloader
'''
    if not all(__import__('importlib').util.find_spec(name) for name in ('aiohttp', 'PIL')):
        pytest.skip('download/browser extras not installed')
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]/'src')})
    assert result.returncode == 0, result.stderr
