"""Own async retry loop and thread-offloaded binary file writes for the downloaders.

They replace tenacity (two ``AsyncRetrying`` loops) and aiofiles (binary open/write/close), keeping
the behaviour the downloaders relied on:

- ``retry_async`` runs ``operation(retry_index)`` up to ``attempts`` times. Only instances of
  ``retry_on`` are retried; any other exception, including cancellation, propagates at once. Waits are
  tenacity's ``wait_exponential(multiplier=1, min=1, max=10)``: 1, 2, 4, 8, 10, 10, ... seconds. Before
  each wait it logs tenacity's ``before_sleep_log`` message. After the last attempt the last exception
  object is re-raised unchanged (tenacity's ``reraise=True``).
- ``open_binary`` opens, writes and closes a file in the default thread executor, so the event loop
  keeps running during slow disk I/O. One write is in flight at a time; ``close`` waits for an
  in-flight write first (also after the writer was cancelled), so there is no close race and no
  leaked handle. Write and close errors (disk full, failed flush) propagate unchanged.
"""
from __future__ import annotations

import asyncio
import logging


def retry_wait(retry_index):
    """Seconds slept before attempt ``retry_index`` (0-based); 0 for the first attempt."""
    return 0 if retry_index <= 0 else min(10, 2 ** (retry_index - 1))


async def retry_async(operation, *, attempts, retry_on, logger, level=logging.INFO, sleep=None):
    attempts = max(1, int(attempts))
    for retry_index in range(attempts):
        try:
            return await operation(retry_index)
        except retry_on as error:
            if retry_index + 1 >= attempts:
                raise
            failure = error
        wait = retry_wait(retry_index + 1)
        # tenacity.before_sleep_log; tenacity names the callable "<unknown>" for AsyncRetrying loops.
        logger.log(level, f"Retrying <unknown> in {'%.3g' % wait} seconds as it raised "
                          f"{failure.__class__.__name__}: {failure}.")
        del failure
        await (sleep or asyncio.sleep)(wait)


class BinaryFile:
    """A binary file handle whose blocking calls run in the default executor."""

    def __init__(self, handle):
        self._handle = handle
        self._pending = None
        self._lock = asyncio.Lock()

    async def write(self, data):
        async with self._lock:  # one write in flight, in call order
            await self._settle()
            self._pending = asyncio.ensure_future(asyncio.to_thread(self._handle.write, data))
            # Shielded: cancelling the writer must not hide a write that is still running in its thread.
            return await asyncio.shield(self._pending)

    async def _settle(self):
        pending, self._pending = self._pending, None
        if pending is not None:
            await asyncio.wait([pending])
            if not pending.cancelled():
                pending.exception()  # retrieved; the writer already saw it or was cancelled

    async def close(self):
        async def finish():
            await self._settle()
            await asyncio.to_thread(self._handle.close)
        await asyncio.shield(asyncio.ensure_future(finish()))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
        return False


class _Opening:
    def __init__(self, path, mode):
        self._path, self._mode, self._file = path, mode, None

    async def __aenter__(self):
        opening = asyncio.ensure_future(asyncio.to_thread(open, self._path, self._mode))
        try:
            handle = await asyncio.shield(opening)
        except asyncio.CancelledError:
            # The open may still complete in its thread: close that handle when it does.
            opening.add_done_callback(lambda done: done.cancelled() or done.exception() or done.result().close())
            raise
        self._file = BinaryFile(handle)
        return self._file

    async def __aexit__(self, *exc):
        await self._file.close()
        return False


def open_binary(path, mode='wb'):
    """``async with open_binary(path) as f: await f.write(chunk)`` (binary modes only)."""
    if 'b' not in mode:
        raise ValueError('open_binary supports binary modes only')
    return _Opening(path, mode)
