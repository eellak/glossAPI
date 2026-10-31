"""K2 batch 2a: the metadata-parquet lock (filelock) across processes.

`parquet_schema._parquet_lock` serializes metadata-parquet writers with `filelock.FileLock`. These tests run
real processes against it: read-modify-write mutual exclusion, waiting and timeouts, release on exceptions,
recovery after a crashed holder, atomic replacement and no unlocked fallback.
"""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='POSIX process control')
# Below the declared floor the lock is not mutually exclusive (filelock 3.x lacks the unlinked-inode retry):
# such an environment is out of specification, see reviews/k2_batch2a/filelock-matrix.log.
filelock = pytest.importorskip('filelock', minversion='4.2.4')


def _spawn(code, *args, **kwargs):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)}
    return subprocess.Popen([sys.executable, '-c', code, *map(str, args)], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, **kwargs)


def _wait_for(path, timeout=20):
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        assert time.monotonic() < deadline, f'timed out waiting for {path}'
        time.sleep(0.01)


RMW = '''
import os, sys, time
from pathlib import Path
from glossapi.parquet_schema import _parquet_lock
target, counter, inside, overlaps, rounds = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
for _ in range(rounds):
    with _parquet_lock(target):
        try:
            os.close(os.open(inside, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            with open(overlaps, 'a') as log: log.write(f'{os.getpid()}\\n')
        value = int(counter.read_text())
        time.sleep(0.001)
        counter.write_text(str(value + 1))
        try: inside.unlink()
        except FileNotFoundError: pass
'''


def test_cross_process_read_modify_write_is_mutually_exclusive(tmp_path):
    target, counter = tmp_path / 'download_results.parquet', tmp_path / 'counter.txt'
    inside, overlaps = tmp_path / 'inside', tmp_path / 'overlaps.log'
    counter.write_text('0')
    workers, rounds = 8, 60
    processes = [_spawn(RMW, target, counter, inside, overlaps, rounds) for _ in range(workers)]
    for process in processes:
        _, err = process.communicate(timeout=180)
        assert process.returncode == 0, err
    assert not overlaps.exists(), f'{len(overlaps.read_text().split())} critical-section overlaps'
    assert int(counter.read_text()) == workers * rounds
    assert not (tmp_path / 'download_results.parquet.lock').exists()


HOLD = '''
import sys, time
from pathlib import Path
from glossapi.parquet_schema import _parquet_lock
target, ready, release, mark = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
with _parquet_lock(target):
    print(mark, 'entered', time.monotonic(), flush=True)
    ready.write_text('held')
    while not release.exists():
        time.sleep(0.01)
    print(mark, 'leaving', time.monotonic(), flush=True)
'''


def test_waiters_block_while_held_and_timeouts_raise(tmp_path):
    target = tmp_path / 'meta.parquet'
    lock_path = Path(str(target) + '.lock')
    holder = _spawn(HOLD, target, tmp_path / 'ready', tmp_path / 'release', 'holder')
    _wait_for(tmp_path / 'ready')
    # A bounded acquisition of the same lock path times out while the holder is inside.
    began = time.monotonic()
    with pytest.raises(filelock.Timeout):
        filelock.FileLock(str(lock_path), timeout=0.3).acquire()
    assert time.monotonic() - began >= 0.3
    # A second writer waits (no timeout in _parquet_lock) instead of entering.
    waiter = _spawn(HOLD, target, tmp_path / 'ready2', tmp_path / 'release2', 'waiter')
    time.sleep(1.0)
    assert not (tmp_path / 'ready2').exists(), 'second writer entered while the lock was held'
    (tmp_path / 'release').write_text('go')
    out, err = holder.communicate(timeout=30)
    assert holder.returncode == 0, err
    _wait_for(tmp_path / 'ready2')
    (tmp_path / 'release2').write_text('go')
    out2, err2 = waiter.communicate(timeout=30)
    assert waiter.returncode == 0, err2
    left = float(out.split('leaving')[1].split()[0])
    entered = float(out2.split('entered')[1].split()[0])
    assert entered >= left
    with filelock.FileLock(str(lock_path), timeout=0.3):
        pass


RAISE = '''
import sys
from pathlib import Path
from glossapi.parquet_schema import _parquet_lock
with _parquet_lock(Path(sys.argv[1])):
    raise RuntimeError('boom inside the lock')
'''


def test_exception_inside_the_lock_propagates_and_releases(tmp_path):
    from glossapi.parquet_schema import _parquet_lock
    target = tmp_path / 'meta.parquet'
    child = _spawn(RAISE, target)
    _, err = child.communicate(timeout=60)
    assert child.returncode == 1 and 'RuntimeError: boom inside the lock' in err
    with filelock.FileLock(str(target) + '.lock', timeout=0):
        pass
    with pytest.raises(KeyError):
        with _parquet_lock(target):
            raise KeyError('in-process')
    with filelock.FileLock(str(target) + '.lock', timeout=0):
        pass


def test_failed_write_keeps_the_previous_parquet_and_releases(tmp_path, monkeypatch):
    from glossapi import parquet_schema
    target = tmp_path / 'download_results.parquet'
    parquet_schema._write_metadata_parquet(pd.DataFrame({'filename': ['a.pdf'], 'url': ['u']}), target)
    before = target.read_bytes()
    def broken(table, path, *args, **kwargs):
        Path(path).write_bytes(b'partial')
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(parquet_schema.pq, 'write_table', broken)
    with pytest.raises(OSError, match='No space left'):
        parquet_schema._write_metadata_parquet(pd.DataFrame({'filename': ['b.pdf'], 'url': ['v']}), target)
    assert target.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['download_results.parquet']
    with filelock.FileLock(str(target) + '.lock', timeout=0):
        pass


CRASH = '''
import os, signal, sys
from pathlib import Path
import pandas as pd
from glossapi import parquet_schema
def die(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)
parquet_schema.os.replace = die  # crash after the temporary parquet is written, before it replaces the target
parquet_schema._write_metadata_parquet(pd.DataFrame({'filename': ['crash.pdf'], 'url': ['x']}), Path(sys.argv[1]))
'''


def test_crashed_holder_releases_the_lock_and_leaves_the_target_intact(tmp_path):
    from glossapi import parquet_schema
    target = tmp_path / 'download_results.parquet'
    parquet_schema._write_metadata_parquet(pd.DataFrame({'filename': ['a.pdf'], 'url': ['u']}), target)
    before = target.read_bytes()
    child = _spawn(CRASH, target)
    child.communicate(timeout=60)
    assert child.returncode == -signal.SIGKILL
    assert target.read_bytes() == before
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name != target.name)
    # Declared: the killed writer leaves its lock file and its uniquely named temporary parquet behind.
    assert any(name.endswith('.tmp') for name in leftovers) and 'download_results.parquet.lock' in leftovers
    began = time.monotonic()
    parquet_schema._write_metadata_parquet(pd.DataFrame({'filename': ['b.pdf'], 'url': ['v']}), target)
    assert time.monotonic() - began < 5
    assert pd.read_parquet(target)['filename'].tolist() == ['b.pdf']


WRITE = '''
import sys
from pathlib import Path
import pandas as pd
from glossapi.parquet_schema import _write_metadata_parquet
index, rounds = int(sys.argv[2]), int(sys.argv[3])
for round_ in range(rounds):
    frame = pd.DataFrame({'filename': [f'w{index}-{round_}-{row}.pdf' for row in range(200)], 'url': ['u'] * 200})
    _write_metadata_parquet(frame, Path(sys.argv[1]))
'''


def test_concurrent_writers_always_leave_one_complete_parquet(tmp_path):
    target = tmp_path / 'download_results.parquet'
    processes = [_spawn(WRITE, target, index, 15) for index in range(5)]
    seen = set()
    while any(process.poll() is None for process in processes):
        try:
            data = target.read_bytes()  # one open: a snapshot of whichever complete file is in place
        except FileNotFoundError:
            data = None
        if data is not None:
            frame = pd.read_parquet(io.BytesIO(data))  # never torn: os.replace under the lock
            assert len(frame) == 200 and frame['filename'].str.rsplit('-', n=1).str[0].nunique() == 1
            seen.add(frame['filename'][0].rsplit('-', 1)[0])
        time.sleep(0.005)
    for process in processes:
        _, err = process.communicate(timeout=60)
        assert process.returncode == 0, err
    assert len(pd.read_parquet(target)) == 200
    assert sorted(p.name for p in tmp_path.iterdir()) == ['download_results.parquet']


def test_no_unlocked_fallback_without_filelock():
    code = '''
import importlib.abc, sys
class NoFilelock(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'filelock':
            raise ModuleNotFoundError('injected absent filelock', name='filelock')
sys.meta_path.insert(0, NoFilelock())
try:
    import glossapi.parquet_schema
except ModuleNotFoundError as exc:
    assert exc.name == 'filelock', exc
else:
    raise AssertionError('metadata parquet writes would run unlocked')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)})
    assert result.returncode == 0, result.stderr



# ---------------------------------------------------------------- K2 batch 3: read-modify-write isolation

UPDATE = '''
import sys, time
from pathlib import Path
import pandas as pd
from glossapi.parquet_schema import ParquetSchema, update_metadata_parquet
target, writer, rows, ready, mode, total = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], int(sys.argv[6])
mine = pd.DataFrame({'filename': [f'w{writer}-r{i}.pdf' for i in range(rows)], 'url': [''] * rows})
def barrier():
    (ready / str(writer)).touch()
    while len(list(ready.iterdir())) < total:
        time.sleep(0.005)
if mode == 'locked':
    def add(current):
        time.sleep(0.02)  # hold the critical section while the other writers queue on the lock
        return pd.concat([current, mine], ignore_index=True)
    update_metadata_parquet(target, add)
else:  # the unlocked pattern this replaces: read, then a (locked) write
    current = pd.read_parquet(target) if target.exists() else pd.DataFrame()
    barrier()
    ParquetSchema().write_metadata_parquet(pd.concat([current, mine], ignore_index=True), target)
'''


@pytest.mark.parametrize('mode', ['locked', 'unlocked-control'])
def test_concurrent_writers_adding_distinct_rows_keep_every_row(tmp_path, mode):
    """N writers each add distinct rows. The locked read-modify-write keeps all of them; the control (read
    outside the lock, as before K2 batch 3) loses rows, so the test is not vacuous."""
    target, ready = tmp_path / 'download_results.parquet', tmp_path / 'ready'
    ready.mkdir()
    pd.DataFrame({'filename': ['seed.pdf'], 'url': ['']}).to_parquet(target, index=False)
    writers, rows = 6, 4
    # The control's barrier makes every writer read before any writes (deterministic loss without the lock).
    processes = [_spawn(UPDATE, target, writer, rows, ready, 'locked' if mode == 'locked' else 'unlocked', writers)
                 for writer in range(writers)]
    for process in processes:
        _, err = process.communicate(timeout=120)
        assert process.returncode == 0, err
    names = set(pd.read_parquet(target)['filename'])
    expected = {'seed.pdf'} | {f'w{w}-r{i}.pdf' for w in range(writers) for i in range(rows)}
    if mode == 'locked':
        assert names == expected
        assert len(pd.read_parquet(target)) == len(expected)
    else:
        assert names < expected and len(expected - names) >= rows, 'the unlocked control unexpectedly kept every row'


ENSURE = '''
import sys, time
from pathlib import Path
import pandas as pd
import glossapi.parquet_schema as schema
base, writer, writers = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
# Delay each writer's write after its scan; earlier writers wait longest, which reverses the write order.
# Before K2 batch 3 the write (the only locked part) came after an unlocked read, so a late writer replaced the
# file with its stale merge; now the read, merge and write are one critical section.
name = '_write_metadata_parquet_locked' if hasattr(schema, '_write_metadata_parquet_locked') else '_write_metadata_parquet'
real = getattr(schema, name)
def slow(*args, **kwargs):
    time.sleep(0.08 * (writers - writer))
    return real(*args, **kwargs)
setattr(schema, name, slow)
time.sleep(0.04 * writer)
(base / 'markdown' / f'doc{writer}.md').write_text('κείμενο', encoding='utf-8')
assert schema.ParquetSchema().ensure_metadata_parquet(base) == base / 'download_results' / 'existing.parquet'
'''


def test_concurrent_ensure_metadata_parquet_keeps_every_writers_rows(tmp_path):
    """`ensure_metadata_parquet` merges its artifact rows into an existing metadata parquet. The existing file
    is now read inside the lock: concurrent writers with interleaved reads keep every document."""
    (tmp_path / 'markdown').mkdir()
    (tmp_path / 'download_results').mkdir()
    pd.DataFrame({'url': ['https://example.org/seed.pdf'], 'filename': ['seed.pdf']}).to_parquet(
        tmp_path / 'download_results' / 'existing.parquet', index=False)
    writers = 6
    processes = [_spawn(ENSURE, tmp_path, writer, writers) for writer in range(writers)]
    for process in processes:
        _, err = process.communicate(timeout=120)
        assert process.returncode == 0, err
    frame = pd.read_parquet(tmp_path / 'download_results' / 'existing.parquet')
    assert set(frame['filename']) == {'seed.pdf'} | {f'doc{w}.pdf' for w in range(writers)}
    assert frame.loc[frame.filename == 'seed.pdf', 'url'].tolist() == ['https://example.org/seed.pdf']
    assert not (tmp_path / 'download_results' / 'download_results.parquet').exists()
    assert sorted(p.name for p in (tmp_path / 'download_results').iterdir()) == ['existing.parquet']
