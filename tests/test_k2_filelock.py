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

