"""K1/W12 trace repairs: LS11-G02 (per-call durability spans) and LS11-G03 (class-qualified names)."""
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.harness import diagtrace
from tools.harness.common import REPO
from tools.harness.trace_prepare import prepare


def test_prepare_qualifies_names_and_marks_writeback(tmp_path):
    prepare(REPO, tmp_path/'trace')
    writer = (tmp_path/'trace/scratch/src/glossapi/ledger/writer.py').read_text()
    assert "@_dt.measured('ledger/writer.py:SyncBatch.close', 'ledger', False)" in writer
    assert "@_dt.measured('ledger/writer.py:LedgerWriter.close', 'ledger', False)" in writer
    assert "@_dt.writeback_measured('ledger/writer.py:SyncBatch.flush', 'ledger')" in writer
    assert "@_dt.writeback_measured('ledger/writer.py:LedgerWriter.durable_barrier', 'ledger')" in writer
    report = json.loads((tmp_path/'trace/instrumentation.json').read_text())
    assert len(report['text_patches']) == 11
    # Production source is untouched.
    assert '_dt.' not in (REPO/'src/glossapi/ledger/writer.py').read_text()


def test_durability_calls_are_recorded_per_call_with_meminfo(tmp_path):
    prepare(REPO, tmp_path/'trace')
    scratch = tmp_path/'trace/scratch'
    trace = tmp_path/'events'
    script = '''
import os, sys
from pathlib import Path
from tools.harness import diagtrace
diagtrace.install_native()
from glossapi.ledger import writer
target = Path(sys.argv[1]); target.mkdir(); (target/'a.txt').write_text('x')
with diagtrace.span('outer:test'):
    with writer.SyncBatch() as batch:
        batch.add_path(target/'a.txt'); batch.add_path(target)
        batch.flush()
diagtrace.dump()
'''
    environment = {**os.environ, 'DIAG_TRACE': str(trace), 'PYTHONPATH': f'{scratch/"src"}{os.pathsep}{scratch}'}
    subprocess.run([sys.executable, '-c', script, str(tmp_path/'data')], check=True, env=environment, cwd=tmp_path)
    events = [json.loads(line) for path in trace.glob('*.jsonl') for line in path.read_text().splitlines()]
    flush = [e for e in events if e['name'] == 'ledger/writer.py:SyncBatch.flush']
    assert len(flush) == 1 and flush[0]['caller'] == 'outer:test' and flush[0]['registered'] == 2
    assert 'meminfo_before' in flush[0] and 'meminfo_after' in flush[0]
    durability = [e for e in events if e['name'] in ('syncfs', 'fsync')]
    # Linux >= 5.8: one syncfs for the filesystem; otherwise one fsync per path.
    assert durability and all(e['caller'] == 'ledger/writer.py:SyncBatch.flush' for e in durability)
    assert all('duration_ns' in e and 'meminfo_after' in e and 'st_dev' in e for e in durability)
    if sys.platform.startswith('linux'):
        assert set(durability[0]['meminfo_before']) == {'Dirty_kB', 'Writeback_kB'}
    stats = [row for path in trace.glob('*.stats.json') for row in json.loads(path.read_text())]
    assert {'ledger/writer.py:SyncBatch.close', 'ledger/writer.py:SyncBatch.flush'} <= {row['name'] for row in stats}
    assert 'ledger/writer.py:LedgerWriter.close' not in {row['name'] for row in stats}


def test_disabled_tracer_is_transparent(monkeypatch):
    monkeypatch.delenv('DIAG_TRACE', raising=False)
    wrapped = diagtrace.writeback_measured('probe')(lambda value: value + 1)
    assert wrapped(1) == 2
    with diagtrace.span('nothing'):
        pass
