"""Measurement-only wall spans for performance attribution (K1/W12).

No profiler hooks; nested exclusive accounting. Nothing runs unless the
environment variable DIAG_TRACE names an output directory.

Origin: the step-10/11 helper (experiments ledger_step11/drivers/diagtrace.py).
K1/W12 repairs:
- LS11-G02: every ``syncfs``/``fsync`` call, ``SyncBatch.flush`` and
  ``LedgerWriter.durable_barrier`` is recorded as its own span, with
  /proc/meminfo Dirty and Writeback (kB) read immediately before and after the
  call, the descriptor's device and path, and the enclosing span name. A
  barrier's writeback can therefore be split into the run's own bytes versus
  other writers' bytes on the same filesystem.
- LS11-G03: names are class-qualified by ``trace_prepare`` (for example
  ``ledger/writer.py:SyncBatch.close`` versus ``ledger/writer.py:LedgerWriter.close``).
"""
import contextlib
import functools
import json
import os
from pathlib import Path
import threading
import time

_tls = threading.local()
_streams = {}
_stats = {}
_clock = time.perf_counter_ns
_MEMINFO_FIELDS = ('Dirty', 'Writeback')


def meminfo():
    """Dirty and Writeback in kB from /proc/meminfo; empty off Linux."""
    try:
        values = {}
        with open('/proc/meminfo') as stream:
            for line in stream:
                key, _, rest = line.partition(':')
                if key in _MEMINFO_FIELDS:
                    values[key + '_kB'] = int(rest.split()[0])
                    if len(values) == len(_MEMINFO_FIELDS):
                        break
        return values
    except OSError:
        return {}


def descriptor_facts(descriptor):
    facts = {}
    try:
        facts['st_dev'] = os.fstat(descriptor).st_dev
    except OSError:
        pass
    try:
        facts['path'] = os.readlink(f'/proc/self/fd/{descriptor}')
    except OSError:
        pass
    return facts


def emit(name, **details):
    directory = os.environ.get('DIAG_TRACE')
    if not directory:
        return
    pid = os.getpid()
    if pid not in _streams:
        Path(directory).mkdir(parents=True, exist_ok=True)
        _streams[pid] = open(Path(directory)/f'{pid}.jsonl', 'a', buffering=1)
    _streams[pid].write(json.dumps(dict(name=name, pid=pid, tid=threading.get_ident(),
        t=_clock(), **details), ensure_ascii=False, default=str)+'\n')


def current_span():
    stack = getattr(_tls, 'stack', None)
    return stack[-1][2] if stack else None


@contextlib.contextmanager
def span(name, category='orchestrator', record=True, details=None):
    if not os.environ.get('DIAG_TRACE'):
        yield
        return
    stack = getattr(_tls, 'stack', None)
    if stack is None:
        stack = _tls.stack = []
    item = [_clock(), 0, name, category, {}]
    stack.append(item)
    try:
        yield
    finally:
        end = _clock()
        stack.pop()
        elapsed = end-item[0]
        own = elapsed-item[1]
        item[4][category] = item[4].get(category, 0)+own
        if stack:
            stack[-1][1] += elapsed
            for bucket, value in item[4].items():
                stack[-1][4][bucket] = stack[-1][4].get(bucket, 0)+value
        phase = next((s[2] for s in stack if s[2].startswith(('phase:', 'barrier:', 'dag:'))), 'outside')
        key = (os.getpid(), threading.get_ident(), phase, category, name)
        old = _stats.get(key, (0, 0, 0))
        _stats[key] = (old[0]+1, old[1]+elapsed-item[1], old[2]+elapsed)
        if record:
            emit(name, start=item[0], end=end, duration_ns=elapsed, exclusive_ns=elapsed-item[1],
                category=category, phase=phase, categories_ns=item[4], **(details or {}))


def measured(name, category='orchestrator', record=True):
    def decorate(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            details = {}
            for value in args:
                if hasattr(value, 'attempt_id'):
                    details = dict(task=value.task_id, attempt=value.attempt_id)
                    if hasattr(value, 'operation'):
                        details['operation'] = value.operation
                    if hasattr(value, 'inputs'):
                        details['inputs'] = [str(a.path) for a in value.inputs]
                    break
            if kwargs.get('planned_task_id'):
                details['task'] = kwargs['planned_task_id']
            with span(name, category, record, details):
                result = function(*args, **kwargs)
                if isinstance(result, tuple) and result and hasattr(result[0], 'attempt_id'):
                    details.update(task=result[0].task_id, attempt=result[0].attempt_id)
                    if hasattr(result[0], 'operation'):
                        details['operation'] = result[0].operation
                return result
        return call
    return decorate


def writeback_measured(name, category='fsync', descriptor_argument=None):
    """Per-call durability span with Dirty/Writeback before and after (LS11-G02).

    The after-sample is taken inside the span, so the span's duration includes
    two /proc/meminfo reads (tens of microseconds)."""
    def decorate(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            if not os.environ.get('DIAG_TRACE'):
                return function(*args, **kwargs)
            details = dict(caller=current_span())
            if descriptor_argument is not None and len(args) > descriptor_argument and isinstance(args[descriptor_argument], int):
                details.update(descriptor_facts(args[descriptor_argument]))
            if args and isinstance(getattr(args[0], 'descriptors', None), dict):
                details['registered'] = len(args[0].descriptors)
            details['meminfo_before'] = meminfo()
            with span(name, category, True, details):
                try:
                    return function(*args, **kwargs)
                finally:
                    details['meminfo_after'] = meminfo()
        call._diag = True
        return call
    return decorate


def dump():
    directory = os.environ.get('DIAG_TRACE')
    if not directory:
        return
    rows = [dict(pid=k[0], tid=k[1], phase=k[2], category=k[3], name=k[4],
        calls=v[0], exclusive_ns=v[1], inclusive_ns=v[2]) for k, v in _stats.items() if k[0] == os.getpid()]
    Path(directory, f'{os.getpid()}.stats.json').write_text(json.dumps(rows, indent=2))


def install_native():
    """Wrap native kernels and every durability call in this process."""
    try:
        import glossapi_rs_cleaner as cleaner
        import glossapi_rs_noise as noise
    except ImportError:
        cleaner = noise = None
    for module, names, category in [(cleaner, ['run_complete_pipeline'], 'kernel'),
            (noise, ['score_markdown_directory_detailed'], 'kernel')]:
        if module is None:
            continue
        for name in names:
            original = getattr(module, name, None)
            if original is not None and not getattr(original, '_diag', False):
                wrapped = measured('native:'+name, category)(original)
                wrapped._diag = True
                setattr(module, name, wrapped)
    if not getattr(os.fsync, '_diag', False):
        os.fsync = writeback_measured('fsync', 'fsync', 0)(os.fsync)
    try:
        from glossapi.ledger import writer as ledger_writer
    except ImportError:
        return
    original = getattr(ledger_writer, 'sync_filesystem', None)
    if original is not None and not getattr(original, '_diag', False):
        ledger_writer.sync_filesystem = writeback_measured('syncfs', 'fsync', 0)(original)
    # SyncBatch.flush and LedgerWriter.durable_barrier get per-call spans from
    # trace_prepare (writeback_measured), under class-qualified names.
