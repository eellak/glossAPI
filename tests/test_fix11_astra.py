"""Fix round 11 (ASTRA-FIXES2c6 on 5d793fb): default-deny recovery.

Every recognised recovery target is resolved by ONE function per kind (``LedgerWriter._resolve_excluded_copy`` at the
run start, ``resolve`` inside ``roll_back_uncommitted_writes``) into ``Resolved`` (positive evidence only) or
``Unresolved(path, reason)``. Only Resolved copies are made durable; any exception inside a resolution is Unresolved,
except an exhausted disk and a post-close-edit refusal.

F1 [HIGH] a copy that vanished at its flush was certified durable (``_fsync_path`` swallowed ENOENT); a ``missing`` or
   ``corrupt`` presence was skipped as if released; a registered generation whose source was lost was taken for a
   caller-seeded file. F2 [MEDIUM] the rollback's installed-bytes failure was a plain ``LedgerError``. F3 [LOW] the
   intent window started at the latest presence observation, not the registration. F4 [LOW] an unverifiable temporary
   kept blocking under a stale lock after its product was resolved.

``test_fault_sweep`` replaces enumerating skips: every filesystem operation recovery performs (one line in
OPERATIONS) x every fault (FAULTS) x every recovery scenario (SCENARIOS). A public resume never reports ``completed``
while a product does not match its committed hash, and an unhooked retry completes with the committed hashes or refuses
typed. The matrix is written to ``FIX11_FAULT_MATRIX`` when that variable names a file (fix11/reply.md).
"""
from __future__ import annotations

import builtins
import errno
import hashlib
import json
import os
import pathlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from glossapi import Corpus
import glossapi.parquet_schema as ps
from glossapi.ledger import encoding
from glossapi.ledger import writer as W
from glossapi.ledger.writer import LedgerWriter
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES, ledger_matches_files
from test_k4c2_fixes import published_copy

CONTENT = b'{"filename":"uncommitted.pdf","text":"bytes a failed attempt put in place"}\n'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refusal(function):
    try:
        return function(), None
    except Exception as error:  # noqa: BLE001 - the typed refusal is the observation
        return None, error


def _conditions(error):
    return [(Path(entry['path']).name, entry['reason']) for entry in getattr(error, 'conditions', [])]


# ===================================================== F1: Resolved needs positive evidence ===================

def test_flush_failure_is_unresolved_and_never_durable(tmp_path, monkeypatch):
    """Step: verified copy -> flush -> durable. The flush fails (the copy vanished): Unresolved ``missing_at_flush``,
    not durable, start refused; the next start finds the copy missing and republishes it. 5d793fb swallowed ENOENT
    in ``_fsync_path`` and marked the copy durable."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
    real = W._fsync_path

    def vanish(target):
        if Path(target) == path and path.exists():
            path.unlink()
        return real(target)
    monkeypatch.setattr(W, '_fsync_path', vanish)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        writer.verify_excluded_copies()
        _, error = _refusal(writer.refuse_ambiguous_copies)
        durable = copy_id in writer._durable_files
    monkeypatch.undo()
    assert _conditions(error) == [('export.jsonl', 'missing_at_flush')], repr(error)
    assert not durable
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        assert writer.verify_excluded_copies() == [copy_id]
        assert _refusal(writer.refuse_ambiguous_copies)[1] is None
    assert path.read_bytes() == source.read_bytes()


@pytest.mark.parametrize('presence', ['missing', 'corrupt'])
def test_bad_presence_without_supersession_is_repaired_not_skipped(tmp_path, presence):
    """Step: named copy observed ``missing``/``corrupt`` -> skip. No supersession in the ledger: the copy is resolved
    only by repairing it from its source (exact committed bytes, flushed). 5d793fb skipped it as released."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:  # no run context: its close records no new barrier
        row = writer.file_catalog()[copy_id]
        path.unlink()
        with writer.transaction():
            writer.event(run, event_kind='artifact', reason_code=presence, file_id=copy_id, version_id=row['version_id'],
                         details={'observation': 'fix11_probe'})
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        restored = writer.verify_excluded_copies()
        _, error = _refusal(writer.refuse_ambiguous_copies)
    assert restored == [copy_id] and error is None
    assert path.read_bytes() == source.read_bytes()


def test_superseded_copy_is_resolved_by_the_later_registration(tmp_path):
    """Positive evidence of supersession: a later registration of the same path. The old named copy is resolved without
    touching the file (which holds the newer generation)."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        newer = b'{"b": 2}\n' * 10
        path.write_bytes(newer)
        with writer.transaction():
            manifest = writer.snapshot('inputs', {'kind': 'inputs', 'files': ['second']})
            task = writer.task(run, 'export', 'export.jsonl', {}, scope_key='corpus:' + manifest)
            attempt = writer.attempt(task, worker_id='w', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
            writer.artifact(run, 'export', 'document_jsonl', path.as_uri(), producer_attempt_id=attempt,
                            sha256=hashlib.sha256(newer).hexdigest(), bytes=len(newer), retention_class='final')
        assert writer.file_catalog()[copy_id]['presence'] in ('missing', 'corrupt')
        assert writer.verify_excluded_copies() == []
        assert _refusal(writer.refuse_ambiguous_copies)[1] is None
    assert path.read_bytes() == newer


def test_rollback_registered_generation_without_source_is_unresolved(tmp_path):
    """Step: committed registration with a lost source -> "nothing to restore". It is Unresolved
    ``no_intact_source``; a genuinely caller-seeded file (no registration of the path) stays resolved. 5d793fb took
    the first for the second."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        seeded = tmp_path / 'seeded.parquet'
        seeded.write_bytes(CONTENT)
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix11-failed')
        for target in (path, seeded):
            target.write_bytes(CONTENT)
            writer.record_uncommitted_write(failed, target, _sha(target), len(CONTENT), wrapper_stage='export', existed=True)
        source.unlink()
        assert writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run) == []
        listed = _conditions(SimpleNamespace(conditions=writer.rollback_unresolved))
    assert listed == [('export.jsonl', 'no_intact_source')]
    assert path.read_bytes() == seeded.read_bytes() == CONTENT


# =========================================================== F2: rollback act failures are typed ==============

def test_rollback_installed_bytes_failure_is_typed(tmp_path, monkeypatch):
    """Step: restore -> installed bytes verified. The copied bytes are not the registered ones: typed
    ``unresolved_recovery`` (``source_changed``), the file kept; an unhooked retry restores. 5d793fb raised a plain
    ``LedgerError`` (CLI ``unsupported_schema``, exit 2). An exhausted disk is not converted (next test)."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix11-failed')
        writer.record_uncommitted_write(failed, path, hashlib.sha256(CONTENT).hexdigest(), len(CONTENT),
                                        wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)
        real = shutil.copyfile

        def changed(src, dst, *args, **kwargs):
            result = real(src, dst, *args, **kwargs)
            with open(dst, 'ab') as handle:
                handle.write(b'X')
            return result
        monkeypatch.setattr(shutil, 'copyfile', changed)
        _, error = _refusal(lambda: writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run))
        monkeypatch.undo()
        kept = path.read_bytes()
        retry = writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run)
    assert getattr(error, 'code', None) == 'unresolved_recovery' and _conditions(error) == [('export.jsonl', 'source_changed')]
    assert kept == CONTENT and retry == [str(ps.resolve_metadata_target(path))] and path.read_bytes() == source.read_bytes()


def test_rollback_disk_full_is_not_converted(tmp_path, monkeypatch):
    """An exhausted disk during the rollback's restore propagates unchanged (typed ``disk_full`` by the run)."""
    from glossapi.stages.execution import disk_full
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix11-failed')
        writer.record_uncommitted_write(failed, path, hashlib.sha256(CONTENT).hexdigest(), len(CONTENT),
                                        wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)

        def full(*args, **kwargs):
            raise OSError(errno.ENOSPC, 'No space left on device')
        monkeypatch.setattr(shutil, 'copyfile', full)
        _, error = _refusal(lambda: writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run))
    assert disk_full(error) and getattr(error, 'code', None) != 'unresolved_recovery', repr(error)


def test_rollback_hard_link_is_typed_and_keeps_the_alias_type(tmp_path):
    """The rollback's hard-link refusal is ``unresolved_recovery`` (CLI exit 3) and still the ``MetadataAliasError`` the
    direct API always raised."""
    path = tmp_path / 'product.jsonl'
    path.write_bytes(CONTENT)
    os.link(path, tmp_path / 'second-name')
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='fix11')
        writer.record_uncommitted_write(run, path, _sha(path), len(CONTENT), wrapper_stage='export', existed=False)
        _, error = _refusal(lambda: writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run))
    assert isinstance(error, ps.MetadataAliasError) and getattr(error, 'code', None) == 'unresolved_recovery'
    assert 'hard-linked' in str(error) and _conditions(error) == [('product.jsonl', 'hard_linked')]


# ========================================================= F3: the window starts at the registration ===========

@pytest.mark.parametrize('first', ['stale', 'canonical'])
def test_presence_observation_does_not_move_the_intent_window(tmp_path, first):
    """Step (N3): records after the copy's registration -> lock agreement. A ``present`` observation between two
    intents does not hide the first: both orders are ``conflicting_locks``. 5d793fb started the window at the latest
    presence event."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        canonical = ps.resolve_metadata_target(path).with_suffix('.jsonl.lock')
        locks = [tmp_path / 'stale.lock', canonical]
        if first == 'canonical':
            locks.reverse()
        digest = hashlib.sha256(CONTENT).hexdigest()
        writer.record_uncommitted_write(run, path, digest, len(CONTENT), wrapper_stage='export', existed=True, lock=locks[0])
        with writer.transaction():
            writer.event(run, event_kind='artifact', reason_code='present', file_id=copy_id, details={'observation': 'fix11_probe'})
        writer.record_uncommitted_write(run, path, digest, len(CONTENT), wrapper_stage='export', existed=True, lock=locks[1])
        path.write_bytes(CONTENT)
        restored = writer.verify_excluded_copies()
        _, error = _refusal(writer.refuse_ambiguous_copies)
    assert restored == [] and path.read_bytes() == CONTENT
    assert _conditions(error) == [('export.jsonl', 'conflicting_locks')], repr(error)


# ============================================================== F4: an unneeded temporary is residue ===========

@pytest.mark.parametrize('case', ['stale', 'conflicting'])
def test_unverifiable_temporary_is_residue_once_the_product_is_resolved(tmp_path, case):
    """Once the product's condition is gone (the target removed), its unverifiable temporary is retained as residue in
    both lock cases: no block, no deletion. 5d793fb kept blocking (``lock_mismatch``) in the stale case."""
    path, temporary = tmp_path / 'physical', tmp_path / '.physical.tmp'
    path.write_bytes(CONTENT)
    temporary.write_bytes(CONTENT)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='fix11')
        locks = [tmp_path / 'stale.lock'] + ([ps.resolve_metadata_target(path).with_suffix('.lock')] if case == 'conflicting' else [])
        for lock in locks:
            writer.record_uncommitted_write(run, path, _sha(path), len(CONTENT), wrapper_stage='clean', existed=False,
                                            temporary=temporary, lock=lock)
        writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        first = [entry['reason'] for entry in writer.rollback_unresolved]
        path.unlink()
        writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        second = list(writer.rollback_unresolved)
    assert first == ['lock_mismatch' if case == 'stale' else 'conflicting_locks']
    assert second == [] and temporary.read_bytes() == CONTENT


# ================================================================================ the fault-injection sweep ===

# One line per filesystem operation recovery and rollback perform: name -> (owner, attribute). ``W.os``/``ps.os`` are
# replaced by a proxy (only the writer's and the metadata choke point's calls are hooked); the others are patched in
# place. Hooks fire only while ``verify_excluded_copies`` or ``roll_back_uncommitted_writes`` runs, on the first call
# that touches the corpus output (path operations) or on the first call (descriptor operations).
OPERATIONS = {
    'open': ('os', 'open'),
    'read': ('os', 'read'),
    'stat': ('os', 'stat'),
    'lstat': ('os', 'lstat'),
    'fsync': ('os', 'fsync'),
    'replace': ('os', 'replace'),
    'unlink': ('os', 'unlink'),
    'path_stat': ('pathlib', 'stat'),
    'path_lstat': ('pathlib', 'lstat'),
    'builtin_open': ('builtins', 'open'),
    'copyfile': ('shutil', 'copyfile'),
    'hash': ('encoding', 'file_hash'),
}
FAULTS = ('missing', 'mismatch', 'oserror')
SCENARIOS = ('startup_intact', 'startup_restore', 'failed_clean', 'publication_crash')
_TEMPLATES: dict[str, tuple[Path, Path, str]] = {}


def _build(scenario, root, monkeypatch):
    """The recovery state of ``scenario`` at ``root``; returns (output, resume run id)."""
    output = root / 'out'
    if scenario == 'publication_crash':
        # A real process death at the export publication's replace, its temporary cut short (fix 8's crash child):
        # the resume's repeated barrier runs the rollback and its temporary cleanup (read, unlink).
        import subprocess
        import sys
        from glossapi.ledger.writer import open_readonly
        from test_fix8_astra import CRASH_CHILD
        root.mkdir(parents=True, exist_ok=True)
        script = root / 'crash_child.py'
        script.write_text(CRASH_CHILD)
        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
               'PYTHONPATH': os.pathsep.join([*filter(None, [os.environ.get('PYTHONPATH')]), str(repo / 'src'), str(repo / 'tests')])}
        done = subprocess.run([sys.executable, '-B', str(script), str(root), 'publication_partial', *map(str, _sources(root))],
                              capture_output=True, text=True, timeout=300, env=env, cwd=root)
        assert done.returncode == 92, done.stderr[-2000:]
        assert [p for p in output.iterdir() if p.name.startswith('.export.jsonl.')], 'the crash must leave the temporary'
        with open_readonly(output / 'corpus.sqlite3') as db:
            run = db.execute("SELECT run_id FROM tasks WHERE operation='export.compatibility_publish' ORDER BY rowid DESC LIMIT 1").fetchone()[0]
        return output, run
    if scenario == 'failed_clean':
        from test_fix8_astra import _failed_clean
        with pytest.MonkeyPatch.context() as patch:
            output, path, run = _failed_clean(root, patch)
        return output, run
    run = Corpus(root / 'empty', output).run(_sources(root), stages=STAGES, settings=SETTINGS6)['run_id']
    if scenario == 'startup_restore':
        path = output / 'export.jsonl'
        with LedgerWriter(output / 'corpus.sqlite3') as writer:
            writer.record_uncommitted_write(run, path, hashlib.sha256(CONTENT).hexdigest(), len(CONTENT),
                                            wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)
    return output, run


def _fresh(scenario, tmp_path_factory, monkeypatch):
    """A fresh copy of the scenario's state at the template's own absolute path (the ledger stores absolute URIs)."""
    if scenario not in _TEMPLATES:
        base = tmp_path_factory.mktemp('fix11-' + scenario)
        work, snapshot = base / 'work', base / 'snapshot'
        output, run = _build(scenario, work, monkeypatch)
        shutil.copytree(work, snapshot, symlinks=True)
        _TEMPLATES[scenario] = (work, snapshot, run)
    work, snapshot, run = _TEMPLATES[scenario]
    shutil.rmtree(work)
    shutil.copytree(snapshot, work, symlinks=True)
    return work, work / 'out', run


def _inject(monkeypatch, operation, fault, output, state):
    owner, name = OPERATIONS[operation]
    root = os.path.realpath(output)
    targets = [output / 'export.jsonl', output / 'download_results' / 'download_results.parquet',
               *sorted(output.glob('.export.jsonl.*.tmp'))]

    def touches(args):
        if operation in ('read', 'fsync'):
            return True
        if not args or not isinstance(args[0], (str, os.PathLike)):
            return False
        return os.path.realpath(os.fspath(args[0])).startswith(root)

    def corrupt(path=None):
        for target in [Path(path)] if path else targets:
            if target.is_file():
                data = target.read_bytes()
                target.write_bytes(bytes([data[0] ^ 1]) + data[1:] if data else b'x')
                return

    def wrap(real):
        def hooked(*args, **kwargs):
            if not state['window'] or state.get('fired') or not touches(args):
                return real(*args, **kwargs)
            state['fired'] = f'{operation}({os.fspath(args[0]) if args and isinstance(args[0], (str, os.PathLike)) else args[:1]!r})'
            if fault == 'missing':
                raise FileNotFoundError(errno.ENOENT, 'injected ' + operation)
            if fault == 'oserror':
                raise OSError(errno.EIO, 'injected ' + operation)
            if operation == 'hash':
                real(*args, **kwargs)
                return '0' * 64
            if operation == 'read':
                data = real(*args, **kwargs)
                return bytes([data[0] ^ 1]) + data[1:] if data else data
            if operation in ('replace', 'copyfile'):
                result = real(*args, **kwargs)
                corrupt(args[1])
                return result
            corrupt()
            return real(*args, **kwargs)
        return hooked

    if owner == 'os':
        for module in (W, ps):
            proxy = SimpleNamespace(**{key: getattr(os, key) for key in dir(os) if not key.startswith('__')})
            setattr(proxy, name, wrap(getattr(os, name)))
            monkeypatch.setattr(module, 'os', proxy)
    elif owner == 'pathlib':
        monkeypatch.setattr(pathlib.Path, name, wrap(getattr(pathlib.Path, name)))
    elif owner == 'builtins':
        monkeypatch.setattr(builtins, 'open', wrap(builtins.open))
    elif owner == 'shutil':
        monkeypatch.setattr(shutil, name, wrap(getattr(shutil, name)))
    else:
        hooked = wrap(encoding.file_hash)
        monkeypatch.setattr(encoding, 'file_hash', hooked)
        monkeypatch.setattr(W, 'file_hash', hooked)
    for method in ('verify_excluded_copies', 'roll_back_uncommitted_writes'):
        real = getattr(LedgerWriter, method)

        def windowed(self, *args, real=real, **kwargs):
            state['window'] = True
            try:
                return real(self, *args, **kwargs)
            finally:
                state['window'] = False
        monkeypatch.setattr(LedgerWriter, method, windowed)


def _outcome(function, output):
    result, error = _refusal(function)
    if error is not None:
        return {'state': 'refused', 'code': getattr(error, 'code', None) or type(error).__name__,
                'reasons': sorted({entry['reason'] for entry in getattr(error, 'conditions', [])})}
    return {'state': result['state'], 'mismatches': ledger_matches_files(output)}


@pytest.mark.parametrize('scenario', SCENARIOS)
@pytest.mark.parametrize('fault', FAULTS)
@pytest.mark.parametrize('operation', list(OPERATIONS))
def test_fault_sweep(tmp_path_factory, monkeypatch, operation, fault, scenario):
    """For every operation x fault x scenario: the hooked public resume never reports ``completed`` while a product
    differs from its committed hash; the unhooked retry completes with the committed hashes, or refuses typed
    (``unresolved_recovery``, or ``post_close_edit`` when the injected fault left bytes nobody recorded)."""
    work, output, run = _fresh(scenario, tmp_path_factory, monkeypatch)
    state = {'window': False}
    _inject(monkeypatch, operation, fault, output, state)
    first = _outcome(lambda: Corpus(work / 'empty', output).run(resume=run, retry_failed=True), output)
    monkeypatch.undo()
    retry = _outcome(lambda: Corpus(work / 'empty', output).run(resume=run, retry_failed=True), output)
    row = {'operation': operation, 'fault': fault, 'scenario': scenario, 'fired': state.get('fired'), 'first': first, 'retry': retry}
    if os.environ.get('FIX11_FAULT_MATRIX'):
        with open(os.environ['FIX11_FAULT_MATRIX'], 'a') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    if first['state'] == 'completed':
        assert not first['mismatches'], row
    assert first['state'] != 'refused' or first['code'] in ('unresolved_recovery', 'post_close_edit', 'DiskFullError'), row
    assert (retry['state'] == 'completed' and not retry['mismatches']) or \
        (retry['state'] == 'refused' and retry['code'] in ('unresolved_recovery', 'post_close_edit')), row
