"""Single-writer SQLite fact journal; no orchestration or backend imports."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import uuid

from .catalogue import EVENT_KINDS, REASON_CODES, STAGES, STATUSES, TERMINAL_STATUSES
from .encoding import canonical_json, evidence, file_hash, snapshot_hash, typed_value, untyped_value
from .schema import APPEND_ONLY_TABLES, SCHEMA_SQL, SCHEMA_VERSION, COMPACT_SCHEMA_SQL, COMPACT_RETENTION_SCHEMA_SQL, FORMAT_SCHEMA_SQL
from .compact import connect, encode_json
from .validation import validate_evidence


_OPEN_WRITERS = set()
# K4(c) part 4 (coordinator decision on K4C3-G4): the full integrity check (PRAGMA integrity_check, every snapshot's
# checksum, canonical JSON and evidence schema, foreign keys) runs once per process per ledger. A writer of this process
# that closes a ledger cleanly (no -wal left) records the file's (dev, ino, size, mtime_ns); the next writer in the same
# process skips the full check only while the file still has exactly that identity and no -wal exists. Anything else
# (first open in a process, a file changed by anyone else, a leftover WAL) runs the full check. The schema/migration
# checks still run at every open.
#
# CP-K4b F1/F3/F4, CP-K4c3 R2: trust is installed only by a writer that validated the ledger at open (full check, or
# an unchanged trusted identity) and closed cleanly through one fence: after its own complete checkpoint it observes the
# file identity (empty WAL), then finds ``PRAGMA data_version`` unchanged since the baseline taken right after
# connecting, and after closing finds that same identity with no WAL. What this guarantees: no other connection
# committed between the open baseline and the version comparison, and the file changed neither by size nor by
# timestamp between the identity observation and the close. It does not cover a same-size write that keeps both mtime
# and ctime (one timestamp tick; K4F1-G1). A failed constructor or close removes the entry and never
# installs one. The identity includes ``st_ctime_ns`` (set by the kernel on any write or metadata change; ``utime``
# cannot restore it). A forked child starts with no trust (``os.register_at_fork``).
_integrity_known: dict[str, tuple[int, int, int, int, int]] = {}


def _content_digest(connection: Any) -> str:
    """sha256 over every table's rows in rowid order (a migration backup's logical identity, CP-K4b F6)."""
    digest = hashlib.sha256()
    tables: list[str] = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for table in tables:
        digest.update(table.encode() + b'\0')
        rows: Any = connection.execute(f'SELECT * FROM "{table}"')
        for row in rows:
            digest.update(repr(tuple(row)).encode())
    return digest.hexdigest()


def _ledger_identity(path: Path, *, allow_wal: bool = False) -> tuple[int, int, int, int, int] | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    if not allow_wal and os.path.exists(str(path)+'-wal'):
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _wal_empty(path: Path) -> bool:
    """No committed frames outstanding: the ``-wal`` is absent or truncated to zero bytes."""
    try:
        return os.stat(str(path)+'-wal').st_size == 0
    except FileNotFoundError:
        return True


def _forget_integrity_after_fork() -> None:
    _integrity_known.clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_forget_integrity_after_fork)


class LedgerError(ValueError):
    """Invalid evidence, ownership, transition or unsupported schema."""


class PostCloseEditError(LedgerError):
    """A copy the last run close left unflushed was changed after that close (CP-K4ac2 R2).

    The edited copy and its managed source are both kept. Nothing runs or publishes until the caller resolves it: move
    or delete the edited copy (it is then republished from the managed source), or write to another corpus.

    Typed for the CLI (``code``): exit 3, a conflict the user resolves, as for the other refusals (K4g B1c)."""
    code = 'post_close_edit'
    EXIT_CODE = 3
    MEANING = ('A corpus copy the last run left unflushed (export.jsonl, a metadata parquet) was changed after that run '
               'closed; it is kept. Move or delete it (it is republished from the managed source) or use another corpus.')

    def __init__(self, copies: list[dict[str, Any]]) -> None:
        self.copies: list[dict[str, Any]] = list(copies)
        names = ', '.join(sorted({str(entry['path']) for entry in self.copies})[:5])
        self.details: dict[str, Any] = {'paths': sorted({str(entry['path']) for entry in self.copies})}
        super().__init__('post_close_edit: changed after the last run closed, kept as is: ' + names +
                         '. Move or delete these files (they are republished from the managed source) or use another corpus')


class UnresolvedRecoveryError(LedgerError):
    """Recovery recognised a file it must repair but could not repair it safely (fix 10, N1/N3: recovery is fail-closed).

    The file is a copy the last run close left unflushed (run start, ``verify_excluded_copies``) or bytes a failed
    publication attempt put in place (the repeated barrier, ``roll_back_uncommitted_writes``), and the repair was
    refused: the target is hard-linked, its intent records bind contradictory locks or a lock its writers no longer
    take, the path was replaced (a symlink), the managed source changed or is missing, or the file changed right after
    its restore. The file is kept as it is. Nothing runs or publishes while a condition stands, and the run never
    reports ``completed``. Every start (and every repeated barrier) re-derives the conditions from the ledger, so
    removing the cause (for example the extra hard link) lets the next resume repair the file and complete.

    ``conditions``: one dict per target, ``path`` and ``reason`` (``hard_linked``, ``conflicting_locks``,
    ``lock_mismatch``, ``path_replaced``, ``source_changed``, ``no_intact_source``, ``changed_after_restore``).
    Typed for the CLI (``code``): exit 3, a conflict the user resolves, as ``post_close_edit``."""
    code = 'unresolved_recovery'
    EXIT_CODE = 3
    MEANING = ('Recovery found a corpus file it must repair (a copy the last run left unflushed, or bytes a failed attempt '
               'put in place) but could not repair it safely (hard-linked, contradictory lock records, path replaced, '
               'managed source changed or missing); the file is kept and nothing ran. Remove the cause (e.g. the extra '
               'hard link) and resume: the file is then repaired.')

    def __init__(self, conditions: list[dict[str, Any]]) -> None:
        self.conditions: list[dict[str, Any]] = [dict(entry) for entry in conditions]
        self.details: dict[str, Any] = {'paths': sorted({str(entry['path']) for entry in self.conditions}),
                                        'conditions': [{key: (str(value) if key == 'path' else value) for key, value in entry.items()}
                                                       for entry in self.conditions]}
        shown = '; '.join(f"{entry['path']} ({entry['reason']})" for entry in self.conditions[:5])
        super().__init__('unresolved_recovery: recovery could not safely repair, kept as is: ' + shown +
                         '. Remove the cause (e.g. an extra hard link) and resume; nothing ran')


def _unique_conditions(conditions):
    """Conditions without repeats of the same (path, reason)."""
    seen, unique = set(), []
    for entry in conditions:
        key = (str(entry['path']), entry['reason'])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def _agreed_lock(locks):
    """The one metadata lock a path's intent records bind (fix 9, F1; fix 10, N3: shared by the repeated-barrier rollback,
    its temporary cleanup and the run-start recovery): ``(lock, conflicting)``. Records binding no lock are not counted
    (the canonical lock is taken for them, ``_recover_locked``); two different bound locks for one physical file cannot
    both be right: ``conflicting`` is True and the path must not be acted on (an unresolved recovery)."""
    bound = {str(lock) for lock in locks if lock}
    return (next(iter(bound)) if len(bound) == 1 else None), len(bound) > 1


def _load_syncfs():
    """Linux syncfs(2); it reports writeback errors only from kernel 5.8 on."""
    import sys
    if not sys.platform.startswith('linux'):return None
    match=re.match(r'(\d+)\.(\d+)',os.uname().release)
    if not match or tuple(map(int,match.groups()))<(5,8):return None
    import ctypes
    try:function=ctypes.CDLL(None,use_errno=True).syncfs
    except (OSError,AttributeError):return None
    function.argtypes=[ctypes.c_int];function.restype=ctypes.c_int
    return function


_SYNCFS=_load_syncfs()  # kept for diagnostics; barriers never call syncfs (K4(c) decision 1)


def sync_filesystem(descriptor):
    """Flush every file and directory of descriptor's filesystem and wait."""
    if _SYNCFS(descriptor)!=0:
        import ctypes
        error=ctypes.get_errno()
        raise OSError(error,os.strerror(error))


class SyncBatch:
    """One durability flush for a batch of changed files and directories.

    Each registered file and then each registered directory is fsynced exactly
    once per batch. A batch never flushes the whole filesystem (syncfs): the
    durability contract makes durable exactly the paths a barrier depends on
    (K4(c) coordinator decision 1).
    """
    def __init__(self):
        self.descriptors={}

    def add(self,descriptor):
        """Register an open file or directory descriptor (it is duplicated)."""
        import stat
        info=os.fstat(descriptor);key=(info.st_dev,info.st_ino)
        if key not in self.descriptors:self.descriptors[key]=(os.dup(descriptor),stat.S_ISDIR(info.st_mode))

    def add_path(self,path):
        descriptor=os.open(path,os.O_RDONLY)
        try:self.add(descriptor)
        finally:os.close(descriptor)

    def flush(self):
        files=[descriptor for descriptor,directory in self.descriptors.values() if not directory]
        directories=[descriptor for descriptor,directory in self.descriptors.values() if directory]
        fsync_all(files)
        fsync_all(directories)

    def close(self):
        for descriptor,_ in self.descriptors.values():os.close(descriptor)
        self.descriptors={}

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def _fsync_path(path):
    try:descriptor=os.open(path,os.O_RDONLY)
    except FileNotFoundError:return  # Missing observations are diagnosed by verification/resume.
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _recover_locked(path, decide, lock=None):
    """Check and act of a recovery in ONE critical section (fix 8, N1) of the lock its writers take (fix 9, F1).

    ``decide(target)`` runs while the lock is held: it re-verifies whatever the caller checked without the lock (the
    bytes, the symlink, the committed generation, a temporary's content) and returns the action, which is then done
    before the lock is released: ``None`` (nothing), ``('restore', source, sha256)`` (republish ``source``'s bytes
    at ``target``, only if the copied bytes are exactly ``sha256``) or ``('remove', file[, identity])`` (``identity``:
    the ``(st_dev, st_ino)`` the caller verified; the name is unlinked only if it still names that single-link regular
    file). Returns the action taken (``None`` if nothing was done).

    The lock is never inferred from the file name (fix 9, F1; fix 8 took it only for a lowercase ``.parquet`` path, so a
    metadata file behind a symlink to ``physical.bin`` or ``physical.PARQUET`` was recovered unlocked and a concurrent
    locked edit was lost). It is always the canonical metadata lock of ``path`` (``parquet_schema.metadata_lock_path``,
    held through ``metadata_target``), the lock every metadata writer of that file takes whatever its name, suffix or
    content (``update_metadata_parquet`` and the other choke-point writers, metadata product publications):
    - ``lock`` given (an intent record that binds the lock its writer held): exactly that lock. If it is not the lock
      the metadata writers of ``path`` take now (the path no longer resolves to the file the writer locked), nothing
      is done.
    - ``lock`` None (an older record, a publication made without a metadata lock (FIX8-G2), a run-start copy check):
      the canonical lock of ``path`` all the same. The metadata APIs accept any path (any suffix or none, any symlink),
      so no path is known to be outside their reach; taking a lock nobody else holds costs one lock file.
    ``target`` is the resolved physical file; nothing is done if the path no longer resolves to it under the lock (a
    symlink swapped in after the unlocked look). A locked writer can no longer commit between the check and the act:
    its edit is either seen by ``decide`` or made after the act, on top of it. A restore of a write made under a
    metadata lock goes through ``_copy_file_locked`` (as the writer's own replace); any other through
    ``_restore_copy``. Both verify the installed bytes first.
    Fix 10 (N1): a refusal raises ``UnresolvedRecoveryError`` (``lock_mismatch``: the recorded lock is not the lock
    the path's writers take now; ``path_replaced``: the path no longer resolves to the file the lock was taken for); a
    hard-linked target raises ``MetadataAliasError`` from ``metadata_target``. ``None`` means only that ``decide``
    found nothing of the recovery's own to act on."""
    path=Path(path)
    from ..parquet_schema import _copy_file_locked, metadata_lock_path, metadata_target, resolve_metadata_target
    canonical=metadata_lock_path(path)
    if lock is not None and Path(lock)!=canonical:
        # Fix 10 (N1): a refusal is never a silent "nothing to do"; the caller records it as an unresolved recovery.
        raise UnresolvedRecoveryError([{'path':str(path),'reason':'lock_mismatch','recorded_lock':str(lock),'canonical_lock':str(canonical)}])
    with metadata_target(path) as physical:
        if resolve_metadata_target(path)!=physical or metadata_lock_path(path)!=canonical:
            raise UnresolvedRecoveryError([{'path':str(path),'reason':'path_replaced'}])
        action=decide(physical)
        if action is None:
            return None
        if action[0]=='restore':
            if lock is not None:
                _copy_file_locked(action[1],physical,expected_sha256=action[2])
            else:
                _restore_copy(action[1],physical,action[2])
        elif len(action)>2 and not _same_single_file(action[1],action[2]):
            return None
        else:
            os.unlink(action[1])
        return action


def _same_single_file(path, identity):
    """``path`` still names the single-link regular file ``identity`` (``(st_dev, st_ino)``), checked by ``lstat``."""
    import stat as _stat
    try:
        info=os.lstat(path)
    except FileNotFoundError:
        return False
    return _stat.S_ISREG(info.st_mode) and info.st_nlink==1 and (info.st_dev,info.st_ino)==tuple(identity)


def _restore_copy(source, path, expected_sha256):
    """Republish a non-metadata copy (export.jsonl, markdown) from its intact managed source (K4C-M2, fix 7 R1):
    temporary file, atomic replace, and only if the copied bytes are exactly ``expected_sha256`` (fix 8: the bytes
    actually installed are verified, so a source that changed after its own check is never published). Called by
    ``_recover_locked``; a .parquet copy goes through metadata_target + _copy_file_locked there instead."""
    import shutil
    temporary=path.with_name('.'+path.name+'.'+new_id()+'.restore')
    try:
        shutil.copyfile(source,temporary)
        if file_hash(temporary)!=expected_sha256:
            raise LedgerError('Managed source changed before its restore; nothing was replaced: '+str(path))
        os.replace(temporary,path)
    finally:
        if os.path.lexists(temporary):os.unlink(temporary)


# Block size of the single pass in ``_verified_temporary`` (a module value so a test can make the pass take several
# reads and act between them).
_TEMPORARY_BLOCK=1<<20


def _read_block(descriptor, size):
    """One read of the verification pass (the seam a test uses to change a file between two reads)."""
    return os.read(descriptor,size)


def _read_exactly(descriptor, size):
    data=b''
    while len(data)<size:
        block=_read_block(descriptor,size-len(data))
        if not block:break
        data+=block
    return data


def _verified_temporary(temporary, details):
    """The ``(st_dev, st_ino)`` of a verified abandoned temporary of a recorded uncommitted write, else ``None`` (fix 8,
    N3; fix 9, F2). Verified means: the exact path the record binds, a regular file with one name (checked on the name
    and on the opened descriptor, which must be the same file), and its bytes either the recorded ones (``bytes`` and
    ``sha256``) or, for a publication temporary whose copy was cut short, a strict prefix of its recorded ``source``.

    The prefix proof reads the source ONCE, through one descriptor: the same bytes are compared with the temporary and
    hashed, and the temporary is a prefix only if those bytes hash to the recorded ``sha256`` (and have the recorded
    length). Fix 8 hashed the source, then reopened it and compared a prefix of whatever it held by then: a source
    changed in between was compared as a different generation and a modified temporary was deleted. A source changed
    during the pass (in place) hashes to something else, and the temporary is kept; one replaced by a new file leaves
    the opened descriptor on the recorded bytes. Never decided by the name. The caller establishes that no writer owns
    the temporary any more (see ``roll_back_uncommitted_writes``), runs this inside ``_recover_locked`` and removes the
    name only if it still names the identity returned here."""
    import stat as _stat
    nofollow=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)
    try:
        info=os.lstat(temporary)
    except FileNotFoundError:
        return None
    if not _stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
        return None
    try:
        descriptor=os.open(temporary,nofollow)
    except OSError:
        return None
    try:
        opened=os.fstat(descriptor)
        identity=(opened.st_dev,opened.st_ino)
        if identity!=(info.st_dev,info.st_ino) or not _stat.S_ISREG(opened.st_mode) or opened.st_nlink!=1:
            return None
        size=opened.st_size
        if size==details.get('bytes'):
            digest=hashlib.sha256();read=0
            while True:
                block=_read_block(descriptor,_TEMPORARY_BLOCK)
                if not block:break
                digest.update(block);read+=len(block)
            if read==size and digest.hexdigest()==details['sha256']:
                return identity
        source=details.get('source')
        if not source:
            return None
        try:
            full=os.open(source,nofollow)
        except OSError:
            return None
        try:
            if not _stat.S_ISREG(os.fstat(full).st_mode):
                return None
            os.lseek(descriptor,0,os.SEEK_SET)
            digest=hashlib.sha256();total=0;compared=0;prefix=True
            while True:
                block=_read_block(full,_TEMPORARY_BLOCK)
                if not block:break
                digest.update(block)
                if prefix and compared<size:
                    piece=_read_exactly(descriptor,min(len(block),size-compared))
                    prefix=bool(piece) and piece==block[:len(piece)]
                    compared+=len(piece)
                total+=len(block)
        finally:
            os.close(full)
        if not prefix or compared!=size or _read_block(descriptor,1):
            return None  # not a prefix of the bytes read, or the temporary holds more than it did at the open
        if size>=total or digest.hexdigest()!=details['sha256'] or details.get('bytes') not in (None,total):
            return None  # not a strict prefix, or the source bytes read are not the recorded generation
        return identity
    finally:
        os.close(descriptor)


def _abandoned_temporary(temporary, details):
    """True only for a verified abandoned temporary of a recorded uncommitted write (``_verified_temporary``)."""
    return _verified_temporary(temporary, details) is not None


def _map_bounded(function,items):
    """Run function over items on at most 8 threads (fsync releases the GIL; ext4 groups the journal commits)."""
    items=list(items)
    if len(items)<4:return [function(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8,len(items))) as pool:
        return list(pool.map(function,items))


def fsync_all(descriptors):
    """fsync every descriptor; concurrent calls let the filesystem group its journal commits."""
    descriptors=list(descriptors)
    if len(descriptors)<4:
        for descriptor in descriptors:os.fsync(descriptor)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8,len(descriptors))) as pool:
        for _ in pool.map(os.fsync,descriptors):pass


# Outputs that are never released within a run: recovery frontiers and evidence.
_RETAINED_PRIVATE_KINDS=frozenset(('download_payload','worker_receipt','generation_manifest','ocr_native','ocr_page_state'))


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def new_id():
    return str(uuid.uuid4())


_CANONICAL_ID = re.compile('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')


def validate_id(value):
    # Equal to str(uuid.UUID(value)) == value (the canonical lowercase spelling), without a UUID object (hot path).
    if not _CANONICAL_ID.fullmatch(value):
        raise LedgerError("IDs must be canonical UUIDs")
    return value


def validate_hash(value, *, nullable=False):
    if value is None and nullable:
        return value
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LedgerError("Expected lowercase SHA-256")
    return value


def validate_time(value):
    if value is None:
        return
    try:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)", value): raise ValueError()
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() != datetime.timedelta(0): raise ValueError()
    except (ValueError, AttributeError):
        raise LedgerError("Expected RFC3339 UTC timestamp") from None


def statements(script):
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            yield statement
            statement = ""
    if statement.strip():
        raise LedgerError("Incomplete migration SQL")


def integrity(connection):
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise LedgerError("Ledger integrity check failed")
    for digest, encoded, version in connection.execute("SELECT sha256,json,schema_version FROM snapshots"):
        if version != 1 or hashlib.sha256(encoded.encode("utf-8")).hexdigest() != digest:
            raise LedgerError("Snapshot checksum/schema mismatch")
        decoded = json.loads(encoded)
        validate_evidence(decoded)
        if canonical_json(decoded) != encoded: raise LedgerError("Snapshot is not canonical JSON")
    if connection.execute("PRAGMA foreign_key_check").fetchone():
        raise LedgerError("Ledger foreign key check failed")


def supported_schema(connection):
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version not in (1, 2, 3, SCHEMA_VERSION):
        raise LedgerError(f"Unsupported ledger schema {version}")
    rows = connection.execute("SELECT version,migration_sha256 FROM schema_migrations ORDER BY version").fetchall()
    if [r[0] for r in rows] != list(range(1, version + 1)):
        raise LedgerError("Schema migration history disagrees with user_version")
    if rows[0][1] != hashlib.sha256(SCHEMA_SQL.encode()).hexdigest():
        raise LedgerError("Schema migration checksum mismatch")
    if version >= 2 and rows[1][1] != hashlib.sha256(COMPACT_SCHEMA_SQL.encode()).hexdigest():
        raise LedgerError("Compact schema migration checksum mismatch")
    if version >= 3 and rows[2][1] != hashlib.sha256(COMPACT_RETENTION_SCHEMA_SQL.encode()).hexdigest():
        raise LedgerError("Compact retention schema migration checksum mismatch")
    if version >= 4 and rows[3][1] != hashlib.sha256(FORMAT_SCHEMA_SQL.encode()).hexdigest():
        raise LedgerError("Ledger format schema migration checksum mismatch")


@contextmanager
def open_readonly(path):
    path = Path(path).absolute()
    wal = path.with_name(path.name + "-wal")
    # SQLite normally creates WAL/SHM sidecars even for mode=ro. A closed,
    # checkpointed ledger needs neither. A live ledger must already have its
    # writer-owned WAL index; never create one as a side effect of reporting.
    if wal.exists() and not path.with_name(path.name + "-shm").exists():
        raise LedgerError("Live WAL has no shared-memory index; retry after writer checkpoint")
    connection = connect(path.as_uri() + ("?mode=ro" if wal.exists() else "?mode=ro&immutable=1"), uri=True, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA query_only=ON")
        supported_schema(connection)
        connection.execute("BEGIN")
        yield connection
    finally:
        if connection.in_transaction: connection.rollback()
        connection.close()


class LedgerWriter:
    """One local process/thread owns writes; workers return observations instead.

    The owner flock is on the existing corpus directory: no extra lock file,
    no conflict with SQLite, and no loss of ownership when a report closes. The
    SQLite connection is intentionally not part of a Corpus (spawn/pickle safe).
    All public mutation methods require an explicit transaction.
    """
    def __init__(self, path, *, code_revision="unknown_not_recorded", storage_version=SCHEMA_VERSION):
        if storage_version not in (1, 2, 3, SCHEMA_VERSION):raise LedgerError("Unsupported creation schema")
        self.path = Path(path).absolute()
        if str(self.path) in _OPEN_WRITERS:
            raise BlockingIOError("Ledger already has an owning writer")
        self.owner = (os.getpid(), threading.get_ident())
        # Typed for the checker (W19, CP-K4b F5): open while the writer is usable, None after close.
        self.connection: Any = None
        self.control_cache: Any = None
        self._file_catalog: Any = None
        self._run_plans = {}
        self._planned_reader_maps = {}
        self.snapshot_cache = set()
        self.produced_files = {}
        # Set by the owning run: {'run_id','debug'}; selects per-barrier records or one run summary.
        self.barrier_context: dict[str, Any] | None = None
        # debug=on keeps intermediates: their copies may rely on the retained private source.
        self.retained_intermediates = False
        self.barrier_totals: dict[str, Any] = dict(barrier_count=0,file_count=0,byte_count=0,directory_count=0,unlink_count=0,flush_ns=0,by_reason={})
        self.lock = os.open(self.path.parent, os.O_RDONLY)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _OPEN_WRITERS.add(str(self.path))
            # The file identity before this process opens it (opening creates the WAL); see _integrity_known.
            opened_identity = _ledger_identity(self.path)
            self.connection = connect(self.path, isolation_level=None)
            # WAL first: switching a new file into WAL mode itself advances data_version.
            self.connection.execute("PRAGMA journal_mode=WAL")
            # The external-commit epoch starts here, before validation or trust reuse (CP-K4bc2 R1): any commit by
            # another connection from now on changes data_version, and the baseline is never reset afterwards.
            self._data_version = self.connection.execute('PRAGMA data_version').fetchone()[0]
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA busy_timeout=5000")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                if self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                    raise LedgerError("Unversioned nonempty database")
                with self.transaction():
                    for statement in statements(SCHEMA_SQL): self.connection.execute(statement)
                    for table in APPEND_ONLY_TABLES:
                        for action in ("UPDATE", "DELETE"):
                            self.connection.execute(f"CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only table'); END")
                    # Hashes may be filled exactly once; other version facts never change.
                    self.connection.execute("""CREATE TRIGGER versions_fill_once BEFORE UPDATE ON document_versions
                      WHEN OLD.document_id IS NOT NEW.document_id OR OLD.version_id IS NOT NEW.version_id
                        OR OLD.revision IS NOT NEW.revision OR OLD.created_at IS NOT NEW.created_at
                        OR (OLD.source_sha256 IS NOT NULL AND OLD.source_sha256 IS NOT NEW.source_sha256)
                        OR (OLD.stored_sha256 IS NOT NULL AND OLD.stored_sha256 IS NOT NEW.stored_sha256)
                      BEGIN SELECT RAISE(ABORT, 'immutable version'); END""")
                    self.connection.execute("INSERT INTO ledger_info VALUES(1,?,?)", (new_id(), utc_now()))
                    self.connection.execute("INSERT INTO schema_migrations VALUES(1,?,?,?)",
                        (utc_now(), hashlib.sha256(SCHEMA_SQL.encode()).hexdigest(), code_revision))
                    self.connection.execute("PRAGMA user_version=1")
                    if storage_version >= 2:
                        for statement in statements(COMPACT_SCHEMA_SQL):self.connection.execute(statement)
                        self.connection.execute("INSERT INTO schema_migrations VALUES(2,?,?,?)",
                            (utc_now(),hashlib.sha256(COMPACT_SCHEMA_SQL.encode()).hexdigest(),code_revision))
                        self.connection.execute("PRAGMA user_version=2")
                    if storage_version >= 3:
                        for statement in statements(COMPACT_RETENTION_SCHEMA_SQL):self.connection.execute(statement)
                        self.connection.execute('INSERT INTO path_root VALUES(1,?)',(self.path.parent.as_uri(),))
                        self.connection.execute("INSERT INTO schema_migrations VALUES(3,?,?,?)",
                            (utc_now(),hashlib.sha256(COMPACT_RETENTION_SCHEMA_SQL.encode()).hexdigest(),code_revision))
                        self.connection.execute('PRAGMA user_version=3')
                    if storage_version >= 4:
                        for statement in statements(FORMAT_SCHEMA_SQL):self.connection.execute(statement)
                        self.connection.execute("INSERT INTO schema_migrations VALUES(4,?,?,?)",
                            (utc_now(),hashlib.sha256(FORMAT_SCHEMA_SQL.encode()).hexdigest(),code_revision))
                        self.connection.execute('PRAGMA user_version=4')
                self.connection.compact = storage_version >= 2
                self.connection.storage_version = storage_version
                if storage_version>=3:self.connection.path_root=self.path.parent.as_uri()
            supported_schema(self.connection)
            if self.connection.execute("SELECT 1 FROM documents WHERE source_key>='url:v2:' AND source_key<'url:v2;' LIMIT 1").fetchone():
                # Development builds before 1 Nov 2026 keyed query-bearing URLs by a corpus secret (never released).
                raise LedgerError("This ledger keys URLs in the withdrawn url:v2 development format; use a new output directory")
            known = _integrity_known.pop(str(self.path.resolve()), None)
            # Trust reuse is fenced against the database actually opened: the file must still have the identity
            # observed before connecting (no swap or rewrite between the stat and the open).
            if known is None or known != opened_identity or _ledger_identity(self.path, allow_wal=True) != opened_identity:
                integrity(self.connection)
            # The scheduler's indexed control projection is process local. Old
            # creation scripts/checksums and read-only historical reports stay exact.
            self.connection.execute(f"CREATE TEMP VIEW current_task_state AS SELECT t.*,e.status,e.reason_code,e.event_id FROM tasks t LEFT JOIN {self.connection.event_table} e ON e.event_id=(SELECT MAX(s.event_id) FROM {self.connection.event_table} s WHERE s.task_id=t.task_id AND s.event_kind='lifecycle')")
            self.connection.execute('CREATE INDEX IF NOT EXISTS snapshots_kind ON snapshots(kind)')
            self.connection.execute('CREATE INDEX IF NOT EXISTS tasks_run_operation ON tasks(run_id,operation)')
            self.connection.execute('CREATE INDEX IF NOT EXISTS tasks_stage_operation ON tasks(stage,operation)')
            self.connection.execute(f"CREATE INDEX IF NOT EXISTS events_decision_task ON {self.connection.event_table}(event_kind,task_id,event_id) WHERE event_kind='decision'")
            file_table='file_records' if self.connection.storage_version>=3 else 'files'
            if self.connection.storage_version<4:
                # Version 4 dropped it: no query plan uses it (K4C-G05).
                self.connection.execute(f'CREATE INDEX IF NOT EXISTS files_stage_kind ON {file_table}(stage,kind)')
            if self.connection.storage_version>=3:
                self.connection.execute('CREATE INDEX IF NOT EXISTS source_records_manifest_lookup ON source_records(manifest_sha256)')
            event_tables=[self.connection.event_table]
            if self.connection.compact:event_tables.append('retention_events')
            if self.connection.storage_version>=3:event_tables.append('completed_deletions')
            self.next_event_id=1+max(self.connection.execute(
                f'SELECT COALESCE(MAX(event_id),0) FROM {table}').fetchone()[0] for table in event_tables)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            # Writeback, not representation: a larger page cache keeps the hot
            # index pages in process, and fewer, larger checkpoints rewrite each
            # hot page (and fsync the database) less often. Records are unchanged.
            self.connection.execute("PRAGMA cache_size=-65536")
            self.connection.execute("PRAGMA wal_autocheckpoint=16384")
            self._durable_files=set()
            self._durable_directories={}
            self.file_catalog()
            self._durable_files=self._durable_after_last_run_close()
            self.connection.owner_writer=self
            # Validated at the epoch taken right after connecting. A commit by another connection during validation or
            # initialisation means what was validated is no longer what is open: validate again now, and keep the
            # original baseline so this writer never installs trust (CP-K4bc2 R1).
            if self.connection.execute('PRAGMA data_version').fetchone()[0] != self._data_version:
                integrity(self.connection)
            self._validated=True
        except BaseException:
            self.close()
            raise

    def _owner(self, *, transaction=True):
        if self.owner != (os.getpid(), threading.get_ident()):
            raise LedgerError("Only the owning orchestrator thread may write")
        if transaction and not self.connection.in_transaction:
            raise LedgerError("An explicit observation transaction is required")

    @contextmanager
    def transaction(self):
        self._owner(transaction=False)
        if self.connection.in_transaction:
            if not getattr(self, '_batch_active', False):
                raise LedgerError("Nested observation transactions are not allowed")
            self.connection.execute('SAVEPOINT observation')
            try:
                yield self
                self.connection.execute('RELEASE observation')
            except BaseException:
                self.control_cache = None
                self._file_catalog = None
                self._run_plans.clear()
                self._planned_reader_maps.clear()
                self.snapshot_cache.clear()
                self.connection.execute('ROLLBACK TO observation')
                self.connection.execute('RELEASE observation')
                raise
            return
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            self.control_cache = None
            self._file_catalog = None
            self._run_plans.clear()
            self._planned_reader_maps.clear()
            self.snapshot_cache.clear()
            raise

    @contextmanager
    def batch(self):
        """Bounded scheduler dispatch/result batches share one durable commit."""
        with self.transaction():
            self._batch_active = True
            try:
                yield self
            finally:
                self._batch_active = False

    def close(self):
        """Run-close durability barrier and VACUUM, then release the connection, the directory lock and the identity
        root in every case (CP-K4a F5); the original failure is re-raised after the release. Integrity trust is
        installed only after a validated, externally unchanged, clean close (CP-K4b F1)."""
        key = str(self.path.resolve()) if getattr(self, 'path', None) is not None else None
        failure = None
        trusted = False
        before = None
        try:
            if self.connection is not None:
                if self.connection.in_transaction:
                    self.connection.rollback()
                    self._file_catalog = None
                if hasattr(self,'_durable_files'):self.durable_barrier()
                if self.barrier_context:self._vacuum_at_run_close()
                if getattr(self, '_validated', False):
                    # The close fence (CP-K4c3 R2). Fold our own WAL into the file completely; then observe the
                    # candidate identity (with an empty WAL) BEFORE the last data_version comparison. A commit by
                    # another connection before that comparison changes data_version; one after it changes the file
                    # (or leaves a WAL), which the post-close comparison below rejects.
                    checkpoint = self.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
                    complete = checkpoint is not None and checkpoint[0] == 0 and checkpoint[1] == checkpoint[2]
                    candidate = _ledger_identity(self.path, allow_wal=True) if complete and _wal_empty(self.path) else None
                    unchanged = self.connection.execute('PRAGMA data_version').fetchone()[0] == self._data_version
                    before = candidate if unchanged else None
                    trusted = before is not None
        except BaseException as error:
            failure = error
        finally:
            if self.connection is not None:
                try:
                    self.connection.close()
                except BaseException as error:
                    failure = failure or error
                self.connection = None
            if key is not None:
                _integrity_known.pop(key, None)
                if trusted and failure is None:
                    # The file must still have the identity observed inside the fence, with no WAL outstanding: a
                    # commit by anyone after the data_version comparison changes it (or leaves a WAL), and then
                    # nothing is trusted. Only a same-size write within the same timestamp tick escapes (K4F1-G1).
                    after = _ledger_identity(self.path)
                    if after is not None and after == before:
                        _integrity_known[key] = after
            if getattr(self, "lock", None) is not None:
                os.close(self.lock)
                self.lock = None
                _OPEN_WRITERS.discard(str(self.path))
        if failure is not None:
            raise failure

    def _vacuum_at_run_close(self) -> None:
        """Representation only (K4(c) part 3, coordinator decision 4): after the run-close barrier, rebuild the
        ledger file without free pages and partly filled pages. Every record, id and order is unchanged (every
        table keeps its rows and rowids; VACUUM is one transaction, so a death inside it leaves the committed
        pre-VACUUM ledger). A reader holding the database (SQLITE_BUSY/LOCKED) only skips the compaction."""
        connection=self.connection
        if connection is None or connection.in_transaction:
            return
        try:
            connection.executescript('VACUUM')
        except sqlite3.OperationalError as error:
            if not any(word in str(error).lower() for word in ('busy','locked')):raise

    def durable_barrier(self, unlinking=(), *, private=False, reason=None):
        """Make durable exactly what a pending unlink (or the run close) depends on, then FULL-commit.

        Ordinary WAL commits protect process death. This is the explicit disk
        boundary used before unlink and at run end (amended durability contract,
        K4(c) coordinator decisions 1-3):
        - ``unlinking`` empty (run close): every present managed file not yet
          made durable, except a byte copy whose retained managed source is (or
          becomes, in this barrier) durable;
        - ``unlinking`` given: the present byte copies of the files to be
          unlinked (same producer attempt and hash, i.e. the user-visible bytes
          that replace them) and, unless ``private``, every output of an attempt
          that read them (the outputs that replace a deleted source or product);
        - ``private`` (bulk release of private scratch after its last planned
          reader): private scratch is never made durable; the stage reruns
          after a crash.
        Each file is fsynced, then each changed parent directory inside the
        corpus, then the ledger commits with synchronous=FULL. No syncfs.
        The cache is by immutable registration, never by stat metadata and
        never used to bypass a hash.
        """
        self._owner(transaction=False)
        if self.connection.in_transaction:
            raise LedgerError('Durability barrier requires a committed observation')
        import time as _time
        began=_time.monotonic_ns()
        catalog=self.file_catalog()
        # Every registration this barrier can see has an event id below this (compact ledgers keep the counter).
        through=self.next_event_id if self.connection.compact and not unlinking else None
        unlinking=set(unlinking)
        if unlinking:
            selected=self._barrier_dependents(catalog,unlinking,private)
        else:
            selected=[row for row in catalog.values() if row['ownership']=='managed' and row['presence']=='present'
                and row['file_id'] not in self._durable_files]
        candidates: set[str] = {row['file_id'] for row in cast(list[Any], selected)}
        selected=self._without_durable_copies(catalog,selected,unlinking)
        # Run close: the byte copies left unflushed because their managed source is durable (decision 3). They are
        # named in the barrier record so that the next run start can verify them and republish a torn one (K4C-M2).
        excluded: list[str] = sorted(candidates-{row['file_id'] for row in cast(list[Any], selected)}) if not unlinking else []
        from urllib.parse import unquote, urlsplit
        import stat
        root=self.path.parent
        directories=set();synchronized=[];paths=[];covered_bytes=0
        for row in selected:
            path=Path(unquote(urlsplit(row['uri']).path))
            try:info=os.stat(path)
            except OSError:continue  # Missing observations are diagnosed by verification/resume.
            if not stat.S_ISREG(info.st_mode):continue
            paths.append(path);covered_bytes+=info.st_size
            directories.update(parent for parent in path.parents if parent.is_relative_to(root))
            synchronized.append(row['file_id'])
        # Each path is opened, flushed and closed by one task, so at most a pool's worth of descriptors is open.
        _map_bounded(_fsync_path,paths)
        ordered=sorted(directories,key=lambda path:len(path.parts),reverse=True)
        started=_time.time_ns()
        cache=self._durable_directories
        def flush_directory(directory):
            try:descriptor=os.open(directory,os.O_RDONLY)
            except OSError:return None
            try:
                info=os.fstat(descriptor)
                key=(info.st_dev,info.st_ino);state=(info.st_mtime_ns,info.st_ctime_ns)
                # A directory whose entries are unchanged since its last fsync needs none. Directory
                # times are coarse (one clock tick), so the skip is trusted only for a state that was
                # already more than a second old when that fsync started.
                cached=cache.get(key)
                if cached and cached[0]==state and max(state)<cached[1]-1_000_000_000:return None
                os.fsync(descriptor)
                return key,state
            finally:os.close(descriptor)
        # Deepest first: a parent is flushed after the entries below it.
        flushed=[]
        for depth in sorted({len(path.parts) for path in ordered},reverse=True):
            flushed.extend(item for item in _map_bounded(flush_directory,[path for path in ordered if len(path.parts)==depth]) if item)
        for key,state in flushed:cache[key]=(state,started)
        synced_directories=len(flushed)
        flushed_ns=_time.monotonic_ns()-began
        record: dict[str, Any]=dict(reason=reason or ('run_close' if not unlinking else 'private_release' if private else 'policy_delete'),
            file_count=len(synchronized),byte_count=covered_bytes,directory_count=synced_directories,unlink_count=len(unlinking),flush_ns=flushed_ns)
        context=self.barrier_context
        summary=self._barrier_totals(record)
        if excluded:
            record['excluded_copy_file_ids']=excluded
            summary['run_close_excluded_copy_file_ids']=excluded
            # Pre-close identity of each unflushed copy, aligned with the ids: [mtime_ns, ctime_ns] (CP-K4a F6). A torn
            # copy keeps (or lowers) them; a write after the close's clock tick (a user's edit, even with mtime
            # restored) raises ctime, and such a copy is never overwritten. A write within the same tick keeps them:
            # the next start then classifies the copy by its bytes (K4F1-G1, ``_left_by_power_loss``).
            identities: list[list[int] | None] = []
            for file_id in excluded:
                try:
                    uri: str = catalog[file_id]['uri']
                    info=os.lstat(unquote(urlsplit(uri).path))
                    identities.append([info.st_mtime_ns,info.st_ctime_ns])
                except (OSError,KeyError):
                    identities.append(None)
            record['excluded_copy_identities']=identities
            summary['run_close_excluded_copy_identities']=identities
        if through is not None:
            # After this commit every present managed file registered below `through` is durable except the
            # excluded copies: the next writer of this ledger starts from that, not from an empty cache.
            record['durable_through_event_id']=through
            summary['run_close_durable_through_event_id']=through
        self.connection.execute('PRAGMA synchronous=FULL')
        try:
            with self.transaction():
                if context and context.get('debug')=='on':
                    # debug=on: every barrier names the files it covered.
                    self.event(context['run_id'],event_kind='retention',reason_code='durability_barrier',message='barrier',
                        details={**record,'file_ids':sorted(synchronized)})
                elif context and not unlinking:
                    # debug=off: one summary of this writer's barriers, written by the run-close barrier.
                    self.event(context['run_id'],event_kind='retention',reason_code='durability_barrier',message='barrier_summary',details=summary)
                else:
                    self.connection.execute('UPDATE ledger_info SET singleton=singleton WHERE singleton=1')
        finally:self.connection.execute('PRAGMA synchronous=NORMAL')
        self._durable_files.update(synchronized)

    def _durable_after_last_run_close(self):
        """Files the latest committed run-close barrier made durable (registered before it, still present, not an
        excluded copy). A later phase or resume then does not fsync them again. Records without the marker (older
        code) yield nothing: the cache stays empty, as before."""
        if not self.connection.compact:return set()
        table=self.connection.event_table
        for message,encoded in self.connection.execute(f"SELECT message,details_json FROM {table} WHERE reason_code='durability_barrier' "
                "AND event_kind='retention' ORDER BY event_id DESC LIMIT 64"):
            details=json.loads(encoded)
            if message=='barrier_summary':
                through,excluded=details.get('run_close_durable_through_event_id'),details.get('run_close_excluded_copy_file_ids') or []
            elif details.get('reason')=='run_close':
                through,excluded=details.get('durable_through_event_id'),details.get('excluded_copy_file_ids') or []
            else:
                continue
            if through is None:return set()
            excluded=set(excluded)
            return {file_id for file_id,row in self.file_catalog().items() if row['ownership']=='managed' and row['presence']=='present'
                and row['event_id'] is not None and row['event_id']<through and file_id not in excluded}
        return set()

    def record_uncommitted_write(self, run_id: str, path: Path, sha256: str, size: int, *, wrapper_stage: str,
                                 existed: bool | None = None, temporary: Any = None, lock: Any = None) -> None:
        """Bind, durably and BEFORE the replacement, the bytes a wrapper is about to put in place at a corpus path whose
        new content it has not committed yet (fix 7, R1: the "record the intent first" design). One entry of
        ``record_uncommitted_writes``; ``temporary``: the file holding those bytes until the replace (fix 8, N3);
        ``lock``: the canonical metadata lock the writer holds for the path (fix 9, F1)."""
        extra = {'temporary': str(Path(temporary).absolute())} if temporary is not None else {}
        if lock is not None:
            extra['lock'] = str(lock)
        self.record_uncommitted_writes(run_id, [(path, sha256, size, existed, extra)], wrapper_stage=wrapper_stage)

    def record_uncommitted_writes(self, run_id: str, entries: Any, *, wrapper_stage: str) -> None:
        """Bind, durably and BEFORE the replacements, the bytes about to be put in place at corpus paths whose new
        content is not committed yet (fix 7, R1: the "record the intent first" design).

        Callers: the metadata choke point (``parquet_schema._replace_in_place``: a legacy wrapper's in-place metadata
        write, committed only when its barrier seals the projection) and the product publication
        (``products._publish_products_locked``: mirrors replaced before the transaction that registers them). The
        commit is synchronous (``PRAGMA synchronous=FULL``, as the durability barrier): once it returns, the ledger names
        these bytes as this run's own uncommitted write, so a later start can tell them from a user's edit and roll them
        back (``verify_excluded_copies``) even if the failed barrier could not, for want of space. If the ledger cannot
        take the record (a full ledger or disk), this raises and the caller replaces nothing.
        ``entries``: ``(path, sha256, bytes[, existed[, extra]])``; the path is recorded as its physical file
        (``resolve_metadata_target``); ``existed`` says whether a file was at the path before this write (a rollback
        removes a file only if the failed attempt created it). ``extra`` (fix 8, N3): ``temporary``, the exact file the
        bytes sit in until the atomic replace, and for a publication copy its ``source`` (the immutable generation), so a
        process death between this record and the replace leaves a temporary the repeated barrier can verify and remove
        (``roll_back_uncommitted_writes``); nothing is ever removed for its name alone. ``extra['lock']`` (fix 9, F1):
        the canonical metadata lock (``parquet_schema.metadata_lock_path``) the writer holds while it makes this write,
        given by the writer that holds it (``held_metadata_lock``), never inferred from the name; recovery, rollback
        and temporary cleanup take exactly that lock (``_recover_locked``)."""
        from ..parquet_schema import resolve_metadata_target
        self._owner(transaction=False)
        if self.connection.in_transaction:
            raise LedgerError('An uncommitted write must be recorded in its own committed observation')
        entries = [(str(resolve_metadata_target(entry[0])), entry[1], entry[2], entry[3] if len(entry) > 3 else None,
                    dict(entry[4]) if len(entry) > 4 and entry[4] else {}) for entry in entries]
        if not entries:
            return
        self.connection.execute('PRAGMA synchronous=FULL')
        try:
            with self.transaction():
                for path, sha256, size, existed, extra in entries:
                    details = {'path': path, 'sha256': sha256, 'bytes': size, 'stage': wrapper_stage}
                    if existed is not None:
                        details['existed'] = bool(existed)
                    for key in ('temporary', 'source', 'lock'):
                        if extra.get(key):
                            details[key] = str(extra[key])
                    self.event(run_id, event_kind='run', reason_code='uncommitted_projection_write',
                        message='bytes put in place before their publication committed', details=details)
        finally:
            self.connection.execute('PRAGMA synchronous=NORMAL')

    def _own_uncommitted_write(self, row: dict[str, Any], path: Path) -> int | None:
        """The event id of a recorded uncommitted write (``record_uncommitted_write``) whose bytes are exactly the
        bytes now at ``path``, recorded after the copy's current registration; ``None`` otherwise. Only such bytes are
        the run's own: a user's edit (any other content) or a newer committed projection (a newer registration) never
        matches."""
        from ..parquet_schema import resolve_metadata_target
        from .encoding import file_hash
        try:
            physical = str(resolve_metadata_target(path))
        except Exception:
            return None
        candidates = self._uncommitted_records(row, physical)
        if not candidates or not path.is_file():
            return None
        current = file_hash(path)
        return next((event_id for event_id, details in candidates if details.get('sha256') == current), None)

    def _uncommitted_records(self, row: dict[str, Any], physical: str) -> list[tuple[int, dict[str, Any]]]:
        """Every recorded uncommitted write of ``physical`` after the copy's current registration, newest first: the
        records that apply to the copy (its own uncommitted bytes, and the locks their writers held)."""
        table = self.connection.event_table
        after = row.get('event_id')
        query = (f"SELECT event_id,details_json FROM {table} WHERE reason_code='uncommitted_projection_write'"
                 + (" AND event_id>?" if after is not None else "") + " ORDER BY event_id DESC")
        candidates = [(event_id, json.loads(encoded)) for event_id, encoded in
                      self.connection.execute(query, (after,) if after is not None else ())]
        return [(event_id, details) for event_id, details in candidates if details.get('path') == physical]

    def roll_back_uncommitted_writes(self, run_ids: Any, output_dir: Any, *, observation_run: str) -> list[str]:
        """Undo the uncommitted writes of failed publication attempts (fix 7, R1), before the publication is repeated.

        ``run_ids``: the wrapper runs of the failed attempts. Every path they recorded with ``record_uncommitted_writes``
        is examined; only a file whose bytes are exactly one of those records is touched (anything else, e.g. a user's
        edit, is left as it is):
        - if the path has a committed registration (its newest registration that is a managed copy of an intact
          immutable generation, not made by the failed attempts) with other bytes, those bytes are restored (temporary
          file and atomic replace: this needs the space the failed attempt lacked, so it raises if the disk is still
          full);
        - if it has none and the failed attempts' first write found no file there (they created it), or that
          registration was released by policy, the file is removed (no space needed); a file that existed before the
          failed attempts with no committed generation (a caller-seeded table) is left as it is (nothing to restore).
        A file already equal to its committed registration is left. Without this, a repeated publication that captures
        the files its callback changes (the OCR publication's caller kernel) sees its own earlier, never-committed bytes
        as unchanged and publishes nothing, and a later wrapper start restores the older committed copy over them.
        The ledger follows the file (events in ``observation_run``, observation ``uncommitted_write_rolled_back``): the
        restored registration is ``present`` again, the failed attempts' registrations of the removed bytes ``missing``.
        Fix 8 (N1): a path that holds one of the records is re-verified, its committed generation chosen again, and
        restored or removed inside one critical section of ``_recover_locked`` (for metadata, the canonical metadata
        lock every locked writer takes); a locked edit that commits after the first, unlocked look is seen there and
        kept. The installed bytes are verified before the replace (``_restore_copy``).
        Fix 8 (N3): a temporary a failed attempt bound in its record (``temporary``) and left behind (a process death
        between the record and the replace) is removed under the same lock, only if ``_abandoned_temporary`` verifies
        it. No writer can own it any more: this runs on the ledger owner's thread under the corpus flock, so the
        recording process has released the ledger (it died or its run ended) or is this thread, outside any write; a
        metadata temporary exists only while its writer holds the metadata lock, which is held here. The removed
        temporaries are listed in ``self.removed_temporaries``.
        Fix 9 (F1, F2): every step takes exactly the lock the records bind (``lock``: the canonical metadata lock their
        writer held), or the canonical metadata lock of the path when they bind none; it is never inferred from the
        file name, so a metadata file whose physical name is ``physical.bin``, ``physical.PARQUET`` or extensionless is
        recovered under the lock its writers take. A path whose records bind two different locks is not acted on. A
        temporary is verified in one pass over the source bytes (``_verified_temporary``) and removed only if its name
        still names the verified file.
        Returns the paths restored or removed."""
        from urllib.parse import unquote, urlsplit
        from .encoding import file_hash
        self.rollback_unresolved: list[dict[str, Any]] = []  # fix 10: fresh for every call, early returns included
        run_ids = [value for value in dict.fromkeys(run_ids) if value]
        if not run_ids:
            return []
        self._owner(transaction=False)
        table = self.connection.event_table
        recorded: dict[str, set[str]] = {}
        created: dict[str, bool] = {}  # path -> the failed attempts' first write found no file there
        temporaries: list[dict[str, Any]] = []  # records binding a temporary (fix 8, N3)
        locks: dict[str, list[tuple[str, Any]]] = {}  # path -> (sha256, metadata lock) of each record (fix 9, F1)
        self.removed_temporaries: list[str] = []
        # Fix 10 (N1, N3): every target recognised here but not resolved (conflicting or stale locks, a replaced path) is
        # listed, never skipped silently; the repeated barrier refuses the run on it (``stages.run.publish_barrier``).
        unresolved: list[dict[str, Any]] = self.rollback_unresolved
        rows_found = []
        for offset in range(0, len(run_ids), 500):
            chunk = run_ids[offset:offset+500]
            rows_found += self.connection.execute(
                    f"SELECT event_id,details_json FROM {table} WHERE reason_code='uncommitted_projection_write' AND run_id IN ("
                    + ','.join('?'*len(chunk)) + ')', chunk).fetchall()
        for _, encoded in sorted(rows_found, key=lambda row: row[0]):
            details = json.loads(encoded)
            if details['path'] not in recorded:
                created[details['path']] = details.get('existed') is False
            recorded.setdefault(details['path'], set()).add(details['sha256'])
            locks.setdefault(details['path'], []).append((details['sha256'], details.get('lock')))
            if details.get('temporary'):
                temporaries.append(details)
        if not recorded:
            return []
        root = Path(output_dir)
        resolved_root = Path(os.path.realpath(root))
        catalog = self.file_catalog()
        index = self._uri_index()
        local = lambda row: Path(unquote(urlsplit(row['uri']).path))
        failed_attempts = {row[0] for chunk in [run_ids[i:i+500] for i in range(0, len(run_ids), 500)]
                           for row in self.connection.execute('SELECT a.attempt_id FROM attempts a JOIN tasks t USING(task_id) WHERE t.run_id IN ('
                                                              + ','.join('?'*len(chunk)) + ')', chunk)}
        changed: list[str] = []
        events: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        def committed_of(rows):
            # The committed state of the path: its newest registration that is a managed copy of an intact immutable
            # generation (a product publication or a document mirror; whatever presence the failed wrapper's own
            # observation gave it afterwards), never a registration made by the failed attempts themselves.
            for row in sorted((row for row in rows if row['ownership'] == 'managed' and row['producer_attempt_id']
                               and row['producer_attempt_id'] not in failed_attempts),
                              key=lambda row: (row.get('created_at') or '', row['event_id'] or 0), reverse=True):
                siblings = (catalog.get(file_id) for file_id in self._files_of_attempts([row['producer_attempt_id']]))
                source = next((other for other in siblings if other is not None and other['file_id'] != row['file_id']
                               and other['presence'] == 'present' and other['ownership'] == 'managed'
                               and other['sha256'] == row['sha256'] and '/.glossapi/artifacts/' in other['uri']
                               and local(other).is_file() and file_hash(local(other)) == other['sha256']), None)
                if source is not None or row['presence'] == 'deleted':
                    return (row, source)
            return None

        def ours(current, committed, physical):
            # Exactly one of the failed attempts' recorded bytes, not the committed bytes, and something to go back to
            # (a committed generation, or no file before the failed attempts' first write).
            return current is not None and current in recorded[physical] \
                and not (committed is not None and committed[0]['sha256'] == current) \
                and (committed is not None or created[physical])

        def lock_of(physical: str, current: Any = False) -> tuple[str | None, bool]:
            # The lock the records of this path bind (``_agreed_lock``, shared with the run-start recovery). ``current``
            # given: only the records that apply to the bytes now at the path (fix 10, N3), so moving or deleting a file
            # whose records contradict each other removes the condition.
            return _agreed_lock(lock for digest, lock in locks.get(physical) or () if current is False or digest == current)

        for physical, digests in recorded.items():
            path = Path(physical)
            if path.is_symlink():
                # The recorded physical file was replaced by a symlink after its record (records are resolved when
                # written): whatever it names is not the recorded file, and following it is never safe (fix 10, N1).
                unresolved.append({'path': physical, 'reason': 'path_replaced'})
                continue
            current = file_hash(path) if path.is_file() else None
            lock, conflicting = lock_of(physical, current)
            if conflicting:
                # The bytes at the path are recorded under contradictory locks: neither the file nor its temporaries are
                # touched, and the run is refused (fix 10, N3: unresolved). Moving or deleting the file removes it.
                unresolved.append({'path': physical, 'reason': 'conflicting_locks',
                                   'locks': sorted({str(lock) for digest, lock in locks[physical] if lock and digest == current})})
                continue
            spellings = {path.as_uri()}
            if path.is_relative_to(resolved_root):
                spellings.add((root.absolute() / path.relative_to(resolved_root)).as_uri())
            rows = [catalog[file_id] for uri in spellings for file_id in index.get(uri, [])]
            committed = committed_of(rows)
            after = current
            if ours(current, committed, physical):
                # The look above holds no lock: a locked writer may commit between it and the act. Check and act are one
                # critical section of the canonical metadata lock (fix 8, N1): symlink, bytes and committed generation
                # are verified again under it, and only then is the file restored or removed.
                seen = {}

                def decide(target, path=path, rows=rows, physical=physical, seen=seen):
                    if Path(target) != path or path.is_symlink():
                        # The path no longer names the recorded physical file: act on nothing, and say so (fix 10).
                        raise UnresolvedRecoveryError([{'path': physical, 'reason': 'path_replaced'}])
                    seen['current'] = file_hash(path) if path.is_file() else None
                    seen['committed'] = committed_of(rows)
                    if not ours(seen['current'], seen['committed'], physical):
                        return None
                    if seen['committed'] is None or seen['committed'][0]['presence'] == 'deleted':
                        # Created by the failed attempt (no file before its first write), or released by policy:
                        # removing it needs no space. A file that existed before with no committed generation to
                        # restore (e.g. a caller-seeded metadata table the failed attempt rewrote) is left as it is,
                        # as before fix 7.
                        return ('remove', path)
                    row, source = seen['committed']
                    assert source is not None
                    return ('restore', local(source), row['sha256'])
                try:
                    action = _recover_locked(path, decide, lock=lock)
                except UnresolvedRecoveryError as refusal:
                    unresolved.extend(refusal.conditions)
                    continue  # nothing changed at this path; no ledger observation either
                if 'current' in seen:
                    current, committed = seen['current'], seen['committed']
                after = current
                if action is not None:
                    if action[0] == 'remove':
                        after = None
                    else:
                        after = file_hash(path)
                        if after != action[2]:
                            raise LedgerError('Rollback of an uncommitted write did not restore the committed bytes: ' + str(path))
                    changed.append(str(path))
            # The ledger follows the file: the committed registration is present again when the file holds it (here, or
            # already at the start, ``verify_excluded_copies``), and the failed attempts' own registrations of bytes the
            # file no longer holds are missing.
            committed_source = committed[1] if committed is not None else None
            if committed is not None and committed_source is not None and committed[0]['presence'] not in ('present', 'deleted') \
                    and after == committed[0]['sha256']:
                events.append((committed[0], 'present', {'observation': 'uncommitted_write_rolled_back',
                                                          'source_file_id': committed_source['file_id'], 'failed_wrapper_runs': run_ids}))
            for other in rows:
                if other['presence'] == 'present' and other['producer_attempt_id'] in failed_attempts \
                        and other['sha256'] in digests and other['sha256'] != after:
                    events.append((other, 'missing', {'observation': 'uncommitted_write_rolled_back', 'failed_wrapper_runs': run_ids}))
        for details in temporaries:
            temporary = Path(details['temporary'])
            if not os.path.lexists(temporary):
                continue
            lock, conflicting = lock_of(details['path'])
            if details.get('lock') and details['lock'] != lock or conflicting:
                # The path's records bind contradictory locks: the temporary is kept (never read as a product; FIX8-G1
                # residue). If the bytes at the path are among them, the loop above already refused the run.
                continue

            def verify_temporary(target, temporary=temporary, details=details):
                identity = _verified_temporary(temporary, details)
                return None if identity is None else ('remove', temporary, identity)
            # Under the lock its writer held (fix 9, F1: the recorded lock; the canonical lock of the path for a record
            # without one): a metadata temporary exists only while its writer holds that lock. The bytes are verified
            # in one pass (fix 9, F2) and the name is removed only if it still names the verified file.
            try:
                if _recover_locked(Path(details['path']), verify_temporary, lock=lock):
                    self.removed_temporaries.append(str(temporary))
            except UnresolvedRecoveryError as refusal:
                unresolved.extend(refusal.conditions)  # the temporary is kept (fix 10)
        unresolved[:] = _unique_conditions(unresolved)
        if events:
            with self.transaction():
                for row, presence, details in events:
                    self.event(observation_run, event_kind='artifact', reason_code=presence, file_id=row['file_id'],
                        version_id=row['version_id'], details=details)
        return changed

    def refuse_ambiguous_copies(self) -> None:
        """Run start, after ``verify_excluded_copies``: an unresolved post-close edit stops the run before any plan
        executes or publishes (CP-K4ac2 R2). The barrier record still names the copy, so every later start refuses
        again until the caller resolves it; a refused start never re-records the copy as durable."""
        if getattr(self, 'unresolved_recoveries', None):
            raise UnresolvedRecoveryError(self.unresolved_recoveries)  # fix 10 (N1): recovery is fail-closed
        if getattr(self, 'ambiguous_copies', None):
            raise PostCloseEditError(self.ambiguous_copies)

    def _left_by_power_loss(self, path: Path, source: Path, row: dict[str, Any]) -> bool:
        """Could a power loss after the run close have left these copy bytes (K4F1-G1, by content, never by time)?

        Two images qualify. (1) A torn image of the managed bytes: no longer than them, and every 512-byte sector
        either equals the managed bytes at that offset or is all zeros (blocks never written back read as zeros;
        a truncation leaves a prefix). (2) Bytes this ledger registered earlier at the same path: the copy is
        published by a temporary file and an atomic replace, so an undurable rename can leave the previous
        registered copy. Anything else is complete content nobody but a user wrote, and is never overwritten.
        The bytes are read only on this rare path (a named copy that no longer matches its registration)."""
        import stat as _stat
        try:
            info=os.lstat(path)
        except FileNotFoundError:
            return True
        if not _stat.S_ISREG(info.st_mode):
            return False
        if info.st_size<=os.stat(source).st_size:
            sector=512;zero=bytes(sector);torn=True
            with open(path,'rb') as copy, open(source,'rb') as managed:
                while torn:
                    block=copy.read(1<<20)
                    if not block:break
                    expected=managed.read(len(block))
                    if block==expected:continue
                    for offset in range(0,len(block),sector):
                        piece=block[offset:offset+sector]
                        if piece!=expected[offset:offset+sector] and piece!=zero[:len(piece)]:
                            torn=False;break
            if torn:
                return True
        earlier={self.file_catalog()[other]['sha256'] for other in self._uri_index().get(row['uri'],[])
            if other!=row['file_id'] and self.file_catalog()[other]['ownership']=='managed'}-{row['sha256']}
        return bool(earlier) and file_hash(path) in earlier

    def _excluded_copy_verdict(self,row,path,before,catalog,local):
        """Classify one copy the last run close left unflushed (``verify_excluded_copies``): ``('intact',)``,
        ``('skip',)`` (no intact source: left for verification to report), ``('ambiguous', details)`` (a user's edit,
        kept and refused) or ``('restore', own_event_id_or_None, source_row)``. Pure check, no write: the caller runs it
        once without a lock and again inside ``_recover_locked`` before it restores (fix 8, N1)."""
        from .encoding import file_hash
        intact=path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and file_hash(path)==row['sha256']
        # Only bytes the barrier left behind are restored (CP-K4a F6). Timestamps can only prove a change: a copy
        # whose mtime or ctime is later than recorded at the barrier was written after the close and is never
        # overwritten; without a recorded identity (an older ledger) the repair is ambiguous and refused. Equal
        # timestamps prove nothing (a write in the barrier's own clock tick keeps them; K4F1-G1), so such a copy is
        # classified by its bytes: only a torn image of the managed bytes, or bytes this ledger registered earlier
        # at the same path (a replace the power loss rolled back), is restored. Any other complete content is a
        # user's edit. Both the copy and its managed source are kept; verification still reports the copy.
        if intact:
            return ('intact',)
        try:info: os.stat_result | None=path.lstat()
        except FileNotFoundError:info=None
        reason=None
        # Bytes a failed wrapper put in place before its barrier committed, recorded as its own before the write
        # (``record_uncommitted_write``; fix 7, R1). They are rolled back to the committed source here, before any
        # post-close-edit classification: the failed barrier's own rollback needs free space and may have failed
        # on the disk that stopped the run. Only an exact match of such a record qualifies; anything else keeps
        # the rules below (a user's edit is kept and refused).
        own=self._own_uncommitted_write(row,path) if info is not None else None
        if own is None and info is not None and (not before or info.st_mtime_ns>before[0] or info.st_ctime_ns>before[1]):
            reason='changed_after_close'
        source=None if reason else next((other for other in catalog.values() if other['file_id']!=row['file_id'] and other['presence']=='present'
            and other['ownership']=='managed' and other['producer_attempt_id']==row['producer_attempt_id']
            and other['sha256']==row['sha256'] and '/.glossapi/artifacts/' in other['uri']
            and local(other).is_file() and file_hash(local(other))==other['sha256']),None)
        if own is None and source is not None and info is not None and not self._left_by_power_loss(path,local(source),row):
            reason='complete_different_content'
        if reason:
            assert info is not None
            return ('ambiguous',dict(recorded_identity=before,current_identity=[info.st_mtime_ns,info.st_ctime_ns],reason=reason))
        if source is None:
            return ('skip',)
        return ('restore',own,source)

    def verify_excluded_copies(self):
        """Run start: verify the byte copies the last run-close barrier left unflushed (K4C-M2).

        Decision 3 skips the fsync of a copy whose managed source is durable, so
        a power loss after run close can tear the copy while its source is
        intact. Each copy named by the latest run-close barrier record is
        checked (size and sha256). A torn or missing copy whose source still
        matches its registration is republished from that source (a changed copy
        counts as torn only by its bytes, ``_left_by_power_loss``; a later
        timestamp or complete different content is a user's edit, kept and
        refused by ``refuse_ambiguous_copies``); every checked
        copy is then made durable, so the window closes at the next run start.
        A copy without an intact source is left for verification to report
        ("Committed publication is missing or corrupt"). A copy whose bytes are
        exactly a recorded uncommitted write of a failed wrapper
        (``record_uncommitted_writes``, recorded after the copy's registration)
        is the run's own and is rolled back to the managed source before any of
        these classifications (fix 7, R1). Fix 8 (N1): a copy to be restored is classified again inside
        ``_recover_locked`` (for metadata, the canonical metadata lock) and restored in the same critical section, with
        the installed bytes verified first; a locked edit committed after the first look is a user's edit there.
        Fix 10 (N1, N3): recovery is fail-closed. A copy recognised here that cannot be resolved (a hard-linked copy,
        records of its uncommitted bytes binding contradictory locks (``_agreed_lock``, shared with the rollback) or a
        lock its writers no longer take, a replaced path, a managed source that changed during the restore or is not
        intact, a copy changed right after its restore) is kept, not made durable, and listed in
        ``self.unresolved_recoveries``; ``refuse_ambiguous_copies`` then refuses the run (``UnresolvedRecoveryError``)
        before any plan executes. "Left for verification" is no longer an outcome.
        Returns the file_ids republished.
        """
        self._owner(transaction=False)
        self.unresolved_recoveries: list[dict[str, Any]] = []
        table=self.connection.event_table
        latest=self.connection.execute(f"SELECT run_id,details_json,recorded_at FROM {table} WHERE reason_code='durability_barrier' AND event_kind='retention' "
            "AND (message='barrier_summary' OR message='barrier') ORDER BY event_id DESC LIMIT 1").fetchone()
        if latest is None:return []
        barrier_run=latest[0];details: dict[str, Any]=json.loads(latest[1])
        identifiers: list[str] = details.get('run_close_excluded_copy_file_ids') or details.get('excluded_copy_file_ids') or []
        if not identifiers:return []
        recorded: list[Any] = details.get('run_close_excluded_copy_identities') or details.get('excluded_copy_identities') or [None]*len(identifiers)
        identity_of: dict[str, Any] = dict(zip(identifiers,recorded))
        self.ambiguous_copies: list[dict[str, Any]] = []
        from urllib.parse import unquote, urlsplit
        from .encoding import file_hash
        local=lambda row:Path(unquote(urlsplit(row['uri']).path))
        catalog=self.file_catalog()
        restored=[];checked=[];durable=[]
        for file_id in identifiers:
            row=catalog.get(file_id)
            if row is None or row['presence']!='present':continue
            path=local(row)
            before=identity_of.get(file_id)
            verdict=self._excluded_copy_verdict(row,path,before,catalog,local)
            unresolved=lambda reason,**extra:self.unresolved_recoveries.append(dict(file_id=file_id,path=str(path),reason=reason,**extra))
            if verdict[0]=='restore':
                # The classification above holds no lock: a locked writer may commit between it and the restore. The
                # copy is classified again and restored inside one critical section of the canonical metadata lock
                # (fix 8, N1); an edit committed meanwhile is then a user's edit, kept and refused below.
                from ..parquet_schema import MetadataAliasError, MetadataStaleGenerationError
                locked: dict[str, Any] = {'verdict':('skip',)}
                def decide(target,row=row,path=path,before=before,locked=locked):
                    locked['verdict']=self._excluded_copy_verdict(row,path,before,catalog,local)
                    return ('restore',local(locked['verdict'][2]),row['sha256']) if locked['verdict'][0]=='restore' else None
                # The lock the records of the copy's uncommitted bytes agree on (fix 9, F1; fix 10, N3: every record after
                # the copy's registration that binds the bytes now at the path, as the rollback's ``_agreed_lock``); none
                # bound: the canonical metadata lock of the path. Contradictory records: kept, unresolved (moving or
                # deleting the copy removes the condition: it is then republished from its source).
                from ..parquet_schema import resolve_metadata_target
                current=file_hash(path) if path.is_file() and not path.is_symlink() else None
                lock,conflicting=_agreed_lock(details.get('lock') for _,details in self._uncommitted_records(row,str(resolve_metadata_target(path)))
                                              if details.get('sha256')==current)
                if conflicting:
                    unresolved('conflicting_locks');continue
                try:
                    action=_recover_locked(path,decide,lock=lock)
                except UnresolvedRecoveryError as refusal:
                    for entry in refusal.conditions:unresolved(entry['reason'])
                    continue
                except MetadataAliasError:
                    unresolved('hard_linked');continue  # the metadata lock refuses a hard-linked copy: kept (fix 10, N1)
                except (LedgerError,MetadataStaleGenerationError):
                    unresolved('source_changed');continue  # the installed bytes would not be the registered ones: kept
                verdict=locked['verdict']
                if action is not None:
                    _,own,source=verdict
                    if file_hash(path)!=row['sha256']:
                        unresolved('changed_after_restore');continue
                    restored.append((row,source,own))
                    checked.append(path);durable.append(file_id)
                    continue
            if verdict[0]=='intact':
                checked.append(path);durable.append(file_id)
            elif verdict[0]=='ambiguous':
                self.ambiguous_copies.append(dict(file_id=file_id,path=str(path),**verdict[1]))
            elif verdict[0]=='skip':
                unresolved('no_intact_source')  # torn/missing copy, no intact managed source to republish (fix 10)
        _map_bounded(_fsync_path,checked)
        for directory in sorted({path.parent for path in checked},key=lambda path:len(path.parts),reverse=True):
            descriptor=os.open(directory,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        # Durable now: this writer's own run-close barrier need not name them again for the next start.
        self._durable_files.update(durable)
        if restored:
            with self.transaction():
                for row,source,own in restored:
                    details={'observation':'unflushed_copy_republished_from_source','source_file_id':source['file_id']}
                    if own is not None:
                        details={'observation':'uncommitted_write_rolled_back','source_file_id':source['file_id'],
                            'uncommitted_write_event_id':own}
                    self.event(barrier_run,event_kind='artifact',reason_code='present',file_id=row['file_id'],version_id=row['version_id'],
                        details=details)
        return [row['file_id'] for row,_,_ in restored]

    def _barrier_totals(self,record: dict[str, Any]) -> dict[str, Any]:
        totals=self.barrier_totals
        totals['barrier_count']+=1
        for key in ('file_count','byte_count','directory_count','unlink_count','flush_ns'):totals[key]+=record[key]
        by=totals['by_reason'].setdefault(record['reason'],dict(barrier_count=0,file_count=0,byte_count=0))
        by['barrier_count']+=1;by['file_count']+=record['file_count'];by['byte_count']+=record['byte_count']
        return {key:(dict(cast(dict[str, Any], value)) if isinstance(value,dict) else value) for key,value in totals.items()}

    def _files_of_attempts(self,attempts):
        """file_ids produced by these attempts (indexed lookup; the catalog supplies current presence)."""
        attempts=[value for value in attempts if value]
        found=[]
        for offset in range(0,len(attempts),500):
            chunk=attempts[offset:offset+500]
            found.extend(value for value, in self.connection.execute(
                'SELECT file_id FROM files WHERE producer_attempt_id IN ('+','.join('?'*len(chunk))+')',chunk))
        return found

    def _barrier_dependents(self,catalog,unlinking,private):
        """Files a pending unlink depends on (see durable_barrier)."""
        bindings={}
        for file_id in unlinking:
            row=catalog.get(file_id)
            if row and row['producer_attempt_id']:bindings.setdefault((row['producer_attempt_id'],row['sha256']),set()).add(row['uri'])
        readers=set()
        if not private:
            identifiers=list(unlinking)
            for offset in range(0,len(identifiers),500):
                chunk=identifiers[offset:offset+500]
                readers.update(value for value, in self.connection.execute(
                    'SELECT DISTINCT attempt_id FROM attempt_inputs WHERE file_id IN ('+','.join('?'*len(chunk))+')',chunk))
        selected=[]
        for file_id in dict.fromkeys(self._files_of_attempts(sorted({producer for producer,_ in bindings}|readers))):
            row=catalog.get(file_id)
            if (row is None or row['ownership']!='managed' or row['presence']!='present' or file_id in self._durable_files
                    or file_id in unlinking):continue
            # A user-visible byte copy (outside the private tree) of a file being unlinked.
            copy=(row['producer_attempt_id'],row['sha256']) in bindings and '/.glossapi/' not in row['uri'] and row['uri'] not in bindings[(row['producer_attempt_id'],row['sha256'])]
            if copy or row['producer_attempt_id'] in readers:selected.append(row)
        return selected

    def _without_durable_copies(self,catalog,selected,unlinking):
        """Decision 3: a byte copy of a retained managed source needs no flush of its own once that source is durable."""
        artifacts='/.glossapi/artifacts/'
        producers={row['producer_attempt_id'] for row in selected if artifacts not in row['uri'] and row['producer_attempt_id']}
        if not producers:return selected
        chosen={row['file_id'] for row in selected}
        sources={}
        for file_id in self._files_of_attempts(sorted(producers)):
            row=catalog.get(file_id)
            if (row is not None and row['ownership']=='managed' and row['presence']=='present' and artifacts in row['uri']
                    and file_id not in unlinking
                    and (row['retention_class']!='intermediate' or row['kind'] in _RETAINED_PRIVATE_KINDS or self.retained_intermediates)
                    and (file_id in self._durable_files or file_id in chosen)):
                sources.setdefault((row['producer_attempt_id'],row['sha256']),[]).append(row['uri'])
        if not sources:return selected
        return [row for row in selected if not (artifacts not in row['uri']
                and any(uri!=row['uri'] for uri in sources.get((row['producer_attempt_id'],row['sha256']),())))]

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def migrate(self, version: int, sql: str, *, checksum: str, code_revision: str, backup_path: Any) -> None:
        """Apply a numbered checksum-bound future migration; rollback on failure."""
        self._owner(transaction=False)
        if self.connection.in_transaction: raise LedgerError("Migration requires no active observation")
        current = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version != current + 1 or hashlib.sha256(sql.encode()).hexdigest() != checksum:
            raise LedgerError("Migration sequence/checksum mismatch")
        integrity(self.connection)
        if Path(backup_path).exists():
            # The live ledger is never its own backup (CP-K4bc2 R4): the same path, a symlink to it or a hard link all
            # resolve to the ledger's (dev, ino); reusing it would upgrade the only copy.
            live, given = os.stat(self.path), os.stat(backup_path)
            if (live.st_dev, live.st_ino) == (given.st_dev, given.st_ino):
                raise LedgerError("Backup path names the live ledger itself; choose an independent backup path")
            # A retry after a failed migration (CP-K4b F6): reuse the earlier backup only when it verifies and holds
            # exactly this (still unmigrated) ledger; any other file at that path is kept and refused.
            existing = None
            try:
                existing = connect(backup_path)
                integrity(existing)
                matches = (existing.execute("PRAGMA user_version").fetchone()[0] == current
                           and _content_digest(existing) == _content_digest(self.connection))
            except (sqlite3.DatabaseError, LedgerError, ValueError):
                matches = False
            finally:
                if existing is not None: existing.close()
            if not matches:
                raise LedgerError("Backup already exists and is not a verified backup of this ledger; choose another backup path")
        else:
            backup = connect(backup_path)
            try:
                self.connection.backup(backup)
                integrity(backup)
            finally: backup.close()
        with self.transaction():
            for statement in statements(sql):
                if re.search(r"\b(BEGIN|COMMIT|ROLLBACK|PRAGMA|ATTACH|DETACH|VACUUM)\b", statement, re.I):
                    raise LedgerError("Migration may not control its own transaction")
                self.connection.execute(statement)
            self.connection.execute("INSERT INTO schema_migrations VALUES(?,?,?,?)", (version, utc_now(), checksum, code_revision))
            integrity(self.connection)
            self.connection.execute(f"PRAGMA user_version={version}")

    def file_catalog(self) -> Any:
        """Version-bound file facts, rebuilt only at open/resume or rollback.

        This is control state, not filesystem byte verification. Publication,
        source deletion and audit still perform their required content checks.
        """
        self._owner(transaction=False)
        if self._file_catalog is None:
            self._file_catalog={row['file_id']:dict(row) for row in self.connection.execute('SELECT * FROM current_file_state')}
            self._catalog_uris=None
        return self._file_catalog

    def _uri_index(self) -> dict[str, list[str]]:
        """URI -> file_ids of the current catalog in registration order (dropped with the catalog on rollback)."""
        catalog=self.file_catalog()
        index: dict[str, list[str]] | None = getattr(self,'_catalog_uris',None)
        if index is None:
            index={}
            for file_id,row in catalog.items():index.setdefault(row['uri'],[]).append(file_id)
            self._catalog_uris=index
        return index

    def _file_presence(self, file_id, presence, event_id):
        row=self.file_catalog()[file_id]
        row.update(presence=presence,event_id=event_id)

    def control(self):
        """Reconstruct once at start/resume, then advance with appended facts.

        A rolled-back observation discards this cache before another write.
        Workers never own this state or use it as persistent authority.
        """
        if self.control_cache is None:
            tasks={row['task_id']:dict(row) for row in self.connection.execute('SELECT * FROM tasks')}
            attempts={row['attempt_id']:dict(row) for row in self.connection.execute('SELECT * FROM attempts')}
            latest={};numbers={};status={};reasons={};task_events={};attempt_events={}
            for identifier,row in attempts.items():
                task=row['task_id'];number=row['attempt_number']
                if number>numbers.get(task,0):numbers[task]=number;latest[task]=identifier
            for event_id,task_id,identifier,value,reason,message in self.connection.execute(f"SELECT event_id,task_id,attempt_id,status,reason_code,message FROM {self.connection.event_table} WHERE event_kind='lifecycle' ORDER BY event_id"):
                event=dict(event_id=event_id,status=value,reason_code=reason,message=message)
                task_events[task_id]=event
                if identifier:
                    status[identifier]=value;reasons[identifier]=reason;attempt_events[identifier]=event
            self.control_cache=dict(tasks=tasks,attempts=attempts,latest=latest,numbers=numbers,status=status,reasons=reasons,task_events=task_events,attempt_events=attempt_events)
        return self.control_cache

    def task_state(self, task_id):
        control=self.control();task=control['tasks'].get(task_id)
        if task is None:return None
        return {**task,**control['task_events'].get(task_id,dict(status=None,reason_code=None,event_id=None,message=''))}

    def task_states(self, run_id):
        return [self.task_state(identifier) for identifier,task in self.control()['tasks'].items() if task['run_id']==run_id]

    def latest_attempt(self, task_id):
        control=self.control();identifier=control['latest'].get(task_id)
        if identifier is None:return None
        return {**control['attempts'][identifier],**control['attempt_events'].get(identifier,dict(status=None,reason_code=None,message=''))}

    def _record_lifecycle(self, task_id, attempt_id, status, reason_code, message, event_id):
        control=self.control();event=dict(status=status,reason_code=reason_code,message=message,event_id=event_id)
        control['task_events'][task_id]=event
        if attempt_id:
            control['status'][attempt_id]=status;control['reasons'][attempt_id]=reason_code
            control['attempt_events'][attempt_id]=event

    def planned_readers(self, run_id, file):
        """Frozen output-kind and explicit file readers, never late guesses."""
        if run_id not in self._planned_reader_maps:
            plan=self._run_plans.get(run_id)
            if plan is None:
                stored=self.connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.plan_sha256 WHERE r.run_id=?',(run_id,)).fetchone()
                if stored is None:raise LedgerError('Unknown reader plan')
                plan=json.loads(stored[0]);self._run_plans[run_id]=plan
            tasks={row['ordinal']:identifier for identifier,row in self.control()['tasks'].items() if row['run_id']==run_id}
            operations=plan.get('operations',())
            keys={operation.get('key'):tasks[index] for index,operation in enumerate(operations) if index in tasks}
            outputs={};files={}
            for index,operation in enumerate(operations):
                if index not in tasks:continue
                reader=tasks[index]
                for parent,kind,_ in operation.get('inputs',()):
                    if parent not in keys:raise LedgerError('Frozen reader producer is missing')
                    outputs.setdefault((keys[parent],kind),set()).add(reader)
                for artifact in operation.get('document',{}).get('artifacts',()):
                    files.setdefault(artifact['file_id'],set()).add(reader)
            self._planned_reader_maps[run_id]=(outputs,files)
        outputs,files=self._planned_reader_maps[run_id]
        producer=self.control()['attempts'].get(file['producer_attempt_id'])
        return (outputs.get((producer['task_id'],file['kind']),set()) if producer else set()) | files.get(file['file_id'],set())

    def reader_state(self, task_id):
        state=self.task_state(task_id)
        return (state['status'],state['reason_code']) if state else (None,None)

    def accepts_result(self, attempt_id: str, lease_token: str) -> bool:
        state=self.control()
        attempt: Any=state['attempts'].get(attempt_id)
        status: Any=state['status'].get(attempt_id)
        return bool(attempt and attempt['lease_token']==lease_token and status not in TERMINAL_STATUSES)

    def snapshot(self, kind, value):
        self._owner()
        value = evidence(value)
        validate_evidence(value)
        encoded = canonical_json(value)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        if digest in self.snapshot_cache:
            return digest
        if getattr(self.connection,'storage_version',0)>=3:
            stored=self.connection.encode_snapshot(value,lambda common:self.snapshot('shared_config',{'kind':'shared_config','value':common}))
        else:
            # The v2 frame only; v3 never stores it, so it is not computed there.
            stored=encode_json(encoded) if self.connection.compact else encoded
        # The snapshot key hashes the full logical value, independent of storage.
        self.connection.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,1,?)",(digest,kind,stored))
        self.snapshot_cache.add(digest)
        return digest

    def run(self, settings, plan, environment, *, code_revision):
        self._owner()
        run_id = new_id()
        hashes = [self.snapshot(kind, {"kind": kind, **value}) for kind, value in
                  (("settings", settings), ("plan", plan), ("environment", environment))]
        self.connection.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, utc_now(), *hashes, code_revision))
        self._run_plans[run_id]=json.loads(canonical_json(evidence({"kind":"plan",**plan})))
        return run_id

    def document(self, source_key):
        self._owner()
        row = self.connection.execute("SELECT document_id FROM documents WHERE source_key=?", (source_key,)).fetchone()
        if row: return row[0]
        identifier = new_id()
        self.connection.execute("INSERT INTO documents VALUES(?,?,?)", (identifier, source_key, utc_now()))
        return identifier

    def version(self, document_id, *, source_sha256=None, stored_sha256=None):
        self._owner(); validate_id(document_id)
        validate_hash(source_sha256, nullable=True); validate_hash(stored_sha256, nullable=True)
        previous = self.connection.execute("SELECT * FROM document_versions WHERE document_id=? ORDER BY revision DESC LIMIT 1", (document_id,)).fetchone()
        if previous and (previous["source_sha256"], previous["stored_sha256"]) == (source_sha256, stored_sha256):
            return previous["version_id"]
        if previous and previous["source_sha256"] is None and previous["stored_sha256"] is None:
            self.fill_version_hashes(previous["version_id"], source_sha256=source_sha256, stored_sha256=stored_sha256)
            return previous["version_id"]
        identifier = new_id()
        self.connection.execute("INSERT INTO document_versions VALUES(?,?,?,?,?,?)", (identifier, document_id, previous["revision"] + 1 if previous else 1, source_sha256, stored_sha256, utc_now()))
        return identifier

    def fill_version_hashes(self, version_id, *, source_sha256, stored_sha256):
        self._owner(); validate_id(version_id)
        validate_hash(source_sha256, nullable=True); validate_hash(stored_sha256, nullable=True)
        self.connection.execute("UPDATE document_versions SET source_sha256=?,stored_sha256=? WHERE version_id=?", (source_sha256, stored_sha256, version_id))

    def source_record(self, document_id, run_id, source_uri, metadata, *, source_kind="local_path", manifest_sha256=None, source_row=None, url_index=None, source_record_id=None):
        self._owner(); validate_id(document_id); validate_id(run_id)
        validate_hash(manifest_sha256, nullable=True)
        if source_row is not None and not isinstance(source_row, str): raise LedgerError("source_row must be text")
        if url_index is not None and (type(url_index) is not int or url_index < 0): raise LedgerError("url_index must be a nonnegative integer")
        encoded = canonical_json({"schema_version": 1, "fields": typed_value(metadata)})
        identifier = source_record_id or snapshot_hash([document_id, run_id, source_uri, source_kind, manifest_sha256, source_row, url_index, encoded])
        validate_hash(identifier)
        self.connection.execute("INSERT INTO source_records VALUES(?,?,?,?,?,?,?,?,?)", (identifier, document_id, run_id, source_uri, source_kind, manifest_sha256, source_row, url_index, encoded))
        return identifier

    def task(self, run_id, stage, operation, specification, *, version_id=None, scope_key=None, ordinal=0):
        self._owner(); validate_id(run_id)
        if version_id is not None: validate_id(version_id)
        if stage not in STAGES: raise LedgerError("Unknown stage")
        identifier = new_id()
        specification_hash = self.snapshot("specification", {"kind": "specification", **specification})
        if version_id is None:
            if not isinstance(scope_key, str) or not scope_key.startswith("corpus:"):
                raise LedgerError("Aggregate task requires a corpus manifest scope")
            validate_hash(scope_key.removeprefix("corpus:"))
            if self.connection.execute("SELECT 1 FROM snapshots WHERE sha256=?", (scope_key.removeprefix("corpus:"),)).fetchone() is None:
                raise LedgerError("Aggregate task manifest is not registered")
        elif scope_key is not None and scope_key != version_id:
            raise LedgerError("Document task scope must equal its version ID")
        control=self.control()
        self.connection.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?)", (identifier, run_id, version_id, scope_key or version_id, stage, operation, ordinal, specification_hash))
        control['tasks'][identifier]=dict(task_id=identifier,run_id=run_id,version_id=version_id,stage=stage,
            operation=operation,scope_key=scope_key or version_id,ordinal=ordinal,specification_sha256=specification_hash)
        return identifier

    def dependency(self, task_id, producer_task_id, input_role):
        self._owner(); validate_id(task_id); validate_id(producer_task_id)
        if self.connection.execute("""WITH RECURSIVE parents(id) AS (
          SELECT producer_task_id FROM task_dependencies WHERE task_id=? UNION
          SELECT d.producer_task_id FROM task_dependencies d JOIN parents p ON d.task_id=p.id)
          SELECT 1 FROM parents WHERE id=?""", (producer_task_id, task_id)).fetchone():
            raise LedgerError("Cyclic dependency")
        self.connection.execute("INSERT INTO task_dependencies VALUES(?,?,?)", (task_id, producer_task_id, input_role))

    def attempt(self, task_id, *, worker_id, device_id, fingerprint, inputs, settings, tools):
        self._owner(); validate_id(task_id)
        control=self.control()
        previous=control['latest'].get(task_id)
        if previous and control['status'].get(previous) not in TERMINAL_STATUSES:
            raise LedgerError("Task already has an active attempt")
        number=control['numbers'].get(task_id,0)+1
        identifier = new_id()
        hashes = [self.snapshot(kind, {"kind": kind, **value}) for kind, value in
                  (("fingerprint", fingerprint), ("inputs", inputs), ("settings", settings), ("tools", tools))]
        lease=new_id()
        created_at=utc_now()
        self.connection.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?)", (identifier, task_id, number, worker_id, device_id, lease, *hashes, created_at))
        control['attempts'][identifier]=dict(attempt_id=identifier,task_id=task_id,attempt_number=number,
            worker_id=worker_id,device_id=device_id,lease_token=lease,fingerprint_sha256=hashes[0],
            input_manifest_sha256=hashes[1],settings_sha256=hashes[2],tools_sha256=hashes[3],created_at=created_at)
        control['latest'][task_id]=identifier
        control['numbers'][task_id]=number
        return identifier

    def input(self, attempt_id, file_id, role, sha256):
        self._owner(); validate_id(attempt_id); validate_id(file_id); validate_hash(sha256)
        row = self.connection.execute("SELECT sha256 FROM files WHERE file_id=?", (file_id,)).fetchone()
        if row is None or row[0] != sha256: raise LedgerError("Input hash differs from registered file")
        self.connection.execute("INSERT INTO attempt_inputs VALUES(?,?,?,?)", (attempt_id, file_id, role, sha256))

    def event(self, run_id, *, event_kind, reason_code, message="", status=None, task_id=None, version_id=None, attempt_id=None, file_id=None, error_type=None, traceback=None, started_at=None, finished_at=None, duration_ns=None, output_manifest_sha256=None, details=None):
        self._owner(); validate_id(run_id)
        for identifier in (task_id, version_id, attempt_id, file_id):
            if identifier is not None: validate_id(identifier)
        if event_kind not in EVENT_KINDS or reason_code not in REASON_CODES or (status is not None and status not in STATUSES):
            raise LedgerError("Unknown event kind, reason or status")
        validate_hash(output_manifest_sha256, nullable=True)
        validate_time(started_at); validate_time(finished_at)
        if attempt_id:
            attempt_task = self.control()['attempts'].get(attempt_id)
            if attempt_task is None or task_id != attempt_task['task_id']: raise LedgerError("Attempt/task mismatch")
        if task_id:
            task = self.control()['tasks'].get(task_id)
            if task is None or task['run_id'] != run_id or task['version_id'] != version_id: raise LedgerError("Task/run/version mismatch")
        if file_id:
            # The owner's catalog holds every registered file's immutable binding (rebuilt after any rollback).
            entry = self._file_catalog.get(file_id) if self._file_catalog is not None else None
            artifact = (entry['version_id'], entry['producer_attempt_id']) if entry is not None else self.connection.execute(
                "SELECT version_id,producer_attempt_id FROM files WHERE file_id=?", (file_id,)).fetchone()
            if artifact is None or artifact[0] != version_id: raise LedgerError("File/version mismatch")
            if attempt_id is not None and artifact[1] is not None and artifact[1] != attempt_id:
                raise LedgerError("File/attempt mismatch")
        if event_kind == "artifact" and file_id is None:
            raise LedgerError("Artifact event requires a file")
        if event_kind == "artifact" and reason_code not in ("present", "deleted", "missing", "corrupt"):
            raise LedgerError("Invalid artifact presence")
        if event_kind == "lifecycle":
            if not task_id or not status: raise LedgerError("Lifecycle requires task and status")
            if status in ("running", *TERMINAL_STATUSES) and not attempt_id:
                raise LedgerError("Running/terminal lifecycle requires an attempt")
            if attempt_id:
                last = self.control()['status'].get(attempt_id)
                if last in TERMINAL_STATUSES: raise LedgerError("Attempt is already terminal")
                if status == "planned" and last: raise LedgerError("Cannot regress an attempt to planned")
        payload = evidence(details or {})
        validate_evidence(payload)
        text = canonical_json(payload)
        if duration_ns is not None and (type(duration_ns) is not int or duration_ns < 0):
            raise LedgerError("Duration must be nonnegative integer nanoseconds")
        encoded=encode_json(text) if self.connection.compact else text
        if self.connection.compact:
            identifier=self.next_event_id
            self.next_event_id+=1
            if event_kind=='retention' and file_id and all(value is None for value in (
                    task_id,attempt_id,status,error_type,traceback,started_at,finished_at,duration_ns,output_manifest_sha256)):
                # All omitted columns are exactly implied by the file binding
                # and event kind. The public events view reconstructs them.
                self.connection.execute('INSERT INTO retention_events VALUES(?,?,?,?,?,?,?)',
                    (identifier,run_id,file_id,reason_code,message,encoded,utc_now()))
            else:
                self.connection.execute('INSERT INTO event_records VALUES('+','.join('?' for _ in range(18))+')',
                    (identifier,run_id,task_id,version_id,attempt_id,file_id,event_kind,status,reason_code,message,
                     error_type,traceback,started_at,finished_at,duration_ns,output_manifest_sha256,encoded,utc_now()))
            if event_kind=='lifecycle':self._record_lifecycle(task_id,attempt_id,status,reason_code,message,identifier)
            if event_kind=='artifact':self._file_presence(file_id,reason_code,identifier)
            return identifier
        cursor = self.connection.execute("""INSERT INTO events(run_id,task_id,version_id,attempt_id,file_id,event_kind,status,reason_code,message,error_type,traceback,started_at,finished_at,duration_ns,output_manifest_sha256,details_json,recorded_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (run_id, task_id, version_id, attempt_id, file_id, event_kind, status, reason_code, message, error_type, traceback, started_at, finished_at, duration_ns, output_manifest_sha256, encoded, utc_now()))
        if event_kind=='lifecycle':self._record_lifecycle(task_id,attempt_id,status,reason_code,message,cursor.lastrowid)
        if event_kind=='artifact':self._file_presence(file_id,reason_code,cursor.lastrowid)
        return cursor.lastrowid

    def artifact(self, run_id, stage, kind, uri, *, sha256=None, bytes=None, version_id=None, producer_attempt_id=None, ownership="managed", retention_class="intermediate", presence="present", page_first=None, page_last=None):
        self._owner(); validate_id(run_id)
        for identifier in (version_id, producer_attempt_id):
            if identifier: validate_id(identifier)
        validate_hash(sha256, nullable=True)
        if producer_attempt_id:
            # The control projection holds every attempt's task and every task's run/stage/version (same facts as a
            # query of attempts JOIN tasks; rebuilt after any rollback).
            control = self.control()
            attempt = control['attempts'].get(producer_attempt_id)
            task = control['tasks'].get(attempt['task_id']) if attempt is not None else None
            producer = (task['run_id'], task['stage'], task['version_id']) if task is not None else None
            if producer is None or producer[0] != run_id or producer[1] != stage or (producer[2] is not None and producer[2] != version_id):
                raise LedgerError("Artifact producer/run/stage/version mismatch")
        for page in (page_first, page_last):
            if page is not None and (type(page) is not int or page < 1): raise LedgerError("Page ranges require positive integers")
        if bytes is not None and (type(bytes) is not int or bytes < 0): raise LedgerError("Artifact bytes must be a nonnegative integer")
        if presence == "present" and (sha256 is None or bytes is None): raise LedgerError("Present artifact requires hash and size")
        if ownership == "managed" and (sha256 is None or bytes is None): raise LedgerError("Managed artifact requires hash and size")
        # Earlier registrations of this location, from the owner's catalog (current presence) by URI.
        catalog = self.file_catalog()
        previous_files = [(row['file_id'], row['version_id'], row['sha256'], row['presence'])
                          for row in (catalog[file_id] for file_id in self._uri_index().get(cast(str, uri), ()))]
        for previous in previous_files:
            changed = presence == "present" and previous[2] is not None and previous[2] != sha256
            absent = presence == "missing"
            if previous[3] == "present" and (changed or absent):
                if changed and ownership == "managed":
                    # glossAPI's own publication replaced the bytes at this fixed path (a re-published mirror or
                    # metadata table): a supersession, not corruption and not a policy deletion (LS4-G02, K1W12-G07).
                    self.event(run_id, event_kind="artifact", reason_code="missing", file_id=previous[0], version_id=previous[1],
                        details={"observation": "replaced_by_later_publication", "observed_sha256": sha256, "policy_deletion": False})
                    continue
                self.event(run_id, event_kind="artifact", reason_code="corrupt" if changed else "missing", file_id=previous[0], version_id=previous[1],
                    details={"observation": "legacy_location_bytes_replaced" if changed else "location_observed_missing", "observed_sha256":sha256})
        identifier = new_id()
        self.connection.execute("INSERT INTO files(file_id,version_id,producer_attempt_id,stage,kind,uri,sha256,bytes,ownership,retention_class,page_first,page_last,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (identifier, version_id, producer_attempt_id, stage, kind, uri, sha256, bytes, ownership, retention_class, page_first, page_last, utc_now()))
        catalog[identifier]=dict(file_id=identifier,version_id=version_id,producer_attempt_id=producer_attempt_id,
            stage=stage,kind=kind,uri=uri,sha256=sha256,bytes=bytes,ownership=ownership,
            retention_class=retention_class,page_first=page_first,page_last=page_last,presence=None,event_id=None)
        self._uri_index().setdefault(cast(str, uri), []).append(identifier)
        self.event(run_id, event_kind="artifact", reason_code=presence, version_id=version_id, file_id=identifier)
        return identifier

    def finish(self, run_id, task_id, attempt_id, *, status, reason_code, outputs=(), required_kinds=(), required_pages=(), details=None, **timing):
        """Register outputs and terminal event in the caller's single transaction."""
        self._owner()
        task = self.connection.execute("SELECT version_id,stage,specification_sha256 FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if task is None: raise LedgerError("Unknown task")
        specification = json.loads(self.connection.execute("SELECT json FROM snapshots WHERE sha256=?", (task[2],)).fetchone()[0])
        if status == "ok":
            kinds = {output['kind'] for output in outputs}
            if not set(required_kinds or specification.get('required_output_kinds', ())) <= kinds:
                raise LedgerError("Required output missing")
            covered = {page for output in outputs if output.get('page_first') is not None for page in range(output['page_first'], output['page_last']+1)}
            if not set(required_pages or specification.get('required_pages', ())) <= covered:
                raise LedgerError("Required page coverage missing")
        roles = [(output.get('role', output['kind']), output.get('page_first'), output.get('page_last')) for output in outputs]
        if len(roles) != len(set(roles)): raise LedgerError("Duplicate output role/range")
        registered = []
        produced = []
        for output in outputs:
            fields = {key: value for key, value in output.items() if key != 'role'}
            file_id = self.artifact(run_id, task[1], version_id=task[0], producer_attempt_id=attempt_id, **fields)
            produced.append({"file_id":file_id, **fields})
            if self.connection.compact:
                entry={"file_id":file_id,"fields":sorted(key for key in output if key!='role')}
                if 'role' in output:entry['role']=output['role']
                registered.append(entry)
            else:registered.append({"file_id": file_id, **output})
        self.produced_files[attempt_id] = produced
        manifest = self.snapshot("outputs", {"kind": "outputs", **({"encoding":"registered_files_v1"} if self.connection.compact else {}), "files": registered})
        return self.event(run_id, event_kind="lifecycle", task_id=task_id, version_id=task[0], attempt_id=attempt_id, status=status, reason_code=reason_code, output_manifest_sha256=manifest, details=details, **timing)
