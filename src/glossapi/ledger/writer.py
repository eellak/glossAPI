"""Single-writer SQLite fact journal; no orchestration or backend imports."""
from __future__ import annotations

from contextlib import contextmanager
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
from .schema import APPEND_ONLY_TABLES, SCHEMA_SQL, SCHEMA_VERSION, COMPACT_SCHEMA_SQL, COMPACT_RETENTION_SCHEMA_SQL
from .compact import connect, encode_json
from .validation import validate_evidence
from .urls import needs_scrub, persisted_url, pop_identity_root, push_identity_root, scrub, scrub_text


_OPEN_WRITERS = set()


class LedgerError(ValueError):
    """Invalid evidence, ownership, transition or unsupported schema."""


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


_SYNCFS=_load_syncfs()


def sync_filesystem(descriptor):
    """Flush every file and directory of descriptor's filesystem and wait."""
    if _SYNCFS(descriptor)!=0:
        import ctypes
        error=ctypes.get_errno()
        raise OSError(error,os.strerror(error))


class SyncBatch:
    """One durability flush for a batch of changed files and directories.

    With syncfs this is one call per filesystem. Otherwise each registered file
    and then each registered directory is fsynced exactly once per batch.
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
        if _SYNCFS is not None:
            devices={}
            for (device,_),(descriptor,_) in self.descriptors.items():devices.setdefault(device,descriptor)
            for descriptor in devices.values():sync_filesystem(descriptor)
            return
        for descriptor,directory in self.descriptors.values():
            if not directory:os.fsync(descriptor)
        for descriptor,directory in self.descriptors.values():
            if directory:os.fsync(descriptor)

    def close(self):
        for descriptor,_ in self.descriptors.values():os.close(descriptor)
        self.descriptors={}

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def new_id():
    return str(uuid.uuid4())


def validate_id(value):
    if str(uuid.UUID(value)) != value:
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
    if version not in (1, 2, SCHEMA_VERSION):
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
        if storage_version not in (1, 2, SCHEMA_VERSION):raise LedgerError("Unsupported creation schema")
        self.path = Path(path).absolute()
        if str(self.path) in _OPEN_WRITERS:
            raise BlockingIOError("Ledger already has an owning writer")
        self.owner = (os.getpid(), threading.get_ident())
        self.connection = None
        self.control_cache = None
        self._file_catalog = None
        self._run_plans = {}
        self._planned_reader_maps = {}
        self.snapshot_cache = set()
        self.produced_files = {}
        self.lock = os.open(self.path.parent, os.O_RDONLY)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _OPEN_WRITERS.add(str(self.path))
            # Lossy URLs are keyed under this corpus's identity key (cycle-2 F2).
            push_identity_root(self.path.parent)
            self._identity_root = self.path.parent
            self.connection = connect(self.path, isolation_level=None)
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
                self.connection.compact = storage_version >= 2
                self.connection.storage_version = storage_version
                if storage_version>=3:self.connection.path_root=self.path.parent.as_uri()
            supported_schema(self.connection)
            integrity(self.connection)
            # The scheduler's indexed control projection is process local. Old
            # creation scripts/checksums and read-only historical reports stay exact.
            self.connection.execute(f"CREATE TEMP VIEW current_task_state AS SELECT t.*,e.status,e.reason_code,e.event_id FROM tasks t LEFT JOIN {self.connection.event_table} e ON e.event_id=(SELECT MAX(s.event_id) FROM {self.connection.event_table} s WHERE s.task_id=t.task_id AND s.event_kind='lifecycle')")
            self.connection.execute('CREATE INDEX IF NOT EXISTS snapshots_kind ON snapshots(kind)')
            self.connection.execute('CREATE INDEX IF NOT EXISTS tasks_run_operation ON tasks(run_id,operation)')
            self.connection.execute('CREATE INDEX IF NOT EXISTS tasks_stage_operation ON tasks(stage,operation)')
            self.connection.execute(f"CREATE INDEX IF NOT EXISTS events_decision_task ON {self.connection.event_table}(event_kind,task_id,event_id) WHERE event_kind='decision'")
            file_table='file_records' if self.connection.storage_version>=3 else 'files'
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
            self._durable_files=set()
            self.file_catalog()
            self.connection.owner_writer=self
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
        if self.connection is not None:
            if self.connection.in_transaction:
                self.connection.rollback()
                self._file_catalog = None
            if hasattr(self,'_durable_files'):self.durable_barrier()
            self.connection.close()
            self.connection = None
        if getattr(self, "lock", None) is not None:
            os.close(self.lock)
            self.lock = None
            _OPEN_WRITERS.discard(str(self.path))
        if getattr(self, "_identity_root", None) is not None:
            pop_identity_root(self._identity_root)
            self._identity_root = None

    def durable_barrier(self):
        """Flush immutable managed products and their parents before a FULL commit.

        Ordinary WAL commits protect process death. This is the explicit disk
        boundary used before unlink and at run end. The cache is by immutable
        registration, never by stat metadata and never used to bypass a hash.
        """
        self._owner(transaction=False)
        if self.connection.in_transaction:
            raise LedgerError('Durability barrier requires a committed observation')
        from urllib.parse import unquote, urlsplit
        import stat
        directories=set()
        synchronized=[];paths=[];devices={}
        for row in self.file_catalog().values():
            if row['ownership']!='managed' or row['presence']!='present':continue
            file_id,uri=row['file_id'],row['uri']
            if file_id in self._durable_files:continue
            path=Path(unquote(urlsplit(uri).path))
            try:info=os.stat(path)
            except OSError:continue  # Missing observations are diagnosed by verification/resume.
            if not stat.S_ISREG(info.st_mode):continue
            devices.setdefault(info.st_dev,path)
            paths.append(path)
            directories.update(parent for parent in path.parents if parent.is_relative_to(self.path.parent))
            synchronized.append(file_id)
        if _SYNCFS is not None:
            # One flush per filesystem covers every listed file and parent
            # directory of this batch; the FULL commit below follows it.
            for directory in directories:devices.setdefault(os.stat(directory).st_dev,directory)
            with SyncBatch() as batch:
                for path in devices.values():batch.add_path(path)
                batch.flush()
        else:
            for path in paths:
                with path.open('rb') as stream:os.fsync(stream.fileno())
            for directory in sorted(directories,key=lambda path:len(path.parts),reverse=True):
                descriptor=os.open(directory,os.O_RDONLY)
                try:os.fsync(descriptor)
                finally:os.close(descriptor)
        self.connection.execute('PRAGMA synchronous=FULL')
        try:
            with self.transaction():
                self.connection.execute('UPDATE ledger_info SET singleton=singleton WHERE singleton=1')
        finally:self.connection.execute('PRAGMA synchronous=NORMAL')
        self._durable_files.update(synchronized)

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def migrate(self, version, sql, *, checksum, code_revision, backup_path):
        """Apply a numbered checksum-bound future migration; rollback on failure."""
        self._owner(transaction=False)
        if self.connection.in_transaction: raise LedgerError("Migration requires no active observation")
        current = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version != current + 1 or hashlib.sha256(sql.encode()).hexdigest() != checksum:
            raise LedgerError("Migration sequence/checksum mismatch")
        if Path(backup_path).exists(): raise LedgerError("Backup already exists")
        integrity(self.connection)
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

    def file_catalog(self):
        """Version-bound file facts, rebuilt only at open/resume or rollback.

        This is control state, not filesystem byte verification. Publication,
        source deletion and audit still perform their required content checks.
        """
        self._owner(transaction=False)
        if self._file_catalog is None:
            self._file_catalog={row['file_id']:dict(row) for row in self.connection.execute('SELECT * FROM current_file_state')}
        return self._file_catalog

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

    def accepts_result(self, attempt_id, lease_token):
        state=self.control()
        attempt=state['attempts'].get(attempt_id)
        return bool(attempt and attempt['lease_token']==lease_token
            and state['status'].get(attempt_id) not in TERMINAL_STATUSES)

    def snapshot(self, kind, value):
        self._owner()
        value = evidence(value)
        validate_evidence(value)
        encoded = canonical_json(value)
        if needs_scrub(encoded):
            # Last recording boundary for URL credentials (CP-LEDGER F2).
            value = scrub(value)
            encoded = canonical_json(value)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        if digest in self.snapshot_cache:
            return digest
        stored=encode_json(encoded) if self.connection.compact else encoded
        if getattr(self.connection,'storage_version',0)>=3:
            stored=self.connection.encode_snapshot(value,lambda common:self.snapshot('shared_config',{'kind':'shared_config','value':common}))
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
        source_key = scrub_text(source_key)
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
        source_uri = persisted_url(source_uri)
        encoded = canonical_json({"schema_version": 1, "fields": typed_value(metadata)})
        if needs_scrub(encoded):
            encoded = canonical_json({"schema_version": 1, "fields": typed_value(scrub(metadata))})
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
            artifact = self.connection.execute("SELECT version_id,producer_attempt_id FROM files WHERE file_id=?", (file_id,)).fetchone()
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
        if needs_scrub(text):
            payload = scrub(payload)
            text = canonical_json(payload)
        message, traceback = scrub_text(message), scrub_text(traceback)
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
            producer = self.connection.execute("SELECT t.run_id,t.stage,t.version_id FROM attempts a JOIN tasks t ON t.task_id=a.task_id WHERE a.attempt_id=?", (producer_attempt_id,)).fetchone()
            if producer is None or producer[0] != run_id or producer[1] != stage or (producer[2] is not None and producer[2] != version_id):
                raise LedgerError("Artifact producer/run/stage/version mismatch")
        for page in (page_first, page_last):
            if page is not None and (type(page) is not int or page < 1): raise LedgerError("Page ranges require positive integers")
        if bytes is not None and (type(bytes) is not int or bytes < 0): raise LedgerError("Artifact bytes must be a nonnegative integer")
        if presence == "present" and (sha256 is None or bytes is None): raise LedgerError("Present artifact requires hash and size")
        if ownership == "managed" and (sha256 is None or bytes is None): raise LedgerError("Managed artifact requires hash and size")
        uri = persisted_url(uri)
        previous_files = self.connection.execute("SELECT file_id,version_id,sha256,presence FROM current_file_state WHERE uri=?", (uri,)).fetchall()
        for previous in previous_files:
            changed = presence == "present" and previous[2] is not None and previous[2] != sha256
            absent = presence == "missing"
            if previous[3] == "present" and (changed or absent):
                self.event(run_id, event_kind="artifact", reason_code="corrupt" if changed else "missing", file_id=previous[0], version_id=previous[1],
                    details={"observation": "legacy_location_bytes_replaced" if changed else "location_observed_missing", "observed_sha256":sha256})
        catalog=self.file_catalog()
        identifier = new_id()
        self.connection.execute("INSERT INTO files(file_id,version_id,producer_attempt_id,stage,kind,uri,sha256,bytes,ownership,retention_class,page_first,page_last,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (identifier, version_id, producer_attempt_id, stage, kind, uri, sha256, bytes, ownership, retention_class, page_first, page_last, utc_now()))
        catalog[identifier]=dict(file_id=identifier,version_id=version_id,producer_attempt_id=producer_attempt_id,
            stage=stage,kind=kind,uri=uri,sha256=sha256,bytes=bytes,ownership=ownership,
            retention_class=retention_class,page_first=page_first,page_last=page_last,presence=None,event_id=None)
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
