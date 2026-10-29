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
from .schema import APPEND_ONLY_TABLES, SCHEMA_SQL, SCHEMA_VERSION
from .validation import validate_evidence


_OPEN_WRITERS = set()


class LedgerError(ValueError):
    """Invalid evidence, ownership, transition or unsupported schema."""


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
    if version != SCHEMA_VERSION:
        raise LedgerError(f"Unsupported ledger schema {version}")
    rows = connection.execute("SELECT version,migration_sha256 FROM schema_migrations ORDER BY version").fetchall()
    if [r[0] for r in rows] != list(range(1, version + 1)):
        raise LedgerError("Schema migration history disagrees with user_version")
    if rows[0][1] != hashlib.sha256(SCHEMA_SQL.encode()).hexdigest():
        raise LedgerError("Schema migration checksum mismatch")


@contextmanager
def open_readonly(path):
    path = Path(path).absolute()
    wal = path.with_name(path.name + "-wal")
    # SQLite normally creates WAL/SHM sidecars even for mode=ro. A closed,
    # checkpointed ledger needs neither. A live ledger must already have its
    # writer-owned WAL index; never create one as a side effect of reporting.
    if wal.exists() and not path.with_name(path.name + "-shm").exists():
        raise LedgerError("Live WAL has no shared-memory index; retry after writer checkpoint")
    connection = sqlite3.connect(path.as_uri() + ("?mode=ro" if wal.exists() else "?mode=ro&immutable=1"), uri=True, isolation_level=None)
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
    def __init__(self, path, *, code_revision="unknown_not_recorded"):
        self.path = Path(path).absolute()
        if str(self.path) in _OPEN_WRITERS:
            raise BlockingIOError("Ledger already has an owning writer")
        self.owner = (os.getpid(), threading.get_ident())
        self.connection = None
        self.lock = os.open(self.path.parent, os.O_RDONLY)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _OPEN_WRITERS.add(str(self.path))
            self.connection = sqlite3.connect(self.path, isolation_level=None)
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
            supported_schema(self.connection)
            integrity(self.connection)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
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
            self.connection.close()
            self.connection = None
        if getattr(self, "lock", None) is not None:
            os.close(self.lock)
            self.lock = None
            _OPEN_WRITERS.discard(str(self.path))

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
        backup = sqlite3.connect(backup_path)
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

    def snapshot(self, kind, value):
        self._owner()
        value = evidence(value)
        validate_evidence(value)
        encoded = canonical_json(value)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        existing = self.connection.execute("SELECT kind,json FROM snapshots WHERE sha256=?", (digest,)).fetchone()
        # Kind is included in the hashed body by callers where semantically needed.
        if existing is None:
            self.connection.execute("INSERT INTO snapshots VALUES(?,?,1,?)", (digest, kind, encoded))
        return digest

    def run(self, settings, plan, environment, *, code_revision):
        self._owner()
        run_id = new_id()
        hashes = [self.snapshot(kind, {"kind": kind, **value}) for kind, value in
                  (("settings", settings), ("plan", plan), ("environment", environment))]
        self.connection.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)", (run_id, utc_now(), *hashes, code_revision))
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
        self.connection.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?)", (identifier, run_id, version_id, scope_key or version_id, stage, operation, ordinal, specification_hash))
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
        if self.connection.execute("""SELECT 1 FROM attempts a WHERE task_id=? AND NOT EXISTS
          (SELECT 1 FROM events e WHERE e.attempt_id=a.attempt_id AND e.event_kind='lifecycle' AND e.status IN ('ok','skipped','failed'))""", (task_id,)).fetchone():
            raise LedgerError("Task already has an active attempt")
        number = self.connection.execute("SELECT COALESCE(MAX(attempt_number),0)+1 FROM attempts WHERE task_id=?", (task_id,)).fetchone()[0]
        identifier = new_id()
        hashes = [self.snapshot(kind, {"kind": kind, **value}) for kind, value in
                  (("fingerprint", fingerprint), ("inputs", inputs), ("settings", settings), ("tools", tools))]
        self.connection.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?)", (identifier, task_id, number, worker_id, device_id, new_id(), *hashes, utc_now()))
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
            attempt_task = self.connection.execute("SELECT task_id FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt_task is None or task_id != attempt_task[0]: raise LedgerError("Attempt/task mismatch")
        if task_id:
            task = self.connection.execute("SELECT run_id,version_id FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if task is None or task[0] != run_id or task[1] != version_id: raise LedgerError("Task/run/version mismatch")
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
                last = self.connection.execute("SELECT status FROM events WHERE attempt_id=? AND event_kind='lifecycle' ORDER BY event_id DESC LIMIT 1", (attempt_id,)).fetchone()
                if last and last[0] in TERMINAL_STATUSES: raise LedgerError("Attempt is already terminal")
                if status == "planned" and last: raise LedgerError("Cannot regress an attempt to planned")
        payload = evidence(details or {})
        validate_evidence(payload)
        if duration_ns is not None and (type(duration_ns) is not int or duration_ns < 0):
            raise LedgerError("Duration must be nonnegative integer nanoseconds")
        cursor = self.connection.execute("""INSERT INTO events(run_id,task_id,version_id,attempt_id,file_id,event_kind,status,reason_code,message,error_type,traceback,started_at,finished_at,duration_ns,output_manifest_sha256,details_json,recorded_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (run_id, task_id, version_id, attempt_id, file_id, event_kind, status, reason_code, message, error_type, traceback, started_at, finished_at, duration_ns, output_manifest_sha256, canonical_json(payload), utc_now()))
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
        previous_files = self.connection.execute("SELECT file_id,version_id,sha256,presence FROM current_file_state WHERE uri=?", (uri,)).fetchall()
        for previous in previous_files:
            changed = presence == "present" and previous[2] is not None and previous[2] != sha256
            absent = presence == "missing"
            if previous[3] == "present" and (changed or absent):
                self.event(run_id, event_kind="artifact", reason_code="corrupt" if changed else "missing", file_id=previous[0], version_id=previous[1],
                    details={"observation": "legacy_location_bytes_replaced" if changed else "location_observed_missing", "observed_sha256":sha256})
        identifier = new_id()
        self.connection.execute("INSERT INTO files VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (identifier, version_id, producer_attempt_id, stage, kind, uri, sha256, bytes, ownership, retention_class, page_first, page_last, utc_now()))
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
        for output in outputs:
            fields = {key: value for key, value in output.items() if key != 'role'}
            file_id = self.artifact(run_id, task[1], version_id=task[0], producer_attempt_id=attempt_id, **fields)
            registered.append({"file_id": file_id, **output})
        manifest = self.snapshot("outputs", {"kind": "outputs", "files": registered})
        return self.event(run_id, event_kind="lifecycle", task_id=task_id, version_id=task[0], attempt_id=attempt_id, status=status, reason_code=reason_code, output_manifest_sha256=manifest, details=details, **timing)
