"""Normative version-1 DDL from LEDGER_DESIGN section 1 (stdlib only)."""
SCHEMA_VERSION = 3
APPEND_ONLY_TABLES = ("events", "source_records", "attempts", "files")
SCHEMA_SQL = r"""
CREATE TABLE ledger_info (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  corpus_id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
);
CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL,
  migration_sha256 TEXT NOT NULL, code_revision TEXT NOT NULL
);
CREATE TABLE snapshots (
  sha256 TEXT PRIMARY KEY, kind TEXT NOT NULL,
  schema_version INTEGER NOT NULL CHECK(schema_version>0), json TEXT NOT NULL
);
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
  settings_sha256 TEXT NOT NULL REFERENCES snapshots(sha256),
  plan_sha256 TEXT NOT NULL REFERENCES snapshots(sha256),
  environment_sha256 TEXT NOT NULL REFERENCES snapshots(sha256),
  code_revision TEXT NOT NULL
);
CREATE TABLE documents (
  document_id TEXT PRIMARY KEY, source_key TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);
CREATE TABLE source_records (
  source_record_id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(document_id),
  run_id TEXT NOT NULL REFERENCES runs(run_id), source_uri TEXT NOT NULL,
  source_kind TEXT NOT NULL CHECK(source_kind IN ('url','local_path','transfer')),
  manifest_sha256 TEXT REFERENCES snapshots(sha256),
  source_row TEXT, url_index INTEGER CHECK(url_index>=0), metadata_json TEXT NOT NULL
);
CREATE TABLE document_versions (
  version_id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(document_id),
  revision INTEGER NOT NULL CHECK(revision>0),
  source_sha256 TEXT, stored_sha256 TEXT, created_at TEXT NOT NULL,
  UNIQUE(document_id,revision)
);
CREATE TABLE tasks (
  task_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  version_id TEXT REFERENCES document_versions(version_id),
  scope_key TEXT NOT NULL,
  stage TEXT NOT NULL CHECK(stage IN
    ('download','extract','clean','ocr','math','sections','annotate','export')),
  operation TEXT NOT NULL, ordinal INTEGER NOT NULL CHECK(ordinal>=0),
  specification_sha256 TEXT NOT NULL REFERENCES snapshots(sha256),
  UNIQUE(run_id,scope_key,stage,operation)
);
CREATE TABLE task_dependencies (
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  producer_task_id TEXT NOT NULL REFERENCES tasks(task_id),
  input_role TEXT NOT NULL,
  PRIMARY KEY(task_id,producer_task_id,input_role), CHECK(task_id<>producer_task_id)
);
CREATE TABLE attempts (
  attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
  attempt_number INTEGER NOT NULL CHECK(attempt_number>0),
  worker_id TEXT NOT NULL, device_id TEXT NOT NULL, lease_token TEXT NOT NULL UNIQUE,
  fingerprint_sha256 TEXT NOT NULL REFERENCES snapshots(sha256),
  input_manifest_sha256 TEXT NOT NULL REFERENCES snapshots(sha256),
  settings_sha256 TEXT NOT NULL REFERENCES snapshots(sha256),
  tools_sha256 TEXT NOT NULL REFERENCES snapshots(sha256),
  created_at TEXT NOT NULL, UNIQUE(task_id,attempt_number)
);
CREATE TABLE files (
  file_id TEXT PRIMARY KEY, version_id TEXT REFERENCES document_versions(version_id),
  producer_attempt_id TEXT REFERENCES attempts(attempt_id),
  stage TEXT NOT NULL CHECK(stage IN
    ('download','extract','clean','ocr','math','sections','annotate','export')),
  kind TEXT NOT NULL, uri TEXT NOT NULL, sha256 TEXT,
  bytes INTEGER CHECK(bytes>=0),
  ownership TEXT NOT NULL CHECK(ownership IN ('managed','initial_external','legacy_external')),
  retention_class TEXT NOT NULL CHECK(retention_class IN
    ('initial','final','intermediate','evidence','recovery','failed_attempt')),
  page_first INTEGER CHECK(page_first>=1), page_last INTEGER,
  created_at TEXT NOT NULL,
  CHECK((page_first IS NULL AND page_last IS NULL) OR
        (page_first IS NOT NULL AND page_last IS NOT NULL AND page_last>=page_first))
);
CREATE TABLE attempt_inputs (
  attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
  file_id TEXT NOT NULL REFERENCES files(file_id), role TEXT NOT NULL,
  sha256 TEXT NOT NULL, PRIMARY KEY(attempt_id,file_id,role)
);
CREATE TABLE events (
  event_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
  task_id TEXT REFERENCES tasks(task_id),
  version_id TEXT REFERENCES document_versions(version_id),
  attempt_id TEXT REFERENCES attempts(attempt_id), file_id TEXT REFERENCES files(file_id),
  event_kind TEXT NOT NULL CHECK(event_kind IN
    ('lifecycle','decision','heartbeat','artifact','retention','import','invalidation','run')),
  status TEXT CHECK(status IN ('planned','running','ok','skipped','failed')),
  reason_code TEXT NOT NULL, message TEXT NOT NULL,
  error_type TEXT, traceback TEXT, started_at TEXT, finished_at TEXT,
  duration_ns INTEGER CHECK(duration_ns>=0),
  output_manifest_sha256 TEXT REFERENCES snapshots(sha256),
  details_json TEXT NOT NULL, recorded_at TEXT NOT NULL
);
CREATE INDEX source_records_document ON source_records(document_id,run_id);
CREATE INDEX source_records_manifest ON source_records(manifest_sha256,source_row,url_index);
CREATE INDEX source_records_uri ON source_records(source_uri);
CREATE INDEX versions_source_hash ON document_versions(source_sha256);
CREATE INDEX versions_stored_hash ON document_versions(stored_sha256);
CREATE INDEX tasks_stage ON tasks(run_id,stage,ordinal);
CREATE INDEX tasks_document ON tasks(version_id,stage,operation);
CREATE INDEX dependencies_producer ON task_dependencies(producer_task_id);
CREATE INDEX attempts_fingerprint ON attempts(fingerprint_sha256);
CREATE INDEX events_task ON events(task_id,event_id DESC);
CREATE INDEX events_run ON events(run_id,event_id);
CREATE INDEX events_attempt ON events(attempt_id,event_id);
CREATE INDEX events_document ON events(version_id,event_id);
CREATE INDEX events_file ON events(file_id,event_id);
CREATE INDEX events_reason ON events(reason_code,event_id);
CREATE INDEX files_hash ON files(sha256);
CREATE INDEX files_uri ON files(uri);
CREATE INDEX files_version ON files(version_id,stage,kind);
CREATE INDEX files_producer ON files(producer_attempt_id);
CREATE INDEX inputs_file ON attempt_inputs(file_id,sha256);
CREATE VIEW current_task_state AS
  SELECT t.*, e.status, e.reason_code, e.event_id
  FROM tasks t LEFT JOIN events e ON e.event_id=(
    SELECT MAX(s.event_id) FROM events s
    WHERE s.task_id=t.task_id AND s.event_kind='lifecycle'
  );
CREATE VIEW current_file_state AS
  SELECT f.*, e.reason_code AS presence, e.event_id
  FROM files f LEFT JOIN events e ON e.event_id=(
    SELECT MAX(s.event_id) FROM events s
    WHERE s.file_id=f.file_id AND s.event_kind='artifact'
  );
"""

# Version 2 retains every logical row and the v1 canonical snapshot hash.
# Physical UUID/hash/JSON encoding is handled by ledger.compact.
COMPACT_SCHEMA_SQL = """
DROP INDEX events_task;
CREATE INDEX events_task ON events(task_id,event_id DESC) WHERE task_id IS NOT NULL;
DROP INDEX events_attempt;
CREATE INDEX events_attempt ON events(attempt_id,event_id) WHERE attempt_id IS NOT NULL;
DROP INDEX events_document;
CREATE INDEX events_document ON events(version_id,event_id) WHERE version_id IS NOT NULL;
DROP INDEX events_file;
CREATE INDEX events_file ON events(file_id,event_id) WHERE file_id IS NOT NULL;
"""

COMPACT_SCHEMA_SQL += """
ALTER TABLE events RENAME TO event_records;
CREATE TABLE retention_events (
 event_id INTEGER PRIMARY KEY,
 run_id TEXT NOT NULL REFERENCES runs(run_id),
 file_id TEXT NOT NULL REFERENCES files(file_id),
 reason_code TEXT NOT NULL, message TEXT NOT NULL,
 details_json TEXT NOT NULL, recorded_at TEXT NOT NULL
);
CREATE INDEX retention_events_file ON retention_events(file_id,event_id);
CREATE INDEX retention_events_run ON retention_events(run_id,event_id);
CREATE INDEX retention_events_reason ON retention_events(reason_code,event_id);
CREATE VIEW events AS SELECT * FROM event_records UNION ALL
 SELECT r.event_id,r.run_id,NULL,f.version_id,NULL,r.file_id,'retention',NULL,
 r.reason_code,r.message,NULL,NULL,NULL,NULL,NULL,NULL,r.details_json,r.recorded_at
 FROM retention_events r JOIN files f ON f.file_id=r.file_id;
CREATE TRIGGER retention_events_no_update BEFORE UPDATE ON retention_events
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
CREATE TRIGGER retention_events_no_delete BEFORE DELETE ON retention_events
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
CREATE TRIGGER events_view_no_update INSTEAD OF UPDATE ON events
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
CREATE TRIGGER events_view_no_delete INSTEAD OF DELETE ON events
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
CREATE TRIGGER events_view_insert INSTEAD OF INSERT ON events BEGIN
 INSERT INTO event_records VALUES(NEW.event_id,NEW.run_id,NEW.task_id,NEW.version_id,
 NEW.attempt_id,NEW.file_id,NEW.event_kind,NEW.status,NEW.reason_code,NEW.message,
 NEW.error_type,NEW.traceback,NEW.started_at,NEW.finished_at,NEW.duration_ns,
 NEW.output_manifest_sha256,NEW.details_json,NEW.recorded_at);
END;
"""

# New ledgers only. Historical v1/v2 creation scripts and checksums stay frozen.
COMPACT_RETENTION_SCHEMA_SQL = """
CREATE TABLE path_root (singleton INTEGER PRIMARY KEY CHECK(singleton=1), uri TEXT NOT NULL);
CREATE TABLE paths (path_id INTEGER PRIMARY KEY, uri TEXT NOT NULL UNIQUE);
ALTER TABLE files RENAME TO file_records;
DROP INDEX files_uri;
CREATE INDEX files_uri ON file_records(CAST(uri AS INTEGER));
CREATE VIEW files AS SELECT f.file_id,f.version_id,f.producer_attempt_id,f.stage,f.kind,
 p.uri,f.sha256,f.bytes,f.ownership,f.retention_class,f.page_first,f.page_last,f.created_at,f.rowid AS rowid
 FROM file_records f JOIN paths p ON p.path_id=CAST(f.uri AS INTEGER);
CREATE TRIGGER files_insert INSTEAD OF INSERT ON files BEGIN
 INSERT OR IGNORE INTO paths(uri) VALUES(NEW.uri);
 INSERT INTO file_records VALUES(NEW.file_id,NEW.version_id,NEW.producer_attempt_id,
 NEW.stage,NEW.kind,(SELECT path_id FROM paths WHERE uri=NEW.uri),NEW.sha256,
 NEW.bytes,NEW.ownership,NEW.retention_class,NEW.page_first,NEW.page_last,NEW.created_at);
END;
CREATE TRIGGER files_view_no_update INSTEAD OF UPDATE ON files
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
CREATE TRIGGER files_view_no_delete INSTEAD OF DELETE ON files
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
DROP VIEW current_file_state;
CREATE VIEW current_file_state AS
 SELECT f.*,e.reason_code AS presence,e.event_id FROM files f
 LEFT JOIN event_records e ON e.event_id=(SELECT MAX(s.event_id) FROM event_records s
 WHERE s.file_id=f.file_id AND s.event_kind='artifact');
CREATE TABLE completed_deletions (
 file_id BLOB PRIMARY KEY REFERENCES file_records(file_id),
 event_id INTEGER NOT NULL UNIQUE,
 run_id BLOB NOT NULL REFERENCES runs(run_id),
 reason_code TEXT NOT NULL,
 deletion_reason TEXT NOT NULL,
 last_consumer_stage TEXT NOT NULL,
 recorded_at TEXT NOT NULL
);
CREATE TRIGGER completed_deletions_no_update BEFORE UPDATE ON completed_deletions
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
CREATE TRIGGER completed_deletions_no_delete BEFORE DELETE ON completed_deletions
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
-- Only this transient journal can collapse after a successful durable delete.
DROP TRIGGER retention_events_no_delete;
CREATE TRIGGER retention_events_no_delete BEFORE DELETE ON retention_events
 WHEN NOT EXISTS (SELECT 1 FROM completed_deletions c WHERE c.file_id=OLD.file_id)
 BEGIN SELECT RAISE(ABORT, 'active retention journal'); END;
DROP VIEW events;
CREATE VIEW events AS SELECT * FROM event_records UNION ALL
 SELECT r.event_id,r.run_id,NULL,f.version_id,NULL,r.file_id,'retention',NULL,
 r.reason_code,r.message,NULL,NULL,NULL,NULL,NULL,NULL,r.details_json,r.recorded_at
 FROM retention_events r JOIN file_records f ON f.file_id=r.file_id UNION ALL
 SELECT c.event_id,c.run_id,NULL,f.version_id,NULL,c.file_id,'retention',NULL,
 c.reason_code,'',NULL,NULL,NULL,NULL,NULL,NULL,'{}',c.recorded_at
 FROM completed_deletions c JOIN file_records f ON f.file_id=c.file_id;
CREATE TRIGGER events_view_no_update INSTEAD OF UPDATE ON events
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
CREATE TRIGGER events_view_no_delete INSTEAD OF DELETE ON events
 BEGIN SELECT RAISE(ABORT, 'append-only table'); END;
CREATE TRIGGER events_view_insert INSTEAD OF INSERT ON events BEGIN
 INSERT INTO event_records VALUES(NEW.event_id,NEW.run_id,NEW.task_id,NEW.version_id,
 NEW.attempt_id,NEW.file_id,NEW.event_kind,NEW.status,NEW.reason_code,NEW.message,
 NEW.error_type,NEW.traceback,NEW.started_at,NEW.finished_at,NEW.duration_ns,
 NEW.output_manifest_sha256,NEW.details_json,NEW.recorded_at);
END;
DROP INDEX source_records_manifest;
DROP INDEX versions_source_hash;
DROP INDEX attempts_fingerprint;
"""
