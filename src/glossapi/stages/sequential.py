"""Single-owner sequential execution and immutable attempt publication.

This is a profile for qualifying adapters, not the streaming scheduler/run CLI.
Compatibility parquet serializers remain in the public phase wrappers.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import os
from pathlib import Path
import time

from .contracts import StageRequest
from .execution import ReceiptCollector, checked_path, sha256_file


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SequentialStageRunner:
    def __init__(self, writer, corpus_directory):
        self.writer = writer
        self.root = Path(corpus_directory).resolve()
        if writer.path.resolve() != self.root / "corpus.sqlite3":
            raise ValueError("Writer must own this corpus")

    def execute(self, stage, document, specification, *, timeout_seconds=600, tools=None, run_id=None):
        from ..ledger.journal import code_revision, loaded_tools, safe_settings
        from ..ledger.writer import new_id, utc_now
        runtime_settings = dict(specification.settings)
        if runtime_settings.get("memory_saving", "normal") != "normal":
            raise ValueError("disk_guard_unavailable: extreme mode requires the qualified quota and retention policy")
        if stage.name == "ocr":
            runtime_settings["input_producers"] = {artifact.file_id: self.writer.connection.execute(
                "SELECT producer_attempt_id FROM files WHERE file_id=?", (artifact.file_id,)).fetchone()[0] for artifact in document.artifacts}
        runtime_settings["required_output_kinds"] = list(specification.required_output_kinds)
        settings = safe_settings(runtime_settings)
        if run_id and stage.name == "clean" and document.artifacts and self.writer.connection.execute(
                "SELECT 1 FROM tasks WHERE run_id=? AND scope_key=? AND stage=? AND operation=?",
                (run_id,document.version_id,stage.name,specification.operation)).fetchone():
            # A canonical text and its chunks can share one document version.
            # Keep their transform operations distinct without inventing documents.
            specification = replace(specification, operation=specification.operation+".artifact:"+document.artifacts[0].file_id)
        specification_receipt = safe_settings(asdict(specification))
        tools = tools or loaded_tools()
        revision = code_revision()
        started_at, started_ns = utc_now(), time.monotonic_ns()
        with self.writer.transaction():
            run = run_id or self.writer.run({"profile": "sequential-adapters", **settings}, specification_receipt, tools, code_revision=revision)
            document_id = document.document_id
            if self.writer.connection.execute("SELECT 1 FROM documents WHERE document_id=?", (document_id,)).fetchone() is None:
                raise ValueError("Snapshot document identity is not registered")
            if stage.name == "download":
                # Reacquisition is a new observation, never a mutation of a
                # filled version. Its hashes are filled in the success commit.
                version = new_id()
                number = self.writer.connection.execute("SELECT COALESCE(MAX(revision),0)+1 FROM document_versions WHERE document_id=?", (document_id,)).fetchone()[0]
                self.writer.connection.execute("INSERT INTO document_versions VALUES(?,?,?,?,?,?)", (version, document_id, number, None, None, utc_now()))
            else:
                version = document.version_id
                row = self.writer.connection.execute("SELECT document_id FROM document_versions WHERE version_id=?", (version,)).fetchone()
                if row is None or row[0] != document_id:
                    raise ValueError("Snapshot version does not belong to document")
            for ordinal, source_record in enumerate(document.source_records):
                self.writer.source_record(document_id, run, document.source_uri, source_record,
                    source_kind="url" if document.source_uri.startswith(("http://", "https://")) else "local_path", source_row=str(ordinal))
            task = self.writer.task(run, stage.name, specification.operation, specification_receipt, version_id=version)
            self.writer.event(run, event_kind="lifecycle", reason_code="stage_planned", status="planned", task_id=task, version_id=version)
            bound_inputs = []
            for ordinal, artifact in enumerate(document.artifacts):
                fact = self.writer.connection.execute("SELECT * FROM files WHERE file_id=?", (artifact.file_id,)).fetchone()
                if fact is None or fact["version_id"] != version or fact["sha256"] != artifact.sha256 or fact["uri"] != artifact.path.absolute().as_uri():
                    raise ValueError("Input is not a committed artifact of this version")
                if fact["producer_attempt_id"]:
                    producer_task = self.writer.connection.execute("SELECT task_id FROM attempts WHERE attempt_id=?",(fact["producer_attempt_id"],)).fetchone()[0]
                    self.writer.dependency(task,producer_task,specification.input_roles[ordinal] if ordinal < len(specification.input_roles) else artifact.kind)
                bound_inputs.append({"file_id": artifact.file_id, "sha256": artifact.sha256, "role": specification.input_roles[ordinal] if ordinal < len(specification.input_roles) else artifact.kind})
            attempt = self.writer.attempt(task, worker_id=f"sequential:{os.getpid()}", device_id=specification.device_id,
                fingerprint={"contract_version": stage.contract_version, "specification": specification_receipt, "inputs": bound_inputs,
                    "tools": tools, "code_revision": revision, "reuse_eligible": False, "reason": "sequential qualification; no cache authority"},
                inputs={"files": bound_inputs}, settings=settings, tools=tools)
            for artifact in bound_inputs:
                self.writer.input(attempt, artifact["file_id"], artifact["role"], artifact["sha256"])
            lease = self.writer.connection.execute("SELECT lease_token FROM attempts WHERE attempt_id=?", (attempt,)).fetchone()[0]
            self.writer.event(run, event_kind="lifecycle", reason_code="stage_started", status="running", task_id=task,
                version_id=version, attempt_id=attempt, started_at=started_at)
        work = self.root / ".glossapi" / "staging" / attempt
        request = StageRequest(task, attempt, lease, document_id, version, specification.operation,
            document.artifacts, runtime_settings, tools, specification.device_id, work, started_ns + int(timeout_seconds*1e9))
        result = stage.run_document(request, reporter=ReceiptCollector())
        expected = (task, attempt, lease)
        if (result.task_id, result.attempt_id, result.lease_token) != expected:
            raise ValueError("Stale worker result; attempt left interrupted, outputs untrusted")
        records = []
        kinds = {output.kind for output in result.outputs}
        if result.status == "ok" and not set(specification.required_output_kinds) <= kinds:
            raise ValueError("Required output missing; no successful publication")
        paths = [output.relative_path for output in result.outputs]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate artifact path")
        for output in result.outputs:
            path = checked_path(work, output.relative_path)
            if not path.is_file():
                raise ValueError("Returned output is absent")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            records.append(dict(kind=output.kind, role=output.relative_path, sha256=sha256_file(path), bytes=path.stat().st_size,
                retention_class=output.retention_class, page_first=output.page_range[0] if output.page_range else None,
                page_last=output.page_range[1] if output.page_range else None))
        for directory in sorted((path for path in work.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
            if directory.is_symlink():
                raise ValueError("Symlink directory in attempt")
            _fsync_directory(directory)
        _fsync_directory(work)
        destination = self.root / ".glossapi" / "artifacts" / version / stage.name / attempt
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError("Attempt publication already exists")
        os.rename(work, destination)
        _fsync_directory(destination.parent)
        with self.writer.transaction():
            if stage.name == "download" and result.status == "ok":
                acquisition = result.decisions[0]["acquisitions"][-1]
                self.writer.fill_version_hashes(version, source_sha256=acquisition["source_sha256"], stored_sha256=acquisition["stored_sha256"])
                for duplicate in self.writer.connection.execute("SELECT document_id,version_id FROM document_versions WHERE stored_sha256=? AND version_id<>?", (acquisition["stored_sha256"], version)).fetchall():
                    self.writer.event(run, event_kind="decision", reason_code="duplicate_content", task_id=task,
                        version_id=version, attempt_id=attempt, details={"duplicate_kind": "content", "hash_basis": "stored",
                            "canonical_document_id": duplicate["document_id"], "canonical_version_id": duplicate["version_id"],
                            "evidence_hash": acquisition["stored_sha256"], "policy": "record_only"})
            for ordinal, decision in enumerate(result.decisions):
                from ..ledger.encoding import pack_evidence
                evidence = self.writer.snapshot("stage_evidence", pack_evidence({"kind": "stage_evidence", "stage": stage.name,
                    "operation": specification.operation, "contract_version": stage.contract_version, **safe_settings(decision)}))
                self.writer.event(run, event_kind="decision", reason_code=result.reason_code, task_id=task, version_id=version,
                    attempt_id=attempt, details={"evidence_sha256": evidence, "ordinal": ordinal})
            outputs = [{**record, "uri": (destination / output.relative_path).as_uri()} for record, output in zip(records, result.outputs)]
            self.writer.finish(run, task, attempt, status=result.status, reason_code=result.reason_code,
                outputs=outputs, required_kinds=specification.required_output_kinds, started_at=started_at, finished_at=utc_now(),
                error_type=result.error.error_type if result.error else None,
                traceback=result.error.traceback if result.error else None,
                message=result.error.message if result.error else "",
                duration_ns=time.monotonic_ns()-started_ns, details={"metrics": result.metrics,
                    "error": {"retryable":result.error.retryable} if result.error else None, "profile": "sequential-adapters"})
        return result, destination
