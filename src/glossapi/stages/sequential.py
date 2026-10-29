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
        self.pending_scratch=[]
        if writer.path.resolve() != self.root / "corpus.sqlite3":
            raise ValueError("Writer must own this corpus")

    def execute(self, stage, document, specification, **options):
        from .replay import replay_result
        replay = replay_result(stage, document, specification)
        if replay is not None:
            result,workspace=replay
            run_id=options.get('run_id')
            if run_id:
                self.verify_projection(stage,document,specification,run_id,result)
            return result,workspace
        request, complete = self.prepare(stage, document, specification, **options)
        from .execution import producer_hash_scope,seal_output_hashes
        with producer_hash_scope():
            result = (stage.finalize if options.get('aggregate') else stage.run_document)(request, reporter=ReceiptCollector())
            result = seal_output_hashes(result,request.work_directory)
        return complete(result)

    def verify_projection(self,stage,document,specification,run_id,result):
        """A cache projection is an exact reader of its supplied alias inputs."""
        from ..ledger.writer import new_id
        if result.status!='ok':return
        for artifact in document.artifacts:
            if artifact.path.stat().st_size!=artifact.bytes or sha256_file(artifact.path)!=artifact.sha256:
                raise ValueError('Projection input hash changed')
        source=self.writer.connection.execute("SELECT task_id FROM attempts WHERE attempt_id=? AND EXISTS (SELECT 1 FROM events WHERE attempt_id=? AND event_kind='lifecycle' AND status='ok')",(result.attempt_id,result.attempt_id)).fetchone()
        if not source:raise ValueError('Projection producer is not successful')
        with self.writer.transaction():
            task=self.writer.task(run_id,stage.name,specification.operation+'.compatibility_projection:'+new_id(),
                {'source_attempt_id':result.attempt_id,'verification':'same document version and exact input hashes'},version_id=document.version_id)
            self.writer.dependency(task,source[0],'verified_producer')
            inputs=[{'file_id':artifact.file_id,'sha256':artifact.sha256} for artifact in document.artifacts]
            attempt=self.writer.attempt(task,worker_id='compatibility-verifier',device_id='cpu',
                fingerprint={'source_attempt_id':result.attempt_id,'reuse_eligible':False},inputs={'files':inputs},settings={},tools={})
            for index,artifact in enumerate(document.artifacts):
                self.writer.input(attempt,artifact.file_id,specification.input_roles[index],artifact.sha256)
            self.writer.finish(run_id,task,attempt,status='ok',reason_code=stage.name+'_completed',
                details={'source_attempt_id':result.attempt_id,'verification':'committed result projected without kernel execution'})

    async def execute_async(self, stage, document, specification, **options):
        request, complete = self.prepare(stage, document, specification, **options)
        result = await stage.run_document_async(request, reporter=ReceiptCollector())
        return complete(result)

    def prepare(self, stage, document, specification, *, timeout_seconds=600, tools=None, run_id=None, aggregate=False,
                planned_task_id=None, fingerprint=None, worker_id=None):
        from ..ledger.journal import code_revision, loaded_tools, safe_settings
        from ..ledger.writer import new_id, utc_now
        from .retention import resolved_settings
        run_settings = {}
        if run_id:
            import json
            row = self.writer.connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.settings_sha256 WHERE r.run_id=?',(run_id,)).fetchone()
            run_settings = json.loads(row[0]) if row else {}
        retention_settings = {key:run_settings[key] for key in ('debug','memory_saving','disk_budget_bytes','max_source_bytes','max_attempt_bytes') if key in run_settings}
        runtime_settings = {**resolved_settings(retention_settings), **dict(specification.settings)}
        if runtime_settings['memory_saving'] == 'extreme':
            from .disk_budget import require_disk_guard
            require_disk_guard(self.root, runtime_settings)
            if any(artifact.bytes>runtime_settings['max_source_bytes'] for artifact in document.artifacts if artifact.kind in ('source_payload','download_payload','initial_input')):
                from .disk_budget import DiskBudgetError
                raise DiskBudgetError('disk_budget_exceeded','known source exceeds max_source_bytes')
            if sum(artifact.bytes for artifact in document.artifacts)>runtime_settings['max_attempt_bytes']:
                from .disk_budget import DiskBudgetError
                raise DiskBudgetError('disk_budget_exceeded','known inputs exceed max_attempt_bytes')
        if stage.name == "ocr":
            runtime_settings["input_producers"] = {artifact.file_id: self.writer.connection.execute(
                "SELECT producer_attempt_id FROM files WHERE file_id=?", (artifact.file_id,)).fetchone()[0] for artifact in document.artifacts}
        runtime_settings["required_output_kinds"] = list(specification.required_output_kinds)
        settings = safe_settings(runtime_settings)
        if run_id and not planned_task_id and not aggregate and document.artifacts and self.writer.connection.execute(
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
            if not aggregate and self.writer.connection.execute("SELECT 1 FROM documents WHERE document_id=?", (document_id,)).fetchone() is None:
                raise ValueError("Snapshot document identity is not registered")
            if aggregate:
                version = None
            elif stage.name == "download" and not planned_task_id:
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
            from ..ledger.encoding import snapshot_hash
            scope = 'corpus:' + self.writer.snapshot('inputs', {'kind':'inputs', 'files':[
                asdict(artifact) | {'path':str(artifact.path)} for artifact in document.artifacts]}) if aggregate else None
            task = planned_task_id or self.writer.task(run, stage.name, specification.operation, specification_receipt, version_id=version, scope_key=scope)
            if planned_task_id:
                bound_task = self.writer.connection.execute('SELECT run_id,version_id,stage,operation FROM tasks WHERE task_id=?',(task,)).fetchone()
                if tuple(bound_task) != (run,version,stage.name,specification.operation):
                    raise ValueError('Frozen task does not match dispatch')
            else:
                self.writer.event(run, event_kind="lifecycle", reason_code="stage_planned", status="planned", task_id=task, version_id=version)
            bound_inputs = []
            producer_roles = set()
            for ordinal, artifact in enumerate(document.artifacts):
                fact = self.writer.connection.execute("SELECT * FROM files WHERE file_id=?", (artifact.file_id,)).fetchone()
                if fact is None or (not aggregate and fact["version_id"] != version) or fact["sha256"] != artifact.sha256 or fact["uri"] != artifact.path.absolute().as_uri():
                    raise ValueError("Input is not a committed artifact of this version")
                if fact["producer_attempt_id"]:
                    producer_task = self.writer.connection.execute("SELECT task_id FROM attempts WHERE attempt_id=?",(fact["producer_attempt_id"],)).fetchone()[0]
                    input_role=specification.input_roles[ordinal] if ordinal < len(specification.input_roles) else artifact.kind
                    # Several distinct files can satisfy one role from the same
                    # producer. Keep every file binding, but one DAG edge.
                    if (producer_task,input_role) not in producer_roles and not self.writer.connection.execute(
                            'SELECT 1 FROM task_dependencies WHERE task_id=? AND producer_task_id=? AND input_role=?',(task,producer_task,input_role)).fetchone():
                        self.writer.dependency(task,producer_task,input_role)
                        producer_roles.add((producer_task,input_role))
                bound_inputs.append({"file_id": artifact.file_id, "sha256": artifact.sha256, "role": specification.input_roles[ordinal] if ordinal < len(specification.input_roles) else artifact.kind})
            attempt = self.writer.attempt(task, worker_id=worker_id or f"sequential:{os.getpid()}", device_id=specification.device_id,
                fingerprint=fingerprint or {"contract_version": stage.contract_version, "specification": specification_receipt, "inputs": bound_inputs,
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
        def complete(result):
            expected = (task, attempt, lease)
            if (result.task_id, result.attempt_id, result.lease_token) != expected:
                raise ValueError("Stale worker result; attempt left interrupted, outputs untrusted")
            current = self.writer.connection.execute("SELECT a.lease_token FROM attempts a WHERE a.attempt_id=? AND NOT EXISTS (SELECT 1 FROM events e WHERE e.attempt_id=a.attempt_id AND e.event_kind='lifecycle' AND e.status IN ('ok','failed','skipped'))",(attempt,)).fetchone()
            if current is None or current[0] != lease:
                raise ValueError('Expired worker lease; outputs untrusted')
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
                records.append(dict(kind=output.kind, role=output.relative_path, sha256=output.sha256 or sha256_file(path), bytes=output.bytes if output.bytes is not None else path.stat().st_size,
                    retention_class=output.retention_class, page_first=output.page_range[0] if output.page_range else None,
                    page_last=output.page_range[1] if output.page_range else None))
            destination = self.root / ".glossapi" / "artifacts" / (version or 'corpus') / stage.name / attempt
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError("Attempt publication already exists")
            cross_filesystem=False
            try:
                os.rename(work, destination)
            except OSError as error:
                import errno, shutil
                if error.errno != errno.EXDEV:raise
                # A hard quota can put scratch on another filesystem. The
                # reservation includes both copies until the durable publication.
                cross_filesystem=True
                shutil.copytree(work,destination)
                for record in records:
                    copied=destination/record['role']
                    if copied.stat().st_size!=record['bytes'] or sha256_file(copied)!=record['sha256']:
                        raise ValueError('Cross-filesystem publication changed bytes')
                # Keep the guarded scratch copy until the writer has registered
                # it and can issue ordinary retention receipts after commit.
            with self.writer.transaction():
                if stage.name == "download" and not aggregate and result.status == "ok":
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
                if cross_filesystem:
                    scratch_files=[]
                    for record in records:
                        scratch_files.append(self.writer.artifact(run,stage.name,'scratch_copy',(work/record['role']).as_uri(),
                            version_id=version,producer_attempt_id=attempt,sha256=record['sha256'],bytes=record['bytes']))
            if cross_filesystem:
                self.pending_scratch.append((run,attempt,work,scratch_files,runtime_settings))
                if not self.writer.connection.in_transaction:self.collect_scratch()
            return result, destination
        return request, complete

    def collect_scratch(self):
        from .retention import RetentionController
        if self.writer.connection.in_transaction:return
        while self.pending_scratch:
            run,attempt,work,files,settings=self.pending_scratch.pop(0)
            controller=RetentionController(self.writer,self.root,run,settings)
            rows=[self.writer.connection.execute('SELECT * FROM files WHERE file_id=?',(file,)).fetchone() for file in files]
            consumer=controller.seal(rows)
            for file in files:controller.declare(file,[consumer],superseded_by=attempt)
            controller.delete_many(files)
            for directory in sorted((path for path in work.rglob('*') if path.is_dir()),key=lambda path:len(path.parts),reverse=True):
                try:directory.rmdir()
                except OSError:pass
            try:work.rmdir()
            except OSError:pass
