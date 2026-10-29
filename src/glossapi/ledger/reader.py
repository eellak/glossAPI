"""Ledger-only source/version/artifact projections and versioned CLI objects.

No payload opens, directory walks, backend imports, or writes occur here.
Shadow and legacy observations never become contractual cache hits.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .catalogue import STAGES, STATUSES
from .encoding import canonical_json, snapshot_hash, unpack_evidence
from .identity import source_key
from .stores import membership_values
from .writer import LedgerError, new_id

STAGE_ORDER = ("download", "extract", "clean", "ocr", "math", "sections", "annotate", "export")


class ReadRequestError(LedgerError):
    def __init__(self, code, message, *, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def _rows(connection, sql, parameters=()):
    cursor = connection.execute(sql, parameters)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def snapshot(connection, digest):
    row = connection.execute("SELECT json FROM snapshots WHERE sha256=?", (digest,)).fetchone()
    if row is None:
        raise LedgerError("Missing evidence snapshot: " + digest)
    value = unpack_evidence(json.loads(row[0]))
    for field, reference in value.pop("shared_fields", {}).items():
        value[field] = membership_values(connection, reference)
    return value


def evidence(connection, event):
    details = json.loads(event["details_json"])
    reference = details.get("evidence_sha256")
    return {**details, **(snapshot(connection, reference) if reference else {})}


def reason_view(event, details):
    return dict(event_id=event["event_id"], reason_code=event["reason_code"], message=event["message"],
                error_type=event["error_type"], traceback=event["traceback"], status=event["status"], details=details)


def source_view(connection, document_id):
    records = _rows(connection, "SELECT * FROM source_records WHERE document_id=? ORDER BY rowid", (document_id,))
    for record in records:
        record["metadata"] = json.loads(record.pop("metadata_json"))
    return records


def version_view(connection, document_id):
    return _rows(connection, "SELECT * FROM document_versions WHERE document_id=? ORDER BY revision", (document_id,))


def artifact_view(connection, version_ids=()):
    files = _rows(connection, "SELECT * FROM current_file_state ORDER BY created_at,file_id")
    return [file for file in files if file["version_id"] in version_ids] if version_ids else files


def resolve_document(connection, selector):
    documents = _rows(connection, "SELECT * FROM documents ORDER BY created_at,document_id")
    exact = {d["document_id"] for d in documents if selector in (d["document_id"], d["source_key"])}
    strong_selector = bool(exact)
    if not exact:
        for record in _rows(connection, "SELECT document_id,source_uri FROM source_records"):
            if selector in (record["source_uri"], _path(record["source_uri"])):
                exact.add(record["document_id"])
        for file in _rows(connection, "SELECT v.document_id,f.uri FROM files f JOIN document_versions v ON v.version_id=f.version_id"):
            if selector in (file["uri"], _path(file["uri"])):
                exact.add(file["document_id"])
    if not strong_selector and urlsplit(selector).scheme.lower() in ("http", "https"):
        normalized = source_key(selector)
        exact.update(d["document_id"] for d in documents if d["source_key"] == normalized)
        for record in _rows(connection, "SELECT document_id,source_uri FROM source_records"):
            if urlsplit(record["source_uri"]).scheme.lower() in ("http", "https") and source_key(record["source_uri"]) == normalized:
                exact.add(record["document_id"])
    if not exact:
        # A filename is a convenience selector only. It never joins identities
        # or establishes artifact ownership; all candidates must be returned.
        for record in _rows(connection, "SELECT document_id,source_uri FROM source_records"):
            if Path(_path(record["source_uri"])).name == selector:
                exact.add(record["document_id"])
        for file in _rows(connection, "SELECT v.document_id,f.uri FROM files f JOIN document_versions v ON v.version_id=f.version_id"):
            if Path(_path(file["uri"])).name == selector:
                exact.add(file["document_id"])
    if len(exact) > 1:
        raise ReadRequestError("ambiguous_file", "Selector matches multiple documents", details={"candidate_document_ids": sorted(exact)})
    if not exact:
        raise ReadRequestError("file_not_found", "No recorded document matches selector", details={"selector": selector})
    return next(d for d in documents if d["document_id"] in exact)


def _path(uri):
    return unquote(urlsplit(uri).path) if uri.startswith("file:") else uri


def select_run(connection, run_id=None):
    rows = _rows(connection, "SELECT rowid AS sequence,* FROM runs ORDER BY rowid DESC")
    if run_id is not None:
        rows = [r for r in rows if r["run_id"] == run_id]
        if not rows:
            raise ReadRequestError("run_not_found", "Unknown run ID", details={"run_id": run_id})
    return rows[0] if rows else None


def _history(connection, run):
    return _rows(connection, "SELECT e.*,t.stage,t.operation FROM events e LEFT JOIN tasks t ON t.task_id=e.task_id WHERE e.run_id IN (SELECT run_id FROM runs WHERE rowid<=?) ORDER BY e.event_id", (run["sequence"],)) if run else []


def report_why(connection, selector, *, run=None, stage=None):
    document = resolve_document(connection, selector)
    versions = version_view(connection, document["document_id"])
    history = _history(connection, run)
    known_versions = {e["version_id"] for e in history if e["version_id"]}
    cutoff = max((e["recorded_at"] for e in history), default=(run or {}).get("created_at", ""))
    versions = [v for v in versions if v["version_id"] in known_versions or v["created_at"] <= cutoff]
    version = versions[-1] if versions else None
    version_ids = {v["version_id"] for v in versions}
    records = source_view(connection, document["document_id"])
    run_ids = {r[0] for r in connection.execute("SELECT run_id FROM runs WHERE rowid<=?", (run["sequence"],))} if run else set()
    records = [r for r in records if r["run_id"] in run_ids]
    owned_files = [f for f in artifact_view(connection, version_ids) if f["version_id"] in version_ids]
    locations = {r["source_uri"] for r in records} | {f["uri"] for f in owned_files}
    locations |= {_path(uri) for uri in locations}
    stages, conflicts, unknowns, inclusion = [], [], [], []
    relevant_events = []
    for event in history:
        details = evidence(connection, event)
        exact_owner = event["version_id"] in version_ids if event["version_id"] else False
        path_owner = isinstance(details.get("path"), str) and details["path"] in locations
        if exact_owner or path_owner:
            relevant_events.append((event, details))
        if event["reason_code"] == "legacy_conflict":
            # Unresolved conflicts have no document owner. Expose the scope
            # honestly so the report cannot hide ambiguous corpus evidence.
            conflicts.append(dict(event_id=event["event_id"], scope="corpus_observation", **details))
    tasks = _rows(connection, "SELECT * FROM tasks WHERE run_id IN (SELECT run_id FROM runs WHERE rowid<=?) ORDER BY rowid", (run["sequence"],)) if run else []
    for task in tasks:
        if task["version_id"] not in version_ids or (stage and task["stage"] != stage):
            continue
        events = [(e, d) for e, d in relevant_events if e["task_id"] == task["task_id"]]
        # Decision hooks belong to the wrapper task but carry exact input
        # locations. Attach only to the explicitly referenced wrapper attempt.
        wrapper_ids = {d.get("wrapper_attempt_id") for e, d in events}
        decisions = [(e, d) for e, d in relevant_events if e["event_kind"] == "decision" and e["attempt_id"] in wrapper_ids]
        lifecycle = [e for e, _ in events if e["event_kind"] == "lifecycle"]
        state = lifecycle[-1] if lifecycle else {}
        attempts = []
        for attempt in _rows(connection, "SELECT * FROM attempts WHERE task_id=? ORDER BY attempt_number", (task["task_id"],)):
            attempt_events = [(e, d) for e, d in events if e["attempt_id"] == attempt["attempt_id"]]
            terminal = next((e for e, _ in reversed(attempt_events) if e["event_kind"] == "lifecycle"), {})
            outputs = snapshot(connection, terminal["output_manifest_sha256"]).get("files", []) if terminal.get("output_manifest_sha256") else []
            inputs = _rows(connection, "SELECT file_id,role,sha256 FROM attempt_inputs WHERE attempt_id=? ORDER BY role,file_id", (attempt["attempt_id"],))
            attempts.append(dict(attempt_id=attempt["attempt_id"], attempt_number=attempt["attempt_number"],
                input_hashes=inputs, output_hashes=outputs, settings_hash=attempt["settings_sha256"], tools_hash=attempt["tools_sha256"],
                started_at=terminal.get("started_at"), finished_at=terminal.get("finished_at"), duration_ns=terminal.get("duration_ns"),
                error=dict(type=terminal.get("error_type"), message=terminal.get("message"), traceback=terminal.get("traceback")) if terminal.get("error_type") or terminal.get("status") == "failed" else None,
                decisions=[reason_view(e, d) for e, d in attempt_events + decisions if e["event_kind"] == "decision"]))
        reasons = [reason_view(e, d)
                   for e, d in events + decisions if e["event_kind"] in ("decision", "lifecycle")]
        stages.append(dict(run_id=task["run_id"], version_id=task["version_id"], stage=task["stage"], operation=task["operation"],
                           status=state.get("status"), reasons=reasons, attempts=attempts))
        if state.get("status") == "failed" and any(d.get("backend_status") == "ok" for _, d in decisions):
            conflicts.append(dict(code="shadow_verdict_conflicts_with_success_observation", task_id=task["task_id"],
                                  event_ids=[e["event_id"] for e, _ in decisions], scope="document"))
    files = []
    for file in owned_files:
        facts = [e for e in history if e["file_id"] == file["file_id"]]
        if not facts:
            continue
        presence = [e for e in facts if e["event_kind"] == "artifact"]
        retention = [dict(event_id=e["event_id"], reason_code=e["reason_code"], details=evidence(connection, e)) for e in facts if e["event_kind"] == "retention"]
        imported_evidence = []
        for event in facts:
            details = evidence(connection, event)
            if details.get("inventory_sha256") and details.get("path"):
                inventory = snapshot(connection, details["inventory_sha256"])
                imported_evidence.extend(entry for entry in inventory.get("files", []) if entry["path"] == details["path"])
        files.append(dict(file_id=file["file_id"], version_id=file["version_id"], kind=file["kind"], path=_path(file["uri"]),
            sha256=file["sha256"], bytes=file["bytes"], presence=presence[-1]["reason_code"] if presence else None,
            ownership=file["ownership"], page_first=file["page_first"], page_last=file["page_last"],
            legacy_evidence=imported_evidence,
            retention=dict(policy="legacy_unknown" if file["ownership"] != "managed" else "shadow_observation", consumer_ok_event_ids=[], outstanding_consumers=[], pins=[], deletions=retention)))
    for event, details in relevant_events:
        if event["reason_code"].startswith("export_"):
            inclusion.append(dict(product=details.get("product", details.get("path")), decision=event["reason_code"], reason_code=event["reason_code"],
                                  record_hash=details.get("record_sha256"), manifest_hash=event["output_manifest_sha256"], event_id=event["event_id"]))
    observations = [dict(event_id=e["event_id"], reason_code=e["reason_code"], details=d) for e, d in relevant_events if e["event_kind"] == "import"]
    imported_stages = {}
    for event, details in relevant_events:
        if event["event_kind"] == "import" and details.get("stage") and (not stage or details["stage"] == stage):
            imported_stages.setdefault((event["run_id"], event["version_id"], details["stage"]), []).append(
                dict(event_id=event["event_id"], reason_code=event["reason_code"], message=event["message"], details=details))
    stages.extend(dict(run_id=r, version_id=v, stage=s, operation="legacy.import", status=None, reasons=reasons, attempts=[])
                  for (r, v, s), reasons in imported_stages.items())
    for field in ("producer_input_settings_bindings", "unreported_model_identity", "per_document_timings", "product_membership_when_unbound", "retention_consumers_and_pins_when_unreported"):
        unknowns.append(dict(code="unknown_not_recorded", field=field))
    if not version or version["source_sha256"] is None:
        unknowns.append(dict(code="legacy_unknown", field="source_sha256", explanation="Original acquired entity bytes were not observed"))
    return dict(document=dict(**document, version_id=version["version_id"] if version else None, versions=versions, source_records=records),
                stages=stages, files=files, inclusion=inclusion, conflicts=conflicts, unknowns=unknowns, observations=observations)


def status_view(connection, *, run=None, stage=None, selector=None):
    document = resolve_document(connection, selector) if selector else None
    rows = _rows(connection, "SELECT s.*,v.document_id FROM current_task_state s LEFT JOIN document_versions v ON v.version_id=s.version_id WHERE s.run_id=? ORDER BY s.ordinal,s.task_id", (run["run_id"],)) if run else []
    rows = [r for r in rows if (not stage or r["stage"] == stage) and (not document or r["document_id"] == document["document_id"])]
    counts = dict.fromkeys(sorted(STATUSES), 0)
    documents = []
    for row in rows:
        if row["status"]:
            counts[row["status"]] += 1
        terminal = _rows(connection, "SELECT * FROM events WHERE event_id=?", (row["event_id"],)) if row["event_id"] else []
        event = terminal[0] if terminal else {}
        attempts = _rows(connection, "SELECT * FROM attempts WHERE task_id=? ORDER BY attempt_number DESC LIMIT 1", (row["task_id"],))
        attempt = attempts[0] if attempts else {}
        heartbeat = connection.execute("SELECT MAX(recorded_at) FROM events WHERE task_id=? AND event_kind='heartbeat'", (row["task_id"],)).fetchone()[0]
        documents.append(dict(document_id=row["document_id"], version_id=row["version_id"], stage=row["stage"], operation=row["operation"],
            status=row["status"], reason_code=row["reason_code"], attempt_number=attempt.get("attempt_number"), worker_id=attempt.get("worker_id"),
            device_id=attempt.get("device_id"), started_at=event.get("started_at"), elapsed_ns=event.get("duration_ns"), last_heartbeat_at=heartbeat,
            heartbeat_age_ns=None, stale_reason="shadow_has_no_executable_leases"))
    state = "completed_with_failures" if counts["failed"] else "running" if counts["running"] else "waiting" if counts["planned"] else "completed"
    last_event = connection.execute("SELECT MAX(event_id) FROM events WHERE run_id=?", (run["run_id"],)).fetchone()[0] if run else None
    present = connection.execute("SELECT SUM(bytes) FROM current_file_state WHERE presence='present'").fetchone()[0]
    return dict(state=state, counts=counts, documents=documents, queues=[], barriers=[], resources=dict(
        disk_budget_bytes=None, present_bytes=present or 0, reserved_bytes=None, free_bytes=None), last_event_id=last_event,
        unknowns=["runtime_queues_resources_not_recorded_in_shadow"])


def report_view(connection, *, run=None, stage=None):
    status = status_view(connection, run=run, stage=stage)
    conflicts = [dict(event_id=e["event_id"], **evidence(connection, e)) for e in _history(connection, run) if e["reason_code"] == "legacy_conflict"]
    rows = status["documents"]
    for row in rows:
        row["source_key"] = connection.execute("SELECT source_key FROM documents WHERE document_id=?", (row["document_id"],)).fetchone()[0] if row["document_id"] else None
        row["scope"] = "document" if row["document_id"] else "corpus"
        task = connection.execute("SELECT task_id FROM tasks WHERE run_id=? AND version_id IS ? AND stage=? AND operation=?",
                                  (run["run_id"], row["version_id"], row["stage"], row["operation"])).fetchone()
        events = _rows(connection, "SELECT * FROM events WHERE task_id=? ORDER BY event_id", (task[0],))
        row["reasons"] = [dict(event_id=e["event_id"], reason_code=e["reason_code"], status=e["status"],
                               error_type=e["error_type"], message=e["message"], traceback=e["traceback"], details=evidence(connection, e))
                          for e in events if e["event_kind"] in ("decision", "lifecycle")]
        row["attempts"] = _rows(connection, "SELECT attempt_id,attempt_number,input_manifest_sha256,settings_sha256,tools_sha256 FROM attempts WHERE task_id=? ORDER BY attempt_number", (task[0],))
    imported = {}
    for event in _history(connection, run):
        if event["run_id"] != (run or {}).get("run_id") or event["event_kind"] != "import" or not event["version_id"]:
            continue
        details = evidence(connection, event)
        observed_stage = details.get("stage")
        if not observed_stage or (stage and stage != observed_stage):
            continue
        imported.setdefault((event["version_id"], observed_stage), []).append(dict(event_id=event["event_id"], reason_code=event["reason_code"], details=details))
    for (version, observed_stage), reasons in imported.items():
        document = connection.execute("SELECT d.document_id,d.source_key FROM documents d JOIN document_versions v USING(document_id) WHERE v.version_id=?", (version,)).fetchone()
        rows.append(dict(document_id=document[0], source_key=document[1], version_id=version, stage=observed_stage,
                         operation="legacy.import", status=None, reason_code="legacy_unknown", reasons=reasons, scope="document"))
    return dict(rows=rows, retention=dict(present_bytes=status["resources"]["present_bytes"], deleted_logical_bytes=None),
                resources=status["resources"], counts=status["counts"], conflicts=conflicts)


def plan_view(connection=None, *, selectors=(), stages=None, settings=None, devices=None, workers=None, manifest=None):
    settings = {"debug": "off", "memory_saving": "normal", **(settings or {})}
    requested = stages or ["extract", "clean", "sections", "export"]
    if any(s not in STAGES for s in requested):
        raise ReadRequestError("invalid_stage", "Unknown stage")
    if settings.get("debug") not in ("on", "off") or settings.get("memory_saving") not in ("normal", "extreme"):
        raise ReadRequestError("invalid_settings", "Invalid debug or memory_saving value")
    prerequisites = dict(download=[], extract=[], clean=["extract"], ocr=["clean"], math=["extract"], sections=["clean"], annotate=["sections"], export=["clean"])
    selected_stages = set()
    def include(stage):
        selected_stages.add(stage)
        for dependency in prerequisites[stage]:
            include(dependency)
    for stage in requested:
        include(stage)
    if settings.get("memory_saving") == "extreme" and selected_stages & {"ocr", "math"}:
        raise ReadRequestError("policy_conflict", "Extreme source retention awaits the last-PDF-consumer policy decision (L1-G01)")
    if not selectors and connection:
        selectors = [r[0] for r in connection.execute("SELECT document_id FROM documents ORDER BY rowid")]
    documents, inventory, seen = [], [], set()
    pooled_stages = set(selected_stages)
    for selector in selectors:
        known = None
        if connection:
            try:
                known = resolve_document(connection, selector)
            except ReadRequestError as error:
                if error.code != "file_not_found":
                    raise
        if known:
            identifier, key = known["document_id"], known["source_key"]
            versions = version_view(connection, identifier)
            version = versions[-1] if versions else None
        else:
            parts = urlsplit(selector)
            if parts.scheme not in ("http", "https", "file") and not Path(selector).is_absolute():
                raise ReadRequestError("file_not_found", "A new local source must be an explicit absolute path", details={"selector": selector})
            identifier, key, version = new_id(), source_key(selector), None
        if key in seen:
            continue
        seen.add(key)
        inventory.append(dict(source_key=key, version=version))
        document_stages = set(selected_stages)
        remote_source = key.startswith("url:v1:")
        if known and not remote_source:
            remote_source = any(r[0] == "url" for r in connection.execute("SELECT source_kind FROM source_records WHERE document_id=?", (identifier,)))
        if remote_source and (version is None or version["stored_sha256"] is None) and "extract" in document_stages:
            document_stages.add("download")
        pooled_stages.update(document_stages)
        steps = []
        for stage in STAGE_ORDER:
            if stage not in document_stages:
                continue
            dependencies = ["download"] if stage == "extract" and "download" in document_stages else prerequisites[stage]
            package = {"download": "download", "annotate": "annotate", "ocr": "ocr_plugin", "math": "layout"}.get(stage, "core")
            steps.append(dict(stage=stage, operation=stage+".planned", action="conditional", reason_code="legacy_unknown" if known else "stage_planned",
                input_roles=[dict(producer=s, role="stage_input") for s in dependencies],
                conditional_on=dict(predicate="selected_adapters_and_verified_inputs_available", input_producer=dependencies, unresolved_fields=["quality", "route", "settings_tool_fingerprint"]),
                packages=[package], device_id=(devices or {}).get(stage, "cpu")))
        documents.append(dict(document_id=identifier, source_key=key, version_id=version["version_id"] if version else None, steps=steps))
    source_inventory_hash = snapshot_hash(dict(records=inventory, manifest=manifest))
    present = connection.execute("SELECT COALESCE(SUM(bytes),0) FROM current_file_state WHERE presence='present'").fetchone()[0] if connection else 0
    pools = [dict(stage=s, device_id=(devices or {}).get(s, "cpu"), workers=(workers or {}).get(s, 1), queue_capacity=max(2, 2*(workers or {}).get(s, 1)), cpu_tokens=1) for s in STAGE_ORDER if s in pooled_stages]
    barriers = [dict(name=name, scope="corpus", waiting_for="sealed_selection", remaining=len(documents))
                for name, stage in (("source_manifest", None), ("sections_selection_ids", "sections"), ("annotation_batch", "annotate"), ("export_shards", "export"))
                if stage is None or stage in pooled_stages]
    body = dict(source_inventory_hash=source_inventory_hash, settings=settings, documents=documents, barriers=barriers, pools=pools,
        resource_limits={k: settings.get(k) for k in ("disk_budget_bytes", "max_source_bytes", "max_attempt_bytes")},
        disk=dict(present_bytes=present, pinned_bytes=None, reserved_bytes=None, estimated_peak_bytes=None, unknown_sizes=[dict(document_id=d["document_id"], reason_code="unknown_not_recorded") for d in documents]),
        requirements=[dict(package_group=p, distributions=[], entry_point="glossapi.ocr" if p == "ocr_plugin" else None, installed=None, reason_code="capabilities_not_recorded") for p in sorted({p for d in documents for s in d["steps"] for p in s["packages"]})],
        legacy_import_required=connection is None or connection.execute("SELECT 1 FROM events WHERE reason_code='legacy_import_completed' LIMIT 1").fetchone() is None,
        execution_available=False, compatibility_profile="legacy_shadow", explicit_manifest=manifest)
    return dict(body, plan_hash=snapshot_hash(body))
