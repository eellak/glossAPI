"""Ledger-only source/version/artifact projections and versioned CLI objects.

No payload opens, directory walks, backend imports, or writes occur here.
Imported legacy observations never become contractual cache hits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .catalogue import STAGES, STATUSES
from .encoding import canonical_json, snapshot_hash, unpack_evidence
from .identity import source_key
from .urls import scrub, source_uri_of_key
from .stores import membership_values
from .writer import LedgerError, new_id
from ..stages.retention import retention_facts, retention_summary

STAGE_ORDER = ("download", "extract", "clean", "ocr", "math", "sections", "annotate", "export")
# Stages `glossapi run` cannot execute in its streaming document DAG, with the
# reason the plan reports. Real math inference is unqualified (LS4-G03).
RUN_UNAVAILABLE = {"math": "math_backend_unavailable"}


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
    if value.get('kind')=='outputs' and value.get('encoding')=='registered_files_v1':
        files=[]
        for reference in value['files']:
            rows=_rows(connection,'SELECT * FROM files WHERE file_id=?',(reference['file_id'],))
            if len(rows)!=1:raise LedgerError('Missing registered output binding')
            entry={'file_id':reference['file_id']}
            for key in reference['fields']:
                if key=='presence':
                    # The output manifest binds initial registration, not a
                    # later deletion/missing observation at the same location.
                    original=connection.execute("SELECT reason_code FROM events WHERE file_id=? AND event_kind='artifact' ORDER BY event_id LIMIT 1",(reference['file_id'],)).fetchone()
                    if original is None:raise LedgerError('Missing initial output presence')
                    entry[key]=original[0]
                elif key in rows[0]:entry[key]=rows[0][key]
                else:raise LedgerError('Unknown registered output field: '+key)
            if 'role' in reference:entry['role']=reference['role']
            files.append(entry)
        value={key:entry for key,entry in value.items() if key not in ('encoding','files')}
        value['files']=files
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
        # A raw URL resolves through its resource identity (keyed reference
        # under the corpus key, read only); a persisted spelling may name
        # several documents (different credentials) and is then ambiguous.
        keyed = _selector_identity(connection, selector)
        exact.update(d["document_id"] for d in documents if keyed is not None and d["source_key"] == keyed)
        spelled = _persisted_spelling(selector)
        keys = {d["document_id"]: d["source_key"] for d in documents}
        spelled_matches = {record["document_id"] for record in _rows(connection, "SELECT document_id,source_uri FROM source_records")
                           if urlsplit(record["source_uri"]).scheme.lower() in ("http", "https") and _persisted_spelling(record["source_uri"]) == spelled}
        spelled_matches |= {d["document_id"] for d in documents if source_uri_of_key(d["source_key"]) is not None
                            and urlsplit(source_uri_of_key(d["source_key"])).scheme.lower() in ("http", "https")
                            and _persisted_spelling(source_uri_of_key(d["source_key"])) == spelled}
        # A document keyed by a different keyed reference provably comes from a
        # different raw URL (other credentials); exclude it only when the
        # selector's own keyed identity is known. Declared-ID and legacy
        # records with the same spelling stay explicit candidates.
        exact.update(identifier for identifier in spelled_matches
                     if keyed is None or not str(keys.get(identifier, "")).startswith("url:v2:") or keys.get(identifier) == keyed)
    filename_selector = False
    if not exact:
        # A filename is a convenience selector only. It never joins identities
        # or establishes artifact ownership; all candidates must be returned.
        filename_selector = True
        for record in _rows(connection, "SELECT document_id,source_uri FROM source_records"):
            if Path(_path(record["source_uri"])).name == selector:
                exact.add(record["document_id"])
        for file in _rows(connection, "SELECT v.document_id,f.uri FROM files f JOIN document_versions v ON v.version_id=f.version_id"):
            if Path(_path(file["uri"])).name == selector:
                exact.add(file["document_id"])
    if len(exact) > 1 and filename_selector:
        aliases = _same_content_aliases(connection, documents, exact)
        if aliases:
            return aliases
    if len(exact) > 1:
        raise ReadRequestError("ambiguous_file", "Selector matches multiple documents", details={"candidate_document_ids": sorted(exact)})
    if not exact:
        raise ReadRequestError("file_not_found", "No recorded document matches selector", details={"selector": selector})
    return next(d for d in documents if d["document_id"] in exact)


def _persisted_spelling(url):
    """Persisted URL form with scheme and host case-folded, fragment dropped."""
    from .urls import persisted_url
    parts = urlsplit(persisted_url(url.split("#", 1)[0]))
    host = parts.netloc.rpartition("@")[2].lower()
    return (parts.scheme.lower(), host, parts.path, parts.query, "?" in persisted_url(url.split("#", 1)[0]))


def _ledger_root(connection):
    try:
        for row in connection.execute("PRAGMA database_list"):
            if row[1] == "main" and row[2]:
                return Path(row[2]).parent
    except Exception:
        return None
    return None


def _selector_identity(connection, selector):
    """Resource identity of a raw URL selector without writing anything."""
    from .urls import keyed_source_key, lossless_url
    if lossless_url(selector):
        return source_key(selector)
    root = _ledger_root(connection)
    if root is None:
        return None
    try:
        return keyed_source_key(selector, root=root, create=False)
    except LedgerError:
        return None


def _same_content_aliases(connection, documents, candidates):
    """One document for a bare filename whose candidates are the same bytes (FG-G04).

    The Corpus API registers one supplied file under its input path and its
    ``downloads/`` copy. When every candidate has exactly one recorded source
    hash and the hashes are equal, the filename names one content: return the
    earliest candidate with every alias listed. Unknown or differing hashes stay
    ambiguous; identities are never joined in the ledger itself.
    """
    hashes = {}
    for document_id in candidates:
        values = {row[0] for row in connection.execute(
            "SELECT source_sha256 FROM document_versions WHERE document_id=?", (document_id,))}
        if len(values) != 1 or None in values:
            return None
        hashes[document_id] = values.pop()
    if len(set(hashes.values())) != 1:
        return None
    ordered = [d for d in documents if d["document_id"] in candidates]
    primary, others = ordered[0], ordered[1:]
    return dict(primary, alias_basis="same_source_sha256", source_sha256=hashes[primary["document_id"]],
                aliases=[dict(document_id=d["document_id"], source_key=d["source_key"]) for d in others])


def _document_ids(document):
    return [document["document_id"], *[alias["document_id"] for alias in document.get("aliases", [])]]


def _path(uri):
    return unquote(urlsplit(uri).path) if uri.startswith("file:") else uri


def child_runs(connection):
    """Runs a parent plan created for its compatibility publication barriers."""
    children = {}
    for parent, encoded in connection.execute("SELECT run_id,details_json FROM events WHERE event_kind='run' AND message LIKE 'wrapper:%'"):
        child = json.loads(encoded).get("wrapper_run_id")
        if child:
            children[child] = parent
    return children


def select_run(connection, run_id=None):
    """Explicit run, else the latest run the user started.

    A streaming ``run`` creates one child run per compatibility wrapper after
    its own run; those children are reached through the parent, never chosen
    as "latest" (LS6-G01/G02). A wrapper the user called directly has no
    parent link and stays selectable.
    """
    rows = _rows(connection, "SELECT rowid AS sequence,* FROM runs ORDER BY rowid DESC")
    if run_id is not None:
        rows = [r for r in rows if r["run_id"] == run_id]
        if not rows:
            raise ReadRequestError("run_not_found", "Unknown run ID", details={"run_id": run_id})
        return rows[0]
    children = child_runs(connection)
    return next((row for row in rows if row["run_id"] not in children), rows[0] if rows else None)


def _run_scope(connection,run):
    if not run:return set()
    identifiers={row[0] for row in connection.execute('SELECT run_id FROM runs WHERE rowid<=?',(run['sequence'],))}
    # Child publication runs are explicitly linked by the parent plan. Their
    # facts remain part of that plan even though their creation sequence is later.
    for encoded, in connection.execute("SELECT details_json FROM events WHERE run_id=? AND message LIKE 'wrapper:%'",(run['run_id'],)):
        child=json.loads(encoded).get('wrapper_run_id')
        if child:identifiers.add(child)
    return identifiers


def _history(connection, run):
    identifiers=_run_scope(connection,run)
    if not identifiers:return []
    parameters=tuple(sorted(identifiers))
    return _rows(connection,'SELECT e.*,t.stage,t.operation FROM events e LEFT JOIN tasks t ON t.task_id=e.task_id WHERE e.run_id IN ('+','.join('?' for _ in parameters)+') ORDER BY e.event_id',parameters)


def report_why(connection, selector, *, run=None, stage=None):
    document = resolve_document(connection, selector)
    # Same-content aliases (FG-G04) report as one document over all their versions.
    document_ids = _document_ids(document)
    versions = [v for identifier in document_ids for v in version_view(connection, identifier)]
    if len(document_ids) > 1:
        versions.sort(key=lambda v: (v["created_at"], v["document_id"] != document["document_id"], v["revision"]))
    history = _history(connection, run)
    known_versions = {e["version_id"] for e in history if e["version_id"]}
    cutoff = max((e["recorded_at"] for e in history), default=(run or {}).get("created_at", ""))
    versions = [v for v in versions if v["version_id"] in known_versions or v["created_at"] <= cutoff]
    version = versions[-1] if versions else None
    version_ids = {v["version_id"] for v in versions}
    records = [record for identifier in document_ids for record in source_view(connection, identifier)]
    run_ids = _run_scope(connection,run)
    records = [r for r in records if r["run_id"] in run_ids]
    owned_files = [f for f in artifact_view(connection, version_ids) if f["version_id"] in version_ids]
    locations = {r["source_uri"] for r in records} | {f["uri"] for f in owned_files}
    locations |= {_path(uri) for uri in locations}
    stages, conflicts, unknowns, inclusion = [], [], [], []
    tasks = [task for task in _rows(connection,'SELECT * FROM tasks ORDER BY rowid') if task['run_id'] in run_ids]
    document_file_ids={file['file_id'] for file in owned_files}
    bound_tasks={task['task_id'] for task in tasks if task['version_id'] in version_ids}
    for task_id,file_id in connection.execute('SELECT a.task_id,i.file_id FROM attempt_inputs i JOIN attempts a USING(attempt_id)'):
        if file_id in document_file_ids:bound_tasks.add(task_id)
    dependencies=list(connection.execute('SELECT task_id,producer_task_id FROM task_dependencies'))
    while True:
        expanded=bound_tasks|{task for task,producer in dependencies if producer in bound_tasks}
        if expanded==bound_tasks:break
        bound_tasks=expanded
    decoded={event['event_id']:evidence(connection,event) for event in history}
    section_keys={key for event in history if event['version_id'] in version_ids for key in decoded[event['event_id']].get('section_keys',[])}
    relevant_events = []
    for event in history:
        details = decoded[event['event_id']]
        if event['version_id'] is None and event['task_id'] in bound_tasks:
            details={**details,'scope':'corpus_operation_with_bound_document_inputs'}
            if 'lineage' in details:details['lineage']=[item for item in details['lineage'] if item.get('section_key') in section_keys]
            if 'selected_versions' in details:
                details['selected_versions']=[item for item in details['selected_versions'] if item.get('version_id') in version_ids]
                stems={item.get('compatibility_stem') for item in details['selected_versions']}
                if 'membership' in details:details['membership']=[{**shard,'records':[item for item in shard['records'] if item['filename'] in stems]} for shard in details['membership']]
        exact_owner = event["version_id"] in version_ids if event["version_id"] else False
        path_owner = isinstance(details.get("path"), str) and details["path"] in locations
        if exact_owner or path_owner or event['task_id'] in bound_tasks:
            relevant_events.append((event, details))
        if event["reason_code"] == "legacy_conflict":
            # Unresolved conflicts have no document owner. Expose the scope
            # honestly so the report cannot hide ambiguous corpus evidence.
            conflicts.append(dict(event_id=event["event_id"], scope="corpus_observation", **details))
    for task in tasks:
        if task["task_id"] not in bound_tasks or (stage and task["stage"] != stage):
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
            error = dict(type=terminal.get("error_type"), message=terminal.get("message"), traceback=terminal.get("traceback"),
                basis="attempt_terminal_event") if terminal.get("error_type") or terminal.get("status") == "failed" else None
            if error and not error["type"]:
                # Some backends return a failed verdict and report the exception
                # only as a structured observation; surface that recorded cause.
                cause = next((item for e, d in attempt_events if e["event_kind"] == "decision"
                              for item in d.get("backend_observations", []) if isinstance(item, dict) and item.get("error_type")), None)
                if cause:
                    error = dict(type=cause["error_type"], message=cause.get("message"), traceback=cause.get("traceback"),
                                 reason_code=cause.get("reason_code"), basis="recorded_backend_observation")
            attempts.append(dict(attempt_id=attempt["attempt_id"], attempt_number=attempt["attempt_number"],
                input_hashes=inputs, output_hashes=outputs, settings_hash=attempt["settings_sha256"], tools_hash=attempt["tools_sha256"],
                started_at=terminal.get("started_at"), finished_at=terminal.get("finished_at"), duration_ns=terminal.get("duration_ns"),
                error=error, decisions=[reason_view(e, d) for e, d in attempt_events + decisions if e["event_kind"] == "decision"]))
        reasons = [reason_view(e, d)
                   for e, d in events + decisions if e["event_kind"] in ("decision", "lifecycle")]
        stages.append(dict(run_id=task["run_id"], version_id=task["version_id"], stage=task["stage"], operation=task["operation"],
                           status=state.get("status"), reasons=reasons, attempts=attempts,scope="document" if task["version_id"] else "corpus"))
        if state.get("status") == "failed" and any(d.get("backend_status") == "ok" for _, d in decisions):
            conflicts.append(dict(code="failed_verdict_conflicts_with_success_observation", task_id=task["task_id"],
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
            retention=retention_facts(connection,file['file_id'],as_of=max((event['event_id'] for event in history),default=0))))
    inclusion_scope = []  # (run_id, version_id) of each inclusion item, for per-occurrence dispositions
    for event, details in relevant_events:
        if details.get('selected_versions'):
            record_hashes={record['filename']:record['record_sha256'] for shard in details.get('membership',[]) for record in shard['records']}
            for selected in details['selected_versions']:
                if selected.get('version_id') in version_ids:
                    terminal=next((item for item in history if item['attempt_id']==event['attempt_id'] and item['event_kind']=='lifecycle' and item['status']=='ok'),{})
                    inclusion.append(dict(product='document_jsonl',decision='include' if selected['included'] else 'omit',
                        reason_code=selected['reason_code'],record_hash=record_hashes.get(selected.get('compatibility_stem')),
                        manifest_hash=terminal.get('output_manifest_sha256'),event_id=event['event_id']))
                    inclusion_scope.append((event['run_id'], selected.get('version_id')))
        elif details.get("product") == "annotated_sections" and event["event_kind"] == "decision":
            # The annotated-section export's own census (PLAN 2.3, W18 M4).
            inclusion.append(dict(product="annotated_sections", decision=details.get("decision"), reason_code=event["reason_code"],
                                  cause=details.get("cause"), section_count=details.get("section_count"),
                                  record_hash=None, manifest_hash=event["output_manifest_sha256"], event_id=event["event_id"]))
            inclusion_scope.append((event['run_id'], event['version_id']))
        elif event["reason_code"].startswith("export_"):
            inclusion.append(dict(product=details.get("product", details.get("path")), decision=event["reason_code"], reason_code=event["reason_code"],
                                  record_hash=details.get("record_sha256"), manifest_hash=event["output_manifest_sha256"], event_id=event["event_id"]))
            inclusion_scope.append((event['run_id'], event['version_id']))
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
    # Every requested source occurrence (for example each manifest row) with its
    # own persisted binding: the version that occurrence's inventory created
    # and the disposition of that version in the run that inventory fed (F4).
    bindings = {}
    for event in _rows(connection, "SELECT run_id,details_json FROM events WHERE reason_code='source_inventory_bound' ORDER BY event_id"):
        details = evidence(connection, event)
        for item in details.get("occurrences", []):
            bindings[item["source_record_id"]] = (item.get("version_id"), details.get("execution_run_id"))
    def execution_scope(run_id):
        children = {json.loads(encoded).get("wrapper_run_id") for encoded, in connection.execute(
            "SELECT details_json FROM events WHERE run_id=? AND message LIKE 'wrapper:%'", (run_id,))}
        return {run_id} | {child for child in children if child}
    def disposition_of(version_id, run_id) -> dict[str, Any]:
        scope = execution_scope(run_id)
        final = None
        for item, (event_run, event_version) in zip(inclusion, inclusion_scope):
            if item.get("product") == "document_jsonl" and event_version == version_id and event_run in scope:
                final = item
        sections = None
        for item, (event_run, event_version) in zip(inclusion, inclusion_scope):
            if item.get("product") == "annotated_sections" and event_version == version_id and event_run in scope:
                sections = item
        products = {"annotated_sections": dict(decision=sections["decision"], reason_code=sections["reason_code"], cause=sections.get("cause"),
                    section_count=sections.get("section_count"), event_id=sections["event_id"])} if sections else {}
        if final is not None:
            products["document_jsonl"] = dict(decision=final["decision"], reason_code=final["reason_code"], event_id=final["event_id"])
            return dict(decision=final["decision"], reason_code=final["reason_code"], event_id=final["event_id"], run_id=run_id, basis="export_inclusion",
                        **({"products": products} if sections else {}))
        terminal = [item for item in stages if item.get("scope") == "document" and item["status"]
                    and item["version_id"] == version_id and item["run_id"] in scope]
        failed = next((item for item in terminal if item["status"] in ("failed", "skipped")), None)
        return dict(decision="failed" if failed and failed["status"] == "failed" else failed["status"] if failed else
                    "completed" if terminal else None,
                    reason_code=(failed["reasons"][-1]["reason_code"] if failed and failed["reasons"] else None),
                    event_id=None, run_id=run_id, basis="document_stage_status", **({"products": products} if products else {}))
    occurrences = []
    for record in records:
        bound_version, execution_run = bindings.get(record["source_record_id"], (None, None))
        disposition = disposition_of(bound_version, execution_run) if execution_run else dict(
            decision=None, reason_code=None, event_id=None, run_id=None, basis="occurrence_binding_not_recorded")
        occurrences.append(dict(source_record_id=record["source_record_id"], run_id=record["run_id"], execution_run_id=execution_run,
                                source_uri=record["source_uri"], source_row=record["source_row"], url_index=record["url_index"],
                                manifest_sha256=record["manifest_sha256"], metadata=record["metadata"], document_id=record["document_id"],
                                version_id=bound_version, disposition=disposition))
    return dict(document=dict(**document, version_id=version["version_id"] if version else None, versions=versions, source_records=records),
                occurrences=occurrences, stages=stages, files=files, inclusion=inclusion, conflicts=conflicts, unknowns=unknowns, observations=observations,retention=retention_summary(connection,run))


_JSON_TAGS = frozenset(("null", "str", "bool", "int", "float", "mapping", "list"))


def _json_native(typed):
    if not isinstance(typed, dict) or typed.get("type") not in _JSON_TAGS:
        return False
    if typed["type"] == "mapping":
        return all(key.get("type") == "str" and _json_native(value) for key, value in typed["value"])
    if typed["type"] == "list":
        return all(_json_native(value) for value in typed["value"])
    return typed["type"] != "float" or float.fromhex(typed["value"]) == float.fromhex(typed["value"])  # NaN stays typed


def json_metadata(stored):
    """The caller's original metadata as plain JSON when every value is JSON-native, else the typed ledger form."""
    from .encoding import untyped_value
    fields = stored.get("fields") if isinstance(stored, dict) and stored.get("schema_version") == 1 else None
    if fields is not None and _json_native(fields):
        return untyped_value(fields), "json"
    return stored, "typed_v1"


def source_metadata(connection, document_ids, run):
    """Each document's source records (the caller's original metadata, verbatim) up to this run."""
    run_ids = _run_scope(connection, run)
    output = []
    for document_id in document_ids:
        for record in source_view(connection, document_id):
            if record["run_id"] in run_ids:
                metadata, encoding = json_metadata(record["metadata"])
                output.append(dict(document_id=document_id, source_record_id=record["source_record_id"], run_id=record["run_id"],
                                   source_uri=record["source_uri"], source_row=record["source_row"], metadata=metadata,
                                   metadata_encoding=encoding))
    return output


def _mark_interruption(connection: Any, run: Any, counts: Any, result: Any) -> None:
    """A run the filesystem stopped says so (W18 M10): exhausted disk, not a live or killed process."""
    if not run or not (counts["running"] or counts["planned"]):
        return
    latest = connection.execute("SELECT event_id,reason_code,message FROM events WHERE run_id=? AND event_kind='run' ORDER BY event_id DESC LIMIT 1",
                                (run["run_id"],)).fetchone()
    if latest is None or latest[1] != "disk_full":
        return
    result["state"] = "interrupted"
    result["interruption"] = dict(cause="disk_full", errno=28, event_id=latest[0], message=latest[2],
                                  action="free space (or enlarge the filesystem), then: glossapi run --resume " + str(run["run_id"]))


def status_view(connection, *, run=None, stage=None, selector=None):
    document = resolve_document(connection, selector) if selector else None
    rows = _rows(connection, "SELECT s.*,v.document_id FROM current_task_state s LEFT JOIN document_versions v ON v.version_id=s.version_id WHERE s.run_id=? ORDER BY s.ordinal,s.task_id", (run["run_id"],)) if run else []
    document_ids = set(_document_ids(document)) if document else set()
    rows = [r for r in rows if (not stage or r["stage"] == stage) and (not document or r["document_id"] in document_ids)]
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
            heartbeat_age_ns=None, stale_reason="no_streaming_leases_recorded_for_run"))
    state = "completed_with_failures" if counts["failed"] else "running" if counts["running"] else "waiting" if counts["planned"] else "completed"
    last_event = connection.execute("SELECT MAX(event_id) FROM events WHERE run_id=?", (run["run_id"],)).fetchone()[0] if run else None
    present = connection.execute("SELECT SUM(bytes) FROM current_file_state WHERE presence='present'").fetchone()[0]
    result: dict[str, Any] = dict(state=state, counts=counts, documents=documents, queues=[], barriers=[], resources=dict(
        disk_budget_bytes=None, present_bytes=present or 0, reserved_bytes=None, free_bytes=None), last_event_id=last_event,
        unknowns=["runtime_queues_resources_not_recorded_for_non_streaming_run"])
    if document:
        # status --file: the selected document's original metadata (K3 manifest round trip).
        result["sources"] = source_metadata(connection, _document_ids(document), run)
    plan = snapshot(connection,run['plan_sha256']) if run else {}
    if plan.get('mode')=='streaming':
        result['unknowns']=[]
        for document in documents:
            document['stale_reason']=None
            if document['last_heartbeat_at']:
                from datetime import datetime, timezone
                age=datetime.now(timezone.utc)-datetime.fromisoformat(document['last_heartbeat_at'].replace('Z','+00:00'))
                document['heartbeat_age_ns']=int(age.total_seconds()*1e9)
                if document['status']=='running' and age.total_seconds()>60:
                    document['stale_reason']='lease_heartbeat_expired'
        for stage_name,count in plan['workers'].items():
            selected=[row for row in rows if row['stage']==stage_name]
            ready=waiting=0
            for row in selected:
                if row['status']!='planned':continue
                pending=connection.execute("SELECT COUNT(*) FROM task_dependencies d JOIN current_task_state p ON p.task_id=d.producer_task_id WHERE d.task_id=? AND p.status NOT IN ('ok','skipped','failed')",(row['task_id'],)).fetchone()[0]
                if pending:waiting+=1
                else:ready+=1
            result['queues'].append(dict(stage=stage_name,device_id='cpu',ready=ready,waiting=waiting,
                capacity=max(2,2*count),active=sum(row['status']=='running' for row in selected)))
        result['resources']['disk_budget_bytes']=plan.get('disk_budget_bytes')
        latest=connection.execute("SELECT details_json FROM events WHERE run_id=? AND event_kind='run' ORDER BY event_id DESC LIMIT 1",(run['run_id'],)).fetchone()
        if latest:result['state']=json.loads(latest[0]).get('state',state)
    _mark_interruption(connection, run, counts, result)
    return result


def run_failure_errors(connection, run_id: str, *, state: str, counts: Any = None) -> list[dict[str, Any]]:
    """Typed errors for a run that ran and found failures (`run` exit 1; docs/cli_json_schema.md "Exit codes").

    One error per failed task (its ledger ``reason_code`` as ``code``, the terminal event's message and error type),
    then one per document whose source was unavailable. A cancelled run, or failures the task table does not name,
    still get one summary error, so a failed run never reports an empty ``errors`` list."""
    rows = _rows(connection, "SELECT s.stage,s.operation,s.status,s.reason_code,s.event_id,s.version_id,v.document_id "
                             "FROM current_task_state s LEFT JOIN document_versions v ON v.version_id=s.version_id "
                             "WHERE s.run_id=? AND (s.status='failed' OR (s.status='skipped' AND s.reason_code='source_unavailable')) "
                             "ORDER BY s.status, s.ordinal, s.task_id", (select_run(connection, run_id)["run_id"],))
    errors: list[dict[str, Any]] = []
    unavailable: set[Any] = set()
    for row in rows:
        if row["status"] == "skipped":
            if row["version_id"] in unavailable:
                continue
            unavailable.add(row["version_id"])
        event = (_rows(connection, "SELECT error_type,message FROM events WHERE event_id=?", (row["event_id"],)) or [{}])[0] \
            if row["event_id"] is not None else {}
        reason = row["reason_code"] or ""
        code = reason if re.fullmatch(r"[a-z][a-z0-9_]*", reason) else "stage_failed"  # the schema's code pattern
        message = event.get("message") or (f"{row['stage']} failed ({reason or 'no reason recorded'})" if row["status"] == "failed"
                                           else "the source was unavailable; later stages were skipped")
        errors.append(dict(code=code, message=message, document_id=row["document_id"], stage=row["stage"],
                           details=dict(status=row["status"], reason_code=reason or None, operation=row["operation"],
                                        version_id=row["version_id"], error_type=event.get("error_type"),
                                        event_id=row["event_id"])))
    if not errors:
        code = "run_cancelled" if state == "cancelled" else "run_completed_with_failures"
        errors.append(dict(code=code, message=f"the run ended {state}; see `status` and `report --why`",
                           document_id=None, stage=None, details=dict(state=state, counts=counts or {})))
    return errors


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
    document_ids = list(dict.fromkeys(row["document_id"] for row in rows if row["document_id"]))
    return dict(rows=rows, retention=retention_summary(connection,run),
                resources=status["resources"], counts=status["counts"], conflicts=conflicts,
                sources=source_metadata(connection, document_ids, run))


def plan_view(connection=None, *, selectors=None, stages=None, settings=None, devices=None, workers=None, manifest=None, identities=None,
              exclusions=None, input_manifest=None):
    """``identities`` (aligned with ``selectors``) carries precomputed resource
    identities of URL selectors; their selectors are persisted spellings.

    ``selectors=None`` (omitted) selects every corpus document. An explicit
    selection (a list, even an empty one) is taken literally: an empty list
    selects nothing and the plan carries the blocking ``empty_selection``
    conflict, so run refuses it (CP-K3 F1).

    Every present local source is hashed: the plan binds the bytes observed
    now, also for a ledger-known source whose stored hash (kept separately as
    ``stored_sha256``) may be stale (CP-K3 F3).

    ``exclusions`` lists requested entries that are not processed (for example
    unselected manifest entries), each with its reason; they are never opened.
    The exact preview (targets, exclusions, capabilities, device allocation,
    deletion policy, conflicts) is part of the hashed body (K3, PLAN 2.4)."""
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
    from ..stages.retention import resolved_settings, DEBUG_HELP, MEMORY_SAVING_HELP
    try:settings=resolved_settings(settings)
    except ValueError as error:raise ReadRequestError('disk_budget_required',str(error)) from error
    selection_empty = selectors is not None and len(selectors) == 0
    if selectors is None:
        selectors = [r[0] for r in connection.execute("SELECT document_id FROM documents ORDER BY rowid")] if connection else []
    documents, inventory, seen = [], [], {}
    exclusions = list(exclusions or [])
    pooled_stages = set(selected_stages)
    for index, selector in enumerate(selectors):
        known = None
        identity = identities[index] if identities else None
        if identity is not None:
            if connection:
                rows = _rows(connection, "SELECT * FROM documents WHERE source_key=?", (identity,))
                known = rows[0] if rows else None
        elif connection:
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
            if identity is None and parts.scheme not in ("http", "https", "file") and not Path(selector).is_absolute():
                raise ReadRequestError("file_not_found", "A new local source must be an explicit absolute path", details={"selector": selector})
            identifier, key, version = new_id(), identity or source_key(selector), None
        if key in seen:
            exclusions.append(dict(selector=selector, reason_code="duplicate_source", duplicate_of_document_id=seen[key],
                                   basis="same source identity as an earlier entry"))
            continue
        seen[key] = identifier
        source_uri=source_uri_of_key(key) or selector
        source_path=Path(_path(source_uri)) if source_uri.startswith('file:') else None
        source_stat=source_path.stat() if source_path is not None and source_path.is_file() else None
        observation=dict(source_uri=source_uri,bytes=source_stat.st_size if source_stat else None,
            mtime_ns=source_stat.st_mtime_ns if source_stat else None,stored_sha256=version.get('stored_sha256') if version else None)
        if source_stat is not None:
            # The exact bytes the plan binds (read only; run refuses if they change),
            # observed now even when the ledger knows an older hash (CP-K3 F3).
            from ..stages.execution import sha256_file
            try:observation['sha256']=sha256_file(source_path)
            except OSError as error:observation['read_error']=type(error).__name__  # the run records it as unavailable
        inventory.append(dict(source_key=key, version=version,observation=observation))
        document_stages = set(selected_stages)
        remote_source = key.startswith(("url:v1:", "url:v2:")) or urlsplit(selector).scheme.lower() in ("http", "https")
        if known and not remote_source:
            remote_source = any(r[0] == "url" for r in connection.execute("SELECT source_kind FROM source_records WHERE document_id=?", (identifier,)))
        if remote_source and "extract" in document_stages:
            # A run acquires every remote source again (no cross-run reuse), including one an earlier run
            # downloaded or deleted under memory_saving=extreme: the preview shows that fetch (K4BP-G5).
            document_stages.add("download")
        pooled_stages.update(document_stages)
        steps = []
        for stage in STAGE_ORDER:
            if stage not in document_stages:
                continue
            dependencies = ["download"] if stage == "extract" and "download" in document_stages else prerequisites[stage]
            package = {"download": "download", "annotate": "annotate", "ocr": "ocr", "math": "layout"}.get(stage, "core")
            from ..ocr.defaults import planned_device
            device = planned_device((devices or {}).get(stage) or settings.get('ocr',{}).get('options',{}).get('device')) if stage=='ocr' else (devices or {}).get(stage,'cpu')
            steps.append(dict(stage=stage, operation=stage+".planned", action="conditional", reason_code="legacy_unknown" if known else "stage_planned",
                input_roles=[dict(producer=s, role="stage_input") for s in dependencies],
                conditional_on=dict(predicate="selected_adapters_and_verified_inputs_available", input_producer=dependencies, unresolved_fields=["quality", "route", "settings_tool_fingerprint"]),
                packages=[package], device_id=device, executable_by_run=stage not in RUN_UNAVAILABLE,
                unavailable_reason=RUN_UNAVAILABLE.get(stage)))
        documents.append(dict(document_id=identifier, source_key=key, version_id=version["version_id"] if version else None, steps=steps,source_observation=observation))
    source_inventory_hash = snapshot_hash(dict(records=inventory, manifest=manifest))
    present = connection.execute("SELECT COALESCE(SUM(bytes),0) FROM current_file_state WHERE presence='present'").fetchone()[0] if connection else 0
    from ..ocr.defaults import planned_device, package_entry
    pools = [dict(stage=s, device_id=planned_device((devices or {}).get(s) or settings.get('ocr',{}).get('options',{}).get('device')) if s=='ocr' else (devices or {}).get(s,'cpu'), workers=(workers or {}).get(s, 1), queue_capacity=max(2, 2*(workers or {}).get(s, 1)), cpu_tokens=1) for s in STAGE_ORDER if s in pooled_stages]
    from ..ocr.defaults import default_values, accelerator_claims
    for pool in pools:
        if pool['stage']=='ocr':
            values={**default_values(pool['device_id']),**settings.get('ocr',{}).get('options',{})}
            pool['detector_device']=values['detector_device']
            pool['accelerator_devices']=sorted(accelerator_claims('ocr',pool['device_id'],{'options':values}))
            pool['cpu_tokens']=int(settings.get('ocr',{}).get('num_threads',1))
    # Every corpus-wide waiting point the run executes, with why it waits and what continues (W18 M13).
    from ..stages.barriers import planned_barriers
    barriers: Any = planned_barriers(pooled_stages, len(documents))
    from ..plan_preview import preview_fields
    planned_stages = [stage for stage in STAGE_ORDER if stage in pooled_stages]
    preview = preview_fields(documents=documents, stages=planned_stages, requested=list(requested), settings=settings,
        pools=pools, explicit_devices=dict(devices or {}), exclusions=exclusions, input_manifest=input_manifest,
        selection_empty=selection_empty)
    available = {row["stage"]: row for row in preview["capabilities"]["stages"]}
    def requirement(group):
        rows = [row for row in available.values() if any(item["package_group"] == group for item in row["packages"])]
        installed = all(not item["blocking"] for row in rows for item in row["packages"] if item["package_group"] == group)
        distributions = sorted({item["requirement"] for row in rows for item in row["packages"] if item["package_group"] == group})
        return installed, distributions
    def requirement_entry(group):
        installed, distributions = requirement(group)
        if group == "ocr":
            entry = package_entry(next(pool["device_id"] for pool in pools if pool["stage"] == "ocr"))
            return dict(entry, installed=installed and available["ocr"]["available"],
                        reason_code=None if available["ocr"]["available"] else available["ocr"]["reason_code"])
        return dict(package_group=group, distributions=distributions, entry_point=None, installed=installed,
                    reason_code=None if installed else "missing_capability")
    body = dict(source_inventory_hash=source_inventory_hash, settings=settings, settings_help={"debug":DEBUG_HELP,"memory_saving":MEMORY_SAVING_HELP}, documents=documents, barriers=barriers, pools=pools,
        resource_limits={k: settings.get(k) for k in ("disk_budget_bytes", "max_source_bytes", "max_attempt_bytes")},
        disk=dict(present_bytes=present, pinned_bytes=None, reserved_bytes=None, estimated_peak_bytes=None, unknown_sizes=[dict(document_id=d["document_id"], reason_code="unknown_not_recorded") for d in documents]),
        requirements=[requirement_entry(p) for p in sorted({p for d in documents for s in d['steps'] for p in s['packages']})],
        legacy_import_required=connection is None or connection.execute("SELECT 1 FROM events WHERE reason_code='legacy_import_completed' LIMIT 1").fetchone() is None,
        execution=dict(command="run", profile="streaming_document_dag",
            available=not any(stage in RUN_UNAVAILABLE for stage in pooled_stages),
            # available: the streaming DAG can execute these stages; ready: and
            # nothing here blocks it (missing installs, device conflicts).
            ready=not any(stage in RUN_UNAVAILABLE for stage in pooled_stages) and not any(
                item["severity"] in ("error", "blocking") for item in preview["conflicts"]),
            blocked_by=sorted({item["code"] for item in preview["conflicts"] if item["severity"] in ("error", "blocking")}),
            unavailable_stages={stage: RUN_UNAVAILABLE[stage] for stage in sorted(pooled_stages) if stage in RUN_UNAVAILABLE},
            basis="run revalidates sources, packages and devices before any dispatch"),
        explicit_manifest=scrub(manifest), **preview)
    return dict(body, plan_hash=snapshot_hash(body))
