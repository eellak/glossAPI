"""Explicit, atomic import of an existing corpus's files into the ledger, as observations; never a scheduling input.

``glossapi run`` calls import_corpus once before a new run (the sole writer); read commands never call it.
Mapping keys are relative file paths, or '<path>#<row>:<url_index>' for rows.
Mappings provide source_key/document_id and, for chunks, a parent input hash
and inclusive range. Unresolved observations stay in the ledger, not tasks.
"""
from __future__ import annotations

import json
import hashlib
import math
import decimal
from contextlib import ExitStack
import shutil
import tempfile
import re
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .encoding import canonical_json, dataframe_records, file_hash, snapshot_hash, typed_value
from .identity import _encoded_source_key, source_key
from .stores import TEXT_FIELDS, identity_present
from .writer import LedgerError, LedgerWriter

_SOURCE_SUFFIXES = {".pdf", ".doc", ".docx", ".html", ".xml", ".pptx", ".csv"}
_SOURCE_FIELDS = {"url", "source_uri", "source_path", "source_id", "document_id", "dataset_id", "filename_base", "filename"}
_CHUNK = re.compile(r"^(.+)__p(\d{4,5})-(\d{4,5})$")


def _paths(root):
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.name not in
            {"corpus.sqlite3", "corpus.sqlite3-wal", "corpus.sqlite3-shm"}
            and not p.name.endswith(("-wal", "-shm"))]


def _stamp(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns, file_hash(p))
            for p in _paths(root)} | {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns, file_hash(p))
        for p in sorted(root.rglob("*-wal")) if p.name != "corpus.sqlite3-wal" and p.is_file()}


def _sqlite_header(path):
    with path.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def _validate_ranges(value):
    if isinstance(value, dict):
        if "page_first" in value or "page_last" in value:
            first, last = value.get("page_first"), value.get("page_last")
            if (first is not None or last is not None) and (type(first) is not int or type(last) is not int or first < 1 or last < first):
                raise ValueError("Invalid one-based inclusive page range")
        for child in value.values():
            _validate_ranges(child)
    elif isinstance(value, list):
        for child in value:
            _validate_ranges(child)


def _product_row(fields):
    return ({"id", "row_id", "filename", "section", "json_section", "section_length", "section_propo"} <= set(fields) or
            "doc_id" in fields and "chunk_id" in fields and any(k in fields for k in ("document", "text", "content")))


def _product_path(path, root):
    parts = path.relative_to(root).parts
    directories = {"sidecars", "json", "sections", "markdown", "clean_markdown", "ocr", "native", ".glossapi", "shards"}
    return (any(part in directories or part.endswith(".parquet.tmp") for part in parts[:-1]) or
            path.name in {"sections_for_annotation.parquet", "classified_sections.parquet", "fully_annotated_sections.parquet"} or
            any(token in path.name for token in (".latex_map.", ".metrics.", ".page_state.", ".triage.")))


def _payload_observation(value, field=None):
    if field in TEXT_FIELDS | {"document"} and isinstance(value, (str, bytes)):
        payload = value.encode("utf-8") if isinstance(value, str) else value
        return dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                    characters=len(value) if isinstance(value, str) else None)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[key] = _payload_observation(item, field=key)
        return result
    if isinstance(value, list):
        return [_payload_observation(item, field=field) for item in value]
    if isinstance(value, tuple):
        return tuple(_payload_observation(item, field=field) for item in value)
    if type(value).__module__.startswith("numpy") and type(value).__name__ == "ndarray":
        return dict(array_dtype=str(value.dtype), array_shape=list(value.shape),
                    values=_payload_observation(value.tolist(), field=field))
    if isinstance(value, decimal.Decimal):
        return {"typed": typed_value(value)}
    if isinstance(value, float) and not math.isfinite(value):
        return {"typed": typed_value(value)}
    return value


def _sqlite_snapshot(path):
    # SQLite's backup API seals committed WAL state. A distributed historical
    # main+WAL pair may have no SHM index: rebuild that index only in a private
    # copy, while inventory's before/after hashes enforce source quiescence.
    wal = path.with_name(path.name + "-wal")
    journal = path.with_name(path.name + "-journal")
    frozen_copy = journal.exists() or wal.exists() and not path.with_name(path.name + "-shm").exists()
    with ExitStack() as resources:
        if frozen_copy:
            temporary = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix="glossapi-legacy-snapshot-")))
            copied = temporary/path.name
            shutil.copyfile(path, copied)
            for sidecar in (wal, journal):
                if sidecar.exists():
                    shutil.copyfile(sidecar, temporary/sidecar.name)
            source = sqlite3.connect(copied, isolation_level=None)
        else:
            uri = path.as_uri() + ("?mode=ro" if wal.exists() else "?mode=ro&immutable=1")
            source = sqlite3.connect(uri, uri=True, isolation_level=None)
        resources.callback(source.close)
        target = sqlite3.connect(":memory:")
        resources.callback(target.close)
        source.execute("PRAGMA query_only=ON")
        source.backup(target)
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise sqlite3.DatabaseError("Invalid legacy SQLite database")
        tables = []
        for name, sql in target.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"):
            quoted = '"' + name.replace('"', '""') + '"'
            cursor = target.execute("SELECT * FROM " + quoted)
            rows = [typed_value(_payload_observation(dict(zip([c[0] for c in cursor.description], row)))) for row in cursor]
            rows.sort(key=canonical_json)
            tables.append(dict(name=name, sql=sql, columns=[c[0] for c in cursor.description], rows=rows))
        return dict(format="sqlite_backup", tables=tables, authority="historical_observations_only",
                    snapshot_method="backup_from_frozen_private_copy" if frozen_copy else "readonly_backup")


def inventory_corpus(root):
    """Freeze bytes, schemas and rows; reject any concurrent corpus mutation."""
    root = Path(root).absolute()
    before = _stamp(root)
    entries = []
    for path in _paths(root):
        relative = str(path.relative_to(root))
        entry = dict(path=relative, uri=path.as_uri(), sha256=before[relative][2], bytes=before[relative][0], mtime_ns=before[relative][1])
        try:
            if path.is_symlink():
                entry.update(format="unresolved_symlink", error="External/symlink ownership requires an explicit mapping")
            elif path.suffix == ".parquet":
                import pandas as pd  # importer only; never the read interface
                frame = pd.read_parquet(path)
                product_rows = _product_path(path, root) or _product_row(frame.columns) or not (_SOURCE_FIELDS & set(frame.columns))
                entry.update(format="parquet", row_role="product" if product_rows else "source_observation", validation="schema_readable", columns=[typed_value(c) for c in frame.columns],
                             dtypes=[str(d) for d in frame.dtypes], rows=[
                                 dict(index=typed_value(index), fields=typed_value(_payload_observation(row) if product_rows else row))
                                 for index, row in dataframe_records(frame)])
            elif path.suffix in (".sqlite", ".sqlite3", ".db") or _sqlite_header(path):
                entry.update(_sqlite_snapshot(path))
                wal_name = relative + "-wal"
                if wal_name in before:
                    entry["wal"] = dict(sha256=before[wal_name][2], bytes=before[wal_name][0], mtime_ns=before[wal_name][1])
            elif path.suffix in (".json", ".jsonl"):
                value = (json.loads(path.read_text(), parse_float=decimal.Decimal) if path.suffix == ".json" else
                         [json.loads(line, parse_float=decimal.Decimal) for line in path.read_text().splitlines() if line.strip()])
                if path.name.endswith(".docling.json") or isinstance(value, dict) and value.get("schema_name") == "DoclingDocument":
                    entry.update(format="document_payload_json", validation="json_syntax_payload_only",
                                 schema_name=value.get("schema_name") if isinstance(value, dict) else None,
                                 document_schema_version=_payload_observation(value.get("version")) if isinstance(value, dict) else None)
                    entries.append(entry)
                    continue
                entry.update(format="json", validation="syntax_and_explicit_ranges_only", evidence=_payload_observation(value))
                # Only source manifests retain full rows; document payloads
                # and diagnostics keep explanatory fields and body hashes.
                if isinstance(value, list) and all(isinstance(r, dict) for r in value) and any(_SOURCE_FIELDS & r.keys() for r in value) and not _product_path(path, root) and not any(_product_row(r) for r in value):
                    entry["rows"] = [dict(index=typed_value(i), fields=typed_value(row)) for i, row in enumerate(value)]
                    entry["validation"] = "source_metadata_observed_not_interpreted"
                else:
                    _validate_ranges(value)
            elif "fatal" in path.name and path.suffix == ".txt":
                entry.update(format="fatal_skiplist", lines=path.read_text().splitlines())
            elif path.suffix == ".pkl":
                entry.update(format="opaque_historical_pickle", unknown="not_unpickled")
            elif path.suffix == ".md":
                path.read_text(encoding="utf-8")
                entry.update(format="markdown", validation="utf8_only")
            else:
                entry["format"] = "payload"
        except (OSError, ValueError, TypeError, sqlite3.DatabaseError) as error:
            entry.update(format="malformed", error_type=type(error).__name__, error=str(error))
        entries.append(entry)
    if before != _stamp(root):
        raise LedgerError("Legacy corpus changed during inventory; quiesce writers and retry")
    body = dict(kind="legacy_inventory", schema_version=1, root=root.as_uri(), files=entries)
    return dict(body, inventory_hash=snapshot_hash(body))


def _value(value):
    from .encoding import untyped_value
    return untyped_value(value)


def _uri(value, root):
    if not isinstance(value, str) or not value.strip():
        return None
    return value if urlsplit(value).scheme else (root / value).absolute().as_uri()


def _file_path(uri):
    return Path(unquote(urlsplit(uri).path)) if uri and urlsplit(uri).scheme == "file" else None


def _stage(path):
    name = path.lower()
    for token, stage in (("annotat", "annotate"), ("classif", "annotate"), ("section", "sections"),
                         ("math", "math"), ("ocr", "ocr"), ("clean", "clean"), ("extract", "extract")):
        if token in name:
            return stage
    return "export" if name.endswith((".jsonl", ".jsonl.zst")) else "download"


def _is_source(path, root):
    return path.suffix.lower() in _SOURCE_SUFFIXES and (len(path.relative_to(root).parts) == 1 or path.parent.name in {"downloads", "input", "sources", "pdfs"})


def _claim_stage(field):
    if not isinstance(field, str):
        return None
    if field.startswith("download"):
        return "download"
    if field in {"filter", "needs_ocr", "extraction", "is_empty"}:
        return "clean"
    for stage in ("extract", "ocr", "math", "annotate", "section", "export"):
        if field.startswith(stage):
            return "sections" if stage == "section" else stage
    return None


def import_corpus(root, *, mapping=None, trust_legacy_outputs=False, code_revision="unknown_not_recorded"):
    """Import unchanged evidence once. A crash rolls back the entire import.

    A changed inventory or mapping is a new, explicit observation run. No old
    fact is overwritten and no imported artifact is eligible for semantic reuse.
    """
    root = Path(root).absolute()
    mapping = mapping or {}
    if not root.is_dir():
        raise LedgerError("Import requires an existing corpus directory")
    with LedgerWriter(root / "corpus.sqlite3", code_revision=code_revision) as writer:
        inventory = inventory_corpus(root)
        identity = snapshot_hash([inventory["inventory_hash"], mapping, trust_legacy_outputs])
        previous_imports = []
        for row in writer.connection.execute("SELECT run_id,details_json FROM events WHERE reason_code='legacy_import_completed'"):
            details = json.loads(row[1])
            previous_imports.append(dict(run_id=row[0], inventory_hash=details["inventory_hash"], import_identity=details["import_identity"]))
            if details.get("import_identity") == identity:
                return dict(details, run_id=row[0], already_imported=True)
        with writer.transaction():
            inventory_hash = writer.snapshot("legacy_inventory", inventory)
            run = writer.run(dict(mode="shadow_import", trust_legacy_outputs=trust_legacy_outputs),
                             dict(mode="legacy_import", inventory_sha256=inventory_hash, mapping=mapping),
                             dict(provenance="legacy_unknown"), code_revision=code_revision)
            if previous_imports:
                writer.event(run, event_kind="import", reason_code="legacy_observed", details=dict(
                    observation="explicit_inventory_reconciliation", previous_imports=previous_imports, inventory_sha256=inventory_hash))
            conflicts, unknowns = [], []
            bindings, versions, uri_documents, file_ids = {}, {}, {}, {}
            artifact_mappings = dict(mapping)
            ambiguous_artifacts = set()
            manifests = {}
            claims = {}
            chunk_ranges = {}

            def conflict(code, **details):
                item = dict(code=code, **details)
                conflicts.append(item)
                writer.event(run, event_kind="import", reason_code="legacy_conflict", details=item)

            def document(key, explicit=None):
                if explicit:
                    found = writer.connection.execute("SELECT source_key FROM documents WHERE document_id=?", (explicit,)).fetchone()
                    if found is None:
                        raise LedgerError("A rename mapping must name an existing document_id")
                    identifier = explicit
                else:
                    identifier = writer.document(key)
                versions.setdefault(identifier, writer.connection.execute(
                    "SELECT version_id FROM document_versions WHERE document_id=? ORDER BY revision DESC LIMIT 1", (identifier,)).fetchone())
                if versions[identifier] is not None and not isinstance(versions[identifier], str):
                    versions[identifier] = versions[identifier][0]
                return identifier

            def register_artifact(stage, kind, uri, **values):
                first, last = values.get("page_first"), values.get("page_last")
                if ((first is None) != (last is None) or
                        first is not None and (type(first) is not int or type(last) is not int or first < 1 or last < first)):
                    raise LedgerError("Artifact ranges require one-based inclusive integers")
                previous = writer.connection.execute("""SELECT file_id FROM current_file_state
                    WHERE uri=? AND sha256 IS ? AND bytes IS ? AND version_id IS ?
                    AND page_first IS ? AND page_last IS ? AND presence=? ORDER BY event_id DESC LIMIT 1""",
                    (uri, values.get("sha256"), values.get("bytes"), values.get("version_id"),
                     values.get("page_first"), values.get("page_last"), values.get("presence", "present"))).fetchone()
                return previous[0] if previous else writer.artifact(run, stage, kind, uri, **values)

            def register_source_record(identifier, uri, metadata, *, record_id, manifest, ordinal, resource_index):
                original = writer.connection.execute(
                    "SELECT document_id,source_uri,run_id FROM source_records WHERE source_record_id=?", (record_id,)).fetchone()
                if original and (original[0], original[1]) != (identifier, uri):
                    # A later explicit mapping is another immutable observation,
                    # not an UPDATE to the owner of the original occurrence.
                    original_id = record_id
                    record_id = snapshot_hash([original_id, "explicit_source_mapping", identifier, uri])
                    metadata = dict(metadata, original_source_record_id=original_id, observation="explicit_source_mapping")
                    conflict("explicit_source_mapping_changed", source_record_id=original_id,
                             original_document_id=original[0], document_id=identifier, source_uri=uri,
                             resolution="explicit_mapping", mapped_source_record_id=record_id)
                    original = writer.connection.execute(
                        "SELECT document_id,source_uri,run_id FROM source_records WHERE source_record_id=?", (record_id,)).fetchone()
                if original:
                    writer.event(run, event_kind="import", reason_code="legacy_observed", version_id=versions[identifier],
                        details=dict(observation="source_occurrence_reobserved", source_record_id=record_id,
                                     original_run_id=original[2], inventory_sha256=inventory_hash))
                    return record_id
                return writer.source_record(identifier, run, uri, metadata,
                    source_kind="url" if urlsplit(uri).scheme in ("http", "https") else "local_path",
                    manifest_sha256=manifest, source_row=str(ordinal), url_index=resource_index,
                    source_record_id=record_id)

            def declared_chunk(entry, override):
                return ("parent_input_sha256" in override or
                        bool(_CHUNK.match((root/entry["path"]).stem)) and
                        ("page_first" in override or "page_last" in override))

            def initial_source(entry):
                path = root/entry["path"]
                override = artifact_mappings.get(entry["path"], {})
                explicit_location = entry["uri"] in uri_documents and path.suffix.lower() in _SOURCE_SUFFIXES
                return (entry["format"] != "unresolved_symlink" and not declared_chunk(entry, override)
                        and (_is_source(path, root) or explicit_location))

            for entry in inventory["files"]:
                if "rows" not in entry or entry.get("row_role") == "product":
                    continue
                file_id = register_artifact(_stage(entry["path"]), "legacy_manifest", entry["uri"],
                    sha256=entry["sha256"], bytes=entry["bytes"], ownership="legacy_external", retention_class="evidence",
                    presence="corrupt" if entry["format"] == "malformed" else "present")
                file_ids[entry["path"]] = file_id
                manifests[entry["path"]] = writer.snapshot("source_manifest", dict(kind="source_manifest", uri=entry["uri"],
                    original_file_sha256=entry["sha256"], original_file_id=file_id, inventory_sha256=inventory_hash,
                    row_home="legacy_inventory.files[path].rows", columns=entry.get("columns", []), row_count=len(entry["rows"])))

            # Source occurrences are identified by explicit records/URI, never
            # basename; store-local rows and duplicate occurrences stay distinct.
            for entry in inventory["files"]:
                if entry.get("row_role") == "product":
                    continue
                for ordinal, encoded in enumerate(entry.get("rows", [])):
                    fields = _value(encoded["fields"])
                    original_values = {_value(key): value for key, value in encoded["fields"]["value"]}
                    if _product_row(fields):
                        continue  # section rows are products, not source records
                    if not (_SOURCE_FIELDS & fields.keys()):
                        continue
                    values = fields.get("url")
                    urls = values if isinstance(values, (list, tuple)) else [values]
                    for url_index, url in enumerate(urls):
                        supplied_index = fields.get("url_index")
                        resource_index = supplied_index if identity_present(supplied_index) and len(urls) == 1 else url_index
                        if type(resource_index) is not int or resource_index < 0:
                            conflict("invalid_resource_index", path=entry["path"], row=ordinal, value=typed_value(resource_index))
                            continue
                        locator = f"{entry['path']}#{ordinal}:{resource_index}"
                        override = mapping.get(locator, {})
                        explicit = override.get("document_id") or fields.get("document_id")
                        requested = fields.get("source_uri")
                        try:
                            if not isinstance(requested, str) or not requested.strip():
                                requested = url
                                if (not isinstance(url, str) or urlsplit(url).scheme.lower() not in ("http", "https", "file")) and fields.get("source_path"):
                                    requested = fields["source_path"]
                            uri = _uri(requested, root)
                        except ValueError as error:
                            conflict("invalid_source_uri", locator=locator, source_path=entry["path"], source_sha256=entry["sha256"], row=ordinal, message=str(error))
                            uri = None
                        if not uri and not explicit and not (fields.get("dataset_id") is not None and fields.get("source_id") is not None) and not override.get("source_key"):
                            conflict("source_identity_unresolved", locator=locator, source_path=entry["path"], source_sha256=entry["sha256"], row=ordinal)
                            continue
                        uri = uri or "urn:glossapi:legacy:" + snapshot_hash([inventory["inventory_hash"], locator])
                        dataset_id = fields.get("dataset_id") if identity_present(fields.get("dataset_id")) else None
                        source_id = fields.get("source_id") if identity_present(fields.get("source_id")) else None
                        if explicit:
                            key = None
                        elif override.get("source_key"):
                            key = override["source_key"]
                        elif source_id is not None and dataset_id is not None:
                            key = _encoded_source_key(original_values["dataset_id"], original_values["source_id"], resource_index)
                        else:
                            key = source_key(uri, dataset_id=dataset_id, source_id=source_id, resource_index=resource_index)
                        identifier = document(key, explicit)
                        bindings[locator] = identifier
                        uri_documents.setdefault(uri, set()).add(identifier)
                        try:
                            location = _uri(fields.get("source_path"), root)
                        except ValueError as error:
                            conflict("invalid_source_location", locator=locator, source_path=entry["path"], row=ordinal, message=str(error))
                            location = None
                        if location:
                            uri_documents.setdefault(location, set()).add(identifier)
                        if versions[identifier] is None:
                            versions[identifier] = writer.version(identifier)
                        record_id = snapshot_hash([inventory["inventory_hash"], entry["path"], ordinal, resource_index])
                        # Keep the ORIGINAL type-tagged row. Decoding values for
                        # relationship lookup must not round nanosecond times or
                        # erase NumPy scalar/array types in their durable home.
                        metadata = dict(original_fields=encoded["fields"], original_fields_encoding="typed_value_v1",
                                        original_index=encoded["index"], original_uri=uri,
                                        naming_policy="source:v1/url:v1/local_path:v1", import_locator=locator,
                                        source_file_sha256=entry["sha256"])
                        register_source_record(identifier, uri, metadata, record_id=record_id,
                            manifest=manifests[entry["path"]], ordinal=ordinal, resource_index=resource_index)
                        artifacts = fields.get("artifacts", [])
                        if isinstance(artifacts, list):
                            for artifact in artifacts:
                                if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
                                    relative = artifact["path"]
                                    proposed = dict(artifact, document_id=identifier)
                                    previous = artifact_mappings.get(relative)
                                    if previous and previous.get("document_id") != identifier:
                                        conflict("manifest_artifact_ownership_ambiguous", path=relative,
                                                 candidates=[previous.get("document_id"), identifier])
                                        if relative not in mapping:
                                            ambiguous_artifacts.add(relative)
                                    elif relative not in mapping:
                                        artifact_mappings[relative] = proposed
                        for field, value in fields.items():
                            if field in {"url", "source_uri", "source_path", "document_id", "dataset_id", "source_id"}:
                                continue
                            claim = dict(document_id=identifier, version_id=versions[identifier], path=entry["path"],
                                         sha256=entry["sha256"], row=ordinal, field=field if isinstance(field, str) else typed_value(field), value=original_values[field], stage=_claim_stage(field), input_binding="legacy_unknown")
                            claims.setdefault((identifier, field), []).append(claim)
                            writer.event(run, event_kind="import", reason_code="legacy_observed", version_id=versions[identifier], details=claim)
            for (_, field), observations in claims.items():
                if len({canonical_json(o["value"]) for o in observations}) > 1:
                    conflict("conflicting_store_claims", field=field, observations=observations)

            # Actual source paths remain independent from URL histories unless
            # an explicit source-path observation or mapping links them.
            for entry in sorted(inventory["files"], key=lambda e: (not initial_source(e), e["path"])):
                path = root / entry["path"]
                override = artifact_mappings.get(entry["path"], {})
                if entry["path"] in ambiguous_artifacts:
                    override = {}
                candidates = set(uri_documents.get(entry["uri"], ()))
                if override.get("document_id"):
                    candidates = {document(None, override["document_id"])}
                elif override.get("source_key"):
                    candidates = {document(override["source_key"])}
                is_source = initial_source(entry)
                if is_source and not candidates:
                    candidates = {document(source_key(entry["uri"]))}
                identifier = next(iter(candidates)) if len(candidates) == 1 and entry["path"] not in ambiguous_artifacts else None
                if len(candidates) > 1:
                    conflict("artifact_ownership_ambiguous", path=entry["path"], candidates=sorted(candidates))
                if identifier and is_source:
                    # Legacy acquisition entity may differ from stored bytes.
                    previous = writer.connection.execute("SELECT version_id,stored_sha256 FROM document_versions WHERE document_id=? ORDER BY revision DESC LIMIT 1", (identifier,)).fetchone()
                    versions[identifier] = previous[0] if previous and previous[1] == entry["sha256"] else writer.version(identifier, source_sha256=None, stored_sha256=entry["sha256"])
                    if not writer.connection.execute("SELECT 1 FROM source_records WHERE document_id=? AND source_uri=?", (identifier, entry["uri"])).fetchone():
                        writer.source_record(identifier, run, entry["uri"], dict(provenance="legacy_unknown", source_hash_unknown="pre_transformation_entity_not_observed"),
                            manifest_sha256=inventory_hash, source_row=entry["path"], source_record_id=snapshot_hash([inventory["inventory_hash"], entry["path"], "local_source"]))
                version = versions.get(identifier) if identifier else None
                if identifier and version is None:
                    version = versions[identifier] = writer.version(identifier)
                first, last = override.get("page_first"), override.get("page_last")
                chunk = _CHUNK.match(path.stem)
                is_chunk = bool(chunk or declared_chunk(entry, override)) and not is_source
                invalid_range = ((first is None) != (last is None) or first is not None and
                                 (type(first) is not int or type(last) is not int or first < 1 or last < first))
                if invalid_range and entry["path"] not in mapping:
                    conflict("invalid_legacy_artifact_range", path=entry["path"],
                             page_first=typed_value(first), page_last=typed_value(last), resolution="explicit_mapping_required")
                    identifier, version, first, last = None, None, None, None
                if is_chunk:
                    parent_hash = override.get("parent_input_sha256")
                    if not (identifier and parent_hash and first is not None and last is not None and
                            writer.connection.execute("SELECT 1 FROM files WHERE version_id=? AND sha256=?", (version, parent_hash)).fetchone()):
                        conflict("chunk_parent_unresolved", path=entry["path"], spelling_range=[int(chunk[2]), int(chunk[3])] if chunk else None)
                        identifier, version, first, last = None, None, None, None
                presence = "corrupt" if entry["format"] == "malformed" else "present"
                if override.get("sha256") and override["sha256"] != entry["sha256"]:
                    presence = "corrupt"
                    conflict("manifest_hash_mismatch", path=entry["path"], expected_sha256=override["sha256"], observed_sha256=entry["sha256"])
                file_id = file_ids.get(entry["path"]) if version is None else None
                file_id = file_id or register_artifact(_stage(entry["path"]), entry["format"], entry["uri"], sha256=entry["sha256"], bytes=entry["bytes"],
                    version_id=version, ownership="legacy_external", retention_class="initial" if is_source else "evidence",
                    presence=presence, page_first=first, page_last=last)
                file_ids[entry["path"]] = file_id
                if is_chunk and version:
                    parent = override["parent_input_sha256"]
                    observation = dict(path=entry["path"], sha256=entry["sha256"], page_first=first, page_last=last, file_id=file_id)
                    siblings = chunk_ranges.setdefault((version, parent), [])
                    for sibling in siblings:
                        if max(first, sibling["page_first"]) <= min(last, sibling["page_last"]):
                            conflict("overlapping_chunk_observations", version_id=version, parent_input_sha256=parent,
                                     observations=[sibling, observation], resolution="explicit_assembly_profile_required")
                    siblings.append(observation)
                if entry["format"] == "malformed":
                    conflict("malformed_artifact", path=entry["path"], sha256=entry["sha256"], error=entry.get("error"))
                writer.event(run, event_kind="import", reason_code="legacy_observed", version_id=version, file_id=file_id,
                    details=dict(path=entry["path"], inventory_sha256=inventory_hash, authority="legacy_evidence_shadow_only",
                                 producer="legacy_unknown", adoption="legacy_unverified" if trust_legacy_outputs and presence == "present" and entry.get("validation") and not is_source and not is_chunk else "legacy_unknown",
                                 stage=_stage(entry["path"]), parent_input_sha256=override.get("parent_input_sha256"),
                                 reuse_eligible=False, source_deletion_allowed=False))
                if not is_source:
                    unknowns.append(dict(path=entry["path"], code="producer_settings_input_binding_unknown", document_id=identifier))

            # Referenced local inputs that disappeared are missing, never deleted.
            for uri, identifiers in uri_documents.items():
                path = _file_path(uri)
                if path is not None and not path.exists():
                    for identifier in sorted(identifiers):
                        register_artifact("download", "referenced_source", uri, version_id=versions[identifier], ownership="legacy_external", retention_class="initial", presence="missing")
            for relative, binding in artifact_mappings.items():
                if "#" in relative or (root/relative).exists() or not binding.get("document_id"):
                    continue
                identifier = binding["document_id"]
                if identifier not in versions:
                    continue
                register_artifact(_stage(relative), "referenced_artifact", (root/relative).absolute().as_uri(),
                    version_id=versions[identifier], ownership="legacy_external", retention_class="evidence", presence="missing")
            current_uris = {entry["uri"] for entry in inventory["files"]}
            for old in writer.connection.execute("SELECT file_id,version_id,uri FROM current_file_state WHERE presence='present'").fetchall():
                path = _file_path(old[2])
                if path and path.is_relative_to(root) and old[2] not in current_uris:
                    writer.event(run, event_kind="artifact", reason_code="missing", file_id=old[0], version_id=old[1],
                                 details=dict(observation="absent_from_reconciled_inventory", policy_deletion="unknown_not_recorded"))
            for entry in inventory["files"]:
                for stem in entry.get("lines", []):
                    candidates = {i for uri, ids in uri_documents.items() if _file_path(uri) and _file_path(uri).stem == stem for i in ids}
                    candidates.update(row[0] for row in writer.connection.execute(
                        "SELECT DISTINCT v.document_id,f.uri FROM files f JOIN document_versions v ON f.version_id=v.version_id")
                        if _file_path(row[1]) and _file_path(row[1]).stem == stem)
                    if len(candidates) != 1:
                        conflict("fatal_stem_unresolved", stem=stem, path=entry["path"], candidates=sorted(candidates))
                    else:
                        identifier = next(iter(candidates))
                        writer.event(run, event_kind="import", reason_code="fatal_skiplist", version_id=versions[identifier], details=dict(stem=stem, cause="legacy_unknown", authority="exclusion_observation_only"))
            for identifier, version in versions.items():
                stored = writer.connection.execute("SELECT stored_sha256 FROM document_versions WHERE version_id=?", (version,)).fetchone()
                if stored and stored[0]:
                    for other, other_version in writer.connection.execute("SELECT document_id,version_id FROM document_versions WHERE stored_sha256=? AND document_id<>?", (stored[0], identifier)):
                        writer.event(run, event_kind="import", reason_code="duplicate_content", version_id=version,
                                     details=dict(other_document_id=other, other_version_id=other_version, hash_basis="stored_sha256", sha256=stored[0], relation="record_only"))
            normalized_uris = {}
            for uri, identifiers in uri_documents.items():
                normalized_uris.setdefault(source_key(uri), set()).update(identifiers)
            for key, identifiers in normalized_uris.items():
                if len(identifiers) > 1:
                    for identifier in sorted(identifiers):
                        writer.event(run, event_kind="import", reason_code="duplicate_url", version_id=versions[identifier],
                                     details=dict(logical_uri_key=key, related_document_ids=sorted(identifiers - {identifier}), relation="record_only"))
            if _stamp(root) != {entry["path"]: (entry["bytes"], entry["mtime_ns"], entry["sha256"]) for entry in inventory["files"]} | {
                entry["path"]+"-wal": (entry["wal"]["bytes"], entry["wal"]["mtime_ns"], entry["wal"]["sha256"])
                for entry in inventory["files"] if "wal" in entry}:
                raise LedgerError("Legacy corpus changed before import commit")
            counts = dict(files=len(file_ids), source_records=len(bindings), documents=len(versions), conflicts=len(conflicts), unknowns=len(unknowns))
            result = dict(import_identity=identity, inventory_hash=inventory["inventory_hash"], inventory_sha256=inventory_hash, counts=counts,
                          conflicts=conflicts, unknowns=unknowns, compatibility_authority="unchanged_shadow_mode")
            writer.event(run, event_kind="import", reason_code="legacy_import_completed", details=result)
        return dict(result, run_id=run, already_imported=False)
