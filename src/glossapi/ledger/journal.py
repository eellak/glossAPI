"""Observation hooks beside legacy stores. Nothing here selects pipeline work.

The active journal belongs to the calling orchestrator, never a backend worker.
Hooks carry structured results; no exception tracing or log parsing is used.
"""
from __future__ import annotations

from ..stages.profile import adapters_enabled

from contextvars import ContextVar
from functools import wraps
import functools
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback as traceback_module
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from .encoding import canonical_json, dataframe_records, file_hash, pack_evidence, typed_value
from .urls import sanitize_url, scrub_text
from .writer import LedgerWriter, new_id, utc_now
from .stores import TEXT_FIELDS, membership_snapshot, read_store, store_states

_active_journal = ContextVar("glossapi_shadow_result_journal", default=None)
_loaded_tools_cache = {}
_SECRET_KEYS = frozenset(("authorization", "cookie", "cookies", "password", "passwd", "token", "api_key", "apikey",
    "headers", "proxy_auth", "auth", "secret", "credentials", "set_cookie", "domain_cookies", "access_key", "private_key",
    "session_id", "sessionid", "bearer", "oauth"))
# Suffix forms cover containers and header names such as ``domain_cookies``,
# ``session_cookie``, ``X-Api-Key`` and ``client_secret``.
_SECRET_SUFFIXES = ("_cookie", "_cookies", "_password", "_passwd", "_secret", "_api_key", "_apikey",
    "_authorization", "_auth", "_credentials", "_access_key", "_private_key", "_session_id")
# ``*_token`` is not a generic suffix: ``lease_token`` (scheduler fencing) and
# ``tail_token`` (LaTeX sanitizer) are evidence, not credentials.
_SECRET_TOKENS = frozenset(("access_token", "refresh_token", "auth_token", "bearer_token", "id_token", "api_token",
    "session_token", "csrf_token", "xsrf_token", "x_auth_token", "x_csrf_token", "x_xsrf_token", "hf_token", "github_token"))
REDACTED = "[redacted]"


def credential_key(key):
    """True for a settings/header key whose value may carry a credential."""
    name = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return name in _SECRET_KEYS or name in _SECRET_TOKENS or name.endswith(_SECRET_SUFFIXES)


def redact_credentials(value):
    """Replace only credential values; every other JSON value is unchanged.

    Strings keep their spelling except URL userinfo and credential query
    fields, which are redacted (CP-LEDGER F2).
    """
    if isinstance(value, dict):
        return {key: (REDACTED if credential_key(key) else redact_credentials(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_credentials(item) for item in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


def safe_settings(value):
    """Recordable copy of settings: credential values never reach a receipt.

    The whole value under a credential key is replaced, including nested
    containers such as ``domain_cookies -> domain -> cookie``.
    """
    if isinstance(value, dict):
        return {str(key): (REDACTED if credential_key(key) else safe_settings(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [safe_settings(item) for item in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, str): return scrub_text(value)
    if isinstance(value, float) and not math.isfinite(value): return typed_value(value)
    if value is None or isinstance(value, (bool, int, float)): return value
    # Do not use repr: backend option objects can hold secrets.
    return {"type": type(value).__module__ + "." + type(value).__name__, "value": "unknown_not_recorded"}



def payload_evidence(value):
    """Keep explanatory structure and hashes, never duplicate text payloads."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in TEXT_FIELDS and isinstance(item, str):
                result[key] = {"sha256": hashlib.sha256(item.encode("utf-8")).hexdigest(), "characters": len(item)}
            else:
                result[key] = payload_evidence(item)
        return safe_settings(result)
    if isinstance(value, list): return [payload_evidence(item) for item in value]
    return safe_settings(value)



def wrapper_return_receipt(value):
    """Describe tabular returns without storing a second parquet payload."""
    if type(value).__module__.startswith("pandas") and type(value).__name__ == "DataFrame":
        description = {"columns": typed_value(value.columns.tolist()), "dtypes": [str(dtype) for dtype in value.dtypes],
                       "index": typed_value(value.index.tolist()),
                       "rows": [typed_value(list(row)) for row in value.itertuples(index=False, name=None)]}
        return {"type": "pandas.DataFrame", "row_count": len(value), "column_count": len(value.columns),
                "columns": description["columns"], "dtypes": description["dtypes"],
                "sha256": hashlib.sha256(canonical_json(description).encode("utf-8")).hexdigest()}
    try:
        return typed_value(value)
    except TypeError:
        return {"type": "unknown_not_recorded", "source_type": type(value).__module__ + "." + type(value).__name__}


def source_key(uri):
    """Stable non-secret identity, never the persisted (redacted) spelling.

    A URL whose persisted form loses nothing keeps ``url:v1:``. A URL with
    userinfo, a query or a fragment value is keyed by an HMAC of the raw URL
    under the corpus key (``url:v2:``): different credentials are different
    resources (CP-LEDGER cycle 2, F2).
    """
    parts = urlsplit(uri)
    if parts.scheme in ("http", "https"):
        from .urls import keyed_source_key, lossless_url
        if not lossless_url(uri):
            return keyed_source_key(uri)
        return "url:v1:" + urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))
    return "local_path:v1:" + uri


def code_revision():
    # The running code cannot change within a process: read the receipt (or ask git) once, not once per attempt.
    return _code_identity()[0]


@functools.lru_cache(maxsize=1)
def _code_identity():
    """(commit, tree_sha256) of the running code: the shipped receipt, else git HEAD and no tree hash."""
    root = Path(__file__).resolve().parents[3]
    shipped = root / ".code-receipt.json"
    if shipped.is_file():
        receipt = json.loads(shipped.read_text())
        return receipt["commit"], receipt["tree_sha256"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip(), None
    except (OSError, subprocess.SubprocessError): return "unknown_not_recorded", None


def loaded_tools():
    """Actual loaded modules only; never import a backend to identify it."""
    packages, native = [], []
    for name in ("pandas", "numpy", "pyarrow", "pypdfium2", "docling", "torch", "sklearn", "glossapi_rs_cleaner", "glossapi_rs_noise", "glossapi_rs_pdfrepair"):
        module = sys.modules.get(name)
        if module is None: continue
        key = (name, getattr(module, "__file__", None))
        if key not in _loaded_tools_cache:
            distribution = {"sklearn": "scikit-learn"}.get(name, name)
            try: version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError: version = getattr(module, "__version__", None)
            path = getattr(module, "__file__", None)
            binary_hash = file_hash(path) if path and Path(path).is_file() and Path(path).suffix in (".so", ".dylib", ".pyd") else None
            _loaded_tools_cache[key] = {"name": name, "version": version, "distribution_hash": None, "build_hash": binary_hash}
        entry = _loaded_tools_cache[key]
        (native if entry["build_hash"] else packages).append(entry)
    code_identity = _code_identity()[1]
    return {"python": sys.version, "platform": platform.platform(), "code_tree_sha256": code_identity, "packages": packages, "native": native,
            "models": [], "dictionaries": [], "unavailable": ["unreported_model_identity", "unreported_dictionary_identity", "distribution_hashes", "font_resolution"]}


def observe(reason_code, *, path=None, error=None, status=None, message="", **details):
    """Record a fact at its branch, preserving the original exception/return."""
    journal = _active_journal.get()
    if journal is None or journal.owner_pid != os.getpid(): return
    journal.observe(reason_code, path=path, error=error, status=status, message=message, **details)
    if reason_code == "delete_completed" and path is not None and isinstance(journal, ResultJournal):
        registration = journal.files.get(str(Path(path).absolute()))
        if registration:
            file_id, version_id, _ = registration
            with journal.writer.transaction():
                journal.writer.event(journal.run_id, event_kind="artifact", reason_code="deleted", version_id=version_id, file_id=file_id, details={"deletion": "observed_legacy_cleanup"})


def observe_store(path, *, kind="legacy_store"):
    journal = _active_journal.get()
    if journal is not None and journal.owner_pid == os.getpid(): journal.store(Path(path), kind=kind)


def observe_source_rows(frame, path, *, url_column="url"):
    journal = _active_journal.get()
    if journal is not None and journal.owner_pid == os.getpid(): journal.source_rows(frame, Path(path), url_column=url_column)


# K4(c) part 4 (coordinator decision on K4C3-G4): the user's originals are hashed only by the stages that read them.
# A reading stage's wrapper records the hash it observed (process-level, keyed by the file's unchanged
# (dev, ino, size, mtime_ns)); a later phase in the same process that never opens the originals binds its source
# observation to that hash instead of reading the bytes again. A new process (or no recorded hash, or any stat change)
# hashes as before.
ORIGINAL_READING_STAGES = frozenset(("extract", "ocr", "math", "download"))
# CP-K4b F3/F4: the key includes st_ctime_ns; a hash taken from this cache is byte-verified before any destructive
# step of the phase (ResultJournal.verify_cached_originals, before retention); a forked child starts empty.
_original_hashes: dict[str, tuple[tuple[int, int, int, int, int], str]] = {}


def _forget_original_hashes_after_fork() -> None:
    _original_hashes.clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_forget_original_hashes_after_fork)


def observed_original_hash(stage: str, path: Path, cached: 'list[tuple[Path, str]] | None' = None) -> str:
    from .encoding import own_hash_key
    key = own_hash_key(path.stat())
    if stage not in ORIGINAL_READING_STAGES:
        recorded = _original_hashes.get(str(path))
        if recorded is not None and recorded[0] == key:
            if cached is not None:
                cached.append((path, recorded[1]))
            return recorded[1]
    digest = file_hash(path)
    if stage in ORIGINAL_READING_STAGES:
        _original_hashes[str(path)] = (key, digest)
    return digest


def inventory(root, produced=()):
    result = {}
    if not Path(root).exists(): return result
    for directory, directories, names in os.walk(root):
        directories[:] = sorted(name for name in directories if name != '.glossapi')
        for name in sorted(names):
            path = Path(directory) / name
            # Adapter generations already have producer-bound file registrations.
            # Do not observe them again as unowned compatibility stores.
            if ".glossapi" in path.relative_to(root).parts:
                continue
            if not path.is_file() or path.name in ("corpus.sqlite3", "corpus.sqlite3-wal", "corpus.sqlite3-shm") or path.suffix in (".log", ".lock"):
                continue
            if str(path.absolute()) in produced:
                continue
            result[str(path.absolute())] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    return result


class ResultJournal:
    def __init__(self, writer, corpus, stage, operation, parameters):
        self.writer, self.corpus, self.stage, self.operation = writer, corpus, stage, operation
        self.owner_pid = os.getpid()
        from ..stages.retention import resolved_settings
        self.retention_settings = resolved_settings(getattr(corpus, '_retention_settings', {}))
        if getattr(corpus,'_scheduler_writer',None) is None:
            self.retention_settings={key:value for key,value in self.retention_settings.items() if key in ('debug','memory_saving','disk_budget_bytes','max_source_bytes','max_attempt_bytes')}
        self.parameters = safe_settings(parameters)
        self.versions = {}
        self.files = {}
        self.document_tasks = {}
        self.document_outcomes = {}
        self.failures, self.skips, self.successes = [], [], []
        self.adapter_stages = set()
        self.adapter_attempts = set()
        self.adapter_profile = adapters_enabled()
        from ..stages.profile import caller_kernel_override
        self.caller_override = caller_kernel_override(corpus,stage)
        self.cutover = self.adapter_profile and getattr(corpus, '_stage_adapters_default', False) and os.environ.get('GLOSSAPI_STAGE_PROFILE') != 'sequential-adapters' and not self.caller_override
        self.mode = 'sequential_cutover' if self.cutover else 'shadow'
        if self.cutover:
            from ..stages.metadata import restore_metadata_projections
            restore_metadata_projections(self)
        self.started_at, self.started_ns = utc_now(), time.monotonic_ns()
        # Producer-bound products already have their immutable byte receipts.
        # Inventory still observes unknown/external files in shadow mode.
        self.before = self.inventory()
        self.source_manifests = set()
        self.revision = code_revision()
        with writer.transaction():
            self.run_id = writer.run({"mode": self.mode, "parameters": self.parameters, **self.retention_settings},
                {"mode": 'wrapper_products' if self.cutover else "observe_existing_wrapper", "stage": stage, "operation": operation}, loaded_tools(), code_revision=self.revision)
            publication=getattr(corpus,'_scheduler_publication',None)
            if publication:
                writer.event(publication['run_id'],event_kind='run',reason_code='legacy_observed',message='wrapper:'+stage,
                    details={'wrapper_run_id':self.run_id,'publication_task_id':publication['task_id']})
            observed_inputs = []
            self.cached_originals: list[tuple[Path, str]] = []
            candidates = parameters.get("file_paths")
            if not candidates:
                candidates = getattr(corpus, "_shadow_source_paths", None)
            if not candidates:
                directory = corpus.downloads_dir if corpus.downloads_dir.exists() else corpus.input_dir
                candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".doc", ".docx", ".xml", ".html", ".pptx", ".csv", ".md")] if directory.exists() else []
            corpus._shadow_source_paths = tuple(str(p) for p in candidates)
            stage_name = str(cast(object, stage))
            for ordinal, candidate in enumerate(candidates):
                path = Path(candidate).absolute()
                present = path.is_file()
                uri = path.as_uri()
                streaming = getattr(corpus, '_streaming_source_versions', {}).get(str(path))
                bound = writer.connection.execute(
                    "SELECT sha256,bytes FROM files WHERE uri=? AND version_id=? ORDER BY rowid DESC LIMIT 1",
                    (uri, streaming)).fetchone() if streaming else None
                digest = bound['sha256'] if bound else observed_original_hash(stage_name, path, self.cached_originals) if present else None
                version_id = getattr(corpus,'_streaming_source_versions',{}).get(str(path))
                if version_id:
                    document_id=writer.connection.execute('SELECT document_id FROM document_versions WHERE version_id=?',(version_id,)).fetchone()[0]
                else:
                    document_id = writer.document(source_key(uri))
                    version_id = writer.version(document_id, source_sha256=digest, stored_sha256=digest)
                self.versions[str(path)] = version_id
                metadata = {"path": str(path), "naming_policy": "local_path:v1", "provenance": "local_input_observed"}
                encoded_metadata = canonical_json({"schema_version": 1, "fields": typed_value(metadata)})
                # The same source occurrence is an input to five wrappers, not
                # five new acquisition rows. Each attempt still binds its file.
                # A streaming run already recorded every requested occurrence in
                # its source inventory; publication wrappers add none (F3/F4).
                if streaming is None and writer.connection.execute("SELECT 1 FROM source_records WHERE document_id=? AND source_uri=? AND source_row=? AND manifest_sha256 IS NULL AND metadata_json=? LIMIT 1",
                        (document_id, uri, str(ordinal), encoded_metadata)).fetchone() is None:
                    writer.source_record(document_id, self.run_id, uri, metadata, source_row=str(ordinal))
                size = path.stat().st_size if present else None
                released = writer.connection.execute("SELECT file_id FROM current_file_state WHERE uri=? AND version_id=? AND presence='deleted' ORDER BY event_id DESC LIMIT 1",(uri,version_id)).fetchone() if not present else None
                if released:
                    continue
                prior = writer.connection.execute("SELECT file_id FROM current_file_state WHERE uri=? AND sha256 IS ? AND bytes IS ? AND version_id=? AND presence=? ORDER BY event_id DESC LIMIT 1",
                    (uri, digest, size, version_id, "present" if present else "missing")).fetchone()
                file_id = prior[0] if prior else writer.artifact(self.run_id, stage, "initial_input", uri, sha256=digest, bytes=size, version_id=version_id, ownership="initial_external", retention_class="initial", presence="present" if present else "missing")
                if digest is not None:
                    observed_inputs.append({"file_id": file_id, "role": f"source_observation:{ordinal}", "sha256": digest})
                if str(path) not in self.document_tasks and not self.adapter_profile:
                    document_task = writer.task(self.run_id, stage, operation + ".document_observation", {"compatibility_profile": "legacy_shadow", "reuse_eligible": False}, version_id=version_id, ordinal=ordinal + 1)
                    self.document_tasks[str(path)] = (document_task, version_id)
                    writer.event(self.run_id, event_kind="lifecycle", reason_code="stage_planned", status="planned", task_id=document_task, version_id=version_id)
            source_inventory_hash = writer.snapshot("inputs", {"kind": "inputs", "files": observed_inputs})
            self.task_id = writer.task(self.run_id, stage, operation, {"compatibility_profile": 'sequential_cutover' if self.cutover else "legacy_shadow", "required_output_kinds": [], "parameters": self.parameters, "source_inventory_sha256": source_inventory_hash}, scope_key="corpus:" + source_inventory_hash)
            self.attempt_id = writer.attempt(self.task_id, worker_id=f"orchestrator:{os.getpid()}", device_id=str(parameters.get("accel_type", "unknown_not_recorded")),
                fingerprint={"reuse_eligible": False, "provenance": "shadow_observation", "code_revision": self.revision, "source_inventory_sha256": source_inventory_hash}, inputs={"files": observed_inputs}, settings=self.parameters, tools=loaded_tools())
            for input_reference in observed_inputs:
                writer.input(self.attempt_id, input_reference["file_id"], input_reference["role"], input_reference["sha256"])
            writer.event(self.run_id, event_kind="lifecycle", reason_code="stage_planned", status="planned", task_id=self.task_id)
            writer.event(self.run_id, event_kind="lifecycle", reason_code="stage_started", status="running", task_id=self.task_id, attempt_id=self.attempt_id, started_at=self.started_at)
            for name, receipt in self.before.items():
                self._register(Path(name), receipt, producer=None)
        # Capture a pre-existing store before its first legacy mutation. New
        # stores start from an empty row set; subsequent calls append deltas.
        for name in self.before:
            self.store(Path(name), kind="legacy_preexisting", preexisting=True)
        original_frame = getattr(corpus, "_shadow_original_source_frame", None)
        if original_frame is not None and corpus.metadata_path is not None:
            self.source_rows(original_frame, Path(corpus.metadata_path), url_column=corpus.url_column)

    def inventory(self):
        produced = set()
        if self.cutover:
            from ..stages.scheduler import local_path
            produced = {str(local_path(row['uri'])) for row in self.writer.file_catalog().values()
                if row['producer_attempt_id'] is not None}
        return inventory(self.corpus.output_dir, produced)

    def _register(self, path, receipt, *, producer, kind=None):
        version_id = self.versions.get(str(path.absolute()))
        # Unbound compatibility files stay aggregate observations. A nearby
        # source, or even a singleton source set, is not evidence of ownership.
        relative = str(path.relative_to(self.corpus.output_dir.absolute())) if path.is_relative_to(self.corpus.output_dir.absolute()) else str(path)
        uri = path.absolute().as_uri()
        previous = self.writer.connection.execute(
            "SELECT file_id,version_id FROM current_file_state WHERE uri=? AND sha256=? AND bytes=? AND presence='present' ORDER BY event_id DESC LIMIT 1",
            (uri, receipt["sha256"], receipt["bytes"])).fetchone()
        if previous is not None:
            self.files[str(path.absolute())] = (previous[0], previous[1], receipt)
            return previous[0]
        file_id = self.writer.artifact(self.run_id, self.stage, kind or relative, uri, **receipt,
            version_id=version_id, producer_attempt_id=producer, ownership="managed" if producer else "legacy_external",
            retention_class="evidence" if path.suffix in (".json", ".jsonl", ".parquet") else "intermediate")
        self.files[str(path.absolute())] = (file_id, version_id, receipt)
        return file_id

    def observe(self, reason_code, *, path=None, error=None, status=None, message="", **details):
        if self.cutover and status == 'pending' and error is None:
            return  # Native progress is not a failed contractual attempt.
        if (self.adapter_profile and self.stage in self.adapter_stages
                and reason_code.startswith(self.stage + "_") and error is None
                and reason_code not in ("math_json_missing", "math_pdf_missing")):
            # The producer result already owns these decisions. Warnings for
            # non-adapter dependencies (e.g. page_quality_invalid) stay shadow.
            # Missing math inputs are detected before invoking an adapter, so
            # another document's producer cannot own those observations.
            return
        reported_error = details.pop("reported_error", None)
        receipt = {"observation": "current_branch", "operation": self.operation, "path": str(path) if path is not None else None,
                   "backend_status": status, "retryable": details.pop("retryable", None), **payload_evidence(details)}
        if error is not None:
            error_type = type(error).__module__ + "." + type(error).__name__
            trace = "".join(traceback_module.format_exception(type(error), error, error.__traceback__))
            message = message or str(error)
        elif reported_error is not None:
            error_type, trace = reported_error["error_type"], reported_error["traceback"]
            message = message or reported_error["message"]
        else: error_type, trace = None, None
        with self.writer.transaction():
            evidence_hash = self.branch_snapshot({"kind": "branch_evidence", **receipt})
            self.writer.event(self.run_id, event_kind="decision", reason_code=reason_code, task_id=self.task_id, attempt_id=self.attempt_id,
                message=message, error_type=error_type, traceback=trace, details={"evidence_sha256": evidence_hash})
        if path is not None:
            exact_path = str(Path(path).absolute())
            if exact_path in self.document_tasks and status in ("ok", "skipped", "failed"):
                self.document_outcomes.setdefault(exact_path, []).append((status, reason_code, error_type, trace, message))
        if status == "failed": self.failures.append(reason_code)
        elif status == "skipped": self.skips.append(reason_code)
        elif status == "ok": self.successes.append(reason_code)

    def branch_snapshot(self, value):
        value = dict(value)
        for field, members in list(value.items()):
            if (isinstance(members, list) and members and
                    (len(members) >= 8 or field in ("stems", "legacy_stems", "processed", "problematic")) and
                    all(isinstance(v, str) for v in members) and members == sorted(set(members))):
                scope = canonical_json([str(value.get("path") or self.corpus.output_dir.absolute().as_uri()), field])
                digest = membership_snapshot(self.writer, scope, members)
                value.pop(field)
                value.setdefault("shared_fields", {})[field] = digest
        return self.writer.snapshot("branch_evidence", pack_evidence(value))

    def store(self, path, *, kind, preexisting=False, register=True):
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file(): self.store(child, kind=kind, preexisting=preexisting, register=register)
            return
        if not path.is_file(): return
        if self.adapter_profile and path.absolute().is_relative_to(Path(self.corpus.output_dir).absolute()/'.glossapi'):
            # A stage's private attempt file (for example the classified parquet the annotate kernel writes in
            # its staging directory): its producer registers it, as inventory() never observes .glossapi either.
            return
        if self.cutover and path.parent == self.corpus.output_dir/'download_results' and path.suffix == '.parquet':
            from ..stages.products import current_product
            if not preexisting or current_product(self.writer.connection,path):
                return  # Producer generation owns the serialized values.
        if self.adapter_profile and self.stage in ("clean", "ocr", "math") and path.suffix == ".parquet" and not preexisting and not getattr(self,"finishing",False):
            return  # Final compatibility values are recorded after their producer facts.
        receipt = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        uri = path.absolute().as_uri()
        states = store_states(self.writer)
        previous = states.get(uri)
        if previous and previous["sha256"] == receipt["sha256"]:
            return
        if self.adapter_profile and kind == "clean_report" and "clean" in self.adapter_stages:
            return
        if self.adapter_profile and self.writer.connection.execute(
                "SELECT 1 FROM files WHERE uri=? AND sha256=? AND producer_attempt_id IS NOT NULL LIMIT 1",
                (uri, receipt["sha256"])).fetchone():
            return
        if self.cutover and self.writer.connection.execute(
                'SELECT 1 FROM files WHERE uri=? AND producer_attempt_id IS NOT NULL LIMIT 1',
                (path.absolute().as_uri(),)).fetchone():
            return
        read_error = None
        try:
            decoded = read_store(path, payload_evidence, kind=kind)
        except Exception as caught:
            read_error = type(caught).__name__
            decoded = ({"format": "unreadable", "read_error_type": type(caught).__name__,
                        "read_error_message": str(caught)}, {})
        if self.adapter_profile and decoded is not None and decoded[0].get("format") == "parquet":
            schema, rows = decoded
            columns=[column.get("value") for column in schema["columns"]]
            fields_by_stage={
                "clean": {"mojibake_badness_score", "mojibake_latin_percentage", "percentage_greek", "latin_percentage",
                    "greek_badness_score", "greek_latin_percentage", "polytonic_ratio", "rejection_reason", "filter",
                    "needs_ocr", "char_count_no_comments", "is_empty", "latin_word_shape_rate", "ocr_pages", "ocr_pages_count",
                    "ocr_pages_unflagged_count", "page_rule_profile", "page_fold_validity", "ocr_noise_suspect", "ocr_noise_flags",
                    "ocr_repeat_phrase_run_max", "ocr_repeat_line_run_max", "ocr_repeat_suspicious_line_count", "ocr_repeat_suspicious_line_ratio"},
                "ocr": {"text", "ocr_success", "ocr_markdown_relpath", "ocr_metrics_relpath", "ocr_text_sha256", "extraction_mode"},
                "math": {"math_enriched", "enriched_math", "math_items", "math_accepted", "math_accept_rate", "math_time_sec"}}
            owners={}
            fields_by_operation={
                "clean.initial": {"mojibake_badness_score","mojibake_latin_percentage","percentage_greek","latin_percentage","greek_badness_score","greek_latin_percentage","polytonic_ratio","rejection_reason","char_count_no_comments","is_empty"},
                "clean.selection": {"filter","needs_ocr","latin_word_shape_rate","ocr_pages","ocr_pages_count","ocr_pages_unflagged_count","page_rule_profile","page_fold_validity"},
                "clean.ocr_artifacts": {"percentage_greek","latin_percentage","polytonic_ratio","ocr_noise_suspect","ocr_noise_flags","ocr_repeat_phrase_run_max","ocr_repeat_line_run_max","ocr_repeat_suspicious_line_count","ocr_repeat_suspicious_line_ratio"},
                "math.triage": {"formula_total","formula_avg_pp","formula_p90_pp","pages_with_formula","pages_total","phase_recommended"}}
            fields_by_operation["clean.score_after_ocr"]=fields_by_operation["clean.initial"]
            for stage,operation,attempt_id,encoded in self.writer.connection.execute(
                    f"SELECT t.stage,t.operation,a.attempt_id,s.json FROM attempts a JOIN tasks t ON t.task_id=a.task_id JOIN snapshots s ON s.sha256=a.settings_sha256 WHERE t.stage IN ('clean','ocr','math') AND EXISTS (SELECT 1 FROM {self.writer.connection.event_table} e WHERE e.attempt_id=a.attempt_id AND e.event_kind='lifecycle' AND e.status='ok') ORDER BY a.rowid"):
                attempt_settings=json.loads(encoded)
                # An aggregate clean selection (K4(c) part 3) owns its fields for every filename it selected.
                filenames=attempt_settings.get("compatibility_filenames") or [attempt_settings.get("compatibility_filename")]
                for filename in filenames:
                    if not filename:continue
                    for field in fields_by_operation.get(operation.split(".artifact:",1)[0],fields_by_stage[stage]): owners[(filename,field)]=attempt_id
            filename_position=columns.index("filename") if "filename" in columns else None
            if owners and filename_position is not None:
                rows={key:{**row,"values":[
                    {"producer_attempt_id":owners[(row["values"][filename_position] if isinstance(row["values"][filename_position],str) else None,column)],"field":column}
                    if (row["values"][filename_position] if isinstance(row["values"][filename_position],str) else None,column) in owners else value
                    for column,value in zip(columns,row["values"])]} for key,row in rows.items()}
                schema={**schema,"adapter_owned_values":"producer_reference_v1; facts are recorded only by their producer"}
                decoded=schema,rows
        self._store_observation(path, kind, receipt, decoded, read_error, preexisting=preexisting, register=register)

    def _store_observation(self, path, kind, receipt, decoded, read_error=None, *, preexisting=False, register=True):
        """Commit local or forwarded rows; workers never open this database."""
        uri = path.absolute().as_uri()
        states = store_states(self.writer)
        previous = states.get(uri)
        if previous and previous["sha256"] == receipt["sha256"]:
            return
        with self.writer.transaction():
            if register:
                self._register(path.absolute(), receipt, producer=None, kind=kind)
            if decoded is None:
                return  # Ordinary PDF/Markdown/JSONL/Docling bodies stay files.
            schema, rows = decoded
            schema_hash = self.writer.snapshot("store_schema", {"kind": "store_schema", **schema})
            old_rows = previous["rows"] if previous else {}
            if read_error:
                rows = old_rows
            delta = {"kind": kind, "encoding": "row_delta_v1", "uri": uri, **receipt,
                     "row_count": None if read_error else len(rows), "schema_sha256": schema_hash,
                     "row_count_unavailable_reason": read_error,
                     "previous_sha256": previous["digest"] if previous else None,
                     "mode": "first_import" if preexisting and previous is None else "delta",
                     "changed": {key: row for key, row in rows.items() if key not in old_rows or old_rows[key] != row},
                     "deleted": sorted(old_rows.keys() - rows.keys()), "binding": "observed_not_verified"}
            digest = self.writer.snapshot("legacy_store", delta)
            self.writer.event(self.run_id, event_kind="import", reason_code="legacy_observed", task_id=self.task_id, attempt_id=self.attempt_id, details={"evidence_sha256": digest})
        # Publish cache only after commit; a rolled-back write cannot become a base.
        states[uri] = {"digest": digest, "rows": rows, "schema_sha256": schema_hash, "sha256": receipt["sha256"]}

    def source_rows(self, frame, path, *, url_column):
        columns = [{"name": str(name), "dtype": str(dtype)} for name, dtype in frame.dtypes.items()]
        frame_digest = hashlib.sha256(canonical_json(columns).encode("utf-8"))
        for index, row in dataframe_records(frame):
            frame_digest.update(canonical_json([typed_value(index), typed_value(row)]).encode("utf-8"))
        frame_hash = frame_digest.hexdigest()
        with self.writer.transaction():
            manifest_hash = file_hash(path) if path.is_file() else None
            for original_digest, encoded in self.writer.connection.execute("SELECT sha256,json FROM snapshots WHERE kind='source_manifest'"):
                original = json.loads(encoded)
                if original.get("uri") == path.absolute().as_uri() and original.get("frame_sha256") == frame_hash and original.get("url_column") == url_column:
                    self.writer.event(self.run_id, event_kind="import", reason_code="legacy_observed", task_id=self.task_id,
                        attempt_id=self.attempt_id, details={"source_manifest_sha256": original_digest,
                            "current_file_sha256": manifest_hash, "binding": "original_frame_reobserved_current_bytes_not_verified"})
                    return
            previous_file = self.writer.connection.execute("SELECT file_id FROM files WHERE uri=? AND sha256 IS ? ORDER BY rowid DESC LIMIT 1", (path.absolute().as_uri(), manifest_hash)).fetchone()
            manifest_file = previous_file[0] if previous_file else self.writer.artifact(self.run_id, self.stage, "source_manifest", path.absolute().as_uri(), sha256=manifest_hash, bytes=path.stat().st_size if path.is_file() else None, ownership="initial_external", retention_class="initial", presence="present" if path.is_file() else "missing")
            manifest = self.writer.snapshot("source_manifest", {"kind": "source_manifest", "uri": path.absolute().as_uri(), "original_file_sha256": manifest_hash, "original_file_id": manifest_file, "binding": "legacy_dataframe_and_current_manifest_bytes_observed_not_verified",
                "columns": columns, "frame_sha256": frame_hash, "url_column": url_column, "row_count": len(frame), "row_indices": [typed_value(index) for index in frame.index],
                "metadata_home": "source_records.metadata_json; source_row is the occurrence ordinal"})
            if manifest in self.source_manifests or self.writer.connection.execute("SELECT 1 FROM source_records WHERE manifest_sha256=? LIMIT 1", (manifest,)).fetchone(): return
            self.source_manifests.add(manifest)
            for ordinal, (index, row) in enumerate(dataframe_records(frame)):
                values = row; uri = values.get(url_column)
                if not isinstance(uri, str) or not uri.strip():
                    uri = path.absolute().as_uri() + f"#row={ordinal}"
                    kind = "local_path"
                else: kind = "url"
                document_id = self.writer.document(source_key(uri))
                self.writer.source_record(document_id, self.run_id, uri, values, source_kind=kind, manifest_sha256=manifest, source_row=str(ordinal))

    def verify_cached_originals(self):
        """Byte-verify every original whose hash this phase took from the process cache, before a destructive step
        (retention) relies on it (CP-K4b F3): a same-stat rewrite that the cache could not see stops the phase."""
        for path, digest in getattr(self, 'cached_originals', ()):
            if not Path(path).is_file() or file_hash(path) != digest:
                _original_hashes.pop(str(path), None)
                raise ValueError(f'original_changed_since_observed: {path} no longer has the hash this phase bound; run the phase again')

    def finish(self, result, error):
        if self.cutover and error is None:
            from ..stages.metadata import seal_metadata_projections
            seal_metadata_projections(self)
        self.finishing = True
        after = self.inventory()
        for name, receipt in after.items():
            if self.before.get(name) != receipt:
                # Terminal output registration below owns this file fact. Early
                # hooks still register transient observations before cleanup;
                # the final inventory must not create a duplicate unowned fact.
                self.store(Path(name), kind=str(Path(name).relative_to(self.corpus.output_dir.absolute())), register=False)
        with self.writer.transaction():
            for name, (file_id, version_id, receipt) in self.files.items():
                if not Path(name).exists() and self.writer.connection.execute("SELECT presence FROM current_file_state WHERE file_id=?", (file_id,)).fetchone()[0] != "deleted":
                    self.writer.event(self.run_id, event_kind="artifact", reason_code="missing", version_id=version_id, file_id=file_id,
                        details={"observation": "absent_after_wrapper", "policy_deletion": "unknown_not_recorded"})
            status = "failed" if error is not None or self.failures else "skipped" if self.skips and not self.successes else "ok"
            reason = self.stage + "_exception" if error is not None else self.failures[0] if self.failures else self.skips[0] if status == "skipped" else self.stage + "_completed"
            if self.cutover and error is None:
                section_aggregate_published = False
                for adapter_attempt in self.adapter_attempts:
                    control=self.writer.control();adapter=control['attempts'].get(adapter_attempt)
                    outcome={**control['attempt_events'].get(adapter_attempt,{}),'operation':control['tasks'][adapter['task_id']]['operation']} if adapter else None
                    if outcome and outcome.get('status') not in ('ok','skipped','failed'):outcome=None
                    if outcome and outcome['status']=='failed':status,reason='failed',outcome['reason_code'];break
                    if outcome and outcome['status']=='ok' and outcome['operation']=='sections.publish':
                        section_aggregate_published = True
                if status=='skipped' and self.stage=='sections' and section_aggregate_published:
                    # Omitted documents remain explicit skipped decisions. The
                    # committed aggregate proves successful section publication,
                    # including when its document fragments were replayed and
                    # therefore emitted no fresh backend success observation.
                    status,reason='ok','sections_completed'
            trace = "".join(traceback_module.format_exception(type(error), error, error.__traceback__)) if error else None
            for source_path, (document_task, version_id) in self.document_tasks.items():
                outcomes = self.document_outcomes.get(source_path, [])
                observed_outcome = next((item for item in outcomes if item[0] == "failed"), outcomes[-1] if outcomes else None)
                document_attempt = self.writer.attempt(document_task, worker_id="observed_by_orchestrator", device_id="unknown_not_recorded",
                    fingerprint={"reuse_eligible": False}, inputs={"files": []}, settings=self.parameters, tools=loaded_tools())
                document_status, document_reason, document_error_type, document_trace, document_message = observed_outcome or (
                    "skipped", "unknown_not_recorded", None, None, "Per-document execution outcome was not reported by the legacy wrapper")
                self.writer.finish(self.run_id, document_task, document_attempt, status=document_status, reason_code=document_reason,
                    error_type=document_error_type, traceback=document_trace, message=document_message,
                    details={"mode": "shadow", "reuse_eligible": False, "wrapper_attempt_id": self.attempt_id,
                             "measurement_basis": "legacy_branch_observation", "duration_unavailable_reason": "per_document_timing_not_reported"})
            outputs = [{"kind": str(Path(name).relative_to(self.corpus.output_dir.absolute())) if Path(name).is_relative_to(self.corpus.output_dir.absolute()) else name, "uri": Path(name).as_uri(), **receipt} for name, receipt in after.items() if self.before.get(name) != receipt and not (
                self.adapter_profile and self.writer.connection.execute(
                    "SELECT 1 FROM files WHERE uri=? AND sha256=? AND producer_attempt_id IS NOT NULL LIMIT 1",
                    (Path(name).as_uri(),receipt["sha256"])).fetchone())]
            self.writer.finish(self.run_id, self.task_id, self.attempt_id, status=status, reason_code=reason, outputs=outputs,
                started_at=self.started_at, finished_at=utc_now(), duration_ns=time.monotonic_ns()-self.started_ns,
                error_type=type(error).__module__ + "." + type(error).__name__ if error else None, traceback=trace, message=str(error) if error else "Legacy wrapper observation",
                details={"wrapper_return": wrapper_return_receipt(result), "wrapper_raised": error is not None, "mode": self.mode, "reuse_eligible": False,
                         "tools_sha256": self.writer.snapshot("tools", {"kind": "tools", **loaded_tools()}),
                         "observed_files": [{"file_id": file_id} for file_id, _, _ in self.files.values()],
                         "unavailable": ['unknown_required_tool_identities_prevent_reuse'] if self.cutover else ["native_rule_spans", "unreported_worker_substeps", "unreported_model_identity"],
                         "measurement_basis": "wrapper_and_observation_wall_monotonic_ns", "cpu_ns": None, "peak_rss_bytes": None})


def shadow_stage(stage, operation=None):
    """Signature/return-preserving boundary hook, with writer lifetime per call."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def recorded(self, *args, **kwargs):
            from ..stages.profile import _public_adapter_scope
            from ..stages.workers import warm_worker_scope
            profile_token = _public_adapter_scope.set(adapters_enabled(self))
            from .encoding import run_hash_scope
            try:
                # One public call is one run (or a streaming run's publication): own-file hashes are reused only
                # inside it (K4C-G08).
                with warm_worker_scope(), run_hash_scope():
                    return invoke(self, *args, **kwargs)
            finally:
                _public_adapter_scope.reset(profile_token)

        def invoke(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            parameters = dict(bound.arguments)
            parameters.pop("self", None)
            parent = _active_journal.get()
            # Existing spawned backend workers send their results to the parent.
            # Never open a second writable ledger for the same corpus there.
            if not getattr(self, "_shadow_ledger_enabled", False) or isinstance(parent, WorkerObservations):
                from ..stages.profile import caller_kernel_override
                if not isinstance(parent,WorkerObservations) and caller_kernel_override(self,stage):
                    from ..stages.caller_kernel import execute_caller_kernel
                    return execute_caller_kernel(self,stage,operation or f'{stage}.{function.__name__}',lambda:function(self,*args,**kwargs))
                return function(self, *args, **kwargs)
            provided_writer=getattr(self,'_scheduler_writer',None)
            own_writer = provided_writer is None and (parent is None or parent.corpus.output_dir != self.output_dir)
            writer = provided_writer or (LedgerWriter(Path(self.output_dir) / "corpus.sqlite3", code_revision=code_revision()) if own_writer else parent.writer)
            try:
                if own_writer:
                    # A phase that owns its writer starts a run: verify the copies the previous run close left unflushed (K4C-M2).
                    writer.verify_excluded_copies()
                journal = ResultJournal(writer, self, stage, operation or f"{stage}.{function.__name__}", parameters)
                if own_writer:
                    # This phase owns the writer: its close is the run-close barrier.
                    writer.barrier_context = {'run_id': journal.run_id, 'debug': journal.retention_settings.get('debug', 'off')}
                    writer.retained_intermediates = journal.retention_settings.get('debug', 'off') == 'on'
                token = _active_journal.set(journal)
                result, error = None, None
                try:
                    if journal.caller_override:
                        from ..stages.caller_kernel import execute_caller_kernel
                        result=execute_caller_kernel(self,stage,operation or f'{stage}.{function.__name__}',lambda:function(self,*args,**kwargs))
                    else:
                        result = function(self, *args, **kwargs)
                    return result
                except BaseException as caught:
                    error = caught
                    raise
                finally:
                    try: journal.finish(result, error)
                    finally: _active_journal.reset(token)
                    if own_writer and journal.cutover and error is None:
                        journal.verify_cached_originals()
                        from ..stages.retention import retain_completed_run
                        retain_completed_run(writer,self.output_dir,journal.run_id,journal.retention_settings,cleanup_paths=getattr(journal,"cleanup_paths",()),observer=getattr(self,"_retention_observer",None))
            finally:
                if own_writer: writer.close()
        return recorded
    return decorate


class WorkerObservations:
    """Serializable results attached to the existing worker result messages."""
    def __init__(self):
        self.owner_pid = os.getpid()
        self.receipts = []

    def observe(self, reason_code, *, path=None, error=None, status=None, message="", **details):
        self.receipts.append({"reason_code": reason_code, "path": str(path) if path else None, "status": status,
            "message": message or (str(error) if error else ""),
            "error_type": type(error).__module__ + "." + type(error).__name__ if error else None,
            "traceback": "".join(traceback_module.format_exception(type(error), error, error.__traceback__)) if error else None,
            "details": {"worker_pid": os.getpid(), "visible_device_ids": os.environ.get("CUDA_VISIBLE_DEVICES"), **payload_evidence(details)}})

    def store(self, path, *, kind):
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file(): self.store(child, kind=kind)
            return
        if path.is_file():
            fact = {"sha256": file_hash(path), "bytes": path.stat().st_size}
            read_error = None
            try:
                decoded = read_store(path, payload_evidence, kind=kind)
            except Exception as caught:
                read_error = type(caught).__name__
                decoded = ({"format": "unreadable", "read_error_type": read_error,
                            "read_error_message": str(caught)}, {})
            self.observe("legacy_observed", path=path, store_kind=kind, **fact)
            # Transport observed rows before legacy cleanup can erase them.
            # Only the owning writer computes/persists the committed row delta.
            self.receipts[-1]["store_observation"] = {"decoded": decoded, "read_error": read_error}

    def source_rows(self, frame, path, *, url_column):
        self.observe("legacy_observed", path=path, source_rows=[{"index":typed_value(index), "fields":typed_value(row)} for index,row in dataframe_records(frame)])


def start_worker_observations():
    if not isinstance(_active_journal.get(), WorkerObservations):
        raise RuntimeError("Worker observations require a scoped worker boundary")


def worker_observation_scope(function):
    @wraps(function)
    def recorded(*args, **kwargs):
        token = _active_journal.set(WorkerObservations())
        try:
            return function(*args, **kwargs)
        finally:
            _active_journal.reset(token)
    return recorded


def drain_worker_observations():
    collector = _active_journal.get()
    if not isinstance(collector, WorkerObservations): return []
    receipts, collector.receipts = collector.receipts, []
    return receipts


def accept_worker_observations(receipts):
    journal = _active_journal.get()
    if not isinstance(journal, ResultJournal): return
    for receipt in receipts:
        forwarded_store = receipt.get("store_observation")
        if forwarded_store is not None:
            journal._store_observation(Path(receipt["path"]), receipt["details"]["store_kind"],
                {key: receipt["details"][key] for key in ("sha256", "bytes")},
                forwarded_store["decoded"], forwarded_store["read_error"])
        with journal.writer.transaction():
            details = dict(receipt["details"])
            if "source_rows" in details:
                details["source_rows_sha256"] = journal.writer.snapshot("worker_source_rows", {
                    "kind": "worker_source_rows", "uri": receipt["path"], "rows": details.pop("source_rows")})
            evidence_hash = journal.branch_snapshot({"kind": "branch_evidence",
                "measurement_basis": "worker_report", "path": receipt["path"],
                "backend_status": receipt["status"], **details})
            journal.writer.event(journal.run_id, event_kind="decision", reason_code=receipt["reason_code"],
                task_id=journal.task_id, attempt_id=journal.attempt_id, message=receipt["message"],
                error_type=receipt["error_type"], traceback=receipt["traceback"],
                details={"measurement_basis": "worker_report", "evidence_sha256": evidence_hash})
        if receipt.get("path"):
            exact_path = str(Path(receipt["path"]).absolute())
            if exact_path in journal.document_tasks and receipt["status"] in ("ok", "skipped", "failed"):
                journal.document_outcomes.setdefault(exact_path, []).append((receipt["status"], receipt["reason_code"], receipt["error_type"], receipt["traceback"], receipt["message"]))
        if receipt["status"] == "failed": journal.failures.append(receipt["reason_code"])
        elif receipt["status"] == "ok": journal.successes.append(receipt["reason_code"])
        elif receipt["status"] == "skipped": journal.skips.append(receipt["reason_code"])


def observe_http_response(response, requested_url, *, retry_index=None, source_entity_bytes=None):
    """Allowlisted response facts captured before validation/cleanup can hide them."""
    from ..stages.receipts import response_receipt
    response_receipt(response, requested_url, retry_index)
    headers = {str(key).lower(): str(value) for key, value in (getattr(response, "headers", {}) or {}).items()
               if str(key).lower() in ("content-type", "content-disposition", "content-length", "content-encoding", "etag", "last-modified", "retry-after")}
    redirects = [{"url": str(getattr(item, "url", "")), "status": getattr(item, "status", None),
                  "location": (getattr(item, "headers", {}) or {}).get("Location")} for item in (getattr(response, "history", ()) or ())]
    observe("legacy_observed", http={"requests": [{"requested_url": requested_url,
        "final_url": str(getattr(response, "url", requested_url)), "http_status": getattr(response, "status", None),
        "redirects": redirects, "response_headers": headers, "retry_index": retry_index,
        "route": "unknown_not_recorded", "policy_hash": None, "sequence": None, "started_at": None, "finished_at": None, "backoff_ms": None, "retry_after_ms": None, "error": None, "timing_unavailable_reason": "legacy_request_did_not_report_timing",
        "source_entity_bytes": source_entity_bytes, "wire_bytes": None}]})


def observe_acquisition(url, path, *, source_observable):
    from ..stages.receipts import acquired_receipt
    acquired_receipt(url, Path(path), source_observable)
    journal = _active_journal.get()
    if not isinstance(journal, ResultJournal): return
    path = Path(path).absolute()
    digest = file_hash(path)
    with journal.writer.transaction():
        document = journal.writer.document(source_key(url))
        version = journal.writer.version(document, source_sha256=digest if source_observable else None, stored_sha256=digest)
        journal.versions[str(path)] = version
        journal.writer.artifact(journal.run_id, "download", "download_payload", path.as_uri(), sha256=digest,
            bytes=path.stat().st_size, version_id=version, producer_attempt_id=journal.attempt_id, retention_class="initial")
        journal.writer.event(journal.run_id, event_kind="decision", reason_code="legacy_observed", version_id=version,
            details={"requested_url":url,"stored_sha256":digest,"source_sha256":digest if source_observable else None,
                     "source_hash_unavailable_reason":None if source_observable else "original_entity_not_observable",
                     "stored_bytes":path.stat().st_size,"wire_bytes":None})
