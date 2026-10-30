"""Private workspaces and a hard deadline spanning conversion AND export.

There is deliberately no pool, scheduler, resume policy or retention here.
"""
from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time
import traceback

from .contracts import OutputArtifact, StageError, StageResult

_producer_hashes=ContextVar('glossapi_producer_hashes',default=None)


@contextmanager
def producer_hash_scope():
    token=_producer_hashes.set({})
    try:yield
    finally:_producer_hashes.reset(token)


def remember_written_hash(path,digest,size):
    values=_producer_hashes.get()
    if values is not None:values[str(Path(path).absolute())]=(digest,size)
    return digest,size


def produced_hash(path):
    values=_producer_hashes.get()
    return values.get(str(Path(path).absolute())) if values is not None else None


def write_text_hashed(path,text,encoding='utf-8'):
    data=text.encode(encoding)
    with Path(path).open('wb') as stream:stream.write(data)
    remember_written_hash(path,hashlib.sha256(data).hexdigest(),len(data))
    return len(text)


def sha256_file(path):
    from ..ledger.encoding import file_hash
    return file_hash(path)


def seal_output_hashes(result, root):
    """Worker-owned byte receipts; no stat identity cache supplies content proof."""
    outputs=[]
    for output in result.outputs:
        path=checked_path(root,output.relative_path)
        written=produced_hash(path)
        digest,size=written if written else (output.sha256 or sha256_file(path),path.stat().st_size)
        remember_written_hash(path,digest,size)
        outputs.append(replace(output,sha256=digest,bytes=size))
    return replace(result,outputs=tuple(outputs))


def copy_hashed(source, destination):
    """Hash the bytes being copied, without reading the new product again."""
    digest=hashlib.sha256();size=0
    with Path(source).open('rb') as reader, Path(destination).open('wb') as writer:
        for block in iter(lambda:reader.read(1024*1024),b''):
            writer.write(block);digest.update(block);size+=len(block)
    return remember_written_hash(destination,digest.hexdigest(),size)


def checked_path(root, relative):
    relative = Path(relative)
    root = Path(root).resolve()
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Artifact path must be a contained relative path")
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("Artifact symlink escapes attempt directory")
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Artifact symlinks are not publishable")
    return candidate


def outputs_in(root, required=()):
    outputs = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("Worker produced a symlink")
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if path.name == "worker-result.json":
            kind = "worker_receipt"
        elif relative.startswith("inputs/"):
            kind = "source_snapshot"
        elif relative.startswith("downloads/"):
            kind = "download_payload"
        elif relative.startswith("repaired_pdfs/"):
            kind = "repaired_pdf"
        elif ".docling.json" in path.name:
            kind = "docling_json"
        elif ".formula_index." in path.name:
            kind = "formula_index"
        elif ".page_quality." in path.name:
            kind = "page_quality"
        elif path.name.endswith(".chunks.json"):
            kind = "chunk_manifest"
        elif relative.startswith("markdown/chunks/"):
            kind = "markdown_chunk"
        elif path.suffix == ".md":
            kind = "raw_markdown"
        elif relative.startswith("sidecars/extract/"):
            kind = "extract_sidecar"
        elif relative.startswith("json/metrics/"):
            kind = "extract_metrics"
        else:
            kind = "attempt_diagnostic"
        outputs.append(OutputArtifact(kind, relative, None,
            "initial" if kind == "download_payload" else "intermediate", kind in required))
    return tuple(outputs)


def failed_result(request, reason, error, *, decisions=(), metrics=None):
    import errno
    if isinstance(error,OSError) and error.errno==errno.ENOSPC:reason='disk_budget_exceeded'
    return StageResult(request.task_id, request.attempt_id, request.lease_token, "failed", reason,
        tuple(replace(output, retention_class="failed_attempt", required=False)
              for output in outputs_in(request.work_directory)), tuple(decisions), metrics or {},
        StageError(type(error).__module__ + "." + type(error).__name__, str(error),
                   "".join(traceback.format_exception(error)), isinstance(error, TimeoutError)))


class ReceiptCollector:
    """No SQLite. A spawned worker writes bounded receipts to its own workspace."""
    def heartbeat(self, lease_token, progress):
        pass

    def evidence(self, lease_token, receipt):
        pass


def _worker(stage, request, connection):
    # A process group also fences any backend/export subprocesses it creates.
    if hasattr(os, "setsid"):
        os.setsid()
    scratch = request.work_directory / "scratch"
    scratch.mkdir()
    # Third-party tempfile users (including browser subprocesses) must share
    # the attempt boundary too, rather than writing to the system temp root.
    os.environ.update(TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    import tempfile
    tempfile.tempdir = str(scratch)
    os.environ["GLOSSAPI_CONVERSION_LOG_FILE"] = str(request.work_directory / "conversion.log")
    try:
        with producer_hash_scope():
            result = stage._run_document(request, reporter=ReceiptCollector())
            result = seal_output_hashes(result,request.work_directory)
        from ..ledger.journal import loaded_tools
        result = seal_output_hashes(replace(result, metrics={**result.metrics, "tools": loaded_tools()}),request.work_directory)
        # Write the large result to the private directory, not a pipe whose
        # capacity could deadlock join(). The pipe carries one small sentinel.
        from dataclasses import asdict
        target = request.work_directory / "worker-result.json"
        from ..ledger.urls import parse_policy, scrub, scrub_known_value
        encoded = json.dumps(scrub_known_value(scrub(asdict(result), parse_policy(request.settings.get("url_policy")))), ensure_ascii=False)
        write_text_hashed(target, encoded)
        data = encoded.encode('utf-8')
        connection.send((hashlib.sha256(data).hexdigest(), len(data)))
    finally:
        connection.close()


def run_fenced(stage, request, *, reporter):
    """Spawn, terminate/kill the process group, then reap before inspecting files."""
    from .workers import warm_fenced
    warm = warm_fenced(stage, request, reporter)
    if warm is not None:
        return warm
    request.work_directory.mkdir(parents=True, exist_ok=False)
    remaining = (request.deadline_monotonic_ns - time.monotonic_ns()) / 1e9
    if remaining <= 0:
        return failed_result(request, stage.name + "_timeout", TimeoutError("Attempt deadline expired"))
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(stage, request, writer))
    process.start()
    writer.close()
    next_heartbeat = time.monotonic()
    try:
        while process.is_alive() and time.monotonic_ns() < request.deadline_monotonic_ns:
            process.join(min(0.1, max(0, (request.deadline_monotonic_ns-time.monotonic_ns()) / 1e9)))
            if time.monotonic() >= next_heartbeat:
                reporter.heartbeat(request.lease_token, {"worker_pid": process.pid})
                next_heartbeat = time.monotonic() + 5
        expired = process.is_alive()
        if expired:
            _stop_process(process)
            return failed_result(request, stage.name + "_timeout", TimeoutError("Conversion/export deadline expired; worker reaped"))
        try:
            received = process.exitcode == 0 and reader.poll() and reader.recv()
        except EOFError:
            received = False
        if not received:
            return failed_result(request, "worker_process_crashed", RuntimeError(f"Worker exited {process.exitcode}"))
        value = json.loads((request.work_directory / "worker-result.json").read_text())
        value["outputs"] = tuple(OutputArtifact(**{**item, "page_range": tuple(item["page_range"]) if item["page_range"] else None}) for item in value["outputs"])
        value["decisions"] = tuple(value["decisions"])
        value["error"] = StageError(**value["error"]) if value["error"] else None
        result = StageResult(**value)
        digest, size = received
        result = replace(result, outputs=(*result.outputs, OutputArtifact("worker_receipt", "worker-result.json", None, "evidence", False, digest, size)))
        if (result.task_id, result.attempt_id, result.lease_token) != (request.task_id, request.attempt_id, request.lease_token):
            raise ValueError("Stale worker result")
        for decision in result.decisions:
            reporter.evidence(request.lease_token, decision)
        return result
    finally:
        if process.is_alive():
            _stop_process(process)
        reader.close()
        process.close()


def _stop_process(process):
    # A browser/exporter can detach into another process group, or ignore TERM.
    # Capture Linux descendants before their parent exits and they are adopted.
    family = {process.pid}
    groups = {process.pid} if hasattr(os, "killpg") else set()
    if Path("/proc/self/stat").exists():
        parents, process_groups = {}, {}
        for entry in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = entry.read_text().rsplit(")", 1)[1].split()
                identifier = int(entry.parent.name)
                parents[identifier], process_groups[identifier] = int(fields[1]), int(fields[2])
            except (OSError, ValueError, IndexError):
                continue
        while True:
            descendants = {identifier for identifier, parent in parents.items() if parent in family}
            if descendants <= family:
                break
            family |= descendants
        groups |= {process_groups[identifier] for identifier in family if process_groups.get(identifier) in family}

    def send(sig):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        for identifier in family:
            try:
                os.kill(identifier, sig)
            except ProcessLookupError:
                pass
    send(signal.SIGTERM)
    process.join(1)
    # The main worker may already have exited while a descendant ignored TERM.
    send(signal.SIGKILL)
    process.join()
    deadline = time.monotonic() + 5
    while Path("/proc/self/stat").exists():
        live = []
        for identifier in family:
            try:
                state = Path(f"/proc/{identifier}/stat").read_text().rsplit(")", 1)[1].split()[0]
                if state != "Z":
                    live.append(identifier)
            except FileNotFoundError:
                pass
        if not live:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Worker descendants could not be fenced; workspace remains untrusted")
        time.sleep(.02)


def freeze_inputs(request):
    """The only reads of external payloads; freeze and rehash before backend use."""
    import shutil
    frozen = []
    # A frozen collision-free publication name (CP-LEDGER F1) spells the source
    # payload, so every product the backend names after it is unique.
    published = request.settings.get("publication_name")
    for ordinal, artifact in enumerate(request.inputs):
        name = published if published and artifact.kind in ("source_payload", "initial_input", "download_payload") else artifact.path.name
        path = request.work_directory / "inputs" / str(ordinal) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        digest,size=copy_hashed(artifact.path,path)
        if size != artifact.bytes or digest != artifact.sha256:
            raise ValueError("input_hash_changed")
        frozen.append(replace(artifact, path=path))
    return replace(request, inputs=tuple(frozen))
