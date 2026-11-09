"""Private workspaces and a hard deadline spanning conversion AND export.

There is deliberately no pool, scheduler, resume policy or retention here.
"""
from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
from contextvars import ContextVar
from typing import cast
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

# Variables the extractors cap to one thread while they run (GlossExtract._apply_thread_caps, the batched adapter).
THREAD_CAP_VARIABLES=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS')


@contextmanager
def thread_caps_scope():
    """Restore the caller's thread variables after an in-process extraction (K4F2-G1).

    Inside the scope the extractors set their caps as before (worker processes started there inherit them). On exit
    every capped variable has its prior value again, and an unset one is unset, so the frozen execution environment a
    same-process resume compares is the caller's own. The caller's settings are never removed or rewritten."""
    prior={name:os.environ.get(name) for name in THREAD_CAP_VARIABLES}
    try:yield
    finally:
        for name,value in prior.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def preserving_thread_caps(function):
    """Decorator form of ``thread_caps_scope`` for an in-process extraction entry."""
    import functools
    @functools.wraps(function)
    def wrapper(*args,**kwargs):
        with thread_caps_scope():
            return function(*args,**kwargs)
    return wrapper


@contextmanager
def producer_hash_scope():
    token=_producer_hashes.set({})
    try:yield
    finally:_producer_hashes.reset(token)


def remember_written_hash(path: 'str | os.PathLike[str]', digest: str, size: int) -> tuple[str, int]:
    values=_producer_hashes.get()
    if values is not None:values[str(Path(path).absolute())]=(digest,size)
    # glossAPI wrote these bytes: reusable for the rest of this run (K4C-G08, ledger.encoding).
    from ..ledger.encoding import remember_own_hash
    remember_own_hash(path,digest,size)
    return digest,size


def produced_hash(path):
    values=_producer_hashes.get()
    return values.get(str(Path(path).absolute())) if values is not None else None


def write_text_hashed(path,text,encoding='utf-8'):
    data=text.encode(encoding)
    with Path(path).open('wb') as stream:stream.write(data)
    remember_written_hash(path,hashlib.sha256(data).hexdigest(),len(data))
    return len(text)


def sha256_file(path: 'str | os.PathLike[str]', *, own: bool = False) -> str:
    from ..ledger.encoding import file_hash
    return file_hash(path, own=own)


def seal_output_hashes(result: StageResult, root: 'str | os.PathLike[str]') -> StageResult:
    """Worker-owned byte receipts; no stat identity cache supplies content proof."""
    outputs=[]
    for output in result.outputs:
        path=checked_path(root,output.relative_path)
        written=produced_hash(path)
        digest,size=written if written else (output.sha256 or sha256_file(path),path.stat().st_size)
        remember_written_hash(path,digest,size)
        outputs.append(replace(output,sha256=digest,bytes=size))
    return replace(result,outputs=tuple(outputs))


def copy_hashed(source: 'str | os.PathLike[str]', destination: 'str | os.PathLike[str]', *,
                expected: 'tuple[str | None, int | None] | None' = None) -> tuple[str, int]:
    """Hash the bytes being copied, without reading the new product again.

    A destination that is a hard link (it may share its inode with a managed
    file, K4(c) decision 3) is unlinked first, so the copy never writes
    through a shared inode. Other destinations keep the in-place write.
    """
    import stat as _stat
    destination=Path(destination)
    try:info=os.lstat(destination)
    except FileNotFoundError:info=None
    if info is not None and _stat.S_ISREG(info.st_mode) and info.st_nlink>1:
        os.unlink(destination)
    digest=hashlib.sha256();size=0
    before=os.stat(source)
    with Path(source).open('rb') as reader, destination.open('wb') as writer:
        for block in iter(lambda:reader.read(1024*1024),b''):
            writer.write(block);digest.update(block);size+=len(block)
    result=remember_written_hash(destination,digest.hexdigest(),size)
    if expected is not None and result==expected:
        # The caller's recorded hash of a file glossAPI wrote was just verified by reading it, and the source did
        # not change while it was read: reusable for the rest of this run (K4C-G08).
        after=os.stat(source)
        if _stable(before,after):
            from ..ledger.encoding import remember_own_hash
            remember_own_hash(source,result[0],result[1],info=after)
    return result


def _stable(before: os.stat_result, after: os.stat_result) -> bool:
    return (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(
        after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)


def copy_reused(source: 'str | os.PathLike[str]', destination: 'str | os.PathLike[str]', sha256: str, size: int) -> tuple[str, int] | None:
    """Copy an own file whose recorded hash this run already holds, without hashing it again (K4C-G08).

    Returns None (the caller hashes while copying) unless the source is
    registered with exactly this hash and size and stays unchanged during the
    copy; the copy's bytes are glossAPI's own write of those bytes.
    """
    import shutil
    from ..ledger.encoding import reused_own_hash
    try:before=os.stat(source)
    except OSError:return None
    if before.st_size!=size or reused_own_hash(before)!=sha256:
        return None
    destination=Path(destination)
    import stat as _stat
    try:info=os.lstat(destination)
    except FileNotFoundError:info=None
    if info is not None and _stat.S_ISREG(info.st_mode) and info.st_nlink>1:
        os.unlink(destination)
    shutil.copyfile(source,destination)
    if not _stable(before,os.stat(source)) or os.stat(destination).st_size!=size:
        raise ValueError('input_hash_changed')
    return remember_written_hash(destination,sha256,size)


# Private outputs a run keeps (recovery frontiers, evidence); everything else private and intermediate is released.
RETAINED_PRIVATE_KINDS=frozenset(('download_payload','worker_receipt','generation_manifest','ocr_native','ocr_page_state'))


def links_released_copies(journal, retention_class, kind):
    """True when a compatibility product may share the inode of its private source (K4(c) decision 3).

    Only with debug=off, and only for a private intermediate that this run
    releases after publication: once released, the product is the file's only
    name, so no retained managed file stays aliased to a user-visible path.
    """
    settings=getattr(journal,'retention_settings',None) or {}
    return (journal is not None and settings.get('debug','off')=='off' and retention_class=='intermediate'
            and kind not in RETAINED_PRIVATE_KINDS)


def link_published(source: 'str | os.PathLike[str]', destination: 'str | os.PathLike[str]', sha256: str, size: 'int | None') -> tuple[str, int] | None:
    """Publish ``source`` at ``destination`` as a hard link (atomic rename of a fresh link).

    The registered producer hash is reused: no new byte stream exists, so the
    bytes are not hashed again (hash once per byte stream). Returns None when
    the filesystem cannot link (other device, link limit, unsupported); the
    caller then copies.
    """
    import uuid
    source=Path(source);destination=Path(destination)
    if destination.suffix=='.parquet' or destination.is_symlink():
        return None  # metadata-shaped or symlinked destinations keep their own publication rules
    info=source.lstat()
    import stat as _stat
    if not _stat.S_ISREG(info.st_mode) or info.st_size!=size:
        raise ValueError('Product changed during final publication')
    try:current=destination.lstat()
    except FileNotFoundError:current=None
    if current is not None and (current.st_dev,current.st_ino)==(info.st_dev,info.st_ino):
        # Already this inode (e.g. republished after a kill): rename() between two links of
        # one file is a no-op that would leave the temporary name behind.
        return remember_written_hash(destination,sha256,info.st_size)
    temporary=destination.parent/('.'+destination.name+'.'+uuid.uuid4().hex+'.link')
    try:os.link(source,temporary,follow_symlinks=False)
    except OSError:return None
    try:os.replace(temporary,destination)
    finally:
        # Defensive: a same-inode rename leaves the source name in place.
        if os.path.lexists(temporary):os.unlink(temporary)
    return remember_written_hash(destination,sha256,info.st_size)


def checked_path(root, relative):
    relative = Path(relative)
    root = Path(os.path.realpath(root))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Artifact path must be a contained relative path")
    candidate = root / relative
    # Below a resolved root, a contained relative path resolves to itself unless one of its components is a
    # link, so the full resolution is needed only to name the error (same checks, same messages; hot path).
    linked = False
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if os.path.islink(parent):
            linked = True
            break
    if linked:
        if not candidate.resolve().is_relative_to(root):
            raise ValueError("Artifact symlink escapes attempt directory")
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


def disk_full(error):
    """True when an exception (or its cause chain) is an exhausted filesystem: ENOSPC/EDQUOT or SQLite's disk-full."""
    import errno
    import sqlite3
    seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error,OSError) and error.errno in (errno.ENOSPC,errno.EDQUOT):return True
        if isinstance(error,sqlite3.Error) and (getattr(error,'sqlite_errorcode',None)==13 or 'disk is full' in str(error)):return True
        error=error.__cause__ or error.__context__
    return False


def failed_result(request, reason, error, *, decisions=(), metrics=None):
    # An exhausted filesystem under a worker's output write is the document's typed cause (W18 M10), not a budget
    # refusal; also when the ENOSPC is the cause of the raised error (K4g fix 3).
    if disk_full(error):reason='disk_full'
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
        encoded = json.dumps(asdict(result), ensure_ascii=False)
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


def _proc_table(attempts=3):
    """(parents, process groups, complete) of the Linux processes in ``/proc``. A process that exits while the
    directory is being enumerated can make advancing the glob itself raise (ESRCH/ENOENT from a vanished entry; fix 12,
    R5): the enumeration is retried. ``complete`` is True only for an enumeration that ran to its end (fix 13, H4);
    after ``attempts`` cut-short enumerations the entries read so far are returned with ``complete=False``. Per-entry
    reads that fail are skipped (the process exited), as before."""
    parents, process_groups = {}, {}
    for _ in range(attempts):
        entries = iter(Path("/proc").glob("[0-9]*/stat"))
        while True:
            try:
                entry = next(entries)
            except StopIteration:
                return parents, process_groups, True
            except OSError:
                break  # the enumeration was cut short by a vanished entry: start it again
            try:
                fields = entry.read_text().rsplit(")", 1)[1].split()
                identifier = int(entry.parent.name)
                parents[identifier], process_groups[identifier] = int(fields[1]), int(fields[2])
            except (OSError, ValueError, IndexError):
                continue
    return parents, process_groups, False


def _stop_process(process):
    # A browser/exporter can detach into another process group, or ignore TERM.
    # Capture Linux descendants before their parent exits and they are adopted.
    # Fix 12 (R5): enumerating /proc can fail while another process exits; the known family (at least the worker
    # itself) is still signalled, joined and fenced.
    family = {process.pid}
    groups = {process.pid} if hasattr(os, "killpg") else set()
    complete = True
    if Path("/proc/self/stat").exists():
        parents, process_groups, complete = _proc_table()
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
            except (FileNotFoundError, ProcessLookupError):
                pass  # gone (ESRCH: the process exited while its stat was read)
        if not live:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Worker descendants could not be fenced; workspace remains untrusted")
        time.sleep(.02)
    if not complete:
        # Fix 13 (H4): the known family was signalled, joined and fenced, but an incomplete /proc enumeration cannot
        # prove that no descendant (one in its own process group) was missed: never certified as fenced.
        raise RuntimeError("Worker descendants could not be fenced; workspace remains untrusted "
                           "(process enumeration was incomplete)")


def freeze_inputs(request):
    """The only reads of external payloads; freeze and rehash before backend use."""
    import shutil
    frozen = []
    # A frozen collision-free publication name (CP-LEDGER F1) spells the source
    # payload, so every product the backend names after it is unique.
    published = cast("str | None", request.settings.get("publication_name"))
    from .contracts import ArtifactReference
    for ordinal, artifact in enumerate(cast('tuple[ArtifactReference, ...]', request.inputs)):
        name = published if published and artifact.kind in ("source_payload", "initial_input", "download_payload") else artifact.path.name
        path = cast(Path, request.work_directory / "inputs" / str(ordinal) / name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A producer-bound input glossAPI wrote (or verified) in this run keeps its recorded hash (K4C-G08). Nothing
        # is registered here: a caller-supplied input (an original, a download, a given text) is hashed as it enters
        # the attempt, every time.
        copied=None if artifact.kind in ("source_payload", "initial_input", "download_payload") else copy_reused(
            artifact.path,path,artifact.sha256,artifact.bytes)
        digest,size=copied or copy_hashed(artifact.path,path)
        if size != artifact.bytes or digest != artifact.sha256:
            raise ValueError("input_hash_changed")
        frozen.append(replace(artifact, path=path))
    return replace(request, inputs=tuple(frozen))
