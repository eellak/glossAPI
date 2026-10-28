"""Experimental cross-document layout batching (Docling 2.81 internal adapter).

No Docling import occurs until DoclingAdapter is constructed. Native PDF handles
stay in spawned CPU workers; only raster images, parsed cells and timings cross
IPC. One owner thread calls both resident inference models. See T3_1_NOTES.md.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, field
import functools
import inspect
import io
import multiprocessing as mp
import pickle
import socket
import struct
import types
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from enum import Enum
from pathlib import Path, PurePath, PosixPath, PurePosixPath, WindowsPath, PureWindowsPath
import queue
import threading
import time
from typing import Any


@dataclass
class PageWork:
    document_id: int
    path: Path
    page_no: int
    payload: Any
    error: str | None = None


@dataclass
class _Document:
    path: Path
    started: float
    deadline: float
    result: Any = None
    count: int = 0
    next_page: int = 1
    pages: dict[int, Any] = field(default_factory=dict)
    error: str | None = None
    exporting: bool = False


def _parse_worker(connection, prepare, options):
    """Persistent worker; a native crash/hang only loses its current document."""
    try:
        while True:
            job = connection.recv()
            if job is None:
                return
            path, page_no = job
            try:
                payload = prepare(Path(path), page_no, **options)
                connection.send((True, payload))
            except Exception as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class _Slot:
    def __init__(self, context, prepare, options):
        self.context, self.prepare, self.options = context, prepare, options
        self.job = None
        self.start()

    def start(self):
        self.connection, child = self.context.Pipe()
        self.process = self.context.Process(
            target=_parse_worker, args=(child, self.prepare, self.options), daemon=True)
        self.process.start()
        child.close()

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        self.connection.close()

    def restart(self):
        self.close()
        self.job = None
        self.start()


def _new_export_object(cls):
    # Reconstruction runs only in the killable worker, without constructors.
    if cls is types.SimpleNamespace:
        return types.SimpleNamespace()
    if issubclass(cls, dict):
        return dict.__new__(cls)
    if issubclass(cls, list):
        return list.__new__(cls)
    return object.__new__(cls)


def _set_export_state(obj, state):
    attributes, slots = state
    if attributes is not None:
        object.__getattribute__(obj, "__dict__").update(attributes)
    for name, value in slots.items():
        object.__setattr__(obj, name, value)


class _ExportTransportPickler(pickle.Pickler):
    """Transport state without executing payload reduction/state hooks in owner.

    A regular multiprocessing.Queue pickles in an uncancellable parent feeder.
    Transfer Python instance dictionaries/slots structurally instead; normal
    pickle hooks (and their semantics) are evaluated inside the worker. Native
    values are limited to explicit standard-library types with bounded C/state
    reducers. Unsupported opaque extension objects fail rather than falling
    back to arbitrary parent-side code.
    """
    def reducer_override(self, obj):
        cls = type(obj)
        if issubclass(cls, type) or cls in (
                types.FunctionType, types.BuiltinFunctionType,
                bytearray, complex, range, slice,
                date, datetime, datetime_time, timedelta, timezone,
                Path, PurePath, PosixPath, PurePosixPath, WindowsPath, PureWindowsPath):
            return NotImplemented
        if cls is functools.partial:
            # Do not look up instance-overridable __reduce__ methods.
            return functools.partial, (obj.func,), (obj.func, obj.args, obj.keywords, obj.__dict__)
        if issubclass(cls, Enum):
            return cls, (object.__getattribute__(obj, "_value_"),)
        if cls is types.MethodType:
            return getattr, (obj.__self__, obj.__func__.__name__)
        attributes = None
        slots = {}
        for base in cls.__mro__:
            for name, descriptor in vars(base).items():
                if name == "__dict__" and isinstance(descriptor, (types.GetSetDescriptorType, types.MemberDescriptorType)):
                    attributes = descriptor.__get__(obj, cls)
                if name not in ("__dict__", "__weakref__") and isinstance(descriptor, types.MemberDescriptorType):
                    try:
                        slots[name] = descriptor.__get__(obj, cls)
                    except AttributeError:
                        pass
        list_items = list.__iter__(obj) if issubclass(cls, list) else None
        dict_items = iter(dict.items(obj)) if issubclass(cls, dict) else None
        if attributes is None and not slots and list_items is None and dict_items is None:
            raise pickle.PicklingError(f"unsupported opaque export value: {cls.__module__}.{cls.__name__}")
        return (_new_export_object, (cls,), (attributes, slots), list_items, dict_items, _set_export_state)


class _DeadlineBuffer(io.BytesIO):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline

    def write(self, value):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("timeout")
        return super().write(value)


class _ExportQueue:
    """Socket-backed bounded queue: capacity one, no parent feeder threads.

    The single writer puts one payload then consumes its receipt before another
    put. Socket backpressure makes put itself deadline-bounded, including a
    worker stalled before reading. Only bytes cross this transport.
    """
    def __init__(self, channel):
        self.channel = channel

    def _timeout(self, deadline):
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("timeout")
        self.channel.settimeout(remaining)

    def put(self, payload, timeout):
        deadline = time.monotonic() + timeout
        for data in (struct.pack("!Q", len(payload)), payload):
            view = memoryview(data)
            while view:
                self._timeout(deadline)
                sent = self.channel.send(view)
                if not sent:
                    raise EOFError("export queue closed")
                view = view[sent:]

    def get(self, deadline=None):
        def read(size):
            data = bytearray()
            while len(data) < size:
                self._timeout(deadline)
                chunk = self.channel.recv(min(size - len(data), 1024 * 1024))
                if not chunk:
                    raise EOFError("export queue closed")
                data.extend(chunk)
            return data
        size, = struct.unpack("!Q", read(8))
        return read(size)

    def close(self):
        self.channel.close()


def _export_worker(channel, epoch):
    """One persistent export process; models remain resident in the owner."""
    jobs = _ExportQueue(channel)
    try:
        jobs.put(pickle.dumps((epoch, "ready")), 10)
        while True:
            data = jobs.get()
            generation, sequence, deadline, task = pickle.loads(data)
            del data
            if generation != epoch or time.monotonic() >= deadline:
                continue
            try:
                # Arbitrary __reduce__/__getstate__/reconstruction hooks belong
                # here, where termination frees both their stack and payload.
                task = pickle.loads(pickle.dumps(task, protocol=5))
                if time.monotonic() >= deadline:
                    raise TimeoutError("timeout")
                task()
                receipt = (epoch, sequence, time.monotonic(), None)
            except BaseException as exc:
                receipt = (epoch, sequence, time.monotonic(),
                           (type(exc).__name__, str(exc)))
            del task
            jobs.put(pickle.dumps(receipt), max(0, deadline - time.monotonic()))
    except (EOFError, OSError, TimeoutError):
        pass
    except BaseException as exc:
        jobs.put(pickle.dumps((epoch, "transport-error", time.monotonic(),
                              (type(exc).__name__, str(exc)))), 1)
    finally:
        jobs.close()


class _ExportSession:
    def __init__(self, context):
        self.context = context
        self.epoch = 0
        self.sequence = 0
        self.process = None
        self.jobs = None

    def close(self):
        if self.process is not None:
            if self.process.pid is not None and self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=0.2)
            if self.process.is_alive():
                # Never start a replacement while an old publisher survives.
                raise RuntimeError("export worker could not be reaped")
            if self.process.pid is not None:
                self.process.join(timeout=0)
            self.process.close()
            self.process = None
        if self.jobs is not None:
            self.jobs.close()
            self.jobs = None

    def _start(self, deadline):
        self.epoch += 1
        parent, child = socket.socketpair()
        self.jobs = _ExportQueue(parent)
        try:
            self.process = self.context.Process(target=_export_worker,
                args=(child, self.epoch), name="batched-export-worker", daemon=True)
            # Only a socket and integer cross spawn: no document hooks or models.
            self.process.start()
        finally:
            child.close()
        if pickle.loads(self.jobs.get(deadline)) != (self.epoch, "ready"):
            raise RuntimeError("invalid export startup receipt")

    def export(self, task, deadline):
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("timeout")
            if self.process is None:
                self._start(deadline)
            self.sequence += 1
            with _DeadlineBuffer(deadline) as buffer:
                _ExportTransportPickler(buffer, protocol=5).dump(
                    (self.epoch, self.sequence, deadline, task))
                self.jobs.put(buffer.getvalue(), max(0, deadline - time.monotonic()))
            generation, sequence, completed, error = pickle.loads(self.jobs.get(deadline))
            if generation == self.epoch and sequence == "transport-error":
                raise RuntimeError(f"export transport: {error[0]}: {error[1]}")
            if (generation, sequence) != (self.epoch, self.sequence):
                raise RuntimeError("stale export receipt")
            if completed > deadline:
                raise TimeoutError("timeout")
            if error:
                name, message = error
                # Preserve the useful public serialization error contract.
                error_type = {"ValueError": ValueError, "TimeoutError": TimeoutError}.get(name, RuntimeError)
                raise error_type(f"{name}: {message}")
        except BaseException:
            # Reap before cleanup/failure publication; replacement is lazy and
            # gets a fresh socket and epoch on the next document's own budget.
            self.close()
            raise


def _bounded_export(context, task, deadline):
    """Single-task compatibility entry; the scheduler owns a persistent session."""
    session = _ExportSession(context)
    try:
        session.export(task, deadline)
    finally:
        session.close()


class BatchedExtractor:
    """Bounded scheduler with a fakeable adapter, deterministic document reassembly.

    Adapter contract: prepare(path, page_no=0) probes a document; other page
    numbers return picklable CPU payloads. open/accept/layout/complete/finish/
    failure run in the owner or writer, never in a parsing subprocess. Export
    runs in one persistent, terminable spawned process per session. Adapters can
    provide export_task, a zero-argument callable containing only export state;
    otherwise their bound export method and result must be transferable. Tasks
    support Python instance dictionaries/slots and standard-library values;
    opaque extension objects must be converted to transferable state. Payload
    pickle hooks run only inside the worker, never in a parent feeder thread.
    Core defaults are workers=2 and layout_batch=8; Corpus auto-CUDA supplies
    workers=8 with layout_batch=8. Batch 16 is never selected automatically.
    Export futures and active documents are bounded as well as prepared pages.
    Timeouts begin at admission, exclude waiting input documents, and are checked
    before/after inference. A wedged native inference call cannot be preempted.
    """
    def __init__(self, adapter=None, *, workers=2, layout_batch=8,
                 queue_max_size=16, export_queue_size=2, document_timeout=600,
                 batch_wait_seconds=0.05, max_active_documents=None):
        for name, value in (("workers", workers), ("layout_batch", layout_batch),
                            ("queue_max_size", queue_max_size),
                            ("export_queue_size", export_queue_size)):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if document_timeout <= 0 or batch_wait_seconds < 0:
            raise ValueError("invalid timeout")
        if max_active_documents is not None and max_active_documents < 1:
            raise ValueError("max_active_documents must be positive")
        self.adapter = adapter if adapter is not None else DoclingAdapter()
        self.converter = getattr(self.adapter, "converter", None)
        self.pipeline = getattr(self.adapter, "pipeline", None)
        self.workers, self.layout_batch = workers, layout_batch
        self.queue_max_size, self.export_queue_size = queue_max_size, export_queue_size
        self.document_timeout, self.batch_wait_seconds = document_timeout, batch_wait_seconds
        self.max_active_documents = max_active_documents or max(2, workers * 2)
        self.batch_log: list[list[tuple[int, int]]] = []
        self.high_water = {"prepared_pages": 0, "parse_jobs": 0,
                           "exports": 0, "active_documents": 0}
        self.processed_files: list[str] = []
        self.problematic_files: list[str] = []
        self.errors: dict[str, str] = {}
        self.page_counts: dict[str, int] = {}

    def extract(self, paths, output_dir, *, max_pages=None):
        paths = [Path(p).resolve() for p in paths]
        # The normal artifact tree is keyed by stem. Refuse ambiguous overwrites.
        if len({p.stem for p in paths}) != len(paths):
            raise ValueError("duplicate filename stems cannot share an artifact directory")
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be positive")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_log.clear()
        self.processed_files.clear()
        self.problematic_files.clear()
        self.errors.clear()
        self.page_counts.clear()
        for key in self.high_water:
            self.high_water[key] = 0
        pending = deque(enumerate(paths))
        active: dict[int, _Document] = {}
        ready = queue.Queue(maxsize=self.queue_max_size)
        rotation = deque()
        exports = {}
        context = mp.get_context("spawn")
        export_session = _ExportSession(context)
        slots = []
        prepare_options = dict(getattr(self.adapter, "prepare_options", {}), max_pages=max_pages)
        first_ready = None

        def fail(doc, message):
            if doc.error is None:
                doc.error = message
                doc.pages.clear()

        def record_failure(doc):
            try:
                # The export child has been reaped before removing partial files:
                # it cannot later replace the timeout receipt with success.
                stem = doc.path.stem
                for path in (output_dir / f"{stem}.md",
                             output_dir.parent / "json" / f"{stem}.docling.json",
                             output_dir.parent / "json" / f"{stem}.docling.json.zst",
                             output_dir.parent / "json/metrics" / f"{stem}.metrics.json",
                             output_dir.parent / "json/metrics" / f"{stem}.per_page.metrics.json"):
                    path.unlink(missing_ok=True)
                self.adapter.failure(doc.path, output_dir, doc.error)
            except Exception as exc:
                # A document-specific artifact path can itself be unwritable.
                # Preserve the error in the returned problem list and continue.
                doc.error = f"{doc.error}; failure recording: {type(exc).__name__}: {exc}"

        def write(doc):
            if time.monotonic() > doc.deadline:
                fail(doc, "timeout")
            if doc.error:
                record_failure(doc)
                return False
            result = self.adapter.finish(doc.result, [doc.pages[i] for i in range(1, doc.count + 1)])
            task = (self.adapter.export_task(result, output_dir)
                    if hasattr(self.adapter, "export_task") else
                    functools.partial(self.adapter.export, result, output_dir))
            try:
                export_session.export(task, doc.deadline)
            except TimeoutError:
                fail(doc, "timeout")
                record_failure(doc)
                return False
            return True

        def infer(batch):
            # Retry a mixed inference error by document, so a bad input does not
            # fail unrelated documents. Never retry an already failed document.
            batch = [p for p in batch if not active[p.document_id].error]
            if not batch:
                return
            self.batch_log.append([(p.document_id, p.page_no) for p in batch])
            try:
                self.adapter.layout(batch, {i: d.result for i, d in active.items()})
            except Exception as exc:
                groups = {}
                for page in batch:
                    groups.setdefault(page.document_id, []).append(page)
                if len(groups) > 1:
                    for group in groups.values():
                        infer(group)
                else:
                    fail(active[batch[0].document_id], f"layout: {type(exc).__name__}: {exc}")
                return
            for page in batch:
                doc = active[page.document_id]
                if page.error:
                    fail(doc, page.error)
                if time.monotonic() > doc.deadline:
                    fail(doc, "timeout")
                if doc.error:
                    continue
                try:
                    self.adapter.complete(doc.result, page)
                    if time.monotonic() > doc.deadline:
                        fail(doc, "timeout")
                    else:
                        doc.pages[page.page_no] = page.payload
                except Exception as exc:
                    fail(doc, f"page {page.page_no}: {type(exc).__name__}: {exc}")

        try:
            for _ in range(self.workers):
                slots.append(_Slot(context, self.adapter.prepare, prepare_options))
            # A single writer owns document assembly/reading-order state; it
            # overlaps inference without concurrent use of the inference models.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="batched-export") as writers:
                while pending or active or exports:
                    now = time.monotonic()
                    for did, future in list(exports.items()):
                        if future.done():
                            doc = active[did]
                            try:
                                ok = future.result()
                            except Exception as exc:
                                fail(doc, f"export: {type(exc).__name__}: {exc}")
                                record_failure(doc)
                                ok = False
                            if ok:
                                self.processed_files.append(doc.path.name)
                                self.page_counts[doc.path.name] = doc.count
                            else:
                                self.problematic_files.append(doc.path.name)
                                self.errors[doc.path.name] = doc.error or "export failure"
                            del exports[did]
                            del active[did]
                    for doc in active.values():
                        if not doc.exporting and now > doc.deadline:
                            fail(doc, "timeout")
                    # A failed/timed-out parser gets terminated and replaced;
                    # we never wait indefinitely on its result or pool shutdown.
                    for slot in slots:
                        if slot.job is not None:
                            did, page_no = slot.job
                            doc = active[did]
                            if doc.error:
                                slot.restart()
                            elif slot.connection.poll():
                                try:
                                    ok, payload = slot.connection.recv()
                                except (EOFError, OSError) as exc:
                                    ok, payload = False, f"parser process exited: {exc}"
                                    slot.restart()
                                slot.job = None
                                if not ok:
                                    fail(doc, str(payload))
                                    slot.restart()
                                else:
                                    try:
                                        if page_no == 0:
                                            doc.result, doc.count = self.adapter.open(doc.path, payload)
                                            if doc.count < 1:
                                                raise ValueError("document has no pages")
                                        else:
                                            page = PageWork(did, doc.path, page_no, payload)
                                            self.adapter.accept(doc.result, page)
                                            ready.put_nowait(page)
                                            if first_ready is None:
                                                first_ready = time.monotonic()
                                    except Exception as exc:
                                        fail(doc, f"parse: {type(exc).__name__}: {exc}")
                                        slot.restart()
                            elif not slot.process.is_alive():
                                fail(doc, f"parser process exited {slot.process.exitcode}")
                                slot.restart()
                    self.high_water["prepared_pages"] = max(self.high_water["prepared_pages"], ready.qsize())
                    # Full export queue stops the inference consumer. Since the
                    # CPU reservation includes in-flight jobs, this propagates
                    # backpressure all the way to parsing and document admission.
                    if len(exports) < self.export_queue_size:
                        waiting_jobs = any(s.job for s in slots)
                        if not ready.empty() and (ready.qsize() >= self.layout_batch or
                                time.monotonic() - first_ready >= self.batch_wait_seconds or
                                not waiting_jobs and not any(d.next_page <= d.count and not d.error
                                                            for d in active.values())):
                            batch = [ready.get_nowait() for _ in range(min(self.layout_batch, ready.qsize()))]
                            infer(batch)
                            first_ready = time.monotonic() if not ready.empty() else None
                        for did, doc in list(active.items()):
                            has_work = any(s.job and s.job[0] == did for s in slots)
                            queued = any(p.document_id == did for p in list(ready.queue))
                            if (not doc.exporting and not has_work and not queued and
                                    (doc.error or doc.count and len(doc.pages) == doc.count)):
                                if len(exports) >= self.export_queue_size:
                                    break
                                doc.exporting = True
                                exports[did] = writers.submit(write, doc)
                    while pending and len(active) < self.max_active_documents and len(exports) < self.export_queue_size:
                        did, path = pending.popleft()
                        started = time.monotonic()
                        try:
                            if hasattr(self.adapter, "admit"):
                                path = self.adapter.admit(path, output_dir)
                            budget = self.adapter.timeout(path) if hasattr(self.adapter, "timeout") else self.document_timeout
                            # The repair remainder is evaluated after repair;
                            # don't add its elapsed time back to the deadline.
                            active[did] = _Document(path, started, time.monotonic() + budget)
                        except Exception as exc:
                            active[did] = _Document(path, started, started,
                                error=f"admission: {type(exc).__name__}: {exc}")
                        rotation.append(did)
                    # Reserve a bounded queue place for every submitted page or
                    # probe. No executor map/eager future list buffers the corpus.
                    for slot in slots:
                        if slot.job is not None or len(exports) >= self.export_queue_size:
                            continue
                        if not slot.process.is_alive():
                            slot.restart()
                        reserved = ready.qsize() + sum(s.job is not None for s in slots)
                        if reserved >= self.queue_max_size:
                            break
                        for _ in range(len(rotation)):
                            did = rotation.popleft()
                            if did not in active:
                                continue
                            rotation.append(did)
                            doc = active[did]
                            if doc.error or doc.exporting:
                                continue
                            if doc.result is None:
                                if any(s.job and s.job[0] == did for s in slots):
                                    continue
                                page_no = 0
                            elif doc.next_page <= doc.count:
                                page_no = doc.next_page
                                doc.next_page += 1
                            else:
                                continue
                            slot.job = (did, page_no)
                            try:
                                slot.connection.send((str(doc.path), page_no))
                            except (BrokenPipeError, OSError) as exc:
                                fail(doc, f"parser dispatch failed: {exc}")
                                slot.restart()
                            break
                    self.high_water["parse_jobs"] = max(self.high_water["parse_jobs"], sum(s.job is not None for s in slots))
                    self.high_water["exports"] = max(self.high_water["exports"], len(exports))
                    self.high_water["active_documents"] = max(self.high_water["active_documents"], len(active))
                    time.sleep(0.002)
        finally:
            export_session.close()
            for slot in slots:
                slot.close()
        # Stable user-facing lists regardless of worker/write completion order.
        order = {p.name: i for i, p in enumerate(paths)}
        self.processed_files.sort(key=order.get)
        self.problematic_files.sort(key=order.get)
        return self.processed_files, self.problematic_files


_CPU_CACHE = None


def _discard_cpu_backend():
    """Invalidate first: even unload/open exceptions must leave no dead cache."""
    global _CPU_CACHE
    previous, _CPU_CACHE = _CPU_CACHE, None
    if previous is not None:
        previous[1]._backend.unload()


def prepare_docling_page(path, page_no, *, images_scale=1.25, max_pages=None):
    try:
        return _prepare_docling_page(path, page_no, images_scale=images_scale, max_pages=max_pages)
    except Exception:
        # Page loading and preprocessing failures can invalidate native state as
        # well as document-open failures. Never reuse it after either failure.
        try:
            _discard_cpu_backend()
        except Exception:
            pass  # preserve the original parse exception
        raise


def _prepare_docling_page(path, page_no, *, images_scale=1.25, max_pages=None):
    """CPU-only worker entry: one cached native backend per worker, no models."""
    global _CPU_CACHE
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
    from docling.datamodel.base_models import InputFormat, Page
    from docling.datamodel.document import InputDocument, ConversionResult
    from docling.datamodel.settings import settings
    from docling.models.stages.page_preprocessing.page_preprocessing_model import (
        PagePreprocessingModel, PagePreprocessingOptions)
    settings.debug.profile_pipeline_timings = True
    if _CPU_CACHE is None or _CPU_CACHE[0] != path:
        _discard_cpu_backend()
        source = InputDocument(path_or_stream=path, format=InputFormat.PDF,
                               backend=DoclingParseDocumentBackend)
        if not source.valid:
            if source._backend is not None:
                source._backend.unload()
            raise ValueError(f"invalid PDF: {path}")
        _CPU_CACHE = (path, source)
    source = _CPU_CACHE[1]
    if page_no == 0:
        return {"page_count": source.page_count,
                "selected_pages": min(source.page_count, max_pages or source.page_count),
                "document_hash": source.document_hash, "filesize": source.filesize}
    backend = source._backend.load_page(page_no - 1)
    try:
        if not backend.is_valid():
            raise ValueError(f"invalid page {page_no}")
        result = ConversionResult(input=source)
        page = Page(page_no=page_no)
        page._backend = backend
        page.size = backend.get_size()
        model = PagePreprocessingModel(PagePreprocessingOptions(images_scale=images_scale))
        page = next(iter(model(result, [page])))
        # TableFormer requires scale 2, which cannot be rasterised by a native
        # handle in the inference owner. Save untouched word cells as well:
        # layout postprocessing mutates parsed_page.
        segmented = deepcopy(backend.get_segmented_page())
        page.get_image(scale=2.0)
        page._backend = None
        return {"page": page, "segmented": segmented,
                "confidence": result.confidence.pages[page_no], "timings": result.timings}
    finally:
        backend.unload()


class _RasterBackend:
    """Minimal duck-typed backend for the inspected layout/table/assembly path."""
    def __init__(self, page, segmented):
        self.page, self.segmented = page, segmented

    def is_valid(self):
        return True

    def get_segmented_page(self):
        return self.segmented

    def get_page_image(self, scale=1, cropbox=None):
        image = self.page._image_cache[scale]
        if cropbox is not None:
            return image.crop(cropbox.scaled(scale).as_tuple())
        return image

    def unload(self):
        pass


class _PredictionReplay:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict_batch(self, images):
        if len(images) != len(self.predictions):
            raise RuntimeError("layout replay size mismatch")
        return self.predictions


class _PerPageLayoutTimings:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, result, pages):
        pages = list(pages)
        previous = result.timings.get("layout")
        offset = len(previous.times) if previous is not None else 0
        output = list(self.model(result, pages))
        item = result.timings.get("layout")
        if item is not None and len(item.times) > offset:
            duration = sum(item.times[offset:])
            timestamp = item.start_timestamps[offset]
            item.times[offset:] = [duration / len(output)] * len(output)
            item.start_timestamps[offset:] = [timestamp] * len(output)
            item.count = len(item.times)
        yield from output


def _docling_export(result, output_dir, options):
    """Rebuild only the export surface, without converter or resident models."""
    import logging
    from glossapi.gloss_extract import GlossExtract
    adapter = DoclingAdapter.__new__(DoclingAdapter)
    adapter.exporter = GlossExtract.__new__(GlossExtract)
    adapter.exporter.__dict__.update(options)
    adapter.exporter._log = logging.getLogger(__name__)
    adapter.export(result, output_dir)


class DoclingAdapter:
    """Version-gated internal hook; does not monkeypatch installed Docling."""
    prepare = staticmethod(prepare_docling_page)

    @staticmethod
    def validate(pipeline):
        from importlib.metadata import version
        if version("docling") != "2.81.0":
            raise RuntimeError("internal batching adapter requires Docling 2.81.0")
        if not hasattr(pipeline.layout_model, "layout_predictor"):
            raise RuntimeError("unsupported layout model")
        if not getattr(pipeline.table_model, "do_cell_matching", False):
            raise RuntimeError("worker raster backend requires table cell matching")
        opts = pipeline.pipeline_options
        if opts.do_ocr or pipeline.keep_backend:
            raise RuntimeError("batched adapter supports layout-only PDFs without backend enrichment")

    def __init__(self, *, device="cpu", images_scale=1.25, layout_batch=8,
                 document_timeout=600, artifacts_path=None, exporter=None):
        from importlib.metadata import version
        if version("docling") != "2.81.0":
            raise RuntimeError("internal batching adapter requires Docling 2.81.0")
        import os
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(key, "1")
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.settings import settings
        from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
        from glossapi.ocr.docling.pipeline import _apply_common_pdf_options, _resolve_accelerator
        options = _apply_common_pdf_options(acc=_resolve_accelerator(device)[0],
            images_scale=images_scale, formula_enrichment=False, code_enrichment=False)
        options.layout_batch_size = layout_batch
        options.document_timeout = document_timeout
        if artifacts_path:
            options.artifacts_path = Path(artifacts_path)
        settings.debug.profile_pipeline_timings = True
        self.converter = DocumentConverter(allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(
                pipeline_options=options, backend=DoclingParseDocumentBackend)})
        self.converter.initialize_pipeline(InputFormat.PDF)
        self.pipeline = next(iter(self.converter.initialized_pipelines.values()))
        self.validate(self.pipeline)
        self.prepare_options = {"images_scale": images_scale}
        if exporter is None:
            # GlossExtract's constructor opens conversion.log in cwd. Construct
            # only its export surface here to avoid an unrequested repo artifact.
            import logging
            from glossapi.gloss_extract import GlossExtract
            exporter = GlossExtract.__new__(GlossExtract)
            exporter._log = logging.getLogger(__name__)
            exporter.export_doc_json = True
            exporter.emit_formula_index = False
            exporter.benchmark_mode = False
            exporter.pdf_backend_name = "docling_parse"
            exporter._font_repair_records = {}
            exporter._font_repair_originals = {}
        self.exporter = exporter

    def open(self, path, metadata):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import InputDocument, ConversionResult
        from docling.datamodel.settings import DocumentLimits
        source = InputDocument.model_construct(file=path, format=InputFormat.PDF,
            document_hash=metadata["document_hash"], filesize=metadata["filesize"],
            page_count=metadata["page_count"], valid=True,
            limits=DocumentLimits(page_range=(1, metadata["selected_pages"])))
        result = ConversionResult(input=source)
        result._t31_started = time.monotonic()
        return result, metadata["selected_pages"]

    def accept(self, result, work):
        packet = work.payload
        page = packet["page"]
        page._backend = _RasterBackend(page, packet["segmented"])
        page._t31_timings = packet["timings"]
        result.confidence.pages[work.page_no] = packet["confidence"]
        work.payload = page

    def layout(self, batch, results):
        from docling.utils.profiling import ProfilingItem, ProfilingScope
        model = self.pipeline.layout_model
        if isinstance(model, _PerPageLayoutTimings):
            model = model.model
        started = time.monotonic()
        predictions = list(model.layout_predictor.predict_batch(
            [p.payload.get_image(scale=1.0) for p in batch]))
        elapsed = time.monotonic() - started
        if not hasattr(self, "forward_log"):
            self.forward_log = []
        self.forward_log.append({"pages": [(p.document_id, p.page_no) for p in batch],
                                 "wall_seconds": elapsed})
        if len(predictions) != len(batch):
            raise RuntimeError("layout predictor returned wrong page count")
        for work, prediction in zip(batch, predictions):
            result = results[work.document_id]
            # A shallow model wrapper retains the very same resident predictor
            # weights, and replays only this page's predictions for postprocessing.
            replay = copy(model)
            replay.layout_predictor = _PredictionReplay([prediction])
            saved_timings = result.timings
            result.timings = {}
            try:
                list(replay(result, [work.payload]))
            except Exception as exc:
                work.error = f"layout postprocess: {type(exc).__name__}: {exc}"
                continue
            finally:
                work.payload._t31_timings.update(result.timings)
                result.timings = saved_timings
            # Replace the replay's near-zero timer with an equal share of the
            # measured mixed-batch forward. Keep the existing metrics key/schema.
            item = work.payload._t31_timings.setdefault("layout", ProfilingItem(scope=ProfilingScope.PAGE))
            item.times = [elapsed / len(batch)]
            item.count = 1

    def per_page_layout_timings(self):
        """Give sequential batches the same per-page forward attribution.

        Installed Docling records one forward duration per layout batch. Split
        it equally over that batch's pages in page-number order in both modes.
        This affects benchmark timing receipts, never layout predictions.
        """
        if not isinstance(self.pipeline.layout_model, _PerPageLayoutTimings):
            self.pipeline.layout_model = _PerPageLayoutTimings(self.pipeline.layout_model)

    def complete(self, result, work):
        saved_timings = result.timings
        result.timings = {}
        try:
            list(self.pipeline.table_model(result, [work.payload]))
            list(self.pipeline.assemble_model(result, [work.payload]))
        finally:
            work.payload._t31_timings.update(result.timings)
            result.timings = saved_timings
        self.pipeline._release_page_resources(type("Item", (), {"payload": work.payload})())

    def finish(self, result, pages):
        from docling.datamodel.base_models import ConversionStatus
        result.pages = pages
        result.status = ConversionStatus.SUCCESS
        # Timings must follow page order too: downstream per-page metrics maps
        # timing arrays by index, not by the asynchronous completion order.
        for page in pages:
            for key, value in page._t31_timings.items():
                if key not in result.timings:
                    result.timings[key] = deepcopy(value)
                else:
                    result.timings[key].times.extend(value.times)
                    result.timings[key].count += value.count
                    result.timings[key].start_timestamps.extend(value.start_timestamps)
            del page._t31_timings
        result = self.pipeline._assemble_document(result)
        result = self.pipeline._enrich_document(result)
        if hasattr(result, "_t31_started"):
            from docling.utils.profiling import ProfilingItem, ProfilingScope
            from datetime import datetime, timezone
            elapsed = time.monotonic() - result._t31_started
            result.timings["pipeline_total"] = ProfilingItem(scope=ProfilingScope.DOCUMENT,
                count=1, times=[elapsed], start_timestamps=[datetime.now(timezone.utc)])
            del result._t31_started
        return result

    def export(self, result, output_dir):
        counts = self.exporter._export_documents([result], output_dir)
        stem = result.input.file.stem
        required = [output_dir / f"{stem}.md"]
        json_dir = output_dir.parent / "json"
        if getattr(self.exporter, "export_doc_json", False) and not any(
                (json_dir / f"{stem}.docling.json{suffix}").exists() for suffix in ("", ".zst")):
            raise RuntimeError("Docling JSON export missing")
        if not getattr(self.exporter, "benchmark_mode", False):
            required.extend(json_dir / "metrics" / f"{stem}.{suffix}.json"
                            for suffix in ("metrics", "per_page.metrics"))
        if counts != (1, 0, 0) or not all(p.exists() for p in required):
            raise RuntimeError(f"export failed: {counts}")

    def export_task(self, result, output_dir):
        options = {key: getattr(self.exporter, key, default) for key, default in (
            ("export_doc_json", True), ("emit_formula_index", False),
            ("benchmark_mode", False), ("pdf_backend_name", "docling_parse"),
            ("_font_repair_records", {}), ("_font_repair_originals", {}))}
        return functools.partial(_docling_export, result, output_dir, options)

    def failure(self, path, output_dir, message):
        # Suppress any partial markdown and emit the existing failure schema.
        (output_dir / f"{path.stem}.md").unlink(missing_ok=True)
        json_dir = output_dir.parent / "json"
        for p in (json_dir / f"{path.stem}.docling.json", json_dir / f"{path.stem}.docling.json.zst",
                  json_dir / "metrics" / f"{path.stem}.metrics.json",
                  json_dir / "metrics" / f"{path.stem}.per_page.metrics.json"):
            p.unlink(missing_ok=True)
        original = self.exporter._original_extract_file(path)
        self.exporter._update_extraction_metadata(output_dir, path,
            "timeout" if message == "timeout" else "failure", "standard",
            page_count=self.exporter._get_pdf_page_count(original),
            no_partial_output=True)
        from shutil import copy2
        directories = ["problematic_files"] + (["timeout_files"] if message == "timeout" else [])
        for name in directories:
            destination = output_dir / name
            destination.mkdir(exist_ok=True)
            copy2(original, destination / original.name)


def corpus_batched_extract(self, original, *args, _auto_cuda=False, **kwargs):
    """Run the normal phase with batched dispatch (auto-CUDA: 8 workers, batch 8)."""
    signature = inspect.signature(original)
    bound = signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    if values.get("use_gpus") != "single" or values.get("formula_enrichment") or values.get("code_enrichment"):
        raise ValueError("batched prototype supports single-device layout-only extraction")
    if values.get("phase1_backend") == "safe":
        raise ValueError("batched extraction requires Docling")
    values["phase1_backend"] = "docling"
    # Configure only an already-existing extractor; the normal prime step
    # builds it if necessary. Intercept the final extract_path dispatch via
    # a temporary prime wrapper, preserving all upstream phase behavior.
    prime = self.prime_extractor
    restore_dispatch = []

    def attach():
        exporter = self.extractor
        saved = exporter.extract_path

        def dispatch(paths, out, skip_existing=True):
            # Match extract_path's per-invocation repair lifecycle. State
            # from a completed or failed call must never shorten this call.
            for attr in ("_font_repair_records", "_font_repair_paths",
                         "_font_repair_originals", "_font_repair_deadlines"):
                setattr(exporter, attr, {})
            exporter._font_repair_output_dir = Path(out)
            # Retain production chunking and other-format behavior. Only
            # normal-size PDFs take the new scheduler path.
            batchable, fallback = [], []
            for path in map(Path, paths):
                count = exporter._get_pdf_page_count(path) if path.suffix.lower() == ".pdf" else None
                if count is None or count > exporter.long_pdf_page_threshold:
                    fallback.append(path)
                else:
                    batchable.append(path)
            fallback_result = saved(fallback, out, skip_existing=skip_existing) if fallback else None
            if not batchable:
                return fallback_result
            from docling.datamodel.base_models import InputFormat
            exporter.converter.initialize_pipeline(InputFormat.PDF)
            adapter = DoclingAdapter.__new__(DoclingAdapter)
            adapter.converter = exporter.converter
            adapter.pipeline = exporter.converter._get_pipeline(InputFormat.PDF)
            adapter.validate(adapter.pipeline)
            adapter.exporter = exporter
            adapter.prepare_options = {"images_scale": adapter.pipeline.pipeline_options.images_scale}
            def budget(path):
                # Same page-derived export budget and repair remainder as
                # sequential, additionally bounded by the adopted pipeline.
                remaining = exporter._remaining_text_timeout(
                    path, exporter._calculate_text_timeout(file_path=path))
                pipeline_limit = getattr(adapter.pipeline.pipeline_options, "document_timeout", None)
                return min(remaining, pipeline_limit) if pipeline_limit is not None else remaining
            adapter.timeout = budget
            # Repair at admission, not while documents wait in the input
            # queue. Repair receives the sequential page-derived deadline.
            adapter.admit = lambda path, out: exporter._prepare_font_repair(path, Path(out))
            # Reuse sequential repair provenance before CPU workers see PDFs.
            original_paths = batchable
            from glossapi.corpus.corpus_state import _ProcessingStateManager
            from glossapi._naming import canonical_stem
            state = _ProcessingStateManager(Path(out) / ".processing_state.pkl",
                url_column=getattr(exporter, "url_column", "url"))
            processed, problematic = state.load()
            selected = [p for p in original_paths if not skip_existing or
                        canonical_stem(p) not in processed | problematic]
            runner = BatchedExtractor(adapter, layout_batch=8,
                workers=8 if _auto_cuda else min(4, values.get("num_threads") or 2))
            ok, bad = runner.extract(selected, out)
            names = {p.name: p.name for p in selected}
            names.update({Path(p).name: exporter._original_extract_file(Path(p)).name
                          for p in exporter._font_repair_originals})
            ok = [names[p] for p in ok]
            bad = [names[p] for p in bad]
            processed.update(canonical_stem(p) for p in ok)
            problematic.difference_update(canonical_stem(p) for p in ok)
            problematic.update(canonical_stem(p) for p in bad)
            state.save(processed, problematic)
            if getattr(exporter, "batch_result_callback", None):
                exporter.batch_result_callback(ok, bad)
            return ok, bad

        exporter.extract_path = dispatch
        restore_dispatch.append((exporter, saved))

    def prime_and_attach(*pa, **pk):
        result = prime(*pa, **pk)
        attach()
        return result

    self.prime_extractor = prime_and_attach
    try:
        if values.get("_prepared"):
            attach()
        return original(*bound.args, **bound.kwargs)
    finally:
        self.prime_extractor = prime
        for exporter, saved in restore_dispatch:
            exporter.extract_path = saved


def install_corpus_opt_in(corpus_class):
    """Compatibility installer for older/custom Corpus classes; reversible.

    Corpus classes with a static batched keyword need no installer.
    This helper never runs at module import and remains useful to older callers.
    """
    original = corpus_class.extract
    signature = inspect.signature(original)

    @functools.wraps(original)
    def extract(self, *args, batched=None, **kwargs):
        from glossapi.corpus.phase_extract import _resolve_batched_docling
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        backend = values.get("phase1_backend", "docling")
        resolver = getattr(self, "_resolve_phase1_backend", None)
        if callable(resolver):
            backend = resolver(backend, force_ocr=values.get("force_ocr", False),
                formula_enrichment=values.get("formula_enrichment", False),
                code_enrichment=values.get("code_enrichment", False))
        elif backend == "auto":
            backend = "docling" if (values.get("formula_enrichment") or values.get("code_enrichment")) else "safe"
        device = values.get("accel_type", "CUDA")
        if values.get("_prepared") and getattr(self, "extractor", None) is not None:
            options = getattr(self.extractor, "pipeline_options", None)
            device = getattr(getattr(options, "accelerator_options", None), "device", device)
        if not _resolve_batched_docling(batched, accel_type=device, resolved_backend=backend,
                use_gpus=values.get("use_gpus", "single"),
                formula_enrichment=values.get("formula_enrichment", False),
                code_enrichment=values.get("code_enrichment", False)):
            return original(self, *args, **kwargs)
        return corpus_batched_extract(self, original, *args, _auto_cuda=batched is None, **kwargs)

    parameters = list(signature.parameters.values())
    if "batched" not in signature.parameters:
        parameters.append(inspect.Parameter("batched", inspect.Parameter.KEYWORD_ONLY, default=None))
    extract.__signature__ = signature.replace(parameters=parameters)
    corpus_class.extract = extract
    return lambda: setattr(corpus_class, "extract", original)
