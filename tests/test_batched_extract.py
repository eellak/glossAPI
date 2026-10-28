"""Scheduler contract tests: spawned CPU workers, no installed Docling import."""
import importlib
import importlib.abc
import functools
import gc
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import tracemalloc
import weakref
from types import SimpleNamespace
from types import ModuleType

import pytest

from glossapi.scripts import batched_extract_benchmark as benchmark
from glossapi.scripts.docling_cost_benchmark import fake_docling, _FakeStatus

core = benchmark.core


def fake_prepare(path, page_no, *, max_pages=None):
    if path.stem == "parse_bad":
        raise ValueError("broken PDF")
    if path.stem == "crash":
        os._exit(17)
    if path.stem == "hang":
        time.sleep(60)
    if page_no == 0:
        return min(3, max_pages or 3)
    # Pages finish out of order; numbering must survive workers and batching.
    time.sleep(0.01 if page_no % 2 else 0.035)
    return {"text": f"{path.stem} page {page_no}", "worker": os.getpid()}


class FakeDocument:
    def __init__(self, pages):
        self.pages = {i + 1: None for i in range(len(pages))}
        self.text = "# Title\n\n" + "\n".join(p["text"] for p in pages) + "\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"

    def export_to_markdown(self):
        return self.text

    def export_to_dict(self, **kwargs):
        return {"text": self.text, "pages": list(self.pages)}


class WrongJsonDocument(FakeDocument):
    def export_to_dict(self, **kwargs):
        return {'wrong_document': True}


class FakeAdapter:
    prepare = staticmethod(fake_prepare)
    prepare_options = {}

    def __init__(self, exporter, *, delay=0, export_started=None, export_release=None):
        self.exporter = exporter
        self.delay = delay
        self.seen = []
        self.worker_pids = set()
        self.export_started, self.export_release = export_started, export_release

    def open(self, path, count):
        return SimpleNamespace(input=SimpleNamespace(file=path), timings={}), count

    def accept(self, result, page):
        self.worker_pids.add(page.payload.pop("worker"))

    def layout(self, batch, results):
        self.seen.append([(p.path.stem, p.page_no) for p in batch])
        if any(p.path.stem == "layout_bad" for p in batch):
            raise ValueError("bad model input")

    def complete(self, result, page):
        if page.path.stem == "table_bad":
            raise ValueError("table failure")

    def finish(self, result, pages):
        result.document = FakeDocument(pages)
        result.status = _FakeStatus.SUCCESS
        return result

    def export(self, result, output_dir):
        if result.input.file.stem == "export_bad":
            raise OSError("disk error")
        if self.export_started:
            self.export_started.set()
            assert self.export_release.wait(10)
        time.sleep(self.delay)
        assert self.exporter._export_documents([result], output_dir) == (1, 0, 0)

    def failure(self, path, out, message):
        self.exporter._update_extraction_metadata(out, path,
            "timeout" if message == "timeout" else "failure", "standard", no_partial_output=True)

    def export_task(self, result, output_dir):
        return functools.partial(fake_export, type(self), result, output_dir,
            self.delay, self.export_started, self.export_release)


def fake_export(adapter_type, result, output_dir, delay, started, release):
    # Spawn imports no installed Docling and carries no model/exporter lambdas.
    with fake_docling('baseline', {}):
        module = importlib.import_module('glossapi.gloss_extract')
        value = module.GlossExtract.__new__(module.GlossExtract)
        value._log = logging.getLogger('t31-test')
        value.benchmark_mode = False
        value.export_doc_json = True
        value.emit_formula_index = False
        value.pdf_backend_name = 'docling_parse'
        value._font_repair_originals = {}
        value._font_repair_records = {}
        value._get_pdf_page_count = lambda path: 3
        value._compute_per_page_metrics = lambda result: {'pages': [{'page_no': p} for p in result.document.pages]}
        adapter_type(value, delay=delay, export_started=started,
                     export_release=release).export(result, output_dir)


@pytest.fixture
def exporter(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "docling_version", lambda: "2.81.0")
    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "docling" or fullname.startswith(("docling.", "torch", "glossapi_rs")):
                raise ModuleNotFoundError(f"forbidden real heavy import: {fullname}")
    monkeypatch.setattr(sys, "meta_path", [Guard(), *sys.meta_path])
    with fake_docling("baseline", {}):
        module = importlib.import_module("glossapi.gloss_extract")
        value = module.GlossExtract.__new__(module.GlossExtract)
        value._log = logging.getLogger("t31-test")
        value.benchmark_mode = False
        value.export_doc_json = True
        value.emit_formula_index = False
        value.pdf_backend_name = "docling_parse"
        value._font_repair_originals = {}
        value._font_repair_records = {}
        value._get_pdf_page_count = lambda path: 3
        value._compute_per_page_metrics = lambda result: {"pages": [{"page_no": p} for p in result.document.pages]}
        yield value


def inputs(tmp_path, names):
    paths = [tmp_path / f"{name}.pdf" for name in names]
    for path in paths:
        path.write_bytes(b"fake source")
    return paths


def artifacts(root):
    # zstd is optional in the lightweight test environment. Compare raw bytes
    # because both fake routes deliberately have deterministic empty timings.
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_mixed_batches_order_reassembly_and_sequential_artifacts(tmp_path, exporter):
    paths = inputs(tmp_path, ["a", "b", "c", "d"])
    adapter = FakeAdapter(exporter)
    runner = core.BatchedExtractor(adapter, workers=3, layout_batch=8,
                                   queue_max_size=12, batch_wait_seconds=0.2)
    ok, bad = runner.extract(paths, tmp_path / "batched" / "markdown")
    assert ok == [p.name for p in paths] and bad == []
    assert any(len({stem for stem, _ in batch}) > 1 for batch in adapter.seen)
    assert len(adapter.worker_pids) >= 2 and os.getpid() not in adapter.worker_pids
    sequential = tmp_path / "sequential" / "markdown"
    for path in paths:
        result, count = adapter.open(path, fake_prepare(path, 0))
        pages = []
        for number in range(1, count + 1):
            page = core.PageWork(0, path, number, fake_prepare(path, number))
            adapter.accept(result, page)
            adapter.layout([page], {0: result})
            adapter.complete(result, page)
            pages.append(page.payload)
        adapter.export(adapter.finish(result, pages), sequential)
    assert artifacts(tmp_path / "batched") == artifacts(tmp_path / "sequential")
    assert any("docling.json" in key for key in artifacts(tmp_path / "batched"))
    assert any("per_page.metrics.json" in key for key in artifacts(tmp_path / "batched"))
    for path in paths:
        text = (tmp_path / "batched" / "markdown" / f"{path.stem}.md").read_text()
        assert text.index("page 1") < text.index("page 2") < text.index("page 3")


@pytest.mark.parametrize("bad_name", ["parse_bad", "layout_bad", "table_bad", "export_bad", "crash", "hang"])
def test_document_failure_isolated_and_parser_replaced(tmp_path, exporter, bad_name):
    paths = inputs(tmp_path, [bad_name, "good", "also_good"])
    adapter = FakeAdapter(exporter)
    # Spawn startup is outside an ordinary GPU forward but within admission.
    # Give healthy documents enough startup time, and override only the hang.
    adapter.timeout = lambda path: 1.5 if path.stem == "hang" else 15
    runner = core.BatchedExtractor(adapter, workers=2, layout_batch=6,
                                   batch_wait_seconds=0.15)
    began = time.monotonic()
    ok, bad = runner.extract(paths, tmp_path / "out" / "markdown")
    assert ok == ["good.pdf", "also_good.pdf"]
    assert bad == [f"{bad_name}.pdf"]
    assert not (tmp_path / "out" / "markdown" / f"{bad_name}.md").exists()
    sidecar = json.loads((tmp_path / "out" / "sidecars" / "extract" / f"{bad_name}.json").read_text())
    assert sidecar["failure_mode"] == ("timeout" if bad_name == "hang" else "failure")
    assert sidecar["no_partial_output"]
    assert time.monotonic() - began < 15


class StalledExportAdapter(FakeAdapter):
    def export(self, result, output_dir):
        if result.input.file.stem == 'export_stall':
            (output_dir.parent / 'export_stall.entered').write_text(str(os.getpid()))
            (output_dir / 'export_stall.md').write_text('incomplete export')
            time.sleep(6)  # Bounded reproduction of the advisory's 60s stall.
        super().export(result, output_dir)


def live_resources():
    """Live RSS, not ru_maxrss (which cannot prove reclamation)."""
    import ctypes
    if sys.platform == 'darwin':
        class TaskInfo(ctypes.Structure):
            _fields_ = [('virtual', ctypes.c_uint64), ('resident', ctypes.c_uint64),
                       ('maximum', ctypes.c_uint64), ('rest', ctypes.c_uint32 * 6)]
        system = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        system.mach_task_self.restype = ctypes.c_uint32
        info = TaskInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
        assert system.task_info(system.mach_task_self(), 20, ctypes.byref(info),
                                ctypes.byref(count)) == 0
        rss = info.resident
        descriptors = len(os.listdir('/dev/fd'))
    else:
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        descriptors = len(os.listdir('/proc/self/fd'))
    return dict(threads=threading.active_count(), descriptors=descriptors, rss_bytes=rss)


def export_resource_baseline():
    # Spawn retains one resource-tracker descriptor for the Python session.
    # Count it before measuring per-worker allocation/reclamation.
    from multiprocessing.resource_tracker import ensure_running
    ensure_running()
    return live_resources()


class RetainedPayload:
    def __init__(self, marker):
        self.marker = marker
        self.retained = bytearray(b'x' * (4 * 1024 * 1024))

    def __reduce__(self):
        self.marker.write_text(str(os.getpid()))
        time.sleep(60)
        return int, (1,)


class ResourceStallAdapter(FakeAdapter):
    def __init__(self, exporter, **kwargs):
        super().__init__(exporter, **kwargs)
        self.samples = []
        self.baseline = None

    def export_task(self, result, output_dir):
        if self.baseline is None:
            self.baseline = live_resources()
        if result.input.file.stem.startswith('held'):
            return functools.partial(write_transferred_payload,
                output_dir / f'{result.input.file.stem}.md',
                RetainedPayload(output_dir.parent / f'{result.input.file.stem}.entered'))
        return super().export_task(result, output_dir)

    def failure(self, path, out, message):
        super().failure(path, out, message)
        self.samples.append(live_resources())


def serialization_stall_resources(tmp_path, exporter, count):
    """Exercise the scheduler and return only scalar resource/artifact receipts."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    gc.collect()
    before = export_resource_baseline()
    children = {p.pid for p in core.mp.active_children()}
    adapter = ResourceStallAdapter(exporter)
    adapter.timeout = lambda path: 2 if path.stem.startswith('held') else 15
    runner = core.BatchedExtractor(adapter, workers=1, layout_batch=1,
        queue_max_size=1, export_queue_size=1, max_active_documents=1)
    names = [f'held{i:03}' for i in range(count)]
    ok, bad = runner.extract(inputs(tmp_path, [*names, 'healthy']),
                             tmp_path / 'out/markdown', max_pages=1)
    assert ok == ['healthy.pdf'] and bad == [f'{name}.pdf' for name in names], runner.errors
    assert all(error == 'timeout' for error in runner.errors.values())
    pids = [int((tmp_path / f'out/{name}.entered').read_text()) for name in names]
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert len(adapter.samples) == count
    assert all(s['threads'] == adapter.baseline['threads'] for s in adapter.samples)
    assert all(s['descriptors'] == adapter.baseline['descriptors'] for s in adapter.samples)
    for name in names:
        assert not (tmp_path / f'out/markdown/{name}.md').exists()
        failure = json.loads((tmp_path / f'out/sidecars/extract/{name}.json').read_text())
        assert failure['failure_mode'] == 'timeout' and failure['no_partial_output']
    assert all(value == 1 for value in runner.high_water.values())
    assert {p.pid for p in core.mp.active_children()} == children
    gc.collect()
    after = live_resources()
    assert after['threads'] == before['threads']
    assert after['descriptors'] == before['descriptors']
    return dict(before=before, baseline=adapter.baseline, samples=adapter.samples,
                after=after, children_before=sorted(children),
                children_after=sorted(p.pid for p in core.mp.active_children()),
                high_water=runner.high_water)


@pytest.mark.linux_dev_node
@pytest.mark.skipif(sys.platform != 'linux', reason='absolute RSS contract runs on the Linux dev node')
def test_twelve_serialization_stalls_have_constant_resources(tmp_path, exporter, record_property):
    # Preserve the original absolute gate on the deployment allocator. macOS
    # keeps reusable malloc pages resident; the portable gate below measures
    # warmed growth and live retention instead. See CP0 reply / rss_slope.json.
    receipt = serialization_stall_resources(tmp_path, exporter, 12)
    record_property('resource_counts', json.dumps(receipt))
    (tmp_path / 'resource-receipt.json').write_text(json.dumps(receipt, indent=2))
    assert max(s['rss_bytes'] for s in receipt['samples']) - receipt['baseline']['rss_bytes'] < 24 * 1024 * 1024


def payload_lifetimes(refs):
    # Return scalars only: the observer must never extend a payload's lifetime.
    live = [payload for ref in refs if (payload := ref()) is not None]
    return dict(created_count=len(refs), retained_count=len(live),
                retained_bytes=sum(len(payload.retained) for payload in live),
                retained_markers=[str(payload.marker) for payload in live])


def serialization_stall_contract(tmp_path, exporter, record_property, *, counts=(96, 12, 48)):
    # CP0's three fresh-process 12/24/48/96 probes show late allocator steps
    # through 96 stalls on macOS, then a common high-water ceiling. Warm that
    # exact workload before comparing independent short/long runs. Every warm-up
    # stall also checks reaping, no partial output, and bounded scheduler queues.
    gc.collect()
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    try:
        refs = {name: [] for name in ('warmup', 'n12', 'n48')}
        checkpoints = {}
        windows = {}
        original_init = RetainedPayload.__init__

        def tracked_init(payload, marker):
            original_init(payload, marker)
            refs[window].append(weakref.ref(payload))

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(RetainedPayload, '__init__', tracked_init)
            gc.collect()
            # Both tracing and the baseline precede the FIRST warm-up payload.
            baseline_current = tracemalloc.get_traced_memory()[0]
            for (window, window_refs), count in zip(refs.items(), counts):
                windows[window] = serialization_stall_resources(tmp_path / window, exporter, count)
                gc.collect()
                checkpoints[window] = payload_lifetimes(window_refs)
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        lifetimes = {name: payload_lifetimes(window_refs) for name, window_refs in refs.items()}
        receipt = dict(**windows, payload_lifetimes=lifetimes,
                       payload_checkpoints=checkpoints,
                       retained_count=sum(s['retained_count'] for s in lifetimes.values()),
                       retained_bytes=sum(s['retained_bytes'] for s in lifetimes.values()),
                       traced_baseline_bytes=baseline_current,
                       traced_current_bytes=current, traced_peak_bytes=peak,
                       rss_growth_bytes=windows['n48']['after']['rss_bytes'] - windows['n12']['after']['rss_bytes'])
        record_property('resource_counts', json.dumps(receipt))
        (tmp_path / 'resource-receipt.json').write_text(json.dumps(receipt, indent=2))
        assert [s['created_count'] for s in lifetimes.values()] == list(counts)
        assert receipt['retained_count'] == 0, (
            f"surviving payloads: {receipt['retained_count']} / {receipt['retained_bytes']} bytes")
        assert all(s['retained_count'] == 0 for s in checkpoints.values()), checkpoints
        # One retained 4 MiB payload must fail even when the OS reuses RSS pages.
        assert current - baseline_current < 1024 * 1024
        # 8 MiB bounds allocator noise (CP0 observed <=6.13 MiB from 12 to 48),
        # while refusing growth proportional to the number of stalled payloads.
        assert receipt['rss_growth_bytes'] < 8 * 1024 * 1024
    finally:
        if owns_tracing:
            tracemalloc.stop()


def test_warmed_serialization_stalls_have_bounded_retention(tmp_path, exporter, record_property):
    serialization_stall_contract(tmp_path, exporter, record_property)


@pytest.mark.parametrize('mutation', ['one-retention', 'growing-retention'])
def test_serialization_resource_oracle_rejects_retention(tmp_path, exporter, record_property,
                                                       monkeypatch, mutation):
    # The reviewer's two production mutations, applied at the same export entry.
    # Shorter windows keep the permanent oracle affordable; the ordinary gate
    # above still exercises the exact warmed 96/12/48 workload and RSS bound.
    kept_tasks = []
    monkeypatch.setattr(core, '_cp0_kept_tasks', kept_tasks, raising=False)
    original_export = core._ExportSession.export

    def retaining_export(session, task, deadline, *args, **kwargs):
        if mutation == 'growing-retention' or not core._cp0_kept_tasks:
            core._cp0_kept_tasks.append(task)
        return original_export(session, task, deadline, *args, **kwargs)

    monkeypatch.setattr(core._ExportSession, 'export', retaining_export)
    try:
        with pytest.raises(AssertionError, match='surviving payloads:'):
            serialization_stall_contract(tmp_path, exporter, record_property, counts=(1, 1, 2))
        receipt = json.loads((tmp_path / 'resource-receipt.json').read_text())
        expected = [1, 0, 0] if mutation == 'one-retention' else [1, 1, 2]
        assert [s['retained_count'] for s in receipt['payload_lifetimes'].values()] == expected
        held = [payload for task in kept_tasks for payload in task.args
                if isinstance(payload, RetainedPayload)]
        assert len(held) == receipt['retained_count'] == sum(expected)
        assert sum(len(payload.retained) for payload in held) == receipt['retained_bytes'] == sum(expected) * 4 * 1024 * 1024
        assert receipt['traced_current_bytes'] - receipt['traced_baseline_bytes'] >= receipt['retained_bytes']
        del held
    finally:
        kept_tasks.clear()
        gc.collect()


class SlowPicklingPayload:
    def __init__(self, marker, delay=5):
        self.marker, self.delay = marker, delay

    def __reduce__(self):
        self.marker.write_text(f'serialization entered {os.getpid()}')
        time.sleep(self.delay)
        return int, (1,)


class SlowStatePayload:
    def __init__(self, marker):
        self.marker = marker

    def __getstate__(self):
        self.marker.write_text(f'state serialization entered {os.getpid()}')
        time.sleep(60)
        return self.__dict__


def write_transferred_payload(path, payload):
    path.write_text(str(payload))


class SlowTransferAdapter(FakeAdapter):
    def export_task(self, result, output_dir):
        if result.input.file.stem == 'transfer_stall':
            return functools.partial(write_transferred_payload,
                output_dir / 'transfer_stall.md',
                SlowPicklingPayload(output_dir.parent / 'transfer.entered'))
        return super().export_task(result, output_dir)


def assert_no_export_helpers(threads):
    assert set(threading.enumerate()) == threads


@pytest.mark.parametrize('payload_type', [SlowPicklingPayload, SlowStatePayload])
def test_export_startup_serialization_is_deadline_bounded(tmp_path, record_property, payload_type):
    threads = set(threading.enumerate())
    children = {p.pid for p in core.mp.active_children()}
    output = tmp_path / 'late.md'
    marker = tmp_path / 'pickling.entered'
    task = functools.partial(write_transferred_payload, output, payload_type(marker))
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            core._bounded_export(core.mp.get_context('spawn'), task, began + 1.5)
        elapsed = time.monotonic() - began
        record_property('bounded_export_wall_seconds', elapsed)
        assert marker.exists()  # The test reached pickling, not just spawn setup.
        assert elapsed < 2, f'export startup took {elapsed:.3f}s for a 1.5s deadline'
    finally:
        assert_no_export_helpers(threads)
    assert not output.exists()  # A late start cannot publish after the timeout.
    assert {p.pid for p in core.mp.active_children()} == children


def test_export_transfer_timeout_preserves_backend_and_next_document(tmp_path, exporter, record_property):
    threads = set(threading.enumerate())
    children = {p.pid for p in core.mp.active_children()}
    adapter = SlowTransferAdapter(exporter)
    adapter.timeout = lambda path: 1.5 if path.stem == 'transfer_stall' else 15
    runner = core.BatchedExtractor(adapter, workers=1, layout_batch=1,
        max_active_documents=1, export_queue_size=1)
    output = tmp_path / 'out/markdown'
    began = time.monotonic()
    try:
        ok, bad = runner.extract(inputs(tmp_path, ['transfer_stall', 'next']),
                                 output, max_pages=1)
        elapsed = time.monotonic() - began
        record_property('scheduler_wall_seconds', elapsed)
        assert ok == ['next.pdf'] and bad == ['transfer_stall.pdf']
        assert runner.errors['transfer_stall.pdf'] == 'timeout'
        assert (output.parent / 'transfer.entered').exists()
        assert elapsed < 4, f'stalled transfer blocked recovery for {elapsed:.3f}s'
        assert runner.adapter is adapter and adapter.exporter is exporter
        assert any(stem == 'next' for batch in adapter.seen for stem, _ in batch)
    finally:
        assert_no_export_helpers(threads)
    assert not (output / 'transfer_stall.md').exists()
    receipt = json.loads((output.parent / 'sidecars/extract/transfer_stall.json').read_text())
    assert receipt['failure_mode'] == 'timeout' and receipt['no_partial_output']
    assert {p.pid for p in core.mp.active_children()} == children


class UnpicklablePayload:
    def __reduce__(self):
        raise ValueError('export payload cannot be serialized')


def test_export_startup_error_propagates_and_reaps(tmp_path):
    children = {p.pid for p in core.mp.active_children()}
    task = functools.partial(write_transferred_payload, tmp_path / 'bad.md', UnpicklablePayload())
    with pytest.raises(ValueError, match='export payload cannot be serialized'):
        core._bounded_export(core.mp.get_context('spawn'), task, time.monotonic() + 5)
    assert not (tmp_path / 'bad.md').exists()
    assert {p.pid for p in core.mp.active_children()} == children


def write_worker_pid(path):
    path.write_text(str(os.getpid()))


def test_export_worker_persists_and_restarts_before_reusing_output(tmp_path):
    session = core._ExportSession(core.mp.get_context('spawn'))
    baseline = export_resource_baseline()
    output = tmp_path / 'slot.md'
    marker = tmp_path / 'serialization.entered'
    try:
        session.export(functools.partial(write_worker_pid, output), time.monotonic() + 5)
        first_pid = int(output.read_text())
        session.export(functools.partial(write_worker_pid, output), time.monotonic() + 5)
        assert int(output.read_text()) == first_pid
        assert session.epoch == 1
        with pytest.raises(TimeoutError):
            session.export(functools.partial(write_transferred_payload, output,
                SlowPicklingPayload(marker)), time.monotonic() + 1.5)
        assert int(marker.read_text().split()[-1]) == first_pid
        with pytest.raises(ProcessLookupError):
            os.kill(first_pid, 0)
        session.export(functools.partial(write_worker_pid, output), time.monotonic() + 5)
        fresh_pid = int(output.read_text())
        assert fresh_pid != first_pid and session.epoch == 2
        saved = output.read_bytes()
        time.sleep(0.1)
        assert output.read_bytes() == saved
    finally:
        session.close()
    assert live_resources()['threads'] == baseline['threads']
    assert live_resources()['descriptors'] == baseline['descriptors']


def delayed_export_start(channel, epoch):
    time.sleep(60)


def unread_export_queue(channel, epoch):
    import pickle
    core._ExportQueue(channel).put(pickle.dumps((epoch, 'ready')), 5)
    time.sleep(60)


@pytest.mark.parametrize('worker', [delayed_export_start, unread_export_queue])
def test_export_startup_and_queue_put_are_bounded(tmp_path, monkeypatch, worker):
    monkeypatch.setattr(core, '_export_worker', worker)
    children = {p.pid for p in core.mp.active_children()}
    baseline = export_resource_baseline()
    task = functools.partial(write_transferred_payload, tmp_path / 'late.md', b'x' * (8 * 1024 * 1024))
    began = time.monotonic()
    with pytest.raises(TimeoutError):
        core._bounded_export(core.mp.get_context('spawn'), task, began + 1.5)
    assert time.monotonic() - began < 2
    assert not (tmp_path / 'late.md').exists()
    assert {p.pid for p in core.mp.active_children()} == children
    assert live_resources()['threads'] == baseline['threads']
    assert live_resources()['descriptors'] == baseline['descriptors']


def test_export_process_start_error_closes_descriptors(tmp_path, monkeypatch):
    baseline = export_resource_baseline()
    def broken_start(self):
        raise OSError('spawn unavailable')
    monkeypatch.setattr(core.mp.process.BaseProcess, 'start', broken_start)
    with pytest.raises(OSError, match='spawn unavailable'):
        core._bounded_export(core.mp.get_context('spawn'),
            functools.partial(write_worker_pid, tmp_path / 'late.md'), time.monotonic() + 5)
    assert live_resources()['descriptors'] == baseline['descriptors']


def test_export_rejects_old_epoch_receipt(tmp_path, monkeypatch):
    import pickle
    session = core._ExportSession(core.mp.get_context('spawn'))
    try:
        session.export(functools.partial(write_worker_pid, tmp_path / 'first'), time.monotonic() + 5)
        original = session.jobs.get
        def old_receipt(deadline):
            epoch, sequence, completed, error = pickle.loads(original(deadline))
            return pickle.dumps((epoch - 1, sequence, completed, error))
        monkeypatch.setattr(session.jobs, 'get', old_receipt)
        with pytest.raises(RuntimeError, match='stale export receipt'):
            session.export(functools.partial(write_worker_pid, tmp_path / 'second'), time.monotonic() + 5)
        assert session.process is None and session.jobs is None
    finally:
        session.close()


def test_export_deadline_isolated_and_resident_backend_survives(tmp_path, exporter):
    paths = inputs(tmp_path, ['export_stall', 'next'])
    adapter = StalledExportAdapter(exporter)
    adapter.timeout = lambda path: 2 if path.stem == 'export_stall' else 15
    runner = core.BatchedExtractor(adapter, workers=1, layout_batch=1,
        max_active_documents=1, export_queue_size=1, document_timeout=2)
    before = time.monotonic()
    ok, bad = runner.extract(paths, tmp_path / 'out/markdown', max_pages=1)
    assert ok == ['next.pdf'] and bad == ['export_stall.pdf']
    assert runner.errors['export_stall.pdf'] == 'timeout'
    export_pid = int((tmp_path / 'out/export_stall.entered').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(export_pid, 0)  # The timed-out child is reaped, not abandoned.
    assert not (tmp_path / 'out/markdown/export_stall.md').exists()
    receipt = json.loads((tmp_path / 'out/sidecars/extract/export_stall.json').read_text())
    assert receipt['failure_mode'] == 'timeout' and receipt['no_partial_output']
    assert any(stem == 'next' for batch in adapter.seen for stem, _ in batch)
    assert time.monotonic() - before < 6


def test_export_backpressure_reaches_inference_and_admission(tmp_path, exporter):
    started, release = FileEvent(tmp_path / 'started'), FileEvent(tmp_path / 'release')
    adapter = FakeAdapter(exporter, export_started=started, export_release=release)
    paths = inputs(tmp_path, [f"doc{i}" for i in range(12)])
    runner = core.BatchedExtractor(adapter, workers=2, layout_batch=2,
        queue_max_size=3, export_queue_size=1, max_active_documents=3)
    errors = []

    def run():
        try:
            runner.extract(paths, tmp_path / "out" / "markdown", max_pages=1)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(10)
        batches = len(adapter.seen)
        time.sleep(0.2)
        assert len(adapter.seen) == batches  # consumer blocked behind writer
        assert runner.high_water["exports"] == 1
        assert runner.high_water["prepared_pages"] <= 3
        assert runner.high_water["active_documents"] <= 3
        assert runner.high_water["parse_jobs"] <= 2
    finally:
        release.set()
        thread.join(15)
    assert not thread.is_alive() and not errors
    assert len(runner.processed_files) == 12


class FileEvent:
    """Cross-process test gate that can travel after persistent worker startup."""
    def __init__(self, path):
        self.path = path

    def set(self):
        self.path.touch()

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while not self.path.exists():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True


def test_duplicate_stems_rejected_before_output(tmp_path, exporter):
    runner = core.BatchedExtractor(FakeAdapter(exporter))
    with pytest.raises(ValueError, match="duplicate"):
        runner.extract([tmp_path / "a.pdf", tmp_path / "other" / "a.pdf"], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_corpus_opt_in_false_preserves_arguments_and_is_reversible():
    class Corpus:
        def extract(self, x=1, *, use_gpus="single", phase1_backend="auto"):
            return x, use_gpus, phase1_backend
    original = Corpus.extract
    restore = core.install_corpus_opt_in(Corpus)
    try:
        assert Corpus().extract(2, batched=False) == (2, "single", "auto")
        with pytest.raises(ValueError, match="single-device"):
            Corpus().extract(batched=True, use_gpus="multi")
    finally:
        restore()
    assert Corpus.extract is original


def test_benchmark_equality_missing_outputs_cannot_pass(tmp_path):
    checks = benchmark.equality(tmp_path, ["a.pdf"])
    assert checks[0]["status"] == "missing_output"
    assert not checks[0]["passes_gate"]
    assert all(checks[0][key] is False for key in ("exact_markdown_equal", "docling_json_equal", "metrics_equal", "per_page_metrics_equal", "sidecar_equal"))


def test_internal_adapter_one_forward_and_document_confidence_isolation(exporter, monkeypatch):
    class ProfilingItem:
        def __init__(self, *, scope):
            self.scope, self.count, self.times, self.start_timestamps = scope, 0, [], []
    profiling = ModuleType("docling.utils.profiling")
    profiling.ProfilingItem = ProfilingItem
    profiling.ProfilingScope = SimpleNamespace(PAGE="page")
    monkeypatch.setitem(sys.modules, "docling.utils.profiling", profiling)
    calls = []

    class Predictor:
        def predict_batch(self, images):
            calls.append(list(images))
            return [[image] for image in images]

    class Model:
        def __init__(self):
            self.layout_predictor = Predictor()
        def __call__(self, result, pages):
            for page, predictions in zip(pages, self.layout_predictor.predict_batch([p.image for p in pages])):
                if page.image == "bad":
                    raise ValueError("postprocessing fails only this document")
                result.confidence.pages[page.page_no] = predictions[0]
                yield page

    adapter = core.DoclingAdapter.__new__(core.DoclingAdapter)
    adapter.pipeline = SimpleNamespace(layout_model=Model())
    adapter.per_page_layout_timings()
    adapter.per_page_layout_timings()  # idempotent; batched replay must unwrap it
    results = {i: SimpleNamespace(confidence=SimpleNamespace(pages={}), timings={}) for i in range(3)}
    batch = []
    for did, label in enumerate(["a", "bad", "c"]):
        page = SimpleNamespace(page_no=1, image=label, _t31_timings={}, get_image=lambda scale, value=label: value)
        batch.append(core.PageWork(did, Path(f"{label}.pdf"), 1, page))
    adapter.layout(batch, results)
    assert calls == [["a", "bad", "c"]]  # only one actual predictor forward
    assert results[0].confidence.pages == {1: "a"}
    assert results[2].confidence.pages == {1: "c"}
    assert batch[1].error.startswith("layout postprocess:")
    assert batch[0].error is None and batch[2].error is None
    assert all(result.timings == {} for result in results.values())


def test_internal_adapter_timings_reassembled_by_page_number(exporter):
    adapter = core.DoclingAdapter.__new__(core.DoclingAdapter)
    adapter.pipeline = SimpleNamespace(_assemble_document=lambda r: r, _enrich_document=lambda r: r)
    def item(value):
        return SimpleNamespace(times=[value], count=1, start_timestamps=[value])
    pages = [SimpleNamespace(page_no=n, _t31_timings={"page_parse": item(n / 10)}) for n in range(1, 4)]
    result = SimpleNamespace(timings={})
    adapter.finish(result, pages)
    assert result.timings["page_parse"].times == [0.1, 0.2, 0.3]
    assert result.timings["page_parse"].count == 3
    assert all(not hasattr(p, "_t31_timings") for p in pages)


@pytest.mark.parametrize("auto_cuda", [False, True])
def test_corpus_opt_in_true_routes_preserves_fallback_state_and_restores(tmp_path, exporter, monkeypatch, auto_cuda):
    paths = inputs(tmp_path, ["a", "b", "long"])
    events = []
    state_data = {"processed": set(), "problematic": set()}
    state_module = ModuleType("glossapi.corpus.corpus_state")

    class State:
        def __init__(self, *args, **kwargs):
            pass
        def load(self):
            return set(state_data["processed"]), set(state_data["problematic"])
        def save(self, processed, problematic):
            state_data.update(processed=processed, problematic=problematic)
    state_module._ProcessingStateManager = State
    monkeypatch.setitem(sys.modules, "glossapi.corpus.corpus_state", state_module)

    class ExistingExtractor:
        converter = SimpleNamespace(initialize_pipeline=lambda fmt: None,
                                     _get_pipeline=lambda fmt: SimpleNamespace(
                                         pipeline_options=SimpleNamespace(images_scale=1.25)))
        long_pdf_page_threshold = 600
        url_column = "url"
        batch_result_callback = None
        def extract_path(self, paths, out, skip_existing=True):
            events.append(("fallback", [p.name for p in paths]))
        def _get_pdf_page_count(self, path):
            return 900 if path.stem == "long" else 3
        def _prepare_font_repair(self, path, out):
            events.append(("repair", path.name))
            return path
        def _remaining_text_timeout(self, path, budget):
            return budget
        def _calculate_text_timeout(self, *, file_path):
            return 60
        def _original_extract_file(self, path):
            return path

    class Runner:
        def __init__(self, adapter, **kwargs):
            assert adapter.converter is existing.converter
            assert kwargs['layout_batch'] == 8
            assert kwargs['workers'] == (8 if auto_cuda else 2)
            self.adapter = adapter
        def extract(self, paths, out):
            for path in paths:
                self.adapter.admit(path, out)
            events.append(("batched", [p.name for p in paths]))
            return [paths[0].name], [paths[1].name]

    monkeypatch.setattr(core.DoclingAdapter, "validate", staticmethod(lambda pipeline: None))
    monkeypatch.setattr(core, "BatchedExtractor", Runner)
    existing = ExistingExtractor()
    saved_dispatch = existing.extract_path

    class Corpus:
        extractor = existing
        def prime_extractor(self):
            events.append(("prime",))
        def extract(self, *, use_gpus="single", phase1_backend="auto", num_threads=2,
                    _prepared=False, skip_existing=True):
            assert phase1_backend == "docling"
            if not _prepared:
                self.prime_extractor()
            self.extractor.extract_path(paths, tmp_path / "out" / "markdown", skip_existing=skip_existing)

    restore = core.install_corpus_opt_in(Corpus)
    corpus = Corpus()
    try:
        if auto_cuda:
            core.corpus_batched_extract(corpus, Corpus.extract.__wrapped__, _auto_cuda=True)
        else:
            corpus.extract(batched=True)
    finally:
        restore()
    assert events == [("prime",), ("fallback", ["long.pdf"]), ("repair", "a.pdf"),
                      ("repair", "b.pdf"), ("batched", ["a.pdf", "b.pdf"])]
    assert state_data == {"processed": {"a"}, "problematic": {"b"}}
    assert existing.extract_path == saved_dispatch


def test_validation_and_timeout_during_inference(tmp_path, exporter):
    with pytest.raises(ValueError, match="positive"):
        core.BatchedExtractor(FakeAdapter(exporter), workers=0)
    adapter = FakeAdapter(exporter)
    adapter.timeout = lambda path: 0.4 if path.stem == "slow_layout" else 10
    def layout(batch, results):
        if any(page.path.stem == "slow_layout" for page in batch):
            time.sleep(0.45)
    adapter.layout = layout
    runner = core.BatchedExtractor(adapter, workers=2, layout_batch=6, batch_wait_seconds=.05)
    ok, bad = runner.extract(inputs(tmp_path, ["slow_layout", "healthy"]), tmp_path / "out" / "markdown")
    assert ok == ["healthy.pdf"] and bad == ["slow_layout.pdf"]
    assert runner.errors["slow_layout.pdf"] == "timeout"


def test_benchmark_both_modes_receipts_and_equality(tmp_path, exporter, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    inputs(pdf_dir, ["a", "b"])
    monkeypatch.setattr(benchmark, "page_count", lambda path: 3)

    def adapter_factory(**kwargs):
        adapter = FakeAdapter(exporter)
        adapter.pipeline = SimpleNamespace(pipeline_options=SimpleNamespace(model_dump=lambda **kw: {"layout_batch_size" if k == "layout_batch" else k: v for k, v in kwargs.items()}))
        def convert(path, *, raises_on_error, page_range):
            result, count = adapter.open(path, fake_prepare(path, 0, max_pages=page_range[1]))
            pages = [{"text": f"{path.stem} page {i}"} for i in range(1, count + 1)]
            result = adapter.finish(result, pages)
            result.pages = pages
            return result
        adapter.converter = SimpleNamespace(convert=convert)
        return adapter
    monkeypatch.setattr(core, "DoclingAdapter", adapter_factory)
    args = SimpleNamespace(pdfs=str(pdf_dir), out=str(tmp_path / "out"), mode="sequential",
        workers=2, layout_batch=8, max_pages=2, device="cpu", artifacts_path=None,
        queue_max_size=4, document_timeout=60)
    assert benchmark.run(args) == 0
    args.mode = "batched"
    assert benchmark.run(args) == 0
    receipt = json.loads((tmp_path / "out" / "batched" / "summary.json").read_text())
    assert receipt["pages"] == receipt["expected_pages"] == 4
    assert receipt["wall_seconds"] > 0
    assert receipt["seconds_per_page"] == receipt["wall_seconds"] / 4
    assert receipt["fidelity_verified"] and receipt["reference_valid"]
    assert all(c["text_similarity"] == 1 and c["docling_json_equal"] for c in receipt["equality_checks"])
    assert not receipt["gpu_claim"]
    with pytest.raises(ValueError, match="nonempty"):
        benchmark.run(args)


def test_failure_sidecar_error_does_not_stop_other_documents(tmp_path, exporter):
    adapter = FakeAdapter(exporter)
    def cannot_record(path, out, message):
        raise OSError("sidecar unwritable")
    adapter.failure = cannot_record
    runner = core.BatchedExtractor(adapter, workers=2)
    ok, bad = runner.extract(inputs(tmp_path, ["export_bad", "healthy"]), tmp_path / "out" / "markdown")
    assert ok == ["healthy.pdf"] and bad == ["export_bad.pdf"]
    assert "failure recording" in runner.errors["export_bad.pdf"]


# CP-B1 reviewer witnesses, inverted to require the corrected behavior.
@pytest.mark.parametrize("device,binding_change", [("cpu", None), ("cuda:0", None),
    ("cpu", "docling_version"), ("cpu", "options_sha256"), ("cpu", "device")])
def test_gate_refuses_wrong_docling_json_and_different_device(tmp_path, exporter, monkeypatch, device, binding_change):
    pdfs = tmp_path/'pdfs'
    pdfs.mkdir()
    inputs(pdfs,['a','b'])
    monkeypatch.setattr(benchmark,'page_count',lambda p:3)
    def factory(**kwargs):
        adapter = FakeAdapter(exporter)
        adapter.pipeline = SimpleNamespace(pipeline_options=SimpleNamespace(model_dump=lambda **kw: {"layout_batch_size" if k == "layout_batch" else k: v for k, v in kwargs.items()}))
        def convert(path, *, raises_on_error, page_range):
            result, count = adapter.open(path, fake_prepare(path,0,max_pages=page_range[1]))
            result = adapter.finish(result,[{'text':f'{path.stem} page {i}'} for i in range(1,count+1)])
            result.pages = list(result.document.pages)
            return result
        adapter.converter=SimpleNamespace(convert=convert)
        original_finish=adapter.finish
        def finish(result,pages):
            result = original_finish(result,pages)
            if args.mode == 'batched':
                # Inject before IPC using a picklable document type. The same
                # wrong JSON still reaches the real exporter/fidelity gate.
                result.document.__class__ = WrongJsonDocument
            return result
        adapter.finish=finish
        return adapter
    monkeypatch.setattr(core,'DoclingAdapter',factory)
    args=SimpleNamespace(pdfs=str(pdfs),out=str(tmp_path/'out'),mode='sequential',
        workers=2,layout_batch=8,max_pages=2,device='cpu',artifacts_path=None,
        queue_max_size=4,document_timeout=60)
    assert benchmark.run(args)==0
    if binding_change:
        baseline = tmp_path / 'out/sequential/summary.json'
        data = json.loads(baseline.read_text())
        data.setdefault('baseline_manifest', {})[binding_change] = 'incompatible'
        baseline.write_text(json.dumps(data))
    args.mode='batched'
    args.device=device
    result=benchmark.run(args)
    receipt=json.loads((tmp_path/'out/batched/summary.json').read_text())
    assert result == 1 and not receipt['fidelity_verified']
    assert receipt['reference_valid'] == (device == 'cpu' and binding_change is None)
    if receipt['reference_valid']:
        assert all(not c['docling_json_equal'] for c in receipt['equality_checks'])
    else:
        assert receipt['comparison_refused']
    if receipt['reference_valid']:
        assert all(c['exact_markdown_equal'] for c in receipt['equality_checks'])

def test_corpus_repeat_refreshes_repair_budget_and_honors_pipeline_timeout(tmp_path, exporter, monkeypatch):
    paths=inputs(tmp_path,['a'])
    repair_calls=[]
    repair_module=ModuleType('glossapi.pdf_repair.extract')
    def repair(path, root, **kwargs):
        repair_calls.append(str(path))
        return path, {'font_repair_status':'skipped','font_repair_fonts':0}
    repair_module.repair_for_extract=repair
    monkeypatch.setitem(sys.modules,'glossapi.pdf_repair.extract',repair_module)
    state_data={'processed':set(),'problematic':set()}
    state_module=ModuleType('glossapi.corpus.corpus_state')
    class State:
        def __init__(self,*args,**kwargs): pass
        def load(self): return set(state_data['processed']),set(state_data['problematic'])
        def save(self,ok,bad): state_data.update(processed=ok,problematic=bad)
    state_module._ProcessingStateManager=State
    monkeypatch.setitem(sys.modules,'glossapi.corpus.corpus_state',state_module)
    exporter.converter=SimpleNamespace(initialize_pipeline=lambda fmt:None,
        _get_pipeline=lambda fmt:SimpleNamespace(pipeline_options=SimpleNamespace(
            images_scale=1.25,document_timeout=0.01)))
    exporter.long_pdf_page_threshold=600
    exporter.url_column='url'
    exporter.batch_result_callback=None
    exporter.repair_fonts=True
    exporter.use_pypdfium_backend=False
    exporter.per_page_timeout_s=1
    exporter.processing_timeout=60
    exporter._font_repair_paths={}
    exporter._font_repair_deadlines={}
    budgets=[]
    class Runner:
        def __init__(self,adapter,**kwargs): self.adapter=adapter
        def extract(self,paths,out):
            path = self.adapter.admit(paths[0], out) if hasattr(self.adapter, 'admit') else paths[0]
            budgets.append(self.adapter.timeout(path))
            return [paths[0].name],[]
    monkeypatch.setattr(core,'BatchedExtractor',Runner)
    monkeypatch.setattr(core.DoclingAdapter,'validate',staticmethod(lambda pipeline:None))
    class Corpus:
        extractor=exporter
        def prime_extractor(self): pass
        def extract(self, *, use_gpus='single',phase1_backend='auto',num_threads=2,
                    _prepared=False,skip_existing=True):
            self.prime_extractor()
            self.extractor.extract_path(paths,tmp_path/'out/markdown',skip_existing=skip_existing)
    restore=core.install_corpus_opt_in(Corpus)
    try:
        corpus=Corpus()
        corpus.extract(batched=True,skip_existing=False)
        exporter._font_repair_deadlines[str(paths[0].resolve())]=time.monotonic()-1
        corpus.extract(batched=True,skip_existing=False)
    finally: restore()
    witness={'repair_calls':repair_calls,'admission_budgets':budgets,'pipeline_document_timeout':0.01}
    assert len(repair_calls) == 2
    assert budgets == [0.01, 0.01]


def test_native_cached_backend_corrupt_healthy_alternation(tmp_path):
    """CP-B1 native_review.py witness; real parser/models/export in offline CPU env."""
    pytest.importorskip("docling", reason="native parser witness requires optional Docling and cached models")
    import subprocess
    import textwrap
    repo = Path(__file__).resolve().parents[1]
    script = tmp_path / "native_witness.py"
    script.write_text(textwrap.dedent(r'''
        import functools
        import json
        import shutil
        import sys
        import time
        from pathlib import Path
        from glossapi.scripts import batched_extract_benchmark as benchmark
        core = benchmark.core
        class NativeStall:
            def __init__(self, marker):
                self.marker = marker
            def __reduce__(self):
                self.marker.write_text(str(__import__('os').getpid()))
                time.sleep(60)
                return int, (1,)
        def stalled_native_export(task, payload):
            task()
        def main():
            root = Path(sys.argv[1])
            original = Path.home() / 'Projects/glossapi-ocr-local/reports/artifacts-20260407-ocr-box/pdfs/ABO_768.pdf'
            healthy = root / 'ABO_768.pdf'
            repeat = root / 'ABO_768_repeat.pdf'
            shutil.copyfile(original, healthy)
            shutil.copyfile(original, repeat)
            corrupts = [root / 'corrupt.pdf', root / 'corrupt_repeat.pdf']
            for path in corrupts:
                path.write_bytes(b'%PDF-1.7\nthis is corrupt\n')
            # Exact good-probe -> corrupt-probe -> good-page cache witness.
            core.prepare_docling_page(healthy, 0, max_pages=1)
            try:
                core.prepare_docling_page(corrupts[0], 0, max_pages=1)
            except Exception:
                pass
            assert core.prepare_docling_page(healthy, 1)['page'].page_no == 1
            adapter = core.DoclingAdapter(device='cpu', document_timeout=120)
            # Native page load/parse failure must invalidate a healthy cache too.
            try:
                core.prepare_docling_page(healthy, 99999)
            except Exception:
                pass
            assert core._CPU_CACHE is None
            resident = tuple(id(obj) for obj in (adapter.converter, adapter.pipeline,
                adapter.pipeline.layout_model, adapter.pipeline.table_model))
            metadata = core.prepare_docling_page(healthy, 0, max_pages=1)
            native_result, count = adapter.open(healthy, metadata)
            page = core.PageWork(0, healthy, 1, core.prepare_docling_page(healthy, 1))
            adapter.accept(native_result, page)
            adapter.layout([page], {0: native_result})
            adapter.complete(native_result, page)
            native_result = adapter.finish(native_result, [page.payload])
            core._discard_cpu_backend()
            session = core._ExportSession(core.mp.get_context('spawn'))
            marker = root / 'native-serialization.entered'
            try:
                # Warm this persistent worker with real Docling export state,
                # then stall its serializer with that same real payload graph.
                warm = root / 'warm/markdown'
                warm.mkdir(parents=True)
                session.export(adapter.export_task(native_result, warm), time.monotonic() + 30)
                native_worker = session.process.pid
                began = time.monotonic()
                try:
                    session.export(functools.partial(stalled_native_export,
                        adapter.export_task(native_result, root / 'late/markdown'),
                        NativeStall(marker)), began + 1.5)
                except TimeoutError:
                    pass
                else:
                    raise AssertionError('native serializer did not time out')
                native_stall_seconds = time.monotonic() - began
                assert native_stall_seconds < 2
                assert int(marker.read_text()) == native_worker
                assert not (root / 'late/markdown/ABO_768.md').exists()
                try:
                    __import__('os').kill(native_worker, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise AssertionError('native serializer worker survived')
            finally:
                session.close()
            assert resident == tuple(id(obj) for obj in (adapter.converter, adapter.pipeline,
                adapter.pipeline.layout_model, adapter.pipeline.table_model))
            runner = core.BatchedExtractor(adapter, workers=1, max_active_documents=2,
                                           layout_batch=4, document_timeout=120)
            ok, bad = runner.extract([corrupts[0], healthy, corrupts[1], repeat],
                                    root / 'batched/markdown', max_pages=1)
            assert ok == [healthy.name, repeat.name], (ok, bad, runner.errors)
            assert bad == [p.name for p in corrupts], (ok, bad, runner.errors)
            adapter.per_page_layout_timings()
            for path in [healthy, repeat]:
                result = adapter.converter.convert(path, raises_on_error=False, page_range=(1, 1))
                assert len(result.pages) == 1
                adapter.export(result, root / 'sequential/markdown')
            checks = benchmark.equality(root, [healthy.name, repeat.name])
            assert all(c['passes_gate'] for c in checks), checks
            (root / 'native-receipt.json').write_text(json.dumps({'processed': ok, 'problematic': bad,
                'errors': runner.errors, 'equality': checks,
                'native_stall_seconds': native_stall_seconds, 'resident_identity_preserved': True}, indent=2))
            print('native corrupt/healthy alternation: both healthy artifacts match sequential')
        if __name__ == '__main__':
            main()
    '''))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(repo / "src"))
    completed = subprocess.run([sys.executable, str(script), str(tmp_path)],
                               env=env, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize('mutation', ['markdown', 'json', 'metrics', 'per_page', 'missing_timer', 'missing_timing_field', 'missing_sidecar'])
def test_strict_artifact_gate_rejects_each_mismatch(tmp_path, exporter, mutation):
    paths = inputs(tmp_path, ['a'])
    adapter = FakeAdapter(exporter)
    for mode in ('sequential', 'batched'):
        result, count = adapter.open(paths[0], 2)
        result = adapter.finish(result, [{'text': f'a page {i}'} for i in (1, 2)])
        adapter.export(result, tmp_path / mode / 'markdown')
    assert benchmark.equality(tmp_path, ['a.pdf'])[0]['passes_gate']
    base = tmp_path / 'batched'
    if mutation == 'markdown':
        path = base / 'markdown/a.md'
        path.write_text(path.read_text() + ' ')  # would pass approximate text gate
    elif mutation == 'json':
        path = base / 'json/a.docling.json'
        path.write_text('{"wrong_document": true}')
    elif mutation == 'missing_sidecar':
        (base / 'sidecars/extract/a.json').unlink()
    else:
        suffix = 'per_page.metrics' if mutation == 'per_page' else 'metrics'
        path = base / f'json/metrics/a.{suffix}.json'
        data = json.loads(path.read_text())
        if mutation == 'per_page':
            data['pages'][0]['formula_count'] = 42
        elif mutation == 'metrics':
            data['file'] = 'wrong.pdf'
        else:
            item = {'scope': 'document', 'count': 1, 'total_sec': 1, 'avg_sec': 1,
                    'p50_sec': 1, 'p90_sec': 1, 'times_sec': [1]}
            left = tmp_path / 'sequential/json/metrics/a.metrics.json'
            original = json.loads(left.read_text())
            original['timings']['pipeline_total'] = item
            left.write_text(json.dumps(original))
            if mutation == 'missing_timing_field':
                data['timings']['pipeline_total'] = dict(item)
                del data['timings']['pipeline_total']['total_sec']
        path.write_text(json.dumps(data))
    assert not benchmark.equality(tmp_path, ['a.pdf'])[0]['passes_gate']


def test_metric_ignore_list_keeps_counts_scope_keys_and_page_content(exporter):
    left = {'file': 'a.pdf', 'timings': {'layout': {'scope': 'page', 'count': 2,
            'total_sec': 3, 'avg_sec': 1.5, 'p50_sec': 1, 'p90_sec': 2, 'times_sec': [1, 2]}}}
    from copy import deepcopy
    right = deepcopy(left)
    right['timings']['layout'].update(total_sec=30, avg_sec=15, p50_sec=10, p90_sec=20, times_sec=[10, 20])
    assert benchmark.metric_content(left) == benchmark.metric_content(right)
    right['timings']['layout']['count'] = 1
    assert benchmark.metric_content(left) != benchmark.metric_content(right)
    assert benchmark.baseline_manifest('cpu', {'layout_batch_size': 4, 'images_scale': 1.25}) == benchmark.baseline_manifest('cpu', {'layout_batch_size': 8, 'images_scale': 1.25})


@pytest.mark.parametrize('mutation,passes', [('bbox_small', True), ('bbox_limit', True),
    ('bbox_large', False), ('markdown', False), ('content', False),
    ('page_size', False), ('coordinate_origin', False), ('structure', False)])
def test_markdown_fidelity_bbox_tolerance(tmp_path, exporter, mutation, passes):
    paths = inputs(tmp_path, ['a'])
    adapter = FakeAdapter(exporter)
    payload = {'texts': [{'text': 'native text', 'prov': [{'bbox': {
        'l': 1.0, 't': 2.0, 'r': 3.0, 'b': 4.0, 'coord_origin': 'BOTTOMLEFT'}}]}],
        'pages': {'1': {'size': {'width': 100.0, 'height': 200.0}}}}
    for mode in ('sequential', 'batched'):
        result, _ = adapter.open(paths[0], 1)
        result = adapter.finish(result, [{'text': 'same text'}])
        adapter.export(result, tmp_path / mode / 'markdown')
        path = tmp_path / mode / 'json/a.docling.json'
        path.with_suffix(path.suffix + '.zst').unlink(missing_ok=True)
        path.write_text(json.dumps(payload))
    target = tmp_path / 'batched/json/a.docling.json'
    changed = json.loads(target.read_text())
    bbox = changed['texts'][0]['prov'][0]['bbox']
    if mutation.startswith('bbox_'):
        bbox['l'] += {'bbox_small': .01, 'bbox_limit': .05, 'bbox_large': .051}[mutation]
    elif mutation == 'markdown':
        (tmp_path / 'batched/markdown/a.md').write_text('changed markdown')
    elif mutation == 'content':
        changed['texts'][0]['text'] = 'changed text'
    elif mutation == 'page_size':
        changed['pages']['1']['size']['width'] += .01
    elif mutation == 'coordinate_origin':
        bbox['coord_origin'] = 'TOPLEFT'
    elif mutation == 'structure':
        changed['texts'].append(changed['texts'][0])
    target.write_text(json.dumps(changed))
    check = benchmark.equality(tmp_path, ['a.pdf'])[0]
    assert check['passes_gate'] is passes
    assert check['fidelity'] == 'markdown' and check['bbox_tolerance'] == .05
    assert not benchmark.equality(tmp_path, ['a.pdf'], fidelity='json')[0]['passes_gate']


@pytest.mark.parametrize('explicit', [None, 'json'])
def test_benchmark_cli_fidelity_defaults_and_override(monkeypatch, explicit):
    args = ['--pdfs', 'unused', '--out', 'unused', '--mode', 'batched']
    if explicit:
        args += ['--fidelity', explicit]
    received = []
    monkeypatch.setattr(benchmark, 'run', lambda options: received.append(options) or 0)
    assert benchmark.main(args) == 0
    assert received[0].fidelity == (explicit or 'markdown')
    assert received[0].layout_batch == 8 and received[0].workers == 8
