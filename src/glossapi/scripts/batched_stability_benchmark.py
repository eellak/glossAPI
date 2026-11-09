"""Offline paired stability instrument; no default/policy changes.

Wall attribution integrates equal shares among active documents, allocating
worker setup/idle/teardown equally. It sums to each worker's conversion wall;
residence time is also retained. Aggregate multi-device wall is the elapsed
worker envelope (per invocation), never the sum of GPU times. Break-even uses
worker seconds, because concurrent recovery costs cannot be subtracted from
aggregate elapsed seconds. Native inference is not preemptible by Docling's
cooperative document timeout. A zero-recovery sample cannot estimate its cost.
"""
from __future__ import annotations

from glossapi.optional_dependencies import optional_dependencies

import argparse
from dataclasses import asdict
import hashlib
import functools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import logging
import signal
import subprocess
import time
import uuid

from glossapi.scripts import batched_extract_benchmark as reference

core = reference.core
STATUSES = ("ok", "failed", "timeout", "incomplete")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def offline():
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def shard_documents(paths, devices):
    if not devices:
        raise ValueError("multi mode requires devices")
    return {device: list(paths[index::len(devices)]) for index, device in enumerate(devices)}


def pin_device(device, fake=False):
    # Run before adapter construction and before importing torch in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ["GLOSSAPI_DOCLING_DEVICE"] = "cpu" if fake else "cuda:0"
    return "cpu" if fake else "cuda:0"


def attributable_wall(documents, wall):
    """Split elapsed active intervals equally; distribute idle overhead equally."""
    if not documents:
        return {}
    boundaries = sorted({0.0, wall, *[min(wall, max(0., d[k])) for d in documents
                        for k in ("started_seconds", "finished_seconds")]})
    allocated = {d["file"]: 0.0 for d in documents}
    for left, right in zip(boundaries, boundaries[1:]):
        active = [d for d in documents if d["started_seconds"] < right and d["finished_seconds"] > left]
        recipients = active or documents
        for doc in recipients:
            allocated[doc["file"]] += (right - left) / len(recipients)
    return allocated


def break_even(saving_per_page, pages_per_document, recovery_count, recovery_seconds):
    """f * recovery_cost/doc == pages/doc * saving/page; raw f may exceed 1."""
    cost = recovery_seconds / recovery_count if recovery_count and recovery_seconds > 0 else None
    rate = max(0., saving_per_page) * pages_per_document / cost if cost else None
    return {"saving_seconds_per_page": saving_per_page,
            "mean_pages_per_document": pages_per_document,
            "measured_seconds_per_recovery": cost, "failures_per_document": rate,
            "failures_per_1000_pages": max(0., saving_per_page) * 1000 / cost if cost else None,
            "formula": "failure_fraction = saving_seconds_per_page * mean_pages_per_document / seconds_per_recovery",
            "reason": "no measured completed recovery cost" if cost is None else
                      "no positive saving" if saving_per_page <= 0 else None}


def recovery_cycles(session):
    counters = session.get("stats", {}).get("counters", {})
    events = session.get("stats", {}).get("events", [])
    cycles, pending = 0, False
    for event in events:
        if event["kind"] == "export_worker_kills" and not pending:
            cycles += 1
            pending = True
        elif event["kind"] in ("export_worker_ready", "export_worker_respawns"):
            pending = False
    if "export_recovery_cycles" in counters:
        cycles = counters["export_recovery_cycles"]
    if not events:
        pending = bool(counters.get("export_recovery_unresolved", 0))
        cycles = counters.get("export_recovery_cycles", counters.get("export_worker_respawns", 0) + int(pending))
    return cycles, int(pending)


def paired_recovery_cost(sessions, common):
    """Cost allocated to exactly the paired worker-time cohort, without add-back.

    Apply the same active/idle equal-share rule to each completed recovery
    interval. A cycle participates only if it allocates positive cost to common
    documents. Pending cycles cannot supply measured cost.
    """
    common = set(common)
    seconds, completed = 0., 0
    for session in sessions:
        documents = session.get("documents", [])
        for event in session.get("stats", {}).get("events", []):
            if event["kind"] != "export_worker_respawns" or event.get("seconds", 0) <= 0:
                continue
            end = event["at_seconds"]
            start = max(0., end - event["seconds"])
            boundaries = sorted({start, end, *[min(end, max(start, d.get(k, 0.))) for d in documents
                for k in ("started_seconds", "finished_seconds")]})
            allocated = 0.
            for left, right in zip(boundaries, boundaries[1:]):
                active = [d for d in documents if d.get("started_seconds", 0) < right and d.get("finished_seconds", 0) > left]
                recipients = active or documents
                if recipients:
                    allocated += (right-left) * sum(d["file"] in common for d in recipients) / len(recipients)
            if allocated > 0:
                seconds += allocated
                completed += 1
    return seconds, completed


def summarize(documents, wall, sessions=(), common=()):
    pages = sum(d["pages"] for d in documents if d["status"] == "ok")
    common_rows = [d for d in documents if d["status"] == "ok" and d["file"] in common]
    common_pages = sum(d["pages"] for d in common_rows)
    paired_wall = sum(d["attributed_wall_seconds"] for d in common_rows)
    worker_wall = sum(s.get("wall_seconds", 0.) for s in sessions)
    # Proportional elapsed attribution keeps the paired metric in the same
    # units as aggregate s/page. Preserve raw worker attribution for recovery
    # modelling and per-document residency; neither is a critical-path claim.
    paired_elapsed = wall * paired_wall / worker_wall if worker_wall else paired_wall
    counters = {}
    events = []
    for session in sessions:
        for key, value in session.get("stats", {}).get("counters", {}).items():
            counters[key] = counters.get(key, 0) + value
        for event in session.get("stats", {}).get("events", []):
            events.append(dict(event, gpu=session["gpu"], session=session["id"]))
    recoveries = sum(recovery_cycles(s)[0] for s in sessions)
    unresolved = sum(recovery_cycles(s)[1] for s in sessions)
    completed = counters.get("export_worker_respawns", 0)
    recovery_seconds = counters.get("recovery_seconds", 0.)
    result = {"documents": len(documents), **{f"documents_{status}": sum(d["status"] == status for d in documents) for status in STATUSES},
        "pages_done": pages, "wall_seconds": wall,
        "seconds_per_page": wall / pages if pages else None,
        "common_documents": len(common_rows), "common_pages": common_pages,
        "common_attributed_wall_seconds": paired_elapsed,
        "common_attributed_worker_seconds": paired_wall,
        "common_seconds_per_page": paired_elapsed / common_pages if common_pages else None,
        "common_worker_seconds_per_page": paired_wall / common_pages if common_pages else None,
        "worker_wall_seconds": worker_wall,
        "initialization_worker_seconds": sum(s.get("initialization_seconds", 0.) for s in sessions),
        "recovery_count": recoveries, "completed_recovery_count": completed,
        "unresolved_recoveries": unresolved,
        "recovery_seconds": recovery_seconds, "counters": counters, "recovery_events": events}
    result["unresolved_recovery_seconds_lower_bound"] = counters.get("unresolved_recovery_seconds_lower_bound", 0.)
    # A recovery with no following export has no ready receipt; never treat its
    # cost as zero. Completed-only mean is explicitly a censored sample.
    result["seconds_per_completed_recovery"] = recovery_seconds / completed if completed else None
    result["recovery_cost_censored"] = result["unresolved_recoveries"] > 0
    result["failure_fraction"] = (result["documents_failed"] + result["documents_timeout"]) / len(documents) if documents else None
    result["unsuccessful_fraction"] = (len(documents) - result["documents_ok"]) / len(documents) if documents else None
    return result


def snapshot_stats(runner):
    snapshot = runner.stats
    return {"counters": dict(snapshot.counters), "events": [asdict(e) for e in snapshot.events]}, [
        dict(v, file=k) for k, v in snapshot.documents.items()]


def normalize_document(row, expected):
    row = dict(row)
    if row.get("status") == "ok" and expected is not None and (expected <= 0 or row.get("pages") != expected):
        row.update(status="incomplete", pages=0, error="exported page count differs from inventory")
    return row


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def read_jsonl(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_bytes().splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            # A killed append can leave a torn final line, never a success.
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise
    return result


def best_effort_recording(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.observer_error()
    return wrapped


class Session:
    def __init__(self, config, gpu, worker, round_id):
        self.config, self.gpu = config, str(gpu)
        self.path = Path(config["mode_out"]) / "sessions" / f"{round_id}-{worker}-{uuid.uuid4().hex}.json"
        self.journal = self.path.with_suffix(".jsonl")
        self.flight = self.path.with_suffix(".flight.json")
        self.data = {"id": self.path.stem, "gpu": self.gpu, "worker": worker, "round": round_id,
                     "created_at_unix": time.time(),
                     "documents": [], "stats": {"counters": {}, "events": []}, "wall_seconds": 0.,
                     "initialization_seconds": 0., "complete": False, "observation_seconds": 0.}
        self.started = None
        self.base = 0.
        self.written = set()
        self.observation_intervals = []
        self.observation_source = None
        self.current_observation_started = None
        self.observation_cursor = 0
        self.conversion_cursor = None
        self.saved_interval_cursor = 0
        self.checkpoint_wall = 0.

    def observer_error(self):
        counters = self.data["stats"]["counters"]
        counters["recorder_errors"] = counters.get("recorder_errors", 0) + 1
        try:
            logging.getLogger(__name__).exception("stability receipt failed")
        except BaseException:
            pass

    def observe(self, callback, *args, **kwargs):
        began = time.monotonic()
        try:
            return callback(*args, **kwargs)
        except BaseException:
            self.observer_error()
        finally:
            finished = time.monotonic()
            self.data["observation_seconds"] += finished - began
            self.observation_intervals.append([began, finished])

    def begin(self):
        self.started = time.monotonic()
        self.data.setdefault("conversion_started_monotonic", self.started)
        self.base = self.data["wall_seconds"]
        if self.conversion_cursor is None:
            self.conversion_cursor = self.started

    def pause(self, excluded_intervals=()):
        if self.started is not None:
            finished = time.monotonic()
            cursor = self.started
            intervals = []
            for left, right in excluded_intervals:
                left, right = max(cursor, left), min(finished, right)
                if left > cursor:
                    intervals.append([cursor, left])
                cursor = max(cursor, right)
            if finished > cursor:
                intervals.append([cursor, finished])
            self.data["wall_seconds"] = self.base + sum(right-left for left,right in intervals)
            self.data["conversion_finished_monotonic"] = finished
            self.data.setdefault("conversion_intervals", []).extend(intervals)
            self.started = None

    @best_effort_recording
    def in_flight(self, names, elapsed=None):
        self.observe(atomic_json, self.flight, {
            "mode": self.config["mode"], "session": self.data["id"], "gpu": self.gpu,
            "documents": list(names), "started_monotonic": time.monotonic(), "pid": os.getpid(),
            "wall_seconds_before": self.data["wall_seconds"] if elapsed is None else self.base + elapsed,
            "input_sha256": self.config["input_sha256"]})

    @best_effort_recording
    def checkpoint(self, name, row, elapsed=None, stats=None):
        if name in self.written:
            return
        row = normalize_document(dict(row, file=name), self.config.get("expected_pages", {}).get(name))
        row.update(gpu=self.gpu, session=self.data["id"], input_sha256=self.config["input_sha256"].get(name))
        wall = self.data["wall_seconds"] if elapsed is None else self.base + elapsed
        now = time.monotonic()
        if self.started is not None:
            # Persist only new conversion segments, including progress-only
            # flight writes before/between terminal checkpoints. The scheduler
            # supplies its entire callback brackets; direct Session users use
            # the IO brackets measured by observe(). Never replay full history.
            source = self.observation_source() if self.observation_source else self.observation_intervals
            excluded = source[self.observation_cursor:]
            self.observation_cursor = len(source)
            if self.observation_source and elapsed is not None:
                excluded = [*excluded, [self.current_observation_started(elapsed), now]]
            cursor = self.conversion_cursor
            intervals = []
            for left, right in excluded:
                left, right = max(cursor, left), min(now, right)
                if left > cursor:
                    intervals.append([cursor, left])
                cursor = max(cursor, right)
            if now > cursor:
                intervals.append([cursor, now])
            self.conversion_cursor = now
            wall = self.checkpoint_wall + sum(right-left for left,right in intervals)
            self.checkpoint_wall = wall
        else:
            intervals = self.data.get("conversion_intervals", [])[self.saved_interval_cursor:]
            self.saved_interval_cursor = len(self.data.get("conversion_intervals", []))
        stats = stats or {"counters": {}, "events": []}
        stats = {"counters": stats["counters"], "events": [asdict(e) for e in stats["events"]]}
        self.observe(append_jsonl, self.journal, {"document": row, "wall_seconds": wall,
            "stats": stats,
            "conversion_intervals": intervals,
            "conversion_started_monotonic": self.data.get("conversion_started_monotonic"),
            "conversion_finished_monotonic": now})
        self.written.add(name)

    @best_effort_recording
    def save(self, complete=False):
        # Full materialization happens only before/after conversion. Incremental
        # terminal checkpoints are one compact append per document.
        self.data["complete"] = complete
        rows = self.data["documents"]
        shares = attributable_wall(rows, self.data["wall_seconds"])
        for row in rows:
            row.update(normalize_document(row, self.config.get("expected_pages", {}).get(row["file"])))
            row.update(attributed_wall_seconds=shares[row["file"]], gpu=self.gpu,
                session=self.data["id"], input_sha256=self.config["input_sha256"].get(row["file"]))
            row["recovery_events"] = [e for e in self.data["stats"]["events"] if e["document"] == row["file"]
                or row["file"] in e.get("affected_documents", [])]
        self.observe(atomic_json, self.path, self.data)


def make_adapter(config, device):
    adapter = core.DoclingAdapter(device=device, layout_batch=4 if config["mode"] == "sequential" else config["layout_batch"],
        document_timeout=config["document_timeout"], artifacts_path=config["artifacts_path"])
    return adapter


def sequential_document(path, adapter, config, output):
    """Production direct convert + ordinary exporter, scoped to one document.

    No observation thread, second PDF handle, parser subprocess, timing model
    wrapper or previous ConversionResult lives across the next convert call.
    """
    with optional_dependencies("layout"):
        from docling.datamodel.base_models import ConversionStatus
    began = time.monotonic()
    row = {"file": path.name, "pages": 0, "prepared_pages": 0, "status": "failed", "error": None}
    result = None
    try:
        expected = config["expected_pages"][path.name]
        if expected == 0:
            raise ValueError(config["inventory_errors"][path.name])
        result = adapter.converter.convert(path, raises_on_error=False, page_range=(1, config["max_pages"]))
        row["prepared_pages"] = len(result.pages)
        if result.status != ConversionStatus.SUCCESS or len(result.pages) != expected:
            row["status"] = "timeout" if "timeout" in str(result.errors).lower() else "incomplete" if len(result.pages) else "failed"
            raise RuntimeError(f"conversion {result.status}: {len(result.pages)}/{expected} pages")
        adapter.export(result, output)
        row.update(status="ok", pages=len(result.pages))
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            row["status"] = "timeout"
        row["error"] = f"{type(exc).__name__}: {exc}"
        # Drop opaque handles before failure metadata opens another PDF handle.
        result = None
        try:
            adapter.failure(path, output, "timeout" if row["status"] == "timeout" else row["error"])
        except Exception as recording:
            row["error"] += f"; failure recording: {recording}"
    finally:
        result = None
    row["wall_seconds"] = time.monotonic() - began
    return row


def sequential_documents(paths, adapter, session, output):
    ok, bad = [], []
    for path in map(Path, paths):
        session.in_flight([path.name])
        session.begin()
        row = sequential_document(path, adapter, session.config, output)
        session.pause()
        row.update(started_seconds=session.base, finished_seconds=session.data["wall_seconds"])
        session.data["documents"].append(row)
        (ok if row["status"] == "ok" else bad).append(path.name)
        session.checkpoint(path.name, row)
        session.in_flight([])
    return ok, bad


def batched_documents(paths, adapter, session, output):
    runner = core.BatchedExtractor(adapter, workers=session.config["workers"],
        layout_batch=session.config["layout_batch"], queue_max_size=session.config["queue_max_size"],
        document_timeout=session.config["document_timeout"])
    session.observation_source = lambda: runner.observation_intervals
    session.current_observation_started = lambda elapsed: runner._stability.started + elapsed
    # Admission inventory is primitive data; there is no observer thread.
    session.begin()
    try:
        runner.extract(paths, output, max_pages=session.config["max_pages"], document_callback=session.checkpoint, progress_callback=session.in_flight)
    finally:
        session.pause(excluded_intervals=runner.observation_intervals)
        stats, documents = snapshot_stats(runner)
        stats["counters"]["recorder_errors"] = stats["counters"].get("recorder_errors", 0) + session.data["stats"]["counters"].get("recorder_errors", 0)
        session.data.update(stats=stats, documents=documents,
            high_water=runner.high_water, mixed_batches=sum(len({p[0] for p in b}) > 1 for b in runner.batch_log))
        session.in_flight([])


def shard_worker(config, paths, gpu, round_id):
    offline()
    device = pin_device(gpu, config["fake_devices"]) if config["use_gpus"] == "multi" else config["device"]
    session = Session(config, gpu, f"gpu{gpu}", round_id)
    session.data["assigned_documents"] = [Path(p).name for p in paths]
    session.save()
    started = time.monotonic()
    try:
        adapter = make_adapter(config, device)
        session.data["initialization_seconds"] = time.monotonic() - started
        session.data["effective_options"] = adapter.pipeline.pipeline_options.model_dump(mode="json")
        session.data["baseline_manifest"] = reference.baseline_manifest(device, session.data["effective_options"])
        output = Path(config["mode_out"]) / "markdown"
        output.mkdir(parents=True, exist_ok=True)
        session.save()  # initialization metadata, before conversion
        if config["mode"] == "batched":
            batched_documents(paths, adapter, session, output)
        else:
            sequential_documents(paths, adapter, session, output)
    except Exception as exc:
        session.data["initialization_seconds"] = time.monotonic() - started if session.started is None and not session.data["documents"] else session.data["initialization_seconds"]
        session.data["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.save(complete=True)


def install_sequential_worker(corpus, config, gpu, worker):
    """Explicit benchmark hook inside the existing multi-GPU queue worker.

    Keep its queue, worker count, GPU binding, crash eviction/respawn and batch
    callbacks. Use the same capped Docling converter/export surface as the
    single-device baseline, bypassing production repair/chunking/state filters.
    No effect unless the private benchmark argument is supplied.
    """
    offline()
    gpu = config.get("device_bindings", {}).get(gpu, gpu)
    device = pin_device(gpu, config["fake_devices"])
    session = Session(config, gpu, worker, config["round"])
    began = time.monotonic()
    adapter = make_adapter(config, device)
    session.data["initialization_seconds"] = time.monotonic() - began
    session.data["effective_options"] = adapter.pipeline.pipeline_options.model_dump(mode="json")
    session.data["baseline_manifest"] = reference.baseline_manifest(device, session.data["effective_options"])
    session.save()
    corpus.extractor = adapter.exporter
    corpus.extractor.release_resources = lambda: None

    def extract(**kwargs):
        session.data.setdefault("assigned_documents", []).extend(Path(p).name for p in kwargs["file_paths"])
        session.save()
        output = Path(config["mode_out"]) / "markdown"
        output.mkdir(parents=True, exist_ok=True)
        ok, bad = sequential_documents(kwargs["file_paths"], adapter, session, output)
        callback = getattr(corpus.extractor, "batch_result_callback", None)
        if callback:
            callback(ok, bad)
        session.save(complete=True)
    corpus.extract = extract


def visible_devices(args):
    if args.fake_devices:
        if args.use_gpus != "multi" or args.device != "cpu":
            raise ValueError("--fake-devices requires --use-gpus multi --device cpu")
        return [str(i) for i in range(args.fake_devices)]
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is not None:
        devices = [d.strip() for d in mask.split(",") if d.strip() and d.strip() != "-1"]
    else:
        import torch
        devices = [str(i) for i in range(torch.cuda.device_count())]
    if not devices:
        raise ValueError("no visible GPUs; CPU sharding requires explicit --fake-devices 2")
    if len(set(devices)) != len(devices):
        raise ValueError("duplicate visible GPUs")
    return devices


def load_sessions(out):
    sessions = []
    for path in sorted((out / "sessions").glob("*.json")):
        if path.name.endswith(".flight.json"):
            continue
        session = json.loads(path.read_text())
        if not session.get("complete"):
            journal = read_jsonl(path.with_suffix(".jsonl"))
            if journal:
                session["documents"] = [record["document"] for record in journal]
                session.update({k: journal[-1][k] for k in ("wall_seconds", "conversion_started_monotonic", "conversion_finished_monotonic")})
                session["conversion_intervals"] = [interval for r in journal for interval in
                    r.get("conversion_intervals", [r["conversion_interval"]] if "conversion_interval" in r else [])]
                session["stats"] = {"counters": journal[-1].get("stats", {}).get("counters", {}),
                    "events": [e for r in journal for e in r.get("stats", {}).get("events", [])]}
                shares = attributable_wall(session["documents"], session["wall_seconds"])
                for row in session["documents"]:
                    row["attributed_wall_seconds"] = shares[row["file"]]
        sessions.append(session)
    return sessions


def collect(out, inventory, expected_pages=None):
    sessions = load_sessions(out)
    if expected_pages is None:
        binding = out / "receipt.json"
        expected_pages = json.loads(binding.read_text()).get("expected_pages", {}) if binding.exists() else {}
    latest = {}
    cumulative = {}
    assignments = {}
    # Later rounds override incomplete records only. Terminal records are never
    # rerun by resume. Saved records carry hashes; the mode receipt binds config.
    for session in sorted(sessions, key=lambda s: s.get("created_at_unix", s.get("conversion_finished_monotonic", 0))):
        for name in session.get("assigned_documents", []):
            assignments[name] = session["gpu"]
        for row in session["documents"]:
            latest[row["file"]] = row
            cumulative[row["file"]] = cumulative.get(row["file"], 0.) + row["attributed_wall_seconds"]
    rows = []
    for path in inventory:
        row = dict(latest.get(path.name, {"file": path.name, "status": "incomplete", "pages": 0,
            "wall_seconds": 0., "attributed_wall_seconds": 0., "error": "no terminal document receipt"}))
        row["attributed_wall_seconds"] = cumulative.get(path.name, 0.)
        if path.name in assignments:
            row.setdefault("gpu", assignments[path.name])
        row = normalize_document(row, expected_pages.get(path.name))
        if row["status"] == "ok" and reference.read_payload(out / "markdown" / f"{path.stem}.md") is None:
            row.update(status="incomplete", pages=0, error="successful receipt has missing Markdown")
        rows.append(row)
    wall = 0.
    for round_id in {s["round"] for s in sessions}:
        timed = [s for s in sessions if s["round"] == round_id and "conversion_started_monotonic" in s]
        intervals = [interval for s in timed for interval in s.get("conversion_intervals", [])]
        if intervals:
            merged = []
            for left, right in sorted(intervals):
                if merged and left <= merged[-1][1]:
                    merged[-1][1] = max(right, merged[-1][1])
                else:
                    merged.append([left, right])
            wall += sum(right - left for left, right in merged)
        elif timed:
            wall += max(s["conversion_finished_monotonic"] for s in timed) - min(s["conversion_started_monotonic"] for s in timed)
    return rows, sessions, wall


def comparison(seq, bat):
    common = sorted({d["file"] for d in seq if d["status"] == "ok"} & {d["file"] for d in bat if d["status"] == "ok"})
    if any(next(d["pages"] for d in seq if d["file"] == name) != next(d["pages"] for d in bat if d["file"] == name) for name in common):
        raise ValueError("paired page counts differ")
    return common


def publish(root, paths, config):
    modes = {}
    loaded = {}
    for mode in ("sequential", "batched"):
        out = root / mode
        if (out / "receipt.json").exists():
            loaded[mode] = collect(out, paths, config.get("expected_pages", {}))
    common = comparison(loaded["sequential"][0], loaded["batched"][0]) if len(loaded) == 2 else []
    fidelity = reference.equality(root, common, config["fidelity"]) if common else []
    def manifest(session):
        # Derive from the persisted options too, so interrupted receipts remain
        # comparable without constructing another converter on resume.
        options = session.get("effective_options")
        if options is None:
            return None
        comparable = {k: v for k, v in options.items() if k not in reference.OPTIONS_HASH_EXCLUSIONS}
        encoded = json.dumps(comparable, sort_keys=True, separators=(",", ":"))
        return {"docling_version": config["docling_version"],
            "options_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
    # Validate the attempts supplying the paired complete outputs. Earlier
    # interrupted attempts retain elapsed/crash history, but supply no output
    # to certify. Unknown options on a successful supplying attempt still fail.
    supplying_sessions = []
    fidelity_excluded_attempts = []
    fidelity_excluded_documents = []
    for mode, (rows, sessions, _) in loaded.items():
        latest_sources = {}
        for session in sorted(sessions, key=lambda s: s.get("created_at_unix", s.get("conversion_finished_monotonic", 0))):
            for row in session["documents"]:
                latest_sources[row["file"]] = session["id"]
        used = {latest_sources[name] for name in common}
        supplying_sessions.extend(s for s in sessions if s["id"] in used)
        for session in sessions:
            if session["documents"] and session["id"] not in used:
                fidelity_excluded_attempts.append(dict(mode=mode, session=session["id"],
                    documents=[r["file"] for r in session["documents"]],
                    reason="interrupted attempt supplies no complete paired output" if session.get("timing_censored")
                        or not session.get("complete", True) else "attempt supplies no current complete paired output"))
        for row in rows:
            if row["file"] not in common:
                fidelity_excluded_documents.append(dict(mode=mode, file=row["file"], status=row["status"],
                    reason=row.get("error") or "no complete successful output in both modes"))
    manifests = [manifest(s) for s in supplying_sessions]
    options_match = bool(manifests) and all(m is not None and m == manifests[0] for m in manifests)
    for mode, (rows, sessions, wall) in loaded.items():
        summary = summarize(rows, wall, sessions, common)
        summary["status"] = "ok" if summary["documents_ok"] == len(paths) else "error"
        summary.update(mode=mode, documents_detail=rows, input_sha256=config["input_sha256"],
            fidelity_checks=fidelity, comparable_options_verified=options_match,
            fidelity_excluded_attempts=fidelity_excluded_attempts,
            fidelity_excluded_documents=fidelity_excluded_documents,
            fidelity_verified=options_match and bool(fidelity) and all(c["passes_gate"] for c in fidelity),
            use_gpus=config["use_gpus"], fake_devices=config["fake_devices"],
            gpu_claim=False,
            gpu_requested=not config["fake_devices"] and (config["use_gpus"] == "multi" or config["device"].lower().startswith("cuda")),
            timing_basis="conversion/export including parsing workers, recovery and teardown; initialization and receipt IO excluded",
            attribution="equal shares across active document intervals, idle/setup/teardown shared equally",
            initialization_errors=[s["error"] for s in sessions if s.get("error")])
        for row in rows:
            row.setdefault("input_sha256", config["input_sha256"].get(row["file"]))
            row["expected_pages"] = config.get("expected_pages", {}).get(row["file"])
            row["attributed_elapsed_seconds"] = (wall * row["attributed_wall_seconds"] / summary["worker_wall_seconds"]
                if summary["worker_wall_seconds"] else 0.)
        summary["total_seconds_including_initialization"] = sum(json.loads(p.read_text())["total_seconds_including_initialization"]
            for p in (root / mode / "rounds").glob("*.json"))
        summary["seconds_per_page_including_initialization"] = (summary["total_seconds_including_initialization"] / summary["pages_done"]
            if summary["pages_done"] else None)
        summary["observation_seconds"] = sum(s.get("observation_seconds", 0.) for s in sessions)
        summary["process_crashes"] = [c for c in read_jsonl(root / "process_crashes.jsonl") if c.get("mode") in (mode, None)]
        summary["process_crash_count"] = len(summary["process_crashes"])
        summary["per_gpu"] = {}
        for gpu in config["devices"]:
            subset = [s for s in sessions if s["gpu"] == gpu]
            gpu_common = common
            if len(loaded) == 2:
                gpu_common = sorted(set(common) & {d["file"] for d in loaded["sequential"][0] if d.get("gpu") == gpu}
                    & {d["file"] for d in loaded["batched"][0] if d.get("gpu") == gpu})
            summary["per_gpu"][gpu] = summarize([d for d in rows if d.get("gpu") == gpu],
                sum(s["wall_seconds"] for s in subset), subset, gpu_common)
            summary["per_gpu"][gpu]["process_crashes"] = [c for c in summary["process_crashes"]
                if c.get("gpu") == gpu or any(f.get("gpu") == gpu for f in c.get("in_flight", []))]
            summary["per_gpu"][gpu]["process_crash_count"] = len(summary["per_gpu"][gpu]["process_crashes"])
        summary["timing_censored"] = any(s.get("timing_censored", False) for s in sessions)
        modes[mode] = summary
    if len(modes) == 2:
        a, b = modes["sequential"], modes["batched"]
        paired_a, paired_b = a["common_worker_seconds_per_page"], b["common_worker_seconds_per_page"]
        net = paired_a - paired_b if paired_a is not None and paired_b is not None else None
        # Incremental threshold uses observed net saving, with recovery cost
        # allocated by the same worker-time attribution to the paired cohort.
        cohort_cost, cohort_completed = paired_recovery_cost(loaded["batched"][1], common)
        line = break_even(net or 0., b["common_pages"] / b["common_documents"] if b["common_documents"] else 0.,
                          cohort_completed, cohort_cost)
        line.update(net_measured_saving_seconds_per_page=net, basis="incremental threshold from observed net paired worker saving; no recovery add-back",
                    cohort=common, cohort_recovery_seconds=cohort_cost, cohort_completed_recoveries=cohort_completed,
                    cost_censored=b["recovery_cost_censored"])
        line["net_measured_elapsed_saving_seconds_per_page"] = (a["common_seconds_per_page"] - b["common_seconds_per_page"]
            if a["common_seconds_per_page"] is not None and b["common_seconds_per_page"] is not None else None)
        if net is None:
            line.update(failures_per_document=None, failures_per_1000_pages=None, reason="no paired successful documents")
        b["break_even"] = line
        b["failure_fraction_difference"] = b["failure_fraction"] - a["failure_fraction"]
        b["unsuccessful_fraction_difference"] = b["unsuccessful_fraction"] - a["unsuccessful_fraction"]
        b["overall_speedup"] = a["seconds_per_page"] / b["seconds_per_page"] if a["seconds_per_page"] and b["seconds_per_page"] else None
        for gpu, gs in b["per_gpu"].items():
            other = a["per_gpu"].get(gpu, {})
            ap, bp = other.get("common_worker_seconds_per_page"), gs["common_worker_seconds_per_page"]
            saving = ap - bp if ap is not None and bp is not None else 0.
            gpu_common = sorted(set(common) & {d["file"] for d in loaded["sequential"][0] if d.get("gpu") == gpu}
                & {d["file"] for d in loaded["batched"][0] if d.get("gpu") == gpu})
            cost, completed = paired_recovery_cost([s for s in loaded["batched"][1] if s["gpu"] == gpu], gpu_common)
            gs["break_even"] = break_even(saving,
                gs["common_pages"] / gs["common_documents"] if gs["common_documents"] else 0.,
                completed, cost)
            gs["break_even"].update(cohort=gpu_common, cohort_recovery_seconds=cost,
                cohort_completed_recoveries=completed, basis=line["basis"])
            gs["break_even"]["cost_censored"] = gs["recovery_cost_censored"]
            if ap is None or bp is None:
                gs["break_even"].update(failures_per_document=None, failures_per_1000_pages=None,
                    reason="no paired successful documents on this GPU")
    for mode, summary in modes.items():
        atomic_json(root / mode / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("mode", "documents_ok", "documents_failed", "documents_timeout", "documents_incomplete",
            "pages_done", "wall_seconds", "seconds_per_page", "common_seconds_per_page", "recovery_count", "recovery_seconds", "fidelity_verified")}, sort_keys=True))
        if "break_even" in summary:
            print("break_even " + json.dumps(summary["break_even"], sort_keys=True))
    return modes


def run(args):
    offline()
    paths = sorted(Path(args.pdfs).resolve().glob("*.pdf"), key=lambda p: p.name)
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("input requires PDFs with unique stems")
    root = Path(args.out).resolve()
    devices = visible_devices(args) if args.use_gpus == "multi" else ["single"]
    hashes, counts, errors = {}, {}, {}
    for path in paths:
        with path.open("rb") as source:
            hashes[path.name] = hashlib.file_digest(source, "sha256").hexdigest()
        try:
            counts[path.name] = min(reference.page_count(path), args.max_pages)
        except Exception as exc:
            counts[path.name] = 0
            errors[path.name] = f"{type(exc).__name__}: {exc}"
    config = dict(workers=args.workers, layout_batch=args.layout_batch, queue_max_size=args.queue_max_size,
        max_pages=args.max_pages, document_timeout=args.document_timeout, artifacts_path=args.artifacts_path,
        device=args.device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        use_gpus=args.use_gpus, devices=devices, fake_devices=args.fake_devices,
        workers_per_device=args.workers_per_device, fidelity=args.fidelity, measurement_schema=2,
        input_sha256=hashes, expected_pages=counts, inventory_errors=errors,
        inputs=[str(p) for p in paths], docling_version=reference.docling_version())
    # All modes must share the same binding; configuration drift invalidates
    # paired fidelity and timing, even if the caller runs one arm at a time.
    binding = root / "inputs.json"
    if binding.exists() and json.loads(binding.read_text()) != config:
        raise ValueError("input/configuration mismatch: use a fresh output root")
    atomic_json(binding, config)
    for mode in ("sequential", "batched") if args.mode == "both" else (args.mode,):
        out = root / mode
        receipt = out / "receipt.json"
        if receipt.exists() and not args.resume:
            raise ValueError(f"mode already exists; use --resume: {mode}")
        atomic_json(receipt, dict(config, mode=mode))
        previous, _, _ = collect(out, paths, config.get("expected_pages", {}))
        finished = {r["file"] for r in previous if r["status"] != "incomplete"}
        pending = [p for p in paths if p.name not in finished]
        if not pending:
            continue
        round_id = uuid.uuid4().hex
        invocation = dict(config, mode=mode, mode_out=str(out), round=round_id)
        began = time.monotonic()
        round_path = out / "rounds" / f"{round_id}.json"
        atomic_json(round_path, {"started_monotonic": began,
            "total_seconds_including_initialization": 0., "complete": False})
        try:
            if args.use_gpus == "multi" and mode == "sequential" and not args.fake_devices:
                from glossapi import Corpus
                # Avoid copying 242 source PDFs: direct paths are authoritative and
                # a downloads marker prevents production input-directory copying.
                marker = out / "downloads" / ".stability-direct-paths"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                empty = out / "empty_input"
                empty.mkdir(exist_ok=True)
                corpus = Corpus(input_dir=empty, output_dir=out)
                native_devices = [int(d) for d in devices] if all(d.isdigit() for d in devices) else list(range(len(devices)))
                if not all(d.isdigit() for d in devices):
                    invocation["device_bindings"] = dict(zip(map(str, native_devices), devices))
                corpus.extract(file_paths=pending, input_format="pdf", use_gpus="multi", devices=native_devices,
                    workers_per_device=args.workers_per_device, phase1_backend="docling", batched=False,
                    repair_fonts=False, skip_existing=False, _stability_benchmark=invocation)
            elif args.use_gpus == "multi":
                # Shard ORIGINAL indices, so resume never moves documents to GPUs.
                shards = shard_documents(paths, devices)
                selected = set(pending)
                processes = []
                context = mp.get_context("spawn")
                for gpu, shard in shards.items():
                    selected_shard = [p for p in shard if p in selected]
                    if selected_shard:
                        process = context.Process(target=shard_worker, args=(invocation, selected_shard, gpu, round_id))
                        process.start()
                        processes.append(process)
                try:
                    crashed = []
                    for process in processes:
                        process.join()
                        if process.exitcode is not None and process.exitcode < 0:
                            crashed.append((-process.exitcode, process.pid))
                    if crashed:
                        for number, pid in crashed:
                            record_process_crash(root, number, pid=pid)
                        raise NativeProcessCrash(crashed[0][0])
                finally:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                            process.join()
                        process.close()
            else:
                shard_worker(invocation, pending, "single", round_id)
        except NativeProcessCrash as exc:
            atomic_json(round_path, {"started_monotonic": began, "process_signal": exc.number,
                "total_seconds_including_initialization": time.monotonic() - began, "complete": True})
            publish(root, paths, config)
            return 128 + exc.number
        except Exception as exc:
            # Persist controller/bootstrap failures too; keep the other arm
            # runnable and let resume retry documents with no terminal receipt.
            failed_session = Session(invocation, "controller", "controller", round_id)
            failed_session.data["error"] = f"{type(exc).__name__}: {exc}"
            failed_session.save(complete=True)
        atomic_json(round_path, {"started_monotonic": began,
            "total_seconds_including_initialization": time.monotonic() - began, "complete": True})
    modes = publish(root, paths, config)
    return 0 if all(s["documents_ok"] == len(paths) for s in modes.values()) and (
        len(modes) < 2 or modes["batched"]["fidelity_verified"]) else 1


class NativeProcessCrash(Exception):
    def __init__(self, number):
        super().__init__(f"worker exited by signal {number}")
        self.number = number


def record_process_crash(root, number, pid=None, *, mode=None, gpu=None, worker=None, round_id=None):
    root = Path(root)
    flights = []
    flight_paths = []
    for path in sorted(root.glob("*/sessions/*.flight.json")):
        flight = json.loads(path.read_text())
        if pid is not None and flight.get("pid") != pid:
            # Worker/round is also an identity boundary (and supports fake
            # process contexts without pretending the controller PID died).
            receipt_path = path.with_name(path.name.replace(".flight.json", ".json"))
            receipt = json.loads(receipt_path.read_text())
            if worker is None or receipt.get("worker") != worker or receipt.get("round") != round_id:
                continue
        flight_paths.append(path)
        if flight["documents"]:
            flights.append(flight)
    now = time.monotonic()
    affected = []
    for flight in flights:
        sessions_dir = root / flight["mode"] / "sessions"
        original = json.loads((sessions_dir / (flight["session"] + ".json")).read_text())
        journal = read_jsonl(sessions_dir / (flight["session"] + ".jsonl"))
        completed_names = {r["document"]["file"] for r in journal if r["document"]["status"] != "incomplete"}
        names = [name for name in flight["documents"] if name not in completed_names]
        if not names:
            continue
        affected.append(dict(mode=flight["mode"], documents=names, gpu=flight["gpu"], session=flight["session"]))
        # Retain interrupted work as an incomplete attempt. This censored
        # interval ends when the supervisor observes the signal, so is an
        # elapsed upper bound (includes signal-detection latency).
        total = flight.get("wall_seconds_before", 0.) + max(0., now - flight["started_monotonic"])
        prior = journal[-1]["wall_seconds"] if journal else 0.
        wall = max(0., total - prior)
        crash_id = "crash-" + uuid.uuid4().hex
        rows = [dict(file=name, status="incomplete", pages=0, prepared_pages=0,
            error=f"process signal {number}", started_seconds=0., finished_seconds=wall,
            wall_seconds=wall, attributed_wall_seconds=wall / len(names), gpu=flight["gpu"],
            session=crash_id, input_sha256=flight["input_sha256"].get(name)) for name in names]
        atomic_json(sessions_dir / (crash_id + ".json"), {
            "id": crash_id, "gpu": flight["gpu"], "round": original["round"], "created_at_unix": time.time(),
            "documents": rows, "stats": {"counters": {}, "events": []}, "wall_seconds": wall,
            "timing_censored": True, "timing_basis": "signal-detection elapsed upper bound after last checkpoint",
            "conversion_started_monotonic": now-wall,
            "conversion_finished_monotonic": now, "conversion_intervals": [[now-wall, now]],
            "effective_options": original.get("effective_options"),
            "source_session": flight["session"], "complete": True})
    modes = {f["mode"] for f in affected}
    names = [name for f in affected for name in f["documents"]]
    append_jsonl(root / "process_crashes.jsonl", {"signal": number,
        "signal_name": signal.Signals(number).name, "mode": next(iter(modes)) if len(modes) == 1 else mode,
        "document_in_flight": names[0] if len(names) == 1 else None,
        "documents_in_flight": names, "in_flight": affected, "pid": pid,
        "gpu": gpu, "worker": worker, "at_unix": time.time()})
    for path in flight_paths:
        path.unlink()
    for path in root.glob("*/rounds/*.json"):
        receipt = json.loads(path.read_text())
        if receipt.get("complete") is False:
            receipt.update(complete=True, process_signal=number,
                total_seconds_including_initialization=max(0., now - receipt["started_monotonic"]))
            atomic_json(path, receipt)


def launch_process(command):
    # Fresh exec with its own process group. A killed owner must not leave an
    # export/parser child publishing artifacts during the resumed invocation.
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait()
        if code < 0:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return code
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def crash_loop(args, argv, launch=None):
    """N signal restarts, each a fresh exec interpreter with --resume.

    Popen's negative status is the signal contract. Ordinary exit 1 is a
    document/controller failure, and must never trigger a restart.
    """
    launch = launch or launch_process
    child_args = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item == "--crash-loop":
            skip = True
        elif not item.startswith("--crash-loop="):
            child_args.append(item)
    command = [sys.executable, "-m", "glossapi.scripts.batched_stability_benchmark", *child_args]
    root = Path(args.out).resolve()
    attempts = 0
    while True:
        code = launch(command)
        if code >= 0 and not (code > 128 and code - 128 in signal.valid_signals()):
            return code
        number = -code if code < 0 else code - 128
        if code < 0:  # spawned shard failures are already recorded by run()
            record_process_crash(root, number)
        if attempts >= args.crash_loop:
            binding = root / "inputs.json"
            if binding.exists():
                config = json.loads(binding.read_text())
                publish(root, [Path(p) for p in config["inputs"]], config)
            return 128 + number
        attempts += 1
        if "--resume" not in command:
            command.append("--resume")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("both", "sequential", "batched"), default="both")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--crash-loop", type=int, default=0, help="maximum signal-exit restarts with --resume")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--layout-batch", type=int, default=8)
    parser.add_argument("--queue-max-size", type=int, default=16)
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--document-timeout", type=float, default=600.)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-gpus", choices=("single", "multi"), default="single")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--fake-devices", type=int, default=0, help="CPU-only multi-path smoke; never GPU evidence")
    parser.add_argument("--artifacts-path")
    parser.add_argument("--fidelity", choices=("markdown", "json"), default="markdown")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if min(args.workers, args.layout_batch, args.queue_max_size, args.max_pages, args.workers_per_device) < 1 or args.document_timeout <= 0 or args.fake_devices < 0 or args.crash_loop < 0:
        parser.error("sizes/timeouts must be positive")
    try:
        return crash_loop(args, argv) if args.crash_loop else run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
