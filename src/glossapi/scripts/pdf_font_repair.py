"""Bounded multiprocessing CLI for PDF font analysis and incremental repair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Iterator

from glossapi.pdf_repair import analyze_pdf, extract_text_stats, repair_pdf
from glossapi.pdf_repair.protection import protect_artifact, protect_sources
from glossapi.pdf_repair.diagnostics import capture_library_warnings


def _pdf_paths(inputs: list[str]) -> list[Path]:
    """Resolve directories, PDF arguments and newline lists without duplicates."""
    paths: set[Path] = set()
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.update(item.resolve() for item in path.iterdir() if item.suffix.lower() == ".pdf")
        elif path.suffix.lower() == ".pdf":
            paths.add(path.resolve())
        elif path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    entry = Path(line.strip()).expanduser()
                    paths.add((entry if entry.is_absolute() else path.parent / entry).resolve())
        else:
            raise ValueError(f"Not a PDF, directory or list: {value}")
    return sorted(paths)


def _worker(result_path: str, path: str, mode: str, destination: str | None,
            wordlist: str | None, sources: tuple[str, ...] = ()) -> None:
    """Atomically publish one JSON result; leave PDF publication to the parent."""
    started = time.monotonic()
    record: dict[str, Any] = {"path": path, "filename": Path(path).name}
    with capture_library_warnings() as counts:
        try:
            report = analyze_pdf(path)
            record.update(report.to_dict())
            record["before_category"] = report.category
            if mode == "repair" and report.error is None:
                record["before"] = extract_text_stats(path, 20, wordlist_path=wordlist)
                if any(font.repair_possible for font in report.fonts):
                    result = repair_pdf(path, destination, protected_sources=sources)
                    record["repair"] = result.to_dict()
                    if result.error:
                        record["error"] = result.error
                    elif result.repaired_fonts:
                        record["after"] = extract_text_stats(destination, 20, wordlist_path=wordlist)
                        after = analyze_pdf(destination)
                        record["after_category"] = after.category
                        if after.error:
                            record["error"] = after.error
                    else:
                        record["skip_reason"] = "No font could be updated in isolation"
                else:
                    record["skip_reason"] = "No repairable fonts"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record.setdefault("category", "unsupported")
        finally:
            record["warning_counts"] = dict(counts)
            record["warning_count"] = sum(counts.values())
            record["elapsed_seconds"] = time.monotonic() - started
            _write_record(Path(result_path), record, sources)


def _write_record(path: Path, record: dict[str, Any],
                  sources: tuple[str, ...] = ()) -> None:
    """Expose only complete JSON files, even if a worker is killed mid-write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    protect_sources(temporary, sources)
    protect_sources(path, sources)
    payload = {**record, "_completed_at": time.monotonic()}
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    protect_sources(path, sources)
    os.replace(temporary, path)


def _stop_process(process: Any) -> None:
    """Reap a child within bounded waits, escalating terminate to kill."""
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.2)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.2)
    else:
        process.join(timeout=0)


def _batch(paths: list[Path], mode: str, out_dir: Path | None, workers: int,
           timeout: float, wordlist: str | None) -> Iterator[dict[str, Any]]:
    """Wait on child exit, never a pipe frame, within each per-file deadline."""
    sources = tuple(str(path) for path in paths)
    stage_root = out_dir / ".e1-staging" if out_dir else None
    if out_dir:
        for path in paths:
            protect_sources(out_dir / path.name, sources)
        protect_sources(stage_root, sources)
        stage_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    pending = iter(paths)
    active: list[dict[str, Any]] = []
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < workers and not exhausted:
                path = next(pending, None)
                if path is None:
                    exhausted = True
                    break
                stage = tempfile.TemporaryDirectory(prefix=".e1-", dir=stage_root)
                destination = str(Path(stage.name) / path.name) if out_dir else None
                result_path = Path(stage.name) / "result.json"
                protect_sources(result_path, sources)
                if destination:
                    protect_sources(destination, sources)
                process = context.Process(target=_worker, args=(str(result_path), str(path), mode, destination, wordlist, sources))
                job = {"path": path, "process": process, "result_path": result_path,
                       "started": time.monotonic(), "stage": stage, "destination": destination}
                try:
                    process.start()
                except BaseException:
                    stage.cleanup()
                    raise
                active.append(job)
            if not active:
                continue
            # Jobs are launched in deadline order. Waiting on the earliest one
            # also lets the other children run without blocking on IPC payloads.
            job = active[0]
            process = job["process"]
            remaining = max(0.0, timeout - (time.monotonic() - job["started"]))
            process.join(remaining)
            expired = process.is_alive()
            elapsed = time.monotonic() - job["started"]
            record = None
            if not expired:
                try:
                    record = json.loads(job["result_path"].read_text(encoding="utf-8"))
                    completed_at = record.pop("_completed_at", job["started"])
                    expired = completed_at > job["started"] + timeout
                    if process.exitcode != 0:
                        raise ValueError("Nonzero worker exit status")
                except (OSError, ValueError) as exc:
                    record = {"path": str(job["path"]), "filename": job["path"].name,
                              "category": "unsupported", "elapsed_seconds": elapsed,
                              "error": f"Worker exited without valid result (exitcode={process.exitcode}): {exc}"}
            if expired:
                _stop_process(process)
                record = {"path": str(job["path"]), "filename": job["path"].name,
                          "category": "timeout", "elapsed_seconds": elapsed,
                          "error": f"Timeout after {timeout:g} seconds"}
            # A killed/crashed worker cannot publish its final warning counts.
            record.setdefault("warning_counts", None)
            record.setdefault("warning_count", None)
            if out_dir:
                # Keep reservations in the report so summarize can protect them
                # even when a worker skipped, failed, or timed out.
                record["planned_output"] = str((out_dir / job["path"].name).resolve())
                record["staging_dir"] = str(stage_root.resolve())
            if record.get("after") is not None and not record.get("error"):
                target = out_dir / job["path"].name
                protect_sources(target, sources)
                os.replace(job["destination"], target)
                record["output"] = str(target)
                record["repair"]["dst"] = str(target)
                record["after"]["path"] = str(target)
            job["stage"].cleanup()
            process.close()
            active.pop(0)
            yield record
    finally:
        for job in active:
            _stop_process(job["process"])
            job["stage"].cleanup()
            job["process"].close()


def _distribution(values: list[float]) -> dict[str, Any]:
    """Describe rates with explicit denominators and fixed histogram boundaries."""
    bins = {"[0,0.3)": 0, "[0.3,0.5)": 0, "[0.5,0.85)": 0, "[0.85,1]": 0}
    for value in values:
        label = "[0,0.3)" if value < .3 else "[0.3,0.5)" if value < .5 else "[0.5,0.85)" if value < .85 else "[0.85,1]"
        bins[label] += 1
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "histogram": bins}


def _rate(record: dict[str, Any], stage: str) -> float | None:
    """Read a nullable total rate; unavailable dictionaries never mean zero."""
    return record.get(stage, {}).get("totals", {}).get("dictionary_valid_rate")


def _page_count(value: Any, filename: str) -> int:
    """Accept manifest floats such as 169.0, rejecting negative/nonfinite pages."""
    count = float(value)
    if not math.isfinite(count) or count < 0:
        raise ValueError(f"Invalid page count for {filename}: {value}")
    return int(count)


def summarize(report_path: Path, manifest: Path | None = None) -> dict[str, Any]:
    """Use manifest pages when supplied, otherwise the report's own page counts."""
    with report_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    pages: dict[str, int] = {}
    if manifest is not None:
        with manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                filename = Path(row["filename"]).name
                if filename in pages:
                    raise ValueError(f"Duplicate manifest filename: {filename}")
                pages[filename] = _page_count(row["pages"], filename)
    else:
        pages = {record["filename"]: _page_count(record["pages"], record["filename"])
                 for record in records if record.get("pages") is not None}
    names = [record["filename"] for record in records]
    if len(set(names)) != len(names):
        raise ValueError("Report contains duplicate filenames; summarize a single batch")
    categories = {category: {"docs": 0, "pages": 0} for category in
                  ("no_text_layer", "repairable", "consistent", "not_embedded", "unsupported", "mixed", "timeout", "error")}
    for record in records:
        category = "error" if record.get("error") else record["category"]
        bucket = categories.setdefault(category, {"docs": 0, "pages": 0})
        bucket["docs"] += 1
        bucket["pages"] += pages.get(record["filename"], 0)
    repaired = [record for record in records if not record.get("error") and record.get("repair", {}).get("repaired_fonts", 0) and "after" in record]
    passing = [record for record in repaired if _rate(record, "after") is not None and _rate(record, "after") >= .85]
    missing = sorted(set(names) - pages.keys())
    if any(record["filename"] not in pages for record in passing):
        raise ValueError("Passing repaired documents are missing page counts")
    worst = sorted(repaired, key=lambda record: (_rate(record, "after") is not None, _rate(record, "after") or 0))[:10]
    return {"report": str(report_path), "categories": categories,
            "page_count_source": "manifest" if manifest is not None else "report",
            "missing_manifest_filenames": missing,
            "dictionary_valid_rate": {stage: _distribution([rate for record in records if not record.get("error") and (rate := _rate(record, stage)) is not None]) for stage in ("before", "after")},
            "paired_dictionary_valid_rate": {stage: _distribution([rate for record in repaired if (rate := _rate(record, stage)) is not None]) for stage in ("before", "after")},
            "docs_after_rate_ge_085": len(passing),
            "passing_filenames": [record["filename"] for record in passing],
            "projected_ocr_pages_removed": sum(pages[record["filename"]] for record in passing),
            "worst_repaired_docs": [{"filename": record["filename"], "pages": pages.get(record["filename"]),
                                      "before_rate": _rate(record, "before"), "after_rate": _rate(record, "after")} for record in worst]}


def main(argv: list[str] | None = None) -> int:
    """Expose classify/repair/summarize with one durable JSONL result per input."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("classify", "repair"):
        command = commands.add_parser(name)
        command.add_argument("--pdfs", nargs="+", required=True)
        command.add_argument("--workers", type=int, default=1)
        command.add_argument("--timeout", type=float, default=120)
        command.add_argument("--wordlist", help="Greek dictionary path (or GLOSSAPI_GREEK_WORDLIST)")
        if name == "classify":
            command.add_argument("--out", type=Path, required=True)
        else:
            command.add_argument("--out-dir", type=Path, required=True)
            command.add_argument("--report", type=Path, required=True)
    command = commands.add_parser("summarize")
    command.add_argument("--report", type=Path, required=True)
    command.add_argument("--manifest", type=Path, help="Optional CSV page counts; defaults to report pages")
    args = parser.parse_args(argv)
    if args.command == "summarize":
        result = summarize(args.report, args.manifest)
        output = args.report.parent / "summary.json"
        with args.report.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        sources = [record.get("path") or record.get("repair", {}).get("src")
                   for record in records]
        sources = [path for path in sources if path]
        planned = [Path(path) for record in records for path in
                   (record.get("planned_output"), record.get("output"),
                    record.get("repair", {}).get("dst")) if path]
        staging = [Path(record["staging_dir"]) for record in records if record.get("staging_dir")]
        try:
            if output.resolve() == args.report.resolve():
                raise ValueError("Summary would overwrite its input report")
            protect_artifact(output, sources, planned, staging)
        except ValueError as exc:
            parser.error(str(exc))
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.workers < 1 or not 0 < args.timeout < float("inf"):
        parser.error("workers and finite timeout must be positive")
    paths = _pdf_paths(args.pdfs)
    if not paths:
        parser.error("No PDFs selected")
    output = args.out if args.command == "classify" else args.report
    out_dir = args.out_dir if args.command == "repair" else None
    try:
        protect_artifact(output, paths,
                         (out_dir / path.name for path in paths) if out_dir else (),
                         (out_dir / ".e1-staging",) if out_dir else ())
        if out_dir:
            if len({path.name for path in paths}) != len(paths):
                parser.error("Duplicate PDF basenames would overwrite each other")
            if any(out_dir.resolve() == path.parent.resolve() for path in paths):
                parser.error("Output directory must differ from every input PDF directory")
            for path in paths:
                protect_sources(out_dir / path.name, paths)
            out_dir.mkdir(parents=True, exist_ok=True)
    except ValueError as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in _batch(paths, args.command, out_dir, args.workers, args.timeout, args.wordlist):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            errors += bool(record.get("error"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
