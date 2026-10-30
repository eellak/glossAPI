"""Offline, deadline-bounded per-page OCR routing reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Iterator

from glossapi.page_quality import page_quality, write_page_quality, PageQuality, PageQualityReport
from glossapi.scripts.pdf_font_repair import _pdf_paths


def _worker(path: str, wordlist: str | None, result_path: str,
            fold_validity: bool = False, rule_profile: str = "v4") -> None:
    """Write one complete result in private staging; sources remain read-only."""
    started = time.perf_counter()
    try:
        record = page_quality(path, wordlist=wordlist, fold_validity=fold_validity,
                              rule_profile=rule_profile).to_dict()
    except Exception as exc:
        record = {"path": path, "filename": Path(path).name,
                  "error": f"{type(exc).__name__}: {exc}"}
    record["worker_elapsed_seconds"] = time.perf_counter() - started
    Path(result_path).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def _stop(process: Any) -> None:
    """Reap each child, escalating termination on expired or crashed jobs."""
    process.join(timeout=0.1)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)
    process.close()


def batch(paths: list[Path], *, workers: int = 1, timeout: float = 120,
          wordlist: str | None = None, fold_validity: bool = False,
          rule_profile: str = "v4") -> Iterator[dict[str, Any]]:
    """Yield exactly one row per input, bounding each child's total lifetime."""
    if workers < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("workers and timeout must be positive and finite")
    if rule_profile not in ("v2", "v3", "v4"):
        raise ValueError("rule_profile must be 'v2', 'v3' or 'v4'")
    context = mp.get_context("spawn")
    pending = iter(enumerate(paths))
    active: list[dict[str, Any]] = []
    exhausted = False
    with tempfile.TemporaryDirectory(prefix="e2-batch-") as stage:
        try:
            while active or not exhausted:
                while len(active) < workers and not exhausted:
                    item = next(pending, None)
                    if item is None:
                        exhausted = True
                        break
                    index, path = item
                    result_path = Path(stage) / f"{index}.json"
                    process = context.Process(target=_worker, args=(str(path), wordlist, str(result_path),
                                                                  fold_validity, rule_profile))
                    started = time.monotonic()
                    process.start()
                    active.append(dict(process=process, path=path, result_path=result_path, started=started))
                for job in active[:]:
                    elapsed = time.monotonic() - job["started"]
                    expired = elapsed >= timeout
                    if not expired and job["process"].is_alive():
                        continue
                    exitcode = job["process"].exitcode
                    _stop(job["process"])
                    active.remove(job)
                    record = {"path": str(job["path"]), "filename": job["path"].name}
                    if expired:
                        record["error"] = f"Timeout after {timeout:g} seconds"
                    elif exitcode != 0:
                        record["error"] = f"Worker exited without result (exitcode={exitcode})"
                    else:
                        try:
                            record = json.loads(job["result_path"].read_text(encoding="utf-8"))
                        except (OSError, ValueError) as exc:
                            record["error"] = f"Invalid worker result: {exc}"
                    record["process_elapsed_seconds"] = elapsed
                    record["fold_validity"] = fold_validity
                    record["rule_profile"] = rule_profile
                    yield record
                if active:
                    time.sleep(0.01)
        finally:
            for job in active:
                _stop(job["process"])


def compare_doc_rule(records: list[dict[str, Any]], manifest: Path) -> dict[str, Any]:
    """Compare every manifest document; failures/missing reports keep all pages.

    Missing documents are disclosed and conservatively retain document routing.
    Integral float CSV page counts are accepted; mismatched successful counts
    fail instead of silently inflating savings. The sample is document-flagged.
    """
    counts: dict[str, int] = {}
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = Path(row["filename"]).name
            value = float(row["pages"])
            if name in counts or not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError(f"Duplicate filename or invalid pages in manifest: {name}")
            counts[name] = int(value)
    matched: set[str] = set()
    full = selected = sum(counts.values())
    errors = 0
    shares: list[float] = []
    for record in records:
        name = record["filename"]
        if name not in counts:
            continue
        if name in matched:
            raise ValueError(f"Duplicate report filename: {name}")
        matched.add(name)
        if record.get("error"):
            errors += 1
        else:
            if record["pages_total"] != counts[name]:
                raise ValueError(f"Manifest/PDF page count mismatch: {name}")
            selected += record["pages_needing_ocr"] - counts[name]
            shares.append(record["share_pages_needing_ocr"])
    bins = {"[0,0.25)": 0, "[0.25,0.5)": 0, "[0.5,0.75)": 0, "[0.75,1]": 0}
    for share in shares:
        label = "[0,0.25)" if share < .25 else "[0.25,0.5)" if share < .5 else "[0.5,0.75)" if share < .75 else "[0.75,1]"
        bins[label] += 1
    return {"documents_compared": len(counts), "documents_with_reports": len(matched),
            "errors_retained_under_doc_rule": errors,
            "manifest_documents_missing": sorted(set(counts) - matched),
            "report_documents_outside_manifest": sorted({r["filename"] for r in records} - set(counts)),
            "pages_under_doc_rule": full, "pages_under_page_rule": selected,
            "pages_saved": full - selected,
            "percent_saved": (full - selected) * 100 / full if full else 0.0,
            "share_pages_needing_ocr": {"count": len(shares), "histogram": bins,
                "mean": statistics.mean(shares) if shares else None,
                "median": statistics.median(shares) if shares else None,
                "min": min(shares) if shares else None, "max": max(shares) if shares else None}}


def main(argv: list[str] | None = None) -> int:
    """Write JSONL and optional sidecars; return nonzero on any file error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--wordlist")
    parser.add_argument("--fold-validity", action="store_true", help="Fold accents, case and final sigma for validity lookup")
    parser.add_argument("--rule-profile", choices=("v2", "v3", "v4"), default="v4")
    parser.add_argument("--output-dir", type=Path, help="Publish json/metrics page-quality sidecars here")
    parser.add_argument("--compare-doc-rule", type=Path)
    args = parser.parse_args(argv)
    paths = _pdf_paths(args.pdfs)
    if args.output_dir and len({p.stem for p in paths}) != len(paths):
        parser.error("Sidecar output requires unique PDF stems")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8") as handle:
        for record in batch(paths, workers=args.workers, timeout=args.timeout, wordlist=args.wordlist,
                            fold_validity=args.fold_validity, rule_profile=args.rule_profile):
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if args.output_dir and not record.get("error"):
                fields = {k: record[k] for k in PageQualityReport.__dataclass_fields__}
                fields["pages"] = [PageQuality(**row) for row in record["pages"]]
                write_page_quality(PageQualityReport(**fields), args.output_dir)
    if args.compare_doc_rule:
        summary = compare_doc_rule(records, args.compare_doc_rule)
        encoded = json.dumps(summary, ensure_ascii=False, indent=2)
        (args.out.parent / "summary.json").write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    return int(any(r.get("error") for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
