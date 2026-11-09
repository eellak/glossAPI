"""Offline T3.1 CPU/GPU smoke and fidelity harness, fresh output per mode.

Run with --out ROOT --mode sequential, then the same ROOT --mode batched.
Default fidelity requires exact markdown and JSON with bbox tolerance 0.05;
--fidelity json retains the byte-exact JSON gate. Both check metrics/sidecars.
Only post-initialization conversion/export wall time is called s/page. Startup
is reported separately. CPU results never constitute GPU performance evidence.
"""
from __future__ import annotations

from glossapi.optional_dependencies import optional_dependencies

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from glossapi.scripts.docling_cost_benchmark import GpuTelemetry, fidelity_metrics


def load_core():
    # Loading this isolated module avoids Corpus/Rust import side effects. The
    # stable name is also installed in spawned workers before IPC unpickling.
    name = "batched_extract"
    if name not in sys.modules:
        source = Path(__file__).resolve().parents[1] / "corpus" / "batched_extract.py"
        # spawn inherits sys.path, but does not inherit dynamic sys.modules.
        # Use an actual importable filename for the worker target and payloads.
        sys.path.insert(0, str(source.parent))
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


core = load_core()


def page_count(path):
    import pypdfium2
    from glossapi.pdfium_guard import pdfium_guard
    with pdfium_guard():
        source = pypdfium2.PdfDocument(str(path))
        try:
            return len(source)
        finally:
            source.close()


# The experiment changes only layout dispatch size. All scientific/model,
# accelerator, timeout and asset options are bound by this hash.
OPTIONS_HASH_EXCLUSIONS = ("layout_batch_size",)
METRIC_TIMING_FIELDS = ("total_sec", "avg_sec", "p50_sec", "p90_sec", "times_sec")
PAGE_TIMING_FIELDS = ("ocr_sec", "parse_sec", "layout_sec", "table_sec", "doc_enrich_share_sec")
TIMING_IGNORE_PATHS = [
    "timings.*.total_sec", "timings.*.avg_sec", "timings.*.p50_sec",
    "timings.*.p90_sec", "timings.*.times_sec[*]",
    *["pages[*]." + key for key in PAGE_TIMING_FIELDS],
    "totals.doc_enrich_total_sec",
]


def docling_version():
    from importlib.metadata import version
    return version("docling")


def baseline_manifest(device, options):
    comparable = {key: value for key, value in options.items()
                  if key not in OPTIONS_HASH_EXCLUSIONS}
    encoded = json.dumps(comparable, sort_keys=True, separators=(",", ":"))
    return {"docling_version": docling_version(), "device": device,
            "options_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def read_payload(path):
    if path.exists():
        return path.read_bytes()
    compressed = path.with_suffix(path.suffix + ".zst")
    if compressed.exists():
        from glossapi.ocr.utils.json_io import zstd
        if zstd is not None:
            return zstd.ZstdDecompressor().decompress(compressed.read_bytes())
    return None


def metric_content(data, per_page=False):
    """Ignore only listed stopwatch values; retain keys, scopes, counts and lengths."""
    from copy import deepcopy
    result = deepcopy(data)
    if per_page:
        for page in result.get("pages", []):
            for key in PAGE_TIMING_FIELDS:
                if key in page:
                    page[key] = "<timing>"
        totals = result.get("totals", {})
        if "doc_enrich_total_sec" in totals:
            totals["doc_enrich_total_sec"] = "<timing>"
    else:
        for item in result.get("timings", {}).values():
            for key in METRIC_TIMING_FIELDS:
                if key in item:
                    item[key] = (["<timing>"] * len(item[key]) if key == "times_sec" else "<timing>")
    return result


def json_bbox_equal(left, right, tolerance=.05, *, _bbox=False):
    """Keep JSON structure/content exact; tolerate only bbox l/t/r/b coordinates.

    Absolute tolerance is in Docling's coordinate units. Page sizes, confidence,
    text, list order, counts and coordinate origins remain exact.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        for key in left:
            a, b = left[key], right[key]
            if _bbox and key in {"l", "t", "r", "b"}:
                if (type(a) not in (int, float) or type(b) not in (int, float) or
                        not math.isfinite(a) or not math.isfinite(b) or
                        abs(a - b) > tolerance + 1e-12):
                    return False
            elif not json_bbox_equal(a, b, tolerance, _bbox=key == "bbox"):
                return False
        return True
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_bbox_equal(a, b, tolerance) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def equality(root, names, fidelity="markdown"):
    if fidelity not in ("json", "markdown"):
        raise ValueError("fidelity must be json or markdown")
    checks = []
    for name in names:
        stem = Path(name).stem
        row = {"file": name}
        pairs = {}
        for label, relative in {
            "exact_markdown_equal": f"markdown/{stem}.md",
            "docling_json_equal": f"json/{stem}.docling.json",
            "metrics_equal": f"json/metrics/{stem}.metrics.json",
            "per_page_metrics_equal": f"json/metrics/{stem}.per_page.metrics.json",
            "sidecar_equal": f"sidecars/extract/{stem}.json",
        }.items():
            left = read_payload(root / "sequential" / relative)
            right = read_payload(root / "batched" / relative)
            pairs[label] = (left, right)
            if left is None or right is None:
                row[label] = False
            elif label in ("metrics_equal", "per_page_metrics_equal"):
                try:
                    row[label] = (metric_content(json.loads(left), label == "per_page_metrics_equal") ==
                                  metric_content(json.loads(right), label == "per_page_metrics_equal"))
                except (ValueError, TypeError, AttributeError) as exc:
                    row[label] = False
                    row.setdefault("artifact_errors", {})[label] = f"{type(exc).__name__}: {exc}"
            else:
                # No volatile keys are ignored in decompressed Docling JSON.
                row[label] = left == right
        left, right = pairs["docling_json_equal"]
        try:
            row["docling_json_bbox_equal"] = (left is not None and right is not None and
                json_bbox_equal(json.loads(left), json.loads(right)))
        except (ValueError, TypeError) as exc:
            row["docling_json_bbox_equal"] = False
            row.setdefault("artifact_errors", {})["docling_json_bbox_equal"] = str(exc)
        row["fidelity"] = fidelity
        row["bbox_tolerance"] = .05 if fidelity == "markdown" else 0
        a, b = pairs["exact_markdown_equal"]
        if a is not None and b is not None:
            approximate = fidelity_metrics(b.decode("utf-8"), a.decode("utf-8"))
            row.update({k: v for k, v in approximate.items() if k != "passes_gate"})
            row["approximate_text_gate"] = approximate["passes_gate"]
        row["status"] = "ok" if all(a is not None and b is not None for a, b in pairs.values()) else "missing_output"
        row["passes_gate"] = (all(row[label] for label in pairs if label != "docling_json_equal") and
            row["docling_json_equal" if fidelity == "json" else "docling_json_bbox_equal"])
        checks.append(row)
    return checks


def run(args):
    # Enforce offline behavior even if the caller has a connected machine.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    paths = sorted(Path(args.pdfs).resolve().glob("*.pdf"))
    if not paths:
        raise ValueError("no PDFs in input directory")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("duplicate stems")
    root = Path(args.out).resolve()
    out = root / args.mode
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    markdown = out / "markdown"
    markdown.mkdir()
    def fingerprint(path):
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    input_hashes = {str(p): fingerprint(p) for p in paths}
    expected_counts, inventory_errors = {}, {}
    for path in paths:
        try:
            expected_counts[path.name] = min(page_count(path), args.max_pages)
        except Exception as exc:
            expected_counts[path.name] = 0
            inventory_errors[path.name] = f"{type(exc).__name__}: {exc}"
    expected_pages = sum(expected_counts.values())
    effective_batch = 4 if args.mode == "sequential" else args.layout_batch
    receipt = {"mode": args.mode, "device": args.device, "workers": args.workers,
               "fidelity": getattr(args, "fidelity", "markdown"),
               "bbox_tolerance": .05 if getattr(args, "fidelity", "markdown") == "markdown" else 0,
               "layout_batch": effective_batch, "requested_layout_batch": args.layout_batch,
               "max_pages_per_document": args.max_pages, "expected_pages": expected_pages,
               "input_sha256": input_hashes,
               "inputs": [str(p) for p in paths], "gpu_claim": False,
               "pages": 0, "wall_seconds": None, "seconds_per_page": None,
               "processed_files": [], "problematic_files": [], "errors": {},
               "inventory_errors": inventory_errors, "timing_ignore_paths": TIMING_IGNORE_PATHS,
               "options_hash_exclusions": OPTIONS_HASH_EXCLUSIONS}
    startup = time.perf_counter()
    with GpuTelemetry(out / "gpu_telemetry.jsonl", args.mode) as telemetry:
        telemetry.context = {"phase": "initialization"}
        try:
            adapter = core.DoclingAdapter(device=args.device,
                layout_batch=effective_batch, document_timeout=args.document_timeout,
                artifacts_path=args.artifacts_path)
            receipt["initialization_seconds"] = time.perf_counter() - startup
            receipt["effective_options"] = adapter.pipeline.pipeline_options.model_dump(mode="json")
            receipt["baseline_manifest"] = baseline_manifest(args.device, receipt["effective_options"])
            if args.mode == "sequential" and hasattr(adapter, "per_page_layout_timings"):
                adapter.per_page_layout_timings()
            telemetry.context = {"phase": "conversion_export"}
            began = time.perf_counter()
            if args.mode == "batched":
                runner = core.BatchedExtractor(adapter, workers=args.workers,
                    layout_batch=args.layout_batch, queue_max_size=args.queue_max_size,
                    document_timeout=args.document_timeout)
                ok, bad = runner.extract(paths, markdown, max_pages=args.max_pages)
                receipt.update(processed_files=ok, problematic_files=bad, errors=runner.errors,
                    pages=sum(runner.page_counts.values()), batch_log=runner.batch_log,
                    high_water=runner.high_water,
                    layout_forward_log=getattr(adapter, "forward_log", []),
                    mixed_batches=sum(len({p[0] for p in batch}) > 1 for batch in runner.batch_log))
            else:
                with optional_dependencies("layout"):
                    from docling.datamodel.base_models import ConversionStatus
                for path in paths:
                    try:
                        result = adapter.converter.convert(path, raises_on_error=False,
                            page_range=(1, args.max_pages))
                        if path.name in inventory_errors:
                            raise RuntimeError(inventory_errors[path.name])
                        expected = expected_counts[path.name]
                        if result.status != ConversionStatus.SUCCESS or len(result.pages) != expected:
                            raise RuntimeError(f"incomplete conversion: {result.status}, "
                                               f"{len(result.pages)}/{expected} pages")
                        adapter.export(result, markdown)
                        receipt["processed_files"].append(path.name)
                        receipt["pages"] += len(result.pages)
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        receipt["errors"][path.name] = message
                        receipt["problematic_files"].append(path.name)
                        adapter.failure(path, markdown, message)
            receipt["wall_seconds"] = time.perf_counter() - began
            receipt["seconds_per_page"] = (receipt["wall_seconds"] / receipt["pages"]
                                            if receipt["pages"] else None)
            receipt["status"] = "ok" if not receipt["problematic_files"] else "error"
        except Exception as exc:
            receipt.update(status="error", initialization_seconds=time.perf_counter() - startup,
                           initialization_error=f"{type(exc).__name__}: {exc}")
        receipt["gpu_telemetry_available"] = telemetry.binary is not None
    receipt["total_seconds"] = time.perf_counter() - startup
    baseline_path = root / "sequential" / "summary.json"
    try:
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    except (ValueError, OSError) as exc:
        baseline = {}
        receipt["baseline_error"] = f"{type(exc).__name__}: {exc}"
    binding = receipt.get("baseline_manifest")
    receipt["reference_valid"] = (bool(binding) and baseline.get("baseline_manifest") == binding and
        baseline.get("status") == "ok" and baseline.get("pages") == expected_pages and
        not baseline.get("inventory_errors") and not inventory_errors and
        baseline.get("input_sha256") == input_hashes and
        baseline.get("max_pages_per_document") == args.max_pages)
    receipt["comparison_refused"] = args.mode == "batched" and not receipt["reference_valid"]
    receipt["equality_checks"] = (equality(root, [p.name for p in paths], getattr(args, "fidelity", "markdown"))
                                  if not receipt["comparison_refused"] else [])
    receipt["fidelity_verified"] = (receipt["status"] == "ok" and
        receipt["pages"] == expected_pages and receipt["reference_valid"] and
        bool(receipt["equality_checks"]) and all(c["passes_gate"] for c in receipt["equality_checks"]))
    (out / "summary.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("mode", "status", "pages", "wall_seconds",
        "seconds_per_page", "initialization_seconds", "fidelity_verified")}, indent=2))
    if "initialization_error" in receipt:
        print(receipt["initialization_error"], file=sys.stderr)
    print(f"receipt: {out / 'summary.json'}")
    return 0 if receipt["status"] == "ok" and (args.mode == "sequential" or receipt["fidelity_verified"]) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("sequential", "batched"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fidelity", choices=("json", "markdown"), default="markdown",
                        help="Exact markdown + JSON bbox tolerance 0.05 (default), or byte-exact JSON")
    parser.add_argument("--layout-batch", type=int, default=8)
    parser.add_argument("--queue-max-size", type=int, default=16)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--document-timeout", type=float, default=600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifacts-path")
    args = parser.parse_args(argv)
    if min(args.workers, args.layout_batch, args.queue_max_size, args.max_pages) < 1 or args.document_timeout <= 0:
        parser.error("sizes and timeouts must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
