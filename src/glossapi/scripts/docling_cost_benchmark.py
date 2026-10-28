"""Isolated E3 layout benchmarks; importing this module never imports Docling/Corpus."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from difflib import SequenceMatcher
from enum import Enum
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Any


VARIANTS = {
    "baseline": {},
    "table_fast": {"table_structure_options.mode": "FAST"},
    "scale_1.0": {"images_scale": 1.0},
    "layout_batch_16": {"layout_batch_size": 16},
    "layout_batch_32": {"layout_batch_size": 32},
    "no_tables": {"do_table_structure": False},
    "no_tables_scale_1.0": {"do_table_structure": False, "images_scale": 1.0},
}
THRESHOLDS = {"text_similarity_min": 0.98, "count_relative_error_max": 0.02,
              "seconds_per_page_ratio_max": 0.75}


class UnsupportedVariant(ValueError):
    pass


def apply_variant(options: Any, name: str) -> None:
    """Declarative, in-place overrides. Baseline is an exact no-op."""
    for field, value in VARIANTS[name].items():
        obj = options
        parts = field.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
        attr = parts[-1]
        if obj is None or not hasattr(obj, attr):
            raise UnsupportedVariant(f"{name}: installed options do not expose {field}")
        if field == "table_structure_options.mode":
            from docling.datamodel.pipeline_options import TableFormerMode
            value = getattr(TableFormerMode, value)
        try:
            setattr(obj, attr, value)
        except (AttributeError, ValueError, TypeError) as exc:
            raise UnsupportedVariant(f"{name}: cannot set {field}: {exc}") from exc


def markdown_metrics(text: str) -> dict[str, int]:
    tables = 0
    in_table = False
    for line in text.splitlines():
        row = line.strip().startswith("|") and line.count("|") >= 2
        if row and not in_table:
            tables += 1
        in_table = row
    return {"heading_count": len(re.findall(r"(?m)^#+ ", text)),
            "table_count": tables, "total_chars": len(text)}


def fidelity_metrics(text: str, baseline: str) -> dict[str, Any]:
    current, reference = markdown_metrics(text), markdown_metrics(baseline)
    normalized, normalized_base = " ".join(text.split()), " ".join(baseline.split())
    result: dict[str, Any] = {
        **current, "baseline": reference,
        "text_similarity": 1.0 if normalized == normalized_base else SequenceMatcher(
            None, normalized_base, normalized, autojunk=False).ratio(),
    }
    counts_ok = True
    for key in ("heading_count", "table_count"):
        base = reference[key]
        error = abs(current[key] - base) / base if base else (0.0 if current[key] == 0 else None)
        result[f"{key}_relative_error"] = error
        counts_ok = counts_ok and error is not None and error <= THRESHOLDS["count_relative_error_max"]
    result["passes_gate"] = bool(counts_ok and result["text_similarity"] >= THRESHOLDS["text_similarity_min"])
    return result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, SimpleNamespace):
        return _jsonable(vars(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class GpuTelemetry:
    """Raw nvidia-smi samples (all visible GPUs), independent of Torch allocation peaks."""
    def __init__(self, path: Path, variant: str):
        self.path, self.variant = path, variant
        self.binary = shutil.which("nvidia-smi")
        self.stop = threading.Event()
        self.context: dict[str, Any] = {"phase": "cold_start"}
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            while not self.stop.is_set():
                sample = {"timestamp": time.time(), "variant": self.variant, **self.context}
                try:
                    result = subprocess.run(
                        [self.binary, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5, check=True,
                    )
                    sample["raw_csv"] = result.stdout.strip()
                    sample["gpus"] = []
                    for index, line in enumerate(result.stdout.splitlines()):
                        fields = line.split(",")
                        def number(raw: str) -> float | None:
                            match = re.search(r"\d+(?:\.\d+)?", raw)
                            return float(match.group()) if match else None
                        sample["gpus"].append({"index": index, "utilization_percent": number(fields[0]),
                                               "memory_used_mib": number(fields[1])})
                except Exception as exc:
                    sample["error"] = str(exc)
                fp.write(json.dumps(sample) + "\n")
                fp.flush()
                self.stop.wait(2.0)

    def __enter__(self):
        self.path.touch()
        if self.binary:
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.binary:
            self.thread.join(timeout=6)


class GpuPeak:
    def __init__(self, device: str, dry_run: bool):
        self.cuda = None
        self.device = device
        self.status = "disabled_dry_run" if dry_run else "unavailable"
        if not dry_run:
            try:
                torch = importlib.import_module("torch")
                if device.startswith("cuda") and torch.cuda.is_available():
                    self.cuda = torch.cuda
                    self.status = "torch_cuda_max_memory_allocated"
            except Exception as exc:
                self.status = f"unavailable: {exc}"

    def begin(self) -> None:
        if self.cuda is not None:
            self.cuda.synchronize(self.device)
            self.cuda.reset_peak_memory_stats(self.device)

    def end(self) -> int | None:
        if self.cuda is None:
            return None
        self.cuda.synchronize(self.device)
        return int(self.cuda.max_memory_allocated(self.device))


# Dry-run stand-ins implement only the Docling surface used by GlossExtract.
# The real GlossExtract, option builder, conversion dispatch and Markdown export
# still execute. FakeConverter can be monkeypatched for in-process tests.
class _FakeOptions(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(do_ocr=False, do_table_structure=True, images_scale=1.25,
                         layout_batch_size=4, table_structure_options=SimpleNamespace(mode="accurate"),
                         accelerator_options=SimpleNamespace(device="cpu"),
                         do_picture_description=False, picture_description_options=None,
                         enable_remote_services=False)
        self.__dict__.update(kwargs)

    def model_dump(self):
        return _jsonable(vars(self))


class FakeConverter:
    def __init__(self, *, variant: str, pages: dict[str, int], format_options, **kwargs):
        self.variant, self.pages = variant, pages
        self.options = format_options[_FakeInputFormat.PDF].pipeline_options

    def markdown(self, path: Path, pages: int) -> str:
        text = f"# {path.stem}\n\n## Content\n\nDeterministic dry-run text, {pages} pages.\n"
        if self.options.do_table_structure:
            text += "\n| Field | Value |\n| --- | --- |\n| sample | 1 |\n"
        return text

    def convert(self, file, *, raises_on_error=False, page_range=None):
        path = Path(file)
        pages = self.pages[str(path)]
        if page_range:
            pages = min(pages, page_range[1]) - page_range[0] + 1
        document = SimpleNamespace(pages={i: None for i in range(1, pages + 1)},
                                   export_to_markdown=lambda: self.markdown(path, pages))
        return SimpleNamespace(input=SimpleNamespace(file=path), document=document,
                               status=_FakeStatus.SUCCESS, errors=[], timings={})

    def convert_all(self, files, **kwargs):
        return [self.convert(file, **kwargs) for file in files]


class _FakeInputFormat(Enum):
    PDF = "pdf"
    DOCX = "docx"
    XML_JATS = "xml_jats"
    HTML = "html"
    PPTX = "pptx"
    CSV = "csv"
    MD = "md"


class _FakeStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


@contextmanager
def fake_docling(variant: str, pages: dict[str, int]):
    """Scoped fake modules: no installed Docling or Torch is loaded or replaced."""
    local_names = {"glossapi.gloss_extract", "glossapi.ocr.docling", "glossapi.ocr.docling.pipeline",
                   "glossapi.ocr.docling_pipeline"}
    def selected(name):
        return name == "docling" or name.startswith("docling.") or name in local_names
    saved = {name: module for name, module in sys.modules.items() if selected(name)}
    parent_attrs = {}
    for name in local_names:
        parent_name, attr = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent:
            parent_attrs[(parent_name, attr)] = (hasattr(parent, attr), getattr(parent, attr, None))
    for name in saved:
        sys.modules.pop(name, None)

    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__path__ = []
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

    class TableFormerMode(Enum):
        ACCURATE = "accurate"
        FAST = "fast"

    class StandardPdfPipeline:
        def __init__(self, options):
            self.options = options

    try:
        for name in ("docling", "docling.datamodel", "docling.pipeline", "docling.backend"):
            module(name)
        module("docling.datamodel.pipeline_options", PdfPipelineOptions=_FakeOptions,
               ThreadedPdfPipelineOptions=_FakeOptions, AcceleratorOptions=SimpleNamespace,
               AcceleratorDevice=SimpleNamespace(CUDA="cuda", CPU="cpu", MPS="mps", AUTO="auto"),
               TableStructureOptions=lambda **kw: SimpleNamespace(do_cell_matching=True, **kw),
               TableFormerMode=TableFormerMode, LayoutOptions=SimpleNamespace,
               PictureDescriptionApiOptions=SimpleNamespace)
        module("docling.datamodel.base_models", InputFormat=_FakeInputFormat, ConversionStatus=_FakeStatus)
        module("docling.datamodel.document", ConversionResult=SimpleNamespace)
        module("docling.datamodel.settings", settings=SimpleNamespace(debug=SimpleNamespace(), perf=SimpleNamespace()))
        module("docling.pipeline.simple_pipeline", SimplePipeline=object)
        module("docling.pipeline.standard_pdf_pipeline", StandardPdfPipeline=StandardPdfPipeline)
        module("docling.backend.docling_parse_backend", DoclingParseDocumentBackend=type("NativeBackend", (), {}))
        module("docling.backend.pypdfium2_backend", PyPdfiumDocumentBackend=type("SafeBackend", (), {}))
        module("docling.document_converter",
               DocumentConverter=lambda **kw: FakeConverter(variant=variant, pages=pages, **kw),
               PdfFormatOption=SimpleNamespace)
        yield
    finally:
        for name in list(sys.modules):
            if selected(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        for name in local_names:
            parent_name, attr = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent:
                existed, value = parent_attrs.get((parent_name, attr), (False, None))
                if existed:
                    setattr(parent, attr, value)
                elif hasattr(parent, attr):
                    delattr(parent, attr)


def run_variant(config: dict[str, Any], variant: str, out: Path) -> dict[str, Any]:
    """Worker entry point, also callable with a monkeypatched FakeConverter."""
    out.mkdir(parents=True, exist_ok=True)
    documents = config["documents"]
    pages = {doc["path"]: doc["source_pages"] for doc in documents}
    metadata: dict[str, Any] = {"variant": variant, "pid": os.getpid(), "dry_run": config["dry_run"],
                                "overrides": VARIANTS[variant], "status": "ok"}
    rows = []
    cold_started = time.perf_counter()
    extractor = None
    with GpuTelemetry(out / "gpu_telemetry.jsonl", variant) as telemetry:
        metadata["telemetry"] = {"binary": telemetry.binary, "interval_seconds": 2.0}
        context = fake_docling(variant, pages) if config["dry_run"] else nullcontext()
        with context:
            try:
                # Never import Corpus: its package initializer can load the Rust cleaner.
                extract_module = importlib.import_module("glossapi.gloss_extract")
                if config["dry_run"]:
                    extract_module._maybe_import_torch = lambda **kw: None
                peak = GpuPeak(config["device"], config["dry_run"])
                metadata["gpu_peak_status"] = peak.status
                peak.begin()
                extractor = extract_module.GlossExtract()
                extractor.benchmark_mode = True
                extractor.configure_batch_policy("docling", max_batch_files=1, prefer_safe_backend=False)
                extractor.enable_accel(threads=2, type=config["device"])
                builder = extract_module.build_layout_pipeline
                def variant_builder(**kwargs):
                    def hook(opts):
                        apply_variant(opts, variant)
                    return builder(**kwargs, options_hook=hook)
                extract_module.build_layout_pipeline = variant_builder
                try:
                    extractor.ensure_extractor(images_scale=VARIANTS[variant].get("images_scale", 1.25),
                                               profile_timings=False)
                finally:
                    extract_module.build_layout_pipeline = builder
                metadata["options"] = _jsonable(extractor._active_pdf_options)
                # Corpus.extract dispatches through _process_batch, which applies
                # these caps before conversion. Preserve that production policy.
                extractor._apply_thread_caps()
                convert_all = extractor._convert_all_with_timeout
                conversion = {}
                def limited_convert(files, timeout_s, **kwargs):
                    if config["max_pages"] is not None:
                        kwargs["page_range"] = (1, min(config["max_pages"], pages[str(Path(files[0]))]))
                    results = convert_all(files, timeout_s, **kwargs)
                    conversion["pages"] = sum(len(res.document.pages) for res in results if res.document is not None)
                    conversion["success"] = all(res.status == extract_module.ConversionStatus.SUCCESS for res in results)
                    return results
                extractor._convert_all_with_timeout = limited_convert

                def measure(doc, phase, repeat, started=None):
                    output_dir = out / phase / str(repeat) / doc["id"] / "markdown"
                    telemetry.context = {"phase": phase, "document": doc["path"], "repeat": repeat}
                    conversion.clear()
                    if started is None:
                        peak.begin()
                        started = time.perf_counter()
                    ok = extractor._process_single_document(Path(doc["path"]), output_dir=output_dir)
                    memory = peak.end()
                    wall = time.perf_counter() - started
                    md = output_dir / f"{Path(doc['path']).stem}.md"
                    actual_pages = conversion.get("pages", 0)
                    expected_pages = min(doc["source_pages"], config["max_pages"] or doc["source_pages"])
                    ok = bool(ok and md.exists() and conversion.get("success") and actual_pages == expected_pages)
                    row = {"variant": variant, "document": doc["path"], "document_id": doc["id"],
                           "repeat": repeat, "phase": phase, "status": "ok" if ok else "error",
                           "wall_seconds": wall, "pages": actual_pages, "expected_pages": expected_pages,
                           "seconds_per_page": wall / actual_pages if actual_pages else None,
                           "peak_gpu_memory_bytes": memory, "markdown_path": str(md) if md.exists() else None}
                    if not ok:
                        row["error"] = "Conversion failed, was partial, omitted Markdown, or returned an unexpected page count"
                    rows.append(row)

                measure(documents[0], "cold_start", 0, started=cold_started)
                for repeat in range(1, config["repeats"] + 1):
                    for doc in documents:
                        measure(doc, "measured", repeat)
                if any(row["status"] != "ok" for row in rows):
                    metadata["status"] = "error"
            except UnsupportedVariant as exc:
                metadata.update(status="unsupported", reason=str(exc))
            except Exception as exc:
                metadata.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            finally:
                if extractor is not None:
                    extractor.release_resources()
    if not rows or (metadata["status"] == "error" and not any(row["status"] == "error" for row in rows)):
        rows.append({"variant": variant, "phase": "cold_start", "status": metadata["status"],
                     "wall_seconds": time.perf_counter() - cold_started, "error": metadata.get("reason")})
    with (out / "results.jsonl").open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, allow_nan=False) + "\n")
    _write_json(out / "worker.json", metadata)
    return metadata


def summarize(rows: list[dict], workers: dict[str, dict], dry_run: bool) -> dict:
    baseline = {(row["document_id"], row["repeat"]): row for row in rows
                if row["variant"] == "baseline" and row["phase"] == "measured" and row["status"] == "ok"}
    for row in rows:
        if row["phase"] != "measured" or row["status"] != "ok":
            continue
        base = baseline.get((row["document_id"], row["repeat"]))
        row["fidelity"] = fidelity_metrics(Path(row["markdown_path"]).read_text(encoding="utf-8"),
                                           Path(base["markdown_path"]).read_text(encoding="utf-8")) if base else None
    variants = {}
    for name, worker in workers.items():
        samples = [row for row in rows if row["variant"] == name and row["phase"] == "measured"]
        successful = [row for row in samples if row["status"] == "ok"]
        fidelity = [row["fidelity"] for row in successful if row.get("fidelity") is not None]
        def median(key):
            values = [row[key] for row in successful if row.get(key) is not None]
            return statistics.median(values) if values else None
        variants[name] = {**worker, "measured_documents": len(samples), "successful_documents": len(successful),
                          "median_wall_seconds": median("wall_seconds"),
                          "median_seconds_per_page": median("seconds_per_page"),
                          "median_peak_gpu_memory_bytes": median("peak_gpu_memory_bytes"),
                          "cold_start": [row for row in rows if row["variant"] == name and row["phase"] == "cold_start"],
                          "fidelity_means": {key: statistics.mean(item[key] for item in fidelity)
                                             for key in ("text_similarity", "heading_count", "table_count", "total_chars")} if fidelity else None,
                          "passes_fidelity_gate": bool(worker["status"] == "ok" and samples and len(fidelity) == len(samples)
                                              and all(item["passes_gate"] for item in fidelity))}
    base_seconds = variants["baseline"]["median_seconds_per_page"]
    baseline_succeeded = variants["baseline"]["passes_fidelity_gate"]
    for variant in variants.values():
        seconds = variant["median_seconds_per_page"]
        ratio = seconds / base_seconds if seconds is not None and base_seconds else None
        variant["ratio_to_baseline"] = ratio
        variant["passes_speed_gate"] = bool(ratio is not None and ratio <= THRESHOLDS["seconds_per_page_ratio_max"])
        variant["passes_gate"] = bool(baseline_succeeded and variant["passes_fidelity_gate"] and variant["passes_speed_gate"])
        variant["accepted"] = bool(not dry_run and variant["passes_gate"])
    return {"dry_run": dry_run, "thresholds": THRESHOLDS, "variants": variants,
            "gate_policy": "passes_gate requires a successful baseline, fidelity thresholds on every measured document/repeat and >=25% faster; accepted also requires a real run",
            "timing_policy": "cold start includes initialization and first conversion; excluded from medians; all documents are then reprocessed each repeat"}


def _pdf_paths(inputs: list[str]) -> list[Path]:
    paths = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() == ".pdf"))
        elif path.suffix.lower() == ".pdf":
            paths.append(path)
        else:
            content = path.read_text(encoding="utf-8")
            entries = json.loads(content) if path.suffix.lower() == ".json" else [
                line.strip() for line in content.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            if not isinstance(entries, list) or not all(isinstance(entry, str) for entry in entries):
                raise ValueError("PDF list must contain path strings")
            for entry in entries:
                entry_path = Path(entry).expanduser()
                paths.append((entry_path if entry_path.is_absolute() else path.parent / entry_path).resolve())
    paths = list(dict.fromkeys(paths))
    if not paths or any(not path.is_file() or path.suffix.lower() != ".pdf" for path in paths):
        raise ValueError("Select at least one existing PDF (directory, paths, or newline/JSON list)")
    return paths


def _count_pages(path: Path) -> int:
    # Lightweight readers only; do not import the Corpus-based checkpoint benchmark.
    from pypdf import PdfReader
    pages = len(PdfReader(str(path)).pages)
    if pages < 1:
        raise ValueError(f"PDF has no pages: {path}")
    return pages


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", nargs="+", help="directory, PDF paths, or newline/JSON path list")
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=_positive, default=1)
    parser.add_argument("--max-pages", type=_positive, help="convert at most the first N pages of each PDF")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=list(VARIANTS), help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    if args.worker:
        if not args.config:
            parser.error("worker requires --config")
        worker = run_variant(json.loads(Path(args.config).read_text(encoding="utf-8")), args.worker, out)
        return int(worker["status"] == "error")
    if not args.pdfs:
        parser.error("--pdfs is required")
    names = [name for raw in args.variants for name in raw.split(",")]
    if names == ["all"]:
        names = list(VARIANTS)
    if any(name not in VARIANTS for name in names):
        parser.error(f"variants must be all or names from {', '.join(VARIANTS)}")
    names = list(dict.fromkeys(["baseline", *names]))
    try:
        paths = _pdf_paths(args.pdfs)
        documents = [{"id": f"{index:04d}", "path": str(path), "source_pages": _count_pages(path)}
                     for index, path in enumerate(paths)]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if out.exists() and any(out.iterdir()):
        parser.error("--out must be new or empty (preserve prior runs)")
    out.mkdir(parents=True, exist_ok=True)
    config = {"documents": documents, "device": args.device, "repeats": args.repeats,
              "max_pages": args.max_pages, "dry_run": args.dry_run, "variants": names}
    _write_json(out / "config.json", config)
    workers, rows = {}, []
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[2]), *[
        str(Path(entry or ".").resolve()) for entry in env.get("PYTHONPATH", "").split(os.pathsep)]])
    for name in names:
        worker_dir = out / name
        worker_dir.mkdir()
        command = [sys.executable, "-m", "glossapi.scripts.docling_cost_benchmark", "--worker", name,
                   "--config", str(out / "config.json"), "--out", str(worker_dir)]
        with (worker_dir / "worker.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=worker_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt = worker_dir / "worker.json"
        workers[name] = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {
            "status": "error", "reason": f"worker exited {result.returncode}; see {worker_dir / 'worker.log'}"}
        if result.returncode:
            workers[name]["status"] = "error"
        worker_results = worker_dir / "results.jsonl"
        if worker_results.exists():
            rows.extend(json.loads(line) for line in worker_results.read_text(encoding="utf-8").splitlines())
        print(f"{name}: {workers[name]['status']}", flush=True)
    summary = {**summarize(rows, workers, args.dry_run), "config": config}
    with (out / "results.jsonl").open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, allow_nan=False) + "\n")
    with (out / "gpu_telemetry.jsonl").open("w", encoding="utf-8") as fp:
        for name in names:
            telemetry = out / name / "gpu_telemetry.jsonl"
            if telemetry.exists():
                fp.write(telemetry.read_text(encoding="utf-8"))
    _write_json(out / "summary.json", summary)
    print(f"summary: {out / 'summary.json'}")
    return int(any(worker["status"] == "error" for worker in workers.values()))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
