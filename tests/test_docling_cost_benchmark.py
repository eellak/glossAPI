import importlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from glossapi.scripts import docling_cost_benchmark as benchmark


@pytest.fixture(autouse=True)
def guard_real_imports(monkeypatch):
    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith(("torch.", "glossapi.corpus", "glossapi_rs")):
                raise AssertionError(f"Forbidden heavy/build-triggering import: {fullname}")
            if fullname == "docling" or fullname.startswith("docling."):
                # Fake modules are already in sys.modules. Missing optional fake
                # modules must fall back rather than load any installed Docling.
                raise ModuleNotFoundError(f"Real Docling import forbidden: {fullname}")
    monkeypatch.setattr(sys, "meta_path", [Guard(), *sys.meta_path])
    for name in list(os.environ):
        if name.startswith("GLOSSAPI_DOCLING_"):
            monkeypatch.delenv(name)


def pdf(path: Path, pages=2):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=100)
    writer.write(path)
    return path


def config(paths, repeats=1, max_pages=None):
    return {"documents": [{"id": f"{i:04d}", "path": str(path), "source_pages": benchmark._count_pages(path)}
                           for i, path in enumerate(paths)],
            "dry_run": True, "repeats": repeats, "max_pages": max_pages, "device": "cuda:0"}


@pytest.mark.parametrize("variant", list(benchmark.VARIANTS))
def test_options_reach_fake_converter_per_variant(tmp_path, monkeypatch, variant):
    path = pdf(tmp_path / "input.pdf")
    seen = []
    class Converter(benchmark.FakeConverter):
        def convert(self, *args, **kwargs):
            seen.append(self.options.model_dump())
            return super().convert(*args, **kwargs)
    monkeypatch.setattr(benchmark, "FakeConverter", Converter)
    # Keep the production extractor's conversion.log in this test's directory.
    monkeypatch.chdir(tmp_path)
    receipt = benchmark.run_variant(config([path]), variant, tmp_path / "out")
    assert receipt["status"] == "ok"
    assert len(seen) == 2  # cold start, then measured reprocessing of the same PDF
    options = seen[-1]
    assert options["do_ocr"] is False
    assert options["images_scale"] == (1.0 if "scale_1.0" in variant else 1.25)
    assert options["table_structure_options"]["mode"] == ("fast" if variant == "table_fast" else "accurate")
    assert options["do_table_structure"] is not variant.startswith("no_tables")
    assert options["layout_batch_size"] == {"layout_batch_16": 16, "layout_batch_32": 32}.get(variant, 4)
    rows = [json.loads(line) for line in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    assert all(row["pages"] == 2 and Path(row["markdown_path"]).is_file() for row in rows)
    assert receipt["gpu_peak_status"] == "disabled_dry_run"


def test_baseline_hook_and_no_override_preserve_builder_field_dict():
    with benchmark.fake_docling("baseline", {}):
        pipeline = importlib.import_module("glossapi.ocr.docling.pipeline")
        before_pipeline, before = pipeline.build_layout_pipeline()
        before_dict = before.model_dump()
        benchmark.apply_variant(before, "baseline")
        assert before.model_dump() == before_dict
        after_pipeline, after = pipeline.build_layout_pipeline(
            options_hook=lambda options: benchmark.apply_variant(options, "baseline"))
        assert after.model_dump() == before_dict
        assert before_pipeline.options is before
        assert after_pipeline.options is after


@pytest.mark.parametrize("variant", ["layout_batch_16", "layout_batch_32"])
def test_missing_batch_field_is_recorded_as_unsupported(tmp_path, monkeypatch, variant):
    class OldOptions(benchmark._FakeOptions):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            del self.layout_batch_size
    monkeypatch.setattr(benchmark, "_FakeOptions", OldOptions)
    monkeypatch.chdir(tmp_path)
    receipt = benchmark.run_variant(config([pdf(tmp_path / "a.pdf")]), variant, tmp_path / "out")
    assert receipt["status"] == "unsupported"
    assert "layout_batch_size" in receipt["reason"]
    rows = [json.loads(line) for line in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "unsupported"


def test_cold_start_excluded_and_all_documents_repeated(tmp_path, monkeypatch):
    paths = [pdf(tmp_path / "a.pdf"), pdf(tmp_path / "b.pdf")]
    clock = SimpleNamespace(value=0.0)
    calls = []
    class TimedConverter(benchmark.FakeConverter):
        def __init__(self, **kwargs):
            clock.value += 50.0
            super().__init__(**kwargs)

        def markdown(self, path, pages):
            calls.append(path.name)
            clock.value += 100.0 if len(calls) == 1 else 4.0
            return super().markdown(path, pages)
    monkeypatch.setattr(benchmark, "FakeConverter", TimedConverter)
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: clock.value, time=lambda: clock.value))
    monkeypatch.chdir(tmp_path)
    receipt = benchmark.run_variant(config(paths, repeats=2), "baseline", tmp_path / "out")
    rows = [json.loads(line) for line in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    summary = benchmark.summarize(rows, {"baseline": receipt}, True)["variants"]["baseline"]
    assert calls == ["a.pdf", "a.pdf", "b.pdf", "a.pdf", "b.pdf"]
    assert summary["cold_start"][0]["wall_seconds"] == 150.0
    assert summary["measured_documents"] == 4
    assert summary["median_wall_seconds"] == 4.0
    assert summary["median_seconds_per_page"] == 2.0
    assert summary["ratio_to_baseline"] == 1.0
    assert summary["passes_fidelity_gate"] is True
    assert summary["passes_gate"] is False  # Baseline is not 25% faster than itself.
    assert summary["accepted"] is False


def test_fidelity_identical_whitespace_and_dropped_heading():
    baseline = "# Title\n\n## Subheading\n\n" + "Some text. " * 100 + "\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    same = benchmark.fidelity_metrics(baseline, baseline)
    assert same["text_similarity"] == 1.0
    assert same["heading_count"] == 2
    assert same["table_count"] == 1
    assert same["total_chars"] == len(baseline)
    assert same["passes_gate"] is True
    collapsed = benchmark.fidelity_metrics(baseline.replace("Some text.", "Some    text."), baseline)
    assert collapsed["text_similarity"] == 1.0
    dropped = benchmark.fidelity_metrics(baseline.replace("## Subheading\n", ""), baseline)
    assert dropped["text_similarity"] >= 0.98
    assert dropped["heading_count_relative_error"] == 0.5
    assert dropped["passes_gate"] is False


def test_fake_variant_dropped_heading_flows_into_fidelity_summary(tmp_path, monkeypatch):
    class DroppedHeading(benchmark.FakeConverter):
        def markdown(self, path, pages):
            text = super().markdown(path, pages) + "Body text. " * 200
            return text.replace("## Content\n", "") if self.variant == "table_fast" else text
    monkeypatch.setattr(benchmark, "FakeConverter", DroppedHeading)
    monkeypatch.chdir(tmp_path)
    cfg = config([pdf(tmp_path / "a.pdf")])
    workers, rows = {}, []
    for variant in ("baseline", "table_fast"):
        out = tmp_path / variant
        workers[variant] = benchmark.run_variant(cfg, variant, out)
        rows.extend(json.loads(line) for line in (out / "results.jsonl").read_text().splitlines())
    variants = benchmark.summarize(rows, workers, True)["variants"]
    assert variants["baseline"]["passes_fidelity_gate"] is True
    assert variants["baseline"]["passes_gate"] is False
    assert variants["table_fast"]["fidelity_means"]["text_similarity"] >= 0.98
    assert variants["table_fast"]["fidelity_means"]["heading_count"] == 1
    assert variants["table_fast"]["passes_gate"] is False


def test_table_groups_and_zero_baseline_counts():
    assert benchmark.markdown_metrics("| a | b |\n| 1 | 2 |\n\n| c | d |\n")["table_count"] == 2
    assert benchmark.fidelity_metrics("text", "text")["passes_gate"] is True
    extra = benchmark.fidelity_metrics("# Added\ntext", "text")
    assert extra["heading_count_relative_error"] is None
    assert extra["passes_gate"] is False
    assert benchmark.fidelity_metrics("text", "| a | b |\n")["passes_gate"] is False


@pytest.mark.parametrize("headings,passes", [(98, True), (97, False), (102, True), (103, False)])
def test_count_gate_two_percent_boundary(headings, passes):
    # Keep text similarity above its threshold while varying count independently.
    body = "body text " * 200
    assert benchmark.fidelity_metrics("# x\n" * headings + body, "# x\n" * 100 + body)["passes_gate"] is passes


def test_summary_ratios_speed_fidelity_and_failed_baseline(tmp_path):
    rows = []
    for variant, wall, text in [("baseline", 4, "# Title\n" + "text " * 100),
                                ("table_fast", 3, "# Title\n" + "text " * 100),
                                ("no_tables", 2, "text " * 100)]:
        path = tmp_path / f"{variant}.md"
        path.write_text(text)
        rows.append({"variant": variant, "phase": "measured", "document_id": "0", "repeat": 1,
                     "status": "ok", "markdown_path": str(path), "wall_seconds": wall,
                     "seconds_per_page": wall / 2, "peak_gpu_memory_bytes": None})
    workers = {variant: {"status": "ok"} for variant in ("baseline", "table_fast", "no_tables")}
    summary = benchmark.summarize(rows, workers, False)["variants"]
    assert summary["table_fast"]["ratio_to_baseline"] == 0.75
    assert summary["table_fast"]["passes_gate"] is True
    assert summary["table_fast"]["passes_speed_gate"] is True
    assert summary["table_fast"]["accepted"] is True
    assert summary["no_tables"]["passes_gate"] is False
    assert summary["no_tables"]["accepted"] is False
    assert summary["no_tables"]["fidelity_means"]["heading_count"] == 0
    rows[0]["status"] = "error"
    workers["baseline"]["status"] = "error"
    summary = benchmark.summarize(rows, workers, False)["variants"]
    assert summary["table_fast"]["passes_gate"] is False
    assert summary["table_fast"]["ratio_to_baseline"] is None


def test_max_pages_reaches_converter_and_measures_actual_pages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    receipt = benchmark.run_variant(config([pdf(tmp_path / "a.pdf", 4)], max_pages=2), "baseline", tmp_path / "out")
    assert receipt["status"] == "ok"
    rows = [json.loads(line) for line in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    assert all(row["pages"] == row["expected_pages"] == 2 for row in rows)
    assert "2 pages" in Path(rows[-1]["markdown_path"]).read_text()


def test_partial_conversion_fails_even_if_markdown_exists(tmp_path, monkeypatch):
    class Partial(benchmark.FakeConverter):
        def convert(self, *args, **kwargs):
            result = super().convert(*args, **kwargs)
            result.status = benchmark._FakeStatus.PARTIAL_SUCCESS
            return result
    monkeypatch.setattr(benchmark, "FakeConverter", Partial)
    monkeypatch.chdir(tmp_path)
    receipt = benchmark.run_variant(config([pdf(tmp_path / "a.pdf")]), "baseline", tmp_path / "out")
    assert receipt["status"] == "error"


def test_gpu_peak_reset_synchronize_and_read(monkeypatch):
    calls = []
    cuda = SimpleNamespace(is_available=lambda: True,
                           synchronize=lambda device: calls.append(("sync", device)),
                           reset_peak_memory_stats=lambda device: calls.append(("reset", device)),
                           max_memory_allocated=lambda device: 1234)
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda name: SimpleNamespace(cuda=cuda))
    meter = benchmark.GpuPeak("cuda:0", False)
    meter.begin()
    assert meter.end() == 1234
    assert calls == [("sync", "cuda:0"), ("reset", "cuda:0"), ("sync", "cuda:0")]


def test_telemetry_samples_required_query_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: "/fake/nvidia-smi")
    sampled = benchmark.threading.Event()
    commands = []
    def query(command, **kwargs):
        commands.append(command)
        sampled.set()
        return SimpleNamespace(stdout="42 %, 123 MiB\n[N/A], 456 MiB\n")
    monkeypatch.setattr(benchmark.subprocess, "run", query)
    path = tmp_path / "telemetry.jsonl"
    with benchmark.GpuTelemetry(path, "baseline") as telemetry:
        assert sampled.wait(2)
    assert not telemetry.thread.is_alive()
    row = json.loads(path.read_text().splitlines()[0])
    assert commands[0][1:] == ["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"]
    assert row["gpus"][0] == {"index": 0, "utilization_percent": 42.0, "memory_used_mib": 123.0}
    assert row["gpus"][1]["utilization_percent"] is None


def test_pdf_lists_relative_paths_and_duplicates(tmp_path):
    path = pdf(tmp_path / "a.pdf")
    listing = tmp_path / "files.txt"
    listing.write_text("# comment\na.pdf\n")
    assert benchmark._pdf_paths([str(listing), str(path)]) == [path]
    listing = tmp_path / "files.json"
    listing.write_text(json.dumps(["a.pdf"]))
    assert benchmark._pdf_paths([str(listing)]) == [path]


def test_cli_dry_run_isolated_processes_all_variants(tmp_path):
    inputs = tmp_path / "pdfs"
    inputs.mkdir()
    pdf(inputs / "a.pdf")
    pdf(inputs / "b.pdf", 3)
    out = tmp_path / "out"
    assert benchmark.main(["--pdfs", str(inputs), "--out", str(out), "--variants", "all",
                           "--dry-run", "--repeats", "2", "--max-pages", "2"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    variants = summary["variants"]
    assert set(variants) == set(benchmark.VARIANTS)
    assert len({worker["pid"] for worker in variants.values()}) == len(benchmark.VARIANTS)
    assert all(worker["pid"] != os.getpid() for worker in variants.values())
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == len(benchmark.VARIANTS) * 5
    assert variants["baseline"]["passes_gate"] is False
    assert variants["baseline"]["passes_fidelity_gate"] is True
    assert variants["table_fast"]["passes_fidelity_gate"] is True
    assert variants["table_fast"]["passes_gate"] == variants["table_fast"]["passes_speed_gate"]
    assert variants["no_tables"]["passes_gate"] is False
    assert all(not variant["accepted"] for variant in variants.values())
    assert all(variant["measured_documents"] == 4 for variant in variants.values())
    assert (out / "gpu_telemetry.jsonl").exists()


def test_cli_rejects_invalid_repeats_and_unknown_variant(tmp_path):
    for extra in (["--repeats", "0"], ["--variants", "unknown"]):
        with pytest.raises(SystemExit) as exc:
            benchmark.main(["--pdfs", str(tmp_path), "--out", str(tmp_path / "out"), *extra])
        assert exc.value.code == 2


# CP2 reviewer regressions: retain measured rows when the worker fails.
def e3_summary(tmp_path, ratio):
    md = tmp_path / 'same.md'
    md.write_text('# Heading\n\nBody text.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n')
    rows = [dict(variant=name, phase='measured', status='ok', document_id='0', repeat=1,
                 markdown_path=str(md), wall_seconds=sec, pages=1, seconds_per_page=sec,
                 peak_gpu_memory_bytes=None) for name, sec in [('baseline', 1.0), ('table_fast', ratio)]]
    return benchmark.summarize(rows, {n: {'status': 'ok'} for n in ['baseline', 'table_fast']}, False)


@pytest.mark.parametrize('ratio,passes', [(1.2, False), (0.750001, False), (0.75, True), (0.5, True)])
def test_e3_passes_gate_requires_25_percent_speed_gain(tmp_path, ratio, passes):
    variant = e3_summary(tmp_path, ratio)['variants']['table_fast']
    assert variant['passes_fidelity_gate'] is True
    assert variant['passes_gate'] is passes
    assert variant['accepted'] is passes


def test_e3_failed_baseline_cannot_approve_variant(tmp_path):
    md = tmp_path / 'same.md'
    md.write_text('same body')
    rows = [dict(variant=name, phase='measured', status='ok', document_id='0', repeat=1,
                 markdown_path=str(md), wall_seconds=sec, pages=1, seconds_per_page=sec,
                 peak_gpu_memory_bytes=None) for name, sec in [('baseline', 1.0), ('table_fast', 0.5)]]
    result = benchmark.summarize(rows, {'baseline': {'status': 'error'}, 'table_fast': {'status': 'ok'}}, False)
    assert result['variants']['table_fast']['accepted'] is False, (
        'E3 accepted=True although baseline worker status=error')
