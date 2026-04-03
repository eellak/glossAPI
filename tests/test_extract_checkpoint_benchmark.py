import json
from pathlib import Path

from glossapi.scripts import extract_checkpoint_benchmark as benchmark


def test_markdown_headers_counts_markdown_headings():
    text = "# Title\n\ntext\n## Subtitle\n\nnot a header\n### Third\n"
    assert benchmark._markdown_headers(text) == 3


def test_compare_inventory_detects_presence_size_header_and_sha_changes():
    baseline = {
        "a": {"present": True, "byte_size": 10, "header_count": 1, "sha256": "old"},
        "b": {"present": True, "byte_size": 20, "header_count": 0, "sha256": "same"},
    }
    current = {
        "a": {"present": True, "byte_size": 12, "header_count": 2, "sha256": "new"},
        "c": {"present": True, "byte_size": 5, "header_count": 0, "sha256": "other"},
    }
    diff = benchmark._compare_inventory(current, baseline)
    assert diff["added_markdown"] == ["c"]
    assert diff["missing_markdown"] == ["b"]
    assert diff["byte_size_changed"] == ["a"]
    assert diff["header_count_changed"] == ["a"]
    assert diff["sha_changed"] == ["a"]


def test_load_baseline_inventory_reads_report_payload(tmp_path):
    report_path = tmp_path / "baseline.json"
    report_path.write_text(
        json.dumps({"markdown_inventory": {"doc": {"present": True, "byte_size": 1, "header_count": 0}}}),
        encoding="utf-8",
    )
    assert benchmark._load_baseline_inventory(report_path)["doc"]["present"] is True


def test_inventory_markdown_marks_missing_files(tmp_path):
    input_pdf = tmp_path / "sample.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n")
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    inventory = benchmark._inventory_markdown(markdown_dir, pdf_paths=[input_pdf])
    assert inventory["sample"]["present"] is False
    assert inventory["sample"]["byte_size"] == 0
    assert inventory["sample"]["header_count"] == 0


def test_runtime_env_snapshot_captures_docling_batch_knobs(monkeypatch):
    monkeypatch.setenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", "2")
    monkeypatch.setenv("GLOSSAPI_DOCLING_BATCH_TARGET_PAGES", "384")
    monkeypatch.setenv("GLOSSAPI_DOCLING_PAGE_BATCH_SIZE", "8")

    snapshot = benchmark._runtime_env_snapshot()

    assert snapshot["GLOSSAPI_DOCLING_MAX_BATCH_FILES"] == "2"
    assert snapshot["GLOSSAPI_DOCLING_BATCH_TARGET_PAGES"] == "384"
    assert snapshot["GLOSSAPI_DOCLING_PAGE_BATCH_SIZE"] == "8"


def test_apply_cli_tuning_overrides_sets_docling_env(monkeypatch):
    for env_name in benchmark.TUNING_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)

    args = benchmark._parse_args(
        [
            "--input-dir",
            "/tmp/in",
            "--output-dir",
            "/tmp/out",
            "--report-path",
            "/tmp/report.json",
            "--docling-max-batch-files",
            "2",
            "--docling-batch-target-pages",
            "512",
            "--docling-page-batch-size",
            "8",
        ]
    )
    benchmark._apply_cli_tuning_overrides(args)

    assert benchmark._runtime_env_snapshot()["GLOSSAPI_DOCLING_MAX_BATCH_FILES"] == "2"
    assert benchmark._runtime_env_snapshot()["GLOSSAPI_DOCLING_BATCH_TARGET_PAGES"] == "512"
    assert benchmark._runtime_env_snapshot()["GLOSSAPI_DOCLING_PAGE_BATCH_SIZE"] == "8"
