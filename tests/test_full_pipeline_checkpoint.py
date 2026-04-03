import json

import pandas as pd

from glossapi.scripts import full_pipeline_checkpoint as checkpoint


def test_read_metadata_counts_handles_missing_and_populated_parquet(tmp_path):
    missing = checkpoint._read_metadata_counts(tmp_path / "missing.parquet")
    assert missing["rows_total"] == 0

    parquet_path = tmp_path / "download_results.parquet"
    pd.DataFrame(
        [
            {"filename": "a.pdf", "needs_ocr": True, "ocr_success": False, "text": ""},
            {"filename": "b.pdf", "needs_ocr": False, "ocr_success": True, "text": "hello"},
        ]
    ).to_parquet(parquet_path, index=False)

    counts = checkpoint._read_metadata_counts(parquet_path)
    assert counts == {
        "rows_total": 2,
        "needs_ocr_true": 1,
        "ocr_success_true": 1,
        "text_nonempty": 1,
    }


def test_full_pipeline_checkpoint_main_writes_summary(tmp_path, monkeypatch):
    class DummyCorpus:
        def __init__(self, input_dir, output_dir):
            self.input_dir = input_dir
            self.output_dir = output_dir

        def _metadata_path(self):
            path = self.output_dir / "download_results" / "download_results.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        def extract(self, **kwargs):
            md = self.output_dir / "markdown"
            md.mkdir(parents=True, exist_ok=True)
            (md / "doc.md").write_text("raw text", encoding="utf-8")
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": False, "ocr_success": False, "text": ""}]
            ).to_parquet(self._metadata_path(), index=False)

        def clean(self, **kwargs):
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": True, "ocr_success": False, "text": ""}]
            ).to_parquet(self._metadata_path(), index=False)

        def ocr(self, **kwargs):
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": False, "ocr_success": True, "text": "fixed text"}]
            ).to_parquet(self._metadata_path(), index=False)

        def jsonl(self, output_path, **kwargs):
            output_path.write_text(json.dumps({"text": "fixed text"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(checkpoint, "Corpus", DummyCorpus)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n")

    output_dir = tmp_path / "out"
    export_path = tmp_path / "export.jsonl"
    report_path = tmp_path / "report.json"

    rc = checkpoint.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--export-path",
            str(export_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["post_clean_counts"]["needs_ocr_true"] == 1
    assert report["post_ocr_counts"]["ocr_success_true"] == 1
    assert report["export_records"] == 1


def test_full_pipeline_checkpoint_can_resume_from_ocr_phase(tmp_path, monkeypatch):
    class DummyCorpus:
        def __init__(self, input_dir, output_dir):
            self.input_dir = input_dir
            self.output_dir = output_dir

        def _metadata_path(self):
            path = self.output_dir / "download_results" / "download_results.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        def extract(self, **kwargs):
            raise AssertionError("extract should have been skipped")

        def clean(self, **kwargs):
            raise AssertionError("clean should have been skipped")

        def ocr(self, **kwargs):
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": False, "ocr_success": True, "text": "fixed text"}]
            ).to_parquet(self._metadata_path(), index=False)

        def jsonl(self, output_path, **kwargs):
            output_path.write_text(json.dumps({"text": "fixed text"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(checkpoint, "Corpus", DummyCorpus)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    metadata_path = output_dir / "download_results" / "download_results.parquet"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"filename": "doc.pdf", "needs_ocr": True, "ocr_success": False, "text": ""}]
    ).to_parquet(metadata_path, index=False)

    export_path = tmp_path / "export.jsonl"
    report_path = tmp_path / "report.json"

    rc = checkpoint.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--export-path",
            str(export_path),
            "--report-path",
            str(report_path),
            "--skip-extract",
            "--skip-clean",
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["skipped_phases"] == ["extract", "clean"]
    assert report["post_extract_counts"]["needs_ocr_true"] == 1
    assert report["post_ocr_counts"]["ocr_success_true"] == 1
    assert report["export_records"] == 1
