import json
from pathlib import Path

import pandas as pd

from glossapi import Corpus


def _write_download_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def _write_metrics(root: Path, stem: str, *, page_count: int, pages: list[dict]) -> None:
    metrics_dir = root / "json" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "file": f"{stem}.pdf",
        "page_count": page_count,
        "pages": pages,
    }
    (metrics_dir / f"{stem}.metrics.json").write_text(json.dumps(data), encoding="utf-8")


def _write_latex_map(root: Path, stem: str, rows: list[dict]) -> None:
    latex_path = root / "json" / f"{stem}.latex_map.jsonl"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in rows]
    latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_jsonl_export_produces_expected_record(tmp_path):
    corpus = Corpus(input_dir=tmp_path / "in", output_dir=tmp_path / "out")

    markdown = corpus.cleaned_markdown_dir / "sample.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    document_text = "## Sample Document\n\nContent line."
    markdown.write_text(document_text, encoding="utf-8")

    _write_download_results(
        corpus.output_dir / "download_results" / "download_results.parquet",
        [
            {
                "filename": "sample.pdf",
                "url": "https://example.com/sample.pdf",
                "download_success": True,
                "download_error": "",
                "download_retry_count": 0,
                "file_ext": "pdf",
                "is_duplicate": False,
                "duplicate_of": "",
                "source_row": 1,
                "url_index": 1,
                "filename_base": "sample",
                "mojibake_badness_score": 0.02,
                "mojibake_latin_percentage": 0.01,
                "greek_badness_score": 12.0,
                "greek_latin_percentage": 0.99,
                "percentage_greek": 0.9,
                "percentage_latin": 0.1,
                "char_count_no_comments": 123,
                "is_empty": False,
                "filter": "ok",
                "needs_ocr": False,
                "page_count": 3,
                "triage_label": "math_dense",
                "extra_meta": "keep-me",
            }
        ],
    )

    _write_metrics(
        corpus.output_dir,
        "sample",
        page_count=3,
        pages=[
            {"page_no": 1, "formula_count": 2, "code_count": 1},
            {"page_no": 2, "formula_count": 1, "code_count": 0},
            {"page_no": 3, "formula_count": 0, "code_count": 1},
        ],
    )

    _write_latex_map(
        corpus.output_dir,
        "sample",
        [
            {"page_no": 1, "item_index": 1, "accept_score": 1.0},
            {"page_no": 2, "item_index": 1, "accept_score": 0.0},
        ],
    )

    out_path = corpus.output_dir / "export.jsonl"
    corpus.jsonl(out_path)

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    record = records[0]

    assert record["document"] == document_text
    assert record["url"] == "https://example.com/sample.pdf"
    assert record["filename"] == "sample.pdf"
    assert record["filetype"] == "pdf"
    assert record["page_count"] == 3
    assert record["formula_total"] == 3
    assert record["code_total"] == 2
    assert record["math_enriched"] is True
    assert record["math_accepted"] == 1
    assert record["char_count_no_comments"] == 123
    assert record["triage_label"] == "math_dense"
    assert record["extra_meta"] == "keep-me"


def test_jsonl_export_handles_missing_math_data(tmp_path):
    corpus = Corpus(input_dir=tmp_path / "in2", output_dir=tmp_path / "out2")

    markdown = corpus.cleaned_markdown_dir / "plain.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("Hello world", encoding="utf-8")

    _write_download_results(
        corpus.output_dir / "download_results" / "download_results.parquet",
        [
            {
                "filename": "plain.pdf",
                "url": "https://example.com/plain.pdf",
                "download_success": True,
                "download_error": "",
                "download_retry_count": 0,
                "file_ext": "pdf",
                "is_duplicate": False,
                "duplicate_of": "",
                "source_row": 2,
                "url_index": 2,
                "filename_base": "plain",
                "mojibake_badness_score": 0.0,
                "mojibake_latin_percentage": 0.0,
                "greek_badness_score": 0.0,
                "greek_latin_percentage": 0.0,
                "percentage_greek": 0.0,
                "percentage_latin": 1.0,
                "char_count_no_comments": 5,
                "is_empty": False,
                "filter": "ok",
                "needs_ocr": False,
            }
        ],
    )

    _write_metrics(
        corpus.output_dir,
        "plain",
        page_count=1,
        pages=[{"page_no": 1, "formula_count": 0, "code_count": 0}],
    )

    out_path = corpus.output_dir / "plain.jsonl"
    corpus.jsonl(out_path)

    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    record = records[0]

    assert record["document"] == "Hello world"
    assert record["math_enriched"] is False
    assert record["math_accepted"] == 0
    assert record["formula_total"] == 0
    assert record["code_total"] == 0
