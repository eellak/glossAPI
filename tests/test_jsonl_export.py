import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

try:
    import datasets  # type: ignore

    _HAS_DATASETS = True
except Exception:  # pragma: no cover - optional dependency
    datasets = None  # type: ignore
    _HAS_DATASETS = False

from glossapi import Corpus


def _write_download_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def _write_source_metadata(path: Path, rows: list[dict]) -> None:
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
    assert record["filename"] == "sample"
    assert record["filetype"] == "pdf"
    assert record["doc_id"] == _expected_doc_id("sample.pdf")
    assert record["chunk_id"] == 0
    assert record["page_count"] == 3
    assert record["formula_total"] == 3
    assert record["code_total"] == 2
    assert record["math_enriched"] is True
    assert record["math_accepted"] == 1
    assert record["char_count_no_comments"] == 123
    assert record["triage_label"] == "math_dense"
    assert record["extra_meta"] == "keep-me"
    assert record.get("webdocs_score") is None


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
    assert record.get("webdocs_score") is None
    assert record["doc_id"] == _expected_doc_id("plain.pdf")
    assert record["chunk_id"] == 0


def test_jsonl_export_supports_custom_text_and_metadata_block(tmp_path):
    corpus = Corpus(input_dir=tmp_path / "in3", output_dir=tmp_path / "out3")

    markdown = corpus.cleaned_markdown_dir / "alpha.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("Document text payload", encoding="utf-8")

    metadata_rows = [
        {
            "filename": "alpha.pdf",
            "filter": "ok",
            "greek_badness_score": 1.23,
            "is_empty": False,
            "latin_percentage": 4.56,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "percentage_greek": 92.1,
            "polytonic_ratio": 0.12,
        }
    ]
    _write_download_results(
        corpus.output_dir / "download_results" / "download_results.parquet",
        metadata_rows,
    )

    out_path = corpus.output_dir / "alpha.jsonl"
    corpus.jsonl(
        out_path,
        text_key="text",
        metadata_key="pipeline_metadata",
        metadata_fields=[
            "filter",
            "greek_badness_score",
            "is_empty",
            "latin_percentage",
            "mojibake_badness_score",
            "needs_ocr",
            "percentage_greek",
            "polytonic_ratio",
        ],
        include_remaining_metadata=False,
    )

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["text"] == "Document text payload"
    assert "document" not in record
    assert set(record.keys()) == {"text", "pipeline_metadata", "filename", "doc_id", "chunk_id"}
    assert record["filename"] == "alpha"
    assert record["doc_id"] == _expected_doc_id("alpha.pdf")
    assert record["chunk_id"] == 0

    metadata = record["pipeline_metadata"]
    for key in [
        "filter",
        "greek_badness_score",
        "is_empty",
        "latin_percentage",
        "mojibake_badness_score",
        "needs_ocr",
        "percentage_greek",
        "polytonic_ratio",
    ]:
        assert key in metadata
    assert metadata["filter"] == "ok"


def test_jsonl_export_includes_source_metadata(tmp_path):
    corpus = Corpus(input_dir=tmp_path / "in4", output_dir=tmp_path / "out4")

    markdown = corpus.cleaned_markdown_dir / "alpha.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("Document text payload", encoding="utf-8")

    pipeline_rows = [
        {
            "filename": "alpha.pdf",
            "filter": "ok",
            "greek_badness_score": 1.23,
            "is_empty": False,
            "latin_percentage": 4.56,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "percentage_greek": 92.1,
            "polytonic_ratio": 0.12,
        }
    ]
    _write_download_results(
        corpus.output_dir / "download_results" / "download_results.parquet",
        pipeline_rows,
    )

    source_rows = [
        {
            "filename": "alpha.pdf",
            "language": "el",
            "handle_url": "https://example.com/handle/alpha",
            "date_accepted": "2020-01-02T00:00:00Z",
        }
    ]
    source_path = corpus.output_dir / "source" / "source.parquet"
    _write_source_metadata(source_path, source_rows)

    out_path = corpus.output_dir / "alpha.jsonl"
    corpus.jsonl(
        out_path,
        text_key="text",
        metadata_key="pipeline_metadata",
        metadata_fields=[
            "filter",
            "greek_badness_score",
            "is_empty",
            "latin_percentage",
            "mojibake_badness_score",
            "needs_ocr",
            "percentage_greek",
            "polytonic_ratio",
        ],
        include_remaining_metadata=False,
        metadata_path=corpus.output_dir / "download_results" / "download_results.parquet",
        source_metadata_key="source_metadata",
        source_metadata_fields=["filename", "language", "handle_url", "date_accepted"],
        source_metadata_path=source_path,
    )

    record = json.loads(out_path.read_text(encoding="utf-8").strip())

    assert record["filename"] == "alpha"
    assert set(record.keys()) == {"text", "pipeline_metadata", "source_metadata", "filename", "doc_id", "chunk_id"}

    source_meta = record["source_metadata"]
    assert source_meta["filename"] == "alpha.pdf"
    assert source_meta["language"] == "el"
    assert source_meta["handle_url"] == "https://example.com/handle/alpha"
    assert source_meta["date_accepted"] == "2020-01-02T00:00:00Z"
    assert record["doc_id"] == _expected_doc_id("alpha.pdf")
    assert record["chunk_id"] == 0


def test_jsonl_export_raises_when_source_metadata_missing_value(tmp_path):
    corpus = Corpus(input_dir=tmp_path / "in5", output_dir=tmp_path / "out5")

    markdown = corpus.cleaned_markdown_dir / "beta.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("Document text payload", encoding="utf-8")

    _write_download_results(
        corpus.output_dir / "download_results" / "download_results.parquet",
        [
            {
                "filename": "beta.pdf",
                "filter": "ok",
                "greek_badness_score": 0.0,
                "is_empty": False,
                "latin_percentage": 1.0,
                "mojibake_badness_score": 0.0,
                "needs_ocr": False,
                "percentage_greek": 99.0,
                "polytonic_ratio": 0.01,
            }
        ],
    )

    source_rows = [
        {
            "filename": "beta.pdf",
            "language": "el",
            "handle_url": None,
            "date_accepted": "2020-01-02T00:00:00Z",
        }
    ]
    source_path = corpus.output_dir / "source" / "source.parquet"
    _write_source_metadata(source_path, source_rows)

    with pytest.raises(ValueError, match="handle_url"):
        corpus.jsonl(
            corpus.output_dir / "beta.jsonl",
            text_key="text",
            metadata_key="pipeline_metadata",
            metadata_fields=[
                "filter",
                "greek_badness_score",
                "is_empty",
                "latin_percentage",
                "mojibake_badness_score",
                "needs_ocr",
                "percentage_greek",
                "polytonic_ratio",
            ],
            include_remaining_metadata=False,
            metadata_path=corpus.output_dir / "download_results" / "download_results.parquet",
            source_metadata_key="source_metadata",
            source_metadata_fields=["filename", "language", "handle_url", "date_accepted"],
            source_metadata_path=source_path,
        )


def test_jsonl_export_sharded(tmp_path):
    corpus = Corpus(input_dir=tmp_path / "in6", output_dir=tmp_path / "out6")

    texts = {
        "doc1": "Alpha text",
        "doc2": "Beta text",
        "doc3": "Gamma text",
    }
    for stem, content in texts.items():
        md_path = corpus.cleaned_markdown_dir / f"{stem}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(content, encoding="utf-8")

    pipeline_rows = [
        {
            "filename": f"{stem}.pdf",
            "filter": "ok",
            "greek_badness_score": float(idx),
            "is_empty": False,
            "latin_percentage": 1.0,
            "mojibake_badness_score": 0.0,
            "needs_ocr": False,
            "percentage_greek": 99.0,
            "polytonic_ratio": 0.0,
        }
        for idx, stem in enumerate(texts, start=1)
    ]
    _write_download_results(
        corpus.output_dir / "download_results" / "download_results.parquet",
        pipeline_rows,
    )

    shard_dir = corpus.output_dir / "export"
    shards = corpus.jsonl_sharded(
        shard_dir,
        shard_size_bytes=50,
        text_key="text",
        metadata_key="pipeline_metadata",
        metadata_fields=[
            "filter",
            "greek_badness_score",
            "is_empty",
            "latin_percentage",
            "mojibake_badness_score",
            "needs_ocr",
            "percentage_greek",
            "polytonic_ratio",
        ],
        include_remaining_metadata=False,
        metadata_path=corpus.output_dir / "download_results" / "download_results.parquet",
    )

    assert len(shards) >= 2
    seen_doc_ids = set()
    decompressor = zstd.ZstdDecompressor()

    for shard_path in shards:
        assert shard_path.suffixes[-2:] == [".jsonl", ".zst"]
        with shard_path.open("rb") as fh:
            data = b"".join(decompressor.read_to_iter(fh))
        for line in data.decode("utf-8").strip().splitlines():
            record = json.loads(line)
            assert record["chunk_id"] == 0
            seen_doc_ids.add(record["doc_id"])

    assert len(seen_doc_ids) == len(texts)


@pytest.mark.skipif(not _HAS_DATASETS, reason="datasets package is not installed")
def test_hf_streaming_loader_example(tmp_path):
    corpus = Corpus(input_dir=tmp_path / "in7", output_dir=tmp_path / "out7")

    markdown = corpus.cleaned_markdown_dir / "gamma.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("Stream me", encoding="utf-8")

    _write_download_results(
        corpus.output_dir / "download_results" / "download_results.parquet",
        [
            {
                "filename": "gamma.pdf",
                "filter": "ok",
                "greek_badness_score": 0.0,
                "is_empty": False,
                "latin_percentage": 0.2,
                "mojibake_badness_score": 0.0,
                "needs_ocr": False,
                "percentage_greek": 99.8,
                "polytonic_ratio": 0.0,
            }
        ],
    )

    shards = corpus.jsonl_sharded(
        corpus.output_dir / "stream_shards",
        shard_size_bytes=1024,
        text_key="text",
        metadata_key="pipeline_metadata",
        metadata_fields=[
            "filter",
            "greek_badness_score",
            "is_empty",
            "latin_percentage",
            "mojibake_badness_score",
            "needs_ocr",
            "percentage_greek",
            "polytonic_ratio",
        ],
        include_remaining_metadata=False,
        metadata_path=corpus.output_dir / "download_results" / "download_results.parquet",
    )

    assert len(shards) == 1
    stream = datasets.load_dataset(  # type: ignore[attr-defined]
        "json",
        data_files={"train": str(shards[0])},
        streaming=True,
    )["train"]
    first = next(iter(stream))
    assert first["text"] == "Stream me"
    assert first["pipeline_metadata"]["filter"] == "ok"


def test_pyarrow_filter_example(tmp_path):
    df = pd.DataFrame(
        [
            {"doc_id": "a", "lang": "el", "year": 2020},
            {"doc_id": "b", "lang": "en", "year": 2021},
            {"doc_id": "c", "lang": "el", "year": 2018},
        ]
    )
    parquet_path = tmp_path / "meta" / "meta.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    import pyarrow.dataset as ds

    dataset = ds.dataset(str(parquet_path), format="parquet")
    table = dataset.to_table(filter=(ds.field("lang") == "el") & (ds.field("year") >= 2019))

    assert set(table.column("doc_id").to_pylist()) == {"a"}
def _expected_doc_id(filename: str) -> str:
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()
