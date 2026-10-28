import json
from pathlib import Path

import pandas as pd

from glossapi import Corpus
from glossapi.corpus.ocr.artifacts import apply_ocr_success_updates
from glossapi.corpus.ocr.config import normalize_ocr_request
from glossapi.corpus.ocr.targets import build_ocr_selection


def _mk_corpus(tmp_path: Path) -> Corpus:
    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_normalize_ocr_request_preserves_plugin_configuration(tmp_path):
    corpus = _mk_corpus(tmp_path)
    options = object()
    request = normalize_ocr_request(
        logger=corpus.logger, fix_bad=True, mode="ocr_bad", backend="docling",
        ocr_options=options, device="cpu", max_pages=None, math_enhance=False,
    )
    assert request.ocr_options is options
    assert request.mode == "ocr_bad"


def test_build_ocr_selection_collapses_chunk_rows_and_skips_completed(tmp_path):
    corpus = _mk_corpus(tmp_path)
    (corpus.input_dir / "needs.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"filename": "needs.pdf", corpus.url_column: "", "needs_ocr": True, "ocr_success": False},
            {"filename": "needs__p0001-0002.pdf", corpus.url_column: "", "needs_ocr": True, "ocr_success": False},
            {"filename": "done.pdf", corpus.url_column: "", "needs_ocr": True, "ocr_success": True},
        ]
    ).to_parquet(dl_dir / "download_results.parquet", index=False)

    selection = build_ocr_selection(
        corpus,
        mode="ocr_bad",
        reprocess_completed=False,
    )

    assert selection.bad_files == ["needs.pdf"]
    assert selection.ocr_candidates_initial == 2
    assert selection.skipped_completed == 1
    assert selection.skipped_skiplist == 0
    assert selection.ocr_done_stems == {"done"}


def test_apply_ocr_success_updates_maps_canonical_artifacts_by_stem(tmp_path):
    markdown_dir = tmp_path / "markdown"
    metrics_dir = tmp_path / "json" / "metrics"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (markdown_dir / "needs.md").write_text("fixed markdown\n", encoding="utf-8")
    (metrics_dir / "needs.metrics.json").write_text('{"page_count": 1}\n', encoding="utf-8")

    df = pd.DataFrame(
        [
            {"filename": "needs.pdf", "needs_ocr": True, "ocr_success": False},
            {"filename": "needs__p0001-0002.pdf", "needs_ocr": True, "ocr_success": False},
        ]
    )

    updated = apply_ocr_success_updates(
        df,
        filenames=["needs.pdf"],
        markdown_dir=markdown_dir,
        metrics_dir=metrics_dir,
        backend_norm="docling",
    ).set_index("filename")

    assert bool(updated.loc["needs.pdf", "ocr_success"]) is True
    assert bool(updated.loc["needs__p0001-0002.pdf", "ocr_success"]) is True
    assert updated.loc["needs.pdf", "text"] == "fixed markdown\n"
    assert updated.loc["needs__p0001-0002.pdf", "text"] == "fixed markdown\n"
    assert updated.loc["needs.pdf", "ocr_markdown_relpath"] == "markdown/needs.md"
    assert updated.loc["needs__p0001-0002.pdf", "ocr_metrics_relpath"] == "json/metrics/needs.metrics.json"
    assert updated.loc["needs.pdf", "extraction_mode"] == "docling"


def test_ocr_pipeline_exports_cleaned_and_raw_text_side_by_side(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "filename": "doc.pdf",
                corpus.url_column: "https://example.com/doc.pdf",
                "needs_ocr": True,
                "ocr_success": False,
            }
        ]
    ).to_parquet(dl_dir / "download_results.parquet", index=False)
    (corpus.input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

    raw_markdown = (
        "Κανονική πρώτη σελίδα.\n"
        "<--- Page Split --->\n"
        "1. 2. 3. 4. 5. 6. 7.\n"
        "0 0 0 0 0 0\n"
        "1.1\n1.1\n1.1\n1.1\n1.1\n1.1\n"
    )

    from glossapi.corpus.ocr import dispatch as runner

    def fake_run_for_files(self_ref, files, **kwargs):
        markdown_dir = self_ref.output_dir / "markdown"
        metrics_dir = self_ref.output_dir / "json" / "metrics"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        for current in files:
            stem = Path(current).stem
            (markdown_dir / f"{stem}.md").write_text(raw_markdown, encoding="utf-8")
            (metrics_dir / f"{stem}.metrics.json").write_text(
                json.dumps(
                    {
                        "page_count": 2,
                        "pages": [
                            {"page_no": 1, "formula_count": 0, "code_count": 0},
                            {"page_no": 2, "formula_count": 0, "code_count": 0},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return {"doc": {"page_count": 2}}

    monkeypatch.setattr(runner, "run_plugin_ocr", fake_run_for_files)

    calls = []
    original_clean_ocr = corpus.clean_ocr
    original_clean = corpus.clean
    original_markdown_dir = corpus.markdown_dir
    original_cleaned_markdown_dir = corpus.cleaned_markdown_dir

    def record_clean_ocr(*args, **kwargs):
        calls.append(
            (
                "clean_ocr",
                Path(str(kwargs.get("input_dir"))),
                kwargs.get("write_cleaned_files", True),
            )
        )
        return original_clean_ocr(*args, **kwargs)

    def record_clean(*args, **kwargs):
        calls.append(
            (
                "clean",
                Path(str(kwargs.get("input_dir"))),
                kwargs.get("write_cleaned_files", True),
            )
        )
        return original_clean(*args, **kwargs)

    monkeypatch.setattr(corpus, "clean_ocr", record_clean_ocr)
    monkeypatch.setattr(corpus, "clean", record_clean)

    corpus.ocr(backend="docling", fix_bad=True, math_enhance=False)

    assert calls[0] == ("clean_ocr", original_markdown_dir, True)
    assert calls[1] == ("clean", original_cleaned_markdown_dir, False)

    raw_text = (original_markdown_dir / "doc.md").read_text(encoding="utf-8")
    cleaned_text = (original_cleaned_markdown_dir / "doc.md").read_text(encoding="utf-8")
    assert raw_text == raw_markdown
    assert cleaned_text != raw_text
    assert "1.1\n1.1" in raw_text
    assert "1.1\n1.1" not in cleaned_text

    out_path = corpus.output_dir / "export.jsonl"
    corpus.jsonl(out_path)
    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    record = records[0]

    assert record["document"] == cleaned_text
    assert record["text"] == raw_text
    assert record["filename"] == "doc"
    assert record["url"] == "https://example.com/doc.pdf"
    assert record["ocr_success"] is True
    assert record["extraction_mode"] == "docling"
    assert record["page_count"] == 2


def test_math_only_forwards_page_routing_to_upstream_selection(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from glossapi.corpus import phase_ocr_math

    corpus = _mk_corpus(tmp_path)
    calls = []

    def select(context, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            bad_files=[], math_done_stems=set(), skip_mgr=None,
            skiplist_path=tmp_path / "skiplist.txt",
        )

    monkeypatch.setattr(phase_ocr_math, "build_ocr_selection", select)
    monkeypatch.setattr(phase_ocr_math, "discover_docling_json_stems", lambda root: [])
    monkeypatch.setattr(corpus, "_run_math_targets", lambda **kwargs: None)
    corpus.ocr(mode="math_only", page_routing="all", fix_bad=False)

    assert calls == [dict(mode="math_only", page_routing="all", reprocess_completed=False)]
