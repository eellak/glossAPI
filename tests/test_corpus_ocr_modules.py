from pathlib import Path

import pandas as pd

from glossapi import Corpus
from glossapi.corpus.ocr.artifacts import apply_ocr_success_updates
from glossapi.corpus.ocr.config import normalize_ocr_request
from glossapi.corpus.ocr.targets import build_ocr_selection
from glossapi.ocr.deepseek.defaults import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_RENDER_DPI


def _mk_corpus(tmp_path: Path) -> Corpus:
    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_normalize_ocr_request_uses_shared_vllm_defaults(tmp_path):
    corpus = _mk_corpus(tmp_path)

    request = normalize_ocr_request(
        logger=corpus.logger,
        fix_bad=True,
        mode="ocr_bad",
        backend="deepseek",
        device=None,
        model_dir=None,
        max_pages=None,
        persist_engine=True,
        precision=None,
        runtime_backend="vllm",
        render_dpi=None,
        gpu_memory_utilization=None,
        math_enhance=False,
        force=None,
        reprocess_completed=None,
        skip_existing=None,
    )

    assert request is not None
    assert request.render_dpi == DEFAULT_RENDER_DPI
    assert request.gpu_memory_utilization == DEFAULT_GPU_MEMORY_UTILIZATION


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
        backend_norm="deepseek",
    ).set_index("filename")

    assert bool(updated.loc["needs.pdf", "ocr_success"]) is True
    assert bool(updated.loc["needs__p0001-0002.pdf", "ocr_success"]) is True
    assert updated.loc["needs.pdf", "text"] == "fixed markdown\n"
    assert updated.loc["needs__p0001-0002.pdf", "text"] == "fixed markdown\n"
    assert updated.loc["needs.pdf", "ocr_markdown_relpath"] == "markdown/needs.md"
    assert updated.loc["needs__p0001-0002.pdf", "ocr_metrics_relpath"] == "json/metrics/needs.metrics.json"
    assert updated.loc["needs.pdf", "extraction_mode"] == "deepseek"
