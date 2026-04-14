import os
from pathlib import Path

import pandas as pd
import pytest


def _mk_corpus(tmp_path: Path):
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_deepseek_backend_ignores_math_flags_and_runs_ocr_only(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    # Seed metadata with one bad file
    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    df = pd.DataFrame([
        {"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}
    ])
    df.to_parquet(dl_dir / "download_results.parquet", index=False)

    # Create stub pdf
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%stub\n")

    # Capture deepseek runner calls and assert math is not invoked
    from glossapi.ocr.deepseek import runner

    calls = {}

    def fake_run_for_files(self_ref, files, **kwargs):
        calls["files"] = list(files)
        return {"doc": {"page_count": 1}}

    monkeypatch.setattr(runner, "run_for_files", fake_run_for_files)

    # Also ensure formula_enrich_from_json would raise if called
    def fail_math(*args, **kwargs):
        raise AssertionError("Phase-2 math should not run for DeepSeek OCR targets")

    monkeypatch.setattr(corpus, "formula_enrich_from_json", fail_math)

    # Run with math flags present — should still just OCR the bad file
    corpus.ocr(backend="deepseek", fix_bad=True, math_enhance=True, mode="ocr_bad_then_math")

    assert calls.get("files") == [fname]


def test_deepseek_backend_forwards_repair_exec_batch_controls(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    df = pd.DataFrame([
        {"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}
    ])
    df.to_parquet(dl_dir / "download_results.parquet", index=False)
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%stub\n")

    from glossapi.ocr.deepseek import runner

    calls = {}

    def fake_run_for_files(self_ref, files, **kwargs):
        calls["files"] = list(files)
        calls["kwargs"] = dict(kwargs)
        return {"doc": {"page_count": 1}}

    monkeypatch.setattr(runner, "run_for_files", fake_run_for_files)

    corpus.ocr(
        backend="deepseek",
        fix_bad=True,
        math_enhance=False,
        mode="ocr_bad",
        repair_exec_batch_target_pages=64,
        repair_exec_batch_target_items=24,
    )

    assert calls.get("files") == [fname]
    assert calls["kwargs"]["repair_exec_batch_target_pages"] == 64
    assert calls["kwargs"]["repair_exec_batch_target_items"] == 24


def test_invalid_backend_is_rejected(tmp_path):
    corpus = _mk_corpus(tmp_path)
    with pytest.raises(ValueError, match="backend must be 'deepseek'"):
        corpus.ocr(backend="bogus", fix_bad=True, math_enhance=False)


def test_deepseek_backend_forwards_parallelism_controls(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    pd.DataFrame(
        [{"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}]
    ).to_parquet(dl_dir / "download_results.parquet", index=False)
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%stub\n")

    from glossapi.ocr.deepseek import runner

    calls = {}

    def fake_run_for_files(self_ref, files, **kwargs):
        calls["files"] = list(files)
        calls["kwargs"] = dict(kwargs)
        return {"doc": {"page_count": 1}}

    monkeypatch.setattr(runner, "run_for_files", fake_run_for_files)

    corpus.ocr(
        backend="deepseek",
        fix_bad=True,
        math_enhance=False,
        use_gpus="multi",
        devices=[1, 3],
        workers_per_gpu=2,
        ocr_profile="plain_ocr",
        attn_backend="sdpa",
        base_size=640,
        image_size=448,
        crop_mode=True,
        render_dpi=120,
        max_pages=7,
        max_new_tokens=2048,
        repetition_penalty=1.08,
        no_repeat_ngram_size=12,
    )

    assert calls["files"] == [fname]
    assert calls["kwargs"]["use_gpus"] == "multi"
    assert calls["kwargs"]["devices"] == [1, 3]
    assert calls["kwargs"]["workers_per_gpu"] == 2
    assert calls["kwargs"]["ocr_profile"] == "plain_ocr"
    assert calls["kwargs"]["attn_backend"] == "sdpa"
    assert calls["kwargs"]["base_size"] == 640
    assert calls["kwargs"]["image_size"] == 448
    assert calls["kwargs"]["crop_mode"] is True
    assert calls["kwargs"]["render_dpi"] == 120
    assert calls["kwargs"]["max_pages"] == 7
    assert calls["kwargs"]["max_new_tokens"] == 2048
    assert calls["kwargs"]["repetition_penalty"] == 1.08
    assert calls["kwargs"]["no_repeat_ngram_size"] == 12


def test_deepseek_rerun_refreshes_with_clean_ocr_then_score_only_clean(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    pd.DataFrame(
        [{"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}]
    ).to_parquet(dl_dir / "download_results.parquet", index=False)
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%stub\n")

    from glossapi.ocr.deepseek import runner

    def fake_run_for_files(self_ref, files, **kwargs):
        markdown_dir = self_ref.output_dir / "markdown"
        metrics_dir = self_ref.output_dir / "json" / "metrics"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        for current in files:
            stem = Path(current).stem
            (markdown_dir / f"{stem}.md").write_text("ocr text\n", encoding="utf-8")
            (metrics_dir / f"{stem}.metrics.json").write_text(
                "{\"page_count\": 1}\n",
                encoding="utf-8",
            )
        return {"doc": {"page_count": 1}}

    monkeypatch.setattr(runner, "run_for_files", fake_run_for_files)

    calls = []
    original_markdown_dir = corpus.markdown_dir

    def fake_clean_ocr(*args, **kwargs):
        calls.append(
            ("clean_ocr", kwargs.get("input_dir"), kwargs.get("write_cleaned_files", True))
        )
        corpus.cleaned_markdown_dir.mkdir(parents=True, exist_ok=True)
        corpus.markdown_dir = corpus.cleaned_markdown_dir

    def fake_clean(*args, **kwargs):
        calls.append(("clean", kwargs.get("input_dir"), kwargs.get("write_cleaned_files", True)))

    monkeypatch.setattr(corpus, "clean_ocr", fake_clean_ocr)
    monkeypatch.setattr(corpus, "clean", fake_clean)

    corpus.ocr(backend="deepseek", fix_bad=True, math_enhance=False)

    assert calls[0][0] == "clean_ocr"
    assert Path(str(calls[0][1])) == original_markdown_dir
    assert calls[0][2] is True
    assert calls[1][0] == "clean"
    assert Path(str(calls[1][1])) == corpus.cleaned_markdown_dir
    assert calls[1][2] is False
