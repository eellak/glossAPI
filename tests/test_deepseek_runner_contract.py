import sys
from pathlib import Path

import pandas as pd
import pytest


def _mk_corpus(tmp_path: Path):
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_deepseek_backend_rejects_stub_mode(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    df = pd.DataFrame(
        [{"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}]
    )
    parquet_path = dl_dir / "download_results.parquet"
    df.to_parquet(parquet_path, index=False)
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%real\n")

    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "1")

    with pytest.raises(RuntimeError, match="stub execution has been removed"):
        corpus.ocr(backend="deepseek", fix_bad=True, math_enhance=False)

    updated = pd.read_parquet(parquet_path).set_index("filename")
    assert bool(updated.loc[fname, "ocr_success"]) is False
    assert bool(updated.loc[fname, "needs_ocr"]) is True


def test_progress_artifacts_stay_out_of_canonical_markdown(tmp_path):
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _write_outputs, _write_progress

    output_dir = tmp_path / "output"
    _write_progress(
        output_dir=output_dir,
        stem="doc",
        page_outputs=["page one"],
        total_pages=5,
        completed_pages=1,
    )

    canonical_markdown = output_dir / "markdown" / "doc.md"
    progress_markdown = output_dir / "sidecars" / "ocr_progress" / "doc.partial.md"
    progress_json = output_dir / "json" / "metrics" / "doc.progress.json"

    assert not canonical_markdown.exists()
    assert progress_markdown.exists()
    assert progress_json.exists()

    _write_outputs(output_dir=output_dir, stem="doc", markdown="final", page_count=5)

    assert canonical_markdown.exists()
    assert canonical_markdown.read_text(encoding="utf-8") == "final\n"
    assert not progress_markdown.exists()


def test_auto_attn_backend_prefers_eager_when_flash_attn_is_unavailable(monkeypatch):
    import builtins

    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _resolve_attn_backend

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flash_attn":
            raise ImportError("flash_attn unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _resolve_attn_backend("auto") == "eager"


def test_runner_uses_downloads_subdir_when_present(tmp_path, monkeypatch):
    from glossapi.ocr.deepseek import runner

    corpus = _mk_corpus(tmp_path)
    downloads_dir = corpus.input_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    (downloads_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%real\n")

    calls = {}

    def fake_run_cli(input_dir, output_dir, **kwargs):
        calls["input_dir"] = input_dir
        md_dir = output_dir / "markdown"
        metrics_dir = output_dir / "json" / "metrics"
        md_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / "doc.md").write_text("ok\n", encoding="utf-8")
        (metrics_dir / "doc.metrics.json").write_text('{"page_count": 1}', encoding="utf-8")

    monkeypatch.setattr(runner, "_run_cli", fake_run_cli)
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv(
        "GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT",
        str(Path(runner.__file__).resolve().parent / "run_pdf_ocr_transformers.py"),
    )
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_PYTHON", sys.executable)

    result = runner.run_for_files(corpus, ["doc.pdf"])

    assert calls["input_dir"] == downloads_dir.resolve()
    assert result["doc"]["page_count"] == 1


def test_build_cli_command_includes_speed_flags(tmp_path):
    from glossapi.ocr.deepseek.runner import _build_cli_command

    cmd = _build_cli_command(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        files=["a.pdf"],
        model_dir=tmp_path / "model",
        python_bin=Path("/usr/bin/python3"),
        script=tmp_path / "run.py",
        max_pages=1,
        content_debug=False,
        device="cuda",
        ocr_profile="plain_ocr",
        attn_backend="flash_attention_2",
        base_size=768,
        image_size=512,
        crop_mode=True,
        render_dpi=144,
        max_new_tokens=1024,
        repetition_penalty=1.05,
        no_repeat_ngram_size=12,
    )

    assert "--ocr-profile" in cmd and "plain_ocr" in cmd
    assert "--attn-backend" in cmd and "flash_attention_2" in cmd
    assert "--base-size" in cmd and "768" in cmd
    assert "--image-size" in cmd and "512" in cmd
    assert "--crop-mode" in cmd
    assert "--render-dpi" in cmd and "144" in cmd
    assert "--max-new-tokens" in cmd and "1024" in cmd
