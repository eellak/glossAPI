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
        page_ranges=None,
        model_dir=tmp_path / "model",
        python_bin=Path("/usr/bin/python3"),
        script=tmp_path / "run.py",
        max_pages=1,
        content_debug=False,
        device="cuda",
        ocr_profile="plain_ocr",
        prompt_override="custom prompt",
        attn_backend="flash_attention_2",
        base_size=768,
        image_size=512,
        crop_mode=True,
        render_dpi=144,
        max_new_tokens=1024,
        repetition_penalty=1.05,
        no_repeat_ngram_size=12,
        runtime_backend="transformers",
        vllm_batch_size=None,
        gpu_memory_utilization=None,
        disable_fp8_kv=False,
        repair_mode=None,
    )

    assert "--ocr-profile" in cmd and "plain_ocr" in cmd
    assert "--prompt-override" in cmd and "custom prompt" in cmd
    assert "--attn-backend" in cmd and "flash_attention_2" in cmd
    assert "--base-size" in cmd and "768" in cmd
    assert "--image-size" in cmd and "512" in cmd
    assert "--crop-mode" in cmd
    assert "--render-dpi" in cmd and "144" in cmd
    assert "--max-new-tokens" in cmd and "1024" in cmd


def test_deepseek_default_max_new_tokens_is_standardized():
    from glossapi.ocr.deepseek import runner
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import DEFAULT_MAX_NEW_TOKENS

    assert DEFAULT_MAX_NEW_TOKENS == 2048
    assert runner.DEFAULT_MAX_NEW_TOKENS == 2048


def test_build_cli_command_includes_vllm_flags(tmp_path):
    from glossapi.ocr.deepseek.runner import _build_cli_command

    cmd = _build_cli_command(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        files=["a.pdf"],
        page_ranges=None,
        model_dir=tmp_path / "model",
        python_bin=Path("/usr/bin/python3"),
        script=tmp_path / "run_vllm.py",
        max_pages=1,
        content_debug=False,
        device="cuda",
        ocr_profile="markdown_grounded",
        prompt_override=None,
        attn_backend="auto",
        base_size=None,
        image_size=None,
        crop_mode=None,
        render_dpi=110,
        max_new_tokens=768,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        runtime_backend="vllm",
        vllm_batch_size=16,
        gpu_memory_utilization=0.92,
        disable_fp8_kv=True,
        repair_mode="auto",
    )

    assert "--batch-size" in cmd and "16" in cmd
    assert "--gpu-memory-utilization" in cmd and "0.92" in cmd
    assert "--disable-fp8-kv" in cmd
    assert "--repair-mode" in cmd and "auto" in cmd


def test_build_cli_command_includes_page_ranges(tmp_path):
    from glossapi.ocr.deepseek.runner import _build_cli_command

    cmd = _build_cli_command(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        files=[],
        page_ranges=["a.pdf:1:64", "b.pdf:65:128"],
        model_dir=tmp_path / "model",
        python_bin=Path("/usr/bin/python3"),
        script=tmp_path / "run_vllm.py",
        max_pages=None,
        content_debug=False,
        device="cuda",
        ocr_profile="markdown_grounded",
        prompt_override=None,
        attn_backend="auto",
        base_size=None,
        image_size=None,
        crop_mode=None,
        render_dpi=144,
        max_new_tokens=1024,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        runtime_backend="vllm",
        vllm_batch_size=32,
        gpu_memory_utilization=0.9,
        disable_fp8_kv=False,
        repair_mode="auto",
    )

    assert "--page-ranges" in cmd
    assert "a.pdf:1:64" in cmd
    assert "b.pdf:65:128" in cmd


def test_vllm_empty_page_detector_is_conservative():
    from glossapi.ocr.deepseek.run_pdf_ocr_vllm import _is_effectively_empty_page

    empty_page = {
        "top_dark_ratio": 0.0004,
        "bottom_dark_ratio": 0.0006,
        "top_third_dark_ratio": 0.0002,
        "middle_third_dark_ratio": 0.0005,
        "bottom_third_dark_ratio": 0.0007,
        "overall_dark_ratio": 0.0008,
    }
    non_empty_sparse_page = {
        "top_dark_ratio": 0.003,
        "bottom_dark_ratio": 0.004,
        "top_third_dark_ratio": 0.0028,
        "middle_third_dark_ratio": 0.0031,
        "bottom_third_dark_ratio": 0.0042,
        "overall_dark_ratio": 0.0022,
    }
    assert _is_effectively_empty_page(empty_page, "auto") is True
    assert _is_effectively_empty_page(non_empty_sparse_page, "auto") is False
    assert _is_effectively_empty_page(empty_page, "off") is False


def test_early_stop_detects_symbol_and_numeric_list_garbage():
    from glossapi.ocr.utils.cleaning import detect_early_stop_index

    symbol_garbage = "Κανονικό κείμενο\n" + (" " * 20)
    numeric_list_garbage = "Πρόλογος\n" + " ".join(f"{idx}." for idx in range(1, 20))

    symbol_cut = detect_early_stop_index(symbol_garbage)
    numeric_cut = detect_early_stop_index(numeric_list_garbage)

    assert symbol_cut is not None
    assert "Κανονικό κείμενο" in symbol_garbage[:symbol_cut]
    assert numeric_cut is not None
    assert "Πρόλογος" in numeric_list_garbage[:numeric_cut]


def test_runner_selects_vllm_script_when_requested(tmp_path, monkeypatch):
    from glossapi.ocr.deepseek import runner

    corpus = _mk_corpus(tmp_path)
    (corpus.input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%real\n")

    calls = {}

    def fake_run_cli(input_dir, output_dir, **kwargs):
        calls["script"] = kwargs["script"]
        calls["runtime_backend"] = kwargs["runtime_backend"]
        md_dir = output_dir / "markdown"
        metrics_dir = output_dir / "json" / "metrics"
        md_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / "doc.md").write_text("ok\n", encoding="utf-8")
        (metrics_dir / "doc.metrics.json").write_text('{"page_count": 1}', encoding="utf-8")

    monkeypatch.setattr(runner, "_run_cli", fake_run_cli)
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_PYTHON", sys.executable)

    result = runner.run_for_files(corpus, ["doc.pdf"], runtime_backend="vllm")

    assert calls["runtime_backend"] == "vllm"
    assert Path(calls["script"]).name == "run_pdf_ocr_vllm.py"
    assert result["doc"]["page_count"] == 1


def test_runner_forwards_scheduler_controls_to_multi_cli(tmp_path, monkeypatch):
    from glossapi.ocr.deepseek import runner

    corpus = _mk_corpus(tmp_path)
    (corpus.input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%real\n")

    calls = {}

    def fake_run_multi_cli(**kwargs):
        calls.update(kwargs)
        md_dir = kwargs["out_root"] / "markdown"
        metrics_dir = kwargs["out_root"] / "json" / "metrics"
        md_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / "doc.md").write_text("ok\n", encoding="utf-8")
        (metrics_dir / "doc.metrics.json").write_text('{"page_count": 1}', encoding="utf-8")

    monkeypatch.setattr(runner, "_run_multi_cli", fake_run_multi_cli)
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_PYTHON", sys.executable)

    result = runner.run_for_files(
        corpus,
        ["doc.pdf"],
        runtime_backend="vllm",
        use_gpus="multi",
        devices=[0, 1],
        scheduler="exact_fill",
        target_batch_pages=196,
        shard_pages=64,
        shard_threshold_pages=256,
    )

    assert calls["scheduler"] == "exact_fill"
    assert calls["target_batch_pages"] == 196
    assert calls["shard_pages"] == 64
    assert calls["shard_threshold_pages"] == 256
    assert result["doc"]["page_count"] == 1
