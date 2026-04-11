import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_page_output_helpers_roundtrip_numbered_blank_pages():
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _join_page_outputs, _split_page_outputs

    page_outputs = ["page one", "", "page three"]

    markdown = _join_page_outputs(page_outputs)

    assert markdown == (
        "page one\n"
        "<!-- page:2 -->\n"
        "<--- Page Split --->\n"
        "\n"
        "<!-- page:3 -->\n"
        "<--- Page Split --->\n"
        "page three"
    )
    assert _split_page_outputs(markdown) == page_outputs


def test_write_outputs_preserves_blank_first_page_structure(tmp_path):
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _join_page_outputs, _split_page_outputs, _write_outputs

    output_dir = tmp_path / "output"
    markdown = _join_page_outputs(["", "page two"])

    _write_outputs(output_dir=output_dir, stem="doc", markdown=markdown, page_count=2)

    written = (output_dir / "markdown" / "doc.md").read_text(encoding="utf-8")
    assert written == (
        "<!-- page:2 -->\n"
        "<--- Page Split --->\n"
        "page two\n"
    )
    assert _split_page_outputs(written) == ["", "page two"]


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


def test_runner_resolves_standard_vllm_defaults_when_omitted(tmp_path, monkeypatch):
    from glossapi.ocr.deepseek import runner
    from glossapi.ocr.deepseek.defaults import DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_RENDER_DPI

    corpus = _mk_corpus(tmp_path)
    (corpus.input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%real\n")

    calls = {}

    def fake_run_cli(input_dir, output_dir, **kwargs):
        calls["input_dir"] = input_dir
        calls["kwargs"] = dict(kwargs)
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
        str(Path(runner.__file__).resolve().parent / "run_pdf_ocr_vllm.py"),
    )
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_PYTHON", sys.executable)

    runner.run_for_files(
        corpus,
        ["doc.pdf"],
        runtime_backend="vllm",
        render_dpi=None,
        gpu_memory_utilization=None,
    )

    assert calls["input_dir"] == corpus.input_dir.resolve()
    assert calls["kwargs"]["render_dpi"] == DEFAULT_RENDER_DPI
    assert calls["kwargs"]["gpu_memory_utilization"] == DEFAULT_GPU_MEMORY_UTILIZATION


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


def test_build_env_prepends_script_src_to_pythonpath(tmp_path, monkeypatch):
    import os

    from glossapi.ocr.deepseek.runner import _build_env

    repo_root = tmp_path / "repo"
    script = repo_root / "src" / "glossapi" / "ocr" / "deepseek" / "run_pdf_ocr_vllm.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# stub\n", encoding="utf-8")
    (repo_root / "src" / "glossapi").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/tmp/old-a", "/tmp/old-b"]))
    env = _build_env(
        python_bin=Path("/usr/bin/python3"),
        visible_device=1,
        script=script,
    )

    assert env["PYTHONPATH"].split(os.pathsep) == [
        str((repo_root / "src").resolve()),
        "/tmp/old-a",
        "/tmp/old-b",
    ]
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


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


def test_repair_disposition_drops_repeat_garbage_cutoff():
    from glossapi.ocr.deepseek.run_pdf_ocr_vllm import _resolve_repair_disposition

    disposition = _resolve_repair_disposition(
        repair_text="garbage",
        repair_postprocess={"early_stops": 1},
    )

    assert disposition == {
        "final_text": "",
        "repair_applied": False,
        "page_dropped_after_repair": True,
        "drop_reason": "repeat_garbage_cutoff",
    }


def test_repair_batch_updates_persisted_outputs_with_repeat_cutoff_drop(tmp_path, monkeypatch):
    from PIL import Image

    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _join_page_outputs, _split_page_outputs, _write_outputs
    from glossapi.ocr.deepseek.run_pdf_ocr_vllm import _run_repair_batch_to_outputs

    output_dir = tmp_path / "output"
    _write_outputs(
        output_dir=output_dir,
        stem="doc",
        markdown=_join_page_outputs(["bad first page", "page two"]),
        page_count=2,
        extra_metrics={
            "repair_mode": "auto",
            "page_metrics": [
                {
                    "page_number": 1,
                    "infer_sec": 1.0,
                    "raw_chars": 20,
                    "final_chars": 14,
                    "repair_strategy": "plain",
                    "repair_reason": "early_stop_markdown_garbage",
                    "repair_attempted": False,
                    "repair_applied": False,
                    "page_dropped_after_repair": False,
                    "empty_page_skipped": False,
                    "garbage_early_stop_applied": True,
                },
                {
                    "page_number": 2,
                    "infer_sec": 0.5,
                    "raw_chars": 8,
                    "final_chars": 8,
                    "repair_strategy": "none",
                    "repair_reason": None,
                    "repair_attempted": False,
                    "repair_applied": False,
                    "page_dropped_after_repair": False,
                    "empty_page_skipped": False,
                    "garbage_early_stop_applied": False,
                },
            ],
            "repair_summary": {"repair_mode": "auto", "pages_flagged": 1, "pages_repaired": 0},
        },
    )

    monkeypatch.setattr(
        "glossapi.ocr.deepseek.run_pdf_ocr_vllm._iter_selected_rendered_pages",
        lambda pdf_path, *, render_dpi, source_page_numbers: [(1, Image.new("RGB", (4, 4), "white"))],
    )
    monkeypatch.setattr(
        "glossapi.ocr.deepseek.run_pdf_ocr_vllm._generate_batch_outputs",
        lambda llm, *, jobs, prompt, batch_size, sampling_params: [
            {"item": jobs[0], "raw_text": "still broken", "infer_sec": 0.25}
        ],
    )
    monkeypatch.setattr(
        "glossapi.ocr.deepseek.run_pdf_ocr_vllm._postprocess_page_text",
        lambda text, *, prompt, content_debug: ("garbage", {"early_stops": 1}),
    )

    result = _run_repair_batch_to_outputs(
        SimpleNamespace(render_dpi=144, batch_size=8, content_debug=False, repair_mode="auto"),
        batch={
            "stem": "doc",
            "pdf_path": str(tmp_path / "doc.pdf"),
            "source_start_page": 1,
            "repair_page_numbers": [1],
        },
        output_dir=output_dir,
        llm=object(),
        plain_prompt="plain prompt",
        sampling_params=object(),
    )

    markdown = (output_dir / "markdown" / "doc.md").read_text(encoding="utf-8")
    metrics = json.loads((output_dir / "json" / "metrics" / "doc.metrics.json").read_text(encoding="utf-8"))

    assert result["pages"] == 1
    assert _split_page_outputs(markdown) == ["", "page two"]
    assert metrics["repair_summary"]["pages_dropped_after_repeat_cutoff"] == 1


def test_repair_batch_pack_updates_multiple_stems(tmp_path, monkeypatch):
    from PIL import Image

    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _join_page_outputs, _split_page_outputs, _write_outputs
    from glossapi.ocr.deepseek.run_pdf_ocr_vllm import _run_repair_batches_to_outputs

    output_dir = tmp_path / "output"
    _write_outputs(
        output_dir=output_dir,
        stem="doc_a",
        markdown=_join_page_outputs(["bad a", "page two a"]),
        page_count=2,
        extra_metrics={
            "repair_mode": "auto",
            "page_metrics": [
                {
                    "page_number": 1,
                    "infer_sec": 1.0,
                    "raw_chars": 10,
                    "final_chars": 5,
                    "repair_strategy": "plain",
                    "repair_reason": "early_stop_markdown_garbage",
                    "repair_attempted": False,
                    "repair_applied": False,
                    "page_dropped_after_repair": False,
                    "empty_page_skipped": False,
                    "garbage_early_stop_applied": True,
                },
                {
                    "page_number": 2,
                    "infer_sec": 0.5,
                    "raw_chars": 9,
                    "final_chars": 9,
                    "repair_strategy": "none",
                    "repair_reason": None,
                    "repair_attempted": False,
                    "repair_applied": False,
                    "page_dropped_after_repair": False,
                    "empty_page_skipped": False,
                    "garbage_early_stop_applied": False,
                },
            ],
            "repair_summary": {"repair_mode": "auto", "pages_flagged": 1, "pages_repaired": 0},
        },
    )
    _write_outputs(
        output_dir=output_dir,
        stem="doc_b",
        markdown=_join_page_outputs(["bad b"]),
        page_count=1,
        extra_metrics={
            "repair_mode": "auto",
            "page_metrics": [
                {
                    "page_number": 1,
                    "infer_sec": 0.7,
                    "raw_chars": 8,
                    "final_chars": 5,
                    "repair_strategy": "plain",
                    "repair_reason": "early_stop_markdown_garbage",
                    "repair_attempted": False,
                    "repair_applied": False,
                    "page_dropped_after_repair": False,
                    "empty_page_skipped": False,
                    "garbage_early_stop_applied": True,
                }
            ],
            "repair_summary": {"repair_mode": "auto", "pages_flagged": 1, "pages_repaired": 0},
        },
    )

    monkeypatch.setattr(
        "glossapi.ocr.deepseek.run_pdf_ocr_vllm._iter_selected_rendered_pages",
        lambda pdf_path, *, render_dpi, source_page_numbers: [
            (page_number, Image.new("RGB", (4, 4), "white")) for page_number in source_page_numbers
        ],
    )
    monkeypatch.setattr(
        "glossapi.ocr.deepseek.run_pdf_ocr_vllm._generate_batch_outputs",
        lambda llm, *, jobs, prompt, batch_size, sampling_params: [
            {"item": job, "raw_text": f"fixed-{job['stem']}-{job['page_number']}", "infer_sec": 0.25}
            for job in jobs
        ],
    )
    monkeypatch.setattr(
        "glossapi.ocr.deepseek.run_pdf_ocr_vllm._postprocess_page_text",
        lambda text, *, prompt, content_debug: (text, {"early_stops": 0}),
    )

    result = _run_repair_batches_to_outputs(
        SimpleNamespace(render_dpi=144, batch_size=8, content_debug=False, repair_mode="auto"),
        batches=[
            {
                "batch_id": 10,
                "stem": "doc_a",
                "pdf_path": str(tmp_path / "doc_a.pdf"),
                "source_start_page": 1,
                "repair_page_numbers": [1],
                "pages": 1,
            },
            {
                "batch_id": 11,
                "stem": "doc_b",
                "pdf_path": str(tmp_path / "doc_b.pdf"),
                "source_start_page": 1,
                "repair_page_numbers": [1],
                "pages": 1,
            },
        ],
        output_dir=output_dir,
        llm=object(),
        plain_prompt="plain prompt",
        sampling_params=object(),
    )

    markdown_a = (output_dir / "markdown" / "doc_a.md").read_text(encoding="utf-8")
    markdown_b = (output_dir / "markdown" / "doc_b.md").read_text(encoding="utf-8")
    metrics_a = json.loads((output_dir / "json" / "metrics" / "doc_a.metrics.json").read_text(encoding="utf-8"))
    metrics_b = json.loads((output_dir / "json" / "metrics" / "doc_b.metrics.json").read_text(encoding="utf-8"))

    assert result["pages"] == 2
    assert result["docs"] == 2
    assert set(result["per_batch_results"]) == {10, 11}
    assert _split_page_outputs(markdown_a)[0] == "fixed-doc_a-1"
    assert _split_page_outputs(markdown_b)[0] == "fixed-doc_b-1"
    assert metrics_a["repair_summary"]["pages_repaired"] == 1
    assert metrics_b["repair_summary"]["pages_repaired"] == 1


def test_vllm_progress_sidecar_keeps_absolute_page_numbers(tmp_path):
    from glossapi.ocr.deepseek.run_pdf_ocr_vllm import _emit_progress

    state = {
        "page_outputs": ["", "page two"],
        "total_pages": 2,
        "completed_pages": 2,
    }

    _emit_progress(tmp_path / "output", "doc", state)

    partial_markdown = (tmp_path / "output" / "sidecars" / "ocr_progress" / "doc.partial.md").read_text(
        encoding="utf-8"
    )
    assert partial_markdown == (
        "<!-- page:2 -->\n"
        "<--- Page Split --->\n"
        "page two\n"
    )


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


def test_runner_prefers_repo_local_deepseek_runtime_when_env_missing(tmp_path, monkeypatch):
    from glossapi.ocr.deepseek import runner, runtime_paths

    corpus = _mk_corpus(tmp_path)
    (corpus.input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%real\n")

    repo_root = tmp_path / "repo"
    python_bin = repo_root / "dependency_setup" / ".venvs" / "deepseek31111" / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("", encoding="utf-8")
    monkeypatch.setattr(runtime_paths, "REPO_ROOT", repo_root)

    calls = {}

    def fake_run_cli(input_dir, output_dir, **kwargs):
        calls["python_bin"] = kwargs["python_bin"]
        md_dir = output_dir / "markdown"
        metrics_dir = output_dir / "json" / "metrics"
        md_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / "doc.md").write_text("ok\n", encoding="utf-8")
        (metrics_dir / "doc.metrics.json").write_text('{"page_count": 1}', encoding="utf-8")

    monkeypatch.setattr(runner, "_run_cli", fake_run_cli)
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("GLOSSAPI_DEEPSEEK_PYTHON", raising=False)
    monkeypatch.delenv("GLOSSAPI_DEEPSEEK_TEST_PYTHON", raising=False)

    result = runner.run_for_files(corpus, ["doc.pdf"], runtime_backend="vllm")

    assert calls["python_bin"] == python_bin
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



def test_runner_reassembles_exact_fill_shards_into_canonical_outputs(tmp_path, monkeypatch):
    import json

    from glossapi.ocr.deepseek import runner
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _join_page_outputs, _write_outputs

    corpus = _mk_corpus(tmp_path)
    downloads_dir = corpus.input_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    (downloads_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n%real\n")

    def fake_run_multi_cli(*, out_root, **kwargs):
        del kwargs
        common_metrics = {
            "source_file": "doc.pdf",
            "source_stem": "doc",
            "ocr_profile": "markdown_grounded",
            "attn_backend": "vllm",
            "runtime_backend": "vllm",
            "batch_size": 96,
            "repair_mode": "auto",
        }
        _write_outputs(
            output_dir=out_root,
            stem="doc__p00001-00002",
            markdown=_join_page_outputs(["page one", "page two"]),
            page_count=2,
            extra_metrics={
                **common_metrics,
                "source_start_page": 1,
                "source_end_page": 2,
                "render_sec": 1.5,
                "infer_sec_total": 2.5,
                "wall_time_sec": 3.5,
                "repair_summary": {"repair_mode": "auto", "pages_flagged": 1, "pages_repaired": 1},
                "page_metrics": [
                    {"page_number": 1, "infer_sec": 1.0, "repair_strategy": "none", "repair_applied": False},
                    {"page_number": 2, "infer_sec": 1.5, "repair_strategy": "plain", "repair_applied": True},
                ],
            },
        )
        _write_outputs(
            output_dir=out_root,
            stem="doc__p00003-00004",
            markdown=_join_page_outputs(["page three", "page four"]),
            page_count=2,
            extra_metrics={
                **common_metrics,
                "source_start_page": 3,
                "source_end_page": 4,
                "render_sec": 0.5,
                "infer_sec_total": 1.5,
                "wall_time_sec": 2.0,
                "repair_summary": {"repair_mode": "auto", "pages_flagged": 0, "pages_repaired": 0},
                "page_metrics": [
                    {"page_number": 1, "infer_sec": 0.7, "repair_strategy": "none", "repair_applied": False},
                    {"page_number": 2, "infer_sec": 0.8, "repair_strategy": "none", "repair_applied": False},
                ],
            },
        )

    monkeypatch.setattr(runner, "_run_multi_cli", fake_run_multi_cli)
    monkeypatch.setattr(runner, "_page_count", lambda path: 4)
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv(
        "GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT",
        str(Path(runner.__file__).resolve().parent / "run_pdf_ocr_vllm.py"),
    )
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_PYTHON", sys.executable)

    result = runner.run_for_files(
        corpus,
        ["doc.pdf"],
        use_gpus="multi",
        devices=[0, 1],
        runtime_backend="vllm",
        scheduler="exact_fill",
        target_batch_pages=2,
    )

    canonical_md = corpus.output_dir / "markdown" / "doc.md"
    canonical_metrics = corpus.output_dir / "json" / "metrics" / "doc.metrics.json"
    assert canonical_md.exists()
    assert canonical_metrics.exists()
    assert canonical_md.read_text(encoding="utf-8") == _join_page_outputs(
        ["page one", "page two", "page three", "page four"]
    ) + "\n"

    metrics = json.loads(canonical_metrics.read_text(encoding="utf-8"))
    assert metrics["reassembled_from_shards"] is True
    assert metrics["reassembled_shard_count"] == 2
    assert [item["page_number"] for item in metrics["page_metrics"]] == [1, 2, 3, 4]
    assert metrics["repair_summary"]["pages_flagged"] == 1
    assert metrics["repair_summary"]["pages_repaired"] == 1
    assert result["doc"]["page_count"] == 4

    assert not (corpus.output_dir / "markdown" / "doc__p00001-00002.md").exists()
    assert (corpus.output_dir / "sidecars" / "ocr_shards" / "markdown" / "doc__p00001-00002.md").exists()
    assert (corpus.output_dir / "sidecars" / "ocr_shards" / "json" / "metrics" / "doc__p00003-00004.metrics.json").exists()


def test_vllm_batch_outputs_accept_in_memory_images_without_disk_roundtrip():
    from PIL import Image

    from glossapi.ocr.deepseek.run_pdf_ocr_vllm import _generate_batch_outputs

    class FakeOutput:
        def __init__(self, text):
            self.outputs = [type("TokenOutput", (), {"text": text})()]

    class FakeLLM:
        def generate(self, prompt_batch, sampling_params=None):
            del sampling_params
            assert len(prompt_batch) == 2
            assert all(item["multi_modal_data"]["image"].mode == "RGB" for item in prompt_batch)
            return [FakeOutput("alpha"), FakeOutput("beta")]

    jobs = [
        {"stem": "doc", "page_number": 1, "image": Image.new("RGB", (4, 4), color="white")},
        {"stem": "doc", "page_number": 2, "image": Image.new("RGB", (4, 4), color="black")},
    ]
    outputs = _generate_batch_outputs(
        FakeLLM(),
        jobs=jobs,
        prompt="prompt",
        batch_size=2,
        sampling_params=object(),
    )

    assert [item["raw_text"] for item in outputs] == ["alpha", "beta"]
    assert jobs[0]["image"].size == (4, 4)
    assert jobs[1]["image"].size == (4, 4)
    for item in jobs:
        item["image"].close()
