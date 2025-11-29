import sys
from pathlib import Path

from glossapi.ocr.deepseek.preflight import check_deepseek_env


def test_preflight_reports_missing_components(tmp_path):
    env = {
        "GLOSSAPI_DEEPSEEK_ALLOW_CLI": "0",
        "GLOSSAPI_DEEPSEEK_ALLOW_STUB": "1",
        "GLOSSAPI_DEEPSEEK_TEST_PYTHON": str(tmp_path / "missing_python"),
        "GLOSSAPI_DEEPSEEK_VLLM_SCRIPT": str(tmp_path / "missing_script.py"),
        "GLOSSAPI_DEEPSEEK_MODEL_DIR": str(tmp_path / "missing_model"),
        "GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH": str(tmp_path / "missing_lib"),
        "PATH": str(tmp_path),  # no cc1plus here
    }
    report = check_deepseek_env(env, check_flashinfer=False)
    names = {c.name for c in report.errors}
    assert "deepseek_python" in names
    assert "vllm_script" in names
    assert "model_dir" in names
    assert "ld_library_path" in names
    assert "cc1plus" in names
    assert not report.ok


def test_preflight_passes_with_complete_env(tmp_path):
    script = tmp_path / "run_pdf_ocr_vllm.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    model_dir = tmp_path / "DeepSeek-OCR"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model-00001-of-000001.safetensors").write_bytes(b"stub")
    lib_dir = tmp_path / "libjpeg"
    lib_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    cc1plus = fake_bin / "cc1plus"
    cc1plus.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cc1plus.chmod(0o755)

    env = {
        "GLOSSAPI_DEEPSEEK_ALLOW_CLI": "1",
        "GLOSSAPI_DEEPSEEK_ALLOW_STUB": "0",
        "GLOSSAPI_DEEPSEEK_TEST_PYTHON": sys.executable,
        "GLOSSAPI_DEEPSEEK_VLLM_SCRIPT": str(script),
        "GLOSSAPI_DEEPSEEK_MODEL_DIR": str(model_dir),
        "GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH": str(lib_dir),
        "PATH": str(fake_bin),
    }
    report = check_deepseek_env(env, check_flashinfer=False)
    assert report.ok
    assert not report.errors
