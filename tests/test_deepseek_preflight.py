import sys

from glossapi.ocr.deepseek.preflight import check_deepseek_env


def test_preflight_reports_missing_components(tmp_path):
    env = {
        "GLOSSAPI_DEEPSEEK_ALLOW_CLI": "0",
        "GLOSSAPI_DEEPSEEK_ALLOW_STUB": "1",
        "GLOSSAPI_DEEPSEEK_TEST_PYTHON": str(tmp_path / "missing_python"),
        "GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT": str(tmp_path / "missing_script.py"),
        "GLOSSAPI_DEEPSEEK_MODEL_DIR": str(tmp_path / "missing_model"),
    }
    report = check_deepseek_env(env, check_torch=False)
    names = {c.name for c in report.errors}
    assert "allow_cli" in names
    assert "allow_stub" in names
    assert "deepseek_python" in names
    assert "runner_script" in names
    assert "model_dir" in names
    assert not report.ok


def test_preflight_passes_with_complete_env(tmp_path):
    script = tmp_path / "run_pdf_ocr_transformers.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    model_dir = tmp_path / "DeepSeek-OCR-2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model-00001-of-000001.safetensors").write_bytes(b"stub")

    env = {
        "GLOSSAPI_DEEPSEEK_ALLOW_CLI": "1",
        "GLOSSAPI_DEEPSEEK_ALLOW_STUB": "0",
        "GLOSSAPI_DEEPSEEK_TEST_PYTHON": sys.executable,
        "GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT": str(script),
        "GLOSSAPI_DEEPSEEK_MODEL_DIR": str(model_dir),
    }
    report = check_deepseek_env(env, check_torch=False)
    assert report.ok
    assert not report.errors
