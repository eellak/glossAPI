"""Keep extractor logs inside each test's disposable output directory."""
import pytest


@pytest.fixture(autouse=True)
def conversion_log_in_tmp_path(tmp_path, monkeypatch):
    # Spawned workers inherit this setting too. Individual override tests can
    # still replace it with their own destination using monkeypatch.setenv.
    monkeypatch.setenv("GLOSSAPI_CONVERSION_LOG_FILE", str(tmp_path / "conversion.log"))
