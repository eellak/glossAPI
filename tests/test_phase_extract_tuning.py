from glossapi.corpus.phase_extract import _resolve_docling_max_batch_files


def test_resolve_docling_max_batch_files_defaults_to_conservative_batch(monkeypatch):
    monkeypatch.delenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", raising=False)
    assert _resolve_docling_max_batch_files() == 1


def test_resolve_docling_max_batch_files_accepts_explicit_override(monkeypatch):
    monkeypatch.setenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", "4")
    assert _resolve_docling_max_batch_files() == 4


def test_resolve_docling_max_batch_files_ignores_invalid_values(monkeypatch):
    monkeypatch.setenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", "not-an-int")
    assert _resolve_docling_max_batch_files() == 1
