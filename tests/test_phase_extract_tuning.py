from pathlib import Path

from glossapi.corpus.phase_extract import (
    _build_extract_work_items,
    _resolve_docling_batch_target_pages,
    _resolve_docling_max_batch_files,
    _resolve_docling_queue_policy,
)


def test_resolve_docling_max_batch_files_defaults_to_conservative_batch(monkeypatch):
    monkeypatch.delenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", raising=False)
    assert _resolve_docling_max_batch_files() == 1


def test_resolve_docling_max_batch_files_accepts_explicit_override(monkeypatch):
    monkeypatch.setenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", "4")
    assert _resolve_docling_max_batch_files() == 4


def test_resolve_docling_max_batch_files_ignores_invalid_values(monkeypatch):
    monkeypatch.setenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", "not-an-int")
    assert _resolve_docling_max_batch_files() == 1


def test_resolve_docling_batch_target_pages_defaults(monkeypatch):
    monkeypatch.delenv("GLOSSAPI_DOCLING_BATCH_TARGET_PAGES", raising=False)
    assert _resolve_docling_batch_target_pages() == 256


def test_resolve_docling_batch_target_pages_accepts_override(monkeypatch):
    monkeypatch.setenv("GLOSSAPI_DOCLING_BATCH_TARGET_PAGES", "384")
    assert _resolve_docling_batch_target_pages() == 384


def test_resolve_docling_queue_policy_uses_env_when_extractor_is_unprimed(monkeypatch):
    monkeypatch.setenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", "2")
    assert _resolve_docling_queue_policy(None) == (2, 600)


def test_resolve_docling_queue_policy_prefers_extractor_values(monkeypatch):
    class Extractor:
        max_batch_files = 3
        long_pdf_page_threshold = 900

    monkeypatch.setenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", "2")
    assert _resolve_docling_queue_policy(Extractor()) == (3, 900)


def test_build_extract_work_items_packs_smaller_files_by_page_budget():
    paths = [Path("a.pdf"), Path("b.pdf"), Path("c.pdf"), Path("d.pdf")]
    pages = {
        "a.pdf": 140,
        "b.pdf": 120,
        "c.pdf": 110,
        "d.pdf": 90,
    }

    items = _build_extract_work_items(
        paths,
        max_batch_files=2,
        target_batch_pages=250,
        long_pdf_page_threshold=600,
        page_counter=lambda path: pages[path.name],
    )

    assert [[p.name for p in item] for item in items] == [["a.pdf", "c.pdf"], ["b.pdf", "d.pdf"]]


def test_build_extract_work_items_keeps_long_pdf_as_standalone_work_item():
    paths = [Path("huge.pdf"), Path("small-a.pdf"), Path("small-b.pdf")]
    pages = {
        "huge.pdf": 1200,
        "small-a.pdf": 100,
        "small-b.pdf": 80,
    }

    items = _build_extract_work_items(
        paths,
        max_batch_files=3,
        target_batch_pages=250,
        long_pdf_page_threshold=600,
        page_counter=lambda path: pages[path.name],
    )

    assert [p.name for p in items[0]] == ["huge.pdf"]
    assert sorted(p.name for p in items[1]) == ["small-a.pdf", "small-b.pdf"]
