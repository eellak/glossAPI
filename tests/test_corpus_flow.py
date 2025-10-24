from pathlib import Path

from glossapi import Corpus
from glossapi.corpus import _ProcessingStateManager


EXPECTED_PHASES = (
    "download",
    "extract",
    "clean",
    "ocr",
    "formula_enrich_from_json",
    "section",
    "annotate",
)


def test_corpus_exposes_pipeline_phases():
    """Ensure the public Corpus API still exposes every documented phase."""
    for name in EXPECTED_PHASES:
        attr = getattr(Corpus, name, None)
        assert callable(attr), f"Corpus missing callable phase '{name}'"


def test_process_all_calls_phases_in_order(tmp_path, monkeypatch):
    """`process_all` should run the high-level phases in documented order."""

    corpus = Corpus(input_dir=tmp_path, output_dir=tmp_path)
    call_order: list[str] = []

    def _capture(name):
        def _inner(*args, **kwargs):
            call_order.append(name)
        return _inner

    monkeypatch.setattr(corpus, "download", _capture("download"))
    monkeypatch.setattr(corpus, "extract", _capture("extract"))
    monkeypatch.setattr(corpus, "section", _capture("section"))
    monkeypatch.setattr(corpus, "annotate", _capture("annotate"))

    corpus.process_all(input_format="pdf", download_first=True)

    assert call_order == ["download", "extract", "section", "annotate"]


def test_extract_skips_processed_files_on_resume(tmp_path, monkeypatch):
    """Multi-GPU resume should skip already processed files."""

    downloads_dir = tmp_path / "downloads"
    markdown_dir = tmp_path / "markdown"
    downloads_dir.mkdir()
    markdown_dir.mkdir()

    processed_pdf = downloads_dir / "already_done.pdf"
    processed_pdf.write_bytes(b"%PDF-1.4\n%dummy\n")

    state_mgr = _ProcessingStateManager(markdown_dir / ".processing_state.pkl")
    state_mgr.save({str(processed_pdf)}, set())

    corpus = Corpus(input_dir=tmp_path, output_dir=tmp_path)

    dummy_calls: list[list[Path]] = []

    class DummyExtractor:
        def configure_batch_policy(self, *args, **kwargs):
            return None

        def ensure_extractor(self, *args, **kwargs):
            return None

        def enable_accel(self, *args, **kwargs):
            return None

        def extract_path(self, files, out_dir, skip_existing=True):
            dummy_calls.append(list(files))

    def _fake_prime(self, **kwargs):
        self.extractor = DummyExtractor()

    def _fake_gpu_preflight(self, **kwargs):
        return None

    monkeypatch.setattr(Corpus, "prime_extractor", _fake_prime, raising=False)
    monkeypatch.setattr(Corpus, "_gpu_preflight", _fake_gpu_preflight, raising=False)

    corpus.extract(
        input_format="pdf",
        use_gpus="multi",
        devices=[0],
    )

    assert dummy_calls == [], "Expected no extraction when files already marked processed"
