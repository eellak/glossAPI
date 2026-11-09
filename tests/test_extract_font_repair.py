"""Worker-local ToUnicode repair integrated with real safe extraction."""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import sys
import time
from types import SimpleNamespace

import pandas as pd
from pypdf import PdfWriter
import pytest

from glossapi import Corpus
from glossapi.pdf_repair import tounicode
from test_pdf_font_repair import greek_fixture, _text


# K2 batch 3: the ledger extract stage runs every document in a fresh spawned worker with a private workspace;
# the repaired PDF is a recorded intermediate (kind repaired_pdf) that is published under repaired_pdfs/ and
# retired after its readers (kept with debug=on). In-process monkeypatches do not reach that worker, so these
# tests inject faults and probes into the worker itself (`worker_hook`: a sitecustomize inherited by spawn) and
# read the repaired PDF from the ledger instead of the corpus directory. Each keeps what it originally asserted.
WORKER_HOOK = """
import hashlib, json, logging, os, sys, time, traceback
LOG = os.environ.get('K2_WORKER_LOG')
if LOG:
    def record(*event):
        with open(LOG, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps([os.getpid(), time.time(), *event], ensure_ascii=False) + '\\n')
    def digest(path):
        try:
            return hashlib.sha256(open(path, 'rb').read()).hexdigest()
        except OSError:
            return None
    def wrap_backend(analyze=None, repair=None):
        from glossapi.pdf_repair import selector
        real = selector.get_backend
        def get_backend(*args, **kwargs):
            backend = real(*args, **kwargs)
            def analyzed(path):
                record('analyze', str(path), digest(path), backend.name)
                return (analyze or (lambda inner, path: inner(path)))(backend.analyze_pdf, path)
            def repaired(src, dst, *rest, **options):
                record('repair', str(src), digest(src), backend.name)
                return (repair or (lambda inner, *a, **k: inner(*a, **k)))(backend.repair_pdf, src, dst, *rest, **options)
            return selector.RepairBackend(backend.name, analyzed, repaired)
        selector.get_backend = get_backend
    class Messages(logging.Handler):
        def emit(self, item):
            if 'Font repair' in item.getMessage():
                record('log', item.getMessage())
    try:
        logging.getLogger('glossapi.gloss_extract').setLevel(logging.INFO)
        logging.getLogger('glossapi.gloss_extract').addHandler(Messages())
%s
        record('hook-ready')
    except Exception:
        record('hook-error', traceback.format_exc())
"""


@pytest.fixture
def worker_hook(tmp_path, monkeypatch):
    """Install ``body`` (indented code using record/digest/wrap_backend) in every spawned worker; return a reader
    of the events those workers recorded."""
    site = tmp_path / "worker-site"
    log = tmp_path / "worker-events.jsonl"

    def install(body=""):
        site.mkdir(exist_ok=True)
        lines = "\n".join("        " + line for line in body.strip("\n").splitlines()) or "        pass"
        (site / "sitecustomize.py").write_text(WORKER_HOOK % lines, encoding="utf-8")
        log.write_text("")
        monkeypatch.setenv("K2_WORKER_LOG", str(log))
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(site), *sys.path]))
        return events

    def events(kind=None):
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        errors = [row for row in rows if row[2] == "hook-error"]
        assert not errors, errors
        return [row[2:] for row in rows if kind is None or row[2] == kind]

    return install


def _repaired_records(root):
    """Ledger records of public repaired PDFs: (relative path, sha256, presence, retention class, ownership)."""
    from glossapi.ledger import open_readonly
    from urllib.parse import unquote, urlsplit
    with open_readonly(Path(root) / "corpus.sqlite3") as db:
        rows = db.execute("SELECT uri, sha256, presence, retention_class, ownership FROM current_file_state "
                          "WHERE kind='repaired_pdf' ORDER BY event_id").fetchall()
    out = []
    for uri, digest, presence, retention, ownership in rows:
        path = Path(unquote(urlsplit(uri).path))
        if ".glossapi" not in path.parts:  # the published spelling, not the private attempt copy
            out.append((str(path.relative_to(Path(root).resolve())), digest, presence, retention, ownership))
    return out


PER_PAGE_BUDGET = 6  # seconds; one page


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expected_repair_sha(source, tmp_path):
    from glossapi.pdf_repair import get_backend
    target = tmp_path / "expected-repair" / Path(source).name
    target.parent.mkdir(exist_ok=True)
    result = get_backend().repair_pdf(source, target)
    assert result.written and result.repaired_fonts >= 1
    return _sha(target)


@pytest.fixture(autouse=True)
def offline_wordlist(monkeypatch):
    monkeypatch.delenv("GLOSSAPI_GREEK_WORDLIST", raising=False)


def _corpus(tmp_path, source, name="paper.with.dots.pdf"):
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    path = downloads / name
    path.write_bytes(source.read_bytes())
    return Corpus(tmp_path, tmp_path), path


def _row(root, name):
    df = pd.read_parquet(root / "download_results" / "download_results.parquet")
    return df.loc[df.filename == name].iloc[0]


def _extract(corpus, **kwargs):
    corpus.extract(phase1_backend="safe", accel_type="CPU", **kwargs)


def test_safe_repairs_original_stem_and_metadata(greek_fixture, tmp_path, monkeypatch, caplog, worker_hook):
    events = worker_hook()
    corpus, source = _corpus(tmp_path, greek_fixture["broken"])
    original = source.read_bytes()
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("\n".join(greek_fixture["text"].split()), encoding="utf-8")
    monkeypatch.setenv("GLOSSAPI_GREEK_WORDLIST", str(wordlist))
    # Corpus loggers intentionally do not propagate to root; the extractor's own log lines are recorded in the
    # worker that runs the extraction (worker_hook).
    corpus.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO):
            _extract(corpus, repair_fonts=True)
    finally:
        corpus.logger.removeHandler(caplog.handler)
    # The repaired PDF keeps the original stem: published as repaired_pdfs/<name>, recorded in the ledger with
    # the bytes of the repair, and retired after extraction read it (debug=off).
    expected = _expected_repair_sha(source, tmp_path)
    assert _repaired_records(tmp_path) == [(f"repaired_pdfs/{source.name}", expected, "deleted", "intermediate", "managed")]
    assert source.read_bytes() == original
    assert greek_fixture["text"] in (tmp_path / "markdown" / f"{source.stem}.md").read_text()
    row = _row(tmp_path, source.name)
    assert row.font_repair_status == "repaired"
    assert row.font_repair_fonts >= 1
    assert row.font_repair_valid_rate_before == 0.0
    assert row.font_repair_valid_rate_after == 1.0
    assert sum("Font repair summary: repaired=1" in record.message for record in caplog.records) == 1
    messages = [event[1] for event in events("log")]
    assert sum("Font repair paper.with.dots: fonts=1 0.000000→1.000000" in message for message in messages) == 1, messages


def test_disabled_uses_garbled_source(greek_fixture, tmp_path, monkeypatch, worker_hook):
    events = worker_hook("wrap_backend()")  # records any analysis in the extraction worker
    corpus, source = _corpus(tmp_path, greek_fixture["broken"])
    _extract(corpus, repair_fonts=False)
    assert events("hook-ready") and not events("analyze"), "Disabled repair must not analyze PDFs"
    assert _text(source).strip() in (tmp_path / "markdown" / f"{source.stem}.md").read_text()
    assert not (tmp_path / "repaired_pdfs").exists() and not _repaired_records(tmp_path)
    row = _row(tmp_path, source.name)
    assert row.font_repair_status == "skipped"
    assert row.font_repair_fonts == 0
    assert pd.isna(row.font_repair_valid_rate_before)
    assert pd.isna(row.font_repair_valid_rate_after)


@pytest.mark.parametrize('native', [False, True])
def test_corpus_repair_uses_public_selector_and_records_backend(greek_fixture, tmp_path, monkeypatch, native, worker_hook):
    # The selector runs in the extraction worker: make the extension unimportable there, or replace it with a
    # recording module that delegates to the Python reference.
    events = worker_hook("""
import importlib.abc, types
if os.environ.get('K2_NATIVE_FAKE') == '1':
    from glossapi.pdf_repair import tounicode
    fake = types.ModuleType('glossapi_rs_pdfrepair')
    def analyze_pdf(path):
        record('native', 'analyze')
        return tounicode.analyze_pdf(path).to_dict()
    def repair_pdf(src, dst, protected):
        record('native', 'repair')
        return tounicode.repair_pdf(src, dst, protected_sources=protected).to_dict()
    fake.analyze_pdf, fake.repair_pdf = analyze_pdf, repair_pdf
    fake.__glossapi_native_api__ = 'GLOSSAPI_NATIVE_API:glossapi_rs_pdfrepair:3;'  # the level the selector needs (K5-A2b)
    sys.modules['glossapi_rs_pdfrepair'] = fake
else:
    class MissingExtension(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'glossapi_rs_pdfrepair':
                raise ImportError('injected unavailable extension')
    sys.modules.pop('glossapi_rs_pdfrepair', None)
    sys.meta_path.insert(0, MissingExtension())
""")
    monkeypatch.setenv('K2_NATIVE_FAKE', '1' if native else '0')
    corpus, source = _corpus(tmp_path, greek_fixture['broken'])
    _extract(corpus, repair_fonts=True)
    row = _row(tmp_path, source.name)
    assert row.font_repair_status == 'repaired'
    assert row.font_repair_backend == ('rust' if native else 'python')
    receipt = json.loads((tmp_path / 'sidecars/extract' / f'{source.stem}.json').read_text())
    assert receipt['font_repair_backend'] == row.font_repair_backend
    assert [event[1] for event in events('native')] == (['analyze', 'repair'] if native else [])


def test_blank_and_corrupt_continue(greek_fixture, tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    blank = downloads / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(blank)
    (downloads / "corrupt.pdf").write_bytes(b"not a PDF")
    (downloads / "good.pdf").write_bytes(greek_fixture["broken"].read_bytes())
    _extract(Corpus(tmp_path, tmp_path))
    assert _row(tmp_path, "blank.pdf").font_repair_status == "no_text_layer"
    assert _row(tmp_path, "corrupt.pdf").font_repair_status == "error"
    assert _row(tmp_path, "good.pdf").font_repair_status == "repaired"
    sidecar = json.loads((tmp_path / "sidecars/extract/corrupt.json").read_text())
    assert sidecar["font_repair_error"]
    assert (tmp_path / "markdown/good.md").is_file()


def test_reuses_sidecar_without_analysis(greek_fixture, tmp_path, monkeypatch, worker_hook):
    """Ledger reuse: an already repaired document is not analysed again. Resume (the default skip_existing) reuses
    the published products; a forced re-extraction is a new attempt in a fresh workspace and analyses again (the
    repaired-PDF mtime cache of the pre-ledger extractor has no workspace to live in), with identical repair."""
    events = worker_hook("wrap_backend()")
    corpus, source = _corpus(tmp_path, greek_fixture["broken"])
    _extract(corpus)
    assert len(events("analyze")) == 1
    first = _repaired_records(tmp_path)
    assert [record[0] for record in first] == [f"repaired_pdfs/{source.name}"]
    markdown = (tmp_path / "markdown" / f"{source.stem}.md").read_bytes()
    _extract(corpus)  # resume
    assert len(events("analyze")) == 1, "a repaired document was analysed again on resume"
    assert _repaired_records(tmp_path) == first
    assert _row(tmp_path, source.name).font_repair_fonts >= 1
    assert _row(tmp_path, source.name).font_repair_status == "repaired"
    _extract(corpus, skip_existing=False)  # forced: declared ledger behaviour
    assert len(events("analyze")) == 2
    assert {record[1] for record in _repaired_records(tmp_path)} == {first[0][1]}
    assert (tmp_path / "markdown" / f"{source.stem}.md").read_bytes() == markdown
    assert _row(tmp_path, source.name).font_repair_status == "repaired"


def test_stale_sidecar_is_rebuilt(greek_fixture, tmp_path, monkeypatch, worker_hook):
    """A repair that no longer matches its source is rebuilt from the current source: the ledger keys the repair
    by the source version, so changed source bytes are analysed again and give a new repaired PDF."""
    events = worker_hook("wrap_backend()")
    corpus, source = _corpus(tmp_path, greek_fixture["broken"])
    _extract(corpus)
    before = _repaired_records(tmp_path)
    source.write_bytes(source.read_bytes() + b"\n% revised\n")  # same document, new bytes (a new version)
    _extract(corpus, skip_existing=False)
    analysed = [event[2] for event in events("analyze")]
    assert analysed[-1] == _sha(source) and len(analysed) == 2
    after = _repaired_records(tmp_path)
    assert len(after) == 2 and after[0] == before[0] and after[1][1] != before[0][1]
    assert after[1][1] == _expected_repair_sha(source, tmp_path)
    assert _row(tmp_path, source.name).font_repair_status == "repaired"


@pytest.mark.parametrize("failure", ["analysis", "repair", "timeout"])
def test_prepass_failure_falls_back(greek_fixture, tmp_path, monkeypatch, failure, worker_hook):
    events = worker_hook("""
failure = os.environ['K2_FAILURE']
def slow(inner, path):
    record('slow-start')
    time.sleep(12)
    record('slow-end')
def fail(inner, *args, **kwargs):
    raise RuntimeError('injected pre-pass failure')
wrap_backend(analyze=slow if failure == 'timeout' else fail if failure == 'analysis' else None,
             repair=fail if failure == 'repair' else None)
import pypdfium2
real_document = pypdfium2.PdfDocument
def capture(path, *args, **kwargs):
    record('opened', str(path), digest(path) if isinstance(path, (str, os.PathLike)) else None)
    return real_document(path, *args, **kwargs)
pypdfium2.PdfDocument = capture
""")
    monkeypatch.setenv("K2_FAILURE", failure)
    corpus, source = _corpus(tmp_path, greek_fixture["broken"])
    from glossapi.gloss_extract import GlossExtract
    corpus.extractor = GlossExtract()
    # Reaches the worker through the stage settings. The attempt deadline also covers the worker's spawn, so the
    # budget leaves room for a loaded machine; the pre-pass deadline is half of it (3 s), the injected stall 12 s.
    corpus.extractor.per_page_timeout_s = PER_PAGE_BUDGET
    _extract(corpus)
    finished = time.time()
    if failure == "timeout":
        # The pre-pass is cut at its deadline (half of the budget), long before the 12 s analysis would end.
        rows = [json.loads(line) for line in (tmp_path / "worker-events.jsonl").read_text().splitlines()]
        start_time = next(row[1] for row in rows if row[2] == "slow-start")
        assert not events("slow-end") and finished - start_time < PER_PAGE_BUDGET * 1.5
    assert _row(tmp_path, source.name).font_repair_status == "error"
    data = json.loads((tmp_path / "sidecars/extract" / f"{source.stem}.json").read_text())
    assert "timeout" in data["font_repair_error"].lower() if failure == "timeout" else "injected" in data["font_repair_error"]
    # Extraction read the original PDF (the worker's frozen copy of the source), never a repaired one.
    opened = events("opened")
    assert any(event[2] == _sha(source) for event in opened)
    assert not any("repaired_pdfs" in event[1] for event in opened)
    assert not _repaired_records(tmp_path)
    assert _text(source).strip() in (tmp_path / "markdown" / f"{source.stem}.md").read_text()
    assert _row(tmp_path, source.name).extract_error


def test_bad_wordlist_is_nullable(greek_fixture, tmp_path, monkeypatch):
    corpus, source = _corpus(tmp_path, greek_fixture["broken"])
    monkeypatch.setenv("GLOSSAPI_GREEK_WORDLIST", str(tmp_path / "missing.txt"))
    _extract(corpus)
    row = _row(tmp_path, source.name)
    assert row.font_repair_status == "repaired"
    assert pd.isna(row.font_repair_valid_rate_before)
    assert pd.isna(row.font_repair_valid_rate_after)


def test_rates_use_only_first_twenty_pages(greek_fixture, tmp_path, monkeypatch):
    writer = PdfWriter()
    writer.append(greek_fixture["broken"], pages=[0] * 20)
    writer.append(greek_fixture["original"])
    source = tmp_path / "twenty_one.pdf"
    writer.write(source)
    corpus, source = _corpus(tmp_path, source)
    words = tmp_path / "words.txt"
    words.write_text("\n".join(greek_fixture["text"].split()))
    monkeypatch.setenv("GLOSSAPI_GREEK_WORDLIST", str(words))
    # Page 21 is readable already; including it would give a positive before rate.
    assert tounicode.extract_text_stats(source)["totals"]["dictionary_valid_rate"] > 0
    _extract(corpus)
    row = _row(tmp_path, source.name)
    assert row.font_repair_valid_rate_before == 0.0
    assert row.font_repair_valid_rate_after == 1.0


def test_timeout_after_repair_does_not_publish(greek_fixture, tmp_path, monkeypatch, worker_hook):
    events = worker_hook("""
from glossapi.pdf_repair import text_stats
real_stats = text_stats.extract_text_stats
def stats(path, **kwargs):
    if '.extract-font-' in str(path):  # the staged repaired PDF, measured before publication
        record('slow-stats', str(path))
        time.sleep(12)
    return {"totals": {"dictionary_valid_rate": 0.0}}
text_stats.extract_text_stats = stats
""")
    corpus, source = _corpus(tmp_path, greek_fixture["broken"])
    from glossapi.gloss_extract import GlossExtract
    corpus.extractor = GlossExtract()
    corpus.extractor.per_page_timeout_s = PER_PAGE_BUDGET
    words = tmp_path / "words.txt"  # a configured wordlist enables the rate measurement
    words.write_text("\n".join(greek_fixture["text"].split()), encoding="utf-8")
    monkeypatch.setenv("GLOSSAPI_GREEK_WORDLIST", str(words))
    _extract(corpus)
    assert events("slow-stats")
    assert _row(tmp_path, source.name).font_repair_status == "error"
    assert not _repaired_records(tmp_path)  # nothing published or recorded as a repaired PDF
    assert not (tmp_path / "repaired_pdfs" / source.name).exists()
    assert not list(tmp_path.rglob(".extract-font-*"))
    assert _text(source).strip() in (tmp_path / "markdown" / f"{source.stem}.md").read_text()


def test_consistent_and_unsupported(greek_fixture, tmp_path, monkeypatch, worker_hook):
    corpus, source = _corpus(tmp_path / "consistent", greek_fixture["original"])
    _extract(corpus)
    assert _row(tmp_path / "consistent", source.name).font_repair_status == "not_needed"
    # An identical re-extraction would reuse that ledger result; classify 'unsupported' in a fresh corpus.
    events = worker_hook("""
from types import SimpleNamespace
wrap_backend(analyze=lambda inner, path: SimpleNamespace(category='unsupported', fonts=[], error=None))
""")
    corpus, source = _corpus(tmp_path / "unsupported", greek_fixture["original"])
    _extract(corpus, skip_existing=False)
    assert len(events("analyze")) == 1
    assert _row(tmp_path / "unsupported", source.name).font_repair_status == "unsupported"


def test_mixed_with_repairable_font(greek_fixture, tmp_path, monkeypatch, worker_hook):
    events = worker_hook("""
def mixed(inner, path):
    report = inner(path)
    report.category = 'mixed'
    record('mixed', report.category)
    return report
wrap_backend(analyze=mixed)
""")
    corpus, source = _corpus(tmp_path, greek_fixture["broken"])
    _extract(corpus)
    assert events("mixed") == [["mixed", "mixed"]]
    assert _row(tmp_path, source.name).font_repair_status == "repaired"
    assert greek_fixture["text"] in (tmp_path / "markdown" / f"{source.stem}.md").read_text()


@pytest.mark.parametrize("batch_files,chunked,name", [(1, False, "paper.pdf"), (2, False, "paper.PDF"), (1, True, "paper.pdf")])
def test_docling_receives_repaired_path(greek_fixture, tmp_path, monkeypatch, batch_files, chunked, name, worker_hook):
    # The scoped fake-Docling converter pattern, installed in the extraction worker: the actual pipeline builder
    # and export path run there without Docling or GPU packages; every converted path is recorded.
    events = worker_hook("""
from collections import defaultdict
from glossapi.scripts import docling_cost_benchmark as benchmark
import pypdfium2
class Converter(benchmark.FakeConverter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pages = defaultdict(lambda: 1)
    def convert(self, file, **kwargs):
        record('converted', str(file), digest(file))
        return super().convert(file, **kwargs)
    def markdown(self, path, pages):
        pdf = pypdfium2.PdfDocument(str(path))
        try:
            page = pdf[0]; text = page.get_textpage(); value = text.get_text_range(); text.close(); page.close()
        finally:
            pdf.close()
        return value
benchmark.FakeConverter = Converter
fake_docling = benchmark.fake_docling('baseline', {})  # module global: keeps the fake installed for the worker
fake_docling.__enter__()
""")
    from glossapi.scripts import docling_cost_benchmark as benchmark
    corpus, source = _corpus(tmp_path, greek_fixture["broken"], name=name)
    second = source.with_name("second.pdf")
    second.write_bytes(source.read_bytes())
    # Explicit-path callers can supply metadata with the exact original casing.
    metadata_dir = tmp_path / "download_results"
    metadata_dir.mkdir(exist_ok=True)
    pd.DataFrame({"filename": [source.name, second.name], "url": ["", ""]}).to_parquet(
        metadata_dir / "download_results.parquet", index=False)
    monkeypatch.setenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES", str(batch_files))
    monkeypatch.setattr(Corpus, "_gpu_preflight", lambda *args, **kwargs: None)
    with benchmark.fake_docling("baseline", {}):
        if chunked:
            module = importlib.import_module("glossapi.gloss_extract")
            corpus.extractor = module.GlossExtract()
            corpus.extractor.long_pdf_page_threshold = 0
            corpus.extractor.chunk_size = 1
        corpus.extract(phase1_backend="docling", repair_fonts=True, export_doc_json=False, file_paths=[source, second])
    converted = events("converted")
    repaired = {record[0]: record[1] for record in _repaired_records(tmp_path)}
    # Docling converted exactly the two repaired PDFs (named <stem>.pdf), never the originals.
    assert sorted(Path(path).name for path, _ in [(e[1], e[2]) for e in converted]) == ["paper.pdf", "second.pdf"]
    assert all(Path(event[1]).parent.name == "repaired_pdfs" for event in converted)
    assert sorted(event[2] for event in converted) == sorted([repaired["repaired_pdfs/paper.pdf"], repaired["repaired_pdfs/second.pdf"]])
    assert _sha(source) not in {event[2] for event in converted}
    assert _row(tmp_path, source.name).font_repair_status == "repaired"
    assert greek_fixture["text"] in (tmp_path / "markdown" / f"{source.stem}.md").read_text()


@pytest.mark.parametrize("enabled", [True, False])
def test_worker_queue_propagates_repair(greek_fixture, tmp_path, monkeypatch, enabled):
    from glossapi.corpus import gpu_extract_worker_queue
    corpus, source = _corpus(tmp_path, greek_fixture["broken"])
    work = queue.Queue()
    work.put([str(source)])
    results = queue.Queue()
    monkeypatch.delenv("GLOSSAPI_WORKER_LOG_DIR", raising=False)
    monkeypatch.setenv("GLOSSAPI_WORKER_LOG_VERBOSE", "0")
    # Workers set these directly; restore them after this in-process worker test.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("GLOSSAPI_DOCLING_DEVICE", "")
    with pytest.raises(SystemExit, match="0"):
        gpu_extract_worker_queue(0, 0, "gpu0-w0", str(tmp_path), str(tmp_path), work,
                                 False, False, False, False, False, "pdf", 1,
                                 False, False, False, "safe", results,
                                 repair_fonts=enabled)
    corpus._finalize_font_repair_metadata([source])
    row = _row(tmp_path, source.name)
    assert row.font_repair_status == ("repaired" if enabled else "skipped")
    assert (tmp_path / "repaired_pdfs" / source.name).exists() is enabled
    expected = greek_fixture["text"] if enabled else _text(source).strip()
    assert expected in (tmp_path / "markdown" / f"{source.stem}.md").read_text()
