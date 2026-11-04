"""K5-D gap sweep: FIX6-G2 (download disk_full in ``process_all``) and FIX8-G1 (run-start sweep of abandoned publication
temporaries).

FIX6-G2. ``Corpus.process_all(download_first=True)`` logged every ``download()`` failure and went on to extract. An
   exhausted filesystem (ENOSPC, EDQUOT, SQLite full) now propagates, as everywhere else in the pipeline (fix 6, F3);
   any other download failure keeps the documented best-effort continuation.
FIX8-G1. The temporary an interrupted publication left (a process death between its intent record and the replace) was
   removed only when that publication was repeated (resume / --retry-failed). Every run start now sweeps the intent
   records that bind a temporary (``LedgerWriter.sweep_abandoned_temporaries``) under exactly the repeated barrier's
   rules: the agreed recorded lock, a regular single-name file, the recorded bytes or a strict prefix of the recorded
   source, never by name alone; an unverifiable one is retained and never blocks. Each test below fails with the sweep
   disabled (the ``sweep='off'`` controls assert the old behaviour) or exercises a retention rule.
"""
from __future__ import annotations

import errno
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

from glossapi import Corpus
import glossapi.parquet_schema as ps
from glossapi.ledger import writer as W
from glossapi.ledger.writer import LedgerWriter, open_readonly
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import _sha
from test_k4g_fix4 import products


# ================================================================================ FIX6-G2: process_all download ===

def _exhausted(code):
    if code == 'sqlite':
        return sqlite3.OperationalError('database or disk is full')
    return OSError(code, 'injected download storage failure')


def _phases(corpus, monkeypatch, download_error):
    calls = []

    def download(*args, **kwargs):
        calls.append('download')
        raise download_error

    def phase(name):
        def run(*args, **kwargs):
            calls.append(name)
        return run
    monkeypatch.setattr(corpus, 'download', download)
    for name in ('extract', 'section', 'annotate'):
        monkeypatch.setattr(corpus, name, phase(name))
    return calls


@pytest.mark.parametrize('code', [errno.ENOSPC, errno.EDQUOT, 'sqlite'])
def test_process_all_download_disk_full_propagates(tmp_path, monkeypatch, code):
    """A full disk under the download step stops ``process_all`` with that error; extraction never starts."""
    corpus = Corpus(input_dir=tmp_path, output_dir=tmp_path)
    error = _exhausted(code)
    calls = _phases(corpus, monkeypatch, error)
    with pytest.raises(type(error)) as raised:
        corpus.process_all(download_first=True)
    assert raised.value is error
    assert calls == ['download']


def test_process_all_download_disk_full_as_a_cause_propagates(tmp_path, monkeypatch):
    """The download's own wrapper error with ENOSPC as its cause is the same condition (``disk_full`` walks the chain)."""
    corpus = Corpus(input_dir=tmp_path, output_dir=tmp_path)
    try:
        try:
            raise OSError(errno.ENOSPC, 'no space')
        except OSError as cause:
            raise RuntimeError('download failed') from cause
    except RuntimeError as wrapped:
        error = wrapped
    calls = _phases(corpus, monkeypatch, error)
    with pytest.raises(RuntimeError):
        corpus.process_all(download_first=True)
    assert calls == ['download']


def test_process_all_other_download_failure_still_continues(tmp_path, monkeypatch):
    """Unchanged: any other download failure is logged and the pipeline continues with the files already present."""
    corpus = Corpus(input_dir=tmp_path, output_dir=tmp_path)
    calls = _phases(corpus, monkeypatch, RuntimeError('network down'))
    corpus.process_all(download_first=True)
    assert calls == ['download', 'extract', 'section', 'annotate']


# ======================================================================= FIX8-G1: the run-start temporary sweep ===

CRASH_CHILD = r'''
import os, sys, json
from pathlib import Path
from types import SimpleNamespace
from glossapi import Corpus
import glossapi.stages.products as P
import glossapi.ledger.writer as W
from test_fix6_astra import SETTINGS6
root, step, sweep = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out = root / 'out'
sources = [Path(p) for p in sys.argv[4:]]
if __name__ == '__main__':
    if sweep == 'off':  # the control: the pre-K5-D start (no sweep)
        W.LedgerWriter.sweep_abandoned_temporaries = lambda self: []
    if step in ('publication_full', 'publication_partial'):
        real = os.replace
        def replace(temporary, written):
            if Path(written).name == 'export.jsonl':
                if step == 'publication_partial':
                    os.truncate(temporary, os.path.getsize(temporary) // 2)
                os._exit(92)
            return real(temporary, written)
        P.os = SimpleNamespace(**{name: getattr(os, name) for name in dir(os) if not name.startswith('__')})
        P.os.replace = replace
    if step == 'fresh_same':      # a fresh run over the same output that repeats every stage (a new run, not a resume)
        result = Corpus(root / 'empty', out).run(sources, stages=('extract', 'clean', 'export'), settings=SETTINGS6)
    elif step == 'fresh_other':   # a fresh run that never repeats the interrupted export publication
        result = Corpus(root / 'empty', out).run(sources, stages=('extract', 'clean'), settings=SETTINGS6)
    elif step == 'legacy_clean':  # a single-phase legacy call (the journal's own run start)
        Corpus(out, out).clean()
        result = {'state': 'legacy'}
    else:
        result = Corpus(root / 'empty', out).run(sources, stages=('extract', 'clean', 'export'), settings=SETTINGS6)
    print(json.dumps({'state': result['state']}))
'''


def _child_runner(tmp_path):
    sources = _sources(tmp_path)
    script = tmp_path / 'crash_child.py'
    script.write_text(CRASH_CHILD)
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'GLOSSAPI_CONVERSION_LOG_FILE': str(tmp_path / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([*filter(None, [os.environ.get('PYTHONPATH')]), str(repo / 'src'), str(repo / 'tests')])}

    def child(root, step, sweep='on'):
        root.mkdir(parents=True, exist_ok=True)
        return subprocess.run([sys.executable, '-B', str(script), str(root), step, sweep, *map(str, sources)],
                              capture_output=True, text=True, timeout=300, env=env, cwd=root)
    return child


def _left_temporaries(output):
    return sorted(p for p in output.iterdir() if p.name.startswith('.export.jsonl.') and p.name.endswith('.tmp'))


def _event_count(output):
    with open_readonly(output / 'corpus.sqlite3') as connection:
        return connection.execute(f'SELECT COUNT(*) FROM {connection.event_table}').fetchone()[0]


@pytest.mark.parametrize('sweep', ['on', 'off'])
@pytest.mark.parametrize('cut', ['publication_full', 'publication_partial'])
@pytest.mark.parametrize('fresh', ['fresh_other', 'fresh_same'])
def test_fresh_run_sweeps_the_abandoned_publication_temporary(tmp_path, cut, fresh, sweep):
    """A process death after the export publication's intent record, at its replace (temporary complete, or its copy
    cut short at half). A FRESH run over the same output (not a resume; ``fresh_other`` never repeats the export) removes
    exactly that verified temporary at its start and completes. ``fresh_same`` ends with the uninterrupted run's
    products. Control ``off``: without the sweep the temporary is still there after the fresh run (FIX8-G1)."""
    child = _child_runner(tmp_path)
    if fresh == 'fresh_same':
        reference = child(tmp_path / 'ref', 'reference')
        assert reference.returncode == 0, reference.stderr[-2000:]
    killed = child(tmp_path / 'faulted', cut)
    assert killed.returncode == 92, killed.stderr[-2000:]
    output = tmp_path / 'faulted/out'
    left = _left_temporaries(output)
    assert len(left) == 1, 'the crash must leave the publication temporary'
    again = child(tmp_path / 'faulted', fresh, sweep)
    assert again.returncode == 0, again.stderr[-2000:]
    assert json.loads(again.stdout.strip().splitlines()[-1])['state'] == 'completed'
    if fresh == 'fresh_same':
        reference_products, fresh_products = products(tmp_path / 'ref/out'), products(output)
        differing = {name for name in set(reference_products) | set(fresh_products)
                     if reference_products.get(name) != fresh_products.get(name)}
        # The interrupted run never closed, so its clean step's deferred temporary report (a diagnostic observation role,
        # ``writer.DIAGNOSTIC_OBSERVATION_ROLES``) was never released; a fresh run does not adopt it. Pre-existing and the
        # same with the sweep off; every other product equals the uninterrupted run's.
        assert differing - {left[0].name} == {'cleaning_report.parquet'}, differing
    if sweep == 'on':
        assert not left[0].exists() and not _left_temporaries(output)
    else:
        assert left[0].exists(), 'control: the pre-K5-D start leaves the temporary in place'


@pytest.mark.parametrize('sweep', ['on', 'off'])
def test_legacy_phase_start_sweeps_the_abandoned_temporary(tmp_path, sweep):
    """The legacy single-phase start (the journal owns its writer: ``Corpus.clean()``) runs the same sweep."""
    child = _child_runner(tmp_path)
    killed = child(tmp_path / 'faulted', 'publication_full')
    assert killed.returncode == 92, killed.stderr[-2000:]
    output = tmp_path / 'faulted/out'
    left = _left_temporaries(output)
    assert len(left) == 1
    again = child(tmp_path / 'faulted', 'legacy_clean', sweep)
    assert again.returncode == 0, again.stderr[-2000:]
    assert left[0].exists() is (sweep == 'off')


def _seed(tmp_path, *, temporary=None, extra=None, lock=None):
    """Unit ledger: one run with one recorded uncommitted write of the metadata file's current bytes (fix 8 shape)."""
    path = tmp_path / 'download_results' / 'download_results.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'filename': ['a.pdf', 'b.pdf']}).to_parquet(path, index=False)
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    with writer.transaction():
        run = writer.run(SETTINGS6, {}, {}, code_revision='k5d')
    entry_extra = dict(extra or {})
    if temporary is not None:
        entry_extra['temporary'] = temporary
    if lock is not None:
        entry_extra['lock'] = lock
    writer.record_uncommitted_writes(run, [(ps.resolve_metadata_target(path), _sha(path), path.stat().st_size, True, entry_extra)],
                                     wrapper_stage='clean')
    writer.close()
    return run, path


def _start_sweep(tmp_path):
    """A later run start: a new owner of the ledger (the recording process is gone) runs the sweep."""
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    try:
        before = writer.connection.execute(f'SELECT COUNT(*) FROM {writer.connection.event_table}').fetchone()[0]
        removed = writer.sweep_abandoned_temporaries()
        after = writer.connection.execute(f'SELECT COUNT(*) FROM {writer.connection.event_table}').fetchone()[0]
    finally:
        writer.close()
    assert before == after, 'the sweep writes no ledger record'
    return removed


def test_sweep_removes_only_verified_temporaries(tmp_path):
    """Recorded path AND recorded bytes AND one name: a recorded temporary holding other bytes, a hard-linked one and an
    unrecorded ``*.tmp`` holding the recorded bytes are all kept; only the verified one is removed (as the rollback)."""
    folder = tmp_path / 'download_results'
    folder.mkdir()
    exact, other, linked = (folder / f'download_results.parquet.{name}.tmp' for name in ('exact', 'other', 'linked'))
    unrecorded = folder / 'download_results.parquet.unrecorded.tmp'
    run, path = _seed(tmp_path, temporary=exact)
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    try:
        digest, size = _sha(path), path.stat().st_size
        for extra in (other, linked):
            writer.record_uncommitted_writes(run, [(path, digest, size, True, {'temporary': extra})], wrapper_stage='clean')
    finally:
        writer.close()
    for target in (exact, linked, unrecorded):
        shutil.copyfile(path, target)
    other.write_bytes(path.read_bytes()[:-1] + b'Z')
    os.link(linked, tmp_path / 'second-name')
    assert _start_sweep(tmp_path) == [str(exact)]
    assert not exact.exists() and other.exists() and linked.exists() and unrecorded.exists() and path.exists()
    assert _start_sweep(tmp_path) == []  # idempotent; the retained residue never blocks


def test_sweep_removes_a_strict_prefix_of_the_recorded_source_only(tmp_path):
    """A publication temporary whose copy was cut short is a strict prefix of its recorded immutable source: removed. A
    temporary longer than the source, or a prefix of a source that changed since the record, is retained."""
    source = tmp_path / '.glossapi' / 'artifacts' / 'export.jsonl'
    source.parent.mkdir(parents=True)
    data = b''.join(b'{"row": %d}\n' % i for i in range(4000))
    source.write_bytes(data)
    target = tmp_path / 'export.jsonl'
    cut, longer = tmp_path / '.export.jsonl.cut.tmp', tmp_path / '.export.jsonl.longer.tmp'
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    try:
        with writer.transaction():
            run = writer.run(SETTINGS6, {}, {}, code_revision='k5d')
        writer.record_uncommitted_writes(run, [(target, _sha(source), len(data), False, {'temporary': t, 'source': source})
                                               for t in (cut, longer)], wrapper_stage='export')
    finally:
        writer.close()
    cut.write_bytes(data[:len(data) // 3])
    longer.write_bytes(data + b'x')
    assert _start_sweep(tmp_path) == [str(cut)]
    assert longer.exists() and source.read_bytes() == data
    # a prefix of a source that no longer holds the recorded generation is never proof
    again = tmp_path / '.export.jsonl.again.tmp'
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    try:
        writer.record_uncommitted_writes(run, [(target, _sha(source), len(data), False, {'temporary': again, 'source': source})],
                                         wrapper_stage='export')
    finally:
        writer.close()
    again.write_bytes(data[:100])
    source.write_bytes(data[:100] + b'changed' + data[107:])
    assert _start_sweep(tmp_path) == []
    assert again.exists()


def test_sweep_retains_a_temporary_whose_records_bind_contradictory_locks(tmp_path):
    """Two records of the same path in the same run binding different locks: not acted on (``_agreed_lock``)."""
    temporary = tmp_path / 'download_results' / 'download_results.parquet.conflict.tmp'
    run, path = _seed(tmp_path, temporary=temporary)
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    try:
        canonical = ps.metadata_lock_path(path)
        for lock in (canonical, canonical.with_name('other.lock')):
            writer.record_uncommitted_writes(run, [(path, _sha(path), path.stat().st_size, True, {'lock': lock})], wrapper_stage='clean')
    finally:
        writer.close()
    shutil.copyfile(path, temporary)
    assert _start_sweep(tmp_path) == []
    assert temporary.exists()


def test_sweep_retains_a_temporary_whose_recorded_lock_is_stale(tmp_path):
    """A recorded lock the path's writers no longer take is a refusal in ``_recover_locked``: retained, never blocks."""
    temporary = tmp_path / 'download_results' / 'download_results.parquet.stale.tmp'
    run, path = _seed(tmp_path, temporary=temporary, lock=tmp_path / 'download_results' / 'not-the-canonical.lock')
    shutil.copyfile(path, temporary)
    assert _start_sweep(tmp_path) == []
    assert temporary.exists()


@pytest.mark.parametrize('protection', ['on', 'off'])
def test_sweep_waits_for_a_live_locked_writer(tmp_path, monkeypatch, protection):
    """A writer still holding the metadata lock owns the recorded temporary and replaces with it before releasing the
    lock. With the lock (``on``) the sweep waits for it and the writer's replace succeeds; ``off`` (the lock bypassed)
    is the control: the sweep deletes the live writer's temporary under it and the replace fails."""
    temporary = tmp_path / 'download_results' / 'download_results.parquet.live.tmp'
    run, path = _seed(tmp_path, temporary=temporary)
    shutil.copyfile(path, temporary)
    if protection == 'off':
        def unlocked(target, decide, **kwargs):
            action = decide(Path(target))
            if action is not None and action[0] == 'remove':
                os.unlink(action[1])
            return action
        monkeypatch.setattr(W, '_recover_locked', unlocked)
    holding, finished, outcome = threading.Event(), threading.Event(), {}

    def live_writer():
        with ps.metadata_target(path) as physical:
            holding.set()
            finished.wait(timeout=1.5)
            try:
                os.replace(temporary, physical)
                outcome['replace'] = 'ok'
            except FileNotFoundError as error:
                outcome['replace'] = repr(error)
    thread = threading.Thread(target=live_writer)
    thread.start()
    try:
        assert holding.wait(timeout=10)
        _start_sweep(tmp_path)
        finished.set()
        thread.join(timeout=20)
    finally:
        finished.set()
        thread.join(timeout=20)
    if protection == 'on':
        assert outcome == {'replace': 'ok'}, outcome
    else:
        assert outcome['replace'].startswith('FileNotFoundError'), outcome


@pytest.mark.parametrize('code', [errno.ENOSPC, errno.EDQUOT])
def test_sweep_propagates_an_exhausted_disk(tmp_path, monkeypatch, code):
    """An exhausted filesystem inside the sweep is never turned into "retained": it propagates (``_must_propagate``)."""
    temporary = tmp_path / 'download_results' / 'download_results.parquet.full.tmp'
    run, path = _seed(tmp_path, temporary=temporary)
    shutil.copyfile(path, temporary)

    def full(*args, **kwargs):
        raise OSError(code, 'injected')
    monkeypatch.setattr(W, '_verified_temporary', full)
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    try:
        with pytest.raises(OSError) as raised:
            writer.sweep_abandoned_temporaries()
    finally:
        writer.close()
    assert raised.value.errno == code and temporary.exists()


def test_sweep_ignores_records_without_a_temporary_and_unreadable_records(tmp_path):
    """Records that bind no temporary (older ledgers) and missing temporaries are nothing to do."""
    run, path = _seed(tmp_path)
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    try:
        writer.record_uncommitted_writes(run, [(path, _sha(path), path.stat().st_size, True,
                                                {'temporary': tmp_path / 'download_results' / 'gone.tmp'})], wrapper_stage='clean')
    finally:
        writer.close()
    assert _start_sweep(tmp_path) == []
    assert path.exists()


# ============================================================ K5A2A-G5: the math raster document is closed, locked ===

def test_page_raster_cache_closes_its_document_under_the_pdfium_lock(tmp_path, monkeypatch):
    """``PageRasterCache.close`` closes the PDF inside ``pdfium_guard`` (P7: no destruction left to the garbage
    collector) and is idempotent. (The module imports Docling's document model: the docling lane runs this.)"""
    pytest.importorskip("docling_core")
    from contextlib import contextmanager
    import glossapi.pdfium_guard as guard
    from glossapi.ocr.math.enrich import PageRasterCache
    from pdf_fixtures import FixturePdf
    fx = FixturePdf()
    fx.insert_text(fx.new_page(200, 200), (20, 40), "raster", fontsize=10)
    path = fx.save(tmp_path / "r.pdf")
    raster = PageRasterCache(path)
    assert raster.get_pil(1, 72).size[0] > 0
    real, held = guard.pdfium_guard, []
    document = raster.doc
    closed = []
    original_close = document.close

    def recording_close():
        closed.append(bool(held))
        original_close()
    document.close = recording_close

    @contextmanager
    def recording_guard():
        with real():
            held.append(1)
            try:
                yield
            finally:
                held.pop()
    monkeypatch.setattr(guard, "pdfium_guard", recording_guard)
    raster.close()
    raster.close()
    assert closed == [True] and raster.doc is None and raster.cache == {}


@pytest.mark.parametrize("has_region", [False, True])
def test_math_enrichment_closes_its_raster_on_every_path(tmp_path, monkeypatch, has_region):
    """Both the no-region return and the inference path close the raster cache they opened."""
    pytest.importorskip("docling_core")
    pytest.importorskip("docling")
    from types import ModuleType, SimpleNamespace
    from PIL import Image
    from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, DoclingDocument, ProvenanceItem
    from glossapi.ocr.math import enrich as enrichment
    from tests.test_ledger_stages import simple_pdf
    code_formula = ModuleType("docling.models.code_formula_model")
    code_formula.CodeFormulaModelOptions = lambda **options: SimpleNamespace(**options)

    class StubModel:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, document, batch):
            for element in batch:
                element.item.text = "x = 1"
                yield element.item
    code_formula.CodeFormulaModel = StubModel
    monkeypatch.setitem(sys.modules, "docling.models.code_formula_model", code_formula)
    opened = []

    class RecordingRaster:
        def __init__(self, *args, **kwargs):
            self.closed = 0
            opened.append(self)

        def get_pil(self, *args, **kwargs):
            return Image.new("RGB", (200, 200), "white")

        def close(self):
            self.closed += 1
    monkeypatch.setattr(enrichment, "PageRasterCache", RecordingRaster)
    source = simple_pdf(tmp_path / "source.pdf")
    document = DoclingDocument(name="k5d-math")
    document.add_text(label=DocItemLabel.TEXT, text="BASE")
    if has_region:
        document.add_text(label=DocItemLabel.FORMULA, text="old", prov=ProvenanceItem(page_no=1,
                          bbox=BoundingBox(l=10, t=40, r=80, b=10, coord_origin=CoordOrigin.BOTTOMLEFT), charspan=(0, 3)))
    json_path = tmp_path / "source.docling.json"
    document.save_as_json(json_path)
    enrichment.enrich_from_docling_json(json_path, source, tmp_path / "out.md", tmp_path / "map.jsonl", device="cpu")
    assert [raster.closed for raster in opened] == [1]
