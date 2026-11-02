"""K4(c) part 3: coordinator decisions 1, 2 and 4 (hash reuse within one run, aggregate clean selection, VACUUM)."""
import hashlib
import time
import json
import os
from pathlib import Path

import pytest

from glossapi.ledger.encoding import file_hash, forget_own_hash, remember_own_hash, reused_own_hash, run_hash_scope


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def test_an_own_file_keeps_its_recorded_hash_only_inside_one_run(tmp_path):
    own, original = tmp_path/'own.md', tmp_path/'original.pdf'
    own.write_bytes(b'written by glossapi'); original.write_bytes(b'supplied by the user')
    sentinel = 'f'*64  # a registered value proves the bytes were not read again
    remember_own_hash(own, sentinel)  # outside a run: no registry
    assert file_hash(own) == hashlib.sha256(own.read_bytes()).hexdigest()
    with run_hash_scope():
        remember_own_hash(own, sentinel, own.stat().st_size)
        assert file_hash(own, own=True) == sentinel
        # A caller-supplied file (own=False) never consults the registry, even for a registered inode (CP-K4b F3).
        assert file_hash(own) == hashlib.sha256(own.read_bytes()).hexdigest()
        # Never registered (a user original): read at every boundary.
        assert file_hash(original) == hashlib.sha256(original.read_bytes()).hexdigest()
        assert reused_own_hash(original.stat()) is None
        with run_hash_scope():  # a nested public call keeps the run's registry
            assert file_hash(own, own=True) == sentinel
        # Reopened for writing (size or mtime changes the key): read again.
        own.write_bytes(b'rewritten by glossapi, longer')
        assert file_hash(own, own=True) == hashlib.sha256(own.read_bytes()).hexdigest()
        digest = file_hash(own, own=True)
        assert reused_own_hash(own.stat()) == digest
        forget_own_hash(own)
        assert reused_own_hash(own.stat()) is None
        remember_own_hash(own, sentinel, size=1)  # size mismatch: refused
        assert reused_own_hash(own.stat()) is None
    # The next run start re-checks everything.
    assert file_hash(own) == hashlib.sha256(own.read_bytes()).hexdigest()


def test_freeze_reuses_producer_bound_inputs_and_hashes_originals(tmp_path):
    from glossapi.stages.contracts import ArtifactReference, StageRequest
    from glossapi.stages.execution import freeze_inputs, remember_written_hash
    produced, original = tmp_path/'produced.md', tmp_path/'source.pdf'
    produced.write_bytes(b'producer output'); original.write_bytes(b'%PDF user bytes')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    request = lambda work, references: StageRequest('t', 'a', 'l', 'd', 'v', 'op', tuple(references), {}, {}, 'cpu', work, 0)
    with run_hash_scope():
        remember_written_hash(produced, digest(produced), produced.stat().st_size)
        frozen = freeze_inputs(request(tmp_path/'w1', [
            ArtifactReference('f1', 'raw_markdown', produced, digest(produced), produced.stat().st_size),
            ArtifactReference('f2', 'source_payload', original, digest(original), original.stat().st_size)]))
        assert [path.read_bytes() for path in (item.path for item in frozen.inputs)] == [b'producer output', b'%PDF user bytes']
        # The original entered the attempt hashed and is never registered for reuse.
        assert reused_own_hash(original.stat()) is None
        # A caller-supplied text (any kind) that glossAPI did not write is hashed and not registered either.
        given = tmp_path/'given.md'; given.write_bytes(b'user text')
        freeze_inputs(request(tmp_path/'w3', [ArtifactReference('f3', 'selected_text', given, digest(given), given.stat().st_size)]))
        assert reused_own_hash(given.stat()) is None
        assert reused_own_hash(frozen.inputs[0].path.stat()) == digest(produced)
        # A registered input whose recorded hash differs from the binding is refused, never trusted.
        with pytest.raises(ValueError, match='input_hash_changed'):
            freeze_inputs(request(tmp_path/'w2', [ArtifactReference('f1', 'raw_markdown', produced, '0'*64, produced.stat().st_size)]))


def _two_documents(tmp_path):
    from test_ledger_fixset_lg import pdf
    return [pdf(tmp_path/'in'/f'{name}.pdf', f'Section heading {name} and a long enough body text for one section')
            for name in ('alpha', 'beta')]


def test_one_aggregate_clean_selection_keeps_a_selection_and_reason_per_document(tmp_path):
    pytest.importorskip('glossapi_rs_cleaner')
    from glossapi import Corpus
    from glossapi.ledger.reader import report_why, select_run
    from glossapi.ledger.writer import open_readonly
    sources = _two_documents(tmp_path)
    corpus = Corpus(tmp_path/'empty', tmp_path/'out')
    result = corpus.run(sources, stages=('extract', 'clean', 'sections', 'export'), settings={'extract': {'repair_fonts': False}})
    assert result['state'] == 'completed', result
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
        tasks = connection.execute("SELECT task_id,version_id FROM tasks WHERE operation='clean.selection'").fetchall()
        assert len(tasks) == 1 and tasks[0][1] is None
        inputs = connection.execute('SELECT COUNT(*) FROM attempt_inputs i JOIN attempts a USING(attempt_id) WHERE a.task_id=?',
                                    (tasks[0][0],)).fetchone()[0]
        assert inputs == 2
        rows = {}
        for source in sources:
            why = report_why(connection, str(source), run=select_run(connection, result['run_id']))
            stage = [item for item in why['stages'] if item['operation'] == 'clean.selection']
            assert len(stage) == 1 and stage[0]['status'] == 'ok', stage
            decisions = [decision for attempt in stage[0]['attempts'] for decision in attempt['decisions']]
            assert decisions and all(decision['reason_code'] == 'clean_completed' for decision in decisions)
            selections = [row for decision in decisions for row in decision['details']['selections']]
            # Its own row only (the other document's row is not shown), with thresholds and policy.
            assert [row['compatibility_filename'] for row in selections] == [source.name], selections
            assert decisions[0]['details']['selection_policy'] == 'legacy_needs_ocr_false_when_drop_bad'
            assert 'effective_thresholds' in decisions[0]['details']
            other = next(item for item in sources if item != source)
            assert other.name not in json.dumps(stage)  # (other corpus aggregates list every document already)
            rows[source.name] = selections[0]
        assert all(row['accepted_for_sections'] is True and row['needs_ocr'] is False for row in rows.values())
    # The sections barrier restored the selection from the aggregate rows.
    assert {json.loads(line)['filename'] for line in (corpus.output_dir/'export.jsonl').read_text().splitlines()} == {
        source.stem for source in sources}


def test_run_close_vacuum_keeps_every_record_and_order(tmp_path, monkeypatch):
    pytest.importorskip('glossapi_rs_cleaner')
    import sqlite3
    from glossapi import Corpus
    from glossapi.ledger.writer import LedgerWriter
    executed = []
    original = LedgerWriter._vacuum_at_run_close
    def vacuum(self):
        before = {table: list(self.connection.execute(f'SELECT rowid,* FROM "{table}" ORDER BY rowid'))
                  for (table,) in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        original(self)
        after = {table: list(self.connection.execute(f'SELECT rowid,* FROM "{table}" ORDER BY rowid')) for table in before}
        executed.append(before == after)
    monkeypatch.setattr(LedgerWriter, '_vacuum_at_run_close', vacuum)
    corpus = Corpus(tmp_path/'empty', tmp_path/'out')
    result = corpus.run(_two_documents(tmp_path), stages=('extract', 'clean'), settings={'extract': {'repair_fonts': False}})
    assert result['state'] == 'completed'
    assert executed and all(executed)
    assert not os.path.exists(str(corpus.output_dir/'corpus.sqlite3')+'-wal')
    connection = sqlite3.connect(corpus.output_dir/'corpus.sqlite3')
    try:
        assert connection.execute('PRAGMA freelist_count').fetchone()[0] == 0
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    finally:
        connection.close()


def test_legacy_model_preloaded_during_the_dag_is_the_object_the_barrier_loads(tmp_path):
    pytest.importorskip('sklearn')
    import shutil
    from pathlib import Path
    import glossapi
    from glossapi import gloss_section_classifier as classifier
    source = Path(glossapi.__file__).parent/'models/section_classifier.joblib'
    frozen = tmp_path/'frozen.joblib'; shutil.copyfile(source, frozen)  # the barrier's frozen copy: same bytes
    try:
        thread = classifier.preload_legacy_model(source)
        assert thread is not None and classifier.preload_legacy_model(source) is thread  # one load per run
        thread.join()
        preloaded, _ = classifier._preloaded(frozen)
        assert preloaded is not None
        loaded = classifier.GlossSectionClassifier().load_model(str(frozen))
        assert loaded is preloaded
        other = tmp_path/'other.joblib'; other.write_bytes(source.read_bytes()+b'\0')  # other bytes: never the preloaded object
        assert classifier._preloaded(other)[0] is None
    finally:
        classifier.release_preloaded_models()
    assert classifier._preloaded(frozen) == (None, None)
    assert classifier.preload_legacy_model(Path(glossapi.__file__).parent/'models/section_classifier') is None  # neutral: no-op


# K4(c) part 4: coordinator decision on K4C3-G4.

def test_originals_are_hashed_by_reading_stages_and_bound_by_the_others(tmp_path, monkeypatch):
    from glossapi.ledger import journal
    original = tmp_path/'source.pdf'; original.write_bytes(b'%PDF user bytes')
    time.sleep(journal.RACY_MARGIN_NS/1e9 + 0.1)  # an original older than the racy margin (K4F1-G1) is cached
    monkeypatch.setattr(journal, '_original_hashes', {})
    reads = []
    real = journal.file_hash
    monkeypatch.setattr(journal, 'file_hash', lambda path, **k: reads.append(str(path)) or real(path, **k))
    expected = hashlib.sha256(original.read_bytes()).hexdigest()
    # No reading stage ran in this process: a non-reading phase hashes and records nothing.
    assert journal.observed_original_hash('clean', original) == expected and len(reads) == 1
    assert journal._original_hashes == {}
    assert journal.observed_original_hash('extract', original) == expected and len(reads) == 2
    for stage in ('clean', 'sections', 'annotate', 'export'):
        assert journal.observed_original_hash(stage, original) == expected
    assert len(reads) == 2  # bound to extract's recorded hash, not read again
    assert journal.observed_original_hash('ocr', original) == expected and len(reads) == 3  # a reading stage reads
    original.write_bytes(b'%PDF user bytes, changed')  # any stat change: hashed again
    assert journal.observed_original_hash('export', original) == hashlib.sha256(original.read_bytes()).hexdigest()
    assert len(reads) == 4


def test_full_integrity_check_once_per_process_until_the_file_changes(tmp_path, monkeypatch):
    from glossapi.ledger import writer as W
    path = tmp_path/'corpus.sqlite3'
    calls = []
    real = W.integrity
    monkeypatch.setattr(W, 'integrity', lambda connection: calls.append(1) or real(connection))
    monkeypatch.setattr(W, '_integrity_known', {})
    with W.LedgerWriter(path) as writer:
        with writer.transaction():
            writer.run({}, {}, {}, code_revision='test')
    assert len(calls) == 1
    with W.LedgerWriter(path):
        pass
    assert len(calls) == 1  # same process, file unchanged since this process closed it
    stat = path.stat(); os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))  # changed by someone else
    with W.LedgerWriter(path):
        pass
    assert len(calls) == 2
    W._integrity_known.clear()  # a new process knows nothing
    with W.LedgerWriter(path):
        pass
    assert len(calls) == 3
    Path(str(path)+'-wal').write_bytes(b'')  # a leftover WAL: full check
    try:
        with W.LedgerWriter(path):
            pass
    finally:
        Path(str(path)+'-wal').unlink(missing_ok=True)
    assert len(calls) == 4
