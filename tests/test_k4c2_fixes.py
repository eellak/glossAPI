"""K4(c) part 2: K4C-H1 (resumed publication), K4C-M1 (extreme admission), K4C-M3 (dictionary identity),
K4C-M2 (excluded corpus copies) and the schema-v4 index migration."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from glossapi.ledger import LedgerWriter
from glossapi.stages import disk_budget
from glossapi.stages.disk_budget import DiskBudget, DiskBudgetError, require_disk_guard
from glossapi.stages.execution import sha256_file


def extreme(**overrides):
    return {**dict(debug='off', memory_saving='extreme', disk_budget_bytes=8*1024**3, max_source_bytes=256*1024**2,
                   max_attempt_bytes=600*1024**2), **overrides}


def run_fixture(writer, settings=None):
    with writer.transaction():
        return writer.run(settings or {}, {}, {}, code_revision='k4c2-test')


# K4C-M1 ------------------------------------------------------------------------------------------------------------

def fake_filesystems(monkeypatch, root, *, corpus, staging):
    """Pretend root and root/.glossapi/staging are two mounted filesystems of the given capacities."""
    staging_path = Path(root).resolve()/'.glossapi/staging'
    staging_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(disk_budget.sys, 'platform', 'linux')
    monkeypatch.setattr(Path, 'is_mount', lambda self: True)
    monkeypatch.setattr(disk_budget, 'filesystem_capacity', lambda path: staging if Path(path) == staging_path else corpus)
    real_stat = Path.stat
    def stat(self, *args, **kwargs):
        value = real_stat(self, *args, **kwargs)
        if self == staging_path:
            fields = {name: getattr(value, name) for name in dir(value) if name.startswith('st_')}
            return SimpleNamespace(**{**fields, 'st_dev': value.st_dev+1})
        return value
    monkeypatch.setattr(Path, 'stat', stat)


def test_guard_refuses_a_corpus_filesystem_that_can_never_admit_an_attempt(tmp_path, monkeypatch):
    mib = 1024**2
    # The part-1 extreme gate filesystems: corpus 1.5 GiB, staging 0.6 GiB, attempt cap = staging capacity.
    fake_filesystems(monkeypatch, tmp_path, corpus=1536*mib, staging=600*mib)
    settings = extreme(disk_budget_bytes=2136*mib, max_attempt_bytes=600*mib)
    with pytest.raises(DiskBudgetError, match='disk_guard_unavailable.*no attempt could be admitted'):
        require_disk_guard(tmp_path, settings)
    # One attempt reservation (2 x 600 MiB) plus the emergency reserve (1 GiB + 64 MiB) fits in 3 GiB.
    fake_filesystems(monkeypatch, tmp_path, corpus=3072*mib, staging=600*mib)
    require_disk_guard(tmp_path, extreme(disk_budget_bytes=3672*mib, max_attempt_bytes=600*mib))


def test_extreme_admission_holds_staging_for_one_attempt_and_never_needs_an_empty_filesystem(tmp_path, monkeypatch):
    mib = 1024**2
    settings = extreme(disk_budget_bytes=3672*mib, max_attempt_bytes=600*mib)
    real = os.statvfs
    def statvfs(path):
        if str(path).endswith('.glossapi/staging'):
            # A staging filesystem never reports its whole capacity free (metadata, lost+found).
            return SimpleNamespace(f_bavail=(600*mib-64*1024)//4096, f_frsize=4096, f_blocks=600*mib//4096)
        if Path(path) == tmp_path.resolve():
            return SimpleNamespace(f_bavail=3000*mib//4096, f_frsize=4096, f_blocks=3072*mib//4096)
        return real(path)
    monkeypatch.setattr(os, 'statvfs', statvfs)
    (tmp_path/'.glossapi/staging').mkdir(parents=True)
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run = run_fixture(writer, settings)
        budget = DiskBudget(writer, tmp_path, run, settings)
        amount = disk_budget.attempt_reservation(settings)
        assert amount == 1200*mib
        # Before K4C-M1 this was False for ever: free staging bytes < max_attempt_bytes even when empty.
        assert budget.can_reserve(amount)
        budget.reserve('first', amount)
        assert not budget.can_reserve(amount)  # staging belongs to the admitted attempt
        budget.release('first')
        assert budget.can_reserve(amount)


# K4C-H1 ------------------------------------------------------------------------------------------------------------

def test_resumed_publication_reuses_its_registration_and_closes_the_leftover_observation(tmp_path):
    from glossapi.stages.compatibility import register_publication
    product = tmp_path/'markdown/doc.md'; product.parent.mkdir(); product.write_text('raw text')
    leftover = tmp_path/'markdown/left.md'; leftover.write_text('left text')
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='k4c2-test')
            document = writer.document('local_path:v1:'+product.as_uri())
            version = writer.version(document, source_sha256='a'*64, stored_sha256='a'*64)
            task = writer.task(run, 'extract', 'extract.document', {}, version_id=version)
            attempt = writer.attempt(task, worker_id='w', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
            # The killed run committed this product's registration; the other product's bytes landed but its
            # registration did not, and the resumed wrapper inventory observed them as an unowned file.
            first = register_publication(writer, run, 'extract', 'raw_markdown', product.as_uri(), version_id=version,
                producer_attempt_id=attempt, sha256=sha256_file(product), bytes=product.stat().st_size, retention_class='intermediate')
            observed = writer.artifact(run, 'extract', 'markdown/left.md', leftover.as_uri(), sha256=sha256_file(leftover),
                bytes=leftover.stat().st_size, ownership='legacy_external')
        with writer.transaction():
            again = register_publication(writer, run, 'extract', 'raw_markdown', product.as_uri(), version_id=version,
                producer_attempt_id=attempt, sha256=sha256_file(product), bytes=product.stat().st_size, retention_class='intermediate')
            claimed = register_publication(writer, run, 'extract', 'raw_markdown', leftover.as_uri(), version_id=version,
                producer_attempt_id=attempt, sha256=sha256_file(leftover), bytes=leftover.stat().st_size, retention_class='intermediate')
        assert again == first
        present = lambda uri: [row[0] for row in writer.connection.execute(
            "SELECT file_id FROM current_file_state WHERE uri=? AND presence='present'", (uri,))]
        assert present(product.as_uri()) == [first]
        assert present(leftover.as_uri()) == [claimed]
        assert writer.connection.execute('SELECT presence FROM current_file_state WHERE file_id=?', (observed,)).fetchone()[0] == 'missing'


# K4C-M2 ------------------------------------------------------------------------------------------------------------

def published_copy(writer, root, *, torn=False):
    """A retained managed source in .glossapi/artifacts and its user-visible byte copy (same producer, same hash).

    ``torn``: the copy's bytes on disk are already incomplete when the run-close barrier records it, as a power loss
    leaves an unflushed copy (its mtime/ctime stay those of the write before the close; CP-K4a F6). A userland
    truncation after the close would advance ctime and count as a later edit."""
    source = root/'.glossapi/artifacts/corpus/export/attempt/export.jsonl'; source.parent.mkdir(parents=True)
    source.write_text('{"a": 1}\n'*1000)
    copy = root/'export.jsonl'; copy.write_bytes(source.read_bytes())
    with writer.transaction():
        run = writer.run({}, {}, {}, code_revision='k4c2-test')
        manifest = writer.snapshot('inputs', {'kind': 'inputs', 'files': []})
        task = writer.task(run, 'export', 'export.jsonl', {}, scope_key='corpus:'+manifest)
        attempt = writer.attempt(task, worker_id='w', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        digest, size = sha256_file(source), source.stat().st_size
        source_id = writer.artifact(run, 'export', 'document_jsonl', source.as_uri(), producer_attempt_id=attempt, sha256=digest,
                                    bytes=size, retention_class='final')
        copy_id = writer.artifact(run, 'export', 'document_jsonl', copy.as_uri(), producer_attempt_id=attempt, sha256=digest,
                                  bytes=size, retention_class='final')
    if torn:
        with copy.open('r+b') as stream: stream.truncate(100)
    writer.barrier_context = {'run_id': run, 'debug': 'off'}
    writer.durable_barrier()
    return run, source, copy, source_id, copy_id


def test_run_close_names_unflushed_copies_and_the_next_start_republishes_a_torn_one(tmp_path):
    import json
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, source, copy, source_id, copy_id = published_copy(writer, tmp_path, torn=True)
        summary = json.loads(writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} "
                                                       "WHERE message='barrier_summary'").fetchone()[0])
        assert summary['run_close_excluded_copy_file_ids'] == [copy_id]
    # Power loss after close: the unflushed copy is incomplete; its bytes and times are those of before the close.
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        assert writer.verify_excluded_copies() == [copy_id]
        assert copy.read_bytes() == source.read_bytes()
        assert writer.connection.execute('SELECT presence FROM current_file_state WHERE file_id=?', (copy_id,)).fetchone()[0] == 'present'
        assert writer.verify_excluded_copies() == []  # intact now: verified and made durable, nothing to restore


def test_next_start_never_overwrites_a_copy_changed_after_the_run_close(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, source, copy, source_id, copy_id = published_copy(writer, tmp_path)
    copy.write_text('edited by the user\n')
    future = copy.stat().st_mtime_ns+3600_000_000_000
    os.utime(copy, ns=(future, future))
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        assert writer.verify_excluded_copies() == []
    assert copy.read_text() == 'edited by the user\n'


# K4C-M3 and schema v4 ----------------------------------------------------------------------------------------------

def indexes(connection):
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")}


def test_v4_header_records_the_dictionary_and_drops_unused_indexes(tmp_path):
    from glossapi.ledger.json_dictionary import DICTIONARY_ID, DICTIONARY_SHA256
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run_fixture(writer)
        assert writer.connection.execute('PRAGMA user_version').fetchone()[0] == 4
        assert tuple(writer.connection.execute('SELECT json_dictionary_id,json_dictionary_sha256 FROM ledger_format').fetchone()) == (DICTIONARY_ID, DICTIONARY_SHA256)
        names = indexes(writer.connection)
    assert not names & {'events_document', 'files_version', 'files_stage_kind'}
    assert {'files_hash', 'files_export_kind', 'events_file', 'files_producer'} <= names


def test_a_ledger_naming_an_unknown_dictionary_is_refused_with_a_typed_error(tmp_path):
    import sqlite3
    from glossapi.ledger.json_dictionary import UnknownJsonDictionary
    from glossapi.ledger.writer import open_readonly
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run_fixture(writer)
    raw = sqlite3.connect(tmp_path/'corpus.sqlite3')
    raw.execute("UPDATE ledger_format SET json_dictionary_id='glossapi-json-v9', json_dictionary_sha256=?", ('0'*64,))
    raw.commit(); raw.close()
    for opener in (lambda: LedgerWriter(tmp_path/'corpus.sqlite3'), lambda: open_readonly(tmp_path/'corpus.sqlite3').__enter__()):
        with pytest.raises(UnknownJsonDictionary) as caught:
            opener()
        assert caught.value.code == 'unknown_json_dictionary' and caught.value.identifier == 'glossapi-json-v9'


def test_an_old_v3_ledger_opens_reads_writes_and_migrates_to_v4(tmp_path):
    import hashlib
    from glossapi.ledger.schema import FORMAT_SCHEMA_SQL
    from glossapi.ledger.writer import integrity, open_readonly
    path = tmp_path/'corpus.sqlite3'
    with LedgerWriter(path, storage_version=3) as writer:
        run = run_fixture(writer)
        with writer.transaction():
            writer.event(run, event_kind='run', reason_code='legacy_observed', details={'value': 'x'*200})
    with LedgerWriter(path) as writer:  # the new code opens the old ledger in place, unchanged in version
        assert writer.connection.execute('PRAGMA user_version').fetchone()[0] == 3
        assert {'events_document', 'files_version', 'files_hash', 'files_stage_kind'} <= indexes(writer.connection)
        with writer.transaction():
            writer.event(run, event_kind='run', reason_code='legacy_observed', details={'value': 'y'*200})
    with open_readonly(path) as connection:
        integrity(connection)
        assert connection.execute(f"SELECT COUNT(*) FROM {connection.event_table} WHERE event_kind='run'").fetchone()[0] == 2
    with LedgerWriter(path) as writer:
        writer.migrate(4, FORMAT_SCHEMA_SQL, checksum=hashlib.sha256(FORMAT_SCHEMA_SQL.encode()).hexdigest(),
                       code_revision='k4c2-test', backup_path=tmp_path/'backup-v3.sqlite3')
    with open_readonly(path) as connection:
        integrity(connection)
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 4
        assert connection.execute('SELECT json_dictionary_id FROM ledger_format').fetchone()[0] == 'glossapi-json-v1'
        assert not indexes(connection) & {'events_document', 'files_version', 'files_stage_kind'}
        details = [connection.execute(f"SELECT details_json FROM {connection.event_table} WHERE event_kind='run' ORDER BY event_id").fetchall()]
        assert len(details[0]) == 2
    with open_readonly(tmp_path/'backup-v3.sqlite3') as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 3


def test_a_new_writer_starts_from_what_the_last_run_close_made_durable(tmp_path):
    import json
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, source, copy, source_id, copy_id = published_copy(writer, tmp_path)  # closes with a run-close barrier
    late = tmp_path/'late.md'; late.write_text('registered by the next phase')
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        assert source_id in writer._durable_files and copy_id not in writer._durable_files  # the copy was excluded
        writer.verify_excluded_copies()
        assert copy_id in writer._durable_files
        with writer.transaction():
            run2 = writer.run({}, {}, {}, code_revision='k4c2-test')
            late_id = writer.artifact(run2, 'export', 'document_jsonl', late.as_uri(), sha256=sha256_file(late),
                                      bytes=late.stat().st_size, retention_class='final')
        writer.barrier_context = {'run_id': run2, 'debug': 'off'}
        writer.durable_barrier()
        summary = json.loads(writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} "
                                                       "WHERE message='barrier_summary' ORDER BY event_id DESC LIMIT 1").fetchone()[0])
        assert summary['by_reason']['run_close']['file_count'] == 1  # only the new registration needed a flush
        assert late_id in writer._durable_files
