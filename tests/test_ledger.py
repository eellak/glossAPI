"""Shadow ledger durability, schema and boundary contracts."""
import datetime
import decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid

import pytest

from glossapi.ledger import LedgerWriter, LedgerError, open_readonly
from glossapi.ledger.encoding import canonical_json, typed_value, untyped_value
from glossapi.ledger.schema import APPEND_ONLY_TABLES, SCHEMA_VERSION


def start(writer, stage="extract", specification=None):
    run = writer.run({}, {}, {}, code_revision="test")
    document = writer.document("local_path:v1:file:///fixture.pdf")
    version = writer.version(document)
    task = writer.task(run, stage, stage + ".test", specification or {}, version_id=version)
    attempt = writer.attempt(task, worker_id="test", device_id="cpu", fingerprint={}, inputs={}, settings={}, tools={})
    writer.event(run, event_kind="lifecycle", reason_code="stage_started", status="running", task_id=task, version_id=version, attempt_id=attempt)
    return run, document, version, task, attempt


def test_schema_pragmas_identity_and_append_only(tmp_path):
    with LedgerWriter(tmp_path / "corpus.sqlite3") as writer:
        assert writer.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert writer.connection.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert writer.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert writer.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert writer.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        with writer.transaction():
            run, document, version, task, attempt = start(writer)
            writer.source_record(document, run, "file:///fixture.pdf", {"value": None})
            writer.artifact(run, "extract", "markdown", "file:///fixture.md", sha256="a"*64, bytes=0, version_id=version, producer_attempt_id=attempt)
        for table in APPEND_ONLY_TABLES:
            for sql in (f"DELETE FROM {table}", f"UPDATE {table} SET {'event_id=event_id' if table=='events' else 'rowid=rowid'}"):
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    with writer.transaction(): writer.connection.execute(sql)
        with pytest.raises(sqlite3.IntegrityError):
            with writer.transaction(): writer.connection.execute("INSERT INTO document_versions VALUES(?,?,1,NULL,NULL,'now')", (str(uuid.uuid4()), str(uuid.uuid4())))
        with pytest.raises(LedgerError):
            with writer.transaction(): writer.task(run, "partial", "bad", {})
        for column, value in (("status", "partial"), ("event_kind", "log")):
            with pytest.raises(sqlite3.IntegrityError):
                with writer.transaction():
                    writer.connection.execute(f"INSERT INTO events(run_id,event_kind,reason_code,message,details_json,recorded_at,{column if column == 'status' else 'status'}) VALUES(?,?,?,?,?,?,?)", (run, "log" if column == "event_kind" else "decision", "unknown_not_recorded", "", "{}", "now", value if column == "status" else None))
        with pytest.raises(LedgerError):
            with writer.transaction(): writer.event(run, event_kind="decision", reason_code="invented")
        with pytest.raises(ValueError):
            with writer.transaction(): writer.snapshot("bad", {"value": float("nan")})


def test_references_inputs_leases_transitions_and_atomic_outputs(tmp_path):
    with LedgerWriter(tmp_path / "corpus.sqlite3") as writer:
        with writer.transaction(): run, document, version, task, attempt = start(writer)
        with pytest.raises(LedgerError, match="active"):
            with writer.transaction(): writer.attempt(task, worker_id="other", device_id="cpu", fingerprint={}, inputs={}, settings={}, tools={})
        with pytest.raises(LedgerError, match="mismatch"):
            with writer.transaction(): writer.event(str(uuid.uuid4()), event_kind="decision", reason_code="legacy_observed", task_id=task, version_id=version, attempt_id=attempt)
        with writer.transaction():
            artifact = writer.artifact(run, "extract", "markdown", "file:///fixture.md", sha256="a"*64, bytes=5, version_id=version)
        with pytest.raises(LedgerError, match="Input hash"):
            with writer.transaction(): writer.input(attempt, artifact, "text", "b"*64)
        with pytest.raises(LedgerError, match="Required output"):
            with writer.transaction(): writer.finish(run, task, attempt, status="ok", reason_code="extract_completed", required_kinds=("markdown",))
        with writer.transaction():
            writer.finish(run, task, attempt, status="failed", reason_code="extract_exception")
        with pytest.raises(LedgerError, match="terminal"):
            with writer.transaction(): writer.event(run, event_kind="lifecycle", status="ok", reason_code="extract_completed", task_id=task, version_id=version, attempt_id=attempt)
        assert writer.connection.execute("SELECT status FROM current_task_state").fetchone()[0] == "failed"
        with writer.transaction():
            second = writer.attempt(task, worker_id="other", device_id="cpu", fingerprint={}, inputs={}, settings={}, tools={})
            writer.finish(run, task, second, status="ok", reason_code="extract_completed", outputs=({"kind":"markdown","uri":"file:///new.md","sha256":"b"*64,"bytes":0},), required_kinds=("markdown",))
        assert writer.connection.execute("SELECT status FROM current_task_state").fetchone()[0] == "ok"
        assert writer.connection.execute("SELECT COUNT(*) FROM files WHERE producer_attempt_id=?", (second,)).fetchone()[0] == 1


def test_versions_dependencies_and_missing_artifacts(tmp_path):
    with LedgerWriter(tmp_path / "corpus.sqlite3") as writer:
        with writer.transaction():
            run, document, version, task, attempt = start(writer)
            writer.fill_version_hashes(version, source_sha256="a"*64, stored_sha256="b"*64)
            other = writer.task(run, "clean", "clean.test", {}, version_id=version)
            writer.dependency(other, task, "text")
            missing = writer.artifact(run, "extract", "legacy", "file:///missing", version_id=version, ownership="legacy_external", presence="missing")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with writer.transaction(): writer.fill_version_hashes(version, source_sha256="c"*64, stored_sha256="b"*64)
        with pytest.raises(LedgerError, match="Cyclic"):
            with writer.transaction(): writer.dependency(task, other, "wrong")
        with pytest.raises(LedgerError, match="requires hash"):
            with writer.transaction(): writer.artifact(run,"extract","new","file:///new")
        assert writer.connection.execute("SELECT presence FROM current_file_state WHERE file_id=?", (missing,)).fetchone()[0] == "missing"


def test_typed_source_fields_preserve_duplicates_and_manifest_checksum(tmp_path):
    fields = {"bytes":b"\x00\xff", "date":datetime.date(2026,10,28), "datetime":datetime.datetime(2026,10,28,tzinfo=datetime.timezone.utc), "decimal":decimal.Decimal("1.2300"), "null":None, "nested":[1,True,{"type":"null"}], "big":2**90, "nan":float("nan")}
    encoded = typed_value(fields)
    canonical_json(encoded)
    decoded = untyped_value(encoded)
    assert decoded.pop("nan") != decoded.get("nan")
    fields.pop("nan")
    assert decoded == fields
    with LedgerWriter(tmp_path / "corpus.sqlite3") as writer:
        with writer.transaction():
            run, document, version, task, attempt = start(writer)
            manifest = writer.snapshot("source_manifest", {"original_file_sha256":"f"*64})
            a = writer.source_record(document,run,"https://example/a",fields,source_kind="url",source_row="0",manifest_sha256=manifest)
            b = writer.source_record(document,run,"https://example/a",fields,source_kind="url",source_row="1",manifest_sha256=manifest)
        assert a != b
        assert writer.connection.execute("SELECT COUNT(*) FROM source_records").fetchone()[0] == 2
        assert json.loads(writer.connection.execute("SELECT json FROM snapshots WHERE sha256=?", (manifest,)).fetchone()[0])["original_file_sha256"] != manifest


def test_migration_backup_rollback_and_fail_closed(tmp_path):
    path=tmp_path/"corpus.sqlite3"
    with LedgerWriter(path) as writer:
        with writer.transaction(): start(writer)
        count = writer.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        sql="CREATE TABLE migration_probe(value TEXT);\nINSERT INTO nonexistent VALUES(1);\n"
        with pytest.raises(sqlite3.OperationalError):
            writer.migrate(SCHEMA_VERSION+1,sql,checksum=hashlib.sha256(sql.encode()).hexdigest(),code_revision="test",backup_path=tmp_path/"backup.sqlite3")
        assert writer.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert not writer.connection.execute("SELECT name FROM sqlite_master WHERE name='migration_probe'").fetchone()
        assert writer.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == count
        with open_readonly(tmp_path/"backup.sqlite3") as backup:
            assert backup.execute("SELECT COUNT(*) FROM events").fetchone()[0] == count
        sql="CREATE TABLE migration_probe(value TEXT);\n"
        writer.migrate(SCHEMA_VERSION+1,sql,checksum=hashlib.sha256(sql.encode()).hexdigest(),code_revision="test",backup_path=tmp_path/"backup2.sqlite3")
    with pytest.raises(LedgerError, match="Unsupported"):
        LedgerWriter(path)
    with pytest.raises(LedgerError, match="Unsupported"):
        with open_readonly(path): pass


def test_exclusive_writer_and_rollback(tmp_path):
    with LedgerWriter(tmp_path/"corpus.sqlite3") as writer:
        with pytest.raises(BlockingIOError): LedgerWriter(writer.path)
        with pytest.raises(RuntimeError):
            with writer.transaction():
                start(writer)
                raise RuntimeError("crash")
        assert writer.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


@pytest.mark.parametrize("commit", [False, True])
def test_crash_on_both_sides_of_observation_commit(tmp_path, commit):
    path=tmp_path/"corpus.sqlite3"
    code="""
import os,sys
from glossapi.ledger import LedgerWriter
writer=LedgerWriter(sys.argv[1])
with writer.transaction():
    run=writer.run({}, {}, {},code_revision='crash-fixture')
    writer.event(run,event_kind='run',reason_code='legacy_observed',details={'receipt':'durable'})
    if sys.argv[2]=='before': os._exit(17)
os._exit(18)
"""
    result=subprocess.run([sys.executable,"-c",code,str(path),"after" if commit else "before"])
    assert result.returncode == (18 if commit else 17)
    with LedgerWriter(path) as writer:
        assert writer.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == int(commit)
        assert writer.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_no_heavy_import_contract():
    code="""
import sys
import glossapi.ledger
import glossapi.ledger.journal
for prefix in ('torch','docling','pandas','numpy','pyarrow','sklearn','pypdfium2','glossapi.corpus'):
    assert not any(name==prefix or name.startswith(prefix+'.') for name in sys.modules),prefix
"""
    subprocess.run([sys.executable,"-c",code],check=True)


def test_boundary_preserves_return_files_removal_and_caught_exception(tmp_path):
    from types import SimpleNamespace
    from glossapi.ledger.journal import shadow_stage, observe, observe_store
    output = tmp_path / "output"; output.mkdir()
    inputs = tmp_path / "inputs"; inputs.mkdir()
    sentinel = {"existing_return": (1, "same")}
    context = SimpleNamespace(output_dir=output,input_dir=inputs,downloads_dir=output/"downloads",_shadow_ledger_enabled=True)

    @shadow_stage("clean", "clean.fixture")
    def legacy_boundary(self):
        temporary = self.output_dir / "report.json"
        temporary.write_text('{"measurements":[1,2],"text":"private payload"}')
        observe_store(temporary,kind="clean_report")
        temporary.unlink()
        observe("delete_completed", path=temporary)
        try: raise OSError("backend failed; legacy continues")
        except OSError as error:
            observe("clean_process_failed", error=error, status="failed", continuation="existing_metrics")
        (self.output_dir / "result.md").write_text("unchanged product")
        return sentinel

    assert legacy_boundary(context) is sentinel
    assert {p.name for p in output.iterdir()} == {"corpus.sqlite3", "result.md"}
    with open_readonly(output/"corpus.sqlite3") as read:
        assert read.execute("SELECT status FROM current_task_state").fetchone()[0] == "failed"
        failure = read.execute("SELECT error_type,traceback FROM events WHERE reason_code='clean_process_failed'").fetchone()
        assert failure[0] == "builtins.OSError" and "backend failed" in failure[1]
        assert read.execute("SELECT presence FROM current_file_state WHERE kind='clean_report'").fetchone()[0] == "deleted"
        stored = read.execute("SELECT json FROM snapshots WHERE kind='legacy_store'").fetchone()[0]
        assert "private payload" not in stored and "sha256" in stored
        terminal = json.loads(read.execute("SELECT details_json FROM events WHERE status='failed'").fetchone()[0])
        assert untyped_value(terminal["wrapper_return"]) == sentinel


def test_real_wrapper_skip_signature_and_no_decision_read(tmp_path):
    import inspect
    from glossapi import Corpus
    corpus = Corpus(tmp_path/"input", tmp_path/"output")
    assert inspect.signature(Corpus.extract) == inspect.signature(Corpus.extract.__wrapped__)
    assert corpus.extract(input_format="doc") is None
    with open_readonly(corpus.output_dir/"corpus.sqlite3") as read:
        assert read.execute("SELECT status,reason_code FROM current_task_state").fetchone() == ("skipped", "unsupported_legacy_doc")
    # Deliberately disagreeing shadow facts never suppress the next invocation.
    assert corpus.extract(input_format="pdf") is None
    with open_readonly(corpus.output_dir/"corpus.sqlite3") as read:
        assert read.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        assert read.execute("SELECT reason_code FROM current_task_state ORDER BY ordinal").fetchall()[-1][0] == "no_matching_inputs"


def test_sqlite_backup_is_wal_aware_and_writer_survives_reader(tmp_path):
    with LedgerWriter(tmp_path/"corpus.sqlite3") as writer:
        with writer.transaction(): start(writer)
        with open_readonly(writer.path) as reader:
            assert reader.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        # A second process must fail while the orchestrator owns the database.
        code="from glossapi.ledger import LedgerWriter; import sys; LedgerWriter(sys.argv[1])"
        result=subprocess.run([sys.executable,"-c",code,str(writer.path)],capture_output=True,text=True)
        assert result.returncode != 0 and "BlockingIOError" in result.stderr


def test_source_numpy_arrays_pandas_nulls_and_nanosecond_timestamp():
    import numpy as np
    import pandas as pd
    fields = {"array": np.array([1, 2, 3]), "missing": pd.NA, "not_a_time": pd.NaT,
              "timestamp": pd.Timestamp("2026-10-28T01:02:03.123456789Z"), "scalar": np.int64(7)}
    encoded = typed_value(fields)
    canonical_json(encoded)
    entries = {untyped_value(key): value for key, value in encoded["value"]}
    assert entries["array"]["shape"] == [3] and untyped_value(entries["array"]) == [1,2,3]
    assert entries["missing"]["type"] == entries["not_a_time"]["type"] == "null"
    assert entries["timestamp"]["value"].endswith("03.123456789+00:00")
    assert untyped_value(entries["scalar"]) == 7


def test_workers_only_return_observations_and_never_open_sqlite(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from glossapi.ledger import journal
    inputs=tmp_path/'inputs'; inputs.mkdir()
    output=tmp_path/'output'; output.mkdir()
    source=inputs/'source.pdf';source.write_bytes(b'fixture')
    corpus=SimpleNamespace(input_dir=inputs,output_dir=output,downloads_dir=output/'downloads',_shadow_ledger_enabled=True)
    with LedgerWriter(output/'corpus.sqlite3') as writer:
        parent=journal.ResultJournal(writer,corpus,'extract','extract.worker_fixture',{'file_paths':[source]})
        collector=journal.WorkerObservations()
        token=journal._active_journal.set(collector)
        monkeypatch.setenv('GLOSSAPI_SHADOW_WORKER','1')
        @journal.shadow_stage('extract')
        def worker(self):
            journal.observe('extract_completed',path=source,status='ok',pages=[1,2])
            return 17
        monkeypatch.setattr(journal,'LedgerWriter',lambda *a,**k:pytest.fail('worker opened SQLite'))
        assert worker(corpus)==17
        receipts=journal.drain_worker_observations()
        assert len(receipts)==1 and not journal.drain_worker_observations()
        journal._active_journal.reset(token)
        token=journal._active_journal.set(parent)
        try: journal.accept_worker_observations(receipts)
        finally: journal._active_journal.reset(token)
        parent.finish(None,None)
        state=writer.connection.execute("SELECT status FROM current_task_state WHERE version_id IS NOT NULL").fetchone()[0]
        assert state=='ok'


def test_worker_scope_restores_parent_after_direct_worker_exit():
    from glossapi.ledger import journal
    previous = journal._active_journal.get()
    @journal.worker_observation_scope
    def worker():
        journal.start_worker_observations()
        journal.observe('extract_exception',status='failed')
        assert len(journal.drain_worker_observations())==1
        raise SystemExit(1)
    with pytest.raises(SystemExit): worker()
    assert journal._active_journal.get() is previous
    assert 'GLOSSAPI_SHADOW_WORKER' not in os.environ


def test_missing_requested_source_still_has_identity_and_presence(tmp_path):
    from types import SimpleNamespace
    from glossapi.ledger.journal import ResultJournal
    output=tmp_path/'output';output.mkdir()
    missing=tmp_path/'missing.pdf'
    corpus=SimpleNamespace(input_dir=tmp_path,output_dir=output,downloads_dir=output/'downloads')
    with LedgerWriter(output/'corpus.sqlite3') as writer:
        journal=ResultJournal(writer,corpus,'extract','extract.missing_fixture',{'file_paths':[missing]})
        journal.finish(None,None)
        assert writer.connection.execute("SELECT presence FROM current_file_state WHERE kind='initial_input'").fetchone()[0]=='missing'
        row=writer.connection.execute('SELECT document_id,version_id FROM document_versions').fetchone()
        with writer.transaction():
            assert writer.version(row[0],source_sha256='a'*64,stored_sha256='a'*64)==row[1]


def test_replaced_legacy_locations_invalidate_presence_without_deleting_history(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with writer.transaction():
            run,document,version,task,attempt=start(writer)
            old=writer.artifact(run,'extract','markdown','file:///mutable.md',sha256='a'*64,bytes=1,version_id=version)
            new=writer.artifact(run,'extract','markdown','file:///mutable.md',sha256='b'*64,bytes=1,version_id=version)
        # A managed registration that replaces the bytes is a supersession (K4g LS4-G02), not corruption.
        assert writer.connection.execute('SELECT presence FROM current_file_state WHERE file_id=?',(old,)).fetchone()[0]=='missing'
        assert writer.connection.execute('SELECT presence FROM current_file_state WHERE file_id=?',(new,)).fetchone()[0]=='present'
        with writer.transaction():
            writer.artifact(run,'extract','markdown','file:///mutable.md',sha256='c'*64,bytes=1,version_id=version,ownership='legacy_external')
        # An observation nobody published that finds other bytes still marks the earlier registration corrupt.
        assert writer.connection.execute('SELECT presence FROM current_file_state WHERE file_id=?',(new,)).fetchone()[0]=='corrupt'
        assert writer.connection.execute('SELECT COUNT(*) FROM files').fetchone()[0]==3


def test_source_manifest_snapshot_keeps_original_byte_hash_and_file_id(tmp_path):
    from types import SimpleNamespace
    import pandas as pd
    from glossapi.ledger.journal import ResultJournal
    output=tmp_path/'out';output.mkdir()
    manifest=tmp_path/'sources.parquet'
    frame=pd.DataFrame({'url':['https://example/a','https://example/a'],'source_row':[0,0], 'detail':[b'\x00',None]})
    frame.to_parquet(manifest)
    context=SimpleNamespace(input_dir=tmp_path/'empty',output_dir=output,downloads_dir=output/'downloads')
    with LedgerWriter(output/'corpus.sqlite3') as writer:
        journal=ResultJournal(writer,context,'download','download.source_fixture',{})
        journal.source_rows(frame,manifest,url_column='url')
        journal.finish(None,None)
        snapshot=json.loads(writer.connection.execute("SELECT json FROM snapshots WHERE kind='source_manifest'").fetchone()[0])
        registered=writer.connection.execute('SELECT sha256,bytes FROM files WHERE file_id=?',(snapshot['original_file_id'],)).fetchone()
        assert registered[0]==snapshot['original_file_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert registered[1]==manifest.stat().st_size
        assert writer.connection.execute('SELECT COUNT(*) FROM source_records').fetchone()[0]==2


def test_active_section_boundary_keeps_parquet_bytes_and_return(tmp_path):
    from glossapi import Corpus
    inputs=tmp_path/'markdown'; inputs.mkdir()
    (inputs/'fixture.md').write_text('# Heading\n\nThis is a real section with several words.\n')
    baseline=Corpus(inputs,tmp_path/'baseline')
    candidate=Corpus(inputs,tmp_path/'candidate')
    for corpus in (baseline,candidate):
        corpus.markdown_dir=inputs
        corpus.good_files=['fixture']
    assert Corpus.section.__wrapped__(baseline) is None
    assert candidate.section() is None
    assert baseline.sections_parquet.read_bytes()==candidate.sections_parquet.read_bytes()
    with open_readonly(candidate.output_dir/'corpus.sqlite3') as read:
        assert read.execute("SELECT COUNT(*) FROM events WHERE reason_code='sections_completed'").fetchone()[0]>=1


def test_tabular_wrapper_return_is_preserved_and_bound_by_hash(tmp_path):
    from types import SimpleNamespace
    import pandas as pd
    from glossapi.ledger.journal import shadow_stage, wrapper_return_receipt
    output=tmp_path/'output';output.mkdir()
    context=SimpleNamespace(input_dir=tmp_path/'input',output_dir=output,downloads_dir=output/'downloads',_shadow_ledger_enabled=True)
    returned=pd.DataFrame({'url':['https://example/a'],'download_success':[True]})
    @shadow_stage('download','download.dataframe_fixture')
    def wrapper(self): return returned
    assert wrapper(context) is returned
    with open_readonly(output/'corpus.sqlite3') as read:
        receipt=json.loads(read.execute("SELECT details_json FROM events WHERE event_kind='lifecycle' AND status='ok'").fetchone()[0])['wrapper_return']
        assert receipt==wrapper_return_receipt(returned)
        assert receipt['type']=='pandas.DataFrame' and receipt['row_count']==1
        mutated=returned.copy();mutated.loc[0,'download_success']=False
        assert wrapper_return_receipt(mutated)['sha256'] != receipt['sha256']


def test_cached_repair_is_not_recorded_as_fresh_repair(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import time
    from glossapi.ledger import journal
    from glossapi.pdf_repair.extract import repair_for_extract
    source=tmp_path/'source.pdf'; source.write_bytes(b'original bytes')
    output=tmp_path/'output';output.mkdir()
    target=output/'repaired_pdfs/source.pdf'; target.parent.mkdir();target.write_bytes(b'cached bytes')
    os.utime(source,ns=(1_000_000_000,1_000_000_000));os.utime(target,ns=(2_000_000_000,2_000_000_000))
    monkeypatch.delenv('GLOSSAPI_GREEK_WORDLIST',raising=False)
    context=SimpleNamespace(input_dir=tmp_path,output_dir=output,downloads_dir=output/'downloads',_shadow_ledger_enabled=True)
    @journal.shadow_stage('extract','extract.repair_cache_fixture')
    def wrapper(self): return repair_for_extract(source,output,enabled=True,deadline=time.monotonic()+10)
    assert wrapper(context)[0]==target
    assert source.read_bytes()==b'original bytes' and target.read_bytes()==b'cached bytes'
    with open_readonly(output/'corpus.sqlite3') as read:
        reasons=[row[0] for row in read.execute("SELECT reason_code FROM events WHERE event_kind='decision'")]
        assert reasons.count('repair_cache_observed')==1
        assert 'font_repaired' not in reasons


def test_safe_backend_failure_keeps_its_backend_reason_and_original_return(tmp_path):
    from glossapi import Corpus
    source=tmp_path/'invalid.pdf'; source.write_bytes(b'not a PDF')
    corpus=Corpus(tmp_path,tmp_path/'output')
    assert corpus.extract(input_format='pdf',file_paths=[source],phase1_backend='safe',accel_type='CPU',repair_fonts=False,skip_existing=False) is None
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as read:
        reasons=[row[0] for row in read.execute('SELECT reason_code FROM events')]
        assert 'pdf_open_failed' in reasons and 'extract_exception' in reasons
        assert 'docling_failed' not in reasons


def test_swallowed_extract_metadata_error_has_real_exception_receipt(tmp_path):
    from types import SimpleNamespace
    import logging
    from glossapi.ledger import journal
    from glossapi.gloss_extract import GlossExtract
    source=tmp_path/'source.pdf';source.write_bytes(b'fixture')
    output=tmp_path/'output';output.mkdir();markdown=output/'markdown';markdown.mkdir()
    (output/'sidecars').write_bytes(b'blocks the sidecar directory')
    extractor=GlossExtract.__new__(GlossExtract)
    extractor._original_extract_file=lambda path:path
    extractor._log=logging.getLogger('metadata_failure_fixture')
    context=SimpleNamespace(input_dir=tmp_path,output_dir=output,downloads_dir=output/'downloads',_shadow_ledger_enabled=True)
    @journal.shadow_stage('extract','extract.metadata_fixture')
    def wrapper(self):
        return extractor._update_extraction_metadata(markdown,source,'success','standard')
    assert wrapper(context) is None
    with open_readonly(output/'corpus.sqlite3') as read:
        error=read.execute("SELECT error_type,traceback FROM events WHERE reason_code='extract_exception'").fetchone()
        assert error[0]=='builtins.NotADirectoryError' and 'NotADirectoryError' in error[1]


def test_numeric_source_frame_keeps_large_integer_id_without_float_coercion(tmp_path):
    from types import SimpleNamespace
    import pandas as pd
    from glossapi.ledger.journal import ResultJournal
    identifier=2**60+1
    frame=pd.DataFrame({'source_id':pd.Series([identifier],dtype='int64'),'score':[0.25]})
    manifest=tmp_path/'source.parquet';frame.to_parquet(manifest)
    output=tmp_path/'output';output.mkdir()
    context=SimpleNamespace(input_dir=tmp_path/'empty',output_dir=output,downloads_dir=output/'downloads')
    with LedgerWriter(output/'corpus.sqlite3') as writer:
        journal=ResultJournal(writer,context,'download','download.typed_numeric_fixture',{})
        journal.source_rows(frame,manifest,url_column='url')
        journal.finish(None,None)
        stored=json.loads(writer.connection.execute('SELECT metadata_json FROM source_records').fetchone()[0])
        decoded=untyped_value(stored['fields'])
        assert type(decoded['source_id']) is int and decoded['source_id']==identifier
        assert type(decoded['score']) is float and decoded['score']==0.25


def test_aggregate_task_scope_is_the_observed_manifest_hash(tmp_path):
    from types import SimpleNamespace
    from glossapi.ledger.journal import ResultJournal
    source=tmp_path/'source.pdf';source.write_bytes(b'fixture')
    output=tmp_path/'output';output.mkdir()
    context=SimpleNamespace(input_dir=tmp_path,output_dir=output,downloads_dir=output/'downloads')
    with LedgerWriter(output/'corpus.sqlite3') as writer:
        journal=ResultJournal(writer,context,'extract','extract.scope_fixture',{'file_paths':[source]})
        row=writer.connection.execute('SELECT t.scope_key,a.input_manifest_sha256 FROM tasks t JOIN attempts a ON a.task_id=t.task_id WHERE t.version_id IS NULL').fetchone()
        assert row['scope_key']=='corpus:'+row['input_manifest_sha256']
        manifest=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(row['input_manifest_sha256'],)).fetchone()[0])
        assert manifest['files'][0]['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        with pytest.raises(LedgerError,match='manifest scope'):
            with writer.transaction(): writer.task(journal.run_id,'export','export.unbound',{})
        journal.finish(None,None)
