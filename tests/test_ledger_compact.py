"""Physical compaction preserves logical facts, mutation detection and old readers."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from glossapi.ledger import LedgerWriter, open_readonly
from glossapi.ledger.compact import connect
from glossapi.ledger.encoding import file_hash
from glossapi.ledger.reader import report_why, select_run
from glossapi.ledger.writer import integrity


@pytest.mark.parametrize('storage_version',[1,2])
def test_old_and_compact_reports_retain_identical_logical_evidence(tmp_path,storage_version):
    database=tmp_path/'corpus.sqlite3'
    with LedgerWriter(database,storage_version=storage_version) as writer:
        with writer.transaction():
            run=writer.run({}, {}, {},code_revision='codec-fixture')
            document=writer.document('local_path:v1:file:///lost.pdf')
            version=writer.version(document,stored_sha256='a'*64)
            task=writer.task(run,'extract','extract.fixture',{},version_id=version)
            attempt=writer.attempt(task,worker_id='fixture',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
            value={'kind':'stage_evidence','stage':'extract','reasons':['preserved cause']*1000}
            digest=writer.snapshot('stage_evidence',value)
            writer.event(run,event_kind='decision',reason_code='extract_exception',task_id=task,version_id=version,attempt_id=attempt,details={'evidence_sha256':digest})
            writer.finish(run,task,attempt,status='failed',reason_code='extract_exception',message='exact failure',traceback='saved traceback')
        integrity(writer.connection)
    before=database.read_bytes()
    with open_readonly(database) as reader:
        report=report_why(reader,document,run=select_run(reader,run))
        assert report['stages'][0]['attempts'][0]['error']['message']=='exact failure'
        assert json.loads(reader.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()[0])=={'schema_version':1,**value}
    assert database.read_bytes()==before
    # Reopening v1 deliberately leaves its format/history intact; no upgrade is needed.
    with LedgerWriter(database) as writer:
        assert writer.connection.execute('PRAGMA user_version').fetchone()[0]==storage_version
        integrity(writer.connection)
    with sqlite3.connect(database) as raw:
        row=raw.execute('SELECT run_id FROM runs').fetchone()[0]
        assert isinstance(row,bytes if storage_version==2 else str)
        assert len(row)==(16 if storage_version==2 else 36)


def test_compact_cursor_and_connection_restore_ids_and_hashes(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with writer.transaction():
            run=writer.run({}, {}, {},code_revision='fixture')
            digest=writer.snapshot('sample',{'long':['repeat']*10000})
        cursor=writer.connection.cursor()
        assert cursor.execute('SELECT run_id FROM runs WHERE run_id=?',(run,)).fetchone()[0]==run
        assert writer.connection.execute('SELECT sha256 FROM snapshots WHERE sha256=?',(digest,)).fetchone()[0]==digest
        with pytest.raises(RuntimeError):
            with writer.transaction():
                writer.snapshot('rolled_back',{'value':'must disappear'})
                raise RuntimeError('rollback')
        assert not writer.connection.execute("SELECT 1 FROM snapshots WHERE kind='rolled_back'").fetchone()
    with connect(tmp_path/'corpus.sqlite3') as reader:integrity(reader)


def test_every_boundary_rehashes_same_size_mutation_and_writer_epoch(tmp_path,monkeypatch):
    import os
    payload=tmp_path/'payload';payload.write_bytes(b'first bytes')
    with LedgerWriter(tmp_path/'corpus.sqlite3'):
        first=file_hash(payload);original=payload.stat()
        assert file_hash(payload)==first
        payload.write_bytes(b'other bytes')
        os.utime(payload,ns=(original.st_atime_ns,original.st_mtime_ns))
        assert file_hash(payload)==hashlib.sha256(b'other bytes').hexdigest()!=first
    assert file_hash(payload)==hashlib.sha256(payload.read_bytes()).hexdigest()


@pytest.mark.parametrize('boundary',['after_intent','after_unlink'])
def test_batched_retention_recovers_process_death_with_every_consumer(tmp_path,boundary):
    import subprocess,sys
    code='''
import hashlib,json,os,signal,sys
from pathlib import Path
from glossapi.ledger import LedgerWriter
from glossapi.stages.retention import RetentionController
root=Path(sys.argv[1])
with LedgerWriter(root/'corpus.sqlite3') as writer:
 with writer.transaction():
  run=writer.run({}, {}, {},code_revision='batch-kill')
  document=writer.document('file:///source');version=writer.version(document)
  files=[]
  for index in range(3):
   path=root/(str(index)+'.md');path.write_text('exact body')
   file=writer.artifact(run,'extract','raw_markdown',path.as_uri(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,version_id=version)
   task=writer.task(run,'clean','clean.'+str(index),{},version_id=version)
   attempt=writer.attempt(task,worker_id='test',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
   writer.input(attempt,file,'text',hashlib.sha256(path.read_bytes()).hexdigest())
   writer.finish(run,task,attempt,status='ok',reason_code='clean_completed')
   files.append((file,task))
 controller=RetentionController(writer,root,run,fault=lambda point,file:os.kill(os.getpid(),signal.SIGKILL) if point==sys.argv[2] else None)
 with writer.batch():
  for file,task in files:controller.declare(file,[task])
 controller.delete_many([file for file,task in files])
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path),boundary])
    assert result.returncode==-9
    from glossapi.stages.retention import RetentionController,retention_facts
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run=writer.connection.execute('SELECT run_id FROM runs').fetchone()[0]
        RetentionController(writer,tmp_path,run).recover()
        for file,presence in writer.connection.execute('SELECT file_id,presence FROM current_file_state'):
            facts=retention_facts(writer.connection,file)
            assert presence=='deleted' and facts['consumer_ok_event_ids'] and not facts['outstanding_consumers']
        integrity(writer.connection)
    assert not list(tmp_path.glob('*.md'))


def test_routing_character_boundaries_keep_the_original_counts():
    from glossapi.page_quality import text_contamination_counts
    text='\x00\x08\t\n\x0b\x0c\r\x0e\x1f\x7f\x9f\xa0\udfff\ue000\uf8ff\uf900αA'
    assert text_contamination_counts(text)=={'pua_char_count':2,'control_char_count':8}


@pytest.mark.parametrize('storage_version',[1,2])
@pytest.mark.parametrize('initial_presence',['missing','present'])
def test_output_manifest_preserves_initial_presence_after_payload_loss(tmp_path,storage_version,initial_presence):
    from glossapi.ledger.reader import snapshot
    payload=tmp_path/'legacy.md'
    if initial_presence=='present':payload.write_bytes(b'exact initial bytes')
    digest=hashlib.sha256(payload.read_bytes()).hexdigest() if payload.exists() else None
    output={'kind':'raw_markdown','role':'body','uri':payload.as_uri(),'ownership':'legacy_external',
            'presence':initial_presence,'sha256':digest,'bytes':payload.stat().st_size if payload.exists() else None}
    with LedgerWriter(tmp_path/'corpus.sqlite3',storage_version=storage_version) as writer:
        with writer.transaction():
            run=writer.run({}, {}, {},code_revision='presence-fixture')
            document=writer.document('local_path:v1:file:///legacy.pdf');version=writer.version(document)
            task=writer.task(run,'extract','extract.fixture',{},version_id=version)
            attempt=writer.attempt(task,worker_id='fixture',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
            event=writer.finish(run,task,attempt,status='failed',reason_code='extract_exception',outputs=(output,))
        file=writer.connection.execute('SELECT file_id FROM files').fetchone()[0]
        if payload.exists():
            payload.unlink()
            with writer.transaction():writer.event(run,event_kind='artifact',reason_code='missing',file_id=file,version_id=version)
        report=report_why(writer.connection,document,run=select_run(writer.connection,run))
        actual=next(stage for stage in report['stages'] if stage['stage']=='extract')['attempts'][0]['output_hashes']
        assert actual==[{'file_id':file,**output}]
        assert report['files'][0]['presence']=='missing'
        integrity(writer.connection)


def test_registered_output_encoding_marker_does_not_reinterpret_other_evidence(tmp_path):
    from glossapi.ledger.reader import snapshot
    value={'kind':'source_manifest','encoding':'registered_files_v1','files':[],
           'explanation':'a literal field in a different evidence kind'}
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with writer.transaction():digest=writer.snapshot('source_manifest',value)
        assert snapshot(writer.connection,digest)=={'schema_version':1,**value}
