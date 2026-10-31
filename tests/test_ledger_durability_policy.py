"""Process-death durability, compact completed history, and shared evidence."""
import hashlib
import json
import sqlite3

import pytest

from glossapi.ledger import LedgerWriter, open_readonly
from glossapi.ledger.writer import integrity
from glossapi.stages.retention import RetentionController, retention_facts


def test_paths_are_relative_interned_and_survive_read_only_reopen(tmp_path):
    path=tmp_path/'products'/'α.md';path.parent.mkdir();path.write_bytes(b'exact')
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with writer.transaction():
            run=writer.run({}, {}, {},code_revision='path-fixture')
            for _ in range(2):
                writer.artifact(run,'extract','raw_markdown',path.as_uri(),sha256=hashlib.sha256(b'exact').hexdigest(),bytes=5)
        assert [tuple(row) for row in writer.connection.execute('SELECT uri FROM files')]==[(path.as_uri(),)]*2
    with sqlite3.connect(tmp_path/'corpus.sqlite3') as physical:
        assert physical.execute('SELECT uri FROM paths').fetchall()==[('@corpus/products/%CE%B1.md',)]
        assert physical.execute('SELECT COUNT(DISTINCT uri) FROM file_records').fetchone()[0]==1
    with open_readonly(tmp_path/'corpus.sqlite3') as reader:
        assert reader.execute('SELECT COUNT(*) FROM files WHERE uri=?',(path.as_uri(),)).fetchone()[0]==2
        integrity(reader)


def test_configuration_references_preserve_exact_evidence_and_rollback(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        settings={'large_common_value':'same configuration '*100,'filename':'one.pdf'}
        with writer.transaction():
            first=writer.snapshot('specification',{'settings':settings,'literal':'@corpus/user-text'})
            second=writer.snapshot('specification',{'settings':{**settings,'filename':'two.pdf'},'literal':'@corpus/user-text'})
        assert writer.connection.execute("SELECT COUNT(*) FROM snapshots WHERE kind='shared_config'").fetchone()[0]==1
        for digest,name in [(first,'one.pdf'),(second,'two.pdf')]:
            value=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()[0])
            assert value['settings']=={**settings,'filename':name}
            assert value['literal']=='@corpus/user-text'
        with pytest.raises(RuntimeError):
            with writer.transaction():
                writer.snapshot('specification',{'settings':{'large_common_value':'rolled back '*100}})
                raise RuntimeError('rollback')
        integrity(writer.connection)


def test_delete_barrier_precedes_unlink_and_compacts_only_after_completion(tmp_path,monkeypatch):
    from test_ledger_retention import fixture, consumer
    import os
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,document,version,file,path=fixture(writer,tmp_path)
        task,_=consumer(writer,run,version,file)
        controller=RetentionController(writer,tmp_path,run)
        controller.declare(file,[task])
        original_barrier=writer.durable_barrier
        calls=[]
        def barrier(*args,**options):
            assert path.exists()
            assert writer.connection.execute("SELECT COUNT(*) FROM retention_events WHERE file_id=?",(file,)).fetchone()[0]>=2
            original_barrier(*args,**options);calls.append('durable')
        monkeypatch.setattr(writer,'durable_barrier',barrier)
        unlink=os.unlink
        def checked_unlink(*args,**kwargs):
            assert calls==['durable']
            return unlink(*args,**kwargs)
        monkeypatch.setattr(os,'unlink',checked_unlink)
        assert controller.delete(file)
        assert not controller.delete(file)
        monkeypatch.setattr(writer,'durable_barrier',original_barrier)
        assert writer.connection.execute('SELECT COUNT(*) FROM retention_events WHERE file_id=?',(file,)).fetchone()[0]==0
        assert writer.connection.execute('SELECT last_consumer_stage FROM completed_deletions WHERE file_id=?',(file,)).fetchone()[0]=='clean'
        facts=retention_facts(writer.connection,file)
        assert facts['consumer_ok_event_ids'] and facts['deletions'][0]['details']['sha256']
        assert facts['policy']=='debug_off_after_last_consumer'
        assert writer.connection.execute("SELECT COUNT(*) FROM snapshots WHERE kind='retention_policy'").fetchone()[0]==0
        integrity(writer.connection)


def test_worker_receipt_uses_bytes_hashed_during_write(tmp_path,monkeypatch):
    from glossapi.stages.contracts import OutputArtifact, StageResult
    from glossapi.stages.execution import producer_hash_scope, write_text_hashed, seal_output_hashes
    from glossapi.stages import execution
    with producer_hash_scope():
        write_text_hashed(tmp_path/'text.md','α exact bytes\n')
        result=StageResult('task','attempt','lease','ok','clean_completed',
            (OutputArtifact('clean_markdown','text.md',None,'intermediate',True),),(),{})
        def forbidden_read(*args):raise AssertionError('second payload hash read')
        monkeypatch.setattr(execution,'sha256_file',forbidden_read)
        output=seal_output_hashes(result,tmp_path).outputs[0]
        assert output.sha256==hashlib.sha256('α exact bytes\n'.encode()).hexdigest()
        assert output.bytes==len('α exact bytes\n'.encode())


def test_history_reason_queries_retain_their_search_indexes(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        plan=[row[-1] for row in writer.connection.execute("EXPLAIN QUERY PLAN SELECT run_id,details_json FROM events WHERE reason_code='legacy_import_completed'")]
        assert any('USING INDEX events_reason' in detail for detail in plan),plan
        plan=[row[-1] for row in writer.connection.execute("EXPLAIN QUERY PLAN SELECT event_id,file_id FROM events WHERE event_kind='retention' AND reason_code='delete_planned'")]
        assert any('USING INDEX retention_events_reason' in detail for detail in plan),plan


def test_interned_path_presence_lookups_keep_both_search_indexes(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        plan=[row[-1] for row in writer.connection.execute('EXPLAIN QUERY PLAN SELECT file_id,presence FROM current_file_state WHERE uri=?',((tmp_path/'text.md').as_uri(),))]
        assert any('USING INDEX files_uri' in detail for detail in plan),plan
        assert any('USING INDEX events_file' in detail for detail in plan),plan
        assert not any('MATERIALIZE events' in detail for detail in plan),plan
        by_identity=[row[-1] for row in writer.connection.execute('EXPLAIN QUERY PLAN SELECT uri,presence FROM current_file_state WHERE file_id=?',('00000000-0000-0000-0000-000000000001',))]
        assert not any(detail.startswith('SCAN p') for detail in by_identity),by_identity


def test_process_death_between_compact_commit_and_policy_pruning(tmp_path):
    import subprocess,sys
    code='''
from pathlib import Path
import hashlib,os,signal,sys
from glossapi.ledger import LedgerWriter
from glossapi.stages.retention import RetentionController
root=Path(sys.argv[1]);path=root/'text.md';path.write_bytes(b'exact bytes')
with LedgerWriter(root/'corpus.sqlite3') as writer:
 with writer.transaction():
  run=writer.run({}, {}, {},code_revision='prune-kill')
  document=writer.document('file:///original');version=writer.version(document)
  digest=hashlib.sha256(b'exact bytes').hexdigest()
  file=writer.artifact(run,'extract','raw_markdown',path.as_uri(),sha256=digest,bytes=11,version_id=version)
  task=writer.task(run,'clean','clean.fixture',{},version_id=version)
  attempt=writer.attempt(task,worker_id='fixture',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
  writer.input(attempt,file,'text',digest)
  writer.finish(run,task,attempt,status='ok',reason_code='clean_completed')
 controller=RetentionController(writer,root,run)
 controller.declare(file,[task])
 controller._prune_completed_policies=lambda:os.kill(os.getpid(),signal.SIGKILL)
 controller.delete_many([file])
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)])
    assert result.returncode==-9
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        assert writer.connection.execute("SELECT COUNT(*) FROM snapshots WHERE kind='retention_policy'").fetchone()[0]==1
        run=writer.connection.execute('SELECT run_id FROM runs').fetchone()[0]
        RetentionController(writer,tmp_path,run).recover()
        assert writer.connection.execute("SELECT COUNT(*) FROM snapshots WHERE kind='retention_policy'").fetchone()[0]==0
        file,presence=writer.connection.execute('SELECT file_id,presence FROM current_file_state').fetchone()
        assert presence=='deleted' and retention_facts(writer.connection,file)['consumer_ok_event_ids']
        integrity(writer.connection)


def test_recovery_finishes_an_interrupted_durability_barrier_before_unlink(tmp_path,monkeypatch):
    import os,subprocess,sys
    code='''
from pathlib import Path
import hashlib,os,signal,sys
from glossapi.ledger import LedgerWriter
from glossapi.stages.retention import RetentionController
root=Path(sys.argv[1]);path=root/'text.md';path.write_bytes(b'exact bytes')
with LedgerWriter(root/'corpus.sqlite3') as writer:
 with writer.transaction():
  run=writer.run({}, {}, {},code_revision='barrier-kill')
  document=writer.document('file:///original');version=writer.version(document)
  digest=hashlib.sha256(b'exact bytes').hexdigest()
  file=writer.artifact(run,'extract','raw_markdown',path.as_uri(),sha256=digest,bytes=11,version_id=version)
  task=writer.task(run,'clean','clean.fixture',{},version_id=version)
  attempt=writer.attempt(task,worker_id='fixture',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
  writer.input(attempt,file,'text',digest)
  writer.finish(run,task,attempt,status='ok',reason_code='clean_completed')
 controller=RetentionController(writer,root,run)
 controller.declare(file,[task])
 writer.durable_barrier=lambda *args,**options:os.kill(os.getpid(),signal.SIGKILL)
 controller.delete_many([file])
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)])
    assert result.returncode==-9 and (tmp_path/'text.md').exists()
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run=writer.connection.execute('SELECT run_id FROM runs').fetchone()[0]
        barriers=[];barrier=writer.durable_barrier
        def record_barrier(*args,**options):barrier(*args,**options);barriers.append('durable')
        monkeypatch.setattr(writer,'durable_barrier',record_barrier)
        unlink=os.unlink
        def checked_unlink(*args,**kwargs):
            assert barriers,'Recovery unlinked before finishing the interrupted barrier'
            return unlink(*args,**kwargs)
        monkeypatch.setattr(os,'unlink',checked_unlink)
        assert RetentionController(writer,tmp_path,run).recover()==1
        assert not (tmp_path/'text.md').exists()
