"""Private release summaries preserve resume proof without reading payloads."""
import json
from pathlib import Path

import pytest

from glossapi.ledger import LedgerWriter
from glossapi.stages.execution import sha256_file
from glossapi.stages.retention import RetentionController, release_private_scratch, retention_facts


def private_producer(writer, root):
    path = root/'.glossapi/artifacts/private/diagnostic.txt'
    path.parent.mkdir(parents=True)
    path.write_text('private producer bytes')
    with writer.transaction():
        run = writer.run({}, {}, {}, code_revision='test')
        scope = writer.snapshot('inputs', {'files':[]})
        task = writer.task(run, 'extract', 'extract.private', {}, scope_key='corpus:'+scope)
        attempt = writer.attempt(task, worker_id='test', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        writer.finish(run, task, attempt, status='ok', reason_code='extract_completed', outputs=[{
            'kind':'attempt_diagnostic', 'uri':path.as_uri(), 'sha256':sha256_file(path), 'bytes':path.stat().st_size}])
    row = writer.connection.execute('SELECT * FROM current_file_state WHERE uri=?', (path.as_uri(),)).fetchone()
    return run, row, path


@pytest.mark.parametrize('boundary', ['complete', 'after_intent', 'after_unlink'])
def test_private_summary_recovers_without_payload_hashing(tmp_path, monkeypatch, boundary):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, row, path = private_producer(writer, tmp_path)
        def fault(point, file_id):
            if point == boundary:
                raise RuntimeError('interrupted release')
        controller = RetentionController(writer, tmp_path, run, fault=fault)
        monkeypatch.setattr(controller, '_identity', lambda *_: pytest.fail('Private release rehashed payload'))
        if boundary == 'complete':
            release_private_scratch(controller, [row])
        else:
            with pytest.raises(RuntimeError, match='interrupted release'):
                release_private_scratch(controller, [row])
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        RetentionController(writer, tmp_path, run).recover()
        assert not path.exists()
        assert writer.connection.execute('SELECT presence FROM current_file_state WHERE file_id=?', (row['file_id'],)).fetchone()[0] == 'deleted'
        facts = retention_facts(writer.connection, row['file_id'])
        assert facts['deletions'][0]['details']['sha256'] == row['sha256']
        summaries = writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE event_kind='retention' AND reason_code='delete_completed'").fetchall()
        assert len(summaries) == 1
        assert json.loads(summaries[0][0])['file_count'] == 1


def test_control_and_consumer_queries_use_physical_indexes(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, row, _ = private_producer(writer, tmp_path)
        queries = [
            (f"SELECT t.stage FROM {writer.connection.event_table} e JOIN tasks t USING(task_id) JOIN attempt_inputs i USING(attempt_id) WHERE i.file_id=? AND e.event_kind='lifecycle' AND e.status='ok' ORDER BY e.event_id DESC LIMIT 1", (row['file_id'],)),
            (f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' AND message=? ORDER BY event_id DESC LIMIT 1", (run,'wrapper:clean')),
            ('SELECT 1 FROM source_records WHERE manifest_sha256=? LIMIT 1', ('a'*64,)),
            (f"SELECT e.details_json FROM {writer.connection.event_table} e JOIN tasks t USING(task_id) WHERE t.stage='clean' AND t.operation LIKE 'clean.selection%' AND e.event_kind='decision'", ()),
            ("SELECT uri FROM current_file_state WHERE stage='export' AND kind='document_jsonl_shard' AND ownership='managed' AND presence='present'", ()),
            ('SELECT status FROM current_task_state WHERE task_id=?', (writer.connection.execute('SELECT task_id FROM tasks WHERE run_id=?', (run,)).fetchone()[0],)),
        ]
        for sql, bindings in queries:
            plan = [value[-1] for value in writer.connection.execute('EXPLAIN QUERY PLAN '+sql, bindings)]
            assert not any('AUTOMATIC' in value or value.startswith('SCAN ') for value in plan), plan


def test_failed_publication_keeps_private_products_for_retry(tmp_path):
    from glossapi import Corpus
    from test_ledger_stages import simple_pdf
    source = simple_pdf(tmp_path/'source.pdf')
    corpus = Corpus(tmp_path/'empty', tmp_path/'output')
    original = corpus.clean
    def fail_publication(*args, **options):
        raise RuntimeError('publisher interrupted before consuming producer records')
    corpus.clean = fail_publication
    result = corpus.run([source], stages=('extract','clean','export'), settings={'extract':{'repair_fonts':False}})
    assert result['state'] == 'completed_with_failures'
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        private = writer.connection.execute("SELECT uri FROM current_file_state WHERE kind='clean_markdown' AND uri LIKE ? AND presence='present'", ((corpus.output_dir/'.glossapi/artifacts').as_uri()+'/%',)).fetchall()
        assert private, 'The unfinished publication still needs its private producer output'
        assert all(Path(__import__('urllib.parse', fromlist=['urlsplit']).urlsplit(uri).path).is_file() for uri, in private)
    corpus.clean = original
    resumed = corpus.run(resume=result['run_id'], retry_failed=True)
    assert resumed['state'] == 'completed'


@pytest.mark.parametrize('storage_version', [1, 2, 3])
def test_retention_history_excludes_artifact_presence_events(tmp_path, storage_version):
    path = tmp_path/'given.md'
    path.write_text('registered presence is not a deletion')
    with LedgerWriter(tmp_path/'corpus.sqlite3', storage_version=storage_version) as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='test')
            file_id = writer.artifact(run, 'extract', 'raw_markdown', path.as_uri(),
                sha256=sha256_file(path), bytes=path.stat().st_size)
        assert retention_facts(writer.connection, file_id)['deletions'] == []


@pytest.mark.parametrize('storage_version', [1, 2, 3])
def test_file_catalog_matches_persisted_state_after_rollback(tmp_path, storage_version):
    from glossapi.ledger.writer import open_readonly
    from glossapi.stages.disk_budget import DiskBudget
    from glossapi.stages.retention import retention_summary
    path=tmp_path/'payload.md';path.write_text('durable identity')
    with LedgerWriter(tmp_path/'corpus.sqlite3', storage_version=storage_version) as writer:
        with writer.transaction():
            run=writer.run({}, {}, {}, code_revision='test')
            file_id=writer.artifact(run,'extract','raw_markdown',path.as_uri(),sha256=sha256_file(path),bytes=path.stat().st_size)
        with pytest.raises(RuntimeError):
            with writer.batch():
                writer.event(run,event_kind='artifact',reason_code='missing',file_id=file_id)
                assert writer.file_catalog()[file_id]['presence']=='missing'
                raise RuntimeError('rolled back observation')
        assert writer.file_catalog()[file_id]['presence']=='present'
        assert DiskBudget(writer,tmp_path,run,{}).present_bytes()==path.stat().st_size
        live=retention_summary(writer.connection)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        assert retention_summary(connection)==live
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        assert writer.file_catalog()[file_id]['presence']=='present'


class PrivateSnapshotProducer:
    name='extract'
    contract_version=1

    def run_document(self, request, *, reporter):
        from glossapi.stages.contracts import OutputArtifact, StageResult
        if request.settings.get('fail'):
            raise RuntimeError('independent prerequisite failed')
        path=request.work_directory/'inputs/snapshot.txt'
        path.parent.mkdir(exist_ok=True)
        path.write_text(request.inputs[0].path.read_text())
        return StageResult(request.task_id,request.attempt_id,request.lease_token,'ok','extract_completed',
            (OutputArtifact('source_snapshot','inputs/snapshot.txt',None,'intermediate',True),),(),{})


class PrivateSnapshotConsumer(PrivateSnapshotProducer):
    name='clean'


def test_unattempted_reader_keeps_private_input_after_other_dependency_fails(tmp_path):
    from dataclasses import replace
    from test_ledger_scheduler import graph
    from glossapi.stages.scheduler import StreamingScheduler
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        producer,consumer=graph(writer,tmp_path,1)
        producer=replace(producer,specification=replace(producer.specification,required_output_kinds=('source_snapshot',)))
        failed=replace(producer,key='failed-prerequisite',specification=replace(producer.specification,
            operation='extract.failed_prerequisite',settings={'fail':True}))
        consumer=replace(consumer,inputs=((producer.key,'source_snapshot','selected_text'),
            (failed.key,'source_snapshot','required_gate')),specification=replace(consumer.specification,required_output_kinds=('source_snapshot',)))
        scheduler=StreamingScheduler(writer,tmp_path,registry={'extract':PrivateSnapshotProducer,'clean':PrivateSnapshotConsumer})
        result=scheduler.execute(scheduler.freeze([producer,failed,consumer]))
        assert result['counts']['failed']==1 and result['counts']['skipped']==1
        snapshot=next(file for file in scheduler._outputs(result['attempts'][producer.key]) if file.kind=='source_snapshot')
        assert snapshot.path.is_file(), 'A skipped upstream_failed reader is still a planned reader, not a successful consumer'
        from glossapi.stages.retention import retain_completed_run
        retain_completed_run(writer,tmp_path,result['run_id'])
        assert snapshot.path.is_file(), 'Final collection must honor the same frozen reader frontier'


@pytest.mark.parametrize('storage_version', [1, 2, 3])
def test_control_catalog_matches_commits_rollback_and_reopen(tmp_path, storage_version):
    database = tmp_path/'corpus.sqlite3'
    with LedgerWriter(database, storage_version=storage_version) as writer:
        run, _, _ = private_producer(writer, tmp_path)
        task = writer.task_states(run)[0]['task_id']
        original = writer.task_state(task)
        def matches_sql():
            for actual in writer.connection.execute('SELECT * FROM current_task_state'):
                cached = writer.task_state(actual['task_id'])
                assert {key:cached[key] for key in actual.keys()} == dict(actual)
            for actual in writer.connection.execute('SELECT * FROM attempts'):
                assert writer.control()['attempts'][actual['attempt_id']] == dict(actual)
        matches_sql()
        with pytest.raises(RuntimeError, match='rollback control'):
            with writer.transaction():
                attempt = writer.attempt(task, worker_id='retry', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
                writer.event(run, event_kind='lifecycle', task_id=task, attempt_id=attempt, status='running', reason_code='stage_started')
                matches_sql()
                raise RuntimeError('rollback control')
        assert writer.task_state(task) == original
        matches_sql()
    with LedgerWriter(database) as writer:
        assert writer.task_state(task) == original
        assert writer.latest_attempt(task)['status'] == 'ok'
        matches_sql()


def test_live_publication_and_retention_do_not_query_task_state(tmp_path, monkeypatch):
    from glossapi import Corpus
    from glossapi.ledger.compact import LedgerCursor
    from test_ledger_stages import simple_pdf
    execute = LedgerCursor.execute
    def guarded_execute(cursor, sql, parameters=()):
        if sql.lstrip().upper().startswith(('SELECT ', 'WITH ')) and 'CURRENT_TASK_STATE' in sql.upper():
            pytest.fail('Live control read queried SQLite instead of the append-driven task catalog: '+sql)
        return execute(cursor, sql, parameters)
    monkeypatch.setattr(LedgerCursor, 'execute', guarded_execute)
    source = simple_pdf(tmp_path/'source.pdf')
    models=Path(__import__('glossapi').__file__).parent/'models'
    corpus = Corpus(tmp_path/'empty', tmp_path/'output',
        section_classifier_model_path=models/'section_classifier.joblib',
        extraction_model_path=models/'kmeans_weights.joblib')
    result = corpus.run([source], stages=('extract','clean','sections','annotate','export'), settings={'extract':{'repair_fonts':False}})
    assert result['state'] == 'completed'
