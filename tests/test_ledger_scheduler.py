"""Real spawned-worker saturation, deadlines, publication and recovery."""
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import subprocess
import sys

import pytest

from glossapi.ledger import LedgerWriter
from glossapi.stages.contracts import ArtifactReference, DocumentSnapshot, OutputArtifact, StageResult, StageSpecification
from glossapi.stages.execution import sha256_file
from glossapi.stages.scheduler import ScheduledOperation, StreamingScheduler


class Producer:
    name='extract'
    contract_version=1
    def run_document(self,request,*,reporter):
        time.sleep(request.settings.get('delay',.01))
        if request.settings.get('crash'):
            os._exit(9)
        text=request.inputs[0].path.read_text() if request.inputs else request.document_id
        if request.settings.get('warn'):
            import logging
            logging.warning('document=%s',text)
        (request.work_directory/'text.txt').write_text(text)
        return StageResult(request.task_id,request.attempt_id,request.lease_token,'ok','extract_completed',
            (OutputArtifact('text','text.txt',None,'final',True),),({'route':True},),{'pid':os.getpid()})


class Consumer(Producer):
    name='clean'
    def run_document(self,request,*,reporter):
        result=super().run_document(request,reporter=reporter)
        return replace(result,reason_code='clean_completed')


def graph(writer,root,count=8,*,delay=.01,consumer_delay=.04,crash=False):
    operations=[]
    with writer.transaction():
        observation=writer.run({}, {}, {},code_revision='test')
        for index in range(count):
            source=root/f'{index}.txt';source.write_text(str(index))
            document=writer.document(source.as_uri())
            version=writer.version(document,source_sha256=sha256_file(source),stored_sha256=sha256_file(source))
            file=writer.artifact(observation,'extract','source',source.as_uri(),version_id=version,
                sha256=sha256_file(source),bytes=source.stat().st_size,ownership='initial_external',retention_class='initial')
            snapshot=DocumentSnapshot(document,version,source.as_uri(),artifacts=(ArtifactReference(file,'source',source,sha256_file(source),source.stat().st_size),))
            operations.append(ScheduledOperation(f'produce:{index}','extract',snapshot,StageSpecification('extract.test',('source',),('text',),settings={'delay':delay,'crash':crash})))
            operations.append(ScheduledOperation(f'consume:{index}','clean',replace(snapshot,artifacts=()),StageSpecification('clean.test',required_output_kinds=('text',),settings={'delay':consumer_delay}),
                inputs=((f'produce:{index}','text','selected_text'),)))
    return operations


def scheduler(writer,root,**options):
    return StreamingScheduler(writer,root,registry={'extract':Producer,'clean':Consumer},**options)


def test_streaming_saturation_slow_consumer_warm_reuse_and_cpu_limit(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine=scheduler(writer,tmp_path,workers={'extract':2,'clean':1},cpu_tokens=2)
        run=engine.freeze(graph(writer,tmp_path,12))
        result=engine.execute(run)
        assert result['state']=='completed' and result['counts']['ok']==24
        for key,water in result['queue_high_water'].items():
            assert water<= (4 if 'extract' in key else 2)
        pids={}
        for key,attempt in result['attempts'].items():
            pids.setdefault(key.split(':')[0],set()).add(engine.result(attempt).metrics['pid'])
        assert len(pids['produce'])<=2 and len(pids['consume'])==1
        trace=result['trace']
        assert next(row['monotonic_ns'] for row in trace if row['event']=='dispatch' and row['key'].startswith('consume:')) < max(row['monotonic_ns'] for row in trace if row['event']=='complete' and row['key'].startswith('produce:'))
        active=peak=0
        for row in trace:
            if row['event']=='dispatch':active+=1
            if row['event']=='complete':active-=1
            peak=max(peak,active)
        assert peak<=2


def test_no_worker_leaves_healthy_files_unattempted_and_planned(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine=scheduler(writer,tmp_path,workers={'extract':0})
        result=engine.execute(engine.freeze(graph(writer,tmp_path)))
        assert result['state']=='waiting' and result['counts']['planned']==16
        assert writer.connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==0


def test_resume_only_unfinished_and_rehashes_committed_outputs(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine=scheduler(writer,tmp_path)
        run=engine.freeze(graph(writer,tmp_path,2))
        first=engine.execute(run)
        before=writer.connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
        resumed=scheduler(writer,tmp_path).execute(run,resume=True)
        assert resumed['state']=='completed' and len(resumed['attempts'])==4
        assert writer.connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==before
        assert first['attempts']==resumed['attempts']
        artifact=next(a for a in engine._outputs(first['attempts']['produce:0']) if a.kind=='text')
        artifact.path.write_text('corruption')
        repaired=scheduler(writer,tmp_path).execute(run,resume=True)
        assert repaired['state']=='completed'
        assert repaired['attempts']['produce:1']==first['attempts']['produce:1']
        assert repaired['attempts']['produce:0']!=first['attempts']['produce:0']


def test_crash_retries_bounded_and_descendants_skip(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine=scheduler(writer,tmp_path,retry_limit=3)
        result=engine.execute(engine.freeze(graph(writer,tmp_path,1,crash=True)))
        assert result['counts']=={'planned':0,'running':0,'ok':0,'skipped':1,'failed':1}
        assert writer.connection.execute("SELECT COUNT(*) FROM attempts a JOIN tasks t USING(task_id) WHERE t.stage='extract'").fetchone()[0]==3


def test_cycle_and_unsupported_device_fail_before_dispatch(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        operations=graph(writer,tmp_path,1)
        engine=scheduler(writer,tmp_path)
        with pytest.raises(ValueError,match='Cyclic'):
            engine.freeze([replace(operations[0],after=('consume:0',)),operations[1]])
        with pytest.raises(ValueError,match='Device'):
            engine.freeze([replace(operations[0],specification=replace(operations[0].specification,device_id='cuda:0'))])
        from glossapi.stages.scheduler import stage_registry
        engine.registry=stage_registry()
        with pytest.raises(ValueError,match='Unsupported download mode'):
            engine.freeze([replace(operations[0],stage='download',specification=StageSpecification('download.acquire',settings={'download_mode':'unknown'}))])
        with pytest.raises(ValueError,match='Unsafe download filename'):
            engine.freeze([replace(operations[0],stage='download',specification=StageSpecification('download.acquire',settings={'filename_base':'../outside'}))])
        with pytest.raises(ValueError,match='CPU tokens'):
            engine.freeze([replace(operations[0],specification=replace(operations[0].specification,settings={'num_threads':2}))])


def test_conditional_skip_is_not_successful_consumption(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        operations=graph(writer,tmp_path,1)
        operations[1]=replace(operations[1],condition=('produce:0','route',False))
        engine=scheduler(writer,tmp_path)
        result=engine.execute(engine.freeze(operations))
        assert result['counts']['ok']==1 and result['counts']['skipped']==1


def test_writer_batch_rolls_back_all_publications_and_excludes_second_owner(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with pytest.raises(BlockingIOError):
            LedgerWriter(tmp_path/'corpus.sqlite3')
        with pytest.raises(RuntimeError):
            with writer.batch():
                with writer.transaction():
                    writer.document('first')
                with writer.transaction():
                    writer.document('second')
                raise RuntimeError('crash before commit')
        assert writer.connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0]==0


def test_public_run_projects_streamed_results_without_reextracting(tmp_path):
    pytest.importorskip('glossapi_rs_cleaner')
    from glossapi import Corpus
    from test_ledger_stages import simple_pdf
    source=simple_pdf(tmp_path/'source.pdf')
    output=tmp_path/'output'
    corpus=Corpus(tmp_path/'empty',output)
    retired_skiplist=output/'skiplists/fatal_skip.txt'
    retired_skiplist.parent.mkdir();retired_skiplist.write_text('source\n')
    result=corpus.run([source],stages=('extract','clean','sections','export'),workers={'extract':1},settings={'extract':{'repair_fonts':False}})
    assert result['state']=='completed'
    assert retired_skiplist.read_text()=='source\n'
    assert (output/'export.jsonl').is_file()
    import sqlite3
    with sqlite3.connect(output/'corpus.sqlite3') as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts a JOIN tasks t USING(task_id) WHERE t.operation='extract.document'").fetchone()[0]==1
        attempts=connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    before=(output/'export.jsonl').read_bytes()
    resumed=corpus.run(resume=result['run_id'])
    assert resumed['state']=='completed' and before==(output/'export.jsonl').read_bytes()
    with sqlite3.connect(output/'corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==attempts


def test_streamed_section_subset_uses_successful_aggregate_not_omission_status(tmp_path):
    pytest.importorskip('glossapi_rs_cleaner')
    from glossapi import Corpus
    from test_ledger_stages import simple_pdf
    omitted=simple_pdf(tmp_path/'omitted.pdf',blank=True)
    selected=simple_pdf(tmp_path/'selected.pdf')
    models=Path(__import__('glossapi').__file__).parent/'models'
    corpus=Corpus(tmp_path/'empty',tmp_path/'output',
        section_classifier_model_path=models/'section_classifier.joblib',
        extraction_model_path=models/'kmeans_weights.joblib')
    result=corpus.run([omitted,selected],stages=('extract','clean','sections','annotate','export'),
        settings={'extract':{'repair_fonts':False}})
    assert corpus.good_files==['selected']
    assert result['state']=='completed'
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        outcomes=dict(writer.connection.execute("SELECT stage,status FROM current_task_state WHERE run_id=? AND operation LIKE '%.compatibility_publish'",(result['run_id'],)))
        assert outcomes['sections']=='ok' and outcomes['annotate']=='ok'
        assert writer.connection.execute("SELECT COUNT(*) FROM current_task_state WHERE operation='sections.wrapper' AND status='ok'").fetchone()[0]==1
        assert writer.connection.execute("SELECT COUNT(*) FROM events WHERE event_kind='decision' AND reason_code='section_not_selected'").fetchone()[0]==1


@pytest.mark.parametrize('boundary',['before_publication','after_publication','before_commit','after_commit'])
def test_restart_quarantines_uncommitted_publication(tmp_path,monkeypatch,boundary):
    import glossapi.stages.sequential as publication
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        operations=graph(writer,tmp_path,1)
        engine=scheduler(writer,tmp_path)
        run=engine.freeze(operations)
        real_rename=publication.os.rename
        real_finish=writer.finish
        real_batch=writer.batch
        def rename(source,destination):
            if boundary=='before_publication':raise RuntimeError('injected crash')
            real_rename(source,destination)
            if boundary=='after_publication':raise RuntimeError('injected crash')
        def finish(*args,**kwargs):
            if boundary=='before_commit':raise RuntimeError('injected crash')
            return real_finish(*args,**kwargs)
        from contextlib import contextmanager
        @contextmanager
        def committed_batch():
            with real_batch():yield writer
            if writer.connection.execute("SELECT COUNT(*) FROM events WHERE status='ok'").fetchone()[0]:
                raise RuntimeError('injected crash')
        with monkeypatch.context() as patch:
            patch.setattr(publication.os,'rename',rename)
            patch.setattr(writer,'finish',finish)
            if boundary=='after_commit':patch.setattr(writer,'batch',committed_batch)
            with pytest.raises(RuntimeError,match='injected crash'):
                engine.execute(run)
        assert writer.connection.execute("SELECT COUNT(*) FROM events WHERE status='ok'").fetchone()[0]==(1 if boundary=='after_commit' else 0)
        result=scheduler(writer,tmp_path).execute(run,resume=True)
        assert result['state']=='completed'
        if boundary!='after_commit':assert list((tmp_path/'.glossapi/quarantine').iterdir())


def test_stale_lease_cannot_publish_and_changed_input_cannot_complete(tmp_path):
    from glossapi.stages.sequential import SequentialStageRunner
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        operation=graph(writer,tmp_path,1)[0]
        request,complete=SequentialStageRunner(writer,tmp_path).prepare(Producer(),operation.document,operation.specification)
        with pytest.raises(ValueError,match='Stale'):
            complete(StageResult(request.task_id,request.attempt_id,'expired','ok','extract_completed',(),(),{}))
        operation.document.artifacts[0].path.write_text('changed input')
        engine=scheduler(writer,tmp_path)
        result=engine.execute(engine.freeze([operation]))
        assert result['counts']['failed']==1
        assert writer.connection.execute("SELECT COUNT(*) FROM events WHERE reason_code='input_hash_changed'").fetchone()[0]==1


def test_killed_orchestrator_resumes_only_unfinished_work(tmp_path):
    driver=tmp_path/'driver.py'
    driver.write_text('''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
from test_ledger_scheduler import graph,scheduler
from glossapi.ledger import LedgerWriter
root=Path(sys.argv[1])
if __name__=='__main__':
 with LedgerWriter(root/'corpus.sqlite3') as writer:
  engine=scheduler(writer,root,workers={'extract':1,'clean':1},cpu_tokens=2)
  if (root/'run-id').exists():
   run=(root/'run-id').read_text()
   result=engine.execute(run,resume=True)
   (root/'result.json').write_text(json.dumps(result))
  else:
   run=engine.freeze(graph(writer,root,6,delay=.15,consumer_delay=.15))
   (root/'run-id').write_text(run)
   engine.execute(run)
''')
    import sqlite3
    command=[sys.executable,str(driver),str(tmp_path),str(Path(__file__).parent)]
    with (tmp_path/'driver.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+15
            while True:
                if (tmp_path/'run-id').exists():
                    with sqlite3.connect(tmp_path/'corpus.sqlite3') as connection:
                        committed=[row[0] for row in connection.execute("SELECT attempt_id FROM events WHERE status='ok'")]
                    if len(committed)>=2:break
                assert time.monotonic()<deadline
                time.sleep(.02)
            process.kill();process.wait()
        finally:
            if process.poll() is None:process.kill();process.wait()
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['state']=='completed'
    assert set(committed)<=set(result['attempts'].values())


@pytest.mark.parametrize('resource,value',[('cpu_tokens',2),('memory_bytes',100),('reservation_bytes',100)])
def test_resource_requests_wait_without_acquiring_an_attempt(tmp_path,resource,value):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        operation=replace(graph(writer,tmp_path,1)[0],**{resource:value})
        engine=scheduler(writer,tmp_path,cpu_tokens=1,memory_bytes=10,disk_budget_bytes=10)
        result=engine.execute(engine.freeze([operation]))
        assert result['state']=='waiting' and result['counts']['planned']==1
        assert writer.connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]==0


def test_execution_fingerprint_change_rejects_resume(tmp_path,monkeypatch):
    import glossapi.stages.scheduler as module
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine=scheduler(writer,tmp_path)
        run=engine.freeze(graph(writer,tmp_path,1))
        identity=module.execution_environment()
        monkeypatch.setattr(module,'execution_environment',lambda:{**identity,'code':'changed'})
        with pytest.raises(ValueError,match='environment changed'):
            engine.execute(run,resume=True)


def test_changed_bound_model_invalidates_frozen_run(tmp_path):
    model=tmp_path/'model.joblib';model.write_bytes(b'pinned-model')
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine=scheduler(writer,tmp_path)
        run=engine.freeze(graph(writer,tmp_path,1),settings={'semantic_files':[{'path':str(model),'sha256':sha256_file(model)}]})
        model.write_bytes(b'changed-model')
        with pytest.raises(ValueError,match='model/dictionary input changed'):
            engine.execute(run,resume=True)


def test_cli_run_single_json_and_supplied_plan_validation(tmp_path):
    pytest.importorskip('glossapi_rs_cleaner')
    from test_ledger_stages import simple_pdf
    from glossapi.ledger.reader import plan_view
    source=simple_pdf(tmp_path/'source.pdf')
    plan=plan_view(selectors=[str(source)],stages=['clean'],settings={'extract':{'repair_fonts':False}})
    path=tmp_path/'plan.json';path.write_text(json.dumps({'data':plan}))
    command=[sys.executable,'-m','glossapi.cli','run','--corpus',str(tmp_path/'output'),'--plan',str(path),'--json']
    completed=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    outcome=json.loads(completed.stdout)
    assert outcome['data']['state']=='completed'
    plan['settings']['debug']='on';path.write_text(json.dumps(plan))
    completed=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert completed.returncode==2 and 'hash does not match' in completed.stdout


def test_one_by_one_http_arrivals_start_extract_and_clean_before_last_download(tmp_path):
    pytest.importorskip('aiohttp')
    pytest.importorskip('glossapi_rs_cleaner')
    from test_ledger_stages import simple_pdf
    from glossapi import Corpus
    from glossapi.stages.run import prepare_streaming
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import threading
    payload=simple_pdf(tmp_path/'source.pdf').read_bytes()
    arrivals=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/last.pdf':time.sleep(2)
            self.send_response(200);self.send_header('Content-Type','application/pdf')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            arrivals.append(time.monotonic_ns())
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        corpus=Corpus(tmp_path/'empty',tmp_path/'output')
        result,replays=prepare_streaming(corpus,[f'http://127.0.0.1:{server.server_port}/{name}.pdf' for name in ('first','last')],
            stages=('extract','clean'),settings={'download':{'rate_limit':0},'extract':{'repair_fonts':False}})
        assert result['state']=='completed',result
        assert all(isinstance(entry[2],ArtifactReference) for entries in replays.values() for entry in entries)
        starts=[row['monotonic_ns'] for row in result['trace'] if row['event']=='dispatch' and row['key'].endswith((':extract',':clean'))]
        assert len(starts)==4 and min(starts)<max(arrivals)
        assert next(row['monotonic_ns'] for row in result['trace'] if row['event']=='dispatch' and row['key']=='0:clean')<max(arrivals)
    finally:
        server.shutdown();server.server_close();thread.join()


def test_warm_worker_never_appends_to_published_attempt_logs(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        operations=[operation for operation in graph(writer,tmp_path,3) if operation.stage=='extract']
        operations=[replace(operation,specification=replace(operation.specification,settings={'warn':True})) for operation in operations]
        engine=scheduler(writer,tmp_path,workers={'extract':1})
        result=engine.execute(engine.freeze(operations))
        assert result['state']=='completed'
        for index in range(3):
            attempt=result['attempts'][f'produce:{index}']
            assert engine._verified(attempt)
            log=next(artifact for artifact in engine._outputs(attempt) if artifact.path.name=='conversion.log')
            content=log.path.read_text()
            assert f'document={index}' in content
            assert all(f'document={other}' not in content for other in range(3) if other!=index)


@pytest.mark.parametrize('include_good',[False,True])
def test_one_failed_document_does_not_block_successful_publication(tmp_path,include_good):
    pytest.importorskip('glossapi_rs_cleaner')
    from test_ledger_stages import simple_pdf
    from glossapi import Corpus
    source=simple_pdf(tmp_path/'good.pdf')
    broken=tmp_path/'broken.pdf';broken.write_bytes(b'not a PDF')
    corpus=Corpus(tmp_path/'empty',tmp_path/'output')
    result=corpus.run([broken,*([source] if include_good else [])],stages=('extract','clean','export'),settings={'extract':{'repair_fonts':False}})
    assert result['state']=='completed_with_failures'
    assert result['document_counts']=={'selected':1+int(include_good),'included':int(include_good),'omitted':1,'failed':1}
    assert (corpus.output_dir/'export.jsonl').is_file()
    assert (corpus.output_dir/'markdown/good.md').is_file()==include_good
    if not include_good:assert (corpus.output_dir/'export.jsonl').read_bytes()==b''
    assert not (corpus.output_dir/'markdown/broken.md').exists()
    import sqlite3
    with sqlite3.connect(corpus.output_dir/'corpus.sqlite3') as connection:
        assert connection.execute("SELECT status FROM current_task_state WHERE run_id=? AND operation='export.compatibility_publish'",(result['run_id'],)).fetchone()[0]=='ok'


def test_annotation_failure_does_not_block_independent_jsonl_branch(tmp_path,monkeypatch):
    pytest.importorskip('glossapi_rs_cleaner')
    from test_ledger_stages import simple_pdf
    from glossapi import Corpus
    source=simple_pdf(tmp_path/'source.pdf')
    corpus=Corpus(tmp_path/'empty',tmp_path/'output')
    def failed_annotation():raise RuntimeError('model worker failed')
    monkeypatch.setattr(corpus,'annotate',failed_annotation)
    result=corpus.run([source],stages=('extract','clean','sections','annotate','export'),settings={'extract':{'repair_fonts':False}})
    assert result['state']=='completed_with_failures'
    assert (corpus.output_dir/'export.jsonl').is_file()
    import sqlite3
    with sqlite3.connect(corpus.output_dir/'corpus.sqlite3') as connection:
        outcomes=dict(connection.execute("SELECT stage,status FROM current_task_state WHERE run_id=? AND operation LIKE '%.compatibility_publish'",(result['run_id'],)))
        assert outcomes['annotate']=='failed' and outcomes['export']=='ok'


@pytest.mark.parametrize('include_failure',[False,True])
def test_remote_run_preserves_download_products_metadata_and_resume(tmp_path,include_failure):
    pytest.importorskip('aiohttp')
    pytest.importorskip('glossapi_rs_cleaner')
    import pandas as pd
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import threading
    from test_ledger_stages import simple_pdf
    from glossapi import Corpus
    payload=simple_pdf(tmp_path/'source.pdf').read_bytes()
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path=='/missing.pdf':
                self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','application/pdf')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        url=f'http://127.0.0.1:{server.server_port}/source.pdf'
        records=[{'url':url,'filename_base':'source','title':'Original metadata','identifier':2**60+1},
                 {'url':url,'filename_base':'duplicate','title':'Duplicate occurrence','identifier':2**60+2}]
        if include_failure:
            records[1]={'url':url.replace('source.pdf','missing.pdf'),'filename_base':'missing',
                'title':'Failed acquisition','identifier':2**60+3}
        download_settings={'rate_limit':0,'max_retries':1,'retry_interval':0}
        manifest=tmp_path/'sources.parquet';pd.DataFrame(records).to_parquet(manifest,index=False)
        sequential=Corpus(tmp_path/'empty-sequential',tmp_path/'sequential')
        sequential.download(manifest,**download_settings)
        sequential.extract(input_format='pdf',phase1_backend='safe',accel_type='CPU',num_threads=1,
            skip_existing=False,batched=False,repair_fonts=False)
        sequential.clean(num_threads=1)
        sequential.jsonl(sequential.output_dir/'export.jsonl')
        sequential_request_count=len(requests)
        streaming=Corpus(tmp_path/'empty-streaming',tmp_path/'streaming')
        result=streaming.run([row['url'] for row in records],stages=('extract','clean','export'),settings={
            'source_manifest':{'documents':records},'download':download_settings,'extract':{'repair_fonts':False}})
        expected_state='completed_with_failures' if include_failure else 'completed'
        assert result['state']==expected_state
        for name in ('download_results/download_results_sources.parquet','download_results/download_results_sources.parquet.partial.parquet'):
            pd.testing.assert_frame_equal(pd.read_parquet(sequential.output_dir/name),pd.read_parquet(streaming.output_dir/name))
        assert (streaming.output_dir/'downloads/source.pdf').read_bytes()==payload
        assert (streaming.output_dir/'export.jsonl').read_bytes()==(sequential.output_dir/'export.jsonl').read_bytes()
        before=len(requests)
        resumed=streaming.run(resume=result['run_id'])
        assert resumed['state']==expected_state and len(requests)==before==2*sequential_request_count
        with LedgerWriter(streaming.output_dir/'corpus.sqlite3') as writer:
            assert writer.connection.execute("SELECT COUNT(*) FROM attempts a JOIN tasks t USING(task_id) WHERE t.run_id=? AND t.stage='download' AND t.operation='download.acquire'",(result['run_id'],)).fetchone()[0]==1+int(include_failure)
    finally:
        server.shutdown();server.server_close();thread.join()
