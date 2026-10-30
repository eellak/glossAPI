"""Task seven: retention matrix, exact readers, crash tombstones and quota guards."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from glossapi.ledger import LedgerWriter
from glossapi.ledger.reader import plan_view, report_why, select_run
from glossapi.stages.disk_budget import DiskBudget, DiskBudgetError, require_disk_guard
from glossapi.stages.execution import sha256_file
from glossapi.stages.retention import RetentionController, retention_facts, resolved_settings


def settings(debug='off', memory_saving='normal'):
    return dict(debug=debug,memory_saving=memory_saving,disk_budget_bytes=4*1024**3,
        max_source_bytes=1024**2,max_attempt_bytes=4*1024**2)


def fixture(writer,root,*,kind='raw_markdown',retention='intermediate',ownership='managed',path=None):
    path=path or root/'payload.md'
    if not path.exists():path.write_text('exact bytes αβγ')
    with writer.transaction():
        run=writer.run(settings(),{}, {},code_revision='retention-test')
        document=writer.document('file:'+str(path))
        version=writer.version(document,source_sha256=sha256_file(path),stored_sha256=sha256_file(path))
        file=writer.artifact(run,'extract',kind,path.as_uri(),version_id=version,
            sha256=sha256_file(path),bytes=path.stat().st_size,ownership=ownership,retention_class=retention)
    return run,document,version,file,path


def consumer(writer,run,version,file,*,status='ok',operation='clean.test',coverage=None):
    with writer.transaction():
        task=writer.task(run,'clean',operation,{},version_id=version)
        attempt=writer.attempt(task,worker_id='test',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
        digest=writer.connection.execute('SELECT sha256 FROM files WHERE file_id=?',(file,)).fetchone()[0]
        writer.input(attempt,file,'selected_text',digest)
        if coverage is not None:
            evidence=writer.snapshot('stage_evidence',{'coverage_allows_source_release':coverage})
            writer.event(run,event_kind='decision',reason_code='clean_completed',task_id=task,
                version_id=version,attempt_id=attempt,details={'evidence_sha256':evidence})
        if status=='running':
            writer.event(run,event_kind='lifecycle',reason_code='stage_started',status='running',
                task_id=task,version_id=version,attempt_id=attempt)
        else:
            writer.finish(run,task,attempt,status=status,reason_code='clean_completed' if status=='ok' else 'clean_process_failed' if status=='failed' else 'not_in_plan')
    return task,attempt


@pytest.mark.parametrize('debug',['off','on'])
@pytest.mark.parametrize('memory_saving',['normal','extreme'])
@pytest.mark.parametrize('kind,retention,ownership,pinned',[
    ('source_payload','initial','initial_external',True),
    ('download_payload','initial','managed',False),
    ('source_snapshot','intermediate','managed',False),
    ('repaired_pdf','intermediate','managed',False),
    ('raw_markdown','intermediate','managed',False),
    ('ocr_native','recovery','managed',True),
    ('docling_json','intermediate','managed',False),
    ('annotation_partition','intermediate','managed',False),
    ('failed_partial','failed_attempt','managed',False),
    ('attempt_diagnostic','intermediate','managed',False),
    ('document_jsonl','final','managed',True),
    ('generation_manifest','evidence','managed',True),
])
def test_artifact_matrix_all_four_settings(tmp_path,debug,memory_saving,kind,retention,ownership,pinned):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,document,version,file,path=fixture(writer,tmp_path,kind=kind,retention=retention,ownership=ownership)
        task,_=consumer(writer,run,version,file)
        controller=RetentionController(writer,tmp_path,run,settings(debug,memory_saving))
        controller.declare(file,[task],source=kind=='download_payload')
        expected=not pinned and (memory_saving=='extreme' if kind=='download_payload' else debug=='off')
        assert controller.delete(file)==expected
        assert path.exists()!=expected
        report=report_why(writer.connection,document,run=select_run(writer.connection,run))
        entry=next(item for item in report['files'] if item['file_id']==file)
        assert entry['presence']==('deleted' if expected else 'present')
        assert entry['retention']['consumer_ok_event_ids']
        if expected:
            assert [item['reason_code'] for item in entry['retention']['deletions']]==(['delete_completed'] if debug=='off' else ['delete_planned','delete_completed'])
            assert entry['retention']['deletions'][-1]['details']['sha256']==entry['sha256']


@pytest.mark.parametrize('status',['failed','skipped','running','unused','partial'])
def test_unused_failed_partial_or_active_input_cannot_be_released(tmp_path,status):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path,kind='download_payload',retention='initial')
        tasks=[]
        if status!='unused':
            task,_=consumer(writer,run,version,file,status='ok' if status=='partial' else status,coverage=False if status=='partial' else None)
            tasks=[task]
        controller=RetentionController(writer,tmp_path,run,settings(memory_saving='extreme'))
        controller.declare(file,tasks,source=True)
        assert not controller.delete(file) and path.exists()


def test_hash_change_cannot_unlock_deletion_and_external_symlinks_survive(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path)
        task,_=consumer(writer,run,version,file)
        controller=RetentionController(writer,tmp_path,run)
        controller.declare(file,[task])
        path.write_text('different bytes')
        assert not controller.delete(file) and path.read_text()=='different bytes'
        outside=tmp_path/'external';outside.write_text('protected')
        path.unlink();path.symlink_to(outside)
        assert not controller.delete(file) and path.is_symlink() and outside.read_text()=='protected'


def test_fanout_reader_and_explicit_unselected_edge(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path)
        first,_=consumer(writer,run,version,file,operation='clean.first')
        second,_=consumer(writer,run,version,file,status='skipped',operation='clean.second')
        controller=RetentionController(writer,tmp_path,run)
        controller.declare(file,[first,second],waived=[second])
        assert controller.delete(file) and not path.exists()


def test_fanout_failed_reader_is_not_an_implicit_waiver(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path)
        first,_=consumer(writer,run,version,file,operation='clean.first')
        second,_=consumer(writer,run,version,file,status='failed',operation='clean.second')
        controller=RetentionController(writer,tmp_path,run)
        controller.declare(file,[first,second])
        assert not controller.delete(file) and path.exists()
        assert retention_facts(writer.connection,file)['outstanding_consumers']==[second]


def test_shared_path_requires_every_logical_reference(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path)
        with writer.transaction():
            other=writer.artifact(run,'extract','duplicate',path.as_uri(),version_id=version,
                sha256=sha256_file(path),bytes=path.stat().st_size)
        first,_=consumer(writer,run,version,file,operation='clean.first')
        second,_=consumer(writer,run,version,other,operation='clean.second')
        controller=RetentionController(writer,tmp_path,run)
        controller.declare(file,[first])
        assert not controller.delete(file)
        controller.declare(other,[second])
        assert controller.delete(file)
        assert {row[0] for row in writer.connection.execute('SELECT presence FROM current_file_state')}=={'deleted'}


def test_hardlinked_blob_cannot_release_an_outstanding_logical_reader(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path)
        shared=tmp_path/'shared.md';os.link(path,shared)
        with writer.transaction():
            other=writer.artifact(run,'extract','shared',shared.as_uri(),version_id=version,
                sha256=sha256_file(shared),bytes=shared.stat().st_size)
        first,_=consumer(writer,run,version,file,operation='clean.first')
        controller=RetentionController(writer,tmp_path,run);controller.declare(file,[first])
        assert not controller.delete(file) and path.exists() and shared.exists()
        second,_=consumer(writer,run,version,other,operation='clean.second')
        controller.declare(other,[second])
        assert controller.delete(file) and controller.delete(other)


@pytest.mark.parametrize('boundary',['after_intent','after_unlink'])
def test_deletion_recovers_on_both_sides_of_unlink(tmp_path,boundary):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path)
        task,_=consumer(writer,run,version,file)
        def crash(point,*_):
            if point==boundary:raise RuntimeError('simulated death')
        controller=RetentionController(writer,tmp_path,run,fault=crash)
        controller.declare(file,[task])
        with pytest.raises(RuntimeError,match='simulated death'):controller.delete(file)
        assert path.exists()==(boundary=='after_intent')
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        controller=RetentionController(writer,tmp_path,run)
        assert controller.recover()==1 and not path.exists()
        assert controller.recover()==0
        assert writer.connection.execute('PRAGMA foreign_key_check').fetchone() is None


@pytest.mark.parametrize('boundary',['after_intent','after_unlink'])
def test_real_process_death_during_deletion_preserves_recovery_receipt(tmp_path,boundary):
    import subprocess,sys
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path)
        task,_=consumer(writer,run,version,file)
        RetentionController(writer,tmp_path,run).declare(file,[task])
    script='''import os,sys
from pathlib import Path
from glossapi.ledger import LedgerWriter
from glossapi.stages.retention import RetentionController
root,run,file,boundary=sys.argv[1:]
def die(point,*_):
    if point==boundary:os._exit(91)
with LedgerWriter(Path(root)/'corpus.sqlite3') as writer:
    RetentionController(writer,root,run,fault=die).delete(file)
'''
    process=subprocess.run([sys.executable,'-c',script,str(tmp_path),run,file,boundary])
    assert process.returncode==91
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        assert RetentionController(writer,tmp_path,run).recover()==1
        assert not path.exists()


def test_unlink_failure_keeps_payload_and_errno(tmp_path,monkeypatch):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path)
        task,_=consumer(writer,run,version,file)
        controller=RetentionController(writer,tmp_path,run);controller.declare(file,[task])
        with monkeypatch.context() as patch:
            patch.setattr(os,'unlink',lambda *a,**k:(_ for _ in ()).throw(PermissionError(13,'denied')))
            assert not controller.delete(file)
        assert path.exists()
        facts=retention_facts(writer.connection,file)
        assert facts['deletions'][-1]['details']['errno']==13


def test_missing_without_intent_is_not_a_retention_tombstone(tmp_path):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,version,file,path=fixture(writer,tmp_path)
        task,_=consumer(writer,run,version,file)
        controller=RetentionController(writer,tmp_path,run);controller.declare(file,[task])
        path.unlink()
        assert not controller.delete(file)
        assert writer.connection.execute('SELECT presence FROM current_file_state WHERE file_id=?',(file,)).fetchone()[0]=='missing'
        assert not retention_facts(writer.connection,file)['deletions']


def test_full_disk_oversized_reservations_and_pinned_payloads(tmp_path,monkeypatch):
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,_,_,_,path=fixture(writer,tmp_path,retention='final')
        budget=DiskBudget(writer,tmp_path,run,{'disk_budget_bytes':10})
        assert not budget.can_reserve(11)
        assert budget.present_bytes()==path.stat().st_size
        with pytest.raises(DiskBudgetError):budget.reserve('too-big',11)
        monkeypatch.setattr(os,'statvfs',lambda *a:SimpleNamespace(f_bavail=0,f_frsize=1))
        assert not DiskBudget(writer,tmp_path,run,{}).can_reserve(1)


def test_unknown_length_source_stops_before_chunk_write():
    from glossapi.stages.receipts import capture_transport, request_receipt, received_bytes
    with capture_transport('standard',{'max_source_bytes':7}),request_receipt('https://example/source') as receipt:
        received_bytes(4)
        with pytest.raises(DiskBudgetError,match='disk_budget_exceeded'):received_bytes(4)
        assert receipt['source_entity_bytes']==4


def test_extreme_requires_caps_and_a_real_enforced_filesystem(tmp_path):
    with pytest.raises(ValueError,match='disk_budget'):resolved_settings({'memory_saving':'extreme'})
    with pytest.raises(DiskBudgetError,match='disk_guard_unavailable'):require_disk_guard(tmp_path,settings(memory_saving='extreme'))
    plan=plan_view(selectors=['https://example/source'],stages=['ocr','math'],settings=settings(memory_saving='extreme'))
    assert plan['settings']['memory_saving']=='extreme'
    assert 'last planned reader' in plan['settings_help']['memory_saving']


def test_debug_off_summary_and_debug_on_native_stages_have_equal_text(tmp_path):
    from test_ledger_text_stages import run_clean
    results={}
    for debug in ('off','on'):
        _,request,result=run_clean(tmp_path/debug,'foo /space bar\n\n\nΚαλημέρα κόσμε',debug=debug)
        assert result.status=='ok',result.error
        results[debug]=(request,result)
    assert (results['off'][0].work_directory/'clean_markdown/source.md').read_bytes()==(results['on'][0].work_directory/'clean_markdown/source.md').read_bytes()
    summary=results['off'][1].decisions[0]['edits']
    assert 'edits' not in summary and 'composition_map' not in summary
    assert summary['ordered_edits_sha256'] and summary['rules']
    assert not (results['off'][0].work_directory/'edit-replacements.json').exists()
    assert (results['on'][0].work_directory/'native_evidence/native_stages/stage2_detailed_table_report.csv').exists()
    assert (results['on'][0].work_directory/'edit-replacements.json').exists()


@pytest.mark.parametrize('debug',['off','on'])
def test_public_clean_keeps_native_stages_only_with_debug_on(tmp_path,debug):
    from glossapi import Corpus
    corpus=Corpus(tmp_path/'input',tmp_path/'output')
    corpus._retention_settings=settings(debug=debug)
    (corpus.markdown_dir/'source.md').write_text('hello /space Καλημέρα κόσμε\n\n')
    corpus.clean(drop_bad=False,num_threads=1)
    assert (corpus.cleaned_markdown_dir/'source.md').exists()
    native=list((corpus.output_dir/'.glossapi/artifacts').rglob('stage2_detailed_table_report.csv'))
    assert bool(native)==(debug=='on')
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        decisions=[]
        for encoded, in writer.connection.execute("SELECT json FROM snapshots WHERE kind='stage_evidence'"):
            from glossapi.ledger.encoding import unpack_evidence
            decisions.append(unpack_evidence(json.loads(encoded)))
        edit=next(value['edits'] for value in decisions if 'edits' in value)
        assert ('edits' in edit)==(debug=='on')


def test_cross_filesystem_publication_retires_only_verified_scratch(tmp_path,monkeypatch):
    from glossapi.stages.contracts import ArtifactReference,DocumentSnapshot,OutputArtifact,StageSpecification,StageResult
    from glossapi.stages.sequential import SequentialStageRunner
    class Stage:
        name='extract';contract_version=1
        def run_document(self,request,*,reporter):
            request.work_directory.mkdir(parents=True)
            (request.work_directory/'final.md').write_text('durable final')
            return StageResult(request.task_id,request.attempt_id,request.lease_token,'ok','extract_completed',
                (OutputArtifact('raw_markdown','final.md',None,'final',True),),(),{})
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run,document,version,file,path=fixture(writer,tmp_path,ownership='initial_external',retention='initial')
        original=os.rename
        def cross_device(source,target):
            if '.glossapi/staging' in str(source):raise OSError(18,'cross device')
            return original(source,target)
        monkeypatch.setattr(os,'rename',cross_device)
        result,workspace=SequentialStageRunner(writer,tmp_path).execute(Stage(),DocumentSnapshot(document,version,path.as_uri()),StageSpecification('extract.test',required_output_kinds=('raw_markdown',)))
        assert (workspace/'final.md').read_text()=='durable final'
        assert not list((tmp_path/'.glossapi/staging').iterdir())
        assert writer.connection.execute("SELECT COUNT(*) FROM current_file_state WHERE kind='scratch_copy' AND presence='deleted'").fetchone()[0]==1
        assert path.exists()


@pytest.mark.parametrize('debug',['off','on'])
def test_extreme_waits_for_last_frozen_pdf_reader_and_removes_all_source_copies(tmp_path,debug):
    from glossapi.stages.retention import retire_downloaded_sources
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        path=tmp_path/'source.pdf';path.write_bytes(b'%PDF-original-entity')
        with writer.transaction():
            document=writer.document('https://example/source.pdf')
            version=writer.version(document,source_sha256=sha256_file(path),stored_sha256=sha256_file(path))
            operations=[{'stage':stage,'document':{'version_id':version},'specification':{'input_roles':[] if stage=='download' else ['source_payload']},'inputs':[]}
                for stage in ('download','extract','math')]
            run=writer.run(settings(debug,'extreme'),{'operations':operations},{},code_revision='test')
            source=writer.artifact(run,'download','download_payload',path.as_uri(),version_id=version,
                sha256=sha256_file(path),bytes=path.stat().st_size,retention_class='initial')
            tasks=[writer.task(run,stage,stage+'.test',{},version_id=version,ordinal=index) for index,stage in enumerate(('download','extract','math'))]
            for task in tasks:
                writer.event(run,event_kind='lifecycle',reason_code='stage_planned',status='planned',task_id=task,version_id=version)
        copy=tmp_path/'source-copy.pdf';copy.write_bytes(path.read_bytes())
        with writer.transaction():
            writer.artifact(run,'extract','source_snapshot',copy.as_uri(),version_id=version,
                sha256=sha256_file(copy),bytes=copy.stat().st_size)
        def finish(task,stage):
            with writer.transaction():
                attempt=writer.attempt(task,worker_id='reader',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
                writer.input(attempt,source,'source_payload',sha256_file(path))
                writer.finish(run,task,attempt,status='ok',reason_code=stage+'_completed')
        finish(tasks[1],'extract')
        retire_downloaded_sources(writer,tmp_path,run,settings(debug,'extreme'))
        assert path.exists()
        finish(tasks[2],'math')
        retire_downloaded_sources(writer,tmp_path,run,settings(debug,'extreme'))
        assert not path.exists() and not copy.exists()
        assert retention_facts(writer.connection,source)['consumer_ok_event_ids']


def test_why_follows_bound_aggregate_lineage_after_fragment_collection(tmp_path):
    from test_ledger_products import corpus_fixture
    corpus=corpus_fixture(tmp_path)
    corpus.markdown_dir=corpus.cleaned_markdown_dir
    corpus.section_classifier_model_path=Path(__file__).resolve().parents[1]/'src/glossapi/models/section_classifier.joblib'
    corpus.section();corpus.annotate();corpus.jsonl(corpus.output_dir/'export.jsonl')
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        report=report_why(writer.connection,str(corpus.cleaned_markdown_dir/'alpha.md'),run=select_run(writer.connection))
        assert {'sections','annotate','export'}<={stage['stage'] for stage in report['stages']}
        included=[item for item in report['inclusion'] if item['decision']=='include']
        assert included and all(item['record_hash'] and item['manifest_hash'] for item in included)
        lineage=[item for stage in report['stages'] for attempt in stage['attempts'] for decision in attempt['decisions'] for item in decision['details'].get('lineage',[])]
        assert lineage and {item['filename'] for item in lineage}=={'alpha'}
        assert any(item['presence']=='deleted' and item['retention']['deletions'] for item in report['files'])


def test_streaming_raw_mirror_matches_saved_snapshot_before_retention(tmp_path):
    import hashlib
    from glossapi import Corpus
    from test_ledger_stages import simple_pdf
    source=simple_pdf(tmp_path/'source.pdf')
    corpus=Corpus(tmp_path/'empty',tmp_path/'output')
    from tools.harness.retention import RetentionSnapshots,verified_snapshots
    corpus._retention_observer=RetentionSnapshots(corpus.output_dir,tmp_path/'retention_snapshots')
    original=corpus.extract;snapshots={}
    from functools import wraps
    @wraps(original)
    def save_snapshot(*args,**options):
        returned=original(*args,**options)
        snapshots.update({path:path.read_bytes() for path in (corpus.output_dir/'markdown').glob('*.md')})
        return returned
    corpus.extract=save_snapshot
    result=corpus.run([source],stages=('extract','clean','export'),settings={'extract':{'repair_fonts':False}})
    assert result['state']=='completed' and snapshots
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        for path,body in snapshots.items():
            assert not path.exists()
            file=writer.connection.execute('SELECT file_id,sha256,presence FROM current_file_state WHERE uri=? ORDER BY event_id DESC LIMIT 1',(path.as_uri(),)).fetchone()
            assert file['presence']=='deleted' and file['sha256']==hashlib.sha256(body).hexdigest()
            facts=retention_facts(writer.connection,file['file_id'])
            assert facts['consumer_ok_event_ids'] and not facts['outstanding_consumers']
        assert (corpus.output_dir/'clean_markdown/source.md').is_file()
        assert source.is_file()
    saved=verified_snapshots(corpus.output_dir,tmp_path/'retention_snapshots')
    assert saved and all(Path(item['snapshot']).is_file() for item in saved)
    from copy import deepcopy
    from tools.harness.common import STAGES
    from tools.harness.corpus import compare_arms
    raw={item['path']:{'sha256':item['sha256'],'bytes':item['bytes']} for item in saved}
    final={'export.jsonl':{'sha256':sha256_file(corpus.output_dir/'export.jsonl'),'bytes':(corpus.output_dir/'export.jsonl').stat().st_size}}
    baseline={'stages':{stage:dict(status='ok',error=None,wrapper_return=None,products={**raw,**final},removed_files=[]) for stage in STAGES}}
    candidate={'stages':{stage:dict(status='ok',error=None,wrapper_return=None,products=dict(final),removed_files=[]) for stage in STAGES},'retention_deletions':saved,'retention_corpus':str(corpus.output_dir),'retention_snapshots':str(tmp_path/'retention_snapshots')}
    assert compare_arms(baseline,candidate)['passes']
    changed=deepcopy(candidate);changed['stages']['export']['products']['export.jsonl']['sha256']='0'*64
    assert not compare_arms(baseline,changed)['passes']
    snapshot=Path(saved[0]['snapshot']);body=snapshot.read_bytes();snapshot.write_bytes(body+b'changed')
    with pytest.raises(AssertionError):verified_snapshots(corpus.output_dir,tmp_path/'retention_snapshots')
    snapshot.write_bytes(body)


@pytest.mark.parametrize('boundary',['after_sealing','after_unlink'])
def test_full_run_resumes_retention_without_repeating_committed_work(tmp_path,boundary):
    from glossapi import Corpus
    from test_ledger_stages import simple_pdf
    source=simple_pdf(tmp_path/'source.pdf')
    corpus=Corpus(tmp_path/'empty',tmp_path/'output');saved={}
    def interrupt_collection(controller,rows):
        if not (corpus.output_dir/'export.jsonl').exists():return
        saved['run_id']=controller.run_id
        saved['export']=(corpus.output_dir/'export.jsonl').read_bytes()
        if boundary=='after_sealing':raise RuntimeError('retention process interrupted')
        def interrupt_unlink(point,*_):
            if point=='after_unlink':raise RuntimeError('retention process interrupted')
        controller.fault=interrupt_unlink
    corpus._retention_observer=interrupt_collection
    with pytest.raises(RuntimeError,match='retention process interrupted'):
        corpus.run([source],stages=('extract','clean','export'),settings={'extract':{'repair_fonts':False}})
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        completed={row[0] for row in writer.connection.execute("SELECT a.attempt_id FROM attempts a JOIN tasks t USING(task_id) JOIN events e USING(attempt_id) WHERE t.run_id=? AND e.event_kind='lifecycle' AND e.status='ok'",(saved['run_id'],))}
    resumed=Corpus(corpus.input_dir,corpus.output_dir).run(resume=saved['run_id'])
    assert resumed['state']=='completed'
    assert (corpus.output_dir/'export.jsonl').read_bytes()==saved['export'] and source.exists()
    assert not (corpus.output_dir/'cleaning_report.parquet').exists()
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        after={row[0] for row in writer.connection.execute("SELECT a.attempt_id FROM attempts a JOIN tasks t USING(task_id) JOIN events e USING(attempt_id) WHERE t.run_id=? AND e.event_kind='lifecycle' AND e.status='ok'",(saved['run_id'],))}
        assert after==completed


@pytest.mark.parametrize('debug',['off','on'])
def test_streaming_parent_applies_debug_policy_to_deferred_native_report(tmp_path,debug):
    from glossapi import Corpus
    from test_ledger_stages import simple_pdf
    source=simple_pdf(tmp_path/'source.pdf')
    corpus=Corpus(tmp_path/'empty',tmp_path/'output')
    result=corpus.run([source],stages=('extract','clean','export'),settings={'debug':debug,'extract':{'repair_fonts':False}})
    assert result['state']=='completed' and source.exists()
    report=corpus.output_dir/'cleaning_report.parquet'
    assert report.exists()==(debug=='on')
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        row=writer.connection.execute("SELECT * FROM current_file_state WHERE uri=? AND kind='temporary_cleaning_report' ORDER BY event_id DESC LIMIT 1",(report.as_uri(),)).fetchone()
        assert row and row['presence']==('deleted' if debug=='off' else 'present')
        if debug=='off':
            facts=retention_facts(writer.connection,row['file_id'])
            assert facts['consumer_ok_event_ids'] and not facts['outstanding_consumers'] and not facts['pins']
            assert any(event['reason_code']=='delete_completed' for event in facts['deletions'])
