"""Stability arithmetic, receipt binding, CPU sharding and native worker hook."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from glossapi.scripts import batched_stability_benchmark as b2


def row(name, status='ok', pages=2, wall=1., gpu='0'):
    return dict(file=name, status=status, pages=pages, attributed_wall_seconds=wall, gpu=gpu)


def test_summary_arithmetic_and_paired_cohort():
    docs = [row('a'), row('b', pages=3, wall=2), row('c','timeout',0), row('d','failed',0), row('e','incomplete',0)]
    sessions = [dict(id='x', gpu='0', wall_seconds=8., initialization_seconds=2., stats=dict(
        counters=dict(export_worker_kills=2, export_recovery_cycles=2, export_recovery_unresolved=1, export_worker_respawns=1, recovery_seconds=1.5), events=[]))]
    result=b2.summarize(docs, 6., sessions, ['b'])
    assert [result[f'documents_{s}'] for s in b2.STATUSES] == [2,1,1,1]
    assert result['pages_done']==5
    assert result['seconds_per_page']==1.2
    assert result['common_seconds_per_page']==.5
    assert result['common_worker_seconds_per_page']==pytest.approx(2/3)
    assert result['worker_wall_seconds']==8
    assert result['recovery_count']==2 and result['completed_recovery_count']==1
    assert result['unresolved_recoveries']==1 and result['recovery_cost_censored']
    assert result['seconds_per_completed_recovery']==1.5


def test_break_even_formula_and_no_cost():
    result=b2.break_even(.2, 40, 4, 32)
    assert result['measured_seconds_per_recovery']==8
    assert result['failures_per_document']==1
    assert result['failures_per_1000_pages']==25
    assert b2.break_even(.1, 10, 0, 0)['failures_per_document'] is None
    assert b2.break_even(-.1, 10, 2, 4)['failures_per_document']==0
    assert b2.break_even(.5, 40, 2, 4)['failures_per_document']==10  # no misleading clipping at 100%


def test_attribution_integrates_overlap_and_idle():
    docs=[dict(file='a',started_seconds=1.,finished_seconds=4.),dict(file='b',started_seconds=2.,finished_seconds=3.)]
    actual=b2.attributable_wall(docs,5.)
    assert actual=={'a':3.5,'b':1.5}
    assert sum(actual.values())==5.


def test_shard_original_indices_and_pin_before_adapter(monkeypatch):
    paths=[Path(f'd{i}.pdf') for i in range(5)]
    assert b2.shard_documents(paths,['3','7'])=={'3':[paths[0],paths[2],paths[4]],'7':[paths[1],paths[3]]}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3,7')
    args=SimpleNamespace(fake_devices=0,use_gpus='multi',device='cuda')
    assert b2.visible_devices(args)==['3','7']
    assert b2.pin_device('7')=='cuda:0'
    assert b2.os.environ['CUDA_VISIBLE_DEVICES']=='7'
    assert b2.pin_device('1',fake=True)=='cpu'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-one,GPU-two')
    assert b2.visible_devices(args)==['GPU-one','GPU-two']


def test_stats_are_immutable_and_snapshots_detached():
    runner=b2.core.BatchedExtractor(adapter=object())
    runner._stability.event('export_deadlines','a.pdf',1)
    snapshot=runner.stats
    assert snapshot.counters['export_deadlines']==1
    with pytest.raises(TypeError): snapshot.counters['export_deadlines']=3
    runner._stability.event('export_deadlines','b.pdf',2)
    assert snapshot.counters['export_deadlines']==1
    assert runner.stats.counters['export_deadlines']==2


def test_collect_resume_and_original_hash_binding(tmp_path):
    out=tmp_path/'batched'
    session=dict(id='x',gpu='0',round='one',documents=[dict(row('a.pdf'), input_sha256='original',
        wall_seconds=1., started_seconds=0.,finished_seconds=1.)],stats=dict(counters={},events=[]),
        wall_seconds=1.,conversion_started_monotonic=10.,conversion_finished_monotonic=11.)
    b2.atomic_json(out/'sessions/x.json',session)
    (out/'markdown').mkdir()
    (out/'markdown/a.md').write_text('a')
    docs,_,wall=b2.collect(out,[Path('a.pdf'),Path('b.pdf')])
    assert docs[0]['status']=='ok' and docs[0]['input_sha256']=='original'
    assert docs[1]['status']=='incomplete' and wall==1
    (out/'markdown/a.md').unlink()
    assert b2.collect(out,[Path('a.pdf')])[0][0]['status']=='incomplete'


def test_install_sequential_native_worker_uses_configured_device(monkeypatch,tmp_path):
    adapter=SimpleNamespace(exporter=SimpleNamespace(),pipeline=SimpleNamespace(pipeline_options=SimpleNamespace(model_dump=lambda **kw:{})))
    seen=[]
    monkeypatch.setattr(b2.reference,'docling_version',lambda:'2.81.0')
    monkeypatch.setattr(b2,'make_adapter',lambda cfg,device:seen.append((device,b2.os.environ['CUDA_VISIBLE_DEVICES'])) or adapter)
    monkeypatch.setattr(b2,'sequential_documents',lambda paths,a,s,out: (['a.pdf'],[]))
    corpus=SimpleNamespace()
    config=dict(fake_devices=0,mode_out=str(tmp_path),round='one',input_sha256={})
    b2.install_sequential_worker(corpus,config,'5','gpu5-w0')
    assert seen==[('cuda:0','5')]
    reports=[]
    corpus.extractor.batch_result_callback=lambda ok,bad:reports.append((ok,bad))
    corpus.extract(file_paths=['a.pdf'])
    assert reports==[(['a.pdf'],[])]


def test_pairing_rejects_different_page_counts():
    with pytest.raises(ValueError,match='page counts'):
        b2.comparison([row('a',pages=1)],[row('a',pages=2)])


def test_native_multi_dispatch_resume_and_hash_drift(monkeypatch,tmp_path):
    import glossapi
    source=tmp_path/'pdfs'
    source.mkdir()
    (source/'a.pdf').write_bytes(b'original input')
    calls=[]
    class FakeCorpus:
        def __init__(self,**kwargs): pass
        def extract(self,**kwargs):
            calls.append(kwargs)
            config=kwargs['_stability_benchmark']
            output=Path(config['mode_out'])
            (output/'markdown').mkdir()
            (output/'markdown/a.md').write_text('content')
            session=dict(id='native',gpu='3',round=config['round'],documents=[dict(row('a.pdf'),gpu='3')],
                stats=dict(counters={},events=[]),wall_seconds=1.,conversion_started_monotonic=1.,conversion_finished_monotonic=2.)
            b2.atomic_json(output/'sessions/native.json',session)
    monkeypatch.setattr(glossapi,'Corpus',FakeCorpus)
    monkeypatch.setattr(b2.reference,'page_count',lambda p:2)
    monkeypatch.setattr(b2.reference,'docling_version',lambda:'2.81.0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3,7')
    argv=['--pdfs',str(source),'--out',str(tmp_path/'out'),'--mode','sequential',
          '--use-gpus','multi','--device','cuda','--workers-per-device','2']
    assert b2.main(argv)==0
    assert calls[0]['use_gpus']=='multi'
    assert calls[0]['devices']==[3,7] and calls[0]['workers_per_device']==2
    assert calls[0]['batched'] is False and calls[0]['repair_fonts'] is False
    assert b2.main(argv+['--resume'])==0 and len(calls)==1
    (source/'a.pdf').write_bytes(b'changed')
    with pytest.raises(SystemExit): b2.main(argv+['--resume'])


def test_real_queue_worker_installs_explicit_hook(monkeypatch,tmp_path):
    import glossapi
    import queue
    from glossapi.corpus import corpus_orchestrator as native
    tasks,results=queue.Queue(),queue.Queue()
    tasks.put(['a.pdf'])
    installed=[]
    class FakeCorpus:
        def __init__(self,**kwargs): pass
        def prime_extractor(self,**kwargs): raise AssertionError('benchmark hook should initialize adapter')
    monkeypatch.setattr(glossapi,'Corpus',FakeCorpus)
    monkeypatch.setattr(native,'_maybe_import_torch',lambda:None)
    monkeypatch.setenv('GLOSSAPI_WORKER_LOG_VERBOSE','0')
    monkeypatch.delenv('GLOSSAPI_WORKER_LOG_DIR',raising=False)
    def install(corpus,config,gpu,worker):
        installed.append((config,gpu,worker))
        corpus.extractor=SimpleNamespace(release_resources=lambda:None)
        def extract(**kwargs):
            assert kwargs['batched'] is False
            corpus.extractor.batch_result_callback(kwargs['file_paths'],[])
        corpus.extract=extract
    monkeypatch.setattr(b2,'install_sequential_worker',install)
    with pytest.raises(SystemExit) as result:
        native.gpu_extract_worker_queue(7,1,'gpu7-w1',str(tmp_path),str(tmp_path),tasks,
            False,False,False,False,False,'pdf',2,False,True,False,'docling',result_q=results,
            stability_benchmark={'test':True})
    assert result.value.code==0
    assert installed==[({'test':True},'7','gpu7-w1')]
    assert results.get_nowait()['processed']==['a.pdf']
    assert results.get_nowait()['event']=='exit'


def test_multi_aggregate_uses_elapsed_envelope_not_worker_sum(monkeypatch,tmp_path):
    paths=[Path('a.pdf'),Path('b.pdf')]
    for mode in ('sequential','batched'):
        out=tmp_path/mode
        b2.atomic_json(out/'receipt.json',{})
        (out/'markdown').mkdir()
        for gpu,name,wall,pages in [('0','a.pdf',4.,2),('1','b.pdf',6.,3)]:
            (out/'markdown'/f'{Path(name).stem}.md').write_text('content')
            session=dict(id=gpu,gpu=gpu,round='round',documents=[row(name,pages=pages,wall=wall,gpu=gpu)],
                stats=dict(counters={},events=[]),wall_seconds=wall,
                conversion_started_monotonic=10.,conversion_finished_monotonic=10.+wall,
                effective_options={'layout_batch_size':4 if mode=='sequential' else 8,'do_ocr':False})
            b2.atomic_json(out/'sessions'/f'{gpu}.json',session)
    monkeypatch.setattr(b2.reference,'equality',lambda *a:[dict(passes_gate=True)])
    config=dict(input_sha256={},fidelity='markdown',use_gpus='multi',fake_devices=2,devices=['0','1'],docling_version='2.81.0',device='cpu')
    result=b2.publish(tmp_path,paths,config)['batched']
    assert result['wall_seconds']==6.
    assert result['worker_wall_seconds']==10.
    assert result['seconds_per_page']==1.2
    assert result['common_seconds_per_page']==1.2
    assert result['common_worker_seconds_per_page']==2.
    assert result['fidelity_verified'] and result['comparable_options_verified']
    assert result['per_gpu']['0']['pages_done']==2
    assert result['per_gpu']['1']['pages_done']==3
    assert result['break_even']['failures_per_document'] is None


def test_batched_shard_adapter_sees_only_pinned_gpu(monkeypatch,tmp_path):
    seen=[]
    options=SimpleNamespace(model_dump=lambda **kw:{'accelerator_options':{'device':'cuda:0'}})
    adapter=SimpleNamespace(pipeline=SimpleNamespace(pipeline_options=options))
    monkeypatch.setattr(b2.reference,'docling_version',lambda:'2.81.0')
    monkeypatch.setattr(b2,'make_adapter',lambda config,device:seen.append((device,b2.os.environ['CUDA_VISIBLE_DEVICES'])) or adapter)
    monkeypatch.setattr(b2,'batched_documents',lambda paths,a,session,out:None)
    config=dict(use_gpus='multi',fake_devices=0,mode='batched',mode_out=str(tmp_path),input_sha256={})
    b2.shard_worker(config,[], 'GPU-mocked-a100','one')
    assert seen==[('cuda:0','GPU-mocked-a100')]


def test_controller_failure_still_publishes_resumable_receipt(monkeypatch,tmp_path):
    import glossapi
    source=tmp_path/'pdfs'
    source.mkdir()
    (source/'a.pdf').write_bytes(b'original')
    class BrokenCorpus:
        def __init__(self,**kwargs): pass
        def extract(self,**kwargs): raise EOFError('manager startup denied')
    monkeypatch.setattr(glossapi,'Corpus',BrokenCorpus)
    monkeypatch.setattr(b2.reference,'page_count',lambda path:2)
    monkeypatch.setattr(b2.reference,'docling_version',lambda:'2.81.0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    assert b2.main(['--pdfs',str(source),'--out',str(tmp_path/'out'),'--mode','sequential',
                    '--use-gpus','multi','--device','cuda'])==1
    import json
    summary=json.loads((tmp_path/'out/sequential/summary.json').read_text())
    assert summary['documents_incomplete']==1
    assert summary['initialization_errors']==['EOFError: manager startup denied']
    assert summary['documents_detail'][0]['input_sha256']


def test_resume_can_finish_after_historical_controller_error(monkeypatch,tmp_path):
    import glossapi
    source=tmp_path/'pdfs'
    source.mkdir()
    (source/'a.pdf').write_bytes(b'original')
    attempts=[]
    class RecoveringCorpus:
        def __init__(self,**kwargs): pass
        def extract(self,**kwargs):
            attempts.append(1)
            if len(attempts)==1:
                raise EOFError('old bootstrap failure')
            config=kwargs['_stability_benchmark']
            out=Path(config['mode_out'])
            (out/'markdown').mkdir(exist_ok=True)
            (out/'markdown/a.md').write_text('content')
            session=dict(id='recovered',gpu='0',round=config['round'],documents=[row('a.pdf')],
                stats=dict(counters={},events=[]),wall_seconds=1.,conversion_started_monotonic=1.,conversion_finished_monotonic=2.)
            b2.atomic_json(out/'sessions/recovered.json',session)
    monkeypatch.setattr(glossapi,'Corpus',RecoveringCorpus)
    monkeypatch.setattr(b2.reference,'page_count',lambda path:2)
    monkeypatch.setattr(b2.reference,'docling_version',lambda:'2.81.0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    args=['--pdfs',str(source),'--out',str(tmp_path/'out'),'--mode','sequential','--use-gpus','multi','--device','cuda']
    assert b2.main(args)==1
    assert b2.main(args+['--resume'])==0


def test_native_benchmark_selection_ignores_stale_production_skip_state(monkeypatch,tmp_path):
    import multiprocessing
    import queue
    from glossapi import Corpus
    from glossapi.corpus import phase_extract as phase
    delivered=[]
    class State:
        def __init__(self,*args,**kwargs): pass
        def load(self): return set(), {'a'}
        def save(self,*args): pass
    class Process:
        pid,exitcode=12345,0
        def __init__(self,target,args,kwargs): self.args,self.kwargs=args,kwargs
        def start(self):
            while True:
                try: items=self.args[5].get_nowait()
                except queue.Empty: break
                delivered.extend(items)
                self.args[17].put(dict(event='batch',processed=items,problematic=[]))
        def is_alive(self): return False
        def join(self,**kwargs): pass
    context=SimpleNamespace(Manager=lambda:SimpleNamespace(dict=dict,shutdown=lambda:None),Queue=queue.Queue,Process=Process)
    monkeypatch.setattr(multiprocessing,'get_context',lambda method:context)
    monkeypatch.setattr(phase,'_ProcessingStateManager',State)
    monkeypatch.setattr(phase._SkiplistManager,'load',lambda self:{'a'})
    monkeypatch.setattr(Corpus,'_finalize_font_repair_metadata',lambda *args:None)
    monkeypatch.setattr(phase,'_estimate_extract_work_pages',lambda path:1)
    source=tmp_path/'downloads/a.pdf'
    source.parent.mkdir()
    source.write_bytes(b'input')
    Corpus(tmp_path,tmp_path).extract(file_paths=[source],phase1_backend='docling',use_gpus='multi',
        devices=[0],batched=False,skip_existing=False,_stability_benchmark={'enabled':True})
    assert delivered==[str(source.resolve())]


def session_config(tmp_path, names=('a.pdf',), mode='sequential'):
    return dict(mode_out=str(tmp_path/mode), mode=mode, expected_pages={n:2 for n in names},
        input_sha256={n:'hash' for n in names}, max_pages=2, inventory_errors={})


def test_incremental_checkpoint_normalizes_partial_and_ignores_torn_tail(tmp_path):
    config = session_config(tmp_path)
    session = b2.Session(config, 'single', 'worker', 'one')
    session.save()
    session.begin()
    session.pause()
    session.checkpoint('a.pdf', dict(row('a.pdf', pages=1), started_seconds=0., finished_seconds=.01))
    records = b2.read_jsonl(session.journal)
    assert len(records) == 1 and records[0]['document']['status'] == 'incomplete'
    assert records[0]['document']['pages'] == 0
    with session.journal.open('ab') as stream: stream.write(b'{"document":')
    docs, _, _ = b2.collect(Path(config['mode_out']), [Path('a.pdf')], config['expected_pages'])
    assert docs[0]['status'] == 'incomplete'


def test_legacy_partial_success_is_retried_on_resume(monkeypatch, tmp_path):
    import json
    source = tmp_path/'pdfs'
    source.mkdir()
    (source/'a.pdf').write_bytes(b'input')
    monkeypatch.setattr(b2.reference, 'page_count', lambda path:2)
    monkeypatch.setattr(b2.reference, 'docling_version', lambda:'2.81.0')
    def worker(config, paths, gpu, round_id):
        session = b2.Session(config, gpu, 'worker', round_id)
        output = Path(config['mode_out'])/'markdown'
        output.mkdir(exist_ok=True)
        (output/'a.md').write_text('content')
        session.data['documents'] = [dict(row('a.pdf', pages=1), started_seconds=0., finished_seconds=1.)]
        # Simulate the old interrupted checkpoint, bypassing new normalization.
        b2.atomic_json(session.path, session.data)
    monkeypatch.setattr(b2, 'shard_worker', worker)
    argv = ['--pdfs', str(source), '--out', str(tmp_path/'out'), '--mode', 'sequential']
    assert b2.main(argv) == 1
    assert json.loads((tmp_path/'out/sequential/summary.json').read_text())['documents_incomplete'] == 1
    calls = []
    monkeypatch.setattr(b2, 'shard_worker', lambda config,paths,*rest:calls.extend(paths))
    assert b2.main(argv+['--resume']) == 1
    assert calls == [source/'a.pdf']


def test_mask_drift_refused_in_single_mode(monkeypatch,tmp_path):
    source=tmp_path/'pdfs'
    source.mkdir()
    (source/'a.pdf').write_bytes(b'input')
    monkeypatch.setattr(b2.reference,'page_count',lambda p:2)
    monkeypatch.setattr(b2.reference,'docling_version',lambda:'2.81.0')
    monkeypatch.setattr(b2,'shard_worker',lambda *a:None)
    argv=['--pdfs',str(source),'--out',str(tmp_path/'out'),'--mode','sequential','--device','cuda:0']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-A100')
    assert b2.main(argv)==1
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-H100')
    with pytest.raises(SystemExit): b2.main(argv+['--resume'])


def test_identical_paired_times_never_manufacture_saving(monkeypatch,tmp_path):
    config=dict(input_sha256={},expected_pages={'a.pdf':10,'b.pdf':10},fidelity='markdown',use_gpus='single',
        fake_devices=0,devices=['single'],docling_version='2.81.0',device='cpu')
    for mode in ('sequential','batched'):
        out=tmp_path/mode
        b2.atomic_json(out/'receipt.json',config)
        (out/'markdown').mkdir()
        for name in ('a','b'): (out/'markdown'/f'{name}.md').write_text(name)
        docs=[dict(row('a.pdf',pages=10,wall=10,gpu='single'),started_seconds=0.,finished_seconds=10.),
            dict(row('b.pdf',status='ok' if mode=='sequential' else 'timeout',pages=10 if mode=='sequential' else 0,
                wall=10,gpu='single'),started_seconds=10.,finished_seconds=20.)]
        stats=dict(counters=dict(export_recovery_cycles=1,export_worker_respawns=1,recovery_seconds=10),events=[
            dict(kind='export_worker_kills',document='b.pdf',epoch=1,at_seconds=10.,seconds=0.),
            dict(kind='export_worker_ready',document=None,epoch=2,at_seconds=20.,seconds=0.),
            dict(kind='export_worker_respawns',document='b.pdf',epoch=2,at_seconds=20.,seconds=10.)]) if mode=='batched' else dict(counters={},events=[])
        b2.atomic_json(out/'sessions/x.json',dict(id='x',gpu='single',round='r',documents=docs,stats=stats,
            wall_seconds=20.,conversion_started_monotonic=10.,conversion_finished_monotonic=30.,effective_options={}))
    monkeypatch.setattr(b2.reference,'equality',lambda *a:[dict(passes_gate=True)])
    result=b2.publish(tmp_path,[Path('a.pdf'),Path('b.pdf')],config)['batched']
    assert result['break_even']['saving_seconds_per_page']==0
    assert result['per_gpu']['single']['break_even']['saving_seconds_per_page']==0
    assert result['break_even']['cohort_recovery_seconds']==0
    assert result['break_even']['failures_per_document'] is None  # no cost in the paired cohort


def test_paired_recovery_cost_uses_same_interval_attribution():
    docs=[dict(row('a'),started_seconds=0.,finished_seconds=10.),
          dict(row('b','timeout',0),started_seconds=5.,finished_seconds=10.)]
    session=dict(documents=docs,stats=dict(events=[dict(kind='export_worker_respawns',at_seconds=10.,seconds=6.)]))
    seconds, completed=b2.paired_recovery_cost([session],['a'])
    assert seconds==3.5 and completed==1  # 1 alone + 5 split equally
    assert b2.break_even(0,2,completed,seconds)['failures_per_document']==0
    assert b2.paired_recovery_cost([session],[])==(0,0)


def test_append_only_checkpoint_scale_and_timer(monkeypatch,tmp_path):
    clock=[100.]
    monkeypatch.setattr(b2.time,'monotonic',lambda:clock[0])
    names=[f'd{i}.pdf' for i in range(242)]
    session=b2.Session(session_config(tmp_path,names), 'single','worker','one')
    session.save()
    writes=[]
    original=b2.append_jsonl
    def slow_append(path,value):
        original(path,value)
        writes.append(Path(path).stat().st_size)
        clock[0]+=1.
    monkeypatch.setattr(b2,'append_jsonl',slow_append)
    for name in names:
        session.begin()
        clock[0]+=.1
        session.pause()
        session.checkpoint(name,dict(row(name),started_seconds=session.base,finished_seconds=session.data['wall_seconds']))
    assert session.data['wall_seconds']==pytest.approx(24.2)
    assert session.data['observation_seconds']==242.
    assert len(b2.read_jsonl(session.journal))==242
    assert writes==sorted(writes)
    assert session.journal.stat().st_size<300_000


def test_sequential_result_released_before_next_document_and_observation(monkeypatch,tmp_path):
    import weakref
    from glossapi.scripts.docling_cost_benchmark import fake_docling, _FakeStatus
    refs=[]
    class Result:
        status=_FakeStatus.SUCCESS
        pages=[1,2]
        errors=[]
    def convert(path,**kwargs):
        assert kwargs==dict(raises_on_error=False,page_range=(1,2))
        assert all(ref() is None for ref in refs)
        result=Result()
        refs.append(weakref.ref(result))
        return result
    adapter=SimpleNamespace(converter=SimpleNamespace(convert=convert),export=lambda *a:None)
    config=session_config(tmp_path,('a.pdf','b.pdf'))
    session=b2.Session(config,'single','worker','one')
    original=b2.append_jsonl
    def observe(path,value):
        assert all(ref() is None for ref in refs)
        assert session.started is None
        original(path,value)
    monkeypatch.setattr(b2,'append_jsonl',observe)
    with fake_docling('baseline',{}):
        assert b2.sequential_documents([Path('a.pdf'),Path('b.pdf')],adapter,session,tmp_path)==(['a.pdf','b.pdf'],[])
    assert all(ref() is None for ref in refs)


def test_receipt_exception_does_not_change_sequential_outcome(monkeypatch,tmp_path,caplog):
    config=session_config(tmp_path)
    session=b2.Session(config,'single','worker','one')
    monkeypatch.setattr(b2,'append_jsonl',lambda *a:(_ for _ in ()).throw(OSError('disk')))
    monkeypatch.setattr(b2,'sequential_document',lambda *a:dict(row('a.pdf'),wall_seconds=1.))
    assert b2.sequential_documents([Path('a.pdf')],object(),session,tmp_path)==(['a.pdf'],[])
    assert session.data['stats']['counters']['recorder_errors']==1
    assert 'stability receipt failed' in caplog.text


def test_crash_loop_real_signal_worker_resumes_and_preserves_crash(monkeypatch,tmp_path):
    import json, sys
    source=tmp_path/'pdfs'
    source.mkdir()
    (source/'a.pdf').write_bytes(b'input')
    root=tmp_path/'out'
    config=dict(session_config(root),inputs=[str(source/'a.pdf')],devices=['single'],fidelity='markdown',use_gpus='single',
        fake_devices=0,device='cpu',docling_version='2.81.0')
    config['mode_out']=str(root/'sequential')
    b2.atomic_json(root/'inputs.json',config)
    b2.atomic_json(root/'sequential/receipt.json',config)
    worker=tmp_path/'crasher.py'
    worker.write_text('''import os, resource, signal, sys
from pathlib import Path
from glossapi.scripts import batched_stability_benchmark as b
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
root=Path(sys.argv[1]); config=b.json.loads((root/'inputs.json').read_text())
s=b.Session(config,'single','worker','one'); s.save(); s.in_flight(['a.pdf'])
if '--resume' not in sys.argv: os.kill(os.getpid(), signal.SIGABRT)
out=root/'sequential/markdown'; out.mkdir(exist_ok=True); (out/'a.md').write_text('content')
s.begin(); s.pause(); s.data['documents']=[dict(file='a.pdf',status='ok',pages=2,prepared_pages=2,wall_seconds=1.,started_seconds=0.,finished_seconds=1.)]
s.save(complete=True); s.in_flight([])
b.publish(root,[Path(config['inputs'][0])],config)
''')
    calls=[]
    def launch(command):
        calls.append(list(command))
        return b2.launch_process([sys.executable,str(worker),str(root),*command[3:]])
    args=SimpleNamespace(out=str(root),crash_loop=1)
    assert b2.crash_loop(args,['--out',str(root),'--crash-loop','1'],launch)==0
    assert len(calls)==2 and '--resume' in calls[1] and '--crash-loop' not in calls[1]
    summary=json.loads((root/'sequential/summary.json').read_text())
    assert summary['documents_ok']==1 and summary['process_crash_count']==1
    assert summary['process_crashes'][0]['signal']==6
    assert summary['process_crashes'][0]['document_in_flight']=='a.pdf'
    assert summary['worker_wall_seconds']>0


def test_crash_loop_restart_limit_and_ordinary_failure(tmp_path):
    args=SimpleNamespace(out=str(tmp_path),crash_loop=2)
    calls=[]
    assert b2.crash_loop(args,['--crash-loop=2'],lambda cmd:calls.append(cmd[:]) or -11)==139
    assert len(calls)==3
    assert len(b2.read_jsonl(tmp_path/'process_crashes.jsonl'))==3
    calls.clear()
    assert b2.crash_loop(args,[],lambda cmd:calls.append(cmd[:]) or 1)==1
    assert len(calls)==1


def test_sequential_does_not_install_layout_wrapper(monkeypatch):
    calls=[]
    def adapter(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(per_page_layout_timings=lambda:pytest.fail('production models must remain unwrapped'))
    monkeypatch.setattr(b2.core,'DoclingAdapter',adapter)
    b2.make_adapter(dict(mode='sequential',layout_batch=16,document_timeout=600,artifacts_path=None),'cuda:0')
    assert calls[0]['layout_batch']==4


def test_receipt_preparation_error_is_isolated(monkeypatch,tmp_path,caplog):
    session=b2.Session(session_config(tmp_path),'single','worker','one')
    def broken(*args): raise OSError('normalizer failed')
    monkeypatch.setattr(b2,'normalize_document',broken)
    session.checkpoint('a.pdf',row('a.pdf'))
    assert session.data['stats']['counters']['recorder_errors']==1
    assert 'stability receipt failed' in caplog.text


def test_group_crash_is_one_event_with_all_in_flight_documents(tmp_path):
    for gpu,name in [('0','a.pdf'),('1','b.pdf')]:
        config=session_config(tmp_path,(name,),'batched')
        session=b2.Session(config,gpu,'worker','one')
        session.save()
        session.in_flight([name])
    b2.record_process_crash(tmp_path,11)
    events=b2.read_jsonl(tmp_path/'process_crashes.jsonl')
    assert len(events)==1
    assert sorted(events[0]['documents_in_flight'])==['a.pdf','b.pdf']
    assert events[0]['document_in_flight'] is None


def test_checkpoint_history_scale_qualification(tmp_path,record_property):
    import time, json
    names=[f'd{i}.pdf' for i in range(242)]
    session=b2.Session(session_config(tmp_path,names,'batched'),'single','worker','one')
    session.save()
    elapsed=0.
    before=time.perf_counter()
    max_checkpoint=0.
    for index,name in enumerate(names):
        events=[b2.core.StabilityEvent('queue_stalls',None,1,index*4+j*.01,seconds=.01,
            affected_documents=tuple(names[index:index+16])) for j in range(5)]
        doc=dict(row(name,wall=4.),started_seconds=index*4.,finished_seconds=(index+1)*4.)
        # Emulate primitive completion data; no converter or model is involved.
        session.begin()
        checkpoint_started=time.perf_counter()
        session.checkpoint(name,doc,(index+1)*4.,dict(counters={'queue_stalls':(index+1)*5},events=events))
        max_checkpoint=max(max_checkpoint,time.perf_counter()-checkpoint_started)
        session.pause()
    elapsed=time.perf_counter()-before
    data=dict(documents=242,events=1210,bytes=session.journal.stat().st_size,
        total_checkpoint_seconds=elapsed,max_checkpoint_seconds=max_checkpoint)
    assert len(b2.read_jsonl(session.journal))==242
    assert data['bytes']<2_000_000
    record_property('incremental_checkpoint_qualification',json.dumps(data,sort_keys=True))


def test_crash_preserves_unfinished_round_elapsed(monkeypatch,tmp_path):
    monkeypatch.setattr(b2.time,'monotonic',lambda:15.)
    b2.atomic_json(tmp_path/'sequential/rounds/one.json',dict(started_monotonic=10.,complete=False,
        total_seconds_including_initialization=0.))
    b2.record_process_crash(tmp_path,11)
    data=b2.json.loads((tmp_path/'sequential/rounds/one.json').read_text())
    assert data['complete'] and data['process_signal']==11
    assert data['total_seconds_including_initialization']==5.


def test_sequential_native_timeout_is_timeout(tmp_path):
    from glossapi.scripts.docling_cost_benchmark import fake_docling
    adapter=SimpleNamespace(converter=SimpleNamespace(convert=lambda *a,**kw:(_ for _ in ()).throw(TimeoutError('native'))),
        failure=lambda *a:None)
    with fake_docling('baseline',{}):
        result=b2.sequential_document(Path('a.pdf'),adapter,session_config(tmp_path),tmp_path)
    assert result['status']=='timeout' and result['pages']==0


def test_interrupted_batched_journal_excludes_flight_observation(tmp_path,monkeypatch):
    clock=[100.]
    monkeypatch.setattr(b2.time,'monotonic',lambda:clock[0])
    cfg=session_config(tmp_path,['a.pdf','b.pdf'],'batched')
    cfg['expected_pages']={'a.pdf':1,'b.pdf':1}
    session=b2.Session(cfg,'single','worker','one')
    session.save()
    original=b2.atomic_json
    def slow_flight(path,value):
        original(path,value)
        if str(path).endswith('.flight.json'): clock[0]+=1.
    monkeypatch.setattr(b2,'atomic_json',slow_flight)
    session.begin()
    session.in_flight(['a.pdf','b.pdf'])  # one second of excluded owner observation
    clock[0]+=3.
    session.checkpoint('a.pdf',dict(status='ok',pages=1,started_seconds=0.,finished_seconds=3.),3.)
    session.in_flight(['b.pdf'])  # same callback follows the terminal append
    clock[0]+=3.
    session.checkpoint('b.pdf',dict(status='ok',pages=1,started_seconds=3.,finished_seconds=6.),6.)
    # Simulate death before the final full Session.save(), so the resume path
    # reconstructs solely from the append-only journal, as a real signal does.
    out=Path(cfg['mode_out']); (out/'markdown').mkdir()
    for n in ('a','b'): (out/'markdown'/f'{n}.md').write_text('content')
    rows,sessions,wall=b2.collect(out,[Path('a.pdf'),Path('b.pdf')],cfg['expected_pages'])
    assert sessions[0]['wall_seconds']==6. and wall==6.
    assert session.data['observation_seconds']==2.
    summary=b2.summarize(rows,wall,sessions,['a.pdf','b.pdf'])
    assert summary['common_seconds_per_page']==3. and summary['common_worker_seconds_per_page']==3.
    assert sessions[0]['conversion_intervals']==[[101.,104.],[105.,108.]]
    session.pause(excluded_intervals=session.observation_intervals)
    session.data['documents']=[record['document'] for record in b2.read_jsonl(session.journal)]
    session.save(complete=True)
    assert b2.collect(out,[Path('a.pdf'),Path('b.pdf')],cfg['expected_pages'])[2]==wall


@pytest.mark.parametrize("automatic", [False, True])
def test_native_queue_real_signal_recorded_and_resume_idempotent(tmp_path,monkeypatch,automatic):
    import queue, subprocess, sys, signal, json
    import multiprocessing
    from glossapi import Corpus
    from glossapi.corpus import phase_extract as phase
    made=[]
    flights=[]
    class State:
        def __init__(self,*a,**kw): pass
        def load(self): return set(),set()
        def save(self,*a): pass
    class Process:
        def __init__(self,target,args,kwargs):
            self.args=args; self.kwargs=kwargs; self.pid=30000+len(made); self.exitcode=-6 if not made else 0; made.append(self)
        def start(self):
            while True:
                try: items=self.args[5].get_nowait()
                except queue.Empty: break
                cfg=self.kwargs['stability_benchmark']
                session=b2.Session(cfg,'0','gpu0-w0',cfg['round'])
                session.data['assigned_documents']=[Path(p).name for p in items]
                session.save()
                if self.exitcode==-6:
                    self.args[18][self.args[2]]=items
                    session.in_flight([Path(p).name for p in items])
                    child = subprocess.Popen([sys.executable, '-c', 'import os, signal, resource; resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); os.kill(os.getpid(), signal.SIGABRT)'])
                    self.pid=child.pid
                    self.exitcode=child.wait()
                    flight=b2.json.loads(session.flight.read_text())
                    flight['pid']=self.pid
                    b2.atomic_json(session.flight,flight)
                    flights.append(session.flight)
                    return
                out=Path(cfg['mode_out'])/'markdown'; out.mkdir(exist_ok=True)
                for p in items: (out/f'{Path(p).stem}.md').write_text('content')
                session.data.update(effective_options={},wall_seconds=1.,conversion_started_monotonic=10.,conversion_finished_monotonic=11.,conversion_intervals=[[10.,11.]],documents=[dict(file=Path(p).name,status='ok',pages=1,started_seconds=0.,finished_seconds=1.,attributed_wall_seconds=1.) for p in items])
                session.save(complete=True)
                self.args[17].put(dict(event='batch',processed=items,problematic=[]))
        def is_alive(self): return False
        def join(self,**kw): pass
    context=SimpleNamespace(Manager=lambda:SimpleNamespace(dict=dict,shutdown=lambda:None),Queue=queue.Queue,Process=Process)
    monkeypatch.setattr(multiprocessing,'get_context',lambda method:context)
    monkeypatch.setattr(phase,'_ProcessingStateManager',State)
    monkeypatch.setattr(phase._SkiplistManager,'load',lambda self:set())
    monkeypatch.setattr(Corpus,'_finalize_font_repair_metadata',lambda *a:None)
    monkeypatch.setattr(phase,'_estimate_extract_work_pages',lambda p:1)
    monkeypatch.setattr(b2.reference,'page_count',lambda p:1)
    monkeypatch.setattr(b2.reference,'docling_version',lambda:'2.81.0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    source=tmp_path/'pdfs/a.pdf'; source.parent.mkdir(); source.write_bytes(b'input')
    root=tmp_path/'out'
    argv=['--pdfs',str(source.parent),'--out',str(root),'--mode','sequential','--use-gpus','multi','--device','cuda']
    launches=[]
    def launch(command): launches.append(command[:]); return b2.main(command[3:])
    first=b2.crash_loop(SimpleNamespace(out=str(root),crash_loop=1),argv+['--crash-loop','1'],launch) if automatic else b2.main(argv)
    assert first==(0 if automatic else 134)
    assert len(launches)==(2 if automatic else 0) and len(made)==(3 if automatic else 2)
    assert made[0].exitcode==-signal.SIGABRT
    initial=json.loads((root/'sequential/summary.json').read_text())
    assert initial['process_crash_count']==1 and initial['worker_wall_seconds']>0.
    assert initial['process_crashes'][0]['signal']==signal.SIGABRT
    assert initial['per_gpu']['0']['process_crash_count']==1
    assert initial['process_crashes'][0]['gpu']=='0'
    assert initial['process_crashes'][0]['document_in_flight']=='a.pdf'
    assert not flights[0].exists() and len(b2.read_jsonl(root/'process_crashes.jsonl'))==1
    assert b2.main(argv+['--resume'])==0
    resumed=json.loads((root/'sequential/summary.json').read_text())
    assert resumed['documents_ok']==1 and resumed['process_crash_count']==1
    assert resumed['per_gpu']['0']['process_crash_count']==1
    before=resumed['worker_wall_seconds']
    assert b2.main(argv+['--resume'])==0
    assert b2.json.loads((root/'sequential/summary.json').read_text())['worker_wall_seconds']==before
    assert resumed['timing_censored']


def test_paired_real_signal_resume_keeps_fidelity(tmp_path):
    source=tmp_path/'pdfs'; source.mkdir()
    for n in ('a.pdf','b.pdf'): (source/n).write_bytes(b'input')
    out=tmp_path/'out'
    import subprocess, sys, json
    driver=Path(__file__).with_name('b2_fake_crash_driver.py')
    command=[sys.executable,str(driver),'--pdfs',str(source),'--out',str(out),'--mode','both','--crash-loop','1']
    first=subprocess.run(command,capture_output=True,text=True,timeout=20)
    # Every complete output remains verifiable after the interrupted attempt.
    assert first.returncode==0,first.stderr
    summary=json.loads((out/'batched/summary.json').read_text())
    seq=json.loads((out/'sequential/summary.json').read_text())
    assert seq['documents_ok']==summary['documents_ok']==2
    assert all(r['passes_gate'] for r in summary['fidelity_checks'])
    assert summary['comparable_options_verified'] is True and summary['fidelity_verified'] is True
    assert summary['fidelity_excluded_attempts']
    assert seq['process_crash_count']==1 and summary['process_crash_count']==0
    crash=seq['process_crashes'][0]
    assert crash['signal']==6 and crash['mode']=='sequential' and crash['document_in_flight']=='b.pdf'
    assert seq['timing_censored'] and seq['worker_wall_seconds']>0
    count_before=len(b2.read_jsonl(out/'conversions.jsonl'))
    second=subprocess.run(command+['--resume'],capture_output=True,text=True,timeout=20)
    assert second.returncode==0 and len(b2.read_jsonl(out/'conversions.jsonl'))==count_before
    again=json.loads((out/'sequential/summary.json').read_text())
    assert again['process_crash_count']==1 and again['worker_wall_seconds']==seq['worker_wall_seconds']
    assert len(b2.read_jsonl(out/'process_crashes.jsonl'))==1
    launches=b2.read_jsonl(out/'launches.jsonl')
    assert '--resume' not in launches[0]['command'] and '--resume' in launches[1]['command']


def test_interrupted_journal_excludes_entire_scheduler_callbacks(monkeypatch, tmp_path):
    clock = [100.]
    monkeypatch.setattr(b2.time, 'monotonic', lambda: clock[0])
    cfg = session_config(tmp_path, ['a.pdf', 'b.pdf'], 'batched')
    cfg['expected_pages'] = {'a.pdf': 1, 'b.pdf': 1}
    session = b2.Session(cfg, 'single', 'worker', 'one')
    session.save()
    brackets = []
    origin = [100.]
    session.observation_source = lambda: brackets
    session.current_observation_started = lambda elapsed: origin[0] + elapsed
    append = b2.append_jsonl
    def slow_append(path, value):
        append(path, value)
        clock[0] += .5
    monkeypatch.setattr(b2, 'append_jsonl', slow_append)
    session.begin()
    session.in_flight(['a.pdf', 'b.pdf'])
    clock[0] += 1.  # Whole progress-only callback, including non-IO work.
    brackets.append([100., 101.])
    origin[0] += 1.
    for index, name in enumerate(('a.pdf', 'b.pdf')):
        clock[0] += 3.
        began = clock[0]
        clock[0] += .2  # Recorder preparation inside the callback.
        session.checkpoint(name, dict(status='ok', pages=1,
            started_seconds=index*3., finished_seconds=(index+1)*3.), (index+1)*3.)
        session.in_flight(['b.pdf'] if index == 0 else [])
        clock[0] += .3  # Callback tail, also outside Session.observe's IO.
        brackets.append([began, clock[0]])
        origin[0] += 1.
    records = b2.read_jsonl(session.journal)
    intervals = [interval for record in records for interval in record['conversion_intervals']]
    assert intervals == [[101., 104.], [105., 108.]]
    assert records[-1]['wall_seconds'] == 6.
    session.pause(excluded_intervals=brackets)
    assert session.data['wall_seconds'] == records[-1]['wall_seconds']


@pytest.mark.parametrize('options', [{'do_ocr': False}, None, {'do_ocr': True}])
def test_fidelity_uses_complete_output_options_and_records_interrupted_exclusions(monkeypatch, tmp_path, options):
    config = dict(input_sha256={}, expected_pages={'a.pdf': 2, 'b.pdf': 2},
        fidelity='markdown', use_gpus='single', fake_devices=0,
        devices=['single'], docling_version='2.81.0', device='cpu')
    for mode in ('sequential', 'batched'):
        out = tmp_path / mode
        b2.atomic_json(out / 'receipt.json', config)
        (out / 'markdown').mkdir()
        (out / 'markdown/a.md').write_text('identical complete output')
        b2.atomic_json(out / 'sessions/ok.json', dict(id='ok', gpu='single', round='one',
            created_at_unix=1., documents=[dict(row('a.pdf', gpu='single'),
                started_seconds=0., finished_seconds=1.)], stats=dict(counters={}, events=[]),
            wall_seconds=1., conversion_started_monotonic=10., conversion_finished_monotonic=11.,
            complete=True, effective_options={'do_ocr':False} if mode == 'sequential' else options))
        b2.atomic_json(out / 'sessions/crash.json', dict(id='crash', gpu='single', round='one',
            created_at_unix=2., documents=[dict(row('b.pdf', 'incomplete', 0, gpu='single'),
                error='process signal 6', started_seconds=0., finished_seconds=1.)],
            stats=dict(counters={}, events=[]), wall_seconds=1., timing_censored=True,
            conversion_started_monotonic=11., conversion_finished_monotonic=12., complete=True))
    compared = []
    def equality(root, names, fidelity):
        compared.extend(names)
        return [dict(file=name, passes_gate=True) for name in names]
    monkeypatch.setattr(b2.reference, 'equality', equality)
    summary = b2.publish(tmp_path, [Path('a.pdf'), Path('b.pdf')], config)['batched']
    assert compared == ['a.pdf']
    assert summary['fidelity_verified'] is (options == {'do_ocr':False})
    assert len(summary['fidelity_excluded_documents']) == 2
    assert all(d['file'] == 'b.pdf' and d['reason'] == 'process signal 6'
        for d in summary['fidelity_excluded_documents'])
    assert len(summary['fidelity_excluded_attempts']) == 2
    assert all('interrupted' in s['reason'] for s in summary['fidelity_excluded_attempts'])
