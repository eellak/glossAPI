"""K4g fix round 2 (CP-K4gc2, cycle 2): the reviewer's targeted probes, ported from review_scratch/CP-K4gc2/test_targeted.py.
R1/R2: recovery and dispatch share one read-only plan; R3/R4: url_index numeric contract in plan and run."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glossapi.input_manifest import ManifestError
from glossapi.stages.run import row_resource_index

HERE = Path(__file__).resolve().parent


def _record(name, data):
    """Receipts stay with the test run (the reviewer wrote them beside the probe)."""
    return data


@pytest.mark.parametrize('value', [np.int64(2**53+1), np.uint64(2**63-1), pd.Series([2**53+1], dtype='Int64').iloc[0], 1.0, np.float64(2.0), -0.0])
def test_f3_integer_scalars_stay_exact_in_both_entries(tmp_path, value):
    from test_k4g_b1 import _plan_keys, _run_keys
    assert row_resource_index({'url_index': value}) == int(value)
    docs = [dict(url='https://fixture.invalid/a.pdf', source_id='S', url_index=value)]
    assert _plan_keys(tmp_path, docs)[0] == _run_keys(tmp_path, docs)


@pytest.mark.parametrize('value', [True, np.bool_(True), '1', '1.0', '9007199254740993', '', -1, np.int64(-1), np.uint64(2**63), float('inf'), float('-inf'), 2**63])
def test_f3_bad_values_refused_in_both_entries(tmp_path, value):
    from test_k4g_b1 import _plan_keys, _run_keys
    docs = [dict(url='https://fixture.invalid/a.pdf', source_id='S', url_index=value)]
    for entry in (_plan_keys, _run_keys):
        with pytest.raises(ManifestError) as caught:
            entry(tmp_path, docs)
        assert caught.value.code == 'invalid_manifest'
        assert caught.value.details['field'] == 'url_index'


@pytest.mark.parametrize('value', [float('nan'), np.float32('nan'), np.float64('nan')])
def test_f3_nan_is_missing_in_both_entries(tmp_path, value):
    from test_k4g_b1 import _plan_keys, _run_keys
    docs = [dict(url='https://fixture.invalid/a.pdf', source_id='S', url_index=value)]
    assert row_resource_index({'url_index': value}) == 0
    assert _plan_keys(tmp_path, docs)[0] == _run_keys(tmp_path, docs)


@pytest.mark.parametrize('value', [Fraction(2**54+1, 2), Fraction(-1, 10**400), Fraction(10**400, 1)], ids=['rounded-fraction','negative-underflow','huge-real'])
def test_f3_non_integral_or_oversized_real_is_refused_typed(value):
    with pytest.raises(ManifestError) as caught:
        row_resource_index({'url_index': value})
    assert caught.value.code == 'invalid_manifest'


from test_ledger_scheduler import Producer, Consumer

class IntermediateProducer(Producer):
    def run_document(self, request, *, reporter):
        result = super().run_document(request, reporter=reporter)
        return replace(result, outputs=tuple(replace(item, retention_class='intermediate') for item in result.outputs))

class OcrConsumer(Consumer):
    name = 'ocr'

class SelectiveProducer(Producer):
    def run_document(self, request, *, reporter):
        result = super().run_document(request, reporter=reporter)
        if request.settings.get('intermediate'):
            result = replace(result, outputs=tuple(replace(item, retention_class='intermediate') for item in result.outputs))
        return result


def test_f2_retired_edge_can_be_regenerated_by_its_invalidated_producer(tmp_path, monkeypatch):
    from test_ledger_scheduler import graph
    from glossapi.ledger import LedgerWriter
    from glossapi.stages.scheduler import StreamingScheduler
    from glossapi.stages.retention import RetentionController, release_private_scratch
    root = tmp_path
    with LedgerWriter(root/'corpus.sqlite3') as writer:
        operations = graph(writer, root, 1)
        registry = {'extract': IntermediateProducer, 'clean': Consumer}
        engine = StreamingScheduler(writer, root, registry=registry)
        run = engine.freeze(operations)
        first = engine.execute(run)
        assert first['state'] == 'completed'
        producer = first['attempts']['produce:0']
        consumer = first['attempts']['consume:0']
        output = next(a for a in engine._outputs(producer, include_deleted=True) if a.kind == 'text')
        if output.path.exists():
            row = writer.connection.execute('SELECT * FROM current_file_state WHERE file_id=?', (output.file_id,)).fetchone()
            release_private_scratch(RetentionController(writer,root,run), [row])
        assert writer.connection.execute('SELECT presence FROM current_file_state WHERE file_id=?', (output.file_id,)).fetchone()[0] == 'deleted'
        # Both outputs need recovery; the original external source is still intact.
        receipt = next(a for a in engine._outputs(producer, include_deleted=True) if a.kind == 'worker_receipt')
        receipt.path.unlink()
        next(a for a in engine._outputs(consumer) if a.kind == 'text').path.unlink()
        assert operations[0].document.artifacts[0].path.is_file()
        fresh = StreamingScheduler(writer,root,registry=registry)
        try:
            actual = fresh.execute(run, resume=True)
        except Exception as error:
            actual = dict(error_code=getattr(error,'code',None),error_type=type(error).__name__,message=str(error))
        data = dict(actual=actual,source_exists=operations[0].document.artifacts[0].path.is_file())
        _record('f2-regenerable.json', data)
        assert actual.get('state') == 'completed', data


@pytest.mark.parametrize('debug', ['off', 'on'])
def test_f1_extreme_partial_retry_and_run_end(tmp_path, debug):
    from test_k4g_fix1 import child
    options = dict(PROBE_EXTREME='1',PROBE_REMOTE='1',PROBE_DEBUG=debug,PROBE_ANNOTATE='1',PROBE_PARAGRAPHS='80')
    runs = [child(tmp_path/'run','start','beta,gamma',**options),
            child(tmp_path/'run','retry-partial','gamma',**options),
            child(tmp_path/'run','retry-complete','',**options)]
    for data in runs:
        for index, entry in enumerate(data['measured']):
            later=max(item['after'] for item in data['measured'][index:])
            assert entry['need'] >= later-entry['before'], (entry,later-entry['before'])
        if debug == 'on' or data['state'] == 'completed_with_failures':
            assert data['credits'] and all(call['credits'] == {} for call in data['credits'])
    assert runs[-1]['state']=='completed' and len(runs[-1]['records'])==3
    if debug == 'off':
        assert runs[-1]['private_md'] == []
    else:
        assert runs[-1]['private_md']
    _record('f1-retry', runs)


@pytest.mark.parametrize('stage,aggregate', [('ocr',False),('sections',True)])
def test_f2_direct_retired_input_shapes_plan_read_only(tmp_path, stage, aggregate):
    from types import SimpleNamespace
    from glossapi.ledger import LedgerWriter
    from glossapi.stages.contracts import DocumentSnapshot
    from glossapi.stages.scheduler import StreamingScheduler
    from glossapi.stages.retention import PublicationInputsReleasedError, RetentionController, release_private_scratch
    from glossapi.stages.execution import sha256_file
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        input_path=tmp_path/'.glossapi/artifacts/input.txt'
        input_path.parent.mkdir(parents=True)
        input_path.write_text('retired reader input')
        output=tmp_path/'.glossapi/artifacts/reader.txt'
        output.write_text('committed output')
        with writer.transaction():
            run=writer.run({}, {}, {}, code_revision='probe')
            document=writer.document('file:///fixture.pdf')
            version=writer.version(document)
            producer_task=writer.task(run,'extract','extract.probe',{},version_id=version)
            producer_attempt=writer.attempt(producer_task,worker_id='probe',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
            writer.finish(run,producer_task,producer_attempt,status='ok',reason_code='extract_completed',outputs=[dict(kind='raw_markdown',uri=input_path.as_uri(),sha256=sha256_file(input_path),bytes=input_path.stat().st_size,retention_class='intermediate')])
            input_id=writer.connection.execute('SELECT file_id FROM files WHERE producer_attempt_id=?',(producer_attempt,)).fetchone()[0]
            task=writer.task(run,stage,stage+'.probe',{},version_id=None if aggregate else version,
                             scope_key='corpus:'+writer.snapshot('inputs',{'files':[input_id]}) if aggregate else None)
            attempt=writer.attempt(task,worker_id='probe',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
            writer.input(attempt,input_id,'selected_text',sha256_file(input_path))
            writer.finish(run,task,attempt,status='ok',reason_code=stage+'_completed',outputs=[dict(kind='text',uri=output.as_uri(),sha256=sha256_file(output),bytes=output.stat().st_size)])
        row=writer.connection.execute('SELECT * FROM current_file_state WHERE file_id=?',(input_id,)).fetchone()
        release_private_scratch(RetentionController(writer,tmp_path,run),[row])
        assert not input_path.exists()
        output.unlink()
        operation=SimpleNamespace(key='reader',stage=stage,document=DocumentSnapshot(document,None if aggregate else version,'file:///fixture.pdf'))
        engine=StreamingScheduler(writer,tmp_path)
        before=writer.connection.total_changes
        operation.inputs=()
        plan=engine._recovery_plan(run,{'reader':task},[operation])
        # The plan is read only. This reader has no planned producer edge (its input was recorded only as an attempt
        # input), so recovery plans to run it again and nothing is refused; planned edges are judged by
        # test_f2_transitively_invalidated_ocr_reader_is_checked_before_writes and the public cases.
        assert writer.connection.total_changes==before
        assert plan['executing']=={'reader'} and plan['unsatisfied']==[] and plan['reusable']=={}


def test_f2_transitively_invalidated_ocr_reader_is_checked_before_writes(tmp_path):
    from test_ledger_scheduler import graph
    from glossapi.ledger import LedgerWriter
    from glossapi.stages.scheduler import StreamingScheduler
    from glossapi.stages.retention import RetentionController, release_private_scratch
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        first, reader, other, _ = graph(writer,tmp_path,2)
        # An inputless producer models an acquisition or independent producer: its retired output is a valid reuse.
        other=replace(other,document=replace(other.document,version_id=first.document.version_id,
                        document_id=first.document.document_id,artifacts=()),
                      specification=replace(other.specification,operation='extract.independent',settings={'intermediate':True}))
        # Same scheduler edge shape as OCR (source + native text + page quality), without invoking an OCR model.
        reader=replace(reader,stage='ocr',specification=replace(reader.specification,operation='ocr.probe'),
                       inputs=((first.key,'text','native_markdown'),(other.key,'text','source_payload')))
        operations=[first,other,reader]
        registry={'extract':SelectiveProducer,'ocr':OcrConsumer}
        engine=StreamingScheduler(writer,tmp_path,registry=registry,workers={'ocr':1})
        run=engine.freeze(operations)
        initial=engine.execute(run)
        assert initial['state']=='completed'
        retired=next(a for a in engine._outputs(initial['attempts'][other.key],include_deleted=True) if a.kind=='text')
        row=writer.connection.execute('SELECT * FROM current_file_state WHERE file_id=?',(retired.file_id,)).fetchone()
        if retired.path.exists():
            release_private_scratch(RetentionController(writer,tmp_path,run),[row])
        assert not retired.path.exists()
        next(a for a in engine._outputs(initial['attempts'][first.key]) if a.kind=='text').path.unlink()
        fresh=StreamingScheduler(writer,tmp_path,registry=registry,workers={'ocr':1})
        before=writer.connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
        try:
            actual=fresh.execute(run,resume=True,retry_failed=True)
        except Exception as error:
            actual=dict(error_code=getattr(error,'code',None),error_type=type(error).__name__,message=str(error))
        after=writer.connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
        reasons=[list(r) for r in writer.connection.execute(f"SELECT event_kind,reason_code,status FROM {writer.connection.event_table} WHERE reason_code IN ('input_hash_changed','output_hash_mismatch')")]
        data=dict(actual=actual,attempts_before=before,attempts_after=after,reasons=reasons)
        _record('f2-transitive.json', data)
        assert actual.get('error_code')=='republish_inputs_released' and after==before, data


def test_f2_public_resume_can_rebuild_a_retired_edge_from_local_source(tmp_path,monkeypatch):
    import test_k4g_fix1 as original
    root=tmp_path/'public'
    started=original.child(root,'start')
    assert started['state']=='completed' and not started['private_md']
    receipts=list((root/'output/.glossapi/artifacts').rglob('worker-result.json'))
    removed=[]
    for path in receipts:
        if any(item['relative_path'].endswith('/alpha.md') for item in json.loads(path.read_text())['outputs']):
            path.unlink(); removed.append(str(path))
    assert len(removed)==2 and (root/'alpha.pdf').is_file()
    actual=original.child(root,'retry-broken',allow_failure=True)
    data=dict(actual=actual,removed=removed,source_exists=(root/'alpha.pdf').is_file())
    assert actual.get('state')=='completed' and sorted(actual['records'])==['alpha','beta','gamma'],data
    _record('f2-public-regenerable.json', data)
    assert actual.get('state')=='completed',data


def test_f2_plain_successful_ocr_resume_is_completed(tmp_path):
    import os
    import subprocess
    import sys
    from test_ocr_same_basename import STUB,SCAN
    from glossapi.ledger.writer import open_readonly
    from glossapi.stages.scheduler import local_path
    root=tmp_path/'public-ocr'
    root.mkdir()
    (root/'stub').mkdir()
    (root/'stub/sitecustomize.py').write_text(STUB)
    source=root/'scan.pdf';source.write_bytes(SCAN.read_bytes())
    env={**os.environ,'GLOSSAPI_TEST_STUB_OCR':'1','PYTHONDONTWRITEBYTECODE':'1','GLOSSAPI_CONVERSION_LOG_FILE':str(root/'conversion.log'),
         'PYTHONPATH':os.pathsep.join([str(root/'stub'),os.environ.get('PYTHONPATH',''),str(HERE.parent/'src'),str(HERE)])}
    def child(mode):
        done=subprocess.run([sys.executable,'-B',str(HERE/'k4g_fix2_ocr_child.py'),str(root),mode,str(source)],env=env,cwd=root,text=True,capture_output=True,timeout=60)
        (root/(mode+'.stdout')).write_text(done.stdout)
        (root/(mode+'.stderr')).write_text(done.stderr)
        assert done.returncode==0,done.stderr[-3000:]
        return json.loads(done.stdout.splitlines()[-1])
    start=child('start')
    assert start['state']=='completed'
    import hashlib
    def products():
        out=root/'output'
        files=[out/'export.jsonl',*sorted((out/'clean_markdown').glob('*.md')),*sorted((out/'markdown').glob('*.md'))]
        return {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    before=products()
    plain=child('plain')
    assert products()==before  # a finished run resumed with nothing changed keeps identical products
    _record('f2-public-ocr-plain.json', dict(start=start,plain=plain))
    assert plain.get('state')=='completed' and plain['attempts']==start['attempts'] and not plain['errors'],plain


def test_f2_guard_precedes_pending_retention_recovery_writes(tmp_path):
    from test_ledger_scheduler import graph
    from glossapi.ledger import LedgerWriter
    from glossapi.stages.scheduler import StreamingScheduler
    from glossapi.stages.retention import RetentionController,release_private_scratch,PublicationInputsReleasedError
    from glossapi.stages.execution import sha256_file
    registry={'extract':IntermediateProducer,'clean':Consumer}
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine=StreamingScheduler(writer,tmp_path,registry=registry)
        run=engine.freeze(graph(writer,tmp_path,1))
        first=engine.execute(run)
        producer=first['attempts']['produce:0']
        retired=next(a for a in engine._outputs(producer,include_deleted=True) if a.kind=='text')
        if retired.path.exists():
            row=writer.connection.execute('SELECT * FROM current_file_state WHERE file_id=?',(retired.file_id,)).fetchone()
            release_private_scratch(RetentionController(writer,tmp_path,run),[row])
        assert not retired.path.exists()
        next(a for a in engine._outputs(first['attempts']['consume:0']) if a.kind=='text').path.unlink()
        pending=retired.path.parent/'late-diagnostic.txt';pending.write_text('pending private release')
        with writer.transaction():
            fid=writer.artifact(run,'extract','attempt_diagnostic',pending.as_uri(),version_id=writer.control()['tasks'][writer.control()['attempts'][producer]['task_id']]['version_id'],
                                producer_attempt_id=producer,sha256=sha256_file(pending),bytes=pending.stat().st_size)
        row=writer.connection.execute('SELECT * FROM current_file_state WHERE file_id=?',(fid,)).fetchone()
        def crash(point,_):
            if point=='after_intent':raise RuntimeError('pending release')
        with pytest.raises(RuntimeError,match='pending release'):
            release_private_scratch(RetentionController(writer,tmp_path,run,fault=crash),[row])
        assert pending.exists()
        before=writer.connection.total_changes
        fresh=StreamingScheduler(writer,tmp_path,registry=registry)
        with pytest.raises(PublicationInputsReleasedError):
            fresh.execute(run,resume=True)
        after=writer.connection.total_changes
        data=dict(changes_before=before,changes_after=after,pending_file_exists=pending.exists(),error_code='republish_inputs_released')
        _record('f2-recovery-order.json', data)
        assert after==before and pending.exists(),data
