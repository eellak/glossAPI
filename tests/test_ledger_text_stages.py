"""Task-four producer evidence, isolation and deterministic plugin/ROI contracts."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pandas as pd
import pytest

from glossapi.stages.contracts import ArtifactReference, StageRequest, StageSpecification
from glossapi.stages.clean import CleanStage
from glossapi.stages.ocr import OcrStage
from glossapi.stages.math import MathStage
from glossapi.stages.edits import EditRecorder, replay_edits, text_hash
from glossapi.stages.execution import ReceiptCollector, sha256_file


def stage_request(root, operation, inputs, **settings):
    return StageRequest(*(str(uuid.uuid4()) for _ in range(5)),operation,
        tuple(ArtifactReference(str(uuid.uuid4()),kind,Path(path),sha256_file(path),Path(path).stat().st_size) for kind,path in inputs),
        {'debug':'on', **settings},{},'cpu',root/'attempt',time.monotonic_ns()+60_000_000_000)


def run_clean(root,text,operation='clean.initial',**settings):
    root.mkdir(parents=True,exist_ok=True)
    source=root/'source.md';source.write_text(text)
    request=stage_request(root,operation,[('selected_text',source)],**settings)
    return source,request,CleanStage().run_document(request,reporter=ReceiptCollector())


def test_stage_metadata_imports_no_optional_backends():
    subprocess.run([sys.executable,'-c',"import sys;from glossapi.stages.clean import CleanStage;from glossapi.stages.ocr import OcrStage;from glossapi.stages.math import MathStage;assert not any(k in sys.modules for k in ('pandas','numpy','torch','docling','pypdfium2','glossapi_rs_cleaner'))"],check=True)
    assert OcrStage().packages(StageSpecification('ocr.document'))[1].entry_point_group=='docling'
    assert MathStage().packages(StageSpecification('math.triage'))==()


@pytest.mark.parametrize('text',[
    'Καλημέρα &amp; κόσμε\r\n\r\nabc\x00def',
    'foo /space bar\n\n\n\nΚαλημέρα\u00a0κόσμε',
    '/uni0301'*14+'\nκαλό κείμενο\n',
    '```\nabc\x00def\n```\n\n:::::::\n',
    '| α | β |\n| --- | --- |\n| 1 | 2 |\n',
    '',
])
def test_native_edits_replay_exactly_before_cleanup(tmp_path,text):
    source,request,result=run_clean(tmp_path,text)
    assert result.status=='ok',result.error
    assert source.read_text()==text.replace('\r\n','\n')
    decision=result.decisions[0]
    replacement=json.loads((request.work_directory/'edit-replacements.json').read_text())
    output=(request.work_directory/'clean_markdown/source.md').read_text()
    assert replay_edits(source.read_bytes().decode(),decision['edits'],replacement)==output
    assert (request.work_directory/'native_evidence/native_stages/stage2_detailed_table_report.csv').exists()
    assert result.metrics['execution']=='in_process'
    assert all('initial_range' in edit for edit in decision['edits']['edits'])


def test_edit_composition_and_tampered_replacement_rejected():
    recorder=EditRecorder('αβγδε')
    recorder.edit('replace',1,3,'XYZ')
    recorder.edit('delete',2,4,'')
    assert recorder.text=='αXδε'
    assert recorder.receipt()['edits'][1]['initial_range']==[1,3]
    assert replay_edits('αβγδε',recorder.receipt(),recorder.replacements)==recorder.text
    replacements=dict(recorder.replacements);replacements[text_hash('XYZ')]='wrong'
    with pytest.raises(ValueError):replay_edits('αβγδε',recorder.receipt(),replacements)


def test_clean_changed_input_fails_without_touching_source(tmp_path):
    source=tmp_path/'source.md';source.write_text('old')
    request=stage_request(tmp_path,'clean.initial',[('selected_text',source)])
    source.write_text('new')
    result=CleanStage().run_document(request,reporter=ReceiptCollector())
    assert result.status=='failed' and result.reason_code=='input_hash_changed'
    assert source.read_text()=='new'


@pytest.mark.parametrize('fresh',[False,True])
def test_wrapper_reclean_preserves_self_input_and_unrelated_sentinel(tmp_path,monkeypatch,fresh):
    from glossapi import Corpus
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE','sequential-adapters')
    corpus=Corpus(tmp_path/'input',tmp_path/'output')
    corpus.markdown_dir.mkdir(exist_ok=True)
    (corpus.markdown_dir/'source.md').write_text('SAME_SENTINEL\n\nReadable native text.')
    corpus.clean(drop_bad=False,num_threads=1)
    before=(corpus.cleaned_markdown_dir/'source.md').read_bytes()
    sentinel=corpus.cleaned_markdown_dir/'unrelated.keep';sentinel.write_text('protected')
    if fresh:corpus=Corpus(tmp_path/'input',tmp_path/'output')
    corpus.clean(input_dir=corpus.cleaned_markdown_dir,drop_bad=False,num_threads=1)
    assert 'SAME_SENTINEL' in (corpus.cleaned_markdown_dir/'source.md').read_text()
    assert sentinel.read_text()=='protected'
    assert before==(corpus.cleaned_markdown_dir/'source.md').read_bytes()


def test_readable_non_greek_is_accepted_without_ocr_need(tmp_path,monkeypatch):
    from glossapi import Corpus
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE','sequential-adapters')
    corpus=Corpus(tmp_path/'input',tmp_path/'output')
    corpus.markdown_dir.mkdir(exist_ok=True)
    (corpus.markdown_dir/'source.md').write_text(('This document describes the research methodology and its results.\n\n')*20)
    import glossapi_rs_noise as noise
    monkeypatch.setattr(noise,'score_markdown_directory_detailed',lambda directory,threads:[
        (str(Path(directory)/'source.md'),80.0,99.0,0.0,0.0)])
    corpus.clean(num_threads=1)
    frame=pd.read_parquet(corpus._get_cached_metadata_parquet())
    row=frame.iloc[0]
    assert 'non_greek' in row['filter'] and not row.needs_ocr
    assert 'source' in corpus.good_files


def plugin_fixture(monkeypatch,pages=4,status='success'):
    from glossapi.corpus.ocr import dispatch
    from glossapi.ocr import page_artifacts
    monkeypatch.setattr(dispatch,'entry_points',lambda **kw:[SimpleNamespace(name='stub-plugin')])
    monkeypatch.setattr(dispatch,'_build_converter',lambda request:SimpleNamespace(convert=lambda source,**kw:
        SimpleNamespace(status=status,document=SimpleNamespace(export_to_markdown=lambda *,page_no:f'STUB OCR {page_no}'))))
    monkeypatch.setattr(page_artifacts,'_page_count',lambda _:pages)
    monkeypatch.setattr(page_artifacts,'uuid',SimpleNamespace(uuid4=lambda:SimpleNamespace(hex='a'*32)))
    return dispatch,page_artifacts


@pytest.mark.parametrize('mode',['replace','supplement'])
def test_partial_and_repeated_plugin_outputs_equal_legacy(tmp_path,monkeypatch,mode):
    dispatch,pages=plugin_fixture(monkeypatch)
    source=tmp_path/'source.pdf';source.write_bytes(b'%PDF-stub-bound')
    native=tmp_path/'native.md';native.write_text(pages._join_page_outputs(['NATIVE 1','NATIVE 2','NATIVE 3','NATIVE 4']) if mode=='replace' else 'WHOLE NATIVE SENTINEL')
    from glossapi.corpus.ocr.config import OcrRequest
    options=SimpleNamespace(kind='stub-plugin')
    configuration=OcrRequest('ocr_bad','docling',options,'cpu',None,False,None,8,220,'single',None,True,False)
    legacy=tmp_path/'legacy';(legacy/'markdown').mkdir(parents=True)
    (legacy/'markdown/source.md').write_bytes(native.read_bytes())
    context=SimpleNamespace(output_dir=legacy,input_dir=tmp_path)
    dispatch.run_plugin_ocr(context,['source.pdf'],request=configuration,page_ranges=['source.pdf:2:2'])
    task=stage_request(tmp_path/'one','ocr.document',[('source_payload',source),('native_markdown',native)],page_ranges=[[2,2]])
    result=OcrStage(options).run_document(task,reporter=ReceiptCollector())
    assert result.status=='ok',result.error
    for relative in ('markdown/source.md','sidecars/ocr_native/source.md','sidecars/ocr_pages/source.json','json/metrics/source.metrics.json'):
        assert (task.work_directory/relative).read_bytes()==(legacy/relative).read_bytes(),relative
    dispatch.run_plugin_ocr(context,['source.pdf'],request=configuration,page_ranges=['source.pdf:3:3'])
    inputs=[('source_payload',source),('native_markdown',task.work_directory/'markdown/source.md'),
        ('ocr_native',task.work_directory/'sidecars/ocr_native/source.md'),('ocr_page_state',task.work_directory/'sidecars/ocr_pages/source.json')]
    second=stage_request(tmp_path/'two','ocr.document',inputs,page_ranges=[[3,3]],prior_state_source_sha256=sha256_file(source))
    repeated=OcrStage(options).run_document(second,reporter=ReceiptCollector())
    assert repeated.status=='ok',repeated.error
    assert (second.work_directory/'markdown/source.md').read_bytes()==(legacy/'markdown/source.md').read_bytes()
    assert all(item['drop_reason']=='unknown_not_reported' for item in repeated.decisions[0]['pages'])
    assert native.read_text().startswith('NATIVE' if mode=='replace' else 'WHOLE')


@pytest.mark.parametrize('missing',['plugin','options'])
def test_missing_plugin_or_options_returns_honest_failed_contract(tmp_path,monkeypatch,missing):
    from glossapi.corpus.ocr import dispatch
    monkeypatch.setattr(dispatch,'entry_points',lambda **kw:[] if missing=='plugin' else [SimpleNamespace(name='stub-plugin')])
    task=stage_request(tmp_path,'ocr.document',[])
    result=OcrStage().run_document(task,reporter=ReceiptCollector())
    assert result.status=='failed' and result.reason_code==f'ocr_{missing}_missing'
    assert result.error.error_type and result.error.traceback


def test_failed_plugin_does_not_publish_canonical_text(tmp_path,monkeypatch):
    plugin_fixture(monkeypatch,status='failure')
    source=tmp_path/'source.pdf';source.write_bytes(b'%PDF-stub')
    task=stage_request(tmp_path,'ocr.document',[('source_payload',source)])
    result=OcrStage(SimpleNamespace(kind='stub-plugin')).run_document(task,reporter=ReceiptCollector())
    assert result.status=='failed' and result.reason_code=='ocr_plugin_conversion_failed'
    assert all(output.retention_class=='failed_attempt' for output in result.outputs)
    assert not (task.work_directory/'markdown/source.md').exists()


def test_prior_page_state_cannot_cross_source_hashes(tmp_path,monkeypatch):
    plugin_fixture(monkeypatch)
    source=tmp_path/'source.pdf';source.write_bytes(b'%PDF-changed')
    state=tmp_path/'state.json';state.write_text('{}')
    task=stage_request(tmp_path,'ocr.document',[('source_payload',source),('ocr_page_state',state)],prior_state_source_sha256='0'*64)
    result=OcrStage(SimpleNamespace(kind='stub-plugin')).run_document(task,reporter=ReceiptCollector())
    assert result.status=='failed' and result.reason_code=='ocr_page_state_invalid'


@pytest.mark.parametrize('outcome',['none','rejected','accepted','failed'])
def test_math_fresh_maps_and_failure_are_distinct(tmp_path,monkeypatch,outcome):
    import glossapi.ocr.math as math_package
    source=tmp_path/'source.pdf';source.write_bytes(b'%PDF-stub')
    document=tmp_path/'source.docling.json';document.write_text('{}')
    stale=tmp_path/'source.latex_map.jsonl';stale.write_text('STALE MAP')
    def infer(document,source,markdown,latex_map,**kw):
        assert latex_map.read_bytes()==b''
        if outcome=='failed':
            markdown.write_text('partial');raise RuntimeError('inference fixture failure')
        markdown.write_text('BASE' if outcome=='none' else 'MATH')
        items=int(outcome!='none');accepted=int(outcome=='accepted')
        if items:latex_map.write_text(json.dumps({'page_no':1,'item_index':1,'latex':'x','accept_score':accepted})+'\n')
        return {'items':items,'accepted':accepted,'time_sec':0.0}
    monkeypatch.setitem(math_package.__dict__,'enrich_from_docling_json',infer)
    task=stage_request(tmp_path,'math.infer',[('source_payload',source),('docling_json',document)])
    result=MathStage().run_document(task,reporter=ReceiptCollector())
    assert stale.read_text()=='STALE MAP'
    if outcome=='failed':
        assert result.status=='failed' and result.error.message=='inference fixture failure'
        assert all(output.retention_class=='failed_attempt' for output in result.outputs)
    else:
        assert result.status=='ok'
        assert result.reason_code==('math_no_regions' if outcome=='none' else 'math_completed')
        assert result.decisions[0]['stats']['accepted']==int(outcome=='accepted')
        assert result.decisions[0]['compile_status']=='not_run'
        assert result.decisions[0]['inputs']['docling_json']['sha256']==sha256_file(document)
        if outcome=='none':assert (task.work_directory/'json/source.latex_map.jsonl').read_bytes()==b''


@pytest.mark.parametrize('missing',['source_payload','docling_json'])
def test_explicit_math_missing_input_fails(tmp_path,missing):
    source=tmp_path/'source.pdf';source.write_bytes(b'%PDF')
    document=tmp_path/'source.docling.json';document.write_text('{}')
    inputs=[item for item in [('source_payload',source),('docling_json',document)] if item[0]!=missing]
    result=MathStage().run_document(stage_request(tmp_path,'math.infer',inputs),reporter=ReceiptCollector())
    assert result.status=='failed' and result.reason_code==('math_pdf_missing' if missing=='source_payload' else 'math_json_missing')


def test_math_triage_is_separate_from_inference(tmp_path,monkeypatch):
    monkeypatch.delenv('GLOSSAPI_TRIAGE_HEURISTIC',raising=False)
    metrics=tmp_path/'per_page.metrics.json';metrics.write_text('{"pages":[{"formula_count":0}]}')
    task=stage_request(tmp_path,'math.triage',[('extract_metrics',metrics)])
    result=MathStage().run_document(task,reporter=ReceiptCollector())
    assert result.status=='ok' and result.reason_code=='math_triage_stop'
    assert 'docling' not in MathStage().packages(StageSpecification('math.triage'))


def test_clean_wrapper_success_products_equal_legacy(tmp_path,monkeypatch):
    from glossapi import Corpus
    products=[]
    for profile in ('legacy','sequential-adapters'):
        monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE',profile)
        root=tmp_path/profile
        corpus=Corpus(root/'input',root/'output')
        corpus._shadow_ledger_enabled=False
        for name,text in [('greek','Καλημέρα κόσμε\n\nκαλό κείμενο.'),('empty',''),('latin','Readable English words with punctuation.\n'*12)]:
            (corpus.markdown_dir/f'{name}.md').write_text(text)
        corpus.clean(num_threads=1,drop_bad=False)
        products.append({str(path.relative_to(corpus.output_dir)):path.read_bytes() for path in corpus.output_dir.rglob('*')
            if path.is_file() and path.suffix not in ('.log','.lock')})
    assert products[0]==products[1]


@pytest.mark.parametrize('newline',['\n','\r\n'])
def test_ocr_clean_producer_spans_replay_and_debug_are_separate(tmp_path,newline):
    text='HEADER\n\n'+' '.join(['αριθμός 1']*22)+'\n\nTAIL<--- Page Split --->SECOND PAGE'
    text=text.replace('\n',newline)
    source,request,result=run_clean(tmp_path,text,'clean.ocr_artifacts',write_debug_files=True)
    assert result.status=='ok',result.error
    assert all('matched_text' not in match and 'matched_text_sha256' in match for match in result.decisions[0]['measurements']['matches'])
    receipt=result.decisions[0]['edits']
    replacements=json.loads((request.work_directory/'edit-replacements.json').read_text())
    assert replay_edits(text,receipt,replacements)==(request.work_directory/'clean_markdown/source.md').read_text()
    assert (request.work_directory/'debug/source.md').exists()
    assert source.read_bytes().decode()==text


@pytest.mark.parametrize('source_root',['given','downloads'])
@pytest.mark.parametrize('fresh',[False,True])
def test_profile_plugin_refresh_uses_fresh_canonical_surface(tmp_path,monkeypatch,source_root,fresh):
    from glossapi import Corpus
    dispatch,pages=plugin_fixture(monkeypatch,pages=1)
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE','sequential-adapters')
    corpus=Corpus(tmp_path/'input',tmp_path/'output')
    root=corpus.input_dir if source_root=='given' else corpus.output_dir/'downloads'
    root.mkdir(exist_ok=True)
    (root/'source.pdf').write_bytes(b'%PDF-stub')
    pages._write_outputs(corpus.output_dir,'source','OLD_NATIVE_SENTINEL\n\nOld native body.',1)
    corpus.clean(drop_bad=False,num_threads=1)
    metadata=corpus._get_cached_metadata_parquet()
    frame=pd.read_parquet(metadata);frame['needs_ocr']=True;frame['ocr_pages']='';frame.to_parquet(metadata,index=False)
    if fresh:corpus=Corpus(tmp_path/'input',tmp_path/'output')
    corpus.ocr(mode='ocr_bad',ocr_options=SimpleNamespace(kind='stub-plugin'))
    raw=(corpus.output_dir/'markdown/source.md').read_text()
    cleaned=(corpus.cleaned_markdown_dir/'source.md').read_text()
    assert 'STUB OCR 1' in raw and 'STUB OCR 1' in cleaned
    assert 'OLD_NATIVE_SENTINEL' not in cleaned
    row=pd.read_parquet(metadata).iloc[0]
    assert row.ocr_success and not row.needs_ocr and row.text==raw
    assert (corpus.output_dir/'sidecars/ocr_native/source.md').read_text().startswith('OLD_NATIVE_SENTINEL')


def test_clean_fact_has_one_home_and_producer_edges(tmp_path,monkeypatch):
    from glossapi import Corpus
    from glossapi.ledger.encoding import unpack_evidence
    import sqlite3
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE','sequential-adapters')
    corpus=Corpus(tmp_path/'input',tmp_path/'output')
    (corpus.markdown_dir/'source.md').write_text('Καλημέρα κόσμε\n\nΚαθαρό κείμενο.')
    corpus.clean(num_threads=1)
    with sqlite3.connect(corpus.output_dir/'corpus.sqlite3') as database:
        evidence=[unpack_evidence(json.loads(row[0])) for row in database.execute("SELECT json FROM snapshots WHERE kind='stage_evidence'")]
        assert sum(item['operation']=='clean.initial' for item in evidence)==1
        assert sum(item['operation']=='clean.selection' for item in evidence)==1
        assert not any('accepted_for_sections' in encoded for encoded, in database.execute("SELECT json FROM snapshots WHERE kind!='stage_evidence'"))
        branches=[json.loads(row[0]) for row in database.execute("SELECT json FROM snapshots WHERE kind='branch_evidence'")]
        assert not any(item.get('spans',{}).get('reason')=='native_backend_does_not_emit_rule_spans' for item in branches)
        assert database.execute('SELECT COUNT(*) FROM task_dependencies').fetchone()[0]>=1
        projected=[json.loads(row[0]) for row in database.execute("SELECT json FROM snapshots WHERE kind='legacy_store'")]
        assert any('producer_attempt_id' in json.dumps(item) for item in projected)


@pytest.mark.parametrize('has_region',[False,True])
def test_real_math_kernel_products_equal_with_deterministic_roi_model(tmp_path,monkeypatch,has_region):
    pytest.importorskip('docling_core')
    pytest.importorskip('docling')
    from glossapi.ocr.math import enrich as enrichment
    from docling_core.types.doc import DoclingDocument,DocItemLabel,ProvenanceItem,BoundingBox,CoordOrigin
    from tests.test_ledger_stages import simple_pdf
    from PIL import Image
    # The retained kernel's historical model import is not in Docling 2.81.
    # Supply the deterministic ROI engine at that boundary; do not qualify or
    # silently substitute a different real model/API (see LS4-G03).
    from types import ModuleType
    code_formula=ModuleType('docling.models.code_formula_model')
    code_formula.CodeFormulaModel=object
    code_formula.CodeFormulaModelOptions=lambda **options:SimpleNamespace(**options)
    monkeypatch.setitem(sys.modules,'docling.models.code_formula_model',code_formula)
    source=simple_pdf(tmp_path/'source.pdf')
    document=DoclingDocument(name='deterministic-math')
    document.add_text(label=DocItemLabel.TEXT,text='BASE SENTINEL')
    if has_region:
        document.add_text(label=DocItemLabel.FORMULA,text='old',prov=ProvenanceItem(page_no=1,
            bbox=BoundingBox(l=10,t=40,r=80,b=10,coord_origin=CoordOrigin.BOTTOMLEFT),charspan=(0,3)))
    json_path=tmp_path/'source.docling.json';document.save_as_json(json_path)
    class StubModel:
        def __init__(self,*args,**kwargs):pass
        def __call__(self,document,batch):
            for element in batch:
                element.item.text='x = 1'
                yield element.item
    class StubRaster:
        def __init__(self,*args,**kwargs):pass
        def get_pil(self,*args,**kwargs):return Image.new('RGB',(200,200),'white')
    monkeypatch.setattr(code_formula,'CodeFormulaModel',StubModel)
    monkeypatch.setattr(enrichment,'PageRasterCache',StubRaster)
    monkeypatch.setattr(enrichment,'time',SimpleNamespace(time=lambda:100.0))
    legacy_md=tmp_path/'legacy.md';legacy_map=tmp_path/'legacy.jsonl'
    legacy=enrichment.enrich_from_docling_json(json_path,source,legacy_md,legacy_map,device='cpu')
    task=stage_request(tmp_path,'math.infer',[('source_payload',source),('docling_json',json_path)])
    result=MathStage().run_document(task,reporter=ReceiptCollector())
    assert result.status=='ok',result.error
    assert (task.work_directory/'markdown/source.md').read_bytes()==legacy_md.read_bytes()
    assert result.decisions[0]['stats']==legacy
    if has_region:
        assert (task.work_directory/'json/source.latex_map.jsonl').read_bytes()==legacy_map.read_bytes()
        regions=result.decisions[0]['regions']
        assert any(item['reason_code']=='math_region_input' and 'bbox' in item['details'] for item in regions)
    else:
        assert (task.work_directory/'json/source.latex_map.jsonl').read_bytes()==b''
        assert not legacy_map.exists()  # Explicitly declared stale/absent-map correction.


def test_clean_keeps_legacy_recursive_and_chunk_products(tmp_path,monkeypatch):
    from glossapi import Corpus
    products=[]
    for profile in ('legacy','sequential-adapters'):
        monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE',profile)
        corpus=Corpus(tmp_path/profile/'input',tmp_path/profile/'output');corpus._shadow_ledger_enabled=False
        for name in ('source.md','source__p00001-00002.md','nested/other.md'):
            path=corpus.markdown_dir/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('Καλημέρα κόσμε\n\nReadable body.')
        corpus.clean(num_threads=1,drop_bad=False)
        products.append({str(path.relative_to(corpus.output_dir)):path.read_bytes() for path in corpus.output_dir.rglob('*')
            if path.is_file() and path.suffix not in ('.log','.lock')})
    assert products[0]==products[1]


def test_canonical_and_chunks_share_version_without_task_identity_collision(tmp_path,monkeypatch):
    from glossapi import Corpus
    from glossapi.ledger import LedgerWriter
    from glossapi.ledger.identity import source_key
    from tests.test_ledger_stages import simple_pdf
    import sqlite3
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE','sequential-adapters')
    corpus=Corpus(tmp_path/'input',tmp_path/'output')
    corpus.input_dir.mkdir(parents=True,exist_ok=True)
    source=simple_pdf(corpus.input_dir/'source.pdf')
    markdown=[corpus.markdown_dir/name for name in ('source.md','source__p00001-00002.md')]
    for path in markdown:path.write_text('Καλημέρα κόσμε\n\nReadable body.')
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        with writer.transaction():
            run=writer.run({}, {}, {}, code_revision='fixture')
            document=writer.document(source_key(source.as_uri()))
            version=writer.version(document,source_sha256=sha256_file(source),stored_sha256=sha256_file(source))
            for path in markdown:
                writer.artifact(run,'extract','raw_markdown',path.as_uri(),version_id=version,
                    sha256=sha256_file(path),bytes=path.stat().st_size,ownership='initial_external',retention_class='initial')
    corpus.clean(num_threads=1,drop_bad=False)
    assert all((corpus.cleaned_markdown_dir/path.name).is_file() for path in markdown)
    with sqlite3.connect(corpus.output_dir/'corpus.sqlite3') as database:
        operations=[row[0] for row in database.execute("SELECT operation FROM tasks WHERE stage='clean' AND version_id=?",(version,))]
        assert sum(operation.split('.artifact:',1)[0]=='clean.initial' for operation in operations)==2
        assert sum(operation.split('.artifact:',1)[0]=='clean.selection' for operation in operations)==2
        assert database.execute('SELECT COUNT(*) FROM documents').fetchone()[0]==1


def test_missing_math_inputs_remain_observed_after_another_adapter(tmp_path, monkeypatch):
    from glossapi import Corpus
    from glossapi.ledger import LedgerWriter
    from glossapi.ledger.journal import ResultJournal
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'sequential-adapters')
    corpus = Corpus(tmp_path/'input', tmp_path/'output')
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        journal = ResultJournal(writer, corpus, 'math', 'enrich_math', {})
        journal.adapter_stages.add('math')
        for reason in ('math_json_missing', 'math_pdf_missing'):
            journal.observe(reason, status='failed', legacy_stem='unprocessed')
        assert journal.failures == ['math_json_missing', 'math_pdf_missing']
        observed = writer.connection.execute(
            "SELECT reason_code FROM events WHERE reason_code IN ('math_json_missing','math_pdf_missing') ORDER BY event_id").fetchall()
        assert [row[0] for row in observed] == journal.failures


def test_native_document_rewrite_exceeds_default_csv_field_limit(tmp_path):
    import csv
    limit=csv.field_size_limit()
    text=('Καλημέρα &amp; κόσμε.\n'*16000)
    source,request,result=run_clean(tmp_path,text)
    assert result.status=='ok',result.error
    assert csv.field_size_limit()==limit
    replacements=json.loads((request.work_directory/'edit-replacements.json').read_text())
    assert max(map(len,replacements.values()))>131072
    assert replay_edits(text,result.decisions[0]['edits'],replacements)==(request.work_directory/'clean_markdown/source.md').read_bytes().decode()
    assert source.read_bytes().decode()==text
