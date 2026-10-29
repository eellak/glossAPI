"""Cutover products: aggregate equivalence, lineage and retired-store authority."""

from glossapi.ledger.compact import connect as ledger_connect
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

from glossapi import Corpus
from glossapi.ledger.encoding import unpack_evidence


def corpus_fixture(tmp_path,*,enabled=True):
    corpus=Corpus(tmp_path/'input',tmp_path/'output')
    corpus._shadow_ledger_enabled=enabled
    for name in ('zeta','alpha','beta'):
        text=f'# {name}\n\nBody of {name}.\n\n## Second\n\nMore text.'
        (corpus.markdown_dir/(name+'.md')).write_text(text)
        (corpus.cleaned_markdown_dir/(name+'.md')).write_text(text)
    metadata=corpus.output_dir/'download_results/download_results.parquet'
    metadata.parent.mkdir()
    pd.DataFrame({'filename':['zeta.pdf','alpha.pdf','beta.pdf'],'filter':['ok']*3,'marker':[0,1,2]}).to_parquet(metadata,index=False)
    return corpus


def evidence(corpus,stage):
    with ledger_connect(corpus.output_dir/'corpus.sqlite3') as database:
        return [value for encoded, in database.execute("SELECT json FROM snapshots WHERE kind='stage_evidence' ORDER BY rowid")
            if (value:=unpack_evidence(json.loads(encoded))).get('stage')==stage]


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('selection',['good','metadata','zero','legacy'])
def test_section_frozen_enumeration_global_ids_and_row_groups(tmp_path,monkeypatch,enabled,selection):
    order=['zeta.md','beta.md','alpha.md']
    real_listdir=os.listdir
    outputs=[]
    for profile in ('legacy','cutover'):
        monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE',profile)
        corpus=corpus_fixture(tmp_path/profile,enabled=enabled)
        corpus.good_files=['alpha','zeta'] if selection=='good' else []
        metadata=corpus.output_dir/'download_results/download_results.parquet'
        frame=pd.read_parquet(metadata)
        if selection in ('zero','legacy'):frame['filter']='bad'
        if selection=='legacy':frame['extraction']=['bad','good','bad']
        frame.to_parquet(metadata,index=False)
        monkeypatch.setattr(os,'listdir',lambda path:order if Path(path)==corpus.markdown_dir else real_listdir(path))
        assert corpus.section() is None
        parquet=pq.ParquetFile(corpus.sections_parquet)
        outputs.append((corpus.sections_parquet.read_bytes(),parquet.read().to_pylist(),
            [parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)]))
    assert outputs[0]==outputs[1]


def install_model(monkeypatch,corpus,*,fail=False,index=False):
    from glossapi.gloss_section_classifier import GlossSectionClassifier
    model=corpus.output_dir/'model.joblib';model.write_text('first')
    corpus.section_classifier_model_path=model
    class Model:
        def __init__(self,label):self.label=label
        def predict(self,frame):
            if fail:raise RuntimeError('deterministic prediction failure')
            return [self.label]*len(frame)
    monkeypatch.setattr(GlossSectionClassifier,'load_model',lambda self,path:Model(Path(path).read_text()))
    if index:monkeypatch.setattr(GlossSectionClassifier,'_compute_likely_index_for_section',lambda *args:(1,[]))
    return model


@pytest.mark.parametrize('fresh',[False,True])
def test_changed_model_and_text_invalidate_numeric_checkpoint(tmp_path,monkeypatch,fresh):
    corpus=corpus_fixture(tmp_path);corpus.good_files=['alpha']
    model=install_model(monkeypatch,corpus)
    corpus.section();corpus.annotate(fully_annotate=False)
    assert set(pd.read_parquet(corpus.classified_parquet).predicted_section)=={'first'}
    old_keys=evidence(corpus,'sections')[0]['section_keys']
    (corpus.markdown_dir/'alpha.md').write_text('# Changed\n\nChanged input with the same numeric ID.')
    model.write_text('second')
    if fresh:
        corpus=Corpus(corpus.input_dir,corpus.output_dir,section_classifier_model_path=model)
        corpus.good_files=['alpha']
    corpus.section();corpus.annotate(fully_annotate=False)
    assert set(pd.read_parquet(corpus.classified_parquet).predicted_section)=={'second'}
    sections=[value for value in evidence(corpus,'sections') if value['operation']=='sections.document']
    assert old_keys!=sections[-1]['section_keys']
    annotations=[value for value in evidence(corpus,'annotate') if 'lineage' in value]
    assert annotations[0]['model_sha256']!=annotations[-1]['model_sha256']
    assert annotations[-1]['lineage'][0]['section_key']==sections[-1]['section_keys'][0]
    assert annotations[-1]['lineage'][0]['confidence'] is None


@pytest.mark.parametrize('failure,index',[(False,False),(True,False),(False,True)])
def test_model_fallback_index_and_full_label_lineage(tmp_path,monkeypatch,failure,index):
    corpus=corpus_fixture(tmp_path);corpus.good_files=['alpha']
    install_model(monkeypatch,corpus,fail=failure,index=index)
    if index:(corpus.markdown_dir/'alpha.md').write_text('\n'.join(f'# Heading {ordinal}\n\nBody {ordinal}.\n' for ordinal in range(10)))
    corpus.section();corpus.annotate(fully_annotate=True)
    lineage=next(value['lineage'] for value in evidence(corpus,'annotate') if 'lineage' in value)
    assert lineage
    for row in lineage:
        assert row['model_label']==('άλλο' if failure else 'first')
        assert row['fallback_reason']==('prediction_exception' if failure else None)
        overridden=index and row['compatibility_id']<=3
        assert row['index_label']==('π' if overridden else row['model_label'])
        assert row['final_label'] is not None
        assert row['heuristic_rules']==(['index_detection'] if overridden else [])


def test_unchanged_text_has_stable_private_keys_and_no_shadow_rows(tmp_path):
    corpus=corpus_fixture(tmp_path);corpus.good_files=['zeta']
    corpus.section();corpus.section()
    rows=[value for value in evidence(corpus,'sections') if 'section_keys' in value]
    assert rows[0]['section_keys']==rows[-1]['section_keys']
    with ledger_connect(corpus.output_dir/'corpus.sqlite3') as database:
        assert not any('sections_for_annotation' in encoded for encoded, in database.execute("SELECT json FROM snapshots WHERE kind='legacy_store'"))
        assert database.execute('SELECT COUNT(*) FROM task_dependencies').fetchone()[0]>=2
    assert 'section_key' not in pd.read_parquet(corpus.sections_parquet).columns


@pytest.mark.parametrize('codec',['none','zstd'])
def test_shorter_sharded_rerun_retires_only_owned_mirrors(tmp_path,codec):
    corpus=corpus_fixture(tmp_path)
    destination=corpus.output_dir/'shards'
    first=corpus.jsonl_sharded(destination,shard_size_bytes=1,compression=codec)
    assert len(first)==3
    sentinel=destination/'unrelated.jsonl';sentinel.write_text('untouched')
    # Select fewer text inputs; mutable metadata is deliberately not authority.
    (corpus.cleaned_markdown_dir/'beta.md').unlink();(corpus.cleaned_markdown_dir/'zeta.md').unlink()
    second=corpus.jsonl_sharded(destination,shard_size_bytes=1,compression=codec)
    assert len(second)==1 and second[0]==first[0]
    assert not first[1].exists() and not first[2].exists() and sentinel.read_text()=='untouched'
    with ledger_connect(corpus.output_dir/'corpus.sqlite3') as database:
        assert database.execute("SELECT COUNT(*) FROM events e JOIN files f USING(file_id) WHERE event_kind='artifact' AND reason_code='deleted' AND f.kind='document_jsonl_shard' AND f.uri NOT LIKE '%/.glossapi/%'").fetchone()[0]==2
        for uri, in database.execute("SELECT uri FROM files WHERE kind='document_jsonl_shard' AND uri LIKE '%/.glossapi/%'"):
            from glossapi.stages.products import local_path
            assert local_path(uri).is_file()


@pytest.mark.parametrize('sharded',[False,True])
def test_empty_export_and_chunk_only_omission(tmp_path,sharded):
    corpus=corpus_fixture(tmp_path)
    for path in corpus.cleaned_markdown_dir.glob('*.md'):path.unlink()
    for path in corpus.markdown_dir.glob('*.md'):path.unlink()
    chunks=corpus.cleaned_markdown_dir/'chunks/alpha';chunks.mkdir(parents=True)
    (chunks/'alpha__p0001-0002.md').write_text('Only a chunk')
    result=corpus.jsonl_sharded(corpus.output_dir/'shards',compression='none') if sharded else corpus.jsonl(corpus.output_dir/'empty.jsonl')
    assert result==[] if sharded else result.read_bytes()==b''
    selected=next(value['selected_versions'] for value in evidence(corpus,'export') if 'selected_versions' in value)
    assert len(selected)==1 and all(not value['included'] for value in selected)
    assert selected[0]['version_id'] is not None and selected[0]['input_file_id']
    metadata=next(value['metadata_decisions'] for value in evidence(corpus,'export') if 'metadata_decisions' in value)
    assert len(metadata)==3 and all(not value['included'] for value in metadata)


def test_metadata_projection_is_authority_with_fresh_wrapper(tmp_path):
    corpus=corpus_fixture(tmp_path)
    corpus.jsonl(corpus.output_dir/'first.jsonl')
    metadata=corpus.output_dir/'download_results/download_results.parquet'
    frame=pd.read_parquet(metadata);frame['marker']=99;frame.to_parquet(metadata,index=False)
    fresh=Corpus(corpus.input_dir,corpus.output_dir)
    second=fresh.jsonl(corpus.output_dir/'second.jsonl')
    assert second.read_bytes()==(corpus.output_dir/'first.jsonl').read_bytes()
    assert pd.read_parquet(metadata).marker.tolist()==[0,1,2]


@pytest.mark.parametrize('method',['download','extract','clean','filter','ocr','section','annotate','jsonl','jsonl_sharded'])
def test_public_wrapper_signatures_are_preserved(method):
    function=getattr(Corpus,method)
    if hasattr(function,'__wrapped__'):assert inspect.signature(function)==inspect.signature(function.__wrapped__)


def test_stage_metadata_imports_no_optional_backend():
    code="import sys; from glossapi.stages.sections import SectionsStage; from glossapi.stages.annotate import AnnotateStage; from glossapi.stages.export import ExportStage; assert not {'pandas','numpy','sklearn','dask','torch','docling'} & set(sys.modules)"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_edit_summary_keeps_per_rule_counts_and_order_hashes():
    from glossapi.stages.edits import EditRecorder
    recorder=EditRecorder('abcd');recorder.edit('first',0,1,'A');recorder.edit('second',3,4,'D');recorder.edit('first',1,2,'B')
    full=recorder.receipt();summary=recorder.summary()
    assert len(full['edits'])==summary['edit_count']==3
    assert summary['rules']['first']['count']==2 and summary['rules']['second']['count']==1
    assert 'edits' not in summary and 'composition_map' not in summary
    assert len(summary['ordered_edits_sha256'])==64 and len(summary['composition_sha256'])==64


def test_generation_manifest_binds_all_immutable_products(tmp_path):
    from glossapi.stages.products import local_path
    from glossapi.ledger.writer import integrity
    corpus=corpus_fixture(tmp_path);corpus.good_files=['alpha']
    corpus.section();corpus.jsonl(corpus.output_dir/'export.jsonl')
    with ledger_connect(corpus.output_dir/'corpus.sqlite3') as database:
        integrity(database)
        for attempt,uri in database.execute("SELECT producer_attempt_id,uri FROM files WHERE kind='generation_manifest'"):
            manifest=json.loads(local_path(uri).read_text())
            for entry in manifest['files']:
                path=local_path(uri).parent/entry['path']
                presence=database.execute('SELECT presence FROM current_file_state WHERE producer_attempt_id=? AND uri=?',(attempt,path.as_uri())).fetchone()[0]
                if presence=='deleted':
                    assert not path.exists()
                    assert database.execute("SELECT 1 FROM events WHERE event_kind='retention' AND reason_code='delete_completed' AND file_id IN (SELECT file_id FROM files WHERE uri=?)",(path.as_uri(),)).fetchone()
                else:
                    assert path.stat().st_size==entry['bytes']
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
                assert database.execute('SELECT COUNT(*) FROM files WHERE producer_attempt_id=? AND uri=? AND sha256=?',
                    (attempt,path.as_uri(),entry['sha256'])).fetchone()[0]==1


def test_corrupt_immutable_metadata_rejects_mirror_consumption(tmp_path):
    from glossapi.stages.products import current_product,local_path
    corpus=corpus_fixture(tmp_path);corpus.jsonl(corpus.output_dir/'first.jsonl')
    with ledger_connect(corpus.output_dir/'corpus.sqlite3') as database:
        product=current_product(database,corpus.output_dir/'download_results/download_results.parquet')
    local_path(product['immutable_uri']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Metadata projection corrupt'):
        Corpus(corpus.input_dir,corpus.output_dir).jsonl(corpus.output_dir/'second.jsonl')


def test_failed_mirror_publication_keeps_prior_generation_selected(tmp_path,monkeypatch):
    from glossapi.stages.products import current_product
    corpus=corpus_fixture(tmp_path);destination=corpus.output_dir/'export.jsonl'
    corpus.jsonl(destination);original=destination.read_bytes()
    with ledger_connect(corpus.output_dir/'corpus.sqlite3') as database:before=current_product(database,destination)
    (corpus.cleaned_markdown_dir/'alpha.md').write_text('Changed text')
    replace=os.replace
    def fail_publication(source,target):
        if Path(target)==destination:raise OSError('deterministic mirror failure')
        return replace(source,target)
    monkeypatch.setattr(os,'replace',fail_publication)
    with pytest.raises(OSError,match='deterministic mirror failure'):corpus.jsonl(destination)
    assert destination.read_bytes()==original
    with ledger_connect(corpus.output_dir/'corpus.sqlite3') as database:assert current_product(database,destination)==before


def test_fresh_section_restores_clean_producer_selection(tmp_path):
    pytest.importorskip('glossapi_rs_cleaner')
    corpus=corpus_fixture(tmp_path);corpus.clean(drop_bad=False,num_threads=1)
    fresh=Corpus(corpus.input_dir,corpus.output_dir);fresh.good_files=['unrelated']
    fresh.section()
    assert set(pd.read_parquet(fresh.sections_parquet).filename)=={'alpha','beta','zeta'}
    assert fresh.markdown_dir==fresh.cleaned_markdown_dir


def test_export_binds_given_text_versions_and_absent_source_omission(tmp_path):
    corpus=corpus_fixture(tmp_path);corpus.input_dir.mkdir()
    (corpus.input_dir/'absent.pdf').write_bytes(b'given source without text')
    destination=corpus.jsonl(corpus.output_dir/'export.jsonl')
    selected=next(value['selected_versions'] for value in evidence(corpus,'export') if 'selected_versions' in value)
    assert len(selected)==4 and all(value['version_id'] and value['document_id'] for value in selected)
    assert sum(value['included'] for value in selected)==3
    omitted=next(value for value in selected if not value['included'])
    assert omitted['source_uri'].endswith('/absent.pdf') and omitted['reason_code']=='export_canonical_markdown_missing'
    payload=destination.read_text()
    assert not any(value['document_id'] in payload or value['version_id'] in payload for value in selected)


def test_public_download_adapter_preserves_transport_hook_and_source_fields(tmp_path,monkeypatch):
    from glossapi.gloss_downloader import GlossDownloader
    from glossapi.stages.receipts import acquired_receipt
    observed=[]
    async def kernel(self,row_index,url,semaphore,rate_limiter,retry_count=0,filename_base=None,referer=None):
        path=Path(self.downloads_dir)/(filename_base+'.pdf')
        path.write_bytes(b'%PDF-verified explicit transport hook')
        acquired_receipt(url,path,True)
        observed.append(path)
        return True,path.name,'pdf','',retry_count
    monkeypatch.setattr(GlossDownloader,'download_file',kernel)
    corpus=Corpus(tmp_path/'input',tmp_path/'output',downloader_config={'sleep':0})
    source=tmp_path/'sources.parquet'
    pd.DataFrame({'url':['https://fixture.invalid/source.pdf'],'source_detail':['original Greek source field']}).to_parquet(source,index=False)
    returned=corpus.download(source)
    assert returned.iloc[0]['source_detail']=='original Greek source field'
    assert bool(returned.iloc[0]['download_success'])
    assert observed and '.glossapi/staging/' in str(observed[0])
    output=corpus.downloads_dir/returned.iloc[0]['filename']
    assert output.read_bytes()==b'%PDF-verified explicit transport hook'
    receipt=next(value for value in evidence(corpus,'download') if 'acquisitions' in value)
    assert receipt['acquisitions'][0]['stored_sha256']==hashlib.sha256(output.read_bytes()).hexdigest()


def test_export_keeps_two_metric_inputs_with_one_producer_role_edge(tmp_path):
    from glossapi.ledger.journal import shadow_stage
    from glossapi.stages.contracts import StageSpecification
    from glossapi.stages.products import execute_aggregate,generation_outputs,publish_products
    corpus=corpus_fixture(tmp_path)
    class MetricPairStage:
        name='extract';contract_version=1
        def finalize(self,request,*,reporter):
            request.work_directory.mkdir(parents=True)
            for name,count in [('alpha.metrics.json',7),('alpha.per_page.metrics.json',9)]:
                (request.work_directory/name).write_text(json.dumps({'page_count':count,'pages':[{'formula_count':2,'code_count':0}]}))
            return generation_outputs(request,[(name,'extract_metrics') for name in ('alpha.metrics.json','alpha.per_page.metrics.json')])
    @shadow_stage('extract','extract.metric_pair_fixture')
    def publish_metrics(self):
        result,workspace=execute_aggregate(self,MetricPairStage(),[],StageSpecification('extract.metric_pair',(),('extract_metrics',)))
        publish_products(self,result,workspace,{name:self.output_dir/'json/metrics'/name for name in ('alpha.metrics.json','alpha.per_page.metrics.json')})
    publish_metrics(corpus)
    records=[json.loads(line) for line in corpus.jsonl(corpus.output_dir/'export.jsonl').read_text().splitlines()]
    assert next(row for row in records if row['filename']=='alpha')['page_count']==7
    with ledger_connect(corpus.output_dir/'corpus.sqlite3') as database:
        task=database.execute("SELECT task_id FROM tasks WHERE operation='export.publish'").fetchone()[0]
        assert database.execute("SELECT COUNT(*) FROM task_dependencies WHERE task_id=? AND input_role='metrics_or_math'",(task,)).fetchone()[0]==1
        assert database.execute("SELECT COUNT(*) FROM attempt_inputs i JOIN attempts a ON a.attempt_id=i.attempt_id WHERE a.task_id=? AND i.role='metrics_or_math'",(task,)).fetchone()[0]==2
