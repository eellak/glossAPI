"""Frozen run plans, streaming document work and compatibility barriers."""
from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import json
from pathlib import Path
import time
from urllib.parse import unquote, urlsplit

from .contracts import ArtifactReference, DocumentSnapshot, StageSpecification
from .execution import sha256_file
from .replay import committed_results
from .scheduler import ScheduledOperation, StreamingScheduler, local_path
from ..ledger.writer import LedgerWriter
from ..ledger.journal import shadow_stage


def extraction_settings(options=None):
    import os
    return {'phase1_backend':'safe','repair_fonts':True,'export_doc_json':True,'emit_formula_index':False,
        'benchmark_mode':False,'num_threads':1,'page_range':None,'long_pdf_page_threshold':600,'chunk_size':200,
        'max_chunk_timeouts':2,'per_page_timeout_s':20,'images_scale':float(os.environ.get('GLOSSAPI_IMAGES_SCALE','1.25')),'formula_enrichment':False,
        'code_enrichment':False,'use_cls':False,**(options or {})}


def prepare_streaming(corpus,sources=(),*,workers=None,stages=('extract','clean','sections'),settings=None,
                      resume=None,retry_failed=False,observer=None,writer=None):
    """Return committed results; public serializers consume them without reruns.

    Acquisition naming is sealed in source order before any worker starts.
    Corpus selection/global IDs, annotation and shard publication remain barriers.
    """
    from ..ledger.journal import source_key
    began=time.monotonic()
    from .retention import resolved_settings
    settings=resolved_settings(settings)
    root=Path(corpus.output_dir).resolve()
    sources=list(sources)
    source_rows=[{'source_uri':str(source)} for source in sources]
    download_manifest=None
    if not resume and any(str(source).startswith(('http://','https://')) for source in sources):
        import pandas as pd
        from ..gloss_downloader import GlossDownloader
        raw_manifest=settings.get('source_manifest')
        records=raw_manifest.get('documents',[]) if isinstance(raw_manifest,dict) else raw_manifest
        records=records or [{'url':str(source)} for source in sources]
        frame=pd.DataFrame([row if isinstance(row,dict) else {'url':row} for row in records])
        # Retain the source frame before expansion/naming for the unchanged
        # download serializer. Resume binds these bytes, not a reconstructed
        # frame with inferred column types or missing source metadata.
        from ..ledger.writer import new_id
        download_manifest=root/'.glossapi/source-manifests'/new_id()/'sources.parquet'
        download_manifest.parent.mkdir(parents=True,exist_ok=False)
        frame.to_parquet(download_manifest,index=False)
        settings={**settings,'download_projection_manifest':{'path':str(download_manifest),
            'sha256':sha256_file(download_manifest),'bytes':download_manifest.stat().st_size},
            'semantic_files':[*settings.get('semantic_files',()),
                {'path':str(download_manifest),'sha256':sha256_file(download_manifest)}]}
        # These retained kernels seal expansion, duplicate policy and collision
        # names before acquisition. They perform no network or backend work.
        naming=GlossDownloader.__new__(GlossDownloader)
        naming.url_column='url';naming._used_filename_bases=set()
        frame=naming.ensure_filename_base(naming._expand_and_mark_duplicates(frame))
        source_rows=frame.to_dict(orient='records')
        sources=[row.get('url',row.get('source_uri')) for row in source_rows]
    with nullcontext(writer) if writer is not None else LedgerWriter(root/'corpus.sqlite3') as writer:
        engine=StreamingScheduler(writer,root,workers=workers,cpu_tokens=settings.get('cpu_tokens'),
            memory_bytes=settings.get('memory_budget_bytes'),disk_budget_bytes=settings.get('disk_budget_bytes'),observer=observer)
        if resume:
            run=resume
            operations,_,run_row=engine._load(run)
            frozen=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(run_row['settings_sha256'],)).fetchone()[0])
            stages=frozen.get('requested_stages',stages)
            settings=frozen
        else:
            operations=[]
            with writer.transaction():
                from ..ledger.journal import redact_credentials
                observation=writer.run(redact_credentials(settings),{'mode':'explicit_source_inventory'}, {},code_revision=__import__('glossapi.ledger.journal',fromlist=['code_revision']).code_revision())
                if download_manifest:
                    writer.artifact(observation,'download','source_manifest',download_manifest.as_uri(),
                        sha256=sha256_file(download_manifest),bytes=download_manifest.stat().st_size,
                        ownership='managed',retention_class='initial')
                manifest=writer.snapshot('source_manifest',{'records':__import__('glossapi.ledger.journal',fromlist=['safe_settings']).safe_settings(source_rows)})
                seen=set()
                for ordinal,given in enumerate(sources):
                    source=str(given)
                    remote=source.startswith(('http://','https://'))
                    uri=source if remote else Path(unquote(urlsplit(source).path) if source.startswith('file:') else source).resolve().as_uri()
                    key=source_key(uri)
                    proposed=settings.get('planned_document_ids',{}).get(key)
                    known=writer.connection.execute('SELECT document_id FROM documents WHERE source_key=?',(key,)).fetchone()
                    if proposed and known is None:
                        from ..ledger.writer import validate_id,utc_now
                        validate_id(proposed)
                        writer.connection.execute('INSERT INTO documents VALUES(?,?,?)',(proposed,key,utc_now()))
                    document=writer.document(key)
                    path=None if remote else local_path(uri)
                    digest=sha256_file(path) if path else None
                    version=writer.version(document,source_sha256=digest,stored_sha256=digest)
                    writer.source_record(document,observation,uri,source_rows[ordinal],source_kind='url' if remote else 'local_path',
                        source_row=str(source_rows[ordinal].get('source_row',ordinal)),url_index=max(0,int(source_rows[ordinal].get('url_index',0))),manifest_sha256=manifest)
                    if key in seen:
                        continue
                    seen.add(key)
                    artifacts=()
                    if path:
                        file_id=writer.artifact(observation,'extract','source_payload',uri,sha256=digest,bytes=path.stat().st_size,
                            version_id=version,ownership='initial_external',retention_class='initial')
                        artifacts=(ArtifactReference(file_id,'source_payload',path,digest,path.stat().st_size),)
                    snapshot=DocumentSnapshot(document,version,uri,artifacts=artifacts)
                    parent=None
                    if remote:
                        # A finite source epoch fixes names and collisions. The
                        # retained downloader validates extensions/redirects.
                        name=source_rows[ordinal]['filename_base']
                        download_settings={'download_mode':'standard','legacy_outer_retry_count':0,**corpus.downloader_config,**settings.get('download',{}),
                            'url':source,'filename_base':name,'row_index':ordinal}
                        parent=f'{ordinal}:download'
                        operations.append(ScheduledOperation(parent,'download',snapshot,
                            StageSpecification('download.acquire',required_output_kinds=('download_payload',),settings=download_settings),
                            skip_reason='duplicate_url' if source_rows[ordinal].get('is_duplicate') else None))
                    if 'extract' in stages:
                        key=f'{ordinal}:extract'
                        operations.append(ScheduledOperation(key,'extract',snapshot,
                            StageSpecification('extract.document',('source_payload',) if artifacts else (),('raw_markdown',),
                                settings=extraction_settings(settings.get('extract'))),
                            inputs=((parent,'download_payload','source_payload'),) if parent else ()))
                        parent=key
                    if 'clean' in stages:
                        key=f'{ordinal}:clean'
                        operations.append(ScheduledOperation(key,'clean',replace(snapshot,artifacts=()),
                            StageSpecification('clean.initial',required_output_kinds=('clean_markdown','clean_metrics'),settings={'num_threads':1}),
                            inputs=((parent,'raw_markdown','selected_text'),)))
                        parent=key
                    if 'ocr' in stages:
                        from ..ocr.defaults import default_values, select_device, cache_root
                        ocr_settings = dict(settings.get('ocr',{}))
                        if ocr_settings.get('entry_point_name','glossapi') != 'glossapi':
                            raise ValueError('This run adapter requires the Greek OCR entry point; pass external options to Corpus.ocr()')
                        device = ocr_settings.get('device_id') or ocr_settings.get('options',{}).get('device') or select_device()
                        values = {**default_values(device), **ocr_settings.get('options',{})}
                        quality_key=f'{ordinal}:quality'
                        operations.append(ScheduledOperation(quality_key,'extract',snapshot,
                            StageSpecification('extract.page_quality',required_output_kinds=('page_quality',),
                                settings={'page_quality':settings.get('page_quality',{})}),
                            inputs=((f'{ordinal}:download','download_payload','source_payload'),) if remote else ()))
                        source_snapshot=replace(snapshot,artifacts=() if remote else snapshot.artifacts)
                        edges=((f'{ordinal}:download','download_payload','source_payload'),) if remote else ()
                        ocr_key=f'{ordinal}:ocr'
                        operations.append(ScheduledOperation(ocr_key,'ocr',source_snapshot,
                            StageSpecification('ocr.document',('source_payload',) if not remote else (),('ocr_markdown','ocr_metrics'),device,
                                {**ocr_settings,'options':values,'entry_point_name':'glossapi',
                                 'cache_root':str(cache_root(ocr_settings.get('cache_root')))}),
                            inputs=(*edges,(f'{ordinal}:extract','raw_markdown','native_markdown'),(quality_key,'page_quality','page_quality')),
                            after=(parent,),cpu_tokens=int(ocr_settings.get('num_threads',1))))
                        parent=f'{ordinal}:clean_after_ocr'
                        operations.append(ScheduledOperation(parent,'clean',replace(snapshot,artifacts=()),
                            StageSpecification('clean.initial.artifact:after_ocr',required_output_kinds=('clean_markdown','clean_metrics'),settings={'num_threads':1}),
                            inputs=((ocr_key,'ocr_markdown','selected_text'),)))
                    if 'sections' in stages:
                        filename=path.stem if path else name
                        operations.append(ScheduledOperation(f'{ordinal}:sections','sections',replace(snapshot,artifacts=()),
                            StageSpecification('sections.document',required_output_kinds=('section_fragment',),settings={'filename':filename}),
                            inputs=((parent,'clean_markdown','selected_text'),)))
            run=engine.freeze(operations,settings={**settings,'requested_stages':list(stages)})
        corpus._retention_settings={**settings,'requested_stages':list(stages)}
        result=engine.execute(run,resume=bool(resume),retry_failed=retry_failed)
        result['wall_seconds']=time.monotonic()-began
        replays={}
        download_replays={}
        selected_sources=[]
        source_versions={}
        per_version={}
        for operation in operations:per_version.setdefault(operation.document.version_id,[]).append(operation.key)
        completed_versions={version for version,keys in per_version.items() if all(key in result['attempts'] for key in keys)}
        for ordinal,operation in enumerate(operations):
            if operation.stage=='download':
                task=next(task['task_id'] for task in writer.task_states(run) if task['ordinal']==ordinal)
                latest=writer.latest_attempt(task)
                if latest:
                    receipts=[artifact for artifact in engine._outputs(latest['attempt_id']) if artifact.kind=='worker_receipt']
                    download_replays[(operation.document.source_uri,operation.specification.settings['filename_base'])]={
                        'attempt_id':latest['attempt_id'],'receipt':receipts[0] if receipts else None,
                        'status':latest['status'],'reason_code':latest['reason_code'],'message':latest['message']}
            attempt=result['attempts'].get(operation.key)
            if not attempt:
                continue
            if operation.stage=='download' and operation.document.version_id in completed_versions and 'extract' not in stages:
                payload=next(a for a in engine._outputs(attempt) if a.kind=='download_payload')
                selected_sources.append(payload.path);source_versions[str(payload.path)]=operation.document.version_id
            if operation.stage=='extract' and operation.document.version_id in completed_versions:
                if operation.document.artifacts:source=operation.document.artifacts[0]
                else:
                    producer=next(edge[0] for edge in operation.inputs if edge[2]=='source_payload')
                    source=next(artifact for artifact in engine._outputs(result['attempts'][producer],include_deleted=True) if artifact.kind=='download_payload')
                selected_sources.append(source.path)
                source_versions[str(source.path)]=operation.document.version_id
            references=writer.connection.execute('SELECT sha256 FROM attempt_inputs WHERE attempt_id=? ORDER BY rowid',(attempt,)).fetchall()
            key=(operation.stage,operation.specification.operation,tuple(row[0] for row in references))
            receipt=next(artifact for artifact in engine._outputs(attempt) if artifact.kind=='worker_receipt')
            # Keep only a hash-bound reference at the corpus barrier. Full edit
            # receipts are read for one projection at a time, never prefetched.
            replays.setdefault(key,[]).append((operation.document.version_id,dict(operation.specification.settings),receipt,receipt.path.parent))
        result['sources']=[str(path) for path in selected_sources]
        failed_versions={task['version_id'] for task in writer.task_states(run) if task['status']=='failed' and task['version_id'] is not None}
        result['document_counts']={'selected':len(per_version),'included':len(completed_versions),
            'omitted':len(per_version)-len(completed_versions),'failed':len(failed_versions)}
        result['plan_hash']=writer.connection.execute('SELECT plan_sha256 FROM runs WHERE run_id=?',(run,)).fetchone()[0]
        if resume:
            settings=json.loads(writer.connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.settings_sha256 WHERE r.run_id=?',(run,)).fetchone()[0])
    corpus._streaming_source_versions=source_versions
    corpus._streaming_download_replays=download_replays if settings.get('download_projection_manifest') else None
    corpus._streaming_download_manifest=settings.get('download_projection_manifest')
    return result,replays


def _run_corpus(corpus,files=(),*,stages=('extract','clean','sections','export'),workers=None,settings=None,
               resume=None,retry_failed=False,plan=None,observer=None):
    """Execute document DAG, then declared compatibility publication barriers."""
    settings=settings or {}
    files=list(files)
    if resume and not retry_failed:
        writer=corpus._scheduler_writer
        row=writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' ORDER BY event_id DESC LIMIT 1",(resume,)).fetchone()
        digest=json.loads(row[0]).get('result_sha256') if row else None
        if digest:
            from ..ledger.encoding import snapshot_hash
            from .scheduler import execution_environment
            run=writer.connection.execute('SELECT * FROM runs WHERE run_id=?',(resume,)).fetchone()
            if snapshot_hash({'schema_version':1,'kind':'environment',**execution_environment()})!=run['environment_sha256']:
                raise ValueError('Frozen execution environment changed; create an amended run')
            saved=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()[0])
            for file_id in saved['retained_file_ids']:
                binding=writer.connection.execute('SELECT uri,sha256,bytes FROM files WHERE file_id=?',(file_id,)).fetchone()
                path=local_path(binding['uri'])
                if not path.is_file() or path.stat().st_size!=binding['bytes'] or sha256_file(path)!=binding['sha256']:
                    raise ValueError('Retained recovery frontier missing or corrupt; reacquire bound ancestors in an amended run')
            from .retention import RetentionController
            frozen=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(run['settings_sha256'],)).fetchone()[0])
            RetentionController(writer,corpus.output_dir,resume,frozen).recover()
            return saved['result']
    prerequisites={'clean':('extract',),'ocr':('extract','clean'),'sections':('clean','extract'),'annotate':('sections','clean','extract'),'export':('clean','extract')}
    stages=tuple(dict.fromkeys([dependency for stage in stages for dependency in (*prerequisites.get(stage,()),stage)]))
    if plan is not None:
        from ..ledger.encoding import snapshot_hash
        body=plan.get('data',plan)
        if snapshot_hash({key:value for key,value in body.items() if key!='plan_hash'})!=body.get('plan_hash'):
            raise ValueError('Supplied plan hash does not match')
        for document in body['documents']:
            observation=document.get('source_observation',{})
            uri=observation.get('source_uri',document['source_key'].split(':v1:',1)[1])
            if uri.startswith('file:'):
                path=local_path(uri)
                if not path.is_file() or (observation.get('bytes') is not None and
                    (path.stat().st_size,path.stat().st_mtime_ns)!=(observation['bytes'],observation['mtime_ns'])):
                    raise ValueError('Supplied plan source inventory changed')
                if observation.get('stored_sha256') and sha256_file(path)!=observation['stored_sha256']:
                    raise ValueError('Supplied plan source hash changed')
        files=[document['source_key'].split(':v1:',1)[1] for document in body['documents']]
        stages=tuple(dict.fromkeys(step['stage'] for document in body['documents'] for step in document['steps']))
        workers={pool['stage']:pool['workers'] for pool in body['pools']}
        settings=body['settings']
        if 'ocr' in stages:
            pool=next(pool for pool in body['pools'] if pool['stage']=='ocr')
            settings={**settings,'ocr':{**settings.get('ocr',{}),'device_id':pool['device_id']}}
        settings={**settings,'planned_document_ids':{document['source_key']:document['document_id'] for document in body['documents']}}
        if body.get('explicit_manifest') is not None:
            settings={**settings,'source_manifest':body['explicit_manifest']}
    if resume and (files or workers or settings):
        raise ValueError('Resume restores its frozen plan; semantic overrides are forbidden')
    if 'math' in stages:
        raise ValueError('Math requires an explicit operation DAG')
    import inspect
    extraction_options=settings.get('extract',{})
    semantic_options=set(extraction_settings()) & set(inspect.signature(corpus.extract).parameters)
    if any(name not in semantic_options for name in extraction_options):
        raise ValueError('Extraction setting requires an explicit operation DAG')
    if int(extraction_options.get('num_threads') or 1)!=1:
        raise ValueError('Native thread overrides require explicit CPU-token declarations in an operation DAG')
    publication_stages=[stage for stage in ('extract','clean','ocr','sections','annotate','export') if stage in stages]
    if not resume and any(str(source).startswith(('http://','https://')) for source in files):
        publication_stages.insert(0,'download')
    if not resume:
        import os
        semantic_files=[Path(corpus.section_classifier_model_path),Path(corpus.extraction_model_path)]
        if os.environ.get('GLOSSAPI_GREEK_WORDLIST'):semantic_files.append(Path(os.environ['GLOSSAPI_GREEK_WORDLIST']))
        settings={**settings,'semantic_files':[{'path':str(path.resolve()),'sha256':sha256_file(path)} for path in semantic_files if path.is_file()]}
    result,replays=prepare_streaming(corpus,files,workers=workers,stages=stages,
        settings={**settings,'publication_barriers':publication_stages},
        resume=resume,retry_failed=retry_failed,observer=observer,writer=corpus._scheduler_writer)
    if result['state'] not in ('completed','completed_with_failures'):
        return result
    corpus._streaming_ocr_replays=replays
    if resume:
        from ..ledger.writer import open_readonly
        with open_readonly(Path(corpus.output_dir)/'corpus.sqlite3') as connection:
            row=connection.execute('SELECT settings_sha256 FROM runs WHERE run_id=?',(resume,)).fetchone()
            settings=json.loads(connection.execute('SELECT json FROM snapshots WHERE sha256=?',row).fetchone()[0])
            stages=settings['requested_stages']
    calls={
        # A resumed run reads redacted credential markers; the publication replays
        # committed acquisitions and never receives a placeholder as a credential.
        'download':lambda:corpus.download(corpus._streaming_download_manifest['path'],**{key:value for key,value in settings.get('download',{}).items()
            if value!=__import__('glossapi.ledger.journal',fromlist=['REDACTED']).REDACTED}),
        'extract':lambda:corpus.extract(**{'input_format':'pdf','file_paths':result['sources'],'phase1_backend':'safe',
            'accel_type':'CPU','num_threads':1,'skip_existing':False,'batched':False,'repair_fonts':True,**settings.get('extract',{})}),
        'clean':lambda:corpus.clean(num_threads=1),
        'ocr':lambda:publish_ocr_results(corpus),
        'sections':corpus.section,'annotate':corpus.annotate,
        'export':lambda:corpus.jsonl(Path(corpus.output_dir)/'export.jsonl')}
    if not result['sources'] and 'export' in stages:
        metadata=Path(corpus.output_dir)/'download_results/download_results.parquet'
        if not metadata.exists():
            import pandas as pd
            from ..parquet_schema import ParquetSchema
            ParquetSchema().write_metadata_parquet(pd.DataFrame({'filename':pd.Series(dtype='str')}),metadata)
        writer=corpus._scheduler_writer
        with writer.transaction():
            for version, in writer.connection.execute('SELECT DISTINCT version_id FROM tasks WHERE run_id=? AND version_id IS NOT NULL',(result['run_id'],)).fetchall():
                writer.event(result['run_id'],event_kind='decision',reason_code='export_canonical_markdown_missing',version_id=version,
                    details={'product':'document_jsonl','included':False,'decision':'omit','cause':'upstream_failed','empty_selected_frontier':True})
    from .disk_budget import DiskBudget
    # One whole-run disk observer for the publication barriers; it starts from
    # the scheduler's persisted peak and observes before each retention step.
    publication_budget=DiskBudget(corpus._scheduler_writer,corpus.output_dir,result['run_id'],getattr(corpus,'_retention_settings',settings))
    with committed_results(replays):
        for stage in ('download','extract','clean','ocr','sections','annotate','export'):
            if stage in stages or (stage=='download' and corpus._streaming_download_manifest):
                published=False
                if not result['sources'] and stage not in ('download','export'):
                    skip_barrier(corpus,result['run_id'],stage)
                else:
                    published=publish_barrier(corpus,result['run_id'],stage,calls[stage],retry_failed=retry_failed,
                        empty_export=not result['sources'] and stage=='export')
                # The next compatibility reader has now consumed these private
                # producer records. Retire at that frontier, not at run close.
                released_stage = {'clean':'extract','sections':'clean','annotate':'sections'}.get(stage)
                if released_stage and published:
                    publication_budget.observe()
                    from .retention import retain_completed_run
                    attempts = [attempt for key,attempt in result['attempts'].items()
                        if key.endswith(':'+released_stage)]
                    retain_completed_run(corpus._scheduler_writer,corpus.output_dir,result['run_id'],
                        getattr(corpus,'_retention_settings',settings),attempts=attempts,
                        observer=getattr(corpus,'_retention_observer',None))
    from .retention import retain_completed_run
    writer=corpus._scheduler_writer
    # Every wrapper generation is a child of this fixed publication plan.
    wrapper_runs=[json.loads(row[0])['wrapper_run_id'] for row in writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' AND message LIKE 'wrapper:%'",(result['run_id'],))]
    all_attempts=list(result['attempts'].values())
    for wrapper in wrapper_runs:
        all_attempts.extend(row[0] for row in writer.connection.execute('SELECT attempt_id FROM attempts a JOIN tasks t USING(task_id) WHERE t.run_id=?',(wrapper,)))
    publication_states=[task['status'] for task in writer.task_states(result['run_id']) if task['operation'].endswith('.compatibility_publish')]
    publication_budget.observe()
    if all(status=='ok' for status in publication_states):
        retain_completed_run(writer,corpus.output_dir,result['run_id'],getattr(corpus,'_retention_settings',settings),attempts=all_attempts,observer=getattr(corpus,"_retention_observer",None))
    from .retention import retention_summary
    # Current present/free bytes after retention; the high-water keeps the
    # scheduler and publication peaks instead of a fresh zero (LS7-G04).
    result['disk']=publication_budget.report()
    result['disk']['deleted_logical_bytes']=retention_summary(writer.connection)['deleted_logical_bytes']
    result['outputs']=[]
    output=Path(corpus.output_dir)/'export.jsonl'
    if output.is_file():
        result['outputs'].append({'kind':'jsonl','path':str(output),'sha256':sha256_file(output),'bytes':output.stat().st_size})
    from collections import Counter
    counts=Counter(task['status'] for task in writer.task_states(result['run_id']))
    result['counts']={status:counts[status] for status in ('planned','running','ok','skipped','failed')}
    result['state']='completed_with_failures' if counts['failed'] else 'completed'
    with corpus._scheduler_writer.transaction():
        corpus._scheduler_writer.event(result['run_id'],event_kind='run',reason_code='legacy_observed',details={'state':result['state'],
            'disk_high_water_bytes':result['disk']['high_water_bytes'],'disk_high_water_scope':'run','result_sha256':corpus._scheduler_writer.snapshot('retained_run_result',{
                'result':{key:value for key,value in result.items() if key!='trace'},
                'retained_file_ids':[row['file_id'] for row in writer.file_catalog().values() if row['presence']=='present'
                    and row['ownership'] in ('managed','initial_external') and local_path(row['uri']).is_file()]})})
    return result


@shadow_stage('ocr', 'ocr.wrapper')
def publish_ocr_results(self):
    """Publish committed OCR bytes at the existing single-writer barrier."""
    from .workers import read_result
    from .compatibility import materialize_stage_outputs
    from ..corpus.ocr.artifacts import persist_ocr_success, refresh_cleaner_after_ocr
    corpus=self
    replays=corpus._streaming_ocr_replays
    filenames=[]
    for (stage,operation,hashes), candidates in replays.items():
        if stage != 'ocr':
            continue
        for version,settings,receipt,workspace in candidates:
            result=read_result(receipt.path)
            if result.reason_code != 'ocr_completed':
                continue
            paths={output.relative_path:corpus.output_dir/output.relative_path for output in result.outputs
                if output.kind in ('ocr_markdown','ocr_metrics','ocr_native','ocr_page_state')}
            materialize_stage_outputs(corpus,result,workspace,paths)
            markdown=next(output for output in result.outputs if output.kind=='ocr_markdown')
            filenames.append(Path(markdown.relative_path).with_suffix('.pdf').name)
    if filenames:
        persist_ocr_success(corpus,filenames=filenames,backend_norm='docling')
        refresh_cleaner_after_ocr(corpus)


def skip_barrier(corpus,run,stage):
    writer=corpus._scheduler_writer
    task=next((task for task in writer.task_states(run) if task['operation']==stage+'.compatibility_publish'),None)
    if task['status'] in ('ok','skipped','failed'):return
    with writer.transaction():
        attempt=writer.attempt(task['task_id'],worker_id='single-publisher',device_id='cpu',fingerprint={'reuse_eligible':False},inputs={'files':[]},settings={},tools={})
        writer.finish(run,task['task_id'],attempt,status='skipped',reason_code='upstream_failed')


def publish_barrier(corpus,run,stage,callback,*,retry_failed=False,empty_export=False):
    """Recover a wrapper commit even if death precedes its barrier commit."""
    writer=corpus._scheduler_writer
    task=next((task for task in writer.task_states(run) if task['operation']==stage+'.compatibility_publish'),None)
    if task is None:
        raise ValueError('Publication barrier is absent from the frozen plan')
    if task['status'] in ('failed','skipped') and not retry_failed:return False
    parents=[writer.task_state(row[0]) for row in writer.connection.execute('SELECT producer_task_id FROM task_dependencies WHERE task_id=?',(task['task_id'],))]
    unavailable=any(parent['operation'].endswith('.compatibility_publish') and parent['stage']!='download' and parent['status']!='ok' for parent in parents)
    if unavailable and not empty_export:
        skip_barrier(corpus,run,stage);return False
    previous=writer.latest_attempt(task['task_id'])
    wrapper=writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' AND message=? ORDER BY event_id DESC LIMIT 1",(run,'wrapper:'+stage)).fetchone()
    wrapper_run=json.loads(wrapper[0])['wrapper_run_id'] if wrapper else None
    wrapper_ok=False
    if wrapper_run:
        wrapper_operations=(stage+'.wrapper',stage+'.clean' if stage=='clean' else stage+'.jsonl' if stage=='export' else stage+'.extract')
        wrapper_ok=any(task['scope_key'].startswith('corpus:') and task['status']=='ok' and task['operation'] in wrapper_operations for task in writer.task_states(wrapper_run))
    if task['status']=='ok':
        for row in writer.connection.execute('SELECT f.uri,i.sha256 FROM attempt_inputs i JOIN files f USING(file_id) WHERE attempt_id=?',(previous['attempt_id'],)):
            path=local_path(row['uri'])
            state=writer.connection.execute('SELECT file_id,presence FROM current_file_state WHERE uri=? AND sha256=? ORDER BY event_id DESC LIMIT 1',(row['uri'],row['sha256'])).fetchone()
            if state and state['presence']=='deleted':
                from .retention import retention_facts
                if any(value['reason_code'] in ('delete_completed','delete_recovered') for value in retention_facts(writer.connection,state['file_id'])['deletions']):continue
            if not path.is_file() or sha256_file(path)!=row['sha256']:
                raise ValueError('Committed publication is missing or corrupt')
        return True
    if previous and task['status']=='running' and not wrapper_ok:
        with writer.transaction():
            writer.finish(run,task['task_id'],previous['attempt_id'],status='failed',reason_code='attempt_interrupted')
    if not (previous and task['status']=='running' and wrapper_ok):
        with writer.transaction():
            attempt=writer.attempt(task['task_id'],worker_id='single-publisher',device_id='cpu',
                fingerprint={'frozen_run_id':run,'stage':stage,'reuse_eligible':False},inputs={'files':[]},settings={},tools={})
            writer.event(run,event_kind='lifecycle',reason_code='stage_started',status='running',task_id=task['task_id'],attempt_id=attempt)
        corpus._scheduler_publication={'run_id':run,'task_id':task['task_id']}
        try:
            callback()
        except Exception as error:
            import traceback
            with writer.transaction():
                writer.finish(run,task['task_id'],attempt,status='failed',reason_code=stage+'_exception',
                    error_type=type(error).__name__,message=str(error),traceback=''.join(traceback.format_exception(error)))
            return False
        finally:
            corpus._scheduler_publication=None
        wrapper=writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' AND message=? ORDER BY event_id DESC LIMIT 1",(run,'wrapper:'+stage)).fetchone()
        wrapper_run=json.loads(wrapper[0])['wrapper_run_id']
    else:
        attempt=previous['attempt_id']
    outputs=writer.connection.execute("SELECT f.* FROM files f JOIN attempts a ON a.attempt_id=f.producer_attempt_id JOIN tasks t ON t.task_id=a.task_id WHERE t.run_id=? AND f.uri LIKE ?",(wrapper_run,(Path(corpus.output_dir).resolve()/'.glossapi/artifacts').as_uri()+'/%')).fetchall()
    wrapper_operation={'download':'download.wrapper','extract':'extract.wrapper','clean':'clean.clean','ocr':'ocr.wrapper','sections':'sections.wrapper','annotate':'annotate.wrapper','export':'export.jsonl'}[stage]
    outcome=next((task for task in writer.task_states(wrapper_run) if task['operation']==wrapper_operation),None)
    with writer.transaction():
        for index,row in enumerate(outputs):
            path=local_path(row['uri'])
            if not path.is_file():
                raise ValueError('Publication artifact is absent')
            writer.input(attempt,row['file_id'],'published:'+str(index),row['sha256'])
        if stage=='extract':
            for source in writer.connection.execute("SELECT DISTINCT i.file_id,i.sha256 FROM attempt_inputs i JOIN attempts a USING(attempt_id) JOIN tasks t USING(task_id) JOIN files f USING(file_id) WHERE t.run_id=? AND f.kind IN ('download_payload','source_payload','initial_input')",(wrapper_run,)):
                writer.input(attempt,source['file_id'],'source_reader:'+source['file_id'],source['sha256'])
        status='ok' if outcome and outcome['status']=='ok' else 'failed'
        writer.finish(run,task['task_id'],attempt,status=status,reason_code=stage+'_completed' if status=='ok' else stage+'_exception',
            details={'wrapper_run_id':wrapper_run,'bound_product_files':len(outputs),'recovered_wrapper_commit':wrapper_ok})
    if status=='ok' and stage=='extract':
        from .retention import retire_downloaded_sources
        retire_downloaded_sources(writer,corpus.output_dir,run,getattr(corpus,'_retention_settings',{}))
    return status=='ok'


def run_corpus(corpus,*args,**options):
    """Hold corpus exclusion across document work and every publication barrier."""
    began=time.monotonic()
    with LedgerWriter(Path(corpus.output_dir)/'corpus.sqlite3') as writer:
        corpus._scheduler_writer=writer
        try:
            result=_run_corpus(corpus,*args,**options)
            result['total_wall_seconds']=time.monotonic()-began
            return result
        finally:
            corpus._scheduler_writer=None
            corpus._streaming_download_replays=None
            corpus._streaming_ocr_replays=None
