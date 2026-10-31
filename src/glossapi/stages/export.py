"""Explicit export inputs, existing record serializer and generation membership."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

from .contracts import PackageRequirement, StagePlan, StageSpecification
from .execution import failed_result, freeze_inputs, sha256_file
from .products import execute_aggregate, generation_outputs, publish_products, resolved_input
from .receipts import capture_backend_observations


class ExportStage:
    name='export'
    contract_version=1

    def __init__(self,corpus=None):self.corpus=corpus

    def packages(self,specification):
        return (PackageRequirement('core',('pandas>=3.0.0','pyarrow>=26,<27','zstandard>=0.22.0')),)

    def plan(self,documents,*,settings):
        return StagePlan((),('export_order_and_shard_membership',))

    def run_document(self,request,*,reporter):
        return self.finalize(request,reporter=reporter)

    def finalize(self,request,*,reporter):
        request.work_directory.mkdir(parents=True,exist_ok=True)
        # The legacy serializer reads a corpus-shaped view of the frozen inputs. The view is a throwaway copy beside
        # the attempt directory, not inside it: inside, every view file became a second registered generation file
        # (byte-identical to its frozen input snapshot), hashed, sealed and released again (K4(c) part 2).
        view_root=request.work_directory.with_name(request.work_directory.name+'.export-view')
        shutil.rmtree(view_root,ignore_errors=True)
        try:
            return self._finalize(request,view_root)
        finally:
            shutil.rmtree(view_root,ignore_errors=True)

    def _finalize(self,request,view_root):
        try:
            request=freeze_inputs(request)
            from ..corpus.phase_export import ExportPhaseMixin
            view=copy.copy(self.corpus);view._inside_product_stage=True
            view.output_dir=view_root
            view.markdown_dir=view.output_dir/'markdown'
            view.cleaned_markdown_dir=view.output_dir/'clean_markdown'
            view.markdown_dir.mkdir(parents=True);view.cleaned_markdown_dir.mkdir()
            for source,relative in zip(request.inputs,request.settings['input_paths']):
                destination=view.output_dir/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source.path,destination)
            options=dict(request.settings['options'])
            options['metadata_path']=view.output_dir/'metadata.parquet'
            if options.get('source_metadata_path') is not None:options['source_metadata_path']=view.output_dir/'source_metadata.parquet'
            with capture_backend_observations() as observations:
                if request.settings['sharded']:
                    produced=ExportPhaseMixin.jsonl_sharded.__wrapped__(view,request.work_directory/'products',**options)
                else:
                    produced=[ExportPhaseMixin.jsonl.__wrapped__(view,request.work_directory/'products/export.jsonl',**options)]
            membership=[]
            for path in produced:
                if path.suffix=='.zst':
                    import zstandard
                    with path.open('rb') as stream,zstandard.ZstdDecompressor().stream_reader(stream) as reader:payload=reader.read()
                else:payload=path.read_bytes()
                records=[json.loads(line) for line in payload.splitlines()]
                membership.append({'path':str(path.relative_to(request.work_directory)),
                    'sha256':sha256_file(path),'bytes':path.stat().st_size,
                    'records':[{'filename':record['filename'],'doc_id':record['doc_id'],
                        'record_sha256':hashlib.sha256((json.dumps(record,ensure_ascii=False)+'\n').encode()).hexdigest()} for record in records]})
            included={record['filename'] for shard in membership for record in shard['records']}
            import pandas as pd
            metadata=pd.read_parquet(view.output_dir/'metadata.parquet')
            selected=[]
            metadata_decisions=[]
            for row in metadata.itertuples():
                filename=str(getattr(row,'filename',''))
                stem=__import__('re').sub(r'__p\d{4}-\d{4}$','',Path(filename).stem)
                metadata_decisions.append({'compatibility_filename':filename,'included':stem in included,
                    'reason_code':'export_included' if stem in included else 'export_canonical_markdown_missing'})
            observed_versions=set()
            for source,relative in zip(request.inputs,request.settings['input_paths']):
                if source.kind!='selected_text':continue
                from .._naming import canonical_stem
                stem=__import__('re').sub(r'__p\d{4}-\d{4}$','',canonical_stem(Path(relative)))
                canonical=Path(relative).parent==Path('clean_markdown')
                present=canonical and stem in included
                identity=request.settings.get('input_versions',{}).get(source.file_id,{'version_id':None,'document_id':None})
                if identity['version_id']:observed_versions.add(identity['version_id'])
                selected.append({**identity,'input_file_id':source.file_id,'input_sha256':source.sha256,'included':present,
                    'reason_code':'export_included' if present else 'export_canonical_markdown_missing' if not canonical else 'export_metadata_row_missing',
                    'identity_basis':'explicit_selected_text_artifact','compatibility_stem':stem})
            for identity in request.settings.get('selected_sources',[]):
                if identity['version_id'] not in observed_versions:
                    selected.append({**identity,'included':False,'reason_code':'export_canonical_markdown_missing','identity_basis':'explicit_wrapper_source_inventory'})
            paths=[(str(path.relative_to(request.work_directory)),'document_jsonl_shard' if request.settings['sharded'] else 'document_jsonl') for path in produced]
            # The view is private scratch; retain input snapshots and products,
            # not a second complete copy of the same text/metadata payloads.
            # The writer seals and retires view copies after aggregate success.
            return generation_outputs(request,paths,reason='export_completed' if included else 'export_empty',
                decisions=({'selected_versions':selected,'metadata_decisions':metadata_decisions,'membership':membership,'backend_observations':observations,
                    'input_precedence':'legacy metrics then per-page metrics; current fresh math map; canonical text only'},),
                metrics={'shards':membership,'records':sum(len(s['records']) for s in membership)})
        except Exception as error:
            return failed_result(request,'export_exception',error)


def export_products(corpus,output,options,*,sharded=False):
    options=dict(options)
    for key in ('metadata_fields','source_metadata_fields'):
        if options.get(key) is not None:options[key]=list(options[key])
    # Preserve the public validation order and exception types before staging.
    if sharded:
        if options['shard_size_bytes']<=0:raise ValueError('shard_size_bytes must be positive')
        if options['compression'].lower() not in ('none','zstd'):raise ValueError(f"Unsupported compression codec '{options['compression'].lower()}'")
    else:
        if options.get('source_metadata_key') and options.get('source_metadata_path') is None:raise ValueError('source_metadata_path must be provided when source_metadata_key is set')
        if options.get('source_metadata_key') and options.get('source_metadata_fields') is None:raise ValueError('source_metadata_fields must be provided when source_metadata_key is set')
        if options.get('source_metadata_path') and not options.get('source_metadata_key'):raise ValueError('source_metadata_key must be provided when source_metadata_path is set')
        if options.get('source_metadata_fields') and not options.get('source_metadata_key'):raise ValueError('source_metadata_key must be provided when source_metadata_fields are set')
    metadata=options.get('metadata_path')
    if metadata is None:
        directory=corpus.output_dir/'download_results';metadata=directory/'download_results.parquet'
        if not metadata.exists():
            candidates=sorted(directory.glob('*.parquet')) if directory.exists() else []
            if not candidates:raise FileNotFoundError(f'Metadata parquet not found in {directory}')
            preferred=[p for p in candidates if p.name.startswith('download_results_')];metadata=preferred[0] if preferred else candidates[0]
    inputs=[('metadata',resolved_input(corpus,metadata))];relative_paths=['metadata.parquet']
    if options.get('source_metadata_path') is not None:
        inputs.append(('source_metadata',Path(options['source_metadata_path'])));relative_paths.append('source_metadata.parquet')
    root=corpus.cleaned_markdown_dir;chunks=root/'chunks'
    if not root.exists() or (not any(root.glob('*.md')) and not (chunks.exists() and any(chunks.rglob('*.md')))):
        root=corpus.markdown_dir;chunks=root/'chunks'
    for path in [*sorted(root.glob('*.md')),*sorted(chunks.rglob('*.md'))]:
        inputs.append(('selected_text',resolved_input(corpus,path)));relative_paths.append('clean_markdown/'+str(path.relative_to(root)))
    for directory,pattern,prefix in [(corpus.output_dir/'json/metrics','*.json','json/metrics'),(corpus.output_dir/'json','*.latex_map.jsonl','json')]:
        for path in sorted(directory.glob(pattern)):
            inputs.append(('metrics_or_math',resolved_input(corpus,path)));relative_paths.append(prefix+'/'+path.name)
    semantic_options={key:str(value) if isinstance(value,Path) else value for key,value in options.items()}
    result,workspace=execute_aggregate(corpus,ExportStage(corpus),inputs,
        StageSpecification('export.publish',tuple(role for role,path in inputs),('generation_manifest',),
            settings={'sharded':sharded,'options':semantic_options,'input_paths':relative_paths}))
    if result.status!='ok':
        # Match existing public error classes for malformed source/options.
        error_type=result.error.error_type.rsplit('.',1)[-1]
        raise {'KeyError':KeyError,'ValueError':ValueError,'FileNotFoundError':FileNotFoundError}.get(error_type,RuntimeError)(result.error.message)
    products={a.relative_path:Path(output)/Path(a.relative_path).name if sharded else Path(output)
        for a in result.outputs if a.kind in ('document_jsonl','document_jsonl_shard')}
    retire=[]
    if sharded:
        from .products import active_journal,local_path
        journal=active_journal()
        if journal:
            import re
            pattern=re.compile(re.escape(options['shard_prefix'])+r'-\d{6}\.jsonl(?:\.zst)?$')
            for uri, in journal.writer.connection.execute("SELECT DISTINCT uri FROM current_file_state WHERE stage='export' AND kind='document_jsonl_shard' AND ownership='managed' AND presence='present'"):
                path=local_path(uri)
                if path.parent==Path(output).absolute() and pattern.fullmatch(path.name) and path not in {p.absolute() for p in products.values()}:
                    retire.append(path)
    publish_products(corpus,result,workspace,products,retire=retire)
    return list(products.values()) if sharded else Path(output)
