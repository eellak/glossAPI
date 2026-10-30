"""OCR through the configured external Docling entry point, then reassembly."""
from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import time

from .contracts import PackageRequirement, PlannedOperation, StagePlan, StageSpecification, StageResult
from .execution import failed_result, freeze_inputs, outputs_in, sha256_file
from .receipts import capture_backend_observations


class OcrStage:
    name = 'ocr'
    contract_version = 2

    def __init__(self, ocr_options=None):
        self.ocr_options = ocr_options

    def packages(self, specification):
        from ..ocr.defaults import package_requirements
        return (PackageRequirement('layout', ('docling==2.81.0',)),
            PackageRequirement('ocr', package_requirements() if specification.settings.get('entry_point_name','glossapi')=='glossapi' else (),
                'docling', specification.settings.get('entry_point_name', 'glossapi')))

    def plan(self, documents, *, settings):
        return StagePlan(tuple(PlannedOperation(document.document_id, document.version_id,
            StageSpecification('ocr.document', ('source_payload','native_markdown','ocr_page_state','docling_json','page_quality'),
                ('ocr_markdown','ocr_metrics'), str(settings.get('device_id') or settings.get('options',{}).get('device') or 'cpu'), settings)) for document in documents))

    def finalize(self, request, *, reporter):
        raise ValueError('OCR has no aggregate finalize operation')

    def run_document(self, request, *, reporter):
        # Compatibility executes here; streaming uses bounded device workers.
        return self._run_document(request, reporter=reporter)

    def _run_document(self, request, *, reporter):
        request.work_directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic_ns()
        reason = 'ocr_configuration_invalid'
        observations = []
        try:
            from ..ocr.defaults import validate_device
            validate_device('ocr',request.device_id,request.settings)
            # DAG edges retain producer kinds; normalize their declared OCR roles.
            aliases={'download_payload':'source_payload','raw_markdown':'native_markdown'}
            request=replace(request,inputs=tuple(replace(item,kind=aliases.get(item.kind,item.kind)) for item in request.inputs))
            from ..corpus.ocr import dispatch
            plugins = [point for point in dispatch.entry_points(group='docling') if point.name != 'docling']
            if not plugins:
                reason = 'ocr_plugin_missing'
                from ..ocr.defaults import INSTALL_MESSAGE
                raise RuntimeError(INSTALL_MESSAGE)
            options = self.ocr_options
            selected_plugin = request.settings.get('entry_point_name', 'glossapi' if options is None else None)
            if selected_plugin and selected_plugin not in [point.name for point in plugins]:
                reason = 'ocr_plugin_missing'
                from ..ocr.defaults import INSTALL_MESSAGE
                raise RuntimeError(INSTALL_MESSAGE)
            if request.device_id != 'cpu' and selected_plugin != 'glossapi' and getattr(options,'kind',None) != 'glossapi':
                reason = 'ocr_backend_unavailable'
                raise ValueError('Accelerated OCR requires the Greek OCR plugin')
            if options is not None and getattr(options,'device',request.device_id) != request.device_id:
                raise ValueError('OCR option device must match its scheduler device pool')
            if options is None and (request.settings.get('options') or {}).get('kind','glossapi') != 'glossapi':
                # Engine selection is validated before routing/early return, without loading models.
                raise ValueError('Configured OCR kind does not match the Greek OCR engine; pass external options to Corpus.ocr()')
            source = next((item for item in request.inputs if item.kind == 'source_payload'), None)
            if source is None:
                raise ValueError('OCR requires an explicit original PDF')
            prior_state = next((item for item in request.inputs if item.kind == 'ocr_page_state'), None)
            if prior_state and request.settings.get('prior_state_source_sha256') != source.sha256:
                reason = 'ocr_page_state_invalid'
                raise ValueError('Prior OCR state is not bound to this source hash')
            originals = request.inputs
            request = freeze_inputs(request)
            root = request.work_directory
            source = next(item for item in request.inputs if item.kind == 'source_payload')
            ranges = request.settings.get('page_ranges', [])
            quality = next((item for item in request.inputs if item.kind == 'page_quality'), None)
            report = None
            if quality or (options is None and not ranges):
                if quality:
                    report = json.loads(quality.path.read_text())
                else:
                    from ..page_quality import page_quality
                    report = page_quality(source.path,output_dir=root).to_dict()
                selected_pages = [int(page['page']) for page in report['pages'] if page['page_needs_ocr'] is True]
                ranges = [[page,page] for page in selected_pages]
                if not selected_pages:
                    native = next(item for item in request.inputs if item.kind == 'native_markdown')
                    target = root/'markdown'/source.path.with_suffix('.md').name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(native.path.read_bytes())
                    metrics = root/'json/metrics'/f'{source.path.stem}.metrics.json'
                    metrics.parent.mkdir(parents=True,exist_ok=True)
                    metrics.write_text(json.dumps({'page_count':report['pages_total'],'ocr_pages':[],
                        'ocr_engine':getattr(options,'kind','glossapi') if options is not None else 'glossapi'}))
                    return StageResult(request.task_id,request.attempt_id,request.lease_token,'ok','ocr_not_routed',
                        tuple(replace(item,kind='ocr_markdown' if item.relative_path.startswith('markdown/') else 'ocr_metrics',required=True)
                              for item in outputs_in(root) if item.relative_path.startswith(('markdown/','json/metrics/'))),
                        ({'operation':request.operation,'pages':[],'needs_ocr':False},), {'wall_ns':time.monotonic_ns()-started})
            destinations = {'native_markdown': root/'markdown'/source.path.with_suffix('.md').name,
                'ocr_native': root/'sidecars/ocr_native'/source.path.with_suffix('.md').name,
                'ocr_page_state': root/'sidecars/ocr_pages'/source.path.with_suffix('.json').name,
                'docling_json': root/'json'/f'{source.path.stem}.docling.json'}
            import shutil
            for artifact in request.inputs:
                destination = destinations.get(artifact.kind)
                if destination:
                    if artifact.kind == 'docling_json' and artifact.path.name.endswith('.zst'):
                        destination = destination.with_suffix(destination.suffix+'.zst')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(artifact.path, destination)
            native_path = root/'sidecars/ocr_native'/source.path.with_suffix('.md').name
            supplied_native = root/'markdown'/source.path.with_suffix('.md').name
            if not native_path.exists() and supplied_native.is_file() and supplied_native.read_text().strip():
                native_path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(supplied_native,native_path)
            context = SimpleNamespace(output_dir=root, input_dir=source.path.parent,
                logger=logging.getLogger(__name__), _inside_ocr_stage=True, _ocr_cache_root=request.settings.get('cache_root'))
            if options is None:
                from ..ocr.defaults import greek_options
                options = greek_options(request.settings.get('options'), device=request.device_id,
                                        routing_sidecar=root/'routing.jsonl')
            from ..corpus.ocr.config import OcrRequest
            configuration = OcrRequest('ocr_bad','docling',options,request.device_id,request.settings.get('max_pages'),
                False,None,8,220,'single',None,True,False,num_threads=int(request.settings.get('num_threads',1)))
            ranges = [f'{source.path.name}:{int(first)}:{int(last)}' for first,last in ranges]
            reason = 'ocr_plugin_conversion_failed'
            with capture_backend_observations() as observations:
                dispatch.run_plugin_ocr(context,[source.path.name], request=configuration, page_ranges=ranges or None)
            canonical = root/'markdown'/source.path.with_suffix('.md').name
            metrics_path = root/'json/metrics'/f'{source.path.stem}.metrics.json'
            if not canonical.is_file() or not metrics_path.is_file():
                raise ValueError('Required canonical OCR output/metrics missing')
            metrics = json.loads(metrics_path.read_text())
            count = int(metrics['page_count'])
            selected = ({int(page['page']) for page in report['pages'] if page['page_needs_ocr'] is True} if report is not None else
                {page for first,last in (request.settings.get('page_ranges') or [(1,count)]) for page in range(int(first), int(last)+1)})
            if configuration.max_pages is not None:
                selected = {page for page in selected if page <= configuration.max_pages}
            if not selected:
                reason = 'ocr_page_selection_invalid'; raise ValueError('OCR selection is empty')
            state_path=root/'sidecars/ocr_pages'/source.path.with_suffix('.json').name
            if selected == set(range(1,count+1)):
                from ..ocr.page_artifacts import _split_page_outputs
                import uuid
                run_id=uuid.uuid4().hex
                state_path.parent.mkdir(parents=True,exist_ok=True)
                state_path.write_text(json.dumps({'page_count':count,'pages':{
                    str(page):{'text':text,'run_id':run_id} for page,text in enumerate(_split_page_outputs(canonical.read_text()),1)}}),encoding='utf-8')
            if time.monotonic_ns() > request.deadline_monotonic_ns:
                raise TimeoutError('OCR deadline exceeded')
            bindings = {item.kind: {'file_id': item.file_id, 'sha256': item.sha256,
                'producer_attempt_id': request.settings.get('input_producers',{}).get(item.file_id)} for item in originals}
            call_log=root/'sidecars/ocr_runtime/calls.jsonl'
            engine_calls={int(row['page']):row for row in map(json.loads,call_log.read_text().splitlines())} if call_log.is_file() else {}
            engine_receipt_hash=sha256_file(call_log) if call_log.is_file() else None
            pages = [{'page': page, 'action': 'ocr' if page in selected else 'retained_native_or_prior_ocr',
                'source_file_id': bindings['source_payload']['file_id'] if page in selected else
                    (bindings.get('ocr_page_state') or bindings.get('ocr_native') or bindings.get('native_markdown') or {}).get('file_id'),
                'source_hash': bindings['source_payload']['sha256'] if page in selected else
                    (bindings.get('ocr_page_state') or bindings.get('ocr_native') or bindings.get('native_markdown') or {}).get('sha256'),
                'producer_attempt_id': request.attempt_id if page in selected else
                    (bindings.get('ocr_page_state') or bindings.get('ocr_native') or bindings.get('native_markdown') or {}).get('producer_attempt_id'),
                'range': [page,page], 'composition_mode': metrics.get('kept_markdown_source'),
                'status': 'produced' if page in selected else 'retained', 'drop_reason': None if page in engine_calls else 'unknown_not_reported',
                'engine_reason': engine_calls.get(page,{}).get('reason',engine_calls.get(page,{}).get('status','unknown_not_reported')),
                'engine_receipt_hash': engine_receipt_hash if page in engine_calls else None} for page in range(1,count+1)]
            outputs = []
            for item in outputs_in(root):
                kind = ('ocr_markdown' if item.relative_path == str(canonical.relative_to(root)) else
                    'ocr_metrics' if item.relative_path == str(metrics_path.relative_to(root)) else
                    'ocr_native' if item.relative_path.startswith('sidecars/ocr_native/') else
                    'ocr_page_state' if item.relative_path.startswith('sidecars/ocr_pages/') else item.kind)
                outputs.append(replace(item, kind=kind, required=kind in ('ocr_markdown','ocr_metrics'),
                    retention_class='recovery' if kind in ('ocr_native','ocr_page_state') else item.retention_class))
            return StageResult(request.task_id,request.attempt_id,request.lease_token,'ok','ocr_completed',tuple(outputs),
                ({'operation':request.operation,'inputs':bindings,'pages':pages,'output_sha256':sha256_file(canonical),
                  'plugin': {'kind':options.kind,'entry_points':[point.name for point in plugins],
                    **(json.loads((root/'sidecars/ocr_runtime/provenance.json').read_text())
                       if (root/'sidecars/ocr_runtime/provenance.json').is_file() else {'models':request.tools.get('models',[])}),
                    'requested_device':request.settings.get('requested_device',request.device_id),
                    'device':request.device_id,'detector_device':getattr(options,'detector_device',None)},
                  'backend_observations':observations},),
                {'wall_ns':time.monotonic_ns()-started,'execution':'in_process_plugin'})
        except Exception as error:
            if isinstance(error, FileNotFoundError) and 'native Markdown' in str(error): reason='ocr_native_missing'
            elif isinstance(error, ValueError) and ('page range' in str(error).lower() or 'max_pages' in str(error)): reason='ocr_page_selection_invalid'
            elif 'Overlapping OCR shards' in str(error): reason='ocr_shards_overlap'
            elif 'Incomplete selected OCR pages' in str(error): reason='ocr_shards_incomplete'
            return failed_result(request,'input_hash_changed' if str(error)=='input_hash_changed' else reason,error,
                decisions=({'backend_observations':observations},),metrics={'wall_ns':time.monotonic_ns()-started})
