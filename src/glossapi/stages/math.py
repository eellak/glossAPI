"""Existing math triage/inference kernels with fresh, hash-bound ROI maps."""
from __future__ import annotations

from dataclasses import replace, asdict
import hashlib
import json
import time

from .contracts import PackageRequirement, PlannedOperation, StagePlan, StageSpecification, StageResult
from .execution import failed_result, freeze_inputs, outputs_in, sha256_file
from .receipts import capture_backend_observations


class MathStage:
    name = 'math'
    contract_version = 1

    def packages(self, specification):
        return () if specification.operation == 'math.triage' else (PackageRequirement('layout', ('docling==2.81.0',)),)

    def plan(self, documents, *, settings):
        operation = settings.get('operation','math.infer')
        roles = ('extract_metrics',) if operation == 'math.triage' else ('source_payload','docling_json','selected_text')
        required = ('math_triage',) if operation == 'math.triage' else ('math_markdown','latex_map','math_metrics')
        return StagePlan(tuple(PlannedOperation(document.document_id,document.version_id,
            StageSpecification(operation,roles,required,str(settings.get('device_id','cpu')),settings)) for document in documents))

    def finalize(self, request, *, reporter):
        raise ValueError('Math publication is per document, not an aggregate operation')

    def run_document(self, request, *, reporter):
        request.work_directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic_ns()
        observations = []
        reason = 'math_exception'
        try:
            originals = request.inputs
            request = freeze_inputs(request)
            root = request.work_directory
            bindings = {item.kind: {'file_id':item.file_id,'sha256':item.sha256} for item in originals}
            if request.operation == 'math.triage':
                from ..ocr.utils.triage import summarize_math_density_from_metrics, recommend_phase
                if len(request.inputs) != 1:
                    raise ValueError('Triage requires one explicit metrics surface')
                summary = summarize_math_density_from_metrics(request.inputs[0].path)
                summary['formula_max_pp']=float(summary.get('formula_p90_pp',0.0))
                decision = {**summary, 'phase_recommended':recommend_phase(summary)}
                (root/'triage.json').write_text(json.dumps(decision,ensure_ascii=False))
                decision = {'operation':request.operation,'inputs':bindings,**decision}
                output_kinds = {'triage.json':'math_triage'}
                reason = 'math_triage_stop' if decision['phase_recommended']=='stop' else 'math_completed'
            elif request.operation == 'math.infer':
                source = next((item for item in request.inputs if item.kind=='source_payload'),None)
                document = next((item for item in request.inputs if item.kind=='docling_json'),None)
                if document is None:
                    reason='math_json_missing'; raise ValueError('Requested math requires Docling JSON')
                if source is None:
                    reason='math_pdf_missing'; raise ValueError('Requested math requires an original PDF')
                if request.device_id != 'cpu':
                    reason='math_devices_unavailable'; raise ValueError('GPU math remains unqualified in this sequential profile')
                from ..ocr.math import enrich_from_docling_json
                from ..text_sanitize import load_latex_policy
                policy = load_latex_policy()
                from ..ledger.encoding import canonical_json
                policy_hash = hashlib.sha256(canonical_json(asdict(policy)).encode()).hexdigest()
                markdown = root/'markdown'/source.path.with_suffix('.md').name
                markdown.parent.mkdir()
                latex_map = root/'json'/f'{source.path.stem}.latex_map.jsonl'
                latex_map.parent.mkdir(); latex_map.write_bytes(b'')
                reason = 'math_exception'
                with capture_backend_observations() as observations:
                    stats = enrich_from_docling_json(document.path,source.path,markdown,latex_map,
                        device=request.device_id,batch_size=int(request.settings.get('batch_size',8)),
                        dpi_base=int(request.settings.get('dpi_base',220)),targets=request.settings.get('targets'))
                if not markdown.is_file():
                    raise ValueError('Fresh math text missing')
                # Validate all fresh rows; an empty map is a successful zero result.
                rows = [json.loads(line) for line in latex_map.read_text().splitlines() if line.strip()]
                if len(rows) != int(stats['items']) or sum(float(row['accept_score'])>=1 for row in rows)!=int(stats['accepted']):
                    raise ValueError('Math map coverage does not match inference result')
                context = {'inputs':bindings,'latex_policy_sha256':policy_hash,
                    'models':request.tools.get('models',[]),
                    'model_identity_unavailable_reason':None if request.tools.get('models') else 'backend_not_reported'}
                context_hash = hashlib.sha256(canonical_json(context).encode()).hexdigest()
                regions = [{**item,'context_sha256':context_hash} for item in observations]
                # Only a real Docling CodeFormula model counts as inference. An
                # injected deterministic/test kernel, or a kernel that does not
                # report its backend, is recorded as such (LS4-G03).
                backend = dict(stats.get('backend') or {'module':None,'qualname':None,'real_inference':False,
                    'unavailable_reason':'backend_not_reported'})
                decision = {'operation':request.operation,**context,'context_sha256':context_hash,
                    'regions':regions,'stats':stats,'compile_status':'not_run',
                    'map_sha256':sha256_file(latex_map),'output_sha256':sha256_file(markdown),
                    'backend':backend,'inference_performed':bool(stats['items']) and backend.get('real_inference') is True}
                (root/'math.metrics.json').write_text(json.dumps(stats,ensure_ascii=False))
                output_kinds = {str(markdown.relative_to(root)):'math_markdown',
                    str(latex_map.relative_to(root)):'latex_map','math.metrics.json':'math_metrics'}
                reason = 'math_no_regions' if not stats['items'] else 'math_completed'
            else:
                raise ValueError('Unknown math operation')
            if time.monotonic_ns()>request.deadline_monotonic_ns:
                raise TimeoutError('Math deadline exceeded')
            outputs = tuple(replace(item,kind=output_kinds.get(item.relative_path,item.kind),
                required=item.relative_path in output_kinds) for item in outputs_in(root))
            return StageResult(request.task_id,request.attempt_id,request.lease_token,'ok',reason,outputs,
                (decision,),{'wall_ns':time.monotonic_ns()-started,'execution':'in_process_backend'})
        except Exception as error:
            unavailable = isinstance(error,ImportError)
            if unavailable: reason='math_backend_unavailable'
            decision = {'backend_observations':observations}
            if unavailable:
                decision.update(backend_status='unavailable',inference_performed=False)
            return failed_result(request,'input_hash_changed' if str(error)=='input_hash_changed' else reason,error,
                decisions=(decision,),metrics={'wall_ns':time.monotonic_ns()-started})
