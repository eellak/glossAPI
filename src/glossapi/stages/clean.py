"""Existing Rust and OCR text kernels, with private inputs and producer edits.

The native cleaner runs in process; no interpreter/model startup per document.
Corpus-level metadata serialization and selection remain compatibility work.
"""
from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
import time

from .contracts import PackageRequirement, PlannedOperation, StagePlan, StageResult, StageSpecification, OutputArtifact
from .execution import failed_result, freeze_inputs, outputs_in, sha256_file
from .edits import EditRecorder, text_hash, capture_ocr_edits
from .receipts import capture_backend_observations


class CleanStage:
    name = 'clean'
    contract_version = 1

    def __init__(self, selection=None):
        self.selection = selection

    def packages(self, specification):
        return (PackageRequirement('core', ('pandas>=1.3.0', 'pyarrow>=26,<27')),)

    def plan(self, documents, *, settings):
        operation = settings.get('operation', 'clean.initial')
        required = ('clean_metrics',) if operation == 'clean.score_after_ocr' else ('clean_markdown', 'clean_metrics')
        return StagePlan(tuple(PlannedOperation(document.document_id, document.version_id,
            StageSpecification(operation, ('selected_text',), required, 'cpu', settings)) for document in documents))

    def finalize(self, request, *, reporter):
        raise ValueError('Clean has no aggregate finalize operation')

    def run_document(self, request, *, reporter):
        request.work_directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic_ns()
        try:
            if request.device_id != 'cpu':
                raise ValueError('Rust clean requires CPU')
            if len(request.inputs) != 1:
                raise ValueError('Clean requires one explicit text surface')
            operation = request.operation.split('.artifact:',1)[0]
            if operation == 'clean.selection':
                request.work_directory.mkdir(parents=True, exist_ok=True)
                source = request.inputs[0]
                if sha256_file(source.path) != source.sha256:
                    raise ValueError('input_hash_changed')
                return StageResult(request.task_id, request.attempt_id, request.lease_token, 'ok', 'clean_completed', (),
                    ({'selected_file_id':source.file_id, **dict(self.selection)},),
                    {'wall_ns': time.monotonic_ns()-started, 'execution': 'in_process'})
            request = freeze_inputs(request)
            source = request.inputs[0]
            root = request.work_directory
            output_dir = root/'clean_markdown'; output_dir.mkdir()
            evidence_dir = root/'native_evidence'; evidence_dir.mkdir()
            output = output_dir/source.path.name
            raw = source.path.read_bytes().decode('utf-8')
            recorder = EditRecorder(raw)
            observations = []
            if operation in ('clean.initial', 'clean.score_after_ocr'):
                import glossapi_rs_cleaner as cleaner
                report = root/'cleaning_report.parquet'
                with capture_backend_observations() as observations:
                    cleaner.run_complete_pipeline(str(source.path.parent), str(output_dir), str(report),
                        ['greek', 'latin'], int(request.settings.get('num_threads') or 1), True,
                        evidence_directory=str(evidence_dir))
                import pandas as pd
                frame = pd.read_parquet(report)
                if len(frame) != 1 or not output.is_file():
                    raise ValueError('Fresh required cleaner report/text missing')
                metric = {key:None if pd.isna(value) else value for key,value in frame.iloc[0].to_dict().items()}
                edit_path = evidence_dir/source.path.with_suffix('.edits.csv').name
                previous_field_limit = csv.field_size_limit()
                try:
                    # Native transforms can rewrite a whole large document.
                    # Bound parsing by the actual captured artifact, not CSV's
                    # unrelated default 128 KiB field limit.
                    csv.field_size_limit(max(previous_field_limit, edit_path.stat().st_size))
                    with edit_path.open(newline='') as stream:
                        for edit in csv.DictReader(stream):
                            recorder.edit(edit['rule_id'], int(edit['start']), int(edit['end']), edit['replacement'])
                finally:
                    csv.field_size_limit(previous_field_limit)
                stage_one = evidence_dir/'native_stages/stage1_cleaned_files'/source.path.name
                if recorder.text != stage_one.read_bytes().decode('utf-8'):
                    raise ValueError('Native edit replay does not match stage-one text')
                # The table removal decision/report and both surfaces are captured
                # by the native producer before it removes its temporary directory.
                recorder.transform('native_table_report_removal', output.read_bytes().decode('utf-8'),
                    reason='batch_remove_tables_from_files; captured detailed_table_report.csv')
                table_report = evidence_dir/'native_stages/stage2_detailed_table_report.csv'
                with table_report.open(newline='') as stream:
                    table_decisions = [{key:value for key,value in row.items() if key != 'file_path'} for row in csv.DictReader(stream)]
                import glossapi_rs_noise as noise
                score_directory = source.path.parent if operation == 'clean.score_after_ocr' else output_dir
                scores = list(noise.score_markdown_directory_detailed(str(score_directory), 1))
                if len(scores) != 1:
                    raise ValueError('Fresh required noise metrics missing')
                scores[0] = list(scores[0]); scores[0][0] = source.path.name
                metrics = {'native_report': metric, 'noise_score': scores[0], 'native_table_decisions':table_decisions,
                    'native_report_surface_sha256':sha256_file(output),'noise_surface_sha256':sha256_file(source.path if operation=='clean.score_after_ocr' else output)}
            elif operation in ('clean.ocr_artifacts', 'clean.content_debug'):
                recorder.transform('python_universal_newlines',raw.replace('\r\n','\n').replace('\r','\n'))
                from ..corpus.phase_clean import _process_combined_ocr_document, DEFAULT_OCR_WORD_REPEAT_WINDOW
                import glossapi_rs_noise as noise
                debug = root/'debug'/source.path.name if request.settings.get('write_debug_files') or operation == 'clean.content_debug' else None
                if debug: debug.parent.mkdir()
                options = {key: int(request.settings.get(key, value)) for key, value in {
                    'min_progress_steps': 10, 'min_repeat_steps': 8, 'min_same_digit_steps': 10,
                    'word_rep_threshold': 4, 'word_min_period': 3, 'word_window': DEFAULT_OCR_WORD_REPEAT_WINDOW}.items()}
                with capture_backend_observations() as observations, capture_ocr_edits(recorder):
                    details = _process_combined_ocr_document(source.path, clean_output_path=output,
                        debug_output_path=debug, noise_mod=noise, include_page_metrics=True, include_match_index=True, **options)
                if recorder.text != output.read_bytes().decode('utf-8'):
                    raise ValueError('OCR producer edit replay does not match cleaned text')
                (root/'debug.details.json').write_text(json.dumps(details,ensure_ascii=False))
                metrics = {'ocr_scores': list(noise.score_markdown_directory_ocr_profile(str(source.path.parent), 1,
                    int(request.settings.get('min_repeat_run', 6))))}
                for row in metrics['ocr_scores']: row['path'] = source.path.name
                # Do not duplicate text or path-bearing debug summary in SQLite.
                metrics['matches'] = [{**{key: value for key, value in item.items()
                    if key not in ('source_path', 'debug_output_path', 'matched_text')},
                    'matched_text_sha256':text_hash(item['matched_text'])}
                    for item in details['match_index_rows']]
            else:
                raise ValueError('Unknown clean operation')
            if time.monotonic_ns() > request.deadline_monotonic_ns:
                raise TimeoutError('Clean deadline exceeded')
            import math
            def nullable(value):
                if isinstance(value,float) and not math.isfinite(value): return None
                if isinstance(value,dict): return {key:nullable(item) for key,item in value.items()}
                if isinstance(value,(list,tuple)): return [nullable(item) for item in value]
                return value
            (root/'clean.metrics.json').write_text(json.dumps(nullable(metrics), ensure_ascii=False, allow_nan=False))
            # Payload replacements stay ordinary files; ledger retains hashes/spans.
            (root/'edit-replacements.json').write_text(json.dumps(recorder.replacements, ensure_ascii=False))
            decision = {'operation': operation, 'raw_file_id': source.file_id, 'raw_hash': source.sha256,
                'clean_hash': text_hash(output.read_bytes().decode('utf-8')), 'edits': recorder.receipt(),
                'edit_summary': recorder.summary(),
                'measurements': metrics, 'backend_observations': observations}
            outputs = []
            for artifact in outputs_in(root):
                kind = 'clean_markdown' if artifact.relative_path.startswith('clean_markdown/') else 'clean_metrics' if artifact.relative_path == 'clean.metrics.json' else artifact.kind
                outputs.append(replace(artifact, kind=kind, required=kind in ('clean_markdown','clean_metrics')))
            return StageResult(request.task_id, request.attempt_id, request.lease_token, 'ok', 'clean_completed',
                tuple(outputs), (decision,), {'wall_ns': time.monotonic_ns()-started, 'execution': 'in_process'})
        except Exception as error:
            return failed_result(request, 'input_hash_changed' if str(error)=='input_hash_changed' else
                'clean_process_failed', error, metrics={'wall_ns': time.monotonic_ns()-started, 'execution': 'in_process'})
