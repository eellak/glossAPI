"""Existing Rust and OCR text kernels, with private inputs and producer edits.

The native cleaner runs in process; no interpreter/model startup per document.
Corpus-level metadata serialization and selection remain compatibility work.
"""
from __future__ import annotations
from .execution import write_text_hashed

from dataclasses import replace
import csv
import json
from pathlib import Path
import time

from typing import Any

from .contracts import PackageRequirement, PlannedOperation, StagePlan, StageRequest, StageResult, StageSpecification, OutputArtifact
from .execution import failed_result, freeze_inputs, outputs_in, sha256_file, remember_written_hash
from .edits import EditRecorder, text_hash, capture_ocr_edits
from .receipts import capture_backend_observations


def _has_supplement_marker(path, marker=b'<!-- ocr-', block=1 << 20):
    """Stream one file for an OCR supplement marker; never retains its payload."""
    overlap = len(marker) - 1
    tail = b''
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(block)
            if not chunk:
                return False
            if marker in tail + chunk:
                return True
            tail = chunk[-overlap:]


def score_ocr_profile_directory(noise, directory, threads, min_repeat_run, *, scratch_root=None):
    """Native OCR-profile text metrics over a directory, without OCR supplement markers.

    The supplement's composition comments are structure, not document text, so
    script shares such as ``latin_percentage`` must not count them (FG-G02).
    The directory is scored in place, exactly as before. Files are inspected
    one at a time; only files that contain a marker are rewritten, one at a
    time, into a marker-free scratch copy that is scored and replaces their
    rows (CP-LEDGER F7). ``scratch_root`` is the caller's owned, accounted
    workspace (the attempt directory, or the corpus staging area); the scratch
    copy is removed before returning.
    """
    import shutil
    import uuid
    from ..ocr.page_artifacts import strip_supplement_markers
    directory = Path(directory)
    marked = [path for path in sorted(directory.rglob('*.md')) if path.is_file() and _has_supplement_marker(path)]
    rows = list(noise.score_markdown_directory_ocr_profile(str(directory), threads, min_repeat_run))
    if not marked:
        return rows
    if scratch_root is None:
        import tempfile
        scratch = Path(tempfile.mkdtemp(prefix='glossapi-ocr-metrics-'))
    else:
        scratch = Path(scratch_root)/('ocr-metrics-'+uuid.uuid4().hex)
        scratch.mkdir(parents=True)
    try:
        for path in marked:
            target = scratch/path.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            text = strip_supplement_markers(path.read_bytes().decode('utf-8', 'surrogateescape'))
            target.write_bytes(text.encode('utf-8', 'surrogateescape'))
        replacements = {Path(row['path']).relative_to(scratch): row
                        for row in noise.score_markdown_directory_ocr_profile(str(scratch), threads, min_repeat_run)}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    result = []
    for row in rows:
        relative = Path(row['path']).relative_to(directory)
        row = dict(replacements.get(relative, row))
        row['path'] = str(directory/relative)
        result.append(row)
    return result


def marker_free_scoring_copy(input_dir, staging_root):
    """A copy of ``input_dir``'s markdown tree with the OCR supplement's composition lines removed, for the native
    scorers of the legacy score-only clean (FG-G02, K4g fix 5); None (score ``input_dir`` itself, as before) when no
    file has a marker. Unmarked files are copied byte for byte, so their report rows are the ones the original gives.
    The caller removes the copy after scoring."""
    import uuid
    from ..ocr.page_artifacts import strip_supplement_markers
    input_dir = Path(input_dir)
    files = sorted(path for path in input_dir.rglob('*.md') if path.is_file())
    if not any(_has_supplement_marker(path) for path in files):
        return None
    target = Path(staging_root)/f'clean-metric-surface-{uuid.uuid4().hex}'
    for path in files:
        copy = target/path.relative_to(input_dir)
        copy.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        if _has_supplement_marker(path):
            data = strip_supplement_markers(data.decode('utf-8', 'surrogateescape')).encode('utf-8', 'surrogateescape')
        copy.write_bytes(data)
    return target


def metric_surface_without_markers(recorder, name, directory):
    """Write ``recorder``'s text without the OCR supplement's composition lines to ``directory/name`` (a directory
    holding only that file, as the native scorers read whole directories) and record each removed line as a producer
    edit on the recorder, so the native edits that follow replay on the same surface. Returns the written path."""
    from ..ocr.page_artifacts import SUPPLEMENT_MARKER_RE
    removed = 0
    for match in list(SUPPLEMENT_MARKER_RE.finditer(recorder.text)):
        recorder.edit('ocr_supplement_marker_excluded_from_metrics', match.start()-removed, match.end()-removed, '',
            reason='FG-G02: text metrics do not count the OCR supplement composition lines')
        removed += match.end()-match.start()
    directory.mkdir(parents=True)
    surface = directory/name
    surface.write_bytes(recorder.text.encode('utf-8'))
    return surface


class CleanStage:
    name = 'clean'
    contract_version = 1

    def __init__(self, selection: 'dict[str, Any] | None' = None) -> None:
        self.selection = selection

    def packages(self, specification):
        return (PackageRequirement('core', ('pandas>=3.0.0', 'pyarrow>=26,<27')),)

    def plan(self, documents, *, settings):
        operation = settings.get('operation', 'clean.initial')
        required = ('clean_metrics',) if operation == 'clean.score_after_ocr' else ('clean_markdown', 'clean_metrics')
        return StagePlan(tuple(PlannedOperation(document.document_id, document.version_id,
            StageSpecification(operation, ('selected_text',), required, 'cpu', settings)) for document in documents))

    def finalize(self, request: StageRequest, *, reporter: object) -> StageResult:
        """One aggregate selection per run (K4(c) part 3): every selected text is a bound input and the decision
        carries one row per document, so ``report --why`` still gives each document its own selection and reason."""
        started = time.monotonic_ns()
        selection = self.selection
        if request.operation != 'clean.selection' or selection is None:
            raise ValueError('Clean has no aggregate finalize operation')
        rows: list[dict[str, Any]] = list(selection['rows'])
        if len(rows) != len(request.inputs):
            raise ValueError('Selection rows do not match the selected texts')
        request.work_directory.mkdir(parents=True, exist_ok=True)
        selections: list[dict[str, Any]] = []
        from .products import active_journal
        journal: Any = active_journal()
        for source, row in zip(request.inputs, rows):
            # Producer ownership decides reuse (CP-K4bc2 R3): a score-only caller text (initial_external) or any
            # caller file is byte-hashed; only a managed product glossAPI wrote may reuse its registered hash.
            fact: Any = journal.writer.connection.execute(
                'SELECT ownership FROM files WHERE file_id=?', (source.file_id,)).fetchone() if journal is not None else None
            owned = bool(fact) and fact[0] == 'managed'
            if sha256_file(source.path, own=owned) != source.sha256 or row['selected_text_sha256'] != source.sha256:
                raise ValueError('input_hash_changed')
            selections.append({'selected_file_id': source.file_id, **row})
        common = {key: value for key, value in selection.items() if key != 'rows'}
        decision: dict[str, Any] = {'selections': selections, **common}
        return StageResult(request.task_id, request.attempt_id, request.lease_token, 'ok', 'clean_completed', (),
            (decision,), {'wall_ns': time.monotonic_ns()-started, 'execution': 'in_process'})

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
                decision: dict[str, Any] = {'selected_file_id':source.file_id, **(self.selection or {})}
                return StageResult(request.task_id, request.attempt_id, request.lease_token, 'ok', 'clean_completed', (),
                    (decision,),
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
                surface = source.path
                if operation == 'clean.score_after_ocr' and _has_supplement_marker(source.path):
                    # The score-only pass after OCR produces the exported text metrics (latin_percentage,
                    # percentage_greek, ...). The OCR supplement's composition lines are structure, not document text,
                    # so the metrics are measured without them (FG-G02, as the OCR profile already is), on an attempt-
                    # private copy; the producer edits record their removal (K4g fix 5).
                    surface = metric_surface_without_markers(recorder, source.path.name, root/'native_evidence'/'metric_surface')
                with capture_backend_observations() as observations:
                    cleaner.run_complete_pipeline(str(surface.parent), str(output_dir), str(report),
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
                import hashlib
                output_bytes = output.read_bytes()
                output_digest = hashlib.sha256(output_bytes).hexdigest()
                remember_written_hash(output,output_digest,len(output_bytes))
                recorder.transform('native_table_report_removal', output_bytes.decode('utf-8'),
                    reason='batch_remove_tables_from_files; captured detailed_table_report.csv')
                table_report = evidence_dir/'native_stages/stage2_detailed_table_report.csv'
                with table_report.open(newline='') as stream:
                    table_decisions = [{key:value for key,value in row.items() if key != 'file_path'} for row in csv.DictReader(stream)]
                import glossapi_rs_noise as noise
                score_directory = surface.parent if operation == 'clean.score_after_ocr' else output_dir
                scores = list(noise.score_markdown_directory_detailed(str(score_directory), 1))
                if len(scores) != 1:
                    raise ValueError('Fresh required noise metrics missing')
                scores[0] = list(scores[0]); scores[0][0] = source.path.name
                noise_surface = (source.sha256 if surface == source.path else sha256_file(surface)) if operation == 'clean.score_after_ocr' else output_digest
                metrics = {'native_report': metric, 'noise_score': scores[0], 'native_table_decisions':table_decisions,
                    'native_report_surface_sha256':output_digest,'noise_surface_sha256':noise_surface}
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
                write_text_hashed(root/'debug.details.json',json.dumps(details,ensure_ascii=False))
                metrics = {'ocr_scores': score_ocr_profile_directory(noise, source.path.parent, 1,
                    int(request.settings.get('min_repeat_run', 6)), scratch_root=root)}
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
            write_text_hashed(root/'clean.metrics.json',json.dumps(nullable(metrics), ensure_ascii=False, allow_nan=False))
            # Payload replacements stay ordinary files; ledger retains hashes/spans.
            if request.settings.get('debug', 'off') == 'on':
                write_text_hashed(root/'edit-replacements.json',json.dumps(recorder.replacements, ensure_ascii=False))
            decision = {'operation': operation, 'raw_file_id': source.file_id, 'raw_hash': source.sha256,
                'clean_hash': output_digest if operation in ('clean.initial','clean.score_after_ocr') else text_hash(output.read_bytes().decode('utf-8')), 'edits': recorder.receipt() if request.settings.get('debug', 'off') == 'on' else recorder.summary(),
                'edit_detail': 'spans' if request.settings.get('debug', 'off') == 'on' else 'per_rule_summary',
                'measurements': metrics, 'backend_observations': observations}
            outputs = []
            for artifact in outputs_in(root):
                kind = 'clean_markdown' if artifact.relative_path.startswith('clean_markdown/') else 'clean_metrics' if artifact.relative_path == 'clean.metrics.json' else 'attempt_diagnostic' if artifact.relative_path.startswith('native_evidence/') else artifact.kind
                outputs.append(replace(artifact, kind=kind, required=kind in ('clean_markdown','clean_metrics')))
            return StageResult(request.task_id, request.attempt_id, request.lease_token, 'ok', 'clean_completed',
                tuple(outputs), (decision,), {'wall_ns': time.monotonic_ns()-started, 'execution': 'in_process'})
        except Exception as error:
            return failed_result(request, 'input_hash_changed' if str(error)=='input_hash_changed' else
                'clean_process_failed', error, metrics={'wall_ns': time.monotonic_ns()-started, 'execution': 'in_process'})
