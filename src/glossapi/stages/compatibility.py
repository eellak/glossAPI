"""Opt-in sequential adapter profile for the existing Corpus serializers.

Until all wrapper products move in step 5, this profile returns the same private
worker outputs through the existing phase controller and its state serializers.
"""
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from .contracts import ArtifactReference, DocumentSnapshot, StageRequest, StageSpecification
from .execution import ReceiptCollector, checked_path, sha256_file
from .extract import ExtractStage


def claim_publication(corpus, destination, version):
    """Reject, never overwrite, a second document version projected onto one location.

    A streaming run publishes each input version under its own frozen name
    (CP-LEDGER F1). Any remaining collision within the run (for example two
    names that a case-insensitive filesystem folds together) is an ambiguous
    projection and fails the barrier instead of silently replacing a product.
    Returns a callable that records the claim after the bytes are written.
    """
    claims = getattr(corpus, '_publication_claims', None)
    if claims is None or version is None:
        return lambda: None
    destination = Path(destination).absolute()
    keys = [('path', os.path.normcase(str(destination)))]
    if destination.exists():
        info = destination.stat()
        keys.append(('inode', info.st_dev, info.st_ino))
    for key in keys:
        owner = claims.get(key)
        if owner is not None and owner != version:
            raise ValueError('Ambiguous publication: '+destination.name+' would replace the product of another document version in this run')
    def record():
        info = destination.stat()
        claims[keys[0]] = version
        claims[('inode', info.st_dev, info.st_ino)] = version
    return record


def register_publication(writer, run_id, stage, kind, uri, *, version_id, producer_attempt_id, sha256, bytes, retention_class):
    """Register one published compatibility product exactly once (K4C-H1).

    A publication killed after its bytes reached the destination but before
    its registration committed is replayed on resume. The replay reuses an
    identical committed registration instead of adding a second present row
    for the same path, and a wrapper inventory's unowned observation of the
    leftover bytes is closed once the product owns the location. Otherwise a
    second present registration pins the path and retention keeps a file the
    uninterrupted run releases.
    """
    rows = writer.connection.execute(
        "SELECT file_id,version_id,producer_attempt_id,kind,sha256,bytes,ownership,retention_class FROM current_file_state WHERE uri=? AND presence='present'",
        (uri,)).fetchall()
    for row in rows:
        if (row['producer_attempt_id'], row['kind'], row['ownership'], row['sha256'], row['bytes'], row['retention_class'], row['version_id']) == (
                producer_attempt_id, kind, 'managed', sha256, bytes, retention_class, version_id):
            return row['file_id']
    file_id = writer.artifact(run_id, stage, kind, uri, version_id=version_id, producer_attempt_id=producer_attempt_id,
        sha256=sha256, bytes=bytes, ownership='managed', retention_class=retention_class)
    for row in rows:
        if row['ownership'] == 'legacy_external' and row['producer_attempt_id'] is None and (row['sha256'], row['bytes']) == (sha256, bytes):
            writer.event(run_id, event_kind='artifact', reason_code='missing', file_id=row['file_id'], version_id=row['version_id'],
                details={'observation': 'unowned_location_claimed_by_product', 'product_file_id': file_id})
    return file_id


def extract_compatibility_document(extractor, source, output_dir, timeout_dir=None):
    from ..ledger.journal import observe, accept_worker_observations, _active_journal, ResultJournal
    source, output_dir = Path(source).absolute(), Path(output_dir).absolute()
    settings = {"phase1_backend": "safe" if extractor.use_pypdfium_backend else "docling",
        "repair_fonts": extractor.repair_fonts, "export_doc_json": extractor.export_doc_json,
        "emit_formula_index": extractor.emit_formula_index, "benchmark_mode": extractor.benchmark_mode,
        "num_threads": 1, "page_range": getattr(extractor, "page_range", None),
        "long_pdf_page_threshold": extractor.long_pdf_page_threshold, "chunk_size": extractor.chunk_size,
        "max_chunk_timeouts": extractor.max_chunk_timeouts, "per_page_timeout_s": extractor.per_page_timeout_s}
    signature = getattr(extractor, "_last_extractor_cfg", None)
    if signature:
        if signature[1] or signature[-1] is not None:
            raise ValueError("OCR plugin extraction belongs to its separate stage adapter")
        settings.update(images_scale=signature[4], formula_enrichment=signature[5], code_enrichment=signature[6], use_cls=signature[7])
    from .products import producer_binding
    journal = _active_journal.get()
    binding = producer_binding(journal if isinstance(journal, ResultJournal) else None, source)
    reference = ArtifactReference(str(uuid.uuid4()), "source_payload", source,
        binding['sha256'] if binding else sha256_file(source), source.stat().st_size)
    # One page-count based timeout per document (it was computed twice, each opening the PDF).
    timeout_seconds = extractor._calculate_text_timeout(file_path=source)
    with tempfile.TemporaryDirectory(prefix="glossapi-sequential-") as temporary:
        identifiers = [str(uuid.uuid4()) for _ in range(5)]
        request = StageRequest(*identifiers, "extract.document", (reference,), settings, {}, "cpu", Path(temporary)/"attempt",
            time.monotonic_ns() + int(timeout_seconds*1e9))
        journal = _active_journal.get()
        if isinstance(journal, ResultJournal):
            from .sequential import SequentialStageRunner
            version = journal.versions.get(str(source))
            if version is None:
                # The legacy facade can stage a symlink/copy into downloads.
                # Bind a resolved alias explicitly; never match a basename.
                aliases={value for path,value in journal.versions.items() if Path(path).resolve()==source.resolve()}
                with journal.writer.transaction():
                    if len(aliases)==1:
                        version=next(iter(aliases))
                    else:
                        from ..ledger.journal import source_key
                        identity=journal.writer.document(source_key(source.as_uri()))
                        version=journal.writer.version(identity,source_sha256=reference.sha256,stored_sha256=reference.sha256)
                    journal.versions[str(source)]=version
                    journal.writer.artifact(journal.run_id,'extract','initial_input',source.as_uri(),version_id=version,
                        sha256=reference.sha256,bytes=reference.bytes,ownership='initial_external',retention_class='initial')
            file_id = journal.writer.connection.execute("SELECT file_id FROM current_file_state WHERE uri=? AND version_id=? AND sha256=? AND presence='present' ORDER BY event_id DESC LIMIT 1",
                (source.as_uri(), version, reference.sha256)).fetchone()[0]
            document_id = journal.writer.connection.execute("SELECT document_id FROM document_versions WHERE version_id=?", (version,)).fetchone()[0]
            snapshot = DocumentSnapshot(document_id, version, source.as_uri(), artifacts=(
                ArtifactReference(file_id, reference.kind, source, reference.sha256, reference.bytes),))
            result, workspace = SequentialStageRunner(journal.writer, journal.corpus.output_dir).execute(
                ExtractStage(), snapshot, StageSpecification("extract.document", ("source_payload",), ("raw_markdown",), settings=settings),
                timeout_seconds=timeout_seconds, run_id=journal.run_id)
            journal.adapter_stages.add('extract')
            journal.adapter_attempts.add(result.attempt_id)
        else:
            result = ExtractStage().run_document(request, reporter=ReceiptCollector())
            workspace = request.work_directory
        # Compatibility products use their original path spellings. This only
        # changes location references in the old serializer's chunk receipts.
        mirror_hashes = {}
        for output in result.outputs:
            relative = output.relative_path
            if relative.startswith(("logs/", "inputs/")) or output.kind in ("attempt_diagnostic", "worker_receipt"):
                continue
            path = checked_path(workspace, relative)
            if output.kind == "chunk_manifest":
                import json
                payload = json.loads(path.read_text())
                payload["file"] = str(output_dir.parent/"repaired_pdfs"/source.name) if (workspace/"repaired_pdfs"/source.name).is_file() else str(source)
                serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2)
            if output.kind == "extract_sidecar":
                import json
                payload = json.loads(path.read_text())
                if payload.get("chunk_manifest_path"):
                    payload["chunk_manifest_path"] = str(output_dir/f"{source.stem}.chunks.json")
                    serialized_payload = json.dumps(payload, ensure_ascii=False)
            destination = output_dir.parent / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            claimed = claim_publication(journal.corpus, destination, version) if isinstance(journal, ResultJournal) else (lambda: None)
            if output.kind == "chunk_manifest" or (output.kind == "extract_sidecar" and payload.get("chunk_manifest_path")):
                from .execution import write_text_hashed
                import hashlib
                write_text_hashed(destination, serialized_payload)
                data = serialized_payload.encode('utf-8')
                mirror_hashes[relative] = (hashlib.sha256(data).hexdigest(), len(data))
            else:
                from .execution import copy_hashed, link_published, links_released_copies
                linked = None
                if output.sha256 and isinstance(journal, ResultJournal) and links_released_copies(journal, output.retention_class, output.kind):
                    linked = link_published(path, destination, output.sha256, output.bytes)
                digest, size = linked or copy_hashed(path, destination)
                if output.sha256 and (digest, size) != (output.sha256, output.bytes):
                    raise ValueError('Product changed during final publication')
                mirror_hashes[relative] = (digest, size)
            claimed()
        if isinstance(journal, ResultJournal):
            with journal.writer.transaction():
                for output in result.outputs:
                    path = workspace/output.relative_path
                    destination = output_dir.parent/output.relative_path
                    if output.relative_path not in mirror_hashes:
                        continue
                    # Compatibility receipts can deliberately re-spell source paths.
                    producer_run=journal.writer.connection.execute('SELECT t.run_id FROM attempts a JOIN tasks t ON t.task_id=a.task_id WHERE a.attempt_id=?',(result.attempt_id,)).fetchone()[0]
                    register_publication(journal.writer, producer_run, "extract", output.kind, destination.as_uri(),
                        version_id=version, producer_attempt_id=result.attempt_id, sha256=mirror_hashes[output.relative_path][0],
                        bytes=mirror_hashes[output.relative_path][1], retention_class=output.retention_class)
        if not isinstance(journal, ResultJournal) or result.status != 'ok':
            for decision in result.decisions:
                accept_worker_observations(decision.get("backend_observations", ()))
        observe(result.reason_code, path=source, status=result.status, adapter_contract_version=1,
            adapter_metrics=result.metrics, decisions=result.decisions,
            error=None, adapter_error=asdict(result.error) if result.error else None)
        return result.status == "ok"


def execute_compatibility_stage(corpus, stage, inputs, specification, *, timeout_seconds=600):
    """Bind explicit compatibility inputs, or protect a newly given text input."""
    from ..ledger.journal import _active_journal, ResultJournal, source_key
    journal = _active_journal.get()
    specification = __import__('dataclasses').replace(specification,settings={**specification.settings,
        'compatibility_filename':specification.settings.get('compatibility_filename',Path(inputs[0][1]).with_suffix('.pdf').name)})
    references = []
    version = document_id = None
    if isinstance(journal, ResultJournal):
        from .sequential import SequentialStageRunner
        from ..ledger.writer import utc_now
        with journal.writer.transaction():
            for role, path in inputs:
                path = Path(path).absolute()
                from .products import producer_binding
                binding = producer_binding(journal, path)
                digest = binding['sha256'] if binding else sha256_file(path)
                fact = journal.writer.connection.execute(
                    "SELECT * FROM current_file_state WHERE uri=? AND sha256=? AND version_id IS NOT NULL AND presence='present' ORDER BY event_id DESC LIMIT 1",
                    (path.as_uri(), digest)).fetchone()
                if version is None:
                    if fact is not None:
                        version = fact['version_id']
                        document_id = journal.writer.connection.execute('SELECT document_id FROM document_versions WHERE version_id=?',(version,)).fetchone()[0]
                    else:
                        document_id = journal.writer.document(source_key(path.as_uri()))
                        version = journal.writer.version(document_id, source_sha256=digest, stored_sha256=digest)
                if fact is None or fact['version_id'] != version:
                    file_id = journal.writer.artifact(journal.run_id, stage.name, role, path.as_uri(),
                        version_id=version, sha256=digest, bytes=path.stat().st_size,
                        ownership='initial_external', retention_class='initial')
                else:
                    file_id = fact['file_id']
                references.append(ArtifactReference(file_id, role, path, digest, path.stat().st_size))
        result, workspace = SequentialStageRunner(journal.writer, corpus.output_dir).execute(stage,
            DocumentSnapshot(document_id, version, references[0].path.as_uri(), artifacts=tuple(references)),
            specification, run_id=journal.run_id, timeout_seconds=timeout_seconds)
        journal.adapter_stages.add(stage.name)
        journal.adapter_attempts.add(result.attempt_id)
        return result, workspace
    for role, path in inputs:
        path = Path(path).absolute()
        references.append(ArtifactReference(str(uuid.uuid4()), role, path, sha256_file(path), path.stat().st_size))
    workspace = Path(tempfile.mkdtemp(prefix='glossapi-adapter-', dir=corpus.output_dir))
    request = StageRequest(*(str(uuid.uuid4()) for _ in range(5)), specification.operation,
        tuple(references), dict(specification.settings), {}, specification.device_id,
        workspace/'attempt', time.monotonic_ns()+int(timeout_seconds*1e9))
    result = stage.run_document(request, reporter=ReceiptCollector())
    return result, request.work_directory


def materialize_stage_outputs(corpus, result, workspace, paths):
    """Copy named successful outputs only; failed outputs stay in their attempt."""
    from ..ledger.journal import _active_journal, ResultJournal
    journal = _active_journal.get()
    if result.status != 'ok':
        raise RuntimeError(f'{result.reason_code}: {result.error.message if result.error else "stage failed"}')
    for relative, destination in paths.items():
        path = checked_path(workspace, relative)
        destination = Path(destination).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        from .execution import copy_hashed
        fact = None
        claimed = lambda: None
        if isinstance(journal, ResultJournal):
            fact = journal.writer.connection.execute('SELECT * FROM files WHERE producer_attempt_id=? AND uri=?',
                (result.attempt_id, path.as_uri())).fetchone()
            if fact is None:
                raise ValueError('Cannot materialize unregistered stage output')
            claimed = claim_publication(corpus, destination, fact['version_id'])
        from .execution import link_published, links_released_copies
        linked = None
        if fact is not None and links_released_copies(journal, fact['retention_class'], fact['kind']):
            linked = link_published(path, destination, fact['sha256'], fact['bytes'])
        digest, size = linked or copy_hashed(path, destination)
        claimed()
        if isinstance(journal, ResultJournal):
            if (digest,size)!=(fact['sha256'],fact['bytes']):
                raise ValueError('Product changed during final publication')
            with journal.writer.transaction():
                producer_run=journal.writer.connection.execute('SELECT t.run_id FROM attempts a JOIN tasks t ON t.task_id=a.task_id WHERE a.attempt_id=?',(result.attempt_id,)).fetchone()[0]
                register_publication(journal.writer, producer_run, fact['stage'], fact['kind'], destination.as_uri(),
                    version_id=fact['version_id'], producer_attempt_id=result.attempt_id,
                    sha256=fact['sha256'], bytes=fact['bytes'], retention_class=fact['retention_class'])
    if not isinstance(journal, ResultJournal):
        # Only the owned temporary attempt, after outputs have been consumed.
        shutil.rmtree(workspace.parent)


def clean_compatibility_kernel(corpus, md_files, report_path, *, input_root=None, num_threads=1, write_cleaned_files=True):
    import json
    import pandas as pd
    from .clean import CleanStage
    frames, scores = [], []
    corpus._stage_clean_surfaces = []
    for source in md_files:
        relative=source.relative_to(input_root) if input_root is not None else Path(source.name)
        destination=corpus.cleaned_markdown_dir/relative
        result, workspace = execute_compatibility_stage(corpus, CleanStage(), [('selected_text', source)],
            StageSpecification('clean.initial' if write_cleaned_files else 'clean.score_after_ocr', ('selected_text',), ('clean_markdown','clean_metrics'),
                settings={'num_threads': num_threads or 1}))
        if result.status != 'ok':
            raise RuntimeError(f'{result.reason_code}: {result.error.message}')
        metrics = json.loads((workspace/'clean.metrics.json').read_text())
        frames.append(pd.DataFrame([metrics['native_report']]))
        score = [float('nan') if value is None else value for value in metrics['noise_score']]; score[0] = str(destination); scores.append(score)
        corpus._stage_clean_surfaces.append((source, destination if write_cleaned_files else source))
        materialize_stage_outputs(corpus, result, workspace,
            {f'clean_markdown/{source.name}': destination} if write_cleaned_files else {})
    if frames:
        # A selectable metadata-shaped report: written through the metadata choke point (K2F round 3), same bytes.
        from ..parquet_schema import pandas_parquet, replace_metadata_parquet
        replace_metadata_parquet(report_path, pd.concat(frames, ignore_index=True), serialize=pandas_parquet)
    return scores


def record_clean_selection(corpus, frame, *, drop_bad, thresholds):
    from .clean import CleanStage
    from ..ledger.journal import _active_journal, ResultJournal
    journal = _active_journal.get()
    if not isinstance(journal, ResultJournal):
        return
    from .products import producer_binding
    # The first row of each filename, found once (was a full-frame filter per document; same row chosen).
    first_rows = {}
    for position, name in enumerate(frame['filename'].astype(str)):
        first_rows.setdefault(name, position)
    for source, output in getattr(corpus, '_stage_clean_surfaces', ()):
        position = first_rows.get(source.with_suffix('.pdf').name)
        if position is None:
            continue
        row = frame.iloc[position]
        # The selected text is a committed product: its producer hash, not a second read of the same bytes.
        binding = producer_binding(journal, output)
        selection = {'accepted_for_sections': source.stem in corpus.good_files,
            'needs_ocr': bool(row.get('needs_ocr',False)), 'filter_tokens': str(row.get('filter','ok')).split(';'),
            'effective_thresholds': thresholds, 'drop_bad': drop_bad,
            'selected_text_sha256': binding['sha256'] if binding else sha256_file(output),
            'selection_policy': 'legacy_needs_ocr_false_when_drop_bad',
            'routing_metrics':{field:(None if __import__('pandas').isna(row[field]) else row[field].item() if hasattr(row[field],'item') else row[field])
                for field in ('latin_word_shape_rate','ocr_pages','ocr_pages_count','ocr_pages_unflagged_count','page_rule_profile','page_fold_validity') if field in row}}

        execute_compatibility_stage(corpus, CleanStage(selection), [('selected_text', output)],
            StageSpecification('clean.selection', ('selected_text',), (), settings={
                'effective_thresholds':thresholds,'drop_bad':drop_bad,'selection_policy':selection['selection_policy']}))


def clean_ocr_compatibility_kernel(corpus, md_files, settings, *, write_cleaned_files, debug_dir):
    import json
    from .clean import CleanStage
    scores, debug_documents = [], []
    corpus._stage_clean_surfaces = []
    for source in md_files:
        result, workspace = execute_compatibility_stage(corpus, CleanStage(), [('selected_text',source)],
            StageSpecification('clean.ocr_artifacts',('selected_text',),('clean_markdown','clean_metrics'),settings=settings))
        if result.status != 'ok':
            raise RuntimeError(f'{result.reason_code}: {result.error.message}')
        metrics = json.loads((workspace/'clean.metrics.json').read_text())
        for row in metrics['ocr_scores']:
            row['path'] = str(source)
            scores.append(row)
        paths = {f'clean_markdown/{source.name}':corpus.cleaned_markdown_dir/source.name} if write_cleaned_files else {}
        corpus._stage_clean_surfaces.append((source,corpus.cleaned_markdown_dir/source.name if write_cleaned_files else source))
        if debug_dir is not None:
            details=json.loads((workspace/'debug.details.json').read_text())
            for row in [details['row'], *details['page_metric_rows'], *details['match_index_rows']]:
                for key,value in list(row.items()):
                    if key == 'source_path': row[key]=str(source)
                    elif key == 'clean_output_path': row[key]=str(corpus.cleaned_markdown_dir/source.name) if write_cleaned_files else None
                    elif key in ('output_path','debug_output_path'): row[key]=str(debug_dir/source.name)
            debug_documents.append(details)
            paths[f'debug/{source.name}'] = debug_dir/source.name
        materialize_stage_outputs(corpus,result,workspace,paths)
    if debug_dir is not None:
        publish_ocr_debug_details(debug_dir,debug_documents,settings)
    return scores


def ocr_compatibility_documents(context, filenames, *, request, page_ranges=None):
    """Keep target selection/persistence outside the plugin result's ownership."""
    import json
    from .ocr import OcrStage
    from ..ocr.defaults import check_requested_device
    # An explicit request and ocr_options.device must agree before a pool/adapter is chosen.
    check_requested_device(request.device, request.ocr_options)
    for name in filenames:
        source = next((directory/name for directory in (context.output_dir/'downloads',context.input_dir/'downloads',context.input_dir)
            if (directory/name).is_file()),context.input_dir/name)
        inputs = [('source_payload',source)]
        for kind, path in [('native_markdown',context.output_dir/'markdown'/source.with_suffix('.md').name),
            ('ocr_native',context.output_dir/'sidecars/ocr_native'/source.with_suffix('.md').name),
            ('ocr_page_state',context.output_dir/'sidecars/ocr_pages'/source.with_suffix('.json').name)]:
            if path.is_file(): inputs.append((kind,path))
        for suffix in ('.docling.json.zst','.docling.json'):
            path=context.output_dir/'json'/f'{source.stem}{suffix}'
            if path.is_file(): inputs.append(('docling_json',path)); break
        ranges=[list(map(int,spec.rsplit(':',2)[1:])) for spec in page_ranges or [] if spec.rsplit(':',2)[0]==name]
        options=request.ocr_options
        from ..ocr.defaults import serialize_options, default_values, select_device
        greek_engine = options is None or getattr(options,'kind',None)=='glossapi'
        # Accelerator auto-selection is only for the qualified Greek OCR engine.
        # An external engine without an explicit device runs on CPU, its only
        # qualified device; an explicit accelerator request is still rejected
        # by validate_device rather than silently replaced.
        device = getattr(options,'device',None) or (select_device(request.device) if greek_engine else (request.device or 'cpu'))
        serialized = serialize_options(options) if options is not None else default_values(device)
        settings={'page_ranges':ranges,'max_pages':request.max_pages,'options':serialized,'requested_device':request.device,
            'entry_point_name':'glossapi' if options is None or getattr(options,'kind',None)=='glossapi' else None,
            'cache_root':str(getattr(context,'_ocr_cache_root', '')) or None}
        # Existing adapter state is reusable only with its recorded source binding.
        from ..ledger.journal import _active_journal, ResultJournal
        journal=_active_journal.get()
        if any(kind=='ocr_page_state' for kind,path in inputs):
            if not isinstance(journal,ResultJournal):
                raise ValueError('Repeated OCR requires a producer-bound prior state ledger')
            state=next(path for kind,path in inputs if kind=='ocr_page_state')
            fact=journal.writer.connection.execute('SELECT producer_attempt_id FROM files WHERE uri=? AND sha256=? ORDER BY rowid DESC LIMIT 1',
                (state.absolute().as_uri(),sha256_file(state))).fetchone()
            if fact and fact[0]:
                rows=[]
                for event in journal.writer.connection.execute(f"SELECT details_json FROM {journal.writer.connection.event_table} WHERE attempt_id=? AND event_kind='decision'",(fact[0],)):
                    digest=json.loads(event[0]).get('evidence_sha256')
                    if digest:
                        row=journal.writer.connection.execute("SELECT json FROM snapshots WHERE sha256=? AND kind='stage_evidence'",(digest,)).fetchone()
                        if row: rows.append(row)
                from ..ledger.encoding import unpack_evidence
                for row in rows:
                    evidence=unpack_evidence(json.loads(row[0])); binding=evidence.get('inputs',{}).get('source_payload',{})
                    if binding.get('sha256')==sha256_file(source): settings['prior_state_source_sha256']=binding['sha256']
        result,workspace=execute_compatibility_stage(context,OcrStage(options),inputs,
            StageSpecification('ocr.document',tuple(kind for kind,path in inputs),('ocr_markdown','ocr_metrics'),device,settings))
        paths={output.relative_path:context.output_dir/output.relative_path for output in result.outputs
            if not output.relative_path.startswith('inputs/') and output.kind not in ('worker_receipt','source_snapshot')}
        materialize_stage_outputs(context,result,workspace,paths)


def math_compatibility_document(corpus, source, document, *, device, batch_size, dpi_base, targets):
    import json
    from .math import MathStage
    inputs=[('source_payload',source),('docling_json',document)]
    chosen=corpus.markdown_dir/source.with_suffix('.md').name
    if chosen.is_file(): inputs.append(('selected_text',chosen))
    result,workspace=execute_compatibility_stage(corpus,MathStage(),inputs,
        StageSpecification('math.infer',tuple(kind for kind,path in inputs),('math_markdown','latex_map','math_metrics'),
            device,{'batch_size':batch_size,'dpi_base':dpi_base,'targets':targets}))
    if result.status!='ok':
        raise RuntimeError(f'{result.reason_code}: {result.error.message}')
    stats=json.loads((workspace/'math.metrics.json').read_text())
    materialize_stage_outputs(corpus,result,workspace,{
        f'markdown/{source.stem}.md':chosen, f'json/{source.stem}.latex_map.jsonl':corpus.output_dir/'json'/f'{source.stem}.latex_map.jsonl'})
    return stats


def publish_ocr_debug_details(debug_dir,documents,settings):
    import json
    from ..corpus.phase_clean import _summarize_metric
    rows=[document['row'] for document in documents]
    page_rows=[row for document in documents for row in document['page_metric_rows']]
    fields={'manifest.jsonl':rows,'page_metrics.jsonl':page_rows,
        'match_index.jsonl':[row for document in documents for row in document['match_index_rows']]}
    for filename,values in fields.items():
        with (debug_dir/filename).open('w',encoding='utf-8') as stream:
            for row in values: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    total_page_times=[row['total_page_seconds'] for row in page_rows]
    table_page_times=[row['table_seconds'] for row in page_rows]
    numeric_page_times=[row['numeric_seconds'] for row in page_rows]
    latex_page_times=[row['latex_seconds'] for row in page_rows]
    hybrid_page_times=[row['hybrid_seconds'] for row in page_rows]
    shared_page_times=[row['shared_repeat_seconds'] for row in page_rows]
    char_eval_times=[row['char_eval_seconds'] for row in page_rows]
    bad_char_ratios=[row['bad_char_ratio'] for row in page_rows]
    word_rep_threshold=settings['word_rep_threshold']
    word_min_period=settings['word_min_period']
    word_window=settings['word_window']
    debug_summary = {
        "doc_count": len(rows),
        "matched_doc_count": sum(1 for row in rows if int(row["matched_page_count"]) > 0),
        "matched_page_count": int(sum(int(row["matched_page_count"]) for row in rows)),
        "match_count": int(sum(int(row.get("match_count", 0)) for row in rows)),
        "table_match_count": int(sum(int(row["table_match_count"]) for row in rows)),
        "numeric_match_count": int(sum(int(row["numeric_match_count"]) for row in rows)),
        "latex_match_count": int(sum(int(row["latex_match_count"]) for row in rows)),
        "hybrid_match_count": int(sum(int(row["hybrid_match_count"]) for row in rows)),
        "word_match_count": int(sum(int(row["word_match_count"]) for row in rows)),
        "word_rep_threshold": int(word_rep_threshold),
        "word_min_period": int(word_min_period),
        "word_window": int(word_window),
        "total_page_seconds": _summarize_metric(total_page_times),
        "table_seconds": _summarize_metric(table_page_times),
        "numeric_seconds": _summarize_metric(numeric_page_times),
        "latex_seconds": _summarize_metric(latex_page_times),
        "hybrid_seconds": _summarize_metric(hybrid_page_times),
        "shared_repeat_seconds": _summarize_metric(shared_page_times),
        "char_eval_seconds": _summarize_metric(char_eval_times),
        "bad_char_ratio": _summarize_metric(bad_char_ratios),
    }
    (debug_dir/'summary.json').write_text(json.dumps(debug_summary,ensure_ascii=False,indent=2),encoding='utf-8')


def math_triage_compatibility_document(corpus, metrics):
    import json
    from .math import MathStage
    result,workspace=execute_compatibility_stage(corpus,MathStage(),[('extract_metrics',metrics)],
        StageSpecification('math.triage',('extract_metrics',),('math_triage',),settings={'compatibility_filename':metrics.name.removesuffix('.per_page.metrics.json')+'.pdf'}))
    if result.status!='ok':raise RuntimeError(f'{result.reason_code}: {result.error.message}')
    decision=json.loads((workspace/'triage.json').read_text())
    materialize_stage_outputs(corpus,result,workspace,{})
    recommendation=decision.pop('phase_recommended')
    return decision,recommendation
