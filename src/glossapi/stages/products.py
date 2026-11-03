"""Immutable product generations and compatibility mirrors, owned by the writer.

The ledger selects generations. Fixed paths are projections, never checkpoints.
The generation manifest is committed together with its complete output set.
"""
from __future__ import annotations
from .execution import write_text_hashed

from dataclasses import replace
import json
import os
from typing import Any
from pathlib import Path
import shutil
import tempfile
import time
import uuid
from urllib.parse import unquote, urlparse

from .contracts import ArtifactReference, DocumentSnapshot, OutputArtifact, StageRequest
from .execution import ReceiptCollector, checked_path, sha256_file


def active_journal():
    from ..ledger.journal import _active_journal, ResultJournal
    journal = _active_journal.get()
    return journal if isinstance(journal, ResultJournal) else None


def local_path(uri):
    return Path(unquote(urlparse(uri).path))


def current_product(connection, logical_path):
    """Resolve only a successful committed generation; never trust a mirror."""
    uri = Path(logical_path).absolute().as_uri()
    journal=active_journal()
    if journal and journal.writer.connection is connection and not connection.in_transaction:
        cache=getattr(journal,'product_projection_index',None)
        if cache is None:
            cache={'cursor':0,'boundary':None,'products':{}}
            journal.product_projection_index=cache
        if cache['cursor']:
            boundary=connection.execute(f'SELECT details_json FROM {connection.event_table} WHERE event_id=?',(cache['cursor'],)).fetchone()
            if boundary is None or boundary[0]!=cache['boundary']:
                cache.update(cursor=0,boundary=None,products={})
        for event_id,encoded in connection.execute(f"SELECT event_id,details_json FROM {connection.event_table} WHERE event_id>? AND reason_code='legacy_observed' AND event_kind='decision' ORDER BY event_id",(cache['cursor'],)):
            digest=json.loads(encoded).get('evidence_sha256')
            row=connection.execute("SELECT json FROM snapshots WHERE sha256=? AND kind='product_projection'",(digest,)).fetchone()
            if row:
                value=json.loads(row[0]);cache['products'][value['product_uri']]=value
            cache.update(cursor=event_id,boundary=encoded)
        return cache['products'].get(uri)
    for encoded, in connection.execute(f"SELECT details_json FROM {connection.event_table} WHERE reason_code='legacy_observed' AND event_kind='decision' ORDER BY event_id DESC"):
        digest=json.loads(encoded).get('evidence_sha256')
        row=connection.execute("SELECT json FROM snapshots WHERE sha256=? AND kind='product_projection'",(digest,)).fetchone()
        if row is None: continue
        value = json.loads(row[0])
        if value.get('product_uri') == uri:
            return value
    return None


def resolved_input(corpus, path):
    """Read registered immutable bytes after cutover, including with fresh Corpus."""
    path = Path(path).absolute()
    journal = active_journal()
    if journal:
        value = current_product(journal.writer.connection, path)
    elif (Path(corpus.output_dir)/'corpus.sqlite3').is_file():
        from ..ledger.writer import open_readonly
        with open_readonly(Path(corpus.output_dir)/'corpus.sqlite3') as connection:
            value = current_product(connection, path)
    else:
        value = None
    if not value:
        return path
    selected = local_path(value['immutable_uri'])
    if not selected.is_file() or (journal is None and sha256_file(selected) != value['sha256']):
        raise ValueError('Committed product is missing or corrupt: '+str(path))
    return selected


def producer_binding(journal, path):
    """Trust a committed producer receipt within its owning execution scope.

    External inputs are hashed by callers; real workers verify every consumed
    input, and the final mirror publisher checks the copied bytes once.
    """
    if journal is None:
        return None
    if Path(path).suffix=='.parquet':
        return None
    return journal.writer.connection.execute(
        "SELECT * FROM current_file_state WHERE uri=? AND producer_attempt_id IS NOT NULL AND presence='present' ORDER BY event_id DESC LIMIT 1",
        (Path(path).absolute().as_uri(),)).fetchone()


def _planned_text_version(journal: Any, path: Path, row: Any) -> Any:
    """The planned version of a selected text that a sibling publication of this run generated (K4g B1b).

    The text must be a registered generation whose producer attempt ran in a wrapper run of the same parent run as
    this aggregate, and its name (chunk suffix removed) must equal the frozen publication name of exactly one source
    this wrapper observed. Anything else (an older run's text, a caller's own file) returns None."""
    import re
    if row is None or not row['producer_attempt_id']:
        return None
    from ..ledger.reader import child_runs
    from .._naming import canonical_stem
    connection = journal.writer.connection
    children = child_runs(connection)
    parent = children.get(journal.run_id)
    producer = connection.execute('SELECT t.run_id FROM attempts a JOIN tasks t USING(task_id) WHERE a.attempt_id=?',
                                  (row['producer_attempt_id'],)).fetchone()
    if parent is None or producer is None or children.get(producer[0]) != parent:
        return None
    name = lambda value: re.sub(r'__p\d{4}-\d{4}$', '', canonical_stem(Path(value)))
    stem = name(path)
    versions = {version for source, version in journal.versions.items() if name(source) == stem}
    return next(iter(versions)) if len(versions) == 1 else None


def execute_aggregate(corpus: Any, stage: Any, inputs: Any, specification: Any) -> Any:
    journal = active_journal()
    references = []
    for role, given in inputs:
        path = Path(given).absolute()
        binding = producer_binding(journal, path)
        digest = binding['sha256'] if binding else sha256_file(path)
        file_id = None
        if journal:
            row = journal.writer.connection.execute("SELECT file_id,version_id,producer_attempt_id FROM current_file_state WHERE uri=? AND sha256=? AND presence='present' ORDER BY event_id DESC LIMIT 1", (path.as_uri(),digest)).fetchone()
            if row and (role!='selected_text' or row['version_id'] is not None):
                file_id = row[0]
            else:
                with journal.writer.transaction():
                    version_id=None
                    if role=='selected_text':
                        # A text this run's own publications generated without a version binding (a caller-kernel
                        # capture, e.g. the after-OCR clean text) belongs to the planned version whose frozen
                        # publication name it bears; only an outside text gets a document of its own (K4g B1b).
                        version_id=_planned_text_version(journal,path,row)
                        if version_id is None:
                            from ..ledger.journal import source_key
                            document_id=journal.writer.document(source_key(path.as_uri()))
                            version_id=journal.writer.version(document_id,source_sha256=digest,stored_sha256=digest)
                    file_id = journal.writer.artifact(journal.run_id,stage.name,role,path.as_uri(),sha256=digest,
                        bytes=path.stat().st_size,version_id=version_id,ownership='initial_external',retention_class='initial')
        references.append(ArtifactReference(file_id or str(uuid.uuid4()),role,path,digest,path.stat().st_size))
    if journal:
        if stage.name=='export':
            versions={}
            for reference in references:
                row=journal.writer.connection.execute('SELECT f.version_id,v.document_id FROM files f LEFT JOIN document_versions v ON v.version_id=f.version_id WHERE file_id=?',(reference.file_id,)).fetchone()
                versions[reference.file_id]={'version_id':row['version_id'],'document_id':row['document_id']}
            selected_sources=[]
            for path,version_id in journal.versions.items():
                row=journal.writer.connection.execute('SELECT document_id FROM document_versions WHERE version_id=?',(version_id,)).fetchone()
                selected_sources.append({'version_id':version_id,'document_id':row[0],'source_uri':Path(path).as_uri()})
            specification=replace(specification,settings={**specification.settings,'input_versions':versions,'selected_sources':selected_sources})
        from .sequential import SequentialStageRunner
        result, workspace = SequentialStageRunner(journal.writer,corpus.output_dir).execute(stage,
            DocumentSnapshot(None,None,'',artifacts=tuple(references)),specification,run_id=journal.run_id,aggregate=True)
        journal.adapter_stages.add(stage.name)
        journal.adapter_attempts.add(result.attempt_id)
    else:
        temporary = Path(tempfile.mkdtemp(prefix='glossapi-products-',dir=corpus.output_dir))
        request = StageRequest(*(str(uuid.uuid4()) for _ in range(5)),specification.operation,
            tuple(references),dict(specification.settings),{},specification.device_id,temporary/'attempt',time.monotonic_ns()+600_000_000_000)
        result = stage.finalize(request,reporter=ReceiptCollector())
        workspace = request.work_directory
    return result,workspace


def generation_outputs(request, paths, *, decisions=(), reason='legacy_observed', metrics=None):
    """Seal all files, including classifier partitions, in an atomic generation."""
    from .contracts import StageResult
    paths=list(paths)
    named={relative for relative,kind in paths}
    for path in sorted(request.work_directory.rglob('*')):
        if path.is_file() and str(path.relative_to(request.work_directory)) not in named:
            relative=str(path.relative_to(request.work_directory))
            paths.append((relative,'source_snapshot' if relative.startswith('inputs/') else 'attempt_diagnostic'))
    files=[]; outputs=[]
    for relative,kind in paths:
        path=checked_path(request.work_directory,relative)
        from .execution import produced_hash
        written=produced_hash(path)
        digest,size=written if written else (sha256_file(path),path.stat().st_size)
        files.append({'path':relative,'kind':kind,'sha256':digest,'bytes':size})
        outputs.append(OutputArtifact(kind,relative,None,'final' if kind in ('sections_parquet','classified_parquet','fully_annotated_parquet','annotated_parquet','metadata_projection','document_jsonl','document_jsonl_shard') else 'intermediate',True,digest,size))
    manifest={'schema_version':1,'operation':request.operation,'inputs':[{'file_id':a.file_id,'sha256':a.sha256,'role':a.kind} for a in request.inputs],
        'files':files,'membership':metrics or {}}
    write_text_hashed(request.work_directory/'generation.json',json.dumps(manifest,ensure_ascii=False,sort_keys=True)+'\n')
    outputs.append(OutputArtifact('generation_manifest','generation.json',None,'evidence',True))
    return StageResult(request.task_id,request.attempt_id,request.lease_token,'ok',reason,tuple(outputs),tuple(decisions),metrics or {})


def publish_products(corpus,result,workspace,products,*,retire=(),physical=None,captured=False):
    """Publish mirrors, then one ledger transaction selecting all immutable files.

    ``physical`` (``{relative: path}``): the file the bytes are atomically written to when it differs from the
    recorded destination, i.e. a metadata target resolved and locked by ``parquet_schema.metadata_target`` by a caller
    that holds it (K2F round 3); the ledger still records the destination as given.

    Every other product whose generation bytes are a selectable metadata parquet (``selectable_metadata_parquet``) is
    published through ``metadata_target`` here (K2F round 4, CP-K2e F1): its resolved physical target is locked (all
    such locks taken in path order before the first replacement) and replaced, so a symlinked metadata file keeps its
    link and a hard-linked one is refused. ``captured=True`` means the products are captures of their own destinations
    (the caller-kernel adapter): a metadata destination that no longer holds the captured bytes when its lock is held
    (a commit after capture, or a removal) raises ``MetadataStaleGenerationError`` before anything is replaced, instead
    of reverting that commit. No metadata lock is held while the generation is produced, so a callback's own metadata
    writes never nest inside it."""
    if result.status != 'ok':
        if result.status == 'skipped':
            return
        raise RuntimeError(f'{result.reason_code}: {result.error.message if result.error else "failed product generation"}')
    from contextlib import ExitStack
    from ..parquet_schema import MetadataStaleGenerationError,metadata_target,resolve_metadata_target,selectable_metadata_parquet
    from .execution import sha256_file
    journal=active_journal()
    with ExitStack() as held:
        written_paths=dict(physical or {})
        guarded=sorted((relative for relative in products if relative not in written_paths
                        and selectable_metadata_parquet(checked_path(workspace,relative))),
                       key=lambda relative:str(resolve_metadata_target(Path(products[relative]).absolute())))
        locked={}
        for relative in guarded:
            # Two product spellings of one physical file (e.g. a symlink and its target, both seen as changed) take its
            # lock once and publish to the same physical path.
            physical_target=resolve_metadata_target(Path(products[relative]).absolute())
            if physical_target not in locked:
                locked[physical_target]=held.enter_context(metadata_target(Path(products[relative]).absolute()))
            written_paths[relative]=locked[physical_target]
        if captured:
            for relative in guarded:
                live=written_paths[relative]
                if not live.is_file() or sha256_file(live)!=sha256_file(checked_path(workspace,relative)):
                    raise MetadataStaleGenerationError(
                        f'{products[relative]} changed after its generation was captured; republishing the capture would '
                        'revert that commit, so nothing was published (the live file keeps the newer bytes).')
        _publish_products_locked(corpus,result,workspace,products,journal,retire,written_paths)


def _publication_temporary(written):
    """A fresh temporary name beside ``written`` (hidden, same directory, so the replace is atomic), chosen before the
    intent record binds it (fix 8, N3); created exclusively and private (0600, as ``NamedTemporaryFile`` made it)."""
    return written.parent/('.'+written.name+'.'+uuid.uuid4().hex+'.tmp')


def _publish_products_locked(corpus,result,workspace,products,journal,retire,written_paths):
    selected=[]
    temporaries={}
    if journal:
        # The mirrors are replaced before the transaction below registers them. Their bytes (the registered immutable
        # generation's) are bound to their paths first, durably, in one record, as this run's own uncommitted writes
        # (fix 7, R1): if that registration then fails on a full ledger, the repeated publication rolls them back
        # instead of capturing them as unchanged or refusing them as a user's edit; if the ledger cannot take even this
        # record, nothing is replaced. The copies below stay one temporary file at a time, as before. Each record also
        # binds its temporary file's exact path and the immutable source it is copied from (fix 8, N3): a process death
        # between this record and a replace leaves a temporary that the repeated barrier verifies (recorded bytes, or a
        # prefix of the recorded source) and removes; nothing is removed for its name alone.
        intents=[]
        for relative,destination in products.items():
            source=checked_path(workspace,relative)
            registered=journal.writer.connection.execute('SELECT sha256,bytes FROM files WHERE producer_attempt_id=? AND uri=?',(result.attempt_id,source.as_uri())).fetchone()
            from .execution import sha256_file
            digest,size=(registered['sha256'],registered['bytes']) if registered else (sha256_file(source),source.stat().st_size)
            target=Path(written_paths.get(relative,Path(destination).absolute()))
            temporaries[relative]=_publication_temporary(target)
            intents.append((target,digest,size,os.path.lexists(target),{'temporary':temporaries[relative],'source':source.absolute()}))
        journal.writer.record_uncommitted_writes(journal.run_id,intents,wrapper_stage=journal.stage)
    for relative,destination in products.items():
        source=checked_path(workspace,relative); destination=Path(destination).absolute()
        written=Path(written_paths.get(relative,destination))
        written.parent.mkdir(parents=True,exist_ok=True)
        registered=journal.writer.connection.execute('SELECT sha256,bytes FROM files WHERE producer_attempt_id=? AND uri=?',(result.attempt_id,source.as_uri())).fetchone() if journal else None
        temporary=temporaries.get(relative) or _publication_temporary(written)
        with os.fdopen(os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as stream:
            try:
                import hashlib
                digest = hashlib.sha256(); size = 0
                with source.open('rb') as reader:
                    for block in iter(lambda:reader.read(1024*1024), b''):
                        stream.write(block); digest.update(block); size += len(block)
                stream.flush()
                if registered and (digest.hexdigest(),size)!=(registered['sha256'],registered['bytes']):
                    raise ValueError('Product hash changed before final publication')
            except BaseException:
                temporary.unlink(missing_ok=True);raise
        os.replace(temporary,written)
        selected.append((source,destination))
    if journal:
        with journal.writer.transaction():
            for source,destination in selected:
                row=journal.writer.connection.execute('SELECT * FROM files WHERE producer_attempt_id=? AND uri=?',(result.attempt_id,source.as_uri())).fetchone()
                if row is None: raise ValueError('Unregistered product generation')
                for prior in journal.writer.connection.execute("SELECT file_id,version_id FROM current_file_state WHERE uri=? AND sha256<>? AND presence='present'",(destination.as_uri(),row['sha256'])).fetchall():
                    journal.writer.event(journal.run_id,event_kind='artifact',reason_code='missing',file_id=prior['file_id'],version_id=prior['version_id'],
                        details={'observation':'compatibility_generation_superseded','replacement_attempt_id':result.attempt_id,
                            'immutable_generation_retained':True,'policy_deletion':False})
                journal.writer.artifact(journal.run_id,row['stage'],row['kind'],destination.as_uri(),version_id=row['version_id'],
                    producer_attempt_id=result.attempt_id,sha256=row['sha256'],bytes=row['bytes'],ownership='managed',retention_class=row['retention_class'])
                digest=journal.writer.snapshot('product_projection',{'kind':'product_projection','product_uri':destination.as_uri(),
                    'immutable_uri':source.as_uri(),'sha256':row['sha256'],'producer_attempt_id':result.attempt_id})
                journal.writer.event(journal.run_id,event_kind='decision',reason_code='legacy_observed',task_id=journal.task_id,
                    attempt_id=journal.attempt_id,details={'evidence_sha256':digest})
        from .retention import RetentionController
        controller=RetentionController(journal.writer,corpus.output_dir,journal.run_id,journal.retention_settings)
        for destination in retire:
            destination=Path(destination).absolute()
            row=journal.writer.connection.execute("SELECT * FROM current_file_state WHERE uri=? AND ownership='managed' AND presence='present' ORDER BY event_id DESC LIMIT 1",(destination.as_uri(),)).fetchone()
            if row is None:continue
            consumer=controller.seal([row])
            controller.declare(row['file_id'],[consumer],superseded_by=result.attempt_id)
            if not controller.delete(row['file_id']):
                raise ValueError('Superseded shard mirror could not be retired')
    else:
        shutil.rmtree(workspace.parent)
