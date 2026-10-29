"""Immutable product generations and compatibility mirrors, owned by the writer.

The ledger selects generations. Fixed paths are projections, never checkpoints.
The generation manifest is committed together with its complete output set.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
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
            boundary=connection.execute('SELECT details_json FROM events WHERE event_id=?',(cache['cursor'],)).fetchone()
            if boundary is None or boundary[0]!=cache['boundary']:
                cache.update(cursor=0,boundary=None,products={})
        for event_id,encoded in connection.execute("SELECT event_id,details_json FROM events WHERE event_id>? AND reason_code='legacy_observed' AND event_kind='decision' ORDER BY event_id",(cache['cursor'],)):
            digest=json.loads(encoded).get('evidence_sha256')
            row=connection.execute("SELECT json FROM snapshots WHERE sha256=? AND kind='product_projection'",(digest,)).fetchone()
            if row:
                value=json.loads(row[0]);cache['products'][value['product_uri']]=value
            cache.update(cursor=event_id,boundary=encoded)
        return cache['products'].get(uri)
    for encoded, in connection.execute("SELECT details_json FROM events WHERE reason_code='legacy_observed' AND event_kind='decision' ORDER BY event_id DESC"):
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
    if not selected.is_file() or sha256_file(selected) != value['sha256']:
        raise ValueError('Committed product is missing or corrupt: '+str(path))
    return selected


def execute_aggregate(corpus, stage, inputs, specification):
    journal = active_journal()
    references = []
    for role, given in inputs:
        path = Path(given).absolute()
        digest = sha256_file(path)
        file_id = None
        if journal:
            row = journal.writer.connection.execute("SELECT file_id,version_id FROM current_file_state WHERE uri=? AND sha256=? AND presence='present' ORDER BY event_id DESC LIMIT 1", (path.as_uri(),digest)).fetchone()
            if row and (role!='selected_text' or row['version_id'] is not None):
                file_id = row[0]
            else:
                with journal.writer.transaction():
                    version_id=None
                    if role=='selected_text':
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
        files.append({'path':relative,'kind':kind,'sha256':sha256_file(path),'bytes':path.stat().st_size})
        outputs.append(OutputArtifact(kind,relative,None,'final' if kind in ('sections_parquet','classified_parquet','fully_annotated_parquet','annotated_parquet','metadata_projection','document_jsonl','document_jsonl_shard') else 'intermediate',True))
    manifest={'schema_version':1,'operation':request.operation,'inputs':[{'file_id':a.file_id,'sha256':a.sha256,'role':a.kind} for a in request.inputs],
        'files':files,'membership':metrics or {}}
    (request.work_directory/'generation.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True)+'\n')
    outputs.append(OutputArtifact('generation_manifest','generation.json',None,'evidence',True))
    return StageResult(request.task_id,request.attempt_id,request.lease_token,'ok',reason,tuple(outputs),tuple(decisions),metrics or {})


def publish_products(corpus,result,workspace,products,*,retire=()):
    """Publish mirrors, then one ledger transaction selecting all immutable files."""
    if result.status != 'ok':
        if result.status == 'skipped':
            return
        raise RuntimeError(f'{result.reason_code}: {result.error.message if result.error else "failed product generation"}')
    journal=active_journal()
    selected=[]
    for relative,destination in products.items():
        source=checked_path(workspace,relative); destination=Path(destination).absolute()
        destination.parent.mkdir(parents=True,exist_ok=True)
        registered=journal.writer.connection.execute('SELECT sha256,bytes FROM files WHERE producer_attempt_id=? AND uri=?',(result.attempt_id,source.as_uri())).fetchone() if journal else None
        if registered and not destination.is_symlink() and destination.is_file() and destination.stat().st_size==registered['bytes'] and sha256_file(destination)==registered['sha256']:
            selected.append((source,destination))
            continue
        with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='.'+destination.name+'.',delete=False) as stream:
            temporary=Path(stream.name)
            try:
                with source.open('rb') as reader: shutil.copyfileobj(reader,stream)
                stream.flush();os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True);raise
        os.replace(temporary,destination)
        from .sequential import _fsync_directory
        _fsync_directory(destination.parent)
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
