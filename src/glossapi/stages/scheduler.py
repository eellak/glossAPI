"""Single-writer, bounded streaming DAG execution and ledger-only recovery.

Ready queues contain task IDs. Excess ready work stays planned in SQLite;
workers cannot publish, acquire a lease, or open a writable ledger.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time

from .contracts import ArtifactReference, DocumentSnapshot, StageSpecification
from .execution import sha256_file
from .sequential import SequentialStageRunner
from .workers import WarmWorkerPool, read_result
from ..ledger.encoding import snapshot_hash
from ..ledger.writer import new_id, utc_now


def execution_environment():
    """Stable executable identity, including unimported native stage binaries."""
    import importlib.metadata
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    files={str(path.relative_to(root)):sha256_file(path) for path in sorted(root.rglob('*.py'))}
    native={}
    for name in ('glossapi_rs_cleaner','glossapi_rs_noise','glossapi_rs_pdfrepair'):
        specification=importlib.util.find_spec(name)
        if specification and specification.origin:
            path=Path(specification.origin)
            binaries=list(path.parent.glob('*.so')) if path.suffix=='.py' else [path]
            native[name]=[sha256_file(binary) for binary in sorted(binaries)]
    packages=sorted((distribution.metadata['Name'],distribution.version) for distribution in importlib.metadata.distributions())
    models=[{'path':str(path.relative_to(root)),'sha256':sha256_file(path)} for path in sorted(root.rglob('*.joblib'))]
    environment={name:os.environ.get(name) for name in ('GLOSSAPI_IMAGES_SCALE','GLOSSAPI_FORMULA_BATCH',
        'DOCLING_ARTIFACTS_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')}
    return {'code':snapshot_hash(files),'environment':environment,'native':[{'name':name,'hashes':hashes} for name,hashes in native.items()],
            'models':models,'packages':[{'name':name,'version':version} for name,version in packages]}


def local_path(uri):
    from urllib.parse import unquote, urlsplit
    return Path(unquote(urlsplit(uri).path))


def stage_registry():
    from .download import DownloadStage
    from .extract import ExtractStage
    from .clean import CleanStage
    from .ocr import OcrStage
    from .math import MathStage
    from .sections import SectionsStage
    return {stage.name: stage for stage in (DownloadStage, ExtractStage, CleanStage, OcrStage, MathStage, SectionsStage)}


@dataclass(frozen=True)
class ScheduledOperation:
    key: str
    stage: str
    document: DocumentSnapshot
    specification: StageSpecification
    # (producer key, output kind, input role); multiple readers are explicit.
    inputs: tuple[tuple[str, str, str], ...] = ()
    after: tuple[str, ...] = ()
    # Optional condition: producer key, decision field, expected JSON value.
    condition: tuple | None = None
    cpu_tokens: int = 1
    memory_bytes: int = 0
    reservation_bytes: int = 0
    skip_reason: str | None = None


def operation_json(operation):
    value = asdict(operation)
    for artifact in value['document']['artifacts']:
        artifact['path'] = str(artifact['path'])
    return json.loads(json.dumps(value))


def operation_from_json(value):
    value = dict(value)
    document = dict(value['document'])
    document['artifacts'] = tuple(ArtifactReference(**{**artifact, 'path': Path(artifact['path']),
        'page_range': tuple(artifact['page_range']) if artifact['page_range'] else None}) for artifact in document['artifacts'])
    document['source_records'] = tuple(document['source_records'])
    specification = dict(value['specification'])
    for field in ('input_roles', 'required_output_kinds'):
        specification[field] = tuple(specification[field])
    return ScheduledOperation(**{**value, 'document': DocumentSnapshot(**document),
        'specification': StageSpecification(**specification), 'inputs': tuple(tuple(edge) for edge in value['inputs']),
        'after': tuple(value['after']), 'condition': tuple(value['condition']) if value['condition'] else None})


class StreamingScheduler:
    def __init__(self, writer, root, *, workers=None, cpu_tokens=None, memory_bytes=None,
                 disk_budget_bytes=None, registry=None, retry_limit=3, observer=None):
        self.writer = writer
        self.root = Path(root).resolve()
        self.runner = SequentialStageRunner(writer, root)
        available = max(1, (os.cpu_count() or 2)-1)
        self.workers = {'download': 4, 'extract': min(2, available), 'clean': 1, 'sections': 2, **(workers or {})}
        if any(type(count) is not int or count < 0 for count in self.workers.values()):
            raise ValueError('Worker counts must be nonnegative integers')
        self.cpu_capacity = available if cpu_tokens is None else cpu_tokens
        if self.cpu_capacity < 1:
            raise ValueError('CPU token capacity must be positive')
        self.memory_capacity = memory_bytes
        self.disk_capacity = disk_budget_bytes
        self.registry = registry or stage_registry()
        self.retry_limit = retry_limit
        self.observer = observer or (lambda *_: None)
        self.trace = []
        self.pools = {}
        self.ready = {}
        self.active = {}
        self.high_water = Counter()
        self.cancelled = False
        self.wait_reasons=set()

    def _wait(self,run,task,operation,reason):
        if (task,reason) in self.wait_reasons:
            return
        with self.writer.transaction():
            self.writer.event(run,event_kind='lifecycle',reason_code=reason,status='planned',task_id=task,
                version_id=operation.document.version_id)
        self.wait_reasons.add((task,reason))

    def _missing_package(self,operation):
        import importlib.metadata
        import re
        stage=self.registry[operation.stage]()
        if not hasattr(stage,'packages'):
            return False
        for requirement in stage.packages(operation.specification):
            for distribution in requirement.distributions:
                try:importlib.metadata.version(re.split(r'[<>=!~ ]',distribution,1)[0])
                except importlib.metadata.PackageNotFoundError:return True
            if requirement.entry_point_group and not any(
                point.name==requirement.entry_point_name for point in importlib.metadata.entry_points(group=requirement.entry_point_group)):
                return True
        return False

    def _trace(self, event, **details):
        row = dict(event=event, monotonic_ns=time.monotonic_ns(), **details)
        self.trace.append(row)
        self.observer(row)

    def freeze(self, operations, *, settings=None):
        operations = tuple(operations)
        keys = {operation.key for operation in operations}
        if len(keys) != len(operations):
            raise ValueError('Duplicate operation key')
        for operation in operations:
            if operation.stage not in self.registry:
                raise ValueError('Unsupported stage: '+operation.stage)
            if operation.specification.device_id != 'cpu':
                raise ValueError('Device is not qualified: '+operation.specification.device_id)
            if operation.stage=='download' and operation.specification.settings.get('download_mode','standard') not in (
                    'standard','default','http','browser','browser_protected','auto','browser_fallback'):
                raise ValueError('Unsupported download mode')
            filename_base=operation.specification.settings.get('filename_base') if operation.stage=='download' else None
            if filename_base is not None and (not isinstance(filename_base,str) or Path(filename_base).name!=filename_base or filename_base in ('.','..')):
                raise ValueError('Unsafe download filename base')
            if operation.cpu_tokens < 1 or operation.memory_bytes < 0 or operation.reservation_bytes < 0:
                raise ValueError('Invalid resource request')
            if operation.cpu_tokens < int(operation.specification.settings.get('num_threads') or 1):
                raise ValueError('Native threads exceed the declared CPU tokens')
            producers = {edge[0] for edge in operation.inputs} | set(operation.after)
            if operation.condition:
                producers.add(operation.condition[0])
            if not producers <= keys or operation.key in producers:
                raise ValueError('Invalid dependency')
        visited = set()
        def visit(key, path):
            if key in path:
                raise ValueError('Cyclic dependency')
            if key in visited:
                return
            operation = next(op for op in operations if op.key == key)
            for parent in {edge[0] for edge in operation.inputs} | set(operation.after) | ({operation.condition[0]} if operation.condition else set()):
                visit(parent, path | {key})
            visited.add(key)
        for key in keys:
            visit(key, set())
        from ..ledger.journal import code_revision, loaded_tools
        plan = dict(schema_version=1, mode='streaming', operations=[operation_json(op) for op in operations],
                    workers=self.workers, cpu_tokens=self.cpu_capacity, memory_bytes=self.memory_capacity,
                    disk_budget_bytes=self.disk_capacity, retry_limit=self.retry_limit,
                    publication_barriers=list((settings or {}).get('publication_barriers',())))
        settings = {'debug': 'off', 'memory_saving': 'normal', **(settings or {})}
        from .retention import resolved_settings
        from .disk_budget import require_disk_guard
        settings=resolved_settings(settings)
        require_disk_guard(self.root,settings)
        self.retention_settings=settings
        with self.writer.transaction():
            run = self.writer.run(settings, plan, execution_environment(), code_revision=code_revision())
            tasks = {}
            for ordinal, operation in enumerate(operations):
                task = self.writer.task(run, operation.stage, operation.specification.operation,
                    {'scheduler_operation': operation_json(operation)}, version_id=operation.document.version_id,
                    scope_key=operation.document.version_id or 'corpus:'+self.writer.snapshot('inputs',{'key':operation.key}), ordinal=ordinal)
                tasks[operation.key] = task
                self.writer.event(run, event_kind='lifecycle', reason_code='stage_planned', status='planned',
                    task_id=task, version_id=operation.document.version_id)
            for operation in operations:
                edges = {(edge[0], edge[2]) for edge in operation.inputs} | {(key, 'barrier') for key in operation.after}
                if operation.condition:
                    edges.add((operation.condition[0], 'condition'))
                for producer, role in edges:
                    self.writer.dependency(tasks[operation.key], tasks[producer], role)
            previous=None
            for index,stage in enumerate(plan['publication_barriers']):
                barrier=self.writer.task(run,stage,stage+'.compatibility_publish',{'barrier':stage},
                    scope_key='corpus:'+self.writer.snapshot('inputs',{'run_id':run}),ordinal=len(operations)+index)
                self.writer.event(run,event_kind='lifecycle',reason_code='waiting_for_corpus_seal',status='planned',task_id=barrier)
                if stage=='export':
                    producers=[self.writer.connection.execute('SELECT task_id FROM tasks WHERE run_id=? AND operation=?',(run,'clean.compatibility_publish')).fetchone()[0]] if 'clean' in plan['publication_barriers'] else list(tasks.values())
                else:producers=[previous] if previous else list(tasks.values())
                for producer in producers:
                    self.writer.dependency(barrier,producer,'corpus_seal')
                previous=barrier
            self.writer.event(run, event_kind='run', reason_code='stage_planned', details={'state':'running','plan_hash':snapshot_hash(plan)})
        return run

    def _load(self, run):
        row = self.writer.connection.execute('SELECT * FROM runs WHERE run_id=?',(run,)).fetchone()
        if row is None:
            raise ValueError('Unknown run')
        plan = json.loads(self.writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(row['plan_sha256'],)).fetchone()[0])
        if plan.get('mode') != 'streaming':
            raise ValueError('Run has no frozen streaming plan')
        self.workers = plan['workers']
        self.cpu_capacity = plan['cpu_tokens']
        self.memory_capacity = plan['memory_bytes']
        self.disk_capacity = plan['disk_budget_bytes']
        self.retry_limit = plan['retry_limit']
        operations = [operation_from_json(value) for value in plan['operations']]
        tasks = {row['ordinal']:row['task_id'] for row in self.writer.connection.execute('SELECT * FROM tasks WHERE run_id=? ORDER BY ordinal',(run,))}
        return operations, {operation.key:tasks[index] for index,operation in enumerate(operations)}, row

    def _outputs(self, attempt, *, include_deleted=False):
        prefix=(self.root/'.glossapi/artifacts').as_uri()+'/%'
        rows = self.writer.connection.execute("SELECT * FROM current_file_state WHERE producer_attempt_id=? AND uri LIKE ? AND (presence='present' OR (? AND presence='deleted')) ORDER BY event_id",(attempt,prefix,include_deleted)).fetchall()
        return tuple(ArtifactReference(row['file_id'], row['kind'], local_path(row['uri']), row['sha256'], row['bytes'],
            (row['page_first'],row['page_last']) if row['page_first'] else None) for row in rows)

    def _verified(self, attempt):
        rows=self.writer.connection.execute('SELECT * FROM current_file_state WHERE producer_attempt_id=? AND uri LIKE ?',
            (attempt,(self.root/'.glossapi/artifacts').as_uri()+'/%')).fetchall()
        if not rows:return False
        for row in rows:
            if row['presence']=='deleted':
                from .retention import retention_facts
                facts=retention_facts(self.writer.connection,row['file_id'])
                if not any(event['reason_code'] in ('delete_completed','delete_recovered') for event in facts['deletions']):return False
            elif row['presence']=='present':
                path=local_path(row['uri'])
                if not path.is_file() or path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:return False
            else:return False
        return True

    def _recover(self, run, tasks, operations):
        done = {}
        with self.writer.batch():
            for operation in operations:
                task = tasks[operation.key]
                row = self.writer.connection.execute("SELECT a.*,e.status,e.reason_code FROM attempts a LEFT JOIN events e ON e.event_id=(SELECT MAX(event_id) FROM events WHERE attempt_id=a.attempt_id AND event_kind='lifecycle') WHERE a.task_id=? ORDER BY a.attempt_number DESC LIMIT 1",(task,)).fetchone()
                if row is None:
                    continue
                if row['status'] == 'running' or row['status'] is None:
                    self.writer.finish(run,task,row['attempt_id'],status='failed',reason_code='attempt_interrupted',
                        details={'lease_token':row['lease_token'],'recovery':'old parent fenced before replacement'})
                elif row['status']=='ok' and self._verified(row['attempt_id']) and all(
                    artifact.path.is_file() and sha256_file(artifact.path)==artifact.sha256 for artifact in operation.document.artifacts):
                    done[operation.key] = row['attempt_id']
                    self._trace('reuse', key=operation.key, attempt_id=row['attempt_id'])
                elif row['status']=='ok':
                    self.writer.event(run,event_kind='invalidation',reason_code='output_hash_mismatch',task_id=task,
                        version_id=operation.document.version_id,details={'attempt_id':row['attempt_id']})
                # Uncommitted published directories never confer completion.
                if operation.key not in done:
                    for parent in (self.root/'.glossapi/staging'/row['attempt_id'],
                                   self.root/'.glossapi/artifacts'/(operation.document.version_id or 'corpus')/operation.stage/row['attempt_id']):
                        if parent.exists() and not self.writer.connection.execute('SELECT 1 FROM files WHERE producer_attempt_id=?',(row['attempt_id'],)).fetchone():
                            quarantine = self.root/'.glossapi/quarantine'/row['attempt_id']
                            quarantine.parent.mkdir(parents=True,exist_ok=True)
                            if not quarantine.exists():
                                os.rename(parent,quarantine)
                            self.writer.event(run,event_kind='invalidation',reason_code='attempt_interrupted',task_id=task,
                                version_id=operation.document.version_id,details={'quarantined_uri':quarantine.as_uri()})
        # A producer invalidation also invalidates every dependent completion.
        for operation in operations:
            if operation.key in done and any(edge[0] not in done for edge in operation.inputs):
                done.pop(operation.key)
        return done

    def execute(self, run, *, resume=False, retry_failed=False):
        from ..ledger.journal import code_revision, loaded_tools
        operations,tasks,run_row = self._load(run)
        if code_revision()!=run_row['code_revision']:
            raise ValueError('Frozen code revision changed; create an amended run')
        if snapshot_hash({'schema_version':1,'kind':'environment',**execution_environment()})!=run_row['environment_sha256']:
            raise ValueError('Frozen execution environment changed; create an amended run')
        frozen_settings=json.loads(self.writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(run_row['settings_sha256'],)).fetchone()[0])
        for binding in frozen_settings.get('semantic_files',[]):
            path=Path(binding['path'])
            if not path.is_file() or sha256_file(path)!=binding['sha256']:
                raise ValueError('Frozen model/dictionary input changed; create an amended run')
        from .disk_budget import DiskBudget, require_disk_guard
        require_disk_guard(self.root,frozen_settings)
        budget=DiskBudget(self.writer,self.root,run,{**frozen_settings,'disk_budget_bytes':self.disk_capacity})
        from .retention import RetentionController
        RetentionController(self.writer,self.root,run,frozen_settings).recover()
        done = self._recover(run,tasks,operations) if resume else {}
        terminal = set(done)
        if resume and not retry_failed:
            for operation in operations:
                state=self.writer.connection.execute('SELECT status,reason_code FROM current_task_state WHERE task_id=?',(tasks[operation.key],)).fetchone()
                count=self.writer.connection.execute('SELECT COUNT(*) FROM attempts WHERE task_id=?',(tasks[operation.key],)).fetchone()[0]
                if state['status']=='failed' and (count>=self.retry_limit or state['reason_code'] not in ('attempt_interrupted','worker_lost','worker_process_crashed','extract_timeout')):
                    terminal.add(operation.key)
                if state['status']=='skipped' and all(edge[0] in done for edge in operation.inputs):
                    terminal.add(operation.key)
        by_key = {operation.key:operation for operation in operations}
        consumers={key:[] for key in by_key}
        for operation in operations:
            for producer,_,_ in operation.inputs:consumers[producer].append(operation)
        tools = loaded_tools()
        active_cpu = active_memory = active_disk = 0
        try:
            while len(terminal)<len(operations):
                completions=[]
                # Drain every pool before admitting any new work.
                for pool in self.pools.values():
                    completions.extend(pool.poll())
                if completions:
                    with self.writer.batch():
                        for request,result in completions:
                            key,complete,operation = self.active.pop(request.attempt_id)
                            complete(result)
                            active_cpu-=operation.cpu_tokens
                            active_memory-=operation.memory_bytes
                            active_disk-=operation.reservation_bytes
                            budget.release(key)
                            self._trace('complete',key=key,attempt_id=request.attempt_id,status=result.status)
                            if result.status=='ok':
                                done[key]=request.attempt_id
                                terminal.add(key)
                            elif result.reason_code not in ('worker_lost','worker_process_crashed','extract_timeout') or self.writer.connection.execute('SELECT COUNT(*) FROM attempts WHERE task_id=?',(tasks[key],)).fetchone()[0]>=self.retry_limit:
                                terminal.add(key)
                self.runner.collect_scratch()
                if completions and not frozen_settings.get('publication_barriers'):
                    from .retention import retire_downloaded_sources
                    retire_downloaded_sources(self.writer,self.root,run,frozen_settings)
                queued={key for queue in self.ready.values() for key in queue}
                active_keys={entry[0] for entry in self.active.values()}
                # Finite scan pages: no eager Future or subprocess per document.
                for offset in range(0,len(operations),128):
                    for operation in operations[offset:offset+128]:
                        key=operation.key
                        if key in terminal or key in queued or key in active_keys:
                            continue
                        if operation.skip_reason:
                            self._skip(run,tasks[key],operation,operation.skip_reason)
                            terminal.add(key)
                            continue
                        dependencies={edge[0] for edge in operation.inputs}|set(operation.after)
                        if operation.condition:
                            dependencies.add(operation.condition[0])
                        if not dependencies<=terminal:
                            continue
                        required={edge[0] for edge in operation.inputs}|({operation.condition[0]} if operation.condition else set())
                        if any(parent not in done for parent in required):
                            self._skip(run,tasks[key],operation,'upstream_failed')
                            terminal.add(key)
                            continue
                        if operation.condition:
                            parent,field,expected=operation.condition
                            result=self.result(done[parent])
                            values=[decision.get(field) for decision in result.decisions if field in decision]
                            if values!=[expected]:
                                self._skip(run,tasks[key],operation,'not_in_plan')
                                terminal.add(key)
                                continue
                        pool_key=(operation.stage,operation.specification.device_id)
                        queue=self.ready.setdefault(pool_key,deque())
                        capacity=max(2,2*self.workers.get(operation.stage,1))
                        if len(queue)<capacity:
                            queue.append(key)
                            queued.add(key)
                            self.high_water[str(pool_key)]=max(self.high_water[str(pool_key)],len(queue))
                pending=[]
                for pool_key,queue in reversed(list(self.ready.items())):
                    stage_name,device=pool_key
                    count=self.workers.get(stage_name,1)
                    if not count:
                        for key in queue:
                            self._wait(run,tasks[key],by_key[key],'workers_exhausted_unattempted')
                        continue
                    if pool_key not in self.pools:
                        if stage_name=='download':
                            from .acquisition_pool import AsyncAcquisitionPool
                            self.pools[pool_key]=AsyncAcquisitionPool(count)
                        else:self.pools[pool_key]=WarmWorkerPool(count)
                    pool=self.pools[pool_key]
                    while queue and pool.active+sum(1 for item in pending if item[0] is pool)<pool.capacity:
                        operation=by_key[queue[0]]
                        if self._missing_package(operation):
                            self._wait(run,tasks[operation.key],operation,'waiting_for_package')
                            break
                        if active_cpu+operation.cpu_tokens>self.cpu_capacity or (
                            self.memory_capacity is not None and active_memory+operation.memory_bytes>self.memory_capacity):
                            self._wait(run,tasks[operation.key],operation,'waiting_for_device')
                            break
                        reservation=operation.reservation_bytes or (2*frozen_settings.get('max_attempt_bytes',0) if frozen_settings.get('memory_saving')=='extreme' else 0)
                        if self.disk_capacity is not None and not budget.can_reserve(reservation):
                            self._wait(run,tasks[operation.key],operation,'waiting_for_disk')
                            break
                        # Require downstream credits. At most queue capacity +
                        # already active producers can be ahead of a consumer.
                        children=consumers[operation.key]
                        if any(len(self.ready.get((child.stage,child.specification.device_id),()))>=max(2,2*self.workers.get(child.stage,1)) for child in children):
                            break
                        queue.popleft()
                        artifacts=list(operation.document.artifacts)
                        roles=list(operation.specification.input_roles[:len(artifacts)])
                        for producer,kind,role in operation.inputs:
                            matches=[artifact for artifact in self._outputs(done[producer]) if artifact.kind==kind]
                            if len(matches)!=1:
                                raise ValueError('Producer output role is absent or ambiguous: '+kind)
                            artifacts.extend(matches);roles.append(role)
                        document=replace(operation.document,artifacts=tuple(artifacts))
                        specification=replace(operation.specification,input_roles=tuple(roles),settings={**{name:frozen_settings[name] for name in ('debug','memory_saving','disk_budget_bytes','max_source_bytes','max_attempt_bytes') if name in frozen_settings},**operation.specification.settings})
                        fingerprint={'contract_version':self.registry[stage_name]().contract_version,
                            'specification':asdict(specification),'inputs':[{'sha256':a.sha256,'kind':a.kind,'range':a.page_range} for a in artifacts],
                            'code_revision':run_row['code_revision'],'environment_sha256':run_row['environment_sha256'],
                            'reuse_eligible':True,'scope':'same frozen run only; no cross-run semantic cache'}
                        pending.append((pool,operation,document,specification,fingerprint))
                        active_cpu+=operation.cpu_tokens
                        active_memory+=operation.memory_bytes
                        active_disk+=operation.reservation_bytes
                        if self.disk_capacity is not None:budget.reserve(operation.key,reservation)
                if pending:
                    # running leases and reservations commit BEFORE dispatch.
                    dispatch=[]
                    with self.writer.batch():
                        for pool,operation,document,specification,fingerprint in pending:
                            stage=self.registry[operation.stage]()
                            request,complete=self.runner.prepare(stage,document,specification,run_id=run,
                                planned_task_id=tasks[operation.key],fingerprint=json.loads(json.dumps(fingerprint)),tools=tools,
                                timeout_seconds=float(specification.settings.get('timeout_seconds',600)),
                                worker_id='warm:'+operation.stage+':'+specification.device_id)
                            self.writer.event(run,event_kind='heartbeat',reason_code='stage_started',task_id=request.task_id,
                                attempt_id=request.attempt_id,version_id=request.version_id,details={'lease_token':request.lease_token,
                                'lease_seconds':60,'reservation_bytes':budget.reservations.get(operation.key,operation.reservation_bytes),'cpu_tokens':operation.cpu_tokens})
                            dispatch.append((pool,stage,operation,request,complete))
                    for pool,stage,operation,request,complete in dispatch:
                        self.active[request.attempt_id]=(operation.key,complete,operation)
                        scheduler=self
                        class Reporter:
                            def heartbeat(self,lease,progress):
                                with scheduler.writer.transaction():
                                    scheduler.writer.event(run,event_kind='heartbeat',reason_code='stage_started',
                                        task_id=self.request.task_id,attempt_id=self.request.attempt_id,version_id=self.request.version_id,
                                        details={'lease_token':lease,**progress})
                        reporter=Reporter();reporter.request=request
                        pool.submit(stage,request,reporter=reporter)
                        self._trace('dispatch',key=operation.key,attempt_id=request.attempt_id)
                if not self.active and not pending:
                    break
                if not completions and not pending:
                    time.sleep(.005)
        except KeyboardInterrupt:
            self.cancelled=True
        finally:
            for pool in self.pools.values():
                pool.close()
            # Active attempts remain unfinished; recovery records interruption.
        counts=Counter(row[0] for row in self.writer.connection.execute('SELECT status FROM current_task_state WHERE run_id=? AND ordinal<?',(run,len(operations))))
        state='cancelled' if self.cancelled else 'waiting' if counts['planned'] or counts['running'] else 'completed_with_failures' if counts['failed'] else 'completed'
        with self.writer.transaction():
            self.writer.event(run,event_kind='run',reason_code='operator_cancelled' if self.cancelled else 'legacy_observed',
                details={'state':'waiting' if state=='completed' and self.writer.connection.execute("SELECT 1 FROM current_task_state WHERE run_id=? AND status='planned'",(run,)).fetchone() else state,
                    'queue_high_water':dict(self.high_water),'result_capacity':2*sum(pool.capacity for pool in self.pools.values())})
        return dict(run_id=run,state=state,counts={status:counts[status] for status in ('planned','running','ok','skipped','failed')},
                    attempts=done,trace=self.trace,queue_high_water=dict(self.high_water),disk=budget.report())

    def _skip(self,run,task,operation,reason):
        with self.writer.transaction():
            attempt=self.writer.attempt(task,worker_id='orchestrator',device_id=operation.specification.device_id,
                fingerprint={'reuse_eligible':False},inputs={'files':[]},settings={},tools={})
            self.writer.finish(run,task,attempt,status='skipped',reason_code=reason)

    def result(self,attempt):
        receipt=next(artifact for artifact in self._outputs(attempt) if artifact.kind=='worker_receipt')
        result=read_result(receipt.path)
        return replace(result,outputs=(*result.outputs,__import__('glossapi.stages.contracts',fromlist=['OutputArtifact']).OutputArtifact('worker_receipt','worker-result.json',None,'evidence',False)))
