"""Single-writer retention, exact consumer bindings and recoverable unlink.

No payload body is required by the read interfaces. A retention declaration is
immutable: later readers require a new plan, never an amendment to an old edge.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import json
import os
from pathlib import Path
import stat
import sqlite3
from urllib.parse import unquote, urlsplit

from .execution import sha256_file
from ..ledger.encoding import snapshot_hash


DEBUG_HELP = 'off keeps final outputs and evidence; on also keeps every intermediate and failed attempt'
MEMORY_SAVING_HELP = 'normal keeps initial files; extreme removes downloaded files after their last planned reader succeeds and requires a disk budget'


def resolved_settings(settings=None):
    result = {'debug': 'off', 'memory_saving': 'normal', **(settings or {})}
    if result['debug'] not in ('off', 'on') or result['memory_saving'] not in ('normal', 'extreme'):
        raise ValueError('Invalid debug or memory_saving setting')
    if result['memory_saving'] == 'extreme':
        for field in ('disk_budget_bytes', 'max_source_bytes', 'max_attempt_bytes'):
            if type(result.get(field)) is not int or result[field] <= 0:
                raise ValueError('disk_budget_exceeded: extreme requires positive '+field)
    return result


def file_path(row):
    return Path(unquote(urlsplit(row['uri']).path))


def retention_facts(connection, file_id, *, as_of=9223372036854775807):
    """Ledger-only facts shared by deletion and report --why."""
    if getattr(connection,'storage_version',0)>=3:
        compact=connection.execute('SELECT c.*,f.sha256,f.bytes FROM completed_deletions c JOIN files f USING(file_id) WHERE c.file_id=? AND c.event_id<=?',(file_id,as_of)).fetchone()
        if compact:
            fields=('file_id','event_id','run_id','reason_code','deletion_reason','last_consumer_stage','recorded_at','sha256','bytes')
            compact=dict(zip(fields,compact))
            consumers=[row[0] for row in connection.execute("SELECT DISTINCT e.event_id FROM attempt_inputs i JOIN events e USING(attempt_id) WHERE i.file_id=? AND i.sha256=? AND e.event_kind='lifecycle' AND e.status='ok' AND e.event_id<?",(file_id,compact['sha256'],compact['event_id']))]
            return dict(policy=compact['deletion_reason'],consumer_ok_event_ids=consumers,
                outstanding_consumers=[],pins=[],declaration={'compact_completed_delete':True,'deletion_reason':compact['deletion_reason']},
                deletions=[dict(event_id=compact['event_id'],reason_code=compact['reason_code'],
                    details={**compact,'history':'compact_completed_delete','deleted_logical_bytes':compact['bytes']})])
    declarations = []
    deletions = []
    for event in connection.execute("SELECT event_id,reason_code,details_json FROM events WHERE file_id=? AND event_kind='retention' AND event_id<=? ORDER BY event_id", (file_id,as_of)):
        value = dict(event_id=event[0], reason_code=event[1], details=json.loads(event[2]))
        if 'retention_plan' in value['details']:
            declarations.append(value['details']['retention_plan'])
        elif 'retention_plan_sha256' in value['details']:
            policy=connection.execute('SELECT json FROM snapshots WHERE sha256=?',(value['details']['retention_plan_sha256'],)).fetchone()
            binding=connection.execute('SELECT sha256,bytes FROM files WHERE file_id=?',(file_id,)).fetchone()
            if policy is None or binding is None:raise ValueError('Missing retention policy binding')
            plan=json.loads(policy[0])['plan']
            declarations.append(dict(file_id=file_id,sha256=binding[0],bytes=binding[1],**plan))
        else:
            if value['details'].get('intent_event_id') and value['reason_code'] in ('delete_completed','delete_recovered'):
                intent=connection.execute('SELECT details_json FROM events WHERE event_id=?',(value['details']['intent_event_id'],)).fetchone()
                if intent:value['details']={**json.loads(intent[0]),**value['details']}
            if value['details'].get('file_binding') == 'registered_file':
                binding=connection.execute('SELECT sha256,bytes FROM files WHERE file_id=?',(file_id,)).fetchone()
                value['details'].update(file_id=file_id,sha256=binding[0],bytes=binding[1])
                value['details'].setdefault('logical_references',[file_id])
            deletions.append(value)
    plan = declarations[-1] if declarations else None
    consumers, outstanding = [], []
    if plan:
        for task in plan['consumer_task_ids']:
            fact = connection.execute("""SELECT e.event_id,e.attempt_id FROM events e
                JOIN attempt_inputs i ON i.attempt_id=e.attempt_id
                WHERE e.task_id=? AND e.event_kind='lifecycle' AND e.status='ok'
                AND i.file_id=? AND i.sha256=? AND e.event_id<=? ORDER BY e.event_id DESC LIMIT 1""",
                (task, file_id, plan['sha256'],as_of)).fetchone()
            if fact:
                consumers.append(fact[0])
            elif task not in plan.get('waived_task_ids', []):
                outstanding.append(task)
    pins=list(plan.get('pins',[])) if plan else ['retention_not_declared']
    row=connection.execute('SELECT ownership,retention_class FROM files WHERE file_id=?',(file_id,)).fetchone()
    if row and row[0]!='managed':pins.append('external_original')
    if plan:
        superseded=plan.get('superseded_by')
        if not superseded and (plan['final'] or row[1]=='final'):pins.append('final_product')
        if row[1] in ('evidence','recovery'):pins.append('durable_evidence' if row[1]=='evidence' else 'incremental_recovery')
        if plan['downloaded_source']:
            if plan['settings']['memory_saving']=='normal':pins.append('initial_source_normal')
        elif row[1]=='initial':pins.append('initial_source')
        elif plan['settings']['debug']=='on' and not superseded:pins.append('debug_on')
        if outstanding:pins.append('outstanding_consumers')
        if not consumers:pins.append('no_successful_consumer')
    return dict(policy=plan['settings'] if plan else 'undeclared_keep',
        consumer_ok_event_ids=consumers, outstanding_consumers=outstanding,
        pins=sorted(set(pins)), deletions=deletions,
        declaration=plan)


class RetentionController:
    def __init__(self, writer, root, run_id, settings=None, *, fault=None):
        self.writer, self.connection = writer, writer.connection
        self.root, self.run_id = Path(root).resolve(), run_id
        self.settings = resolved_settings(settings)
        self.fault = fault or (lambda *_: None)

    def declare(self, file_id, consumers=(), *, pins=(), waived=(), final=False, source=False, superseded_by=None):
        row = self.connection.execute('SELECT * FROM files WHERE file_id=?', (file_id,)).fetchone()
        if row is None:
            raise ValueError('Unknown retention artifact')
        plan = dict(file_id=file_id, sha256=row['sha256'], bytes=row['bytes'],
            settings={key:self.settings[key] for key in ('debug', 'memory_saving')},
            consumer_task_ids=sorted(set(consumers)), waived_task_ids=sorted(set(waived)),
            pins=sorted(set(pins)), final=final, downloaded_source=source, superseded_by=superseded_by)
        for task in set(consumers) | set(waived):
            if self.connection.execute('SELECT 1 FROM tasks WHERE task_id=?', (task,)).fetchone() is None:
                raise ValueError('Unknown planned consumer')
        if not set(waived)<=set(consumers):raise ValueError('Only a declared edge can be waived')
        for task in waived:
            state=self.connection.execute('SELECT status,reason_code FROM current_task_state WHERE task_id=?',(task,)).fetchone()
            if tuple(state)!=('skipped','not_in_plan'):raise ValueError('Only explicitly unselected edges can be waived')
        previous = retention_facts(self.connection, file_id)['declaration']
        if previous:
            if previous != plan:
                raise ValueError('Retention consumers are frozen')
            return
        with self.writer.transaction():
            details={'retention_plan':plan}
            if self.connection.compact:
                shared={key:value for key,value in plan.items() if key not in ('file_id','sha256','bytes')}
                details={'retention_plan_sha256':self.writer.snapshot('retention_policy',{'plan':shared})}
            self.writer.event(self.run_id, event_kind='retention', reason_code='retention_pinned',
                file_id=file_id, version_id=row['version_id'], details=details)

    def eligibility(self, file_id):
        row = self.connection.execute('SELECT * FROM current_file_state WHERE file_id=?', (file_id,)).fetchone()
        facts = retention_facts(self.connection, file_id)
        if row['presence']=='deleted':return row,facts,['already_deleted']
        plan = facts['declaration']
        reasons = list(facts['pins'])
        if row['ownership'] != 'managed' or not file_path(row).absolute().is_relative_to(self.root):
            reasons.append('external_original')
        if not plan:
            return row, facts, reasons
        superseded=plan.get('superseded_by')
        if superseded and not self.connection.execute("SELECT 1 FROM events WHERE attempt_id=? AND event_kind='lifecycle' AND status='ok'",(superseded,)).fetchone():
            reasons.append('replacement_not_successful')
        if not superseded and (plan['final'] or row['retention_class'] == 'final'):
            reasons.append('final_product')
        if row['retention_class'] == 'recovery':
            reasons.append('incremental_recovery')
        if row['retention_class'] == 'evidence':
            reasons.append('durable_evidence')
        if plan['downloaded_source']:
            if self.settings['memory_saving'] != 'extreme':
                reasons.append('initial_source_normal')
        elif row['retention_class'] == 'initial':
            reasons.append('initial_source')
        elif self.settings['debug'] == 'on' and not superseded:
            reasons.append('debug_on')
        if facts['outstanding_consumers']:
            reasons.append('outstanding_consumers')
        if not facts['consumer_ok_event_ids']:
            reasons.append('no_successful_consumer')
        if self.connection.execute("""SELECT 1 FROM attempt_inputs i JOIN current_task_state t
            ON t.task_id=(SELECT task_id FROM attempts WHERE attempt_id=i.attempt_id)
            WHERE i.file_id=? AND t.status='running' LIMIT 1""", (file_id,)).fetchone():
            reasons.append('active_reader')
        # Partial extract success is a historical fact, never release authority.
        for task in plan['consumer_task_ids']:
            for details, in self.connection.execute("SELECT details_json FROM events WHERE task_id=? AND event_kind='decision'", (task,)):
                digest=json.loads(details).get('evidence_sha256')
                stored=self.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()
                if not stored:continue
                encoded=stored[0]
                from ..ledger.encoding import unpack_evidence
                evidence = unpack_evidence(json.loads(encoded))
                if evidence.get('coverage_allows_source_release') is False:
                    reasons.append('incomplete_coverage')
        return row, facts, sorted(set(reasons))

    def _identity(self, row):
        path = file_path(row)
        # Open every parent without following links, including corpus root.
        relative = path.absolute().relative_to(self.root)
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for name in relative.parts[:-1]:
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor); descriptor = child
            handle = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                before = os.fstat(handle)
                if not stat.S_ISREG(before.st_mode) or before.st_size != row['bytes']:
                    raise ValueError('Artifact identity changed')
                import hashlib
                digest = hashlib.sha256()
                with os.fdopen(os.dup(handle), 'rb') as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(chunk)
                after = os.fstat(handle)
                current = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
                identity = lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
                if identity(before) != identity(after) or identity(after) != identity(current) or digest.hexdigest() != row['sha256']:
                    raise ValueError('Artifact hash or identity changed')
            finally:
                os.close(handle)
            return descriptor, relative.name, identity(after)
        except FileNotFoundError:
            # Recovery can observe an unlink whose completion was interrupted.
            # Flush its nearest existing parent before recording completion.
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
            raise
        except BaseException:
            os.close(descriptor); raise

    def delete(self, file_id, *, recovering=None):
        if self.connection.in_transaction and recovering is None:
            raise ValueError('Deletion intent must commit before unlink')
        row, facts, pins = self.eligibility(file_id)
        if pins or row['presence'] == 'deleted':
            return False
        # Several logical registrations may share the same physical pathname.
        aliases = self.connection.execute("SELECT file_id FROM current_file_state WHERE uri=? AND presence='present'", (row['uri'],)).fetchall()
        if any(self.eligibility(alias[0])[2] for alias in aliases):
            return False
        path=file_path(row)
        if path.exists() and not path.is_symlink() and path.stat().st_nlink>1:
            identity=path.stat()
            for shared in self.connection.execute("SELECT * FROM current_file_state WHERE presence='present' AND uri<>?",(row['uri'],)):
                candidate=file_path(shared)
                try:other=candidate.lstat()
                except OSError:continue
                if (other.st_dev,other.st_ino)==(identity.st_dev,identity.st_ino) and self.eligibility(shared['file_id'])[2]:return False
        intent = recovering
        details = dict(file_id=file_id, sha256=row['sha256'], bytes=row['bytes'],
            policy_sha256=snapshot_hash(facts['declaration']), consumer_ok_event_ids=facts['consumer_ok_event_ids'],
            pin_check=[], logical_references=[alias[0] for alias in aliases])
        descriptor = None
        try:
            try:
                descriptor, name, identity = self._identity(row)
            except FileNotFoundError:
                if intent is None:
                    with self.writer.transaction():
                        self.writer.event(self.run_id,event_kind='artifact',reason_code='missing',file_id=file_id,
                            version_id=row['version_id'],details={'policy_deletion':False})
                    return False
                return self._complete(row, details, 'delete_recovered', intent)
            if intent is None:
                with self.writer.transaction():
                    intent = self.writer.event(self.run_id,event_kind='retention',reason_code='delete_planned',
                        file_id=file_id,version_id=row['version_id'],details=self._compact_receipt({**details,'identity':list(identity)}))
                self.writer.durable_barrier()
                self.fault('after_intent', file_id)
            # Pins and leases can only change through this writer. Recheck the
            # path identity under the same opened directory immediately before unlink.
            if self.eligibility(file_id)[2]:
                return False
            current = os.stat(name,dir_fd=descriptor,follow_symlinks=False)
            if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns) != identity:
                raise ValueError('Artifact replaced before unlink')
            free_before = os.statvfs(self.root).f_bavail * os.statvfs(self.root).f_frsize
            os.unlink(name,dir_fd=descriptor)
            if hasattr(self,'_directory_syncs'):
                directory=os.fstat(descriptor);key=(directory.st_dev,directory.st_ino)
                if key not in self._directory_syncs:self._directory_syncs[key]=os.dup(descriptor)
            else:os.fsync(descriptor)
            self.fault('after_unlink', file_id)
            free_after = os.statvfs(self.root).f_bavail * os.statvfs(self.root).f_frsize
            details['observed_free_bytes_change'] = free_after-free_before
            return self._complete(row,details,'delete_completed',intent)
        except (OSError, ValueError) as error:
            with self.writer.transaction():
                self.writer.event(self.run_id,event_kind='retention',reason_code='delete_failed',file_id=file_id,
                    version_id=row['version_id'],message=str(error),details={**details,'intent_event_id':intent,
                        'errno':getattr(error,'errno',None),'error_type':type(error).__name__})
            return False
        finally:
            if descriptor is not None:os.close(descriptor)

    def _compact_receipt(self,details):
        if not self.connection.compact:return details
        result={key:value for key,value in details.items() if key not in ('file_id','sha256','bytes')}
        if result.get('logical_references')==[details.get('file_id')]:result.pop('logical_references')
        if 'intent_event_id' in result and 'deleted_logical_bytes' in result:
            result={key:value for key,value in result.items() if key not in ('policy_sha256','consumer_ok_event_ids','pin_check')}
        return {'file_binding':'registered_file',**result}

    def _complete(self,row,details,reason,intent):
        with self.writer.transaction():
            for file_id in details['logical_references']:
                self.writer.event(self.run_id,event_kind='artifact',reason_code='deleted',file_id=file_id,
                    version_id=self.connection.execute('SELECT version_id FROM files WHERE file_id=?',(file_id,)).fetchone()[0],details={'intent_event_id':intent,'sha256':row['sha256']})
            if getattr(self.connection,'storage_version',0)>=3 and self.settings['debug']=='off':
                for alias in details['logical_references']:
                    pending=getattr(self,'_completed_policy_hashes',set())
                    for encoded, in self.connection.execute("SELECT details_json FROM retention_events WHERE file_id=? AND reason_code='retention_pinned'",(alias,)):
                        digest=json.loads(encoded).get('retention_plan_sha256')
                        if digest:pending.add(digest)
                    self._completed_policy_hashes=pending
                    plan=retention_facts(self.connection,alias)['declaration']
                    deletion_reason=('extreme_after_last_planned_reader' if plan['downloaded_source'] else
                        'superseded_generation_after_last_consumer' if plan.get('superseded_by') else
                        'debug_off_after_last_consumer')
                    consumer=self.connection.execute("SELECT t.stage FROM events e JOIN tasks t USING(task_id) JOIN attempt_inputs i USING(attempt_id) WHERE i.file_id=? AND e.event_kind='lifecycle' AND e.status='ok' ORDER BY e.event_id DESC LIMIT 1",(alias,)).fetchone()
                    if consumer is None:raise ValueError('Completed deletion lacks an exact successful consumer')
                    from ..ledger.writer import utc_now
                    event_id=self.writer.next_event_id;self.writer.next_event_id+=1
                    self.connection.execute('INSERT INTO completed_deletions VALUES(?,?,?,?,?,?,?)',
                        (alias,event_id,self.run_id,reason,deletion_reason,consumer[0],utc_now()))
                    self.connection.execute('DELETE FROM retention_events WHERE file_id=?',(alias,))
                if not getattr(self.writer,'_batch_active',False):self._prune_completed_policies()
                return True
            for alias in details['logical_references']:
                if alias==row['file_id']:continue
                self.writer.event(self.run_id,event_kind='retention',reason_code=reason,file_id=alias,
                    version_id=self.connection.execute('SELECT version_id FROM files WHERE file_id=?',(alias,)).fetchone()[0],
                    details=self._compact_receipt({**details,'intent_event_id':intent,'shared_path_primary_file_id':row['file_id'],'deleted_logical_bytes':0}))
            self.writer.event(self.run_id,event_kind='retention',reason_code=reason,file_id=row['file_id'],
                version_id=row['version_id'],details=self._compact_receipt({**details,'intent_event_id':intent,'deleted_logical_bytes':row['bytes']}))
        return True

    def _prune_completed_policies(self):
        """Drop only completed policies with no remaining live journal reference."""
        if getattr(self.connection,'storage_version',0)<3 or self.settings['debug']!='off':return
        pending=getattr(self,'_completed_policy_hashes',set())
        if not pending:
            # A process can die after the completion commit and before pruning.
            pending={row[0] for row in self.connection.execute("SELECT sha256 FROM snapshots WHERE kind='retention_policy'")}
        if not pending:return
        live=set()
        for encoded, in self.connection.execute("SELECT details_json FROM retention_events WHERE reason_code='retention_pinned'"):
            digest=json.loads(encoded).get('retention_plan_sha256')
            if digest:live.add(digest)
        with nullcontext() if self.connection.in_transaction else self.writer.transaction():
            for digest in pending-live:
                try:self.connection.execute("DELETE FROM snapshots WHERE sha256=? AND kind='retention_policy'",(digest,))
                except sqlite3.IntegrityError:
                    # A separately registered fact still requires this snapshot.
                    pass
        self._completed_policy_hashes=set()

    def delete_many(self,file_ids):
        intents=[]
        # All intents reach stable storage before any unlink. Completions are
        # another bounded transaction; a death anywhere leaves recoverable intents.
        for offset in range(0,len(file_ids),128):
            with self.writer.transaction():
                for file_id in file_ids[offset:offset+128]:
                    row,facts,pins=self.eligibility(file_id)
                    if pins or row['presence']!='present':continue
                    details=dict(file_id=file_id,sha256=row['sha256'],bytes=row['bytes'],
                        policy_sha256=snapshot_hash(facts['declaration']),consumer_ok_event_ids=facts['consumer_ok_event_ids'],pin_check=[])
                    intent=self.writer.event(self.run_id,event_kind='retention',reason_code='delete_planned',
                        file_id=file_id,version_id=row['version_id'],details=self._compact_receipt(details))
                    intents.append((file_id,intent))
            self.writer.durable_barrier()
            batch=intents;intents=[]
            self._directory_syncs={}
            try:
                with self.writer.batch():
                    try:
                        for file_id,intent in batch:
                            self.fault('after_intent',file_id)
                            self.delete(file_id,recovering=intent)
                    finally:
                        # Every affected parent reaches stable storage before
                        # this transaction can commit any deletion completion.
                        for descriptor in self._directory_syncs.values():os.fsync(descriptor)
            finally:
                for descriptor in self._directory_syncs.values():os.close(descriptor)
                del self._directory_syncs
            self._prune_completed_policies()

    def recover(self):
        self._prune_completed_policies()
        intents = list(self.connection.execute("""SELECT event_id,file_id FROM events e
            WHERE event_kind='retention' AND reason_code='delete_planned' AND NOT EXISTS
            (SELECT 1 FROM events c WHERE c.file_id=e.file_id AND c.event_id>e.event_id
             AND c.reason_code IN ('delete_completed','delete_recovered','delete_failed'))"""))
        # The dead writer may have committed its NORMAL intent and died while
        # flushing consumers, before its FULL barrier. Never infer durability
        # merely from a visible intent after process death.
        if intents:self.writer.durable_barrier()
        return sum(self.delete(row[1],recovering=row[0]) for row in intents)

    def seal(self, rows):
        """A real, hash-bound reader for otherwise unconsumed diagnostics."""
        from ..ledger.writer import new_id
        rows = list(rows)
        if not rows:return None
        # Read and classify every byte before recording the diagnostic consumer.
        files = []
        for row in rows:
            path=file_path(row)
            if path.is_symlink() or not path.is_file() or sha256_file(path)!=row['sha256']:
                raise ValueError('Evidence-sealing input changed')
            files.append({'file_id':row['file_id'],'sha256':row['sha256']})
        with self.writer.transaction():
            manifest=self.writer.snapshot('inputs',{'kind':'inputs','files':files})
            ordinal=self.connection.execute('SELECT COALESCE(MAX(ordinal),-1)+1 FROM tasks WHERE run_id=?',(self.run_id,)).fetchone()[0]
            task=self.writer.task(self.run_id,rows[0]['stage'],rows[0]['stage']+'.seal_evidence:'+new_id(),
                {'input_manifest_sha256':manifest,'diagnostic_bodies':'debug_on_only'},scope_key='corpus:'+manifest,ordinal=ordinal)
            attempt=self.writer.attempt(task,worker_id='evidence-sealer',device_id='cpu',
                fingerprint={'reuse_eligible':False},inputs={'files':files},settings=self.settings,tools={})
            for row in rows:self.writer.input(attempt,row['file_id'],'sealed:'+row['file_id'],row['sha256'])
            self.writer.finish(self.run_id,task,attempt,status='ok',reason_code=rows[0]['stage']+'_completed',
                details={'input_manifest_sha256':manifest,'sealed_file_count':len(files),'payload_bodies_retained':self.settings['debug']=='on'})
        return task


def retain_completed_run(writer,root,run_id,settings=None,*,attempts=None,cleanup_paths=(),observer=None):
    """Seal and collect only owned private attempts after their publication seal.

    Public wrapper products are requested outputs. The streaming plan supplies
    all document readers and publication barriers before this final collection.
    """
    controller=RetentionController(writer,root,run_id,settings)
    controller.recover()
    if attempts is None:
        attempts=[row[0] for row in writer.connection.execute('SELECT attempt_id FROM attempts a JOIN tasks t USING(task_id) WHERE t.run_id=?',(run_id,))]
    rows=[]
    for attempt in attempts:
        rows.extend(writer.connection.execute("SELECT * FROM current_file_state WHERE producer_attempt_id=? AND ownership='managed' AND presence='present'",(attempt,)).fetchall())
    private=Path(root).resolve()/'.glossapi/artifacts'
    projections={}
    for encoded, in writer.connection.execute("SELECT json FROM snapshots WHERE kind='product_projection' ORDER BY rowid"):
        value=json.loads(encoded);projections[value['product_uri']]=value
    current_backings={value['immutable_uri'] for value in projections.values()}
    candidates=[]
    for row in rows:
        path=file_path(row)
        superseded_public_raw=(row['kind']=='raw_markdown' and path.is_relative_to(Path(root).resolve()/'markdown') and 'clean' in controller.settings.get('requested_stages',()))
        public_temporary=row['kind'] in ('annotation_partition','repaired_pdf','temporary_cleaning_report')
        if not path.is_relative_to(private) and path not in cleanup_paths and not superseded_public_raw and not public_temporary:continue
        if row['retention_class'] in ('final','evidence','recovery') or (row['uri'] in current_backings and row['kind']!='annotation_partition'):continue
        if controller.settings['debug']=='on' and not (row['kind']=='download_payload' and controller.settings['memory_saving']=='extreme'):continue
        if row['kind']=='download_payload' and not writer.connection.execute('SELECT 1 FROM attempt_inputs WHERE file_id=?',(row['file_id'],)).fetchone():continue
        # Worker receipts are recovery frontiers and must remain replayable.
        if row['kind'] in ('worker_receipt','generation_manifest','ocr_native','ocr_page_state'):continue
        producer=writer.connection.execute("SELECT status FROM current_task_state WHERE task_id=(SELECT task_id FROM attempts WHERE attempt_id=?)",(row['producer_attempt_id'],)).fetchone()
        if not producer or producer[0]!='ok':continue
        if row['stage']=='extract':
            from ..ledger.encoding import unpack_evidence
            partial=False
            for detail, in writer.connection.execute("SELECT details_json FROM events WHERE attempt_id=? AND event_kind='decision'",(row['producer_attempt_id'],)):
                digest=json.loads(detail).get('evidence_sha256')
                stored=writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()
                if stored and unpack_evidence(json.loads(stored[0])).get('coverage_allows_source_release') is False:partial=True
            if partial:continue
        candidates.append(row)
    by_stage=defaultdict(list)
    for row in candidates:
        if not retention_facts(writer.connection,row['file_id'])['declaration']:by_stage[row['stage']].append(row)
    for stage,files in by_stage.items():
        sealer=controller.seal(files)
        for offset in range(0,len(files),128):
            with writer.batch():
                for row in files[offset:offset+128]:
                    consumers=[item[0] for item in writer.connection.execute('SELECT DISTINCT a.task_id FROM attempt_inputs i JOIN attempts a USING(attempt_id) WHERE i.file_id=?',(row['file_id'],))]
                    # Raw is a final product when no real text reader selected it.
                    final=row['kind'] in ('clean_markdown','ocr_markdown','math_markdown') or (row['kind']=='raw_markdown' and len(consumers)==1)
                    controller.declare(row['file_id'],consumers,final=final,source=row['kind']=='download_payload')
    if observer:observer(controller,candidates)
    controller.delete_many([row['file_id'] for row in candidates])
    return controller


def defer_owned_cleanup(corpus,path):
    from .products import active_journal
    journal=active_journal()
    if not journal:return False
    path=Path(path).absolute()
    with journal.writer.transaction():
        journal.writer.artifact(journal.run_id,journal.stage,'temporary_cleaning_report',path.as_uri(),
            producer_attempt_id=journal.attempt_id,sha256=sha256_file(path),bytes=path.stat().st_size)
    journal.cleanup_paths=getattr(journal,'cleanup_paths',set())|{path}
    return True


def retention_summary(connection,run=None):
    groups={}
    for kind,presence,size in connection.execute("SELECT kind,presence,COALESCE(SUM(bytes),0) FROM current_file_state GROUP BY kind,presence"):
        group=groups.setdefault(kind,dict(present=0,deleted=0,missing=0,corrupt=0,pinned=0))
        group[presence]=size
    for kind,size in connection.execute("SELECT kind,COALESCE(SUM(bytes),0) FROM current_file_state WHERE presence='present' AND (ownership<>'managed' OR retention_class IN ('initial','final','evidence','recovery')) GROUP BY kind"):
        groups[kind]['pinned']=size
    settings={};resources=dict(present_bytes=None,reserved_bytes=None,free_bytes=None)
    if run:
        stored=connection.execute('SELECT json FROM snapshots WHERE sha256=?',(run['settings_sha256'],)).fetchone()
        if stored:settings=json.loads(stored[0])
        latest=connection.execute("SELECT details_json FROM events WHERE run_id=? AND event_kind='run' ORDER BY event_id DESC LIMIT 1",(run['run_id'],)).fetchone()
        reference=json.loads(latest[0]).get('result_sha256') if latest else None
        recorded=connection.execute('SELECT json FROM snapshots WHERE sha256=?',(reference,)).fetchone()
        if recorded:resources=json.loads(recorded[0])['result'].get('disk',resources)
    deleted=connection.execute("SELECT COALESCE(SUM(bytes),0) FROM (SELECT uri,MAX(bytes) bytes FROM current_file_state WHERE presence='deleted' GROUP BY uri)").fetchone()[0]
    return dict(settings={key:settings.get(key,default) for key,default in [('debug','off'),('memory_saving','normal')]},
        source_exception=MEMORY_SAVING_HELP,by_kind_bytes=groups,deleted_logical_bytes=deleted,resources=resources,
        disk_budget_bytes=settings.get('disk_budget_bytes'),free_space_basis='last recorded statvfs resource observation; report does not scan payloads')


def retire_downloaded_sources(writer,root,run_id,settings):
    """Release original download bytes after the frozen plan's last PDF reader.

    Compatibility extraction is an actual reader and is therefore a planned
    publication consumer. Evaluation and report commands never add reader edges.
    """
    if settings.get('memory_saving','normal')!='extreme':return
    controller=RetentionController(writer,root,run_id,settings)
    run=writer.connection.execute('SELECT plan_sha256 FROM runs WHERE run_id=?',(run_id,)).fetchone()
    plan=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',run).fetchone()[0])
    operations=plan.get('operations',[])
    tasks={row['ordinal']:row for row in writer.connection.execute('SELECT * FROM current_task_state WHERE run_id=?',(run_id,))}
    versions={value['document']['version_id'] for value in operations if value['stage']=='download'}
    publication=writer.connection.execute("SELECT * FROM current_task_state WHERE run_id=? AND operation='extract.compatibility_publish'",(run_id,)).fetchone()
    if publication and publication['status']!='ok':return
    # The original payload and byte-identical staging snapshots are one source
    # footprint; extreme must not leave a hidden whole-PDF copy behind.
    for version in versions:
        readers=[];waived=[];ready=True
        for ordinal,value in enumerate(operations):
            if value['document']['version_id']!=version:continue
            roles=value['specification']['input_roles']+[edge[2] for edge in value['inputs']]
            if not set(roles)&{'source_payload','pdf','source_pdf'}:continue
            task=tasks[ordinal]
            if task['status']=='skipped' and task['reason_code']=='not_in_plan':waived.append(task['task_id'])
            elif task['status']!='ok':ready=False
            else:readers.append(task['task_id'])
        if not ready or not readers:continue
        if publication:readers.append(publication['task_id'])
        digest=writer.connection.execute('SELECT stored_sha256 FROM document_versions WHERE version_id=?',(version,)).fetchone()[0]
        rows=writer.connection.execute("SELECT * FROM current_file_state WHERE version_id=? AND sha256=? AND ownership='managed' AND presence='present' AND kind IN ('download_payload','source_snapshot','scratch_copy')",(version,digest)).fetchall()
        if not rows:continue
        # Verify source-hash bindings and full coverage on the real readers.
        valid=True
        for task in readers:
            if not writer.connection.execute("SELECT 1 FROM attempt_inputs i JOIN attempts a USING(attempt_id) JOIN events e USING(attempt_id) WHERE a.task_id=? AND i.sha256=? AND e.event_kind='lifecycle' AND e.status='ok'",(task,digest)).fetchone():valid=False
            for detail, in writer.connection.execute("SELECT details_json FROM events WHERE task_id=? AND event_kind='decision'",(task,)):
                reference=json.loads(detail).get('evidence_sha256')
                stored=writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(reference,)).fetchone()
                if stored:
                    from ..ledger.encoding import unpack_evidence
                    value=unpack_evidence(json.loads(stored[0]))
                    if 'coverage_allows_source_release' in value and value['coverage_allows_source_release'] is not True:valid=False
        if not valid:continue
        new_rows=[row for row in rows if not retention_facts(writer.connection,row['file_id'])['declaration']]
        sealer=controller.seal(new_rows)
        if sealer:
            with writer.transaction():
                for reader in readers:writer.dependency(sealer,reader,'source_release_gate')
        for row in new_rows:
            exact=[task for task in readers if writer.connection.execute("SELECT 1 FROM attempt_inputs i JOIN attempts a USING(attempt_id) WHERE a.task_id=? AND i.file_id=? AND i.sha256=?",(task,row['file_id'],digest)).fetchone()]
            controller.declare(row['file_id'],[*exact,sealer],source=True)
        controller.delete_many([row['file_id'] for row in rows])
