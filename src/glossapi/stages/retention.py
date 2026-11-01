"""Single-writer retention, exact consumer bindings and recoverable unlink.

No payload body is required by the read interfaces. A retention declaration is
immutable: later readers require a new plan, never an amendment to an old edge.
"""
from __future__ import annotations

from typing import Any, cast

from collections import defaultdict
from contextlib import nullcontext
import functools
import json
import os
from pathlib import Path
import stat
import sqlite3
from urllib.parse import unquote, urlsplit

from .execution import sha256_file
from ..ledger.encoding import snapshot_hash
from ..settings_contract import setting_help


# The owner's settings text (PLAN section 2.2), from its single source.
DEBUG_HELP = setting_help('debug')
MEMORY_SAVING_HELP = setting_help('memory_saving')

# Retention contract (one rule per ownership class; PLAN.md section 2.3):
# - supplied local originals (ownership initial_external / legacy_external, or
#   any path outside the corpus root) are never deleted, in any setting;
# - downloads are deleted only with memory_saving=extreme, only when this run
#   (or a compatibility wrapper run it created) registered them, only after
#   the last planned reader in the frozen plan succeeded on the exact hash,
#   and only after a durability barrier covers the replacement outputs;
# - deletion opens every path component without following links, so a
#   symlinked file or directory alias is never followed to its target;
# - debug=on keeps intermediates and failed attempts, but does not cancel the
#   extreme exception for owned downloads (the run summary reports it).


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
    return _uri_path(row['uri'])


@functools.lru_cache(maxsize=1<<16)
def _uri_path(uri):
    return Path(unquote(urlsplit(uri).path))


def path_alias(root, path):
    """True when ``path`` is outside ``root`` or reached through any symlink below it."""
    root=Path(root)
    try:relative=Path(path).absolute().relative_to(root)
    except ValueError:return True
    current=root
    for part in relative.parts:
        current=current/part
        if current.is_symlink():return True
    return False


def run_family(connection, run_id):
    """This run plus the compatibility wrapper runs it created."""
    family={run_id}
    table=getattr(connection,'event_table','events')
    for encoded, in connection.execute(f"SELECT details_json FROM {table} WHERE run_id=? AND event_kind='run' AND message LIKE 'wrapper:%'",(run_id,)):
        child=json.loads(encoded).get('wrapper_run_id')
        if child:family.add(child)
    return family


def registering_run(connection, file_id):
    """The run whose first artifact fact registered this file (its producer's run)."""
    table=getattr(connection,'event_table','events')
    row=connection.execute(f"SELECT run_id FROM {table} WHERE file_id=? AND event_kind='artifact' ORDER BY event_id LIMIT 1",(file_id,)).fetchone()
    return row[0] if row else None


# Compatibility copies published outside .glossapi/ that every setting keeps (the debug-off product set: cleaned text and
# the per-document extraction sidecar). Declared here by kind; the plan preview lists them (W18 S2).
KEPT_PUBLISHED_KINDS=frozenset(('clean_markdown','extract_sidecar'))


def retention_facts(connection, file_id, *, as_of=9223372036854775807):
    """Ledger-only facts shared by deletion and report --why."""
    # Bulk private releases keep one stage summary and the existing indexed
    # artifact presence facts, rather than another per-file retention history.
    released = connection.execute(f"SELECT event_id,details_json FROM {connection.event_table} WHERE file_id=? AND event_kind='artifact' AND reason_code='deleted' ORDER BY event_id DESC LIMIT 1", (file_id,)).fetchone()
    if released and released[0] <= as_of:
        detail = json.loads(released[1])
        if detail.get('private_release_intent'):
            intent = connection.execute(f'SELECT details_json FROM {connection.event_table} WHERE event_id=?', (detail['private_release_intent'],)).fetchone()
            summary = json.loads(intent[0])
            binding = connection.execute('SELECT sha256,bytes FROM files WHERE file_id=?', (file_id,)).fetchone()
            consumers = [row[0] for row in connection.execute(f"SELECT DISTINCT e.event_id FROM attempt_inputs i JOIN {connection.event_table} e USING(attempt_id) WHERE i.file_id=? AND i.sha256=? AND e.event_kind='lifecycle' AND e.status='ok' AND e.event_id<?", (file_id,binding[0],released[0]))]
            return dict(policy='private_scratch_after_last_planned_reader', consumer_ok_event_ids=consumers,
                outstanding_consumers=[], pins=[], declaration={'private_release_summary':True},
                deletions=[dict(event_id=released[0],reason_code='delete_completed',details={
                    'file_id':file_id,'sha256':binding[0],'bytes':binding[1],
                    'deleted_logical_bytes':binding[1], 'summary':summary,
                    'history':'private_stage_summary'})])
    if getattr(connection,'storage_version',0)>=3:
        compact=connection.execute('SELECT c.*,f.sha256,f.bytes FROM completed_deletions c JOIN files f USING(file_id) WHERE c.file_id=? AND c.event_id<=?',(file_id,as_of)).fetchone()
        if compact:
            fields=('file_id','event_id','run_id','reason_code','deletion_reason','last_consumer_stage','recorded_at','sha256','bytes')
            compact=dict(zip(fields,compact))
            consumers=[row[0] for row in connection.execute(f"SELECT DISTINCT e.event_id FROM attempt_inputs i JOIN {connection.event_table} e USING(attempt_id) WHERE i.file_id=? AND i.sha256=? AND e.event_kind='lifecycle' AND e.status='ok' AND e.event_id<?",(file_id,compact['sha256'],compact['event_id']))]
            return dict(policy=compact['deletion_reason'],consumer_ok_event_ids=consumers,
                outstanding_consumers=[],pins=[],declaration={'compact_completed_delete':True,'deletion_reason':compact['deletion_reason']},
                deletions=[dict(event_id=compact['event_id'],reason_code=compact['reason_code'],
                    details={**compact,'history':'compact_completed_delete','deleted_logical_bytes':compact['bytes']})])
    retention_table = 'retention_events' if connection.compact else 'events'
    retention_filter = '' if connection.compact else " AND event_kind='retention'"
    declarations = []
    deletions = []
    for event in connection.execute(f"SELECT event_id,reason_code,details_json FROM {retention_table} WHERE file_id=? AND event_id<=?{retention_filter} ORDER BY event_id", (file_id,as_of)):
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
                intent=connection.execute(f'SELECT details_json FROM {retention_table} WHERE event_id=?',(value['details']['intent_event_id'],)).fetchone()
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
            fact = connection.execute(f"SELECT e.event_id,e.attempt_id FROM {connection.event_table} e\n                JOIN attempt_inputs i ON i.attempt_id=e.attempt_id\n                WHERE e.task_id=? AND e.event_kind='lifecycle' AND e.status='ok'\n                AND i.file_id=? AND i.sha256=? AND e.event_id<=? ORDER BY e.event_id DESC LIMIT 1",
                (task, file_id, plan['sha256'],as_of)).fetchone()
            if fact:
                consumers.append(fact[0])
            elif task not in plan.get('waived_task_ids', []):
                outstanding.append(task)
    row=connection.execute('SELECT ownership,retention_class,kind,uri FROM files WHERE file_id=?',(file_id,)).fetchone()
    if plan:pins=list(plan.get('pins',[]))
    elif row and row[0]=='managed' and row[2] in KEPT_PUBLISHED_KINDS and '/.glossapi/' not in row[3]:
        # A compatibility copy the contract declares kept in every setting (W18 S2): declared by kind, no per-file record.
        pins=['published_compatibility_copy']
    else:pins=['retention_not_declared']
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
            state=self.writer.task_state(task)
            if (state['status'],state['reason_code'])!=('skipped','not_in_plan'):raise ValueError('Only explicitly unselected edges can be waived')
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
        elif path_alias(self.root, file_path(row)):
            # A link (file or any parent directory) could point at a file this
            # run does not own; never follow it for deletion.
            reasons.append('path_alias')
        if not plan:
            return row, facts, reasons
        superseded=plan.get('superseded_by')
        if superseded and self.writer.control()['status'].get(superseded)!='ok':
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
            elif registering_run(self.connection, file_id) not in run_family(self.connection, self.run_id):
                # Extreme releases only downloads this run owns, never a file
                # another run acquired, even with an identical version/hash.
                reasons.append('foreign_run_source')
        elif row['retention_class'] == 'initial':
            reasons.append('initial_source')
        elif self.settings['debug'] == 'on' and not superseded:
            reasons.append('debug_on')
        if facts['outstanding_consumers']:
            reasons.append('outstanding_consumers')
        if not facts['consumer_ok_event_ids']:
            reasons.append('no_successful_consumer')
        readers=self.connection.execute('SELECT attempt_id FROM attempt_inputs WHERE file_id=?',(file_id,))
        if any(self.writer.task_state(self.writer.control()['attempts'][reader[0]]['task_id'])['status']=='running' for reader in readers):
            reasons.append('active_reader')
        # Partial extract success is a historical fact, never release authority.
        for task in plan['consumer_task_ids']:
            for details, in self.connection.execute(f"SELECT details_json FROM {self.connection.event_table} WHERE task_id=? AND event_kind='decision'", (task,)):
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
                # A file glossAPI wrote or verified in this run and has not reopened for writing keeps its recorded
                # hash (K4C-G08: the seal/delete double hash); anything else is read and hashed here.
                from ..ledger.encoding import reused_own_hash
                value: str
                if reused_own_hash(before) == row['sha256']:
                    value = row['sha256']
                else:
                    import hashlib
                    digest = hashlib.sha256()
                    with os.fdopen(os.dup(handle), 'rb') as stream:
                        for chunk in iter(lambda: stream.read(1024*1024), b''):
                            digest.update(chunk)
                    value = digest.hexdigest()
                after = os.fstat(handle)
                current = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
                identity = lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
                if identity(before) != identity(after) or identity(after) != identity(current) or value != row['sha256']:
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
        # Every retention fact, pin, lease and task outcome is an event of this writer: an unchanged event counter
        # means an unchanged eligibility (used to skip the identical recheck below when nothing was written).
        # (compact ledgers only: version 1 assigns event ids in SQLite and keeps no counter).
        checked_at = self.writer.next_event_id if self.connection.compact else None
        # Several logical registrations may share the same physical pathname.
        aliases = self.connection.execute("SELECT file_id FROM current_file_state WHERE uri=? AND presence='present'", (row['uri'],)).fetchall()
        # This file's own eligibility was computed just above (no state change in between).
        if any(self.eligibility(alias[0])[2] for alias in aliases if alias[0] != file_id):
            return False
        path=file_path(row)
        if path.exists() and not path.is_symlink() and path.stat().st_nlink>1:
            identity=path.stat()
            for shared in self.writer.file_catalog().values():
                if shared['presence']!='present' or shared['uri']==row['uri']:continue
                candidate=file_path(shared)
                try:other=candidate.lstat()
                except OSError:continue
                if (other.st_dev,other.st_ino)==(identity.st_dev,identity.st_ino) and self.eligibility(shared['file_id'])[2]:return False
        intent = recovering
        details: dict[str, Any] = dict(file_id=file_id, sha256=row['sha256'], bytes=row['bytes'],
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
                self.writer.durable_barrier([file_id,*(alias[0] for alias in aliases)])
                self.fault('after_intent', file_id)
            # Pins and leases can only change through this writer. Recheck the
            # path identity under the same opened directory immediately before unlink
            # (the eligibility again only if this writer recorded anything since it was computed).
            if (checked_at is None or self.writer.next_event_id != checked_at) and self.eligibility(file_id)[2]:
                return False
            current = os.stat(name,dir_fd=descriptor,follow_symlinks=False)
            if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns) != identity:
                raise ValueError('Artifact replaced before unlink')
            free_before = os.statvfs(self.root).f_bavail * os.statvfs(self.root).f_frsize
            os.unlink(name,dir_fd=descriptor)
            if hasattr(self,'_directory_syncs'):self._directory_syncs.add(descriptor)
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
                    consumer=self.connection.execute(f"SELECT t.stage FROM {self.connection.event_table} e JOIN tasks t USING(task_id) JOIN attempt_inputs i USING(attempt_id) WHERE i.file_id=? AND e.event_kind='lifecycle' AND e.status='ok' ORDER BY e.event_id DESC LIMIT 1",(alias,)).fetchone()
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
        from ..ledger.writer import SyncBatch
        intents=[]
        # Every intent of this frontier commits, then one durability barrier
        # precedes any unlink. Completions are bounded transactions; a death
        # anywhere leaves recoverable intents.
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
        if not intents:
            if file_ids:self._prune_completed_policies()
            return
        self.writer.durable_barrier([file_id for file_id,_ in intents])
        for offset in range(0,len(intents),128):
            self._directory_syncs=SyncBatch()
            try:
                with self.writer.batch():
                    try:
                        for file_id,intent in intents[offset:offset+128]:
                            self.fault('after_intent',file_id)
                            self.delete(file_id,recovering=intent)
                    finally:
                        # Every affected parent reaches stable storage, once per
                        # batch, before this transaction commits its completions.
                        self._directory_syncs.flush()
            finally:
                self._directory_syncs.close()
                del self._directory_syncs
            self._prune_completed_policies()

    def recover(self):
        recover_private_releases(self)
        self._prune_completed_policies()
        table = 'retention_events' if self.connection.compact else 'events'
        retention_filter = '' if self.connection.compact else " AND e.event_kind='retention'"
        intents = list(self.connection.execute(f"""SELECT event_id,file_id FROM {table} e
            WHERE reason_code='delete_planned' AND file_id IS NOT NULL {retention_filter} AND NOT EXISTS
            (SELECT 1 FROM {table} c WHERE c.file_id=e.file_id AND c.event_id>e.event_id
             AND c.reason_code IN ('delete_completed','delete_recovered','delete_failed'))"""))
        # The dead writer may have committed its NORMAL intent and died while
        # flushing consumers, before its FULL barrier. Never infer durability
        # merely from a visible intent after process death.
        if intents:self.writer.durable_barrier([row[1] for row in intents])
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
            # A managed file: the verified hash is reused by its retention delete in this run (K4C-G08).
            if path.is_symlink() or not path.is_file() or sha256_file(path,own=True)!=row['sha256']:
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
    # One pass over the owner's file catalog (current presence) instead of one view query per attempt;
    # rows are grouped in the given attempt order, each attempt's files in registration order.
    by_producer=defaultdict(list)
    for entry in writer.file_catalog().values():
        if entry['producer_attempt_id'] and entry['ownership']=='managed' and entry['presence']=='present':
            by_producer[entry['producer_attempt_id']].append(entry)
    rows=[]
    for attempt in attempts:
        rows.extend(dict(entry) for entry in by_producer.get(attempt,()))
    private=Path(root).resolve()/'.glossapi/artifacts'
    projections={}
    for encoded, in writer.connection.execute("SELECT json FROM snapshots WHERE kind='product_projection' ORDER BY rowid"):
        value: dict[str, Any]=json.loads(encoded);projections[value['product_uri']]=value
    current_backings={value['immutable_uri'] for value in projections.values()}
    candidates=[]
    partial_coverage={}
    # The OCR publication re-cleans every raw markdown in markdown/ (clean_ocr over the directory), so public raw
    # markdown is superseded only once that barrier has completed (it was kept until then only by an accidental
    # duplicate registration, which the K4C-H1 fix removed).
    requested=controller.settings.get('requested_stages',())
    raw_still_read='ocr' in requested and any(task['operation']=='ocr.compatibility_publish' and task['status']!='ok'
        for task in writer.task_states(run_id))
    for row in rows:
        path=file_path(row)
        superseded_public_raw=(row['kind']=='raw_markdown' and path.is_relative_to(Path(root).resolve()/'markdown') and 'clean' in requested
            and not raw_still_read)
        public_temporary=row['kind'] in ('annotation_partition','repaired_pdf','temporary_cleaning_report')
        if not path.is_relative_to(private) and path not in cleanup_paths and not superseded_public_raw and not public_temporary:continue
        if row['retention_class'] in ('final','evidence','recovery') or (row['uri'] in current_backings and row['kind']!='annotation_partition'):continue
        if controller.settings['debug']=='on' and not (row['kind']=='download_payload' and controller.settings['memory_saving']=='extreme'):continue
        if row['kind']=='download_payload' and not writer.connection.execute('SELECT 1 FROM attempt_inputs WHERE file_id=?',(row['file_id'],)).fetchone():continue
        # Worker receipts are recovery frontiers and must remain replayable.
        if row['kind'] in ('worker_receipt','generation_manifest','ocr_native','ocr_page_state'):continue
        producer=writer.control()['attempts'].get(row['producer_attempt_id'])
        if not producer or writer.task_state(producer['task_id'])['status']!='ok':continue
        if row['stage']=='extract':
            # One verdict per producer attempt (its decisions are immutable), not one per output file.
            partial=partial_coverage.get(row['producer_attempt_id'])
            if partial is None:
                from ..ledger.encoding import unpack_evidence
                partial=False
                for detail, in writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE attempt_id=? AND event_kind='decision'",(row['producer_attempt_id'],)):
                    digest=json.loads(detail).get('evidence_sha256')
                    stored=writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()
                    if stored and unpack_evidence(json.loads(stored[0])).get('coverage_allows_source_release') is False:partial=True
                partial_coverage[row['producer_attempt_id']]=partial
            if partial:continue
        candidates.append(row)
    private_candidates = [row for row in candidates if file_path(row).is_relative_to(private)
        and row['retention_class']=='intermediate' and row['kind']!='download_payload'
        and not (controller.settings['memory_saving']=='extreme' and row['kind']=='source_snapshot')]
    # Freeze future publication readers: only unconsumed snapshots/diagnostics
    # can retire during the DAG. Other outputs reach this function after their
    # compatibility publication frontier is complete.
    if private_candidates:
        release_private_scratch(controller, private_candidates)
        released = {row['file_id'] for row in private_candidates}
        candidates = [row for row in candidates if row['file_id'] not in released]
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
    owner=getattr(connection,'owner_writer',None)
    deleted_paths={}
    if owner is not None:
        for row in owner.file_catalog().values():
            group=groups.setdefault(row['kind'],dict(present=0,deleted=0,missing=0,corrupt=0,pinned=0))
            size=row['bytes'] or 0
            group[row['presence']]=group.get(row['presence'],0)+size
            if row['presence']=='present' and (row['ownership']!='managed' or row['retention_class'] in ('initial','final','evidence','recovery')):
                group['pinned']+=size
            if row['presence']=='deleted':
                deleted_paths[row['uri']]=max(deleted_paths.get(row['uri'],0),size)
        deleted=sum(deleted_paths.values())
    else:
        # Read-only reports reconstruct from the ledger; running control uses
        # the owner's append-driven catalog instead of repeated SQL censuses.
        for kind,presence,size in connection.execute("SELECT kind,presence,COALESCE(SUM(bytes),0) FROM current_file_state GROUP BY kind,presence"):
            group=groups.setdefault(kind,dict(present=0,deleted=0,missing=0,corrupt=0,pinned=0))
            group[presence]=size
        for kind,size in connection.execute("SELECT kind,COALESCE(SUM(bytes),0) FROM current_file_state WHERE presence='present' AND (ownership<>'managed' OR retention_class IN ('initial','final','evidence','recovery')) GROUP BY kind"):
            groups[kind]['pinned']=size
        deleted=connection.execute("SELECT COALESCE(SUM(bytes),0) FROM (SELECT uri,MAX(bytes) bytes FROM current_file_state WHERE presence='deleted' GROUP BY uri)").fetchone()[0]
    settings={};resources=dict(present_bytes=None,reserved_bytes=None,free_bytes=None)
    if run:
        stored=connection.execute('SELECT json FROM snapshots WHERE sha256=?',(run['settings_sha256'],)).fetchone()
        if stored:settings=json.loads(stored[0])
        latest=connection.execute(f"SELECT details_json FROM {connection.event_table} WHERE run_id=? AND event_kind='run' ORDER BY event_id DESC LIMIT 1",(run['run_id'],)).fetchone()
        reference=json.loads(latest[0]).get('result_sha256') if latest else None
        recorded=connection.execute('SELECT json FROM snapshots WHERE sha256=?',(reference,)).fetchone()
        if recorded:resources=json.loads(recorded[0])['result'].get('disk',resources)
    return dict(settings={key:settings.get(key,default) for key,default in [('debug','off'),('memory_saving','normal')]},
        source_exception=MEMORY_SAVING_HELP,by_kind_bytes=groups,deleted_logical_bytes=deleted,resources=resources,
        disk_budget_bytes=settings.get('disk_budget_bytes'),free_space_basis='last recorded statvfs resource observation; report does not scan payloads')


def _streaming_retirement(writer,run_id)->'Any':
    """Per-run reader map for streaming retirement, built once per writer (the frozen plan does not change)."""
    states=writer.__dict__.setdefault('_streaming_retirement',{})
    state=states.get(run_id)
    if state is None:
        run=writer.connection.execute('SELECT plan_sha256 FROM runs WHERE run_id=?',(run_id,)).fetchone()
        operations=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',run).fetchone()[0]).get('operations',[])
        ordinal_task={row['ordinal']:row['task_id'] for row in writer.task_states(run_id)}
        downloads={value['document']['version_id'] for value in operations if value['stage']=='download'}
        readers={}
        for ordinal,value in enumerate(operations):
            if value['document']['version_id'] not in downloads:continue
            roles=value['specification']['input_roles']+[edge[2] for edge in value['inputs']]
            if set(roles)&{'source_payload','pdf','source_pdf'}:
                readers.setdefault(value['document']['version_id'],[]).append(ordinal_task[ordinal])
        extract=[ordinal_task[ordinal] for ordinal,value in enumerate(operations) if value['specification'].get('operation')=='extract.document']
        state=states[run_id]=dict(versions=downloads,readers=readers,settled=set(),extract=extract,repaired_done=set())
    return state


def release_repaired_pdfs(writer,root,run_id,settings):
    """memory_saving=extreme with debug=off: release each repaired PDF once its document's extraction succeeded.

    A repaired PDF is a whole-PDF copy that no planned document stage reads; its
    only use was the ``repaired_pdfs/`` compatibility mirror, which debug=off
    releases again at the clean frontier (it is not a final product). Keeping it
    until the corpus-wide publication would hold a second copy of every
    repaired source, so a corpus larger than the budget could never stream (W18
    M9). The publication skips a released one; products are unchanged.
    """
    if settings.get('memory_saving','normal')!='extreme' or settings.get('debug','off')=='on':return
    state=_streaming_retirement(writer,run_id)
    controller=RetentionController(writer,root,run_id,settings)
    private=(controller.root/'.glossapi/artifacts').as_uri()+'/'
    rows=[]
    for task in state['extract']:
        if task in state['repaired_done'] or writer.task_state(task)['status']!='ok':continue
        state['repaired_done'].add(task)
        attempt=writer.control()['latest'].get(task)
        rows+=[row for row in writer.connection.execute("SELECT * FROM current_file_state WHERE producer_attempt_id=? AND kind='repaired_pdf' AND presence='present'",(attempt,)).fetchall()
            if row['ownership']=='managed' and row['retention_class']=='intermediate' and row['uri'].startswith(private)]
    if rows:release_private_scratch(controller,rows)


def retire_downloaded_sources(writer: Any, root: Any, run_id: str, settings: dict[str, Any]) -> None:
    """Release original download bytes after the frozen plan's last PDF reader.

    memory_saving=extreme streams (owner design, W18 M9): a run-owned download
    is deleted as soon as every planned document stage that reads it succeeded
    and its outputs are durable, while the run goes on. The compatibility
    publications later bind it by its recorded hash and never read it again.
    Evaluation and report commands never add reader edges. Called after every
    completion batch, it visits only versions it has not settled yet.
    """
    if settings.get('memory_saving','normal')!='extreme':return
    release_repaired_pdfs(writer,root,run_id,settings)
    state=_streaming_retirement(writer,run_id)
    controller=None
    # The original payload and byte-identical staging snapshots are one source
    # footprint; extreme must not leave a hidden whole-PDF copy behind.
    for version in state['versions']-state['settled']:
        # Not before this document's own source readers are terminal (cheap; the download itself precedes them).
        if any(writer.task_state(task_id)['status'] not in ('ok','skipped','failed') for task_id in state['readers'].get(version,())):continue
        controller=controller or RetentionController(writer,root,run_id,settings)
        digest=writer.connection.execute('SELECT stored_sha256 FROM document_versions WHERE version_id=?',(version,)).fetchone()[0]
        family=run_family(writer.connection,run_id)
        rows=[row for row in writer.connection.execute("SELECT * FROM current_file_state WHERE version_id=? AND sha256=? AND ownership='managed' AND presence='present' AND kind IN ('download_payload','source_snapshot','scratch_copy')",(version,digest)).fetchall()
            if registering_run(writer.connection,row['file_id']) in family and not path_alias(controller.root,file_path(row))]
        if not rows:
            state['settled'].add(version);continue
        # Every frozen reader of these bytes (CP-K4a F2): the producer/output-kind edges and explicit file bindings of
        # the plan (``planned_readers``), whatever role the edge names and whatever document or aggregate the reading
        # task belongs to, plus this document's source readers by role. Deletion waits for all of them.
        candidates: set[str] = set(state['readers'].get(version,()))
        for row in rows:
            planned: set[str] = writer.planned_readers(run_id,row)
            candidates|=planned
        readers=[];waived=[];ready=True
        for task_id in sorted(candidates):
            task=writer.task_state(task_id)
            if task['status']=='skipped' and task['reason_code']=='not_in_plan':waived.append(task_id)
            elif task['status']!='ok':ready=False
            else:readers.append(task_id)
        if not ready:continue
        if not readers:
            state['settled'].add(version);continue
        # Verify source-hash bindings and full coverage on the real readers.
        valid=True
        for task in readers:
            if not writer.connection.execute(f"SELECT 1 FROM attempt_inputs i JOIN attempts a USING(attempt_id) JOIN {writer.connection.event_table} e USING(attempt_id) WHERE a.task_id=? AND i.sha256=? AND e.event_kind='lifecycle' AND e.status='ok'",(task,digest)).fetchone():valid=False
            for detail, in writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE task_id=? AND event_kind='decision'",(task,)):
                reference=json.loads(detail).get('evidence_sha256')
                stored=writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(reference,)).fetchone()
                if stored:
                    from ..ledger.encoding import unpack_evidence
                    value=unpack_evidence(json.loads(stored[0]))
                    if 'coverage_allows_source_release' in value and value['coverage_allows_source_release'] is not True:valid=False
        if not valid:
            state['settled'].add(version);continue
        new_rows=[row for row in rows if not retention_facts(writer.connection,row['file_id'])['declaration']]
        sealer=controller.seal(new_rows)
        if sealer:
            with writer.transaction():
                for reader in readers:writer.dependency(sealer,reader,'source_release_gate')
        for row in new_rows:
            exact=[task for task in readers if writer.connection.execute("SELECT 1 FROM attempt_inputs i JOIN attempts a USING(attempt_id) WHERE a.task_id=? AND i.file_id=? AND i.sha256=?",(task,row['file_id'],digest)).fetchone()]
            controller.declare(row['file_id'],[*exact,sealer],source=True)
        controller.delete_many([row['file_id'] for row in rows])
        state['settled'].add(version)


def _hard_linked_copies(controller):
    """(producer attempt, sha256) -> present managed copies outside the private tree (decision 3 hard links)."""
    private=(controller.root/'.glossapi/artifacts').as_uri()+'/'
    copies={}
    for row in controller.writer.file_catalog().values():
        if row['ownership']=='managed' and row['presence']=='present' and row['producer_attempt_id'] and not row['uri'].startswith(private):
            copies.setdefault((row['producer_attempt_id'],row['sha256']),[]).append(row)
    return copies


def _open_private_parent(controller, parts, parents):
    """The parent directory of a private file, opened component by component without following links.

    ``parents`` keeps the descriptor of the most recent parent: consecutive
    files of one attempt directory share one chain of no-follow opens.
    """
    if parents is not None and parents.get('parts')==parts:
        return parents['descriptor']
    descriptor=os.open(controller.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for component in parts:
            child=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
            os.close(descriptor);descriptor=child
    except BaseException:
        os.close(descriptor);raise
    if parents is not None:
        _close_private_parent(parents)
        parents.update(parts=parts,descriptor=descriptor)
    return descriptor


def _close_private_parent(parents):
    if parents and parents.get('descriptor') is not None:
        os.close(parents['descriptor'])
    if parents is not None:parents.update(parts=None,descriptor=None)


def _unlink_private_file(controller, row, intent, copies, parents=None):
    """Unlink only a producer-owned, unexposed regular private file; no rehash.

    A private file may share its inode only with the registered public copies
    of the same producer attempt and bytes (the compatibility product published
    by hard link, decision 3); any other extra link is ambiguous.
    """
    path = file_path(row)
    private = controller.root/'.glossapi/artifacts'
    if (row['ownership']!='managed' or not row['producer_attempt_id']
            or row['retention_class']!='intermediate' or not path.is_relative_to(private)
            or row['kind']=='download_payload'):
        raise ValueError('Bulk release requires unexposed producer-owned scratch')
    relative=path.relative_to(controller.root)
    descriptor=None
    try:
        try:
            descriptor=_open_private_parent(controller,relative.parts[:-1],parents)
            info=os.stat(relative.name,dir_fd=descriptor,follow_symlinks=False)
            links=1
            if stat.S_ISREG(info.st_mode) and info.st_nlink>1:
                # One count per distinct path: a resumed publication can register the same copy again.
                for uri in sorted({copy['uri'] for copy in copies.get((row['producer_attempt_id'],row['sha256']),())}):
                    try:other=file_path({'uri':uri}).lstat()
                    except OSError:continue
                    if (other.st_dev,other.st_ino)==(info.st_dev,info.st_ino):links+=1
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=links:
                raise ValueError('Private scratch identity is ambiguous')
            os.unlink(relative.name,dir_fd=descriptor)
            controller.fault('after_unlink',row['file_id'])
        except FileNotFoundError:
            pass
        controller.writer.event(controller.run_id,event_kind='artifact',reason_code='deleted',
            file_id=row['file_id'],version_id=row['version_id'],
            details={'private_release_intent':intent})
    finally:
        if parents is None and descriptor is not None:os.close(descriptor)


def _finish_private_release(controller, intent, detail):
    connection=controller.connection
    manifest=json.loads(connection.execute('SELECT json FROM snapshots WHERE sha256=?',
        (detail['manifest_sha256'],)).fetchone()[0])
    # Private scratch released within the run is never made durable, neither
    # its bytes nor its unlink (K4(c) decision 2): after a crash the release
    # intent is recovered (a missing file is an observed unlink) and the stage
    # reruns from kept inputs. The intent itself committed with FULL first.
    copies=_hard_linked_copies(controller)
    catalog=controller.writer.file_catalog()
    parents={}
    try:
        for offset in range(0,len(manifest['file_ids']),128):
            with controller.writer.batch():
                for file_id in manifest['file_ids'][offset:offset+128]:
                    # The owner's append-driven catalog holds current presence (control state, not byte proof).
                    row=catalog.get(file_id) or connection.execute('SELECT * FROM current_file_state WHERE file_id=?',(file_id,)).fetchone()
                    if row['presence']=='deleted':continue
                    _unlink_private_file(controller,row,intent,copies,parents)
    finally:
        _close_private_parent(parents)
    with controller.writer.transaction():
        controller.writer.event(controller.run_id,event_kind='retention',reason_code='delete_completed',
            message='private_stage_release',details={**detail,'intent_event_id':intent})


def release_private_scratch(controller, rows):
    """Group release after the producer/last planned reader; retain all hashes."""
    if controller.settings['debug']=='on':return
    by_stage=defaultdict(list)
    for row in rows:
        producer=controller.writer.control()['status'].get(row['producer_attempt_id'])=='ok'
        readers=controller.connection.execute('SELECT attempt_id FROM attempt_inputs WHERE file_id=?',(row['file_id'],))
        unfinished=any(controller.writer.task_state(controller.writer.control()['attempts'][reader[0]]['task_id'])['status'] not in (None,'ok') for reader in readers)
        ready=True
        for reader in controller.writer.planned_readers(controller.run_id,row):
            status,reason=controller.writer.reader_state(reader)
            if status=='skipped' and reason=='not_in_plan':continue
            if status!='ok':ready=False;break
            exact=controller.connection.execute(f"SELECT 1 FROM attempt_inputs i JOIN attempts a USING(attempt_id) JOIN {controller.connection.event_table} e USING(attempt_id) WHERE a.task_id=? AND i.file_id=? AND i.sha256=? AND e.event_kind='lifecycle' AND e.status='ok' LIMIT 1",(reader,row['file_id'],row['sha256'])).fetchone()
            if not exact:ready=False;break
        if producer and not unfinished and ready:
            by_stage[row['stage']].append(row)
    # Every stage summary of this frontier commits in one transaction; one
    # durability barrier then covers all of them before any unlink.
    releases=[]
    if by_stage:
        with controller.writer.transaction():
            for stage,files in by_stage.items():
                manifest=controller.writer.snapshot('private_release_manifest',{'file_ids':[row['file_id'] for row in files],
                    'planned_reader_task_ids':sorted({reader for row in files for reader in controller.writer.planned_readers(controller.run_id,row)})})
                detail={'stage':stage,'file_count':len(files),'bytes':sum(row['bytes'] for row in files),
                    'manifest_sha256':manifest,'policy':'private_scratch_after_last_planned_reader',
                    'producer_attempt_ids':sorted({row['producer_attempt_id'] for row in files})}
                intent=controller.writer.event(controller.run_id,event_kind='retention',reason_code='delete_planned',
                    message='private_stage_release',details=detail)
                releases.append((intent,detail,files))
        controller.writer.durable_barrier([row['file_id'] for _,_,files in releases for row in files],private=True)
    for intent,detail,files in releases:
        controller.fault('after_intent', files[0]['file_id'])
        _finish_private_release(controller,intent,detail)


def recover_private_releases(controller):
    connection=controller.connection
    rows=connection.execute(f"SELECT event_id,details_json FROM {connection.event_table} e WHERE run_id=? AND event_kind='retention' AND reason_code='delete_planned' AND message='private_stage_release'", (controller.run_id,)).fetchall()
    # details_json can be compressed physically, so match completion links using
    # the decoded public row values rather than SQLite JSON extraction.
    pending=[]
    for intent,encoded in rows:
        completed=False
        for value, in connection.execute(f"SELECT details_json FROM {connection.event_table} WHERE run_id=? AND reason_code='delete_completed' AND message='private_stage_release' AND event_id>?",(controller.run_id,intent)):
            if json.loads(value).get('intent_event_id')==intent:completed=True;break
        if not completed:pending.append((intent,json.loads(encoded)))
    # A dead writer may have committed its NORMAL intent without its FULL
    # barrier; only an unfinished release needs that barrier before unlink.
    if pending:
        released=[]
        for intent,detail in pending:
            manifest=json.loads(connection.execute('SELECT json FROM snapshots WHERE sha256=?',(detail['manifest_sha256'],)).fetchone()[0])
            released.extend(manifest['file_ids'])
        controller.writer.durable_barrier(released,private=True)
    for intent,detail in pending:_finish_private_release(controller,intent,detail)


def released_source(connection, uri):
    """The ledger row of a source file at ``uri`` that a completed policy deletion removed, or None.

    memory_saving=extreme deletes a run-owned download after its last planned
    document reader (streaming, W18 M9). Later compatibility projections bind
    the committed bytes by this row; they never read, recreate or re-consume them.
    """
    row=connection.execute("SELECT * FROM current_file_state WHERE uri=? AND presence='deleted' AND kind IN ('download_payload','source_snapshot') "
        "ORDER BY event_id DESC LIMIT 1",(uri,)).fetchone()
    if row is None or Path(file_path(row)).exists():
        return None
    deletions = cast(list[dict[str, Any]], retention_facts(connection,row['file_id'])['deletions'])
    if not any(item['reason_code'] in ('delete_completed','delete_recovered') for item in deletions):
        return None
    return row


def _download_footprint(writer,row):
    """A staging snapshot of a run-owned download's bytes: extreme retires it with the download (one source footprint)."""
    return writer.connection.execute("SELECT 1 FROM current_file_state WHERE version_id=? AND sha256=? AND kind='download_payload' LIMIT 1",
        (row['version_id'],row['sha256'])).fetchone() is not None


def retire_completed_scratch(writer, root, run, settings, attempts, *, protected=()):
    """During the DAG, only outputs with no planned publication reader retire."""
    if settings.get('debug','off')=='on':return
    rows=[]
    for attempt in attempts:
        for row in writer.connection.execute("SELECT * FROM current_file_state WHERE producer_attempt_id=? AND presence='present' AND retention_class='intermediate'",(attempt,)):
            if (attempt,row['kind']) in protected:continue
            if row['kind'] not in ('source_snapshot','attempt_diagnostic'):continue
            if settings.get('memory_saving')=='extreme' and row['kind']=='source_snapshot' and _download_footprint(writer,row):continue
            if not file_path(row).is_relative_to(Path(root).resolve()/'.glossapi/artifacts'):continue
            rows.append(row)
    release_private_scratch(RetentionController(writer,root,run,settings),rows)
