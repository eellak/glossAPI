"""Conservative disk admission and verification of an enforced scratch quota.

A user-provisioned, size-limited filesystem is the native write guard. Merely
polling directory sizes, RLIMIT_FSIZE (one file), or a settings assertion cannot
qualify a third-party backend's total scratch writes.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

from .retention import resolved_settings


class DiskBudgetError(ValueError):
    def __init__(self, code, message):
        self.code=code
        super().__init__(code+': '+message)


def filesystem_capacity(path):
    state=os.statvfs(path)
    return state.f_blocks*state.f_frsize


def require_disk_guard(root, settings):
    """Read-only validation; never mount or silently provision a filesystem."""
    settings=resolved_settings(settings)
    if settings['memory_saving']!='extreme':return
    root=Path(root).resolve()
    staging=root/'.glossapi/staging'
    if not sys.platform.startswith('linux') or not root.is_mount() or not staging.is_mount():
        raise DiskBudgetError('disk_guard_unavailable','extreme needs dedicated size-limited corpus and staging filesystems')
    if filesystem_capacity(root)+filesystem_capacity(staging)>settings['disk_budget_bytes'] or filesystem_capacity(staging)>settings['max_attempt_bytes']:
        raise DiskBudgetError('disk_guard_unavailable','filesystem capacity exceeds the declared corpus or attempt cap')
    # Separate devices prevent the backend consuming the durable evidence reserve.
    if root.stat().st_dev==staging.stat().st_dev:
        raise DiskBudgetError('disk_guard_unavailable','staging must have its own enforced capacity')


HIGH_WATER_BASIS=('maximum managed present+reserved bytes over the observations of this run: every admission '
    'check, the scheduler start/end, resume and the final retention boundary, persisted in run events; '
    'not a continuous time series of all filesystem writes')


def recorded_high_water(connection, run_id):
    """Largest disk high-water this run has persisted in its run events."""
    table=getattr(connection,'event_table','events')
    peak=0
    import json
    for encoded, in connection.execute(f"SELECT details_json FROM {table} WHERE run_id=? AND event_kind='run'",(run_id,)):
        try:value=json.loads(encoded).get('disk_high_water_bytes')
        except (TypeError,ValueError,AttributeError):value=None
        if isinstance(value,int):peak=max(peak,value)
    return peak


class DiskBudget:
    def __init__(self, writer, root, run_id, settings):
        self.writer,self.root,self.run_id=writer,Path(root).resolve(),run_id
        self.settings=resolved_settings(settings)
        self.limit=self.settings.get('disk_budget_bytes')
        self.reservations={}
        # Persisted peaks of earlier boundaries of this run (scheduler,
        # resume, final retention) seed this one; a fresh budget never
        # reports a smaller peak than the run already observed (LS7-G04).
        self.high_water=recorded_high_water(writer.connection,run_id) if run_id else 0
        self.observations=0

    def present_bytes(self):
        # Count pathname copies conservatively, while duplicate observations of
        # one pathname do not reserve its bytes several times.
        by_path={}
        for row in self.writer.file_catalog().values():
            if row['presence']=='present' and row['ownership']=='managed':
                by_path[row['uri']]=max(by_path.get(row['uri'],0),row['bytes'] or 0)
        return sum(by_path.values())

    def observe(self, present=None):
        """Record one managed present+reserved observation; return present bytes."""
        present=self.present_bytes() if present is None else present
        self.high_water=max(self.high_water,present+sum(self.reservations.values()))
        self.observations+=1
        return present

    def can_reserve(self, amount):
        present=self.observe()
        reserved=sum(self.reservations.values())
        reserve=max(1024**3,int(.05*(self.limit or 0)))+64*1024**2
        free=os.statvfs(self.root).f_bavail*os.statvfs(self.root).f_frsize
        scratch_available=True
        if self.settings['memory_saving']=='extreme':
            scratch=os.statvfs(self.root/'.glossapi/staging')
            scratch_available=scratch.f_bavail*scratch.f_frsize>=self.settings['max_attempt_bytes']
        return (self.limit is None or present+reserved+amount<=self.limit) and free-reserved-amount>=reserve and scratch_available

    def reserve(self, key, amount):
        if not self.can_reserve(amount):
            raise DiskBudgetError('waiting_for_disk','managed payloads, reservations or emergency free-space reserve prevent admission')
        self.reservations[key]=amount

    def release(self,key):
        self.reservations.pop(key,None)

    def report(self):
        state=os.statvfs(self.root)
        present=self.observe()
        return dict(disk_budget_bytes=self.limit,present_bytes=present,
            reserved_bytes=sum(self.reservations.values()),high_water_bytes=self.high_water,
            high_water_basis=HIGH_WATER_BASIS,
            free_bytes=state.f_bavail*state.f_frsize,free_space_basis='statvfs available bytes; logical copies counted by pathname',
            emergency_reserve_bytes=max(1024**3,int(.05*(self.limit or 0)))+64*1024**2)
