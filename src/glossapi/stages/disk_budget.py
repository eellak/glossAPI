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


class DiskBudget:
    def __init__(self, writer, root, run_id, settings):
        self.writer,self.root,self.run_id=writer,Path(root).resolve(),run_id
        self.settings=resolved_settings(settings)
        self.limit=self.settings.get('disk_budget_bytes')
        self.reservations={}
        self.high_water=0

    def present_bytes(self):
        # Count pathname copies conservatively, while duplicate observations of
        # one pathname do not reserve its bytes several times.
        return self.writer.connection.execute("""SELECT COALESCE(SUM(bytes),0) FROM
            (SELECT uri,MAX(bytes) bytes FROM current_file_state WHERE presence='present'
             AND ownership='managed' GROUP BY uri)""").fetchone()[0]

    def can_reserve(self, amount):
        present=self.present_bytes()
        reserved=sum(self.reservations.values())
        reserve=max(1024**3,int(.05*(self.limit or 0)))+64*1024**2
        free=os.statvfs(self.root).f_bavail*os.statvfs(self.root).f_frsize
        self.high_water=max(self.high_water,present+reserved)
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
        return dict(disk_budget_bytes=self.limit,present_bytes=self.present_bytes(),
            reserved_bytes=sum(self.reservations.values()),high_water_bytes=self.high_water,
            free_bytes=state.f_bavail*state.f_frsize,free_space_basis='statvfs available bytes; logical copies counted by pathname',
            emergency_reserve_bytes=max(1024**3,int(.05*(self.limit or 0)))+64*1024**2)
