"""A disk_full record allocated before it is needed (K4BI-G2, W18 M10).

A filesystem too full for one more ledger page cannot record why the run stopped. Every run allocates this small
file at its start (``.glossapi/interruption.json``, a fixed size), and an exhausted disk overwrites it in place: no
new block is needed. ``status``, ``report`` and the resume read it when the ledger holds no ``disk_full`` event.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SIZE = 4096
NAME = '.glossapi/interruption.json'


def _payload(value: dict[str, Any]) -> bytes:
    data = json.dumps(value, sort_keys=True).encode()[:SIZE-1]
    return data + b' '*(SIZE-len(data))


def allocate(root: Any) -> None:
    """Create the fixed-size record at run start (kept as is while it names an interruption the resume must read)."""
    path = Path(root)/NAME
    if path.is_file() and path.stat().st_size == SIZE:
        return
    clear(root)


def clear(root: Any) -> None:
    """Reset the record to ``clear`` in place (a completed run, or first allocation); best effort."""
    path = Path(root)/NAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            os.pwrite(descriptor, _payload({'state': 'clear'}), 0)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def record(root: Any, run_id: str, error: object) -> bool:
    """Overwrite the allocated record in place with the disk_full cause; never grows the file."""
    path = Path(root)/NAME
    try:
        descriptor = os.open(path, os.O_WRONLY)
    except OSError:
        return False
    try:
        from datetime import datetime, timezone
        os.pwrite(descriptor, _payload({'state': 'interrupted', 'cause': 'disk_full', 'errno': 28, 'run_id': run_id,
                                        'recorded_at': datetime.now(timezone.utc).isoformat(), 'error': str(error)[:300]}), 0)
        os.fsync(descriptor)
        return True
    except OSError:
        return False
    finally:
        os.close(descriptor)


def read(root: Any, run_id: str | None = None) -> dict[str, Any] | None:
    """The recorded, not yet consumed disk_full cause for ``run_id`` (any run when None), or None."""
    try:
        value: dict[str, Any] = json.loads((Path(root)/NAME).read_bytes().decode().strip() or '{}')
    except (OSError, ValueError):
        return None
    if value.get('state') != 'interrupted' or value.get('cause') != 'disk_full' or (run_id is not None and value.get('run_id') != run_id):
        return None
    return value


def consume(root: Any, run_id: str) -> bool:
    """After a recovery has attributed the recorded cause, mark it consumed in place (CP-K4a F3): it explains the
    attempts that were running when it was written and never a later death of their replacements."""
    value = read(root, run_id)
    path = Path(root)/NAME
    if value is None:
        return False
    try:
        from datetime import datetime, timezone
        descriptor = os.open(path, os.O_WRONLY)
        try:
            os.pwrite(descriptor, _payload({**value, 'state': 'consumed', 'consumed_at': datetime.now(timezone.utc).isoformat()}), 0)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return True
    except OSError:
        return False


def _instant(value: object) -> Any:
    from datetime import datetime
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def verified_cause(connection: Any, root: Any, run_id: str, *, attempt_id: str | None = None, table: str = 'events') -> dict[str, Any] | None:
    """The disk_full explanation of an interruption, tied to an attempt epoch (CP-K4a F3), or None.

    The epoch is the interrupted attempt's start (``attempt_id``) or, for a whole run (status, report --why), the start
    of its latest attempt. A ledger ``disk_full`` event explains it only when recorded after that start; the record
    allocated at run start (when the ledger could not take the event) only when written after it and not yet
    consumed by an earlier recovery. An older cause never explains a later death.
    """
    if attempt_id is not None:
        epoch: Any = connection.execute(f"SELECT MIN(event_id) FROM {table} WHERE attempt_id=?", (attempt_id,)).fetchone()[0]
        row: Any = connection.execute("SELECT created_at FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        started: Any = row[0] if row else None
    else:
        epoch = connection.execute(f"SELECT MAX(e.event_id) FROM {table} e JOIN tasks t ON t.task_id=e.task_id WHERE t.run_id=? "
                                   "AND e.event_kind='lifecycle' AND e.reason_code='stage_started'", (run_id,)).fetchone()[0]
        started = connection.execute("SELECT MAX(a.created_at) FROM attempts a JOIN tasks t ON t.task_id=a.task_id WHERE t.run_id=?",
                                     (run_id,)).fetchone()[0]
    marker: Any = connection.execute(f"SELECT event_id,message FROM {table} WHERE run_id=? AND event_kind='run' AND reason_code='disk_full' "
                                "AND event_id>? ORDER BY event_id DESC LIMIT 1", (run_id, epoch or 0)).fetchone()
    if marker:
        return dict(cause='disk_full', errno=28, event_id=marker[0], message=marker[1], recorded_at=None, basis='ledger event')
    recorded = read(root, run_id) if root else None
    if recorded and (started is None or _instant(recorded.get('recorded_at')) >= _instant(started)):
        return dict(cause='disk_full', errno=28, event_id=None, message=recorded.get('error'), recorded_at=recorded.get('recorded_at'),
                    basis='record allocated at run start')
    return None
