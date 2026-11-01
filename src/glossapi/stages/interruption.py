"""A disk_full record allocated before it is needed (K4BI-G2, W18 M10).

A filesystem too full for one more ledger page cannot record why the run stopped. Every run allocates this small
file at its start (``.glossapi/interruption.json``, a fixed size), and an exhausted disk overwrites it in place: no
new block is needed. ``status``, ``report`` and the resume read it when the ledger holds no ``disk_full`` event.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SIZE = 4096
NAME = '.glossapi/interruption.json'


def _payload(value):
    data = json.dumps(value, sort_keys=True).encode()[:SIZE-1]
    return data + b' '*(SIZE-len(data))


def allocate(root):
    """Create the fixed-size record at run start (kept as is while it names an interruption the resume must read)."""
    path = Path(root)/NAME
    if path.is_file() and path.stat().st_size == SIZE:
        return
    clear(root)


def clear(root):
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


def record(root, run_id, error):
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


def read(root, run_id=None):
    """The recorded disk_full cause for ``run_id`` (any run when None), or None."""
    try:
        value = json.loads((Path(root)/NAME).read_bytes().decode().strip() or '{}')
    except (OSError, ValueError):
        return None
    if value.get('cause') != 'disk_full' or (run_id is not None and value.get('run_id') != run_id):
        return None
    return value
