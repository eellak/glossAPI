"""The corpus validation mark (fix 15, H3: a refusal that cannot be recorded must still revoke completion).

A fixed-size file, ``.glossapi/validation.json``, overwritten in place like the disk_full record
(``stages.interruption``): once it exists, writing it needs no new block, so it works on a full disk or a full ledger.

- ``mark_pending`` runs at every ``Corpus.run`` start BEFORE start-up validation or any repair. If it cannot be
  written, the start aborts before any repair (``UnresolvedRecoveryError``, reason ``validation_mark_unwritable``).
- ``mark_validated`` runs only after a passing completion gate persisted its verdict (``LedgerWriter.completion_gate``).
- Every read surface (``ledger.reader.certification``) treats ``pending`` as not certified: a start whose refusal could
  not be recorded in the ledger (or that died) leaves the corpus ``unverified``, never ``completed``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SIZE = 4096
NAME = '.glossapi/validation.json'


def _payload(value: dict[str, Any]) -> bytes:
    data = json.dumps(value, sort_keys=True).encode()[:SIZE-1]
    return data + b' '*(SIZE-len(data))


def _write(root: Any, value: dict[str, Any]) -> None:
    path = Path(root)/NAME
    created = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        written = os.pwrite(descriptor, _payload(value), 0)
        if written != SIZE:
            raise OSError(f'short write of the validation mark ({written} of {SIZE} bytes)')
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if created:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def mark_pending(root: Any, run_id: str | None) -> None:
    """Durably mark the corpus validation pending; raises ``OSError`` if the mark cannot be written."""
    from datetime import datetime, timezone
    _write(root, {'state': 'pending', 'run_id': run_id, 'pid': os.getpid(), 'since': datetime.now(timezone.utc).isoformat()})


def mark_validated(root: Any, run_id: str, event_id: int, fingerprint: str) -> bool:
    """After a passing gate: the corpus is validated at that verdict. Best effort (a failure leaves it pending: the
    refuse direction)."""
    from datetime import datetime, timezone
    try:
        _write(root, {'state': 'validated', 'run_id': run_id, 'event_id': event_id, 'authority_fingerprint': fingerprint,
                      'at': datetime.now(timezone.utc).isoformat()})
        return True
    except OSError:
        return False


def read(root: Any) -> dict[str, Any] | None:
    """The mark, or None when this corpus never had one (a ledger written before fix 15)."""
    try:
        text = (Path(root)/NAME).read_bytes().decode().strip()
    except OSError:
        return None
    try:
        value = json.loads(text or '{}')
    except ValueError:
        return {'state': 'unreadable'}
    return value if isinstance(value, dict) else {'state': 'unreadable'}
