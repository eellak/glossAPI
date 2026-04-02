"""Durable batch queue helpers for multi-GPU DeepSeek OCR runs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
QUEUE_MAIN = "main"
QUEUE_REPAIR = "repair"


def _empty_counts() -> Dict[str, int]:
    return {
        STATUS_PENDING: 0,
        STATUS_RUNNING: 0,
        STATUS_DONE: 0,
        STATUS_FAILED: 0,
        "total": 0,
    }


def _normalize_queue_name(queue_name: str) -> str:
    queue_norm = str(queue_name or QUEUE_MAIN).strip().lower()
    if queue_norm not in {QUEUE_MAIN, QUEUE_REPAIR}:
        raise ValueError(f"Unsupported queue name: {queue_name}")
    return queue_norm


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_work_db(db_path: Path, *, batches: Iterable[Dict[str, Any]], replace: bool = True) -> None:
    db_path = Path(db_path).expanduser().resolve()
    if replace and db_path.exists():
        db_path.unlink()
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS work_items (
                batch_id INTEGER PRIMARY KEY,
                queue_name TEXT NOT NULL,
                queue_key TEXT NOT NULL UNIQUE,
                batch_json TEXT NOT NULL,
                pages INTEGER NOT NULL,
                status TEXT NOT NULL,
                worker_id TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                started_at REAL,
                finished_at REAL,
                last_heartbeat REAL,
                last_error TEXT,
                result_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_work_items_status ON work_items(status);
            CREATE INDEX IF NOT EXISTS idx_work_items_queue_status ON work_items(queue_name, status);
            CREATE INDEX IF NOT EXISTS idx_work_items_worker ON work_items(worker_id);
            """
        )
        rows = [
            (
                int(batch["batch_id"]),
                QUEUE_MAIN,
                str(batch.get("queue_key") or f"{QUEUE_MAIN}:{int(batch['batch_id'])}"),
                json.dumps(batch, sort_keys=True),
                int(batch.get("pages", 0)),
                STATUS_PENDING,
            )
            for batch in batches
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO work_items(batch_id, queue_name, queue_key, batch_json, pages, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def enqueue_batches(
    db_path: Path,
    *,
    queue_name: str,
    batches: Iterable[Dict[str, Any]],
) -> list[int]:
    queue_norm = _normalize_queue_name(queue_name)
    inserted_ids: list[int] = []
    with _connect(db_path) as conn:
        _with_transaction(conn)
        next_batch_id = int(
            conn.execute("SELECT COALESCE(MAX(batch_id), -1) + 1 AS next_id FROM work_items").fetchone()["next_id"]
        )
        for batch in batches:
            payload = dict(batch)
            queue_key = str(payload.get("queue_key") or f"{queue_norm}:{next_batch_id}")
            row = conn.execute(
                "SELECT batch_id FROM work_items WHERE queue_key = ?",
                (queue_key,),
            ).fetchone()
            if row is None:
                batch_id = int(payload.get("batch_id", next_batch_id))
                next_batch_id = max(next_batch_id, batch_id + 1)
            else:
                batch_id = int(row["batch_id"])
            payload["batch_id"] = batch_id
            payload["queue_name"] = queue_norm
            payload_json = json.dumps(payload, sort_keys=True)
            pages = int(payload.get("pages", 0))
            if row is None:
                conn.execute(
                    """
                    INSERT INTO work_items(batch_id, queue_name, queue_key, batch_json, pages, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (batch_id, queue_norm, queue_key, payload_json, pages, STATUS_PENDING),
                )
            else:
                conn.execute(
                    """
                    UPDATE work_items
                    SET queue_name = ?, batch_json = ?, pages = ?, status = ?, worker_id = NULL, attempt_count = 0,
                        started_at = NULL, finished_at = NULL, last_heartbeat = NULL, last_error = NULL, result_json = NULL
                    WHERE batch_id = ?
                    """,
                    (queue_norm, payload_json, pages, STATUS_PENDING, batch_id),
                )
            inserted_ids.append(batch_id)
        conn.commit()
    return inserted_ids


def _with_transaction(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def requeue_stale_running_batches(
    db_path: Path,
    *,
    stale_after_sec: float,
    now_ts: Optional[float] = None,
) -> int:
    now_value = float(now_ts) if now_ts is not None else float(time.time())
    cutoff = now_value - float(max(1.0, stale_after_sec))
    with _connect(db_path) as conn:
        _with_transaction(conn)
        cursor = conn.execute(
            """
            UPDATE work_items
            SET status = ?, worker_id = NULL, started_at = NULL, finished_at = NULL
            WHERE status = ? AND COALESCE(last_heartbeat, started_at, 0) < ?
            """,
            (STATUS_PENDING, STATUS_RUNNING, cutoff),
        )
        conn.commit()
        return int(cursor.rowcount or 0)


def requeue_worker_batches(
    db_path: Path,
    *,
    worker_id: str,
    error: Optional[str] = None,
    max_attempts: int = 2,
) -> int:
    max_attempts_value = max(1, int(max_attempts))
    with _connect(db_path) as conn:
        _with_transaction(conn)
        # `attempt_count` is incremented on claim. With the default max_attempts=2
        # each work item gets one retry after its first failed claim, then becomes
        # terminally failed instead of bouncing forever between workers.
        cursor = conn.execute(
            """
            UPDATE work_items
            SET status = CASE WHEN attempt_count < ? THEN ? ELSE ? END,
                worker_id = CASE WHEN attempt_count < ? THEN NULL ELSE ? END,
                started_at = NULL,
                finished_at = NULL,
                last_heartbeat = NULL,
                last_error = ?,
                result_json = NULL
            WHERE status = ? AND worker_id = ?
            """,
            (
                max_attempts_value,
                STATUS_PENDING,
                STATUS_FAILED,
                max_attempts_value,
                str(worker_id),
                str(error) if error else None,
                STATUS_RUNNING,
                str(worker_id),
            ),
        )
        conn.commit()
        return int(cursor.rowcount or 0)


def claim_next_batch(
    db_path: Path,
    *,
    worker_id: str,
    stale_after_sec: float,
    queue_name: str = QUEUE_MAIN,
    now_ts: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    queue_norm = _normalize_queue_name(queue_name)
    now_value = float(now_ts) if now_ts is not None else float(time.time())
    cutoff = now_value - float(max(1.0, stale_after_sec))
    with _connect(db_path) as conn:
        _with_transaction(conn)
        conn.execute(
            """
            UPDATE work_items
            SET status = ?, worker_id = NULL, started_at = NULL, finished_at = NULL
            WHERE status = ? AND COALESCE(last_heartbeat, started_at, 0) < ?
            """,
            (STATUS_PENDING, STATUS_RUNNING, cutoff),
        )
        row = conn.execute(
            """
            SELECT batch_id, batch_json
            FROM work_items
            WHERE status = ? AND queue_name = ?
            ORDER BY batch_id ASC
            LIMIT 1
            """,
            (STATUS_PENDING, queue_norm),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            """
            UPDATE work_items
            SET status = ?, worker_id = ?, attempt_count = attempt_count + 1, started_at = ?, last_heartbeat = ?, last_error = NULL
            WHERE batch_id = ?
            """,
            (STATUS_RUNNING, str(worker_id), now_value, now_value, int(row["batch_id"])),
        )
        conn.commit()
    return json.loads(str(row["batch_json"]))


def heartbeat_batch(db_path: Path, *, batch_id: int, worker_id: str, now_ts: Optional[float] = None) -> None:
    now_value = float(now_ts) if now_ts is not None else float(time.time())
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE work_items
            SET last_heartbeat = ?
            WHERE batch_id = ? AND status = ? AND worker_id = ?
            """,
            (now_value, int(batch_id), STATUS_RUNNING, str(worker_id)),
        )


def mark_batch_done(
    db_path: Path,
    *,
    batch_id: int,
    worker_id: str,
    result: Optional[Dict[str, Any]] = None,
    now_ts: Optional[float] = None,
) -> None:
    now_value = float(now_ts) if now_ts is not None else float(time.time())
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE work_items
            SET status = ?, finished_at = ?, last_heartbeat = ?, result_json = ?
            WHERE batch_id = ? AND worker_id = ?
            """,
            (
                STATUS_DONE,
                now_value,
                now_value,
                json.dumps(result, sort_keys=True) if result is not None else None,
                int(batch_id),
                str(worker_id),
            ),
        )


def mark_batch_failed(
    db_path: Path,
    *,
    batch_id: int,
    worker_id: str,
    error: str,
    max_attempts: int = 2,
    now_ts: Optional[float] = None,
) -> None:
    now_value = float(now_ts) if now_ts is not None else float(time.time())
    max_attempts_value = max(1, int(max_attempts))
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE work_items
            SET status = CASE WHEN attempt_count < ? THEN ? ELSE ? END,
                worker_id = CASE WHEN attempt_count < ? THEN NULL ELSE ? END,
                started_at = NULL,
                finished_at = ?,
                last_heartbeat = ?,
                last_error = ?,
                result_json = NULL
            WHERE batch_id = ? AND worker_id = ?
            """,
            (
                max_attempts_value,
                STATUS_PENDING,
                STATUS_FAILED,
                max_attempts_value,
                str(worker_id),
                now_value,
                now_value,
                str(error),
                int(batch_id),
                str(worker_id),
            ),
        )


def work_queue_counts(db_path: Path) -> Dict[str, int]:
    counts = _empty_counts()
    counts["by_queue"] = {
        QUEUE_MAIN: _empty_counts(),
        QUEUE_REPAIR: _empty_counts(),
    }
    with _connect(db_path) as conn:
        for row in conn.execute("SELECT queue_name, status, COUNT(*) AS count FROM work_items GROUP BY queue_name, status"):
            queue_name = _normalize_queue_name(str(row["queue_name"]))
            status = str(row["status"])
            count = int(row["count"])
            counts[status] = int(counts.get(status, 0)) + count
            counts["total"] += count
            counts["by_queue"][queue_name][status] = count
            counts["by_queue"][queue_name]["total"] += count
    return counts


def iter_work_items(db_path: Path) -> Iterable[Dict[str, Any]]:
    with _connect(db_path) as conn:
        for row in conn.execute(
            """
            SELECT batch_id, queue_name, queue_key, batch_json, pages, status, worker_id, attempt_count, started_at,
                   finished_at, last_heartbeat, last_error, result_json
            FROM work_items
            ORDER BY batch_id ASC
            """
        ):
            item = json.loads(str(row["batch_json"]))
            item.update(
                {
                    "queue_name": str(row["queue_name"]),
                    "queue_key": str(row["queue_key"]),
                    "status": str(row["status"]),
                    "worker_id": row["worker_id"],
                    "attempt_count": int(row["attempt_count"]),
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                    "last_heartbeat": row["last_heartbeat"],
                    "last_error": row["last_error"],
                    "result": json.loads(str(row["result_json"])) if row["result_json"] else None,
                    "pages": int(row["pages"]),
                }
            )
            yield item
