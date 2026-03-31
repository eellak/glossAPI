from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class TransferItem:
    canonical_filename: str
    remote_path: str
    remote_size_bytes: int
    remote_name: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS transfer_items (
    canonical_filename TEXT PRIMARY KEY,
    remote_path TEXT NOT NULL,
    remote_size_bytes INTEGER NOT NULL,
    remote_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    priority_rank INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    transfer_started_at TEXT,
    transfer_finished_at TEXT,
    last_seen_size_bytes INTEGER NOT NULL DEFAULT 0
);
"""

PDF_NAME_PATTERN = re.compile(r"([A-Za-z0-9._-]+\.pdf(?:\.[A-Za-z0-9_-]+)?)", re.IGNORECASE)
FILENAME_KEYS = ("filename", "canonical_filename", "md_filename", "source_filename")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_pdf_stage_pull",
        description="Resumable staged pull of OpenArchives PDFs from the Greece storage box.",
    )
    p.add_argument("--manifest", required=True, help="TSV manifest with canonical_filename, remote_path, remote_size_bytes, remote_name.")
    p.add_argument("--work-root", required=True, help="Root directory for downloads, partials, logs, and state.")
    p.add_argument("--remote-host", default="debian@83.212.80.170")
    p.add_argument("--password-env", default="GREECE_BOX_PASSWORD", help="Environment variable containing the remote SSH password.")
    p.add_argument("--transport", choices=("sftp", "rsync"), default="sftp")
    p.add_argument("--max-attempts", type=int, default=20)
    p.add_argument("--connect-timeout", type=int, default=30)
    p.add_argument("--io-timeout", type=int, default=180)
    p.add_argument("--sleep-after-failure", type=float, default=10.0)
    p.add_argument("--summary-interval-seconds", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=0, help="Optional limit for testing.")
    p.add_argument(
        "--priority-dir",
        default=None,
        help="Directory of dynamic priority files or filename lists. Items here are transferred first.",
    )
    return p.parse_args(argv)


class TransferState:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(SCHEMA)
        self._ensure_columns()
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _ensure_columns(self) -> None:
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(transfer_items)").fetchall()}
        if "priority_rank" not in cols:
            self.conn.execute("ALTER TABLE transfer_items ADD COLUMN priority_rank INTEGER NOT NULL DEFAULT 0")

    def sync_manifest(self, items: Iterable[TransferItem]) -> None:
        rows = [
            (item.canonical_filename, item.remote_path, int(item.remote_size_bytes), item.remote_name)
            for item in items
        ]
        self.conn.executemany(
            """
            INSERT INTO transfer_items (
                canonical_filename, remote_path, remote_size_bytes, remote_name, status
            ) VALUES (?, ?, ?, ?, 'pending')
            ON CONFLICT(canonical_filename) DO UPDATE SET
                remote_path=excluded.remote_path,
                remote_size_bytes=excluded.remote_size_bytes,
                remote_name=excluded.remote_name
            """,
            rows,
        )
        self.conn.commit()

    def reset_stale_in_progress(self) -> None:
        self.conn.execute(
            """
            UPDATE transfer_items
            SET status='pending',
                last_error=CASE
                    WHEN last_error = '' THEN 'Recovered from interrupted transfer'
                    ELSE last_error || ' | Recovered from interrupted transfer'
                END
            WHERE status='in_progress'
            """
        )
        self.conn.commit()

    def mark_completed_if_present(self, downloads_dir: Path, partial_dir: Path) -> None:
        cur = self.conn.execute(
            "SELECT canonical_filename, remote_size_bytes, status FROM transfer_items"
        )
        updates = []
        for canonical_filename, remote_size_bytes, status in cur.fetchall():
            final_path = downloads_dir / canonical_filename
            if final_path.exists() and final_path.stat().st_size == int(remote_size_bytes):
                updates.append((int(remote_size_bytes), utc_now(), canonical_filename))
                continue
            part_path = partial_dir / f"{canonical_filename}.part"
            if part_path.exists() and status == "completed":
                self.conn.execute(
                    """
                    UPDATE transfer_items
                    SET status='pending',
                        last_error='Final file missing; resuming from partial',
                        transfer_finished_at=NULL
                    WHERE canonical_filename=?
                    """,
                    (canonical_filename,),
                )
        if updates:
            self.conn.executemany(
                """
                UPDATE transfer_items
                SET status='completed',
                    last_seen_size_bytes=?,
                    transfer_finished_at=?,
                    last_error=''
                WHERE canonical_filename=?
                """,
                updates,
            )
        self.conn.commit()

    def next_item(self, *, max_attempts: int) -> Optional[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.execute(
            """
            SELECT *
            FROM transfer_items
            WHERE status IN ('pending', 'failed')
              AND attempts < ?
            ORDER BY priority_rank DESC, attempts ASC, canonical_filename ASC
            LIMIT 1
            """,
            (max_attempts,),
        )
        return cur.fetchone()

    def mark_in_progress(self, canonical_filename: str, current_size: int) -> None:
        self.conn.execute(
            """
            UPDATE transfer_items
            SET status='in_progress',
                attempts=attempts+1,
                transfer_started_at=?,
                last_seen_size_bytes=?,
                last_error=''
            WHERE canonical_filename=?
            """,
            (utc_now(), int(current_size), canonical_filename),
        )
        self.conn.commit()

    def mark_completed(self, canonical_filename: str, size_bytes: int) -> None:
        self.conn.execute(
            """
            UPDATE transfer_items
            SET status='completed',
                transfer_finished_at=?,
                last_seen_size_bytes=?,
                last_error=''
            WHERE canonical_filename=?
            """,
            (utc_now(), int(size_bytes), canonical_filename),
        )
        self.conn.commit()

    def mark_failed(self, canonical_filename: str, error: str, size_bytes: int) -> None:
        self.conn.execute(
            """
            UPDATE transfer_items
            SET status='failed',
                last_error=?,
                last_seen_size_bytes=?
            WHERE canonical_filename=?
            """,
            (str(error), int(size_bytes), canonical_filename),
        )
        self.conn.commit()

    def counts(self) -> dict[str, int]:
        cur = self.conn.execute(
            """
            SELECT status, COUNT(*) AS c
            FROM transfer_items
            GROUP BY status
            """
        )
        counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        for status, count in cur.fetchall():
            counts[str(status)] = int(count)
        counts["total"] = sum(counts.values())
        return counts

    def byte_counts(self) -> dict[str, int]:
        cur = self.conn.execute(
            """
            SELECT
                COALESCE(SUM(remote_size_bytes), 0) AS bytes_total,
                COALESCE(SUM(CASE WHEN status = 'completed' THEN remote_size_bytes ELSE 0 END), 0) AS bytes_completed,
                COALESCE(SUM(CASE WHEN status = 'in_progress' THEN last_seen_size_bytes ELSE 0 END), 0) AS bytes_in_progress
            FROM transfer_items
            """
        )
        row = cur.fetchone()
        bytes_total = int(row[0] or 0)
        bytes_completed = int(row[1] or 0)
        bytes_in_progress = int(row[2] or 0)
        bytes_remaining = max(0, bytes_total - bytes_completed)
        return {
            "bytes_total": bytes_total,
            "bytes_completed": bytes_completed,
            "bytes_in_progress": bytes_in_progress,
            "bytes_remaining": bytes_remaining,
        }

    def set_priorities(self, canonical_filenames: set[str]) -> None:
        self.conn.execute("UPDATE transfer_items SET priority_rank=0 WHERE priority_rank != 0")
        if canonical_filenames:
            batch = []
            for name in sorted(canonical_filenames):
                batch.append(name)
                if len(batch) >= 500:
                    placeholders = ",".join("?" for _ in batch)
                    self.conn.execute(
                        f"UPDATE transfer_items SET priority_rank=100 WHERE canonical_filename IN ({placeholders})",
                        batch,
                    )
                    batch.clear()
            if batch:
                placeholders = ",".join("?" for _ in batch)
                self.conn.execute(
                    f"UPDATE transfer_items SET priority_rank=100 WHERE canonical_filename IN ({placeholders})",
                    batch,
                )
        self.conn.commit()

    def priority_counts(self) -> dict[str, int]:
        cur = self.conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN priority_rank > 0 THEN 1 ELSE 0 END), 0) AS priority_total,
                COALESCE(SUM(CASE WHEN priority_rank > 0 AND status='pending' THEN 1 ELSE 0 END), 0) AS priority_pending,
                COALESCE(SUM(CASE WHEN priority_rank > 0 AND status='completed' THEN 1 ELSE 0 END), 0) AS priority_completed,
                COALESCE(SUM(CASE WHEN priority_rank > 0 AND status='failed' THEN 1 ELSE 0 END), 0) AS priority_failed
            FROM transfer_items
            """
        )
        row = cur.fetchone()
        return {
            "priority_total": int(row[0] or 0),
            "priority_pending": int(row[1] or 0),
            "priority_completed": int(row[2] or 0),
            "priority_failed": int(row[3] or 0),
        }


def read_manifest(path: Path) -> list[TransferItem]:
    items: list[TransferItem] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"canonical_filename", "remote_path", "remote_size_bytes", "remote_name"}
        if not required.issubset(reader.fieldnames or set()):
            raise SystemExit(f"Manifest missing required columns: {sorted(required)}")
        for row in reader:
            items.append(
                TransferItem(
                    canonical_filename=str(row["canonical_filename"]).strip(),
                    remote_path=str(row["remote_path"]).strip(),
                    remote_size_bytes=int(row["remote_size_bytes"]),
                    remote_name=str(row["remote_name"]).strip(),
                )
            )
    return items


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sshpass_env(password_env: str) -> dict[str, str]:
    env = os.environ.copy()
    secret = env.get(password_env)
    if not secret:
        raise SystemExit(f"Password env var '{password_env}' is not set.")
    env["SSHPASS"] = secret
    return env


def ssh_transport_options(connect_timeout: int) -> list[str]:
    return [
        "-o",
        "BatchMode=no",
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=yes",
        "-o",
        f"ConnectTimeout={int(connect_timeout)}",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ConnectionAttempts=3",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/tmp/greece_box_known_hosts",
    ]


def canonicalize_pdf_name(raw: str) -> Optional[str]:
    text = os.path.basename(str(raw).strip())
    if not text:
        return None
    lower = text.lower()
    marker = ".pdf."
    if marker in lower:
        idx = lower.index(marker)
        return text[: idx + 4]
    if lower.endswith(".pdf"):
        return text
    return None


def _walk_json_strings(obj) -> Iterable[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str):
                yield key
            yield from _walk_json_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_json_strings(item)
    elif isinstance(obj, str):
        yield obj


def _extract_priority_filenames_from_csv(path: Path) -> set[str]:
    results: set[str] = set()
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {field.strip() for field in (reader.fieldnames or []) if field}
        keyed = any(key in fields for key in FILENAME_KEYS)
        for row in reader:
            if keyed:
                for key in FILENAME_KEYS:
                    value = row.get(key)
                    if value:
                        canonical = canonicalize_pdf_name(value)
                        if canonical is not None:
                            results.add(canonical)
                            break
            else:
                for value in row.values():
                    if not value:
                        continue
                    for match in PDF_NAME_PATTERN.findall(str(value)):
                        canonical = canonicalize_pdf_name(match)
                        if canonical is not None:
                            results.add(canonical)
    return results


def _extract_priority_filenames_from_json(path: Path) -> set[str]:
    results: set[str] = set()
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    for text in _walk_json_strings(data):
        canonical = canonicalize_pdf_name(text)
        if canonical is not None:
            results.add(canonical)
            continue
        for match in PDF_NAME_PATTERN.findall(text):
            canonical = canonicalize_pdf_name(match)
            if canonical is not None:
                results.add(canonical)
    return results


def _extract_priority_filenames_from_text(path: Path) -> set[str]:
    results: set[str] = set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        canonical = canonicalize_pdf_name(line)
        if canonical is not None:
            results.add(canonical)
    for match in PDF_NAME_PATTERN.findall(text):
        canonical = canonicalize_pdf_name(match)
        if canonical is not None:
            results.add(canonical)
    return results


def load_priority_filenames(priority_dir: Path) -> set[str]:
    results: set[str] = set()
    if not priority_dir.exists():
        return results
    for path in sorted(priority_dir.rglob("*")):
        if not path.is_file():
            continue
        direct = canonicalize_pdf_name(path.name)
        if direct is not None:
            results.add(direct)
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                results.update(_extract_priority_filenames_from_csv(path))
            elif suffix == ".json":
                results.update(_extract_priority_filenames_from_json(path))
            elif suffix in {".txt", ".list", ".lst", ".log"}:
                results.update(_extract_priority_filenames_from_text(path))
            else:
                continue
        except Exception:
            continue
    return results


def rsync_one(
    *,
    remote_host: str,
    remote_path: str,
    temp_path: Path,
    password_env: str,
    connect_timeout: int,
    io_timeout: int,
) -> subprocess.CompletedProcess[str]:
    ssh_cmd = (
        "ssh "
        "-o BatchMode=no "
        "-o PreferredAuthentications=password "
        "-o PubkeyAuthentication=no "
        "-o KbdInteractiveAuthentication=yes "
        f"-o ConnectTimeout={int(connect_timeout)} "
        "-o ServerAliveInterval=15 "
        "-o ServerAliveCountMax=3 "
        "-o ConnectionAttempts=3 "
        "-o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/tmp/greece_box_known_hosts"
    )
    cmd = [
        "sshpass",
        "-e",
        "rsync",
        "-av",
        "--partial",
        "--append-verify",
        "--inplace",
        f"--timeout={int(io_timeout)}",
        "-e",
        ssh_cmd,
        f"{remote_host}:{remote_path}",
        str(temp_path),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=sshpass_env(password_env))


def sftp_one(
    *,
    remote_host: str,
    remote_path: str,
    temp_path: Path,
    password_env: str,
    connect_timeout: int,
    io_timeout: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "sshpass",
        "-e",
        "sftp",
        *ssh_transport_options(connect_timeout),
        "-b",
        "-",
        remote_host,
    ]
    batch = f'reget "{remote_path}" "{temp_path}"\n'
    return subprocess.run(cmd, capture_output=True, text=True, env=sshpass_env(password_env), input=batch)


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    priority_dir = Path(args.priority_dir).expanduser().resolve() if args.priority_dir else (work_root / "unreachable_from_source_20260331")
    downloads_dir = work_root / "downloads"
    partial_dir = work_root / "partials"
    logs_dir = work_root / "logs"
    state_dir = work_root / "state"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    state = TransferState(state_dir / "transfer_state.sqlite3")
    items = read_manifest(manifest_path)
    if args.limit and int(args.limit) > 0:
        items = items[: int(args.limit)]
    state.sync_manifest(items)
    state.reset_stale_in_progress()
    state.mark_completed_if_present(downloads_dir, partial_dir)
    manifest_names = {item.canonical_filename for item in items}

    stop_requested = False

    def _handle_signal(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[transfer] signal {signum} received; stopping after current file", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    last_summary_ts = 0.0
    current_path = state_dir / "current_transfer.json"
    summary_path = state_dir / "summary.json"
    events_path = logs_dir / "events.jsonl"
    priority_summary_path = state_dir / "priority_summary.json"
    priority_available_path = state_dir / "priority_available_in_manifest.txt"
    priority_missing_path = state_dir / "priority_missing_in_manifest.txt"
    last_priority_set: Optional[set[str]] = None

    def refresh_priorities() -> dict[str, int]:
        nonlocal last_priority_set
        requested = load_priority_filenames(priority_dir)
        if last_priority_set is None or requested != last_priority_set:
            available = requested & manifest_names
            missing = requested - manifest_names
            state.set_priorities(available)
            priority_available_path.write_text(
                "".join(f"{name}\n" for name in sorted(available)),
                encoding="utf-8",
            )
            priority_missing_path.write_text(
                "".join(f"{name}\n" for name in sorted(missing)),
                encoding="utf-8",
            )
            write_json(
                priority_summary_path,
                {
                    "updated_at": utc_now(),
                    "priority_dir": str(priority_dir),
                    "requested_total": len(requested),
                    "available_in_manifest_total": len(available),
                    "missing_in_manifest_total": len(missing),
                },
            )
            last_priority_set = requested
        return state.priority_counts()

    priority_counts = refresh_priorities()

    while not stop_requested:
        priority_counts = refresh_priorities()
        row = state.next_item(max_attempts=int(args.max_attempts))
        if row is None:
            write_json(summary_path, {"updated_at": utc_now(), **state.counts(), **state.byte_counts(), **priority_counts, "done": True})
            break

        canonical = str(row["canonical_filename"])
        remote_path = str(row["remote_path"])
        remote_size = int(row["remote_size_bytes"])
        final_path = downloads_dir / canonical
        temp_path = partial_dir / f"{canonical}.part"
        current_size = temp_path.stat().st_size if temp_path.exists() else 0

        state.mark_in_progress(canonical, current_size)
        write_json(
            current_path,
            {
                "updated_at": utc_now(),
                "transport": str(args.transport),
                "canonical_filename": canonical,
                "remote_path": remote_path,
                "remote_size_bytes": remote_size,
                "partial_path": str(temp_path),
                "partial_size_bytes": current_size,
                "attempt_number": int(row["attempts"]) + 1,
            },
        )
        append_event(
            events_path,
            {
                "ts": utc_now(),
                "event": "start",
                "transport": str(args.transport),
                "canonical_filename": canonical,
                "remote_path": remote_path,
                "remote_size_bytes": remote_size,
                "partial_size_bytes": current_size,
                "attempt_number": int(row["attempts"]) + 1,
            },
        )

        transfer_kwargs = {
            "remote_host": str(args.remote_host),
            "remote_path": remote_path,
            "temp_path": temp_path,
            "password_env": str(args.password_env),
            "connect_timeout": int(args.connect_timeout),
            "io_timeout": int(args.io_timeout),
        }
        if str(args.transport) == "rsync":
            result = rsync_one(**transfer_kwargs)
        else:
            result = sftp_one(**transfer_kwargs)

        if result.returncode == 0 and temp_path.exists():
            actual_size = temp_path.stat().st_size
            if remote_size > 0 and actual_size != remote_size:
                state.mark_failed(
                    canonical,
                    f"Size mismatch after transfer: expected {remote_size}, got {actual_size}",
                    actual_size,
                )
            else:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp_path, final_path)
                state.mark_completed(canonical, actual_size)
                append_event(
                    events_path,
                    {
                        "ts": utc_now(),
                        "event": "completed",
                        "transport": str(args.transport),
                        "canonical_filename": canonical,
                        "size_bytes": actual_size,
                    },
                )
        else:
            actual_size = temp_path.stat().st_size if temp_path.exists() else 0
            error = (result.stderr or result.stdout or "").strip()[-4000:]
            state.mark_failed(canonical, error or f"transfer failed with code {result.returncode}", actual_size)
            append_event(
                events_path,
                {
                    "ts": utc_now(),
                    "event": "failed",
                    "transport": str(args.transport),
                    "canonical_filename": canonical,
                    "return_code": int(result.returncode),
                    "partial_size_bytes": actual_size,
                    "error": error or f"transfer failed with code {result.returncode}",
                },
            )
            time.sleep(float(args.sleep_after_failure))

        now = time.time()
        if now - last_summary_ts >= float(args.summary_interval_seconds):
            priority_counts = refresh_priorities()
            write_json(summary_path, {"updated_at": utc_now(), **state.counts(), **state.byte_counts(), **priority_counts, "done": False})
            last_summary_ts = now

    if current_path.exists():
        try:
            current_path.unlink()
        except Exception:
            pass

    priority_counts = refresh_priorities()
    write_json(summary_path, {"updated_at": utc_now(), **state.counts(), **state.byte_counts(), **priority_counts, "done": True})
    state.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
