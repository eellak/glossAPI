"""Canonical JSON and lossless typed source values, without pandas imports."""
from __future__ import annotations

import os
import base64
import datetime
import decimal
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping
from typing import Any


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def snapshot_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def typed_value(value):
    """Tag containers too, so user columns named '$type' cannot collide."""
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, decimal.Decimal):
        return {"type": "decimal", "value": str(value)}
    scalar_type = type(value).__module__ + "." + type(value).__name__
    if type(value).__module__.startswith("pandas") and type(value).__name__ in ("NAType", "NaTType"):
        return {"type": "null", "value": None, "source_type": scalar_type}
    if type(value).__module__.startswith("numpy") and type(value).__name__ == "ndarray":
        return {"type": "array", "dtype": str(value.dtype), "shape": list(value.shape), "value": typed_value(value.tolist())}
    # Timestamp.isoformat retains nanoseconds; tag its original scalar class.
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return {"type": type(value).__name__, "value": value.isoformat(),
                "fold": getattr(value, "fold", 0)}
    if isinstance(value, datetime.timedelta):
        return {"type": "timedelta", "value": [value.days, value.seconds, value.microseconds]}
    if isinstance(value, Mapping):
        return {"type": "mapping", "value": [[typed_value(k), typed_value(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "value": [typed_value(v) for v in value]}
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return {"type": "scalar", "source_type": scalar_type, "value": typed_value(value.item())}
    raise TypeError(f"Unsupported source value type: {scalar_type}")


def untyped_value(value):
    tag, payload = value["type"], value["value"]
    if tag == "null": return None
    if tag in ("str", "bool"): return payload
    if tag == "int": return int(payload)
    if tag == "float": return float.fromhex(payload)
    if tag == "bytes": return base64.b64decode(payload, validate=True)
    if tag == "decimal": return decimal.Decimal(payload)
    if tag in ("datetime", "Timestamp"):
        return datetime.datetime.fromisoformat(payload).replace(fold=value.get("fold", 0))
    if tag == "date": return datetime.date.fromisoformat(payload)
    if tag == "time": return datetime.time.fromisoformat(payload).replace(fold=value.get("fold", 0))
    if tag == "timedelta": return datetime.timedelta(days=payload[0], seconds=payload[1], microseconds=payload[2])
    if tag == "mapping": return {untyped_value(k): untyped_value(v) for k, v in payload}
    if tag == "list": return [untyped_value(v) for v in payload]
    if tag == "tuple": return tuple(untyped_value(v) for v in payload)
    if tag == "path": return Path(payload)
    if tag in ("scalar", "array"): return untyped_value(payload)
    raise ValueError(f"Unknown source type: {tag}")


def evidence(value):
    """Versioned envelope used by every details and snapshot writer."""
    if not isinstance(value, dict):
        raise TypeError("Evidence must be an object")
    return {"schema_version": 1, **value}


# Run-scoped reuse of the hashes of files glossAPI itself wrote (K4(c) part 3, coordinator decision 1, K4C-G08).
# Within one public call (a streaming run with its publication barriers, or one sequential phase), a file glossAPI
# wrote, or whose recorded producer hash glossAPI has just verified by reading it, and has not reopened for writing is
# not re-hashed: its recorded hash is reused, keyed by (st_dev, st_ino, st_size, st_mtime_ns). Only callers that know
# the file is glossAPI's own register it (remember_own_hash); the user's originals and downloads are never registered,
# so every boundary that hashes them still reads their bytes. The registry exists only inside run_hash_scope() and is
# discarded at the end of the call, so anything that could change between runs is re-checked at the next run start.
_run_hashes: dict[tuple[int, int, int, int], str] | None = None


def own_hash_key(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


class run_hash_scope:
    """Activate the registry for one public call; a nested call keeps the outer call's registry."""
    outer: dict[tuple[int, int, int, int], str] | None = None

    def __enter__(self) -> 'run_hash_scope':
        global _run_hashes
        self.outer = _run_hashes
        if _run_hashes is None:
            _run_hashes = {}
        return self

    def __exit__(self, *exc: object) -> None:
        global _run_hashes
        _run_hashes = self.outer


def remember_own_hash(path: 'str | os.PathLike[str]', digest: str | None, size: int | None = None, *,
                      info: os.stat_result | None = None) -> None:
    """Register the hash of a file glossAPI wrote or just verified (no-op outside a run scope)."""
    registry = _run_hashes
    if registry is None or digest is None:
        return
    try:
        info = info or os.stat(path)
    except OSError:
        return
    if size is not None and info.st_size != size:
        return
    registry[own_hash_key(info)] = digest


def forget_own_hash(path: 'str | os.PathLike[str]') -> None:
    """Drop a registration before glossAPI reopens the file for writing (same inode)."""
    registry = _run_hashes
    if registry is None:
        return
    try:
        info = os.stat(path)
    except OSError:
        return
    registry.pop(own_hash_key(info), None)


def reused_own_hash(info: os.stat_result) -> str | None:
    """The registered hash of an unchanged own file, or None."""
    registry = _run_hashes
    return None if registry is None else registry.get(own_hash_key(info))


def file_hash(path: 'str | os.PathLike[str]', *, own: bool = False) -> str:
    """Rehash bytes at every boundary, fencing descriptor/path replacement.

    Filesystem timestamps can alias rapid same-size writes, so they are never
    accepted as proof for a file glossAPI did not write. The one exception is
    the run-scoped registry above: a file glossAPI itself wrote (or verified)
    in this run and has not reopened for writing keeps its recorded hash.
    ``own=True`` registers the computed hash (the caller knows the file is
    glossAPI's own, e.g. a managed artifact whose recorded hash it checks).
    """
    path = Path(path)
    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    info = path.stat()
    reused = reused_own_hash(info)
    if reused is not None:
        return reused
    before = identity(info)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if identity(os.fstat(stream.fileno())) != before:
            raise ValueError('File changed before hashing')
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if identity(os.fstat(stream.fileno())) != before or identity(path.stat()) != before:
            raise ValueError('File changed while hashing')
    value=digest.hexdigest()
    if own:
        remember_own_hash(path, value, info=info)
    return value


def dataframe_records(frame):
    """Preserve column scalar types; iterrows can round mixed numeric rows."""
    for index, values in zip(frame.index, frame.itertuples(index=False, name=None)):
        yield index, dict(zip(frame.columns, values))


def pack_evidence(value):
    """Factor repeated field names and categorical values, losslessly.

    Font/page diagnostic arrays often repeat a wide schema and long reason
    strings thousands of times. This is a JSON table with dictionaries, not a
    text payload or an opaque compressed blob. Every original field survives.
    """
    def pack(item):
        if isinstance(item, list):
            if len(item) >= 4 and all(isinstance(row, dict) for row in item):
                columns = sorted(item[0])
                if all(sorted(row) == columns for row in item):
                    rows = [[pack(row[key]) for key in columns] for row in item]
                    dictionaries = {}
                    for index in range(len(columns)):
                        values = [row[index] for row in rows]
                        if all(v is None or isinstance(v, str) for v in values):
                            unique = list(dict.fromkeys(values))
                            codes = {v: n for n, v in enumerate(unique)}
                            encoded = [codes[v] for v in values]
                            if len(canonical_json([unique, encoded])) + 24 < len(canonical_json(values)):
                                dictionaries[str(index)] = unique
                                for row, code in zip(rows, encoded): row[index] = code
                    return {"_table": {"columns": columns, "rows": rows, "dictionaries": dictionaries}}
            return [pack(v) for v in item]
        if isinstance(item, dict):
            result = {key: pack(v) for key, v in item.items()}
            return {"_object": result} if "_table" in item or "_object" in item else result
        return item
    encoded = {"kind": value.get("kind"), "encoding": "columnar_json_v1", "value": pack(value)}
    return encoded if len(canonical_json(encoded)) + 128 < len(canonical_json(value)) else value


def unpack_evidence(value: Any) -> Any:
    """Recover all original explanatory JSON fields without opening payloads."""
    if value.get("encoding") != "columnar_json_v1":
        return value
    def unpack(item: Any) -> Any:
        if isinstance(item, list): return [unpack(v) for v in item]
        if isinstance(item, dict):
            if "_table" in item:
                table = item["_table"]
                return [{key: unpack(table["dictionaries"][str(i)][row[i]] if str(i) in table["dictionaries"] else row[i])
                         for i, key in enumerate(table["columns"])} for row in table["rows"]]
            if "_object" in item:
                return {key: unpack(v) for key, v in item["_object"].items()}
            return {key: unpack(v) for key, v in item.items()}
        return item
    return unpack(value["value"])
