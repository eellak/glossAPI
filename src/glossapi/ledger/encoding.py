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


def file_hash(path):
    """Rehash bytes at every boundary, fencing descriptor/path replacement.

    Filesystem timestamps can alias rapid same-size writes. They are never
    accepted as proof that a previously hashed file remains unchanged.
    """
    path = Path(path)
    identity = lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    before = identity(path.stat())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if identity(os.fstat(stream.fileno())) != before:
            raise ValueError('File changed before hashing')
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if identity(os.fstat(stream.fileno())) != before or identity(path.stat()) != before:
            raise ValueError('File changed while hashing')
    value=digest.hexdigest()
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


def unpack_evidence(value):
    """Recover all original explanatory JSON fields without opening payloads."""
    if value.get("encoding") != "columnar_json_v1":
        return value
    def unpack(item):
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
