"""Canonical JSON and lossless typed source values, without pandas imports."""
from __future__ import annotations

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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_records(frame):
    """Preserve column scalar types; iterrows can round mixed numeric rows."""
    for index, values in zip(frame.index, frame.itertuples(index=False, name=None)):
        yield index, dict(zip(frame.columns, values))
