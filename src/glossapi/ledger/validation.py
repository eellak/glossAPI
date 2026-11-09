"""Version-1 JSON evidence validation, independent of SQLite JSON extensions."""
import math
import re


def validate_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value): raise ValueError("Evidence numbers must be finite")
        return
    if type(value) is list:
        for item in value: validate_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values(): validate_json(item)
        return
    raise ValueError("Evidence must contain only JSON objects, arrays and scalar values")


def validate_evidence(value, snapshot_kind=None):
    """Validate one evidence object. ``snapshot_kind`` is the authoritative kind of a snapshot: the kind the writer
    records it under (``LedgerWriter.snapshot``) or the stored ``snapshots.kind`` row (``integrity``); None for other
    evidence, such as event details."""
    validate_json(value)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported evidence schema")
    for name in ("evidence_sha256", "tools_sha256"):
        digest = value.get(name)
        if digest is not None and (not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)):
            raise ValueError(f"Invalid {name}")
    # A settings snapshot records public API parameters as given: a selector such as files=None (auto-discovery) is
    # not an inventory. Every other evidence kind keeps the inventory-array rule (FINAL-ASTRA F03). Only the
    # authoritative snapshot kind grants that exemption; a payload's own "kind" field cannot, and a payload that
    # claims or denies "settings" against its snapshot kind is refused (FINAL-ASTRA-c2 C2-01). Other kinds may
    # legitimately carry their own payload kind (for example legacy_store deltas), so only "settings" is bound.
    payload_kind = value.get("kind")
    if (snapshot_kind is not None and payload_kind is not None and payload_kind != snapshot_kind
            and "settings" in (snapshot_kind, payload_kind)):
        raise ValueError(f"Evidence kind {payload_kind!r} conflicts with its snapshot kind {snapshot_kind!r}")
    if snapshot_kind != "settings":
        for name in ("files", "rows", "columns", "unavailable", "models", "packages", "native", "dictionaries"):
            if name in value and not isinstance(value[name], list): raise ValueError(f"{name} must be an array")
    if "wrapper_raised" in value and type(value["wrapper_raised"]) is not bool:
        raise ValueError("wrapper_raised must be boolean")
    if "reuse_eligible" in value and type(value["reuse_eligible"]) is not bool:
        raise ValueError("reuse_eligible must be boolean")
