"""Content-addressed, replayable observations of mutable compatibility stores.

These deltas explain legacy writes; they never supply pipeline state. Row keys
are local to a store, not document identities. Text bodies remain hashed files.
"""
from collections import Counter
import hashlib
import json
import math
import sys

from .encoding import canonical_json, dataframe_records, typed_value


TEXT_FIELDS = frozenset(("text", "content", "title", "markdown", "latex", "body",
                         "section", "json_section", "header"))


def compact_value(value, *, field=None):
    if field in TEXT_FIELDS and isinstance(value, str):
        return {"text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                "characters": len(value), "bytes": len(value.encode("utf-8"))}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    # Tags wrap *all* containers, so user data cannot impersonate a type tag.
    return {"typed": typed_value(value)}


def identity_present(value):
    encoded = typed_value(value)
    while encoded["type"] == "scalar": encoded = encoded["value"]
    if encoded["type"] == "null": return False
    if encoded["type"] == "str": return bool(encoded["value"])
    if encoded["type"] == "float": return encoded["value"] not in ("nan", "inf", "-inf")
    if encoded["type"] in ("list", "tuple", "mapping"): return bool(encoded["value"])
    return True


def row_key(values, index, counts, *, content=None):
    fields = next((keys for keys in (("filename", "id"), ("filename", "row_id"),
        ("filename_base",), ("source_row", "url_index"), ("url", "url_index"),
        ("url",), ("row_id",), ("id",), ("source_row",), ("filename",), ("page",))
        if all(k in values and identity_present(values[k]) for k in keys)), ())
    if fields:
        identity = [list(fields), [typed_value(values[k]) for k in fields]]
    elif content is not None:
        identity = ["row_content_sha256", hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()]
    else:
        identity = ["index", typed_value(index)]
    base = canonical_json(identity)
    occurrence = counts[base]
    counts[base] += 1
    return canonical_json([json.loads(base), occurrence])


def array_rows(value):
    counts, rows = Counter(), {}
    for index, item in enumerate(value):
        key = row_key(item, index, counts) if isinstance(item, dict) else canonical_json(["index", index])
        rows[key] = {"index": index, "value": item}
    return rows


def read_store(path, sanitize, *, kind=None):
    """Return schema and keyed rows, keeping types without repeating column names."""
    if path.suffix == ".parquet" and "pandas" in sys.modules:
        frame = sys.modules["pandas"].read_parquet(path)
        columns = list(frame.columns)
        schema = {"format": "parquet", "columns": [typed_value(name) for name in columns],
                  "dtypes": [str(dtype) for dtype in frame.dtypes],
                  "identity": "nonnull_named_key_then_content_or_explicit_index; duplicate_occurrence_v1"}
        positional = (frame.index.name is None and frame.index.dtype.kind in ("i", "u")
                      and frame.index.equals(sys.modules["pandas"].RangeIndex(len(frame))))
        schema["index"] = "generated_position_not_row_data" if positional else "explicit_label_per_row"
        counts, prepared = Counter(), []
        for index, values in dataframe_records(frame):
            row = {"values": [compact_value(v, field=k) for k, v in values.items()]}
            if not positional:
                row["index"] = compact_value(index)
            key = row_key(values, index, counts, content=row if positional else None)
            prepared.append((key, row))
        rows, content_counts = {}, Counter()
        for key, row in prepared:
            identity = json.loads(key)[0]
            if identity[0] == "row_content_sha256" or counts[canonical_json(identity)] > 1:
                # A repeated supplied key is ambiguous. Content identity keeps
                # those observations stable under insertion/reordering while
                # still preserving truly identical duplicate occurrences.
                key = row_key({}, None, content_counts, content=row)
            rows[key] = row
        return schema, rows
    if path.suffix == ".json" and not path.name.endswith(".docling.json"):
        value = sanitize(json.loads(path.read_text(encoding="utf-8")))
        if isinstance(value, dict):
            rows, collections = {}, {}
            collection_names = ("pages", "rows", "records", "documents", "results", "fonts")
            for key, item in value.items():
                if isinstance(item, dict) and (key in collection_names or item and all(isinstance(v, dict) for v in item.values())):
                    collections[key] = "object"
                    rows.update({canonical_json([key, child]): row for child, row in item.items()})
                elif isinstance(item, list) and (item or key in collection_names) and all(isinstance(v, dict) for v in item):
                    collections[key] = "array"
                    rows.update({canonical_json([key, json.loads(child)]): row for child, row in array_rows(item).items()})
                else:
                    rows[canonical_json(key)] = item
            schema = {"format": "json_object", "identity": "object_key"}
            if collections:
                schema.update(identity="object_key_or_nested_collection_row", collections=collections)
            return schema, rows
        if isinstance(value, list):
            return {"format": "json_array", "identity": "named_row_key_then_duplicate_occurrence_v1; index_fallback"}, array_rows(value)
        return {"format": "json_scalar", "identity": "root"}, {"root": sanitize(value)}
    if kind == "fatal_skiplist" or path.name == "fatal_skip.txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        ordered = lines == sorted(lines)
        counts, rows = Counter(), {}
        for index, line in enumerate(lines):
            occurrence = counts[line]; counts[line] += 1
            rows[canonical_json([line, occurrence])] = line if ordered else {"index": index, "value": line}
        return {"format": "line_store", "identity": "line_value_and_duplicate_occurrence",
                "order": "sorted" if ordered else "explicit_index"}, rows
    return None


def restore_json_store(schema, rows):
    """Recover diagnostic structure (with body hashes), not deleted text bodies."""
    if schema["format"] == "json_scalar": return rows["root"]
    if schema["format"] == "json_array":
        return [row["value"] for row in sorted(rows.values(), key=lambda r: r["index"])]
    result = {key: {} if kind == "object" else [] for key, kind in schema.get("collections", {}).items()}
    arrays = {}
    for encoded, row in rows.items():
        key = json.loads(encoded)
        if isinstance(key, list):
            parent, child = key
            if schema["collections"][parent] == "object": result[parent][child] = row
            else: arrays.setdefault(parent, []).append(row)
        else: result[key] = row
    for key, values in arrays.items():
        result[key] = [row["value"] for row in sorted(values, key=lambda r: r["index"])]
    return result


def membership_snapshot(writer, scope, values):
    """Delta a sorted set in branch evidence instead of copying it on each load."""
    if not hasattr(writer, "_membership_states"):
        states, by_content = {}, {}
        for digest, encoded in writer.connection.execute("SELECT sha256,json FROM snapshots WHERE kind='membership_delta' ORDER BY rowid"):
            delta = json.loads(encoded); prior = states.get(delta["scope"])
            if delta["previous_sha256"] != (prior["digest"] if prior else None):
                raise ValueError("Broken membership evidence chain")
            members = prior["members"] if prior else set()
            members.difference_update(delta["removed"]); members.update(delta["added"])
            if len(members) != delta["row_count"]: raise ValueError("Membership row count mismatch")
            states[delta["scope"]] = {"digest": digest, "members": members}
            if "members_sha256" in delta: by_content[(delta["scope"], delta["members_sha256"])] = digest
        writer._membership_states = states
        writer._membership_by_content = by_content
    states = writer._membership_states
    previous = states.get(scope)
    # A failed outer observation may have rolled back a cached snapshot. Never
    # use that uncommitted digest as the base of a later committed observation.
    if previous and writer.connection.execute("SELECT 1 FROM snapshots WHERE sha256=?", (previous["digest"],)).fetchone() is None:
        del writer._membership_states
        return membership_snapshot(writer, scope, values)
    members = set(values)
    if previous and members == previous["members"]: return previous["digest"]
    content_hash = hashlib.sha256(canonical_json(sorted(members)).encode("utf-8")).hexdigest()
    reused = writer._membership_by_content.get((scope, content_hash))
    if reused and writer.connection.execute("SELECT 1 FROM snapshots WHERE sha256=?", (reused,)).fetchone():
        # Keep the latest *new* delta as the cache base. Events may reference
        # any earlier committed value without forking the replay chain.
        return reused
    old = previous["members"] if previous else set()
    digest = writer.snapshot("membership_delta", {"kind": "membership_delta", "encoding": "sorted_set_delta_v1",
        "scope": scope, "previous_sha256": previous["digest"] if previous else None,
        "row_count": len(members), "members_sha256": content_hash,
        "added": sorted(members-old), "removed": sorted(old-members)})
    states[scope] = {"digest": digest, "members": members}
    writer._membership_by_content[(scope, content_hash)] = digest
    return digest


def membership_values(connection, digest):
    chain = []
    while digest:
        delta = json.loads(connection.execute("SELECT json FROM snapshots WHERE sha256=? AND kind='membership_delta'", (digest,)).fetchone()[0])
        chain.append(delta); digest = delta["previous_sha256"]
    members = set()
    for delta in reversed(chain):
        members.difference_update(delta["removed"]); members.update(delta["added"])
        if len(members) != delta["row_count"]: raise ValueError("Membership row count mismatch")
    return sorted(members)


def store_states(writer):
    """Rebuild the observation cache after reopen/crash from committed deltas."""
    if not hasattr(writer, "_store_states"):
        states = {}
        for digest, encoded in writer.connection.execute(
                "SELECT sha256,json FROM snapshots WHERE kind='legacy_store' ORDER BY rowid"):
            delta = json.loads(encoded)
            if delta.get("encoding") != "row_delta_v1":
                continue  # Original step-1 history remains untouched.
            uri = delta["uri"]
            previous = states.get(uri)
            if delta["previous_sha256"] != (previous["digest"] if previous else None):
                raise ValueError("Broken legacy-store evidence chain")
            rows = previous["rows"] if previous else {}
            for key in delta["deleted"]:
                rows.pop(key)
            rows.update(delta["changed"])
            if delta["row_count"] is not None and len(rows) != delta["row_count"]:
                raise ValueError("Legacy-store evidence row count mismatch")
            states[uri] = {"digest": digest, "rows": rows, "schema_sha256": delta["schema_sha256"],
                           "sha256": delta["sha256"]}
        writer._store_states = states
    return writer._store_states
