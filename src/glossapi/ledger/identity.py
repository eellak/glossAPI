"""Versioned logical identities. Filenames and byte equality are never joins."""
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .encoding import snapshot_hash, typed_value
from .writer import LedgerError


def _encoded_source_key(dataset_value, source_value, resource_index):
    if type(resource_index) is not int or resource_index < 0:
        raise LedgerError("A source resource index must be a nonnegative integer")
    return "source:v1:" + snapshot_hash([dataset_value, source_value, resource_index])


def source_key(uri, *, dataset_id=None, source_id=None, resource_index=0):
    if source_id is not None:
        if dataset_id is None:
            raise LedgerError("An explicit source ID requires a dataset namespace")
        return _encoded_source_key(typed_value(dataset_id), typed_value(source_id), resource_index)
    parts = urlsplit(str(uri))
    if parts.scheme.lower() in ("http", "https"):
        # Lowercase the host, not credentials, path or escaping. Preserve ports,
        # queries (including a trailing empty query) and meaningful URL bytes.
        authority = parts.netloc
        user, separator, host = authority.rpartition("@")
        authority = (user + separator if separator else "") + host.lower()
        normalized = urlunsplit((parts.scheme.lower(), authority, parts.path, parts.query, ""))
        if "?" in str(uri).split("#", 1)[0] and not parts.query:
            normalized += "?"
        return "url:v1:" + normalized
    if parts.scheme == "file":
        return "local_path:v1:" + str(uri)
    return "local_path:v1:" + Path(uri).absolute().as_uri()
