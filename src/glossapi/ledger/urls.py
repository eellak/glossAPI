"""Persisted URL form, header redaction and resource identity (CP-LEDGER cycle 2).

The policy is an allowlist, not a list of exceptions. Everything glossAPI
writes (ledger, snapshots, manifests, receipts, logs, reports, errors and
parquet) keeps only the scheme, host, port and path of a URL:

- userinfo is always dropped;
- every query and fragment value is replaced by ``PLACEHOLDER`` unless the
  caller declared that field public (``settings['url_policy']``);
- network-path (``//host/...``) and relative references get the same treatment.

``persisted_url`` is the one function that does this; ``scrub_text`` finds
URLs and references in free text and applies it, and ``scrub`` walks JSON-like
values. Raw URLs stay in process memory for transport only.

A persisted spelling is not an identity: two raw URLs that differ only in a
credential persist identically. ``resource_identity`` therefore keys a lossy
URL by an HMAC of the raw URL under a per-corpus random key kept outside the
ledger (``.glossapi/keys/resource-identity.key``, owner-only), or by a stable
caller-declared ``source_id``. A URL that loses nothing keeps its plain
``url:v1:`` identity, so credential- and query-free corpora are unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import hmac
import logging
import os
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import unquote_plus, urlsplit, urlunsplit

# Percent-encoded marker: valid inside a query or fragment, never a credential.
PLACEHOLDER = "%5Bredacted%5D"
URL_REDACTED = PLACEHOLDER
HEADER_REDACTED = "[redacted]"
DEFAULT_SECRET_HEADERS = frozenset(("authorization", "proxy-authorization", "cookie", "set-cookie"))

_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*")
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s\"'<>\\`]+")
# ``//host...`` not preceded by a scheme colon, a word or another slash.
_NETWORK_PATH = re.compile(r"(?<![\w+.\-:/\\])//(?=[^\s/\"'<>\\`])[^\s\"'<>\\`]+")
# A token with a ``?name=value`` query (relative redirect targets, paths).
_RELATIVE = re.compile(r"(?<![^\s\"'(=,\[{<>])([^\s\"'<>\\`?]*)(\?[A-Za-z0-9_.~%+\-]+=[^\s\"'<>\\`]*)")
# Safety net: a credential-named field anywhere after a query-like separator.
_FIELD = re.compile(r"(?<=[?&;#])([A-Za-z0-9_.~%+\-]+)=([^&;#\s\"'<>\\`]+)")
_PREFILTER = re.compile(r"//|\?[A-Za-z0-9_.~%+\-]|[&;#][A-Za-z0-9_.~%+\-]+=|authorization|cookie", re.IGNORECASE)
_SIGNATURE_FIELDS = frozenset(("sig", "signature", "x_amz_signature", "x_amz_credential", "x_amz_security_token",
    "x_goog_signature", "x_goog_credential", "x_goog_security_token", "awsaccesskeyid", "googleaccessid",
    "client_secret", "jwt", "x_api_key", "private_token", "key_secret", "key"))


# ------------------------------------------------------------------ policy ---

class UrlPolicy:
    """Caller declarations: public query/fragment fields and extra secret headers."""
    __slots__ = ("public_fields", "secret_headers")

    def __init__(self, public_fields=(), secret_headers=()):
        self.public_fields = frozenset(_field_name(name) for name in public_fields)
        self.secret_headers = DEFAULT_SECRET_HEADERS | frozenset(str(name).strip().lower() for name in secret_headers)

    def declared(self):
        return {"public_query_fields": sorted(self.public_fields),
                "secret_headers": sorted(self.secret_headers - DEFAULT_SECRET_HEADERS)}


DEFAULT_POLICY = UrlPolicy()
# A context variable, so concurrent asyncio acquisitions and threads never see
# another run's declarations; a fresh thread falls back to the strict default.
_policy_var = ContextVar("glossapi_url_policy", default=DEFAULT_POLICY)


def _field_name(name):
    return unquote_plus(str(name)).strip().lower()


def parse_policy(value):
    """Validate ``settings['url_policy']``; ``None`` means the default policy."""
    if value is None:
        return DEFAULT_POLICY
    if isinstance(value, UrlPolicy):
        return value
    if not isinstance(value, dict) or set(value) - {"public_query_fields", "secret_headers"}:
        raise ValueError("url_policy accepts only public_query_fields and secret_headers")
    fields, headers = value.get("public_query_fields", ()), value.get("secret_headers", ())
    if isinstance(fields, str) or isinstance(headers, str) or not all(isinstance(item, str) and item for item in [*fields, *headers]):
        raise ValueError("url_policy fields must be lists of non-empty names")
    return UrlPolicy(fields, headers)


def current_policy():
    return _policy_var.get()


@contextmanager
def url_policy(value=None):
    """Make a declared policy active in this context (orchestrator or worker)."""
    token = _policy_var.set(parse_policy(value))
    try:
        yield _policy_var.get()
    finally:
        _policy_var.reset(token)


# --------------------------------------------------------- persisted form ---

def _fields(text, policy):
    """Placeholder every non-empty value of a query or fragment component list."""
    parts = []
    for separator, part in re.findall(r"(^|[&;])([^&;]*)", text):
        name, equals, value = part.partition("=")
        if equals:
            if value and value != PLACEHOLDER and _field_name(name) not in policy.public_fields:
                part = name + "=" + PLACEHOLDER
        elif part and part != PLACEHOLDER and _field_name(part) not in policy.public_fields:
            part = PLACEHOLDER  # a bare component is a value (it can be a token)
        parts.append(separator + part)
    return "".join(parts)


def _tail(tail, policy):
    path, hash_mark, fragment = tail.partition("#")
    path, question, query = path.partition("?")
    if query:
        query = _fields(query, policy)
    if fragment:
        fragment = _fields(fragment, policy)
    return path + question + query + hash_mark + fragment


def _authority_and_tail(rest):
    end = len(rest)
    for character in "/?#":
        position = rest.find(character)
        if position != -1:
            end = min(end, position)
    authority, tail = rest[:end], rest[end:]
    if "@" in authority:
        authority = authority.rpartition("@")[2]  # userinfo is always dropped
    return authority, tail


def persisted_url(url, policy=None):
    """The one persisted spelling of a URL or URL reference (allowlist).

    Keeps scheme, host, port and path; drops userinfo; replaces every query and
    fragment value by ``PLACEHOLDER`` unless the field is declared public.
    Absolute, network-path (``//host``) and relative references are handled.
    Idempotent. A string without userinfo, query or fragment is returned as is.
    """
    if not isinstance(url, str) or not url or ("@" not in url and "?" not in url and "#" not in url):
        return url
    policy = policy or _policy_var.get()
    scheme, colon, rest = url.partition(":")
    if colon and _SCHEME.fullmatch(scheme):
        if rest.startswith("//"):
            authority, tail = _authority_and_tail(rest[2:])
            result = scheme + "://" + authority + _tail(tail, policy)
        else:
            result = scheme + ":" + _tail(rest, policy)  # opaque URI (magnet:?xt=...)
    elif url.startswith("//"):
        authority, tail = _authority_and_tail(url[2:])
        result = "//" + authority + _tail(tail, policy)
    else:
        result = _tail(url, policy)
    return url if result == url else result


# Backwards-compatible name used by earlier boundaries and tests.
sanitize_url = persisted_url


def has_credentials(url):
    """True when the persisted spelling loses information (userinfo/query/fragment)."""
    return isinstance(url, str) and persisted_url(url) != url


def was_redacted(value):
    """True when a recorded URL string lost values to the persisted form."""
    return isinstance(value, str) and PLACEHOLDER in value and ("://" in value or value.startswith("//"))


def credential_field(name):
    from .journal import credential_key
    normalized = _field_name(name).replace("-", "_").replace(" ", "_")
    return credential_key(normalized) or normalized in _SIGNATURE_FIELDS


# --------------------------------------------------------------- headers ---

def redact_headers(headers, policy=None):
    """A plain dict copy of request/response headers with secret values replaced.

    Structural: applied before any formatting. Authorization,
    Proxy-Authorization, Cookie, Set-Cookie and caller-declared secret headers.
    """
    policy = policy or _policy_var.get()
    if headers is None:
        return {}
    try:
        items = list(headers.items())
    except AttributeError:
        items = list(headers)
    result = {}
    for key, value in items:
        name = str(key)
        if name.strip().lower() in policy.secret_headers or credential_field(name):
            value = HEADER_REDACTED
        else:
            value = scrub_text(str(value))
        if name in result:  # repeated header (multidict): keep every value
            result[name] = result[name] + ", " + value
        else:
            result[name] = value
    return result


def describe_request(info, policy=None):
    """Formatting-safe description of an aiohttp RequestInfo-like object."""
    if info is None:
        return "No request info available"
    method = getattr(info, "method", "?")
    url = persisted_url(str(getattr(info, "url", "")), policy)
    real = persisted_url(str(getattr(info, "real_url", url)), policy)
    return f"RequestInfo(method={method!r}, url={url!r}, real_url={real!r}, headers={redact_headers(getattr(info, 'headers', None), policy)!r})"


def _header_pattern(policy):
    names = sorted({re.escape(name) for name in policy.secret_headers}, key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(names) + r")([\"']?)\s*[:=]\s*", re.IGNORECASE)


_header_patterns = {}


def _redact_header_text(text, policy):
    pattern = _header_patterns.get(policy.secret_headers)
    if pattern is None:
        pattern = _header_patterns[policy.secret_headers] = _header_pattern(policy)
    if not pattern.search(text):
        return text
    out, position = [], 0
    for match in pattern.finditer(text):
        if match.start() < position:
            continue
        start = match.end()
        if start < len(text) and text[start] in "\"'":
            quote = text[start]
            end = text.find(quote, start + 1)
            end = len(text) if end == -1 else end
            out.append(text[position:start + 1])
        else:
            end = len(text)
            for stop in "\r\n":
                found = text.find(stop, start)
                if found != -1:
                    end = min(end, found)
            out.append(text[position:start])
        value = text[start + 1:end] if start < len(text) and text[start] in "\"'" else text[start:end]
        out.append(HEADER_REDACTED if value else value)
        position = end
    out.append(text[position:])
    return "".join(out)


# ------------------------------------------------------------- free text ---

def _field(match):
    if match.group(2) != PLACEHOLDER and credential_field(match.group(1)):
        return match.group(1) + "=" + PLACEHOLDER
    return match.group(0)


def scrub_text(text, policy=None):
    """Apply the persisted URL form and header redaction to free text.

    Absolute URLs, network-path references and ``?name=value`` references are
    rewritten by ``persisted_url``; secret header values are replaced.
    """
    if not isinstance(text, str) or not _PREFILTER.search(text):
        return text
    policy = policy or _policy_var.get()
    result = text
    if "://" in result:
        result = _URL.sub(lambda match: persisted_url(match.group(0), policy), result)
    if "//" in result:
        result = _NETWORK_PATH.sub(lambda match: persisted_url(match.group(0), policy), result)
    if "?" in result:
        result = _RELATIVE.sub(lambda match: persisted_url(match.group(1) + match.group(2), policy), result)
    if "=" in result:
        result = _FIELD.sub(_field, result)
    result = _redact_header_text(result, policy)
    return text if result == text else result


def scrub(value, policy=None):
    """Recursively apply ``scrub_text`` to every string of a JSON-like value."""
    policy = policy or _policy_var.get()
    if isinstance(value, str):
        return scrub_text(value, policy)
    if isinstance(value, dict):
        return {scrub_text(key, policy) if isinstance(key, str) else key: scrub(item, policy) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, policy) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub(item, policy) for item in value)
    return value


def needs_scrub(encoded):
    """Cheap check over an encoded JSON/text value before a structural scrub."""
    return isinstance(encoded, str) and scrub_text(encoded) != encoded


# --------------------------------------------------------------- logging ---

class CredentialLogFilter(logging.Filter):
    """Applies the URL and header policy to the fully formatted record.

    The message, cached ``exc_text`` (formatted from ``exc_info`` here, so no
    handler formats the raw traceback later) and ``stack_info`` are scrubbed.
    """

    def filter(self, record):
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        cleaned = scrub_text(message)
        if cleaned != message or record.args:
            record.msg, record.args = cleaned, ()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_info = None
        if record.exc_text:
            record.exc_text = scrub_text(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub_text(record.stack_info)
        return True


def protect_handler(handler):
    """Install the one credential filter on a glossAPI handler (idempotent)."""
    if not any(isinstance(item, CredentialLogFilter) for item in handler.filters):
        handler.addFilter(CredentialLogFilter())
    return handler


def protect_logger(logger):
    if not any(isinstance(item, CredentialLogFilter) for item in logger.filters):
        logger.addFilter(CredentialLogFilter())
    for handler in logger.handlers:
        protect_handler(handler)
    return logger


# -------------------------------------------------------------- identity ---

KEY_RELATIVE_PATH = Path(".glossapi") / "keys" / "resource-identity.key"
_active_roots = []
_roots_lock = threading.Lock()


def push_identity_root(root):
    with _roots_lock:
        _active_roots.append(Path(root))


def pop_identity_root(root):
    with _roots_lock:
        for index in range(len(_active_roots) - 1, -1, -1):
            if _active_roots[index] == Path(root):
                del _active_roots[index]
                return


def active_identity_root():
    with _roots_lock:
        return _active_roots[-1] if _active_roots else None


def identity_key_path(root):
    return Path(root) / KEY_RELATIVE_PATH


def identity_key(root, *, create=True):
    """The per-corpus HMAC key (32 random bytes), kept outside the ledger.

    Created on first need with owner-only permissions (directory 0700, file
    0600). An existing key readable by group or others is refused.
    """
    from .writer import LedgerError
    path = identity_key_path(root)
    if not path.exists():
        if not create:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(secrets.token_hex(32))
                stream.flush()
                os.fsync(stream.fileno())
    mode = path.stat().st_mode
    if mode & 0o077:
        raise LedgerError("Resource identity key must be readable by its owner only: " + str(path))
    value = path.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise LedgerError("Resource identity key is malformed: " + str(path))
    return bytes.fromhex(value)


def _identity_parts(url):
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port is not None:
        host = (f"[{host}]" if ":" in host else host) + ":" + str(parts.port)
    elif parts.netloc.rpartition("@")[2].endswith(":"):
        host += ":"
    return parts, host


def lossless_url(url, policy=None):
    """True when the persisted form of ``url`` (ignoring the fragment) loses nothing."""
    without_fragment = url.split("#", 1)[0]
    return persisted_url(without_fragment, policy) == without_fragment


def keyed_reference(url, key):
    """HMAC-SHA256 of the raw URL (scheme and host lowercased, fragment removed)."""
    parts = urlsplit(url)
    userinfo, separator, host = parts.netloc.rpartition("@")
    canonical = urlunsplit((parts.scheme.lower(), userinfo + separator + host.lower(), parts.path, parts.query, ""))
    if "?" in url.split("#", 1)[0] and not parts.query:
        canonical += "?"
    return hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def keyed_source_key(url, *, root=None, create=True, policy=None):
    """``url:v2:<keyed reference>:<persisted spelling>`` for a lossy URL."""
    from .writer import LedgerError
    root = root or active_identity_root()
    if root is None:
        raise LedgerError("A URL with userinfo, a query or a fragment needs the corpus resource identity key")
    key = identity_key(root, create=create)
    if key is None:
        return None
    parts = urlsplit(url)
    host = parts.netloc.rpartition("@")[2].lower()
    display = urlunsplit((parts.scheme.lower(), host, parts.path, parts.query, ""))
    if "?" in url.split("#", 1)[0] and not parts.query:
        display += "?"
    return "url:v2:" + keyed_reference(url, key) + ":" + persisted_url(display, policy)


def source_uri_of_key(key):
    """The persisted source URI a source key spells, or None (``source:v1``)."""
    if not isinstance(key, str):
        return None
    for prefix in ("url:v1:", "local_path:v1:"):
        if key.startswith(prefix):
            return key[len(prefix):]
    if key.startswith("url:v2:"):
        parts = key.split(":", 3)
        return parts[3] if len(parts) == 4 else None
    return None


def is_keyed_key(key):
    return isinstance(key, str) and (key.startswith("url:v2:") or key.startswith("source:v1:"))
