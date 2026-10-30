"""Credential-free URL spellings for every recorded boundary (CP-LEDGER F2).

A source URL can carry transport credentials: HTTP userinfo
(``https://user:secret@host/...``) or credential query/fragment fields
(``?access_token=...``, signed-URL signatures). Those values are kept in memory
for the request only. Everything that is recorded (source records, manifests,
snapshots, plans, specifications, evidence, errors, logs and reports) uses the
sanitized spelling, which is also the stable non-secret source identity.

Sanitization is the identity on URLs without credentials, so credential-free
corpora record exactly the same bytes as before.
"""
from __future__ import annotations

from functools import lru_cache
import re
from urllib.parse import unquote_plus

# Percent-encoded marker: valid inside userinfo, query and fragment, and never
# a value a server would accept as a credential.
URL_REDACTED = "%5Bredacted%5D"
# Signed-URL and SAS fields that carry secrets but are not ``*_token``/``*_key`` names.
_SIGNATURE_FIELDS = frozenset(("sig", "signature", "x_amz_signature", "x_amz_credential", "x_amz_security_token",
    "x_goog_signature", "x_goog_credential", "x_goog_security_token", "awsaccesskeyid", "googleaccessid",
    "client_secret", "jwt", "x_api_key", "private_token", "key_secret"))
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s\"'<>\\`]+")
# A credential field of a relative reference (for example a redirect
# ``Location: /file.pdf?sig=...``) or of any query-like text.
_FIELD = re.compile(r"(?<=[?&;#])([A-Za-z0-9_.~%+\-]+)=([^&;#\s\"'<>\\`]+)")


@lru_cache(maxsize=4096)
def credential_field(name):
    from .journal import credential_key
    normalized = unquote_plus(str(name)).strip().lower().replace("-", "_").replace(" ", "_")
    return credential_key(normalized) or normalized in _SIGNATURE_FIELDS


def _fields(text):
    parts = []
    changed = False
    for separator, part in re.findall(r"(^|[&;])([^&;]*)", text):
        name, equals, value = part.partition("=")
        if equals and value and value != URL_REDACTED and credential_field(name):
            part = name + "=" + URL_REDACTED
            changed = True
        parts.append(separator + part)
    return "".join(parts), changed


def sanitize_url(url):
    """Return ``url`` with userinfo and credential query/fragment values redacted.

    Non-URL strings and credential-free URLs are returned unchanged (the same
    object), so this is safe to apply to any recorded string.
    """
    if not isinstance(url, str) or "://" not in url or ("@" not in url and "?" not in url and "#" not in url):
        return url
    scheme, separator, rest = url.partition("://")
    if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9+.\-]*", scheme):
        return url
    end = len(rest)
    for character in "/?#":
        position = rest.find(character)
        if position != -1:
            end = min(end, position)
    authority, tail = rest[:end], rest[end:]
    changed = False
    if "@" in authority:
        userinfo, _, host = authority.rpartition("@")
        if userinfo != URL_REDACTED:
            authority = URL_REDACTED + "@" + host
            changed = True
    path, hash_mark, fragment = tail.partition("#")
    path, question, query = path.partition("?")
    if query:
        query, query_changed = _fields(query)
        changed = changed or query_changed
    if fragment:
        fragment, fragment_changed = _fields(fragment)
        changed = changed or fragment_changed
    if not changed:
        return url
    return scheme + "://" + authority + path + question + query + hash_mark + fragment


def has_credentials(url):
    return isinstance(url, str) and sanitize_url(url) != url


def was_redacted(value):
    """True when a recorded URL string lost credentials to sanitization."""
    return isinstance(value, str) and URL_REDACTED in value and "://" in value


def _field(match):
    if match.group(2) != URL_REDACTED and credential_field(match.group(1)):
        return match.group(1) + "=" + URL_REDACTED
    return match.group(0)


def scrub_text(text):
    """Sanitize every URL, and every credential field of a relative reference,
    embedded in free text (messages, tracebacks, logs, redirect locations)."""
    if not isinstance(text, str):
        return text
    if "://" in text:
        text = _URL.sub(lambda match: sanitize_url(match.group(0)), text)
    if "=" in text:
        text = _FIELD.sub(_field, text)
    return text


def scrub(value):
    """Recursively sanitize URL strings in a JSON-like value."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {scrub_text(key) if isinstance(key, str) else key: scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub(item) for item in value)
    return value


def needs_scrub(encoded):
    """Cheap check over an encoded JSON/text value before a structural scrub."""
    if "=" not in encoded and "@" not in encoded:
        return False
    if "://" in encoded and any(sanitize_url(match.group(0)) != match.group(0) for match in _URL.finditer(encoded)):
        return True
    return "=" in encoded and any(match.group(2) != URL_REDACTED and credential_field(match.group(1))
                                  for match in _FIELD.finditer(encoded))
