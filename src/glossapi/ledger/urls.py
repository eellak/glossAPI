"""Credentials unsupported: glossAPI downloads public targets only (PLAN §2.1, credentials contract v2).

``refuse_credentials`` is the one check. It refuses, with ``CredentialsUnsupportedError`` (code
``credentials_unsupported``), a URL that carries userinfo (``https://user:pass@host/...``, also schemeless
``user:pass@host/...`` as the downloader would request it) or an obvious credential query/fragment field (``?token=``,
``&X-Amz-Signature=``, ``#accessToken=`` ...), and download settings that would send credentials (``domain_cookies``,
``credentials``, cookie/header/auth/token-named keys, the same inside a download policy: its defaults and rules, as a
dict, a policy object or a policy file).

Hard guarantees (contract v2 items 1, 3, 5): every request boundary (``GlossDownloader._refuse_credentials``: the
per-file kernel and the batch expansion every direct, sync, async, browser, legacy and streaming download passes) applies
it before a request is built; errors name only scheme, host, path and the field or setting name, never a value; nothing
here imports policy or YAML code unless a policy is actually given. The main entries (run/plan, CLI, manifests,
scheduler freeze/reload, the legacy download wrapper) apply it early, best effort; what a caller supplies may be recorded
as given before a refusal (item 4: no confidentiality of records, no redaction). A secret under an innocuous name
(``?x=SECRET``) is a public URL as far as glossAPI can tell.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterator
from urllib.parse import unquote_plus, urlsplit, urlunsplit

CODE = "credentials_unsupported"
MEANING = ("A source URL carries userinfo (user:pass@) or a credential query/fragment field (token, key, signature, ...), "
           "or a download setting would send credentials (domain_cookies, credentials): glossAPI downloads public "
           "targets only. Refused before anything is planned, recorded or fetched; supply the public URL.")
# Credential names after normalisation (case-insensitive, camelCase/kebab/dot read as snake_case); each also matches
# without underscores (``accesstoken``), and any name ending in a suffix below matches too.
CREDENTIAL_FIELDS = frozenset((
    "access_key", "access_token", "api_key", "auth", "authorization", "aws_access_key_id", "awsaccesskeyid", "bearer",
    "client_secret", "cookie", "cookies", "credential", "credentials", "googleaccessid", "id_token", "jwt", "key",
    "passwd", "password", "private_key", "pwd", "refresh_token", "secret", "secret_key", "session_id",
    "session_token", "sig", "signature", "token", "x_amz_credential", "x_amz_security_token",
    "x_amz_signature", "x_goog_credential", "x_goog_security_token", "x_goog_signature"))
CREDENTIAL_SUFFIXES = ("_token", "_secret", "_password", "_passwd", "_signature", "_credential", "_credentials",
                       "_cookie", "_cookies", "_api_key", "_access_key", "_secret_key", "_private_key", "_session_id")
_COMPACT_FIELDS = frozenset(name.replace("_", "") for name in CREDENTIAL_FIELDS)
_COMPACT_SUFFIXES = tuple(suffix.replace("_", "") for suffix in CREDENTIAL_SUFFIXES)
# Download settings that only ever sent credentials (removed 1 Nov 2026), besides credential-named keys.
CREDENTIAL_SETTINGS = frozenset(("credentials", "domain_cookies", "cookies", "headers", "basic_auth", "proxy_auth"))
_KEY_PREFIXES = ("url:v1:", "local_path:v1:")
_EMBEDDED = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s;,'\"<>\\]+")
_SELECTOR_KEYS = ("url", "href", "link", "source_uri", "source_key", "path", "document_id")


class CredentialsUnsupportedError(ValueError):
    """A credentialed source URL or download setting; ``details`` never holds the secret itself."""
    code = CODE

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details


def _normal_name(name: Any) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", unquote_plus(str(name)).strip())
    return re.sub(r"[-. ]+", "_", text.lower())


def credential_field(name: Any) -> bool:
    """True for a query/fragment field or setting name that obviously carries a credential."""
    normalized = _normal_name(name)
    compact = normalized.replace("_", "")
    return (normalized in CREDENTIAL_FIELDS or compact in _COMPACT_FIELDS or normalized.endswith(CREDENTIAL_SUFFIXES)
            or compact.endswith(_COMPACT_SUFFIXES))


def source_links(value: Any) -> list[str]:
    """The source URLs one URL cell names, in the downloader's execution grammar (lists/tuples of URLs or
    ``{url|href|link}`` dicts, a dict, a JSON or Python-literal list/object string, ``;``/newline-separated or several
    embedded URLs, else the string itself). ``GlossDownloader._expand_and_mark_duplicates`` expands cells with it."""
    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if item is None or (isinstance(item, float) and item != item):
                continue
            if isinstance(item, dict):
                link = item.get('url') or item.get('href') or item.get('link')
                if link:
                    out.append(str(link))
            else:
                out.append(str(item))
        return out
    if isinstance(value, dict):
        link = value.get('url') or value.get('href') or value.get('link')
        return [str(link)] if link else []
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith('[') and text.endswith(']')) or (text.startswith('{') and text.endswith('}')):
            for decode in (json.loads, ast.literal_eval):
                try:
                    return source_links(decode(text))
                except Exception:
                    continue
        if any(separator in text for separator in ('\n', ';')) and 'http' in text:
            parts = [part.strip() for part in re.split(r'[;\n]+', text) if part.strip()]
            if parts:
                return parts
        embedded = re.findall(r'https?://\S+', text)
        if len(embedded) > 1:
            cleaned = [link.rstrip('.,)') for link in embedded if link]
            if cleaned:
                return cleaned
        return [text]
    try:
        return [str(value)]
    except Exception:
        return []


def _candidates(value: Any) -> Iterator[str]:
    """Every URL spelling a cell, row or selector can reach the downloader or a plan with: the execution grammar,
    every string leaf (numpy arrays included), known key prefixes, embedded URLs, scheme case and whitespace."""
    if isinstance(value, str):
        text = value.strip()
        yield text
        if "://" not in text and not text.startswith(("/", ".", "~")) and ("@" in text.split("/", 1)[0] or "?" in text):
            yield "https://" + text  # the downloader requests a schemeless cell as https://
        for prefix in _KEY_PREFIXES:
            if text.startswith(prefix):
                yield text[len(prefix):]
        yield from _EMBEDDED.findall(text)
        for link in source_links(text):
            if link != text:
                yield from _candidates(link)
    elif isinstance(value, dict):
        for key in _SELECTOR_KEYS:
            if key in value:
                yield from _candidates(value[key])
    elif isinstance(value, (list, tuple)) or hasattr(value, 'tolist'):
        items = value.tolist() if hasattr(value, 'tolist') else value
        for item in items if isinstance(items, (list, tuple)) else [items]:
            yield from _candidates(item)


def credential_reason(url: Any) -> str | None:
    """Why a URL is credentialed (``'userinfo'`` or ``'field:<name>'``), or None for a public URL or a path."""
    if not isinstance(url, str) or "://" not in url:
        return None
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if "@" in parts.netloc:
        return "userinfo"
    for component in (parts.query, parts.fragment):
        for field in component.replace(";", "&").split("&"):
            name, equals, value = field.partition("=")
            if equals and value and credential_field(name):
                return "field:" + unquote_plus(name).strip()
    return None


def _public_part(url: str) -> str:
    """Scheme, host and path only: what an error may name without repeating the secret."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return "(unparseable URL)"
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


def _policy_blocks(policy: Any) -> Iterator[Any]:
    """The route-option mappings of a download policy (a dict or a DownloadPolicy-like object); imports nothing, so an
    absent policy never needs the optional download extra (contract v2 item 5)."""
    if hasattr(policy, "default_options") and hasattr(policy, "rules"):
        yield policy.default_options or {}
        yield from (getattr(rule, "options", None) or {} for rule in policy.rules)
    elif isinstance(policy, dict):
        yield policy.get("default") or {}
        yield from (rule for rule in policy.get("rules") or [] if isinstance(rule, dict))


def _settings_violation(settings: Any) -> str | None:
    if not isinstance(settings, dict):
        return None
    for key, value in settings.items():
        empty = value is None or (isinstance(value, (str, dict, list, tuple)) and not value)
        if not empty and (_normal_name(key) in CREDENTIAL_SETTINGS or credential_field(key)):
            return str(key)
    for block in _policy_blocks(settings.get("download_policy")):
        found = _settings_violation(block)
        if found:
            return "download_policy." + found
    path = settings.get("download_policy_file")
    if path:
        from ..download_policy import read_policy_file  # only when a policy file is actually given
        for block in _policy_blocks(read_policy_file(path)):
            found = _settings_violation(block)
            if found:
                return "download_policy_file." + found
    return None


def refuse_credentials(*, urls: Any = (), settings: Any = ()) -> None:
    """THE entry rule: raise ``CredentialsUnsupportedError`` when any URL value (a URL, selector, cell or manifest row;
    ``urls`` is an iterable of them) is credentialed, or any download settings mapping in ``settings`` (one mapping or
    an iterable of mappings) holds a credential setting, its download policy included."""
    for value in ([urls] if isinstance(urls, (str, dict)) else urls):
        for candidate in _candidates(value):
            reason = credential_reason(candidate)
            if reason is not None:
                what = "userinfo (user:pass@)" if reason == "userinfo" else "the credential field " + repr(reason[6:])
                raise CredentialsUnsupportedError(
                    f"{CODE}: credentialed downloads are not supported; the source URL {_public_part(candidate)} carries "
                    f"{what}. glossAPI downloads public targets only: supply the public URL.", url=_public_part(candidate),
                    reason=reason if reason == "userinfo" else "credential_field", field=reason[6:] or None)
    for mapping in ([settings] if isinstance(settings, dict) else settings):
        found = _settings_violation(mapping)
        if found:
            raise CredentialsUnsupportedError(
                f"{CODE}: credentialed downloads are not supported; the download setting {found} would send credentials. "
                "glossAPI downloads public targets only.", setting=found, reason="credential_setting")


def without_userinfo(url: Any) -> Any:
    """``url`` with any userinfo dropped (an installed package's VCS URL can carry an access token)."""
    if not isinstance(url, str) or "@" not in url or "://" not in url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if "@" not in parts.netloc:
        return url
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, parts.query, parts.fragment))
