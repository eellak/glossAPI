"""Source URL entry rule: glossAPI downloads public targets only (owner decision 1 Nov 2026, PLAN §9).

A source URL that carries userinfo (``https://user:pass@host/...``) or an obvious credential query or fragment field
(``?token=``, ``&X-Amz-Signature=``, ``#access_token=`` ...) is refused at entry with ``CredentialsUnsupportedError``
(code ``credentials_unsupported``), before any plan is frozen, any ledger row is written or any request is made. So is
a download setting that would send credentials (``domain_cookies``, ``credentials``). Every other URL is a public
target and is recorded as given: in the plan, the ledger, receipts, logs and result tables. No redaction is needed,
so there is none.

The check is a guard against an obvious mistake, not a sanitizer: a secret under an innocuous field name is a public
URL as far as glossAPI can tell, and is recorded as given.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import unquote_plus, urlsplit, urlunsplit

CODE = "credentials_unsupported"
MEANING = ("A source URL carries userinfo (user:pass@) or a credential query/fragment field (token, key, signature, ...), "
           "or a download setting would send credentials (domain_cookies, credentials): glossAPI downloads public "
           "targets only. Refused before anything is planned, recorded or fetched; supply the public URL.")
# Field names (lowercase, '-' and ' ' read as '_') that carry a credential in a URL, and suffixes that do.
CREDENTIAL_FIELDS = frozenset((
    "access_token", "api_key", "apikey", "auth", "auth_token", "authorization", "awsaccesskeyid", "bearer",
    "client_secret", "credential", "credentials", "googleaccessid", "id_token", "jwt", "key", "passwd", "password",
    "private_token", "pwd", "refresh_token", "secret", "session_token", "sig", "signature", "token", "x_amz_credential",
    "x_amz_security_token", "x_amz_signature", "x_api_key", "x_goog_credential", "x_goog_security_token",
    "x_goog_signature"))
CREDENTIAL_SUFFIXES = ("_token", "_secret", "_password", "_passwd", "_signature", "_api_key", "_apikey", "_credential")
# Download settings that only ever sent credentials (removed 1 Nov 2026); naming one is refused, never ignored.
CREDENTIAL_SETTINGS = frozenset(("credentials", "domain_cookies"))


class CredentialsUnsupportedError(ValueError):
    """A credentialed source URL or download setting; ``details`` never holds the secret itself."""
    code = CODE

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details


def credential_field(name: str) -> bool:
    """True for a query/fragment field name that obviously carries a credential."""
    normalized = unquote_plus(str(name)).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in CREDENTIAL_FIELDS or normalized.endswith(CREDENTIAL_SUFFIXES)


def credential_reason(url: Any) -> str | None:
    """Why a URL is credentialed (``'userinfo'`` or ``'field:<name>'``), or None for a public URL."""
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
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


def refuse_credentialed(url: Any) -> None:
    """Raise ``CredentialsUnsupportedError`` for a credentialed source URL; any other value passes."""
    reason = credential_reason(url)
    if reason is not None:
        what = "userinfo (user:pass@)" if reason == "userinfo" else "the credential field " + repr(reason[6:])
        raise CredentialsUnsupportedError(
            f"{CODE}: credentialed downloads are not supported; the source URL {_public_part(url)} carries {what}. "
            "glossAPI downloads public targets only: supply the public URL.",
            url=_public_part(url), reason=reason if reason == "userinfo" else "credential_field", field=reason[6:] or None)


def refuse_credential_settings(*mappings: Any) -> None:
    """Raise ``CredentialsUnsupportedError`` when a download settings mapping holds a credential setting."""
    for mapping in mappings:
        found = sorted(name for name in CREDENTIAL_SETTINGS if isinstance(mapping, dict) and mapping.get(name))
        if found:
            raise CredentialsUnsupportedError(
                f"{CODE}: credentialed downloads are not supported; the download setting {found[0]} would send credentials. "
                "glossAPI downloads public targets only.", setting=found[0], reason="credential_setting")


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
