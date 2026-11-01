"""Policy routing for downloader selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

from .optional_dependencies import optional_dependencies

with optional_dependencies("download"):
    import yaml

VALID_DOWNLOADERS = {"standard", "browser", "auto"}
ROUTE_OPTION_KEYS = {
    "request_timeout",
    "ssl_verify",
    "ssl_cafile",
    "request_method",
    "sleep",
    "per_domain_concurrency",
    "domain_concurrency_floor",
    "domain_concurrency_ceiling",
    "skip_failed_after",
    "browser_timeout_ms",
    "browser_post_load_wait_ms",
    "browser_engine",
    "browser_headless",
    "browser_session_ttl_seconds",
}


def _normalize_downloader(value: Any, default: str = "standard") -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"default", "http"}:
        normalized = "standard"
    if normalized in {"browser_fallback"}:
        normalized = "auto"
    if normalized in {"browser_protected"}:
        normalized = "browser"
    if normalized not in VALID_DOWNLOADERS:
        raise ValueError(f"Unsupported downloader route: {value}")
    return normalized


@dataclass(frozen=True)
class DownloadPolicyMatch:
    domains: tuple[str, ...] = ()
    url_regex: Optional[re.Pattern[str]] = None

    def matches(self, url: str) -> bool:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if self.domains:
            matched_domain = any(
                hostname == domain or hostname.endswith(f".{domain}")
                for domain in self.domains
            )
            if not matched_domain:
                return False
        if self.url_regex and not self.url_regex.search(url):
            return False
        return True


@dataclass(frozen=True)
class DownloadPolicyRule:
    matcher: DownloadPolicyMatch
    downloader: str
    options: Dict[str, Any]

    def matches(self, url: str) -> bool:
        return self.matcher.matches(url)


@dataclass(frozen=True)
class DownloadPolicy:
    default_downloader: str = "standard"
    default_options: Dict[str, Any] | None = None
    rules: tuple[DownloadPolicyRule, ...] = ()

    def resolve(self, url: str) -> tuple[str, Dict[str, Any]]:
        for rule in self.rules:
            if rule.matches(url):
                return rule.downloader, dict(rule.options)
        return self.default_downloader, dict(self.default_options or {})


def _extract_route_options(data: Dict[str, Any]) -> Dict[str, Any]:
    from .ledger.urls import refuse_credentials
    refuse_credentials(settings=data)  # public targets only: a policy never sends credentials
    return {key: value for key, value in data.items() if key in ROUTE_OPTION_KEYS}


def _build_matcher(raw: Dict[str, Any]) -> DownloadPolicyMatch:
    domains = tuple(str(item).strip().lower() for item in (raw.get("domains") or []) if str(item).strip())
    url_regex = raw.get("url_regex")
    compiled = re.compile(str(url_regex)) if url_regex else None
    return DownloadPolicyMatch(domains=domains, url_regex=compiled)


def build_download_policy(data: Dict[str, Any]) -> DownloadPolicy:
    default_block = dict(data.get("default") or {})
    default_downloader = _normalize_downloader(default_block.get("downloader"), default="standard")
    default_options = _extract_route_options(default_block)

    rules = []
    for raw_rule in data.get("rules") or []:
        raw_rule = dict(raw_rule or {})
        matcher = _build_matcher(dict(raw_rule.get("match") or {}))
        downloader = _normalize_downloader(raw_rule.get("downloader"), default=default_downloader)
        options = _extract_route_options(raw_rule)
        rules.append(DownloadPolicyRule(matcher=matcher, downloader=downloader, options=options))

    return DownloadPolicy(
        default_downloader=default_downloader,
        default_options=default_options,
        rules=tuple(rules),
    )


class PolicyFileError(ValueError):
    """An unreadable download policy file; the message names the file and position only, never its content."""
    code = "invalid_settings"


def read_policy_file(path: str | Path) -> Dict[str, Any]:
    """The decoded policy mapping. A YAML error is re-raised without the parser's source excerpt or exception chain
    (the excerpt could repeat a value; contract v2 item 3)."""
    policy_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None) or getattr(error, "context_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark is not None else ""
        raise PolicyFileError(f"Download policy file is not valid YAML: {policy_path}{where}") from None
    if not isinstance(payload, dict):
        raise PolicyFileError("Download policy file must define a mapping at the top level")
    return payload


def load_download_policy(path: str | Path) -> DownloadPolicy:
    return build_download_policy(read_policy_file(path))


__all__ = [
    "DownloadPolicy",
    "DownloadPolicyMatch",
    "DownloadPolicyRule",
    "VALID_DOWNLOADERS",
    "build_download_policy",
    "load_download_policy",
    "read_policy_file",
]
