"""Browser-capable downloader mode for browser-gated file endpoints."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any, Dict, Optional, Tuple

import aiofiles
import aiohttp

from .download_policy import DownloadPolicy, load_download_policy
from .gloss_downloader import GlossDownloader


@dataclass
class BrowserSessionState:
    user_agent: str
    cookie_header: str
    cached_at: float


class BrowserGlossDownloader(GlossDownloader):
    """
    Downloader variant that retries browser-gated file endpoints via Playwright.

    This mode only targets file endpoints that are protected by browser/session
    checks. It intentionally does not attempt viewer-style extraction.
    """

    def __init__(
        self,
        *args,
        browser_timeout_ms: int = 60000,
        browser_post_load_wait_ms: int = 3000,
        browser_engine: str = "chromium",
        browser_headless: bool = True,
        browser_session_ttl_seconds: int = 900,
        browser_max_parallel_bootstraps: int = 2,
        default_download_route: str = "auto",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.browser_timeout_ms = int(browser_timeout_ms)
        self.browser_post_load_wait_ms = int(browser_post_load_wait_ms)
        self.browser_engine = str(browser_engine or "chromium")
        self.browser_headless = bool(browser_headless)
        self.browser_session_ttl_seconds = int(browser_session_ttl_seconds)
        self.browser_max_parallel_bootstraps = max(1, int(browser_max_parallel_bootstraps))
        self.browser_bootstrap_semaphore = asyncio.Semaphore(self.browser_max_parallel_bootstraps)
        self._browser_session_cache: Dict[str, BrowserSessionState] = {}
        self._browser_session_locks: Dict[str, asyncio.Lock] = {}
        self.default_download_route = str(default_download_route or "auto").strip().lower()
        self.policy = self._load_policy()

    def _load_policy(self) -> Optional[DownloadPolicy]:
        if self.download_policy is not None:
            return self.download_policy
        if self.download_policy_file:
            return load_download_policy(self.download_policy_file)
        return None

    def _resolve_route(self, url: str) -> tuple[str, Dict[str, Any]]:
        if self.policy is not None:
            return self.policy.resolve(url)
        return self.default_download_route, {}

    def _route_setting(self, route_options: Dict[str, Any], name: str, fallback: Any) -> Any:
        return route_options.get(name, fallback)

    def _domain_key(self, url: str) -> str:
        return self._extract_base_domain(url) or (urlparse(url).hostname or "").lower()

    def _choose_browser_bootstrap_url(self, url: str) -> str:
        if self._url_looks_like_file_endpoint(url):
            return self.get_base_url(url)
        return url

    def _should_ignore_navigation_exception(self, url: str, exc: Exception) -> bool:
        message = str(exc)
        if self._url_looks_like_file_endpoint(url) and "net::ERR_ABORTED" in message:
            return True
        return False

    def _session_lock_for_domain(self, domain_key: str) -> asyncio.Lock:
        lock = self._browser_session_locks.get(domain_key)
        if lock is None:
            lock = asyncio.Lock()
            self._browser_session_locks[domain_key] = lock
        return lock

    def _is_browser_session_fresh(self, state: BrowserSessionState, route_options: Dict[str, Any]) -> bool:
        ttl = int(self._route_setting(route_options, "browser_session_ttl_seconds", self.browser_session_ttl_seconds))
        if ttl <= 0:
            return False
        return (time.time() - state.cached_at) < ttl

    def _should_attempt_browser_recovery(self, url: str, html_issue: str) -> bool:
        issue = str(html_issue or "").lower()
        if "document viewer returned" in issue:
            return False
        if "challenge page returned" in issue:
            return True
        if "cookie bootstrap is required" in issue:
            return True
        if "expected a file-like response but received html instead" in issue:
            return self._url_looks_like_file_endpoint(url)
        return False

    def _build_ssl_connector(self) -> Optional[aiohttp.TCPConnector]:
        connector = None
        if not self.ssl_verify:
            connector = aiohttp.TCPConnector(ssl=False)
        elif self.ssl_cafile:
            import ssl as _ssl

            ctx = _ssl.create_default_context(cafile=self.ssl_cafile)
            connector = aiohttp.TCPConnector(ssl=ctx)
        return connector

    def _domain_cookies_for_url(self, url: str) -> Dict[str, str]:
        cookies: Dict[str, str] = {}
        for domain_pattern, domain_cookies in self.domain_cookies.items():
            if domain_pattern in url:
                cookies.update(domain_cookies)
        return cookies

    async def _write_recovered_file(self, row_index: int, filename: str, body: bytes) -> None:
        tmp_path = self.downloads_dir / f".part_browser_{row_index}"
        async with aiofiles.open(tmp_path, "wb") as handle:
            await handle.write(body)
        final_path = self.downloads_dir / filename
        os.replace(tmp_path, final_path)

    async def _fetch_with_browser_session_state(
        self,
        *,
        url: str,
        referer: Optional[str],
        state: BrowserSessionState,
    ) -> Tuple[bytes, Dict[str, str], Dict[str, Any]]:
        request_headers = {
            "User-Agent": state.user_agent,
            "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
        }
        if state.cookie_header:
            request_headers["Cookie"] = state.cookie_header
        if referer:
            request_headers["Referer"] = referer

        connector = self._build_ssl_connector()
        timeout = aiohttp.ClientTimeout(total=min(max(self.request_timeout, 30), 180))
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, headers=request_headers, timeout=timeout) as response:
                response.raise_for_status()
                body = await response.read()
                response_headers = {str(k): str(v) for k, v in (response.headers or {}).items()}
        return body, response_headers, {"candidate_url": url, "session_reused": True}

    async def _bootstrap_browser_session_state(
        self,
        *,
        url: str,
        referer: Optional[str],
        route_options: Dict[str, Any],
    ) -> tuple[BrowserSessionState, list[tuple[str, Dict[str, str], str]]]:
        timeout_ms = int(self._route_setting(route_options, "browser_timeout_ms", self.browser_timeout_ms))
        post_load_wait_ms = int(
            self._route_setting(route_options, "browser_post_load_wait_ms", self.browser_post_load_wait_ms)
        )
        browser_engine = str(self._route_setting(route_options, "browser_engine", self.browser_engine))
        browser_headless = bool(self._route_setting(route_options, "browser_headless", self.browser_headless))

        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise RuntimeError(
                "Browser download mode requires the optional 'browser' dependencies "
                "(install Playwright and browser binaries)"
            ) from exc

        accepted_responses: list[tuple[str, Dict[str, str], str]] = []
        bootstrap_url = self._choose_browser_bootstrap_url(url)

        async with self.browser_bootstrap_semaphore:
            async with async_playwright() as playwright:
                browser_type = getattr(playwright, browser_engine, None)
                if browser_type is None:
                    raise RuntimeError(f"Unsupported browser engine: {browser_engine}")

                browser = await browser_type.launch(headless=browser_headless)
                context = await browser.new_context(ignore_https_errors=not self.ssl_verify)
                parsed = urlparse(url)
                browser_cookies = [
                    {
                        "name": key,
                        "value": str(value),
                        "domain": parsed.hostname or "",
                        "path": "/",
                    }
                    for key, value in self._domain_cookies_for_url(url).items()
                ]
                if browser_cookies:
                    await context.add_cookies(browser_cookies)
                page = await context.new_page()
                if referer:
                    await page.set_extra_http_headers({"Referer": referer})

                async def _route_filter(route: Any) -> None:
                    req = route.request
                    if req.resource_type in {"image", "media", "font"}:
                        await route.abort()
                        return
                    req_url = str(req.url or "")
                    if "googletagmanager" in req_url or "google-analytics.com" in req_url:
                        await route.abort()
                        return
                    await route.continue_()

                await page.route("**/*", _route_filter)

                def _record_response(response: Any) -> None:
                    try:
                        response_headers = {str(k): str(v) for k, v in (response.headers or {}).items()}
                        file_ext = self.infer_file_extension(response.url, response_headers, b"")
                        if file_ext and file_ext != "html" and self.is_supported_format(file_ext):
                            accepted_responses.append((response.url, response_headers, file_ext))
                    except Exception:
                        return

                page.on("response", _record_response)

                try:
                    main_response = None
                    try:
                        main_response = await page.goto(bootstrap_url, wait_until="networkidle", timeout=timeout_ms)
                    except Exception as exc:
                        if not self._should_ignore_navigation_exception(bootstrap_url, exc):
                            raise
                    if main_response is not None:
                        main_headers = {str(k): str(v) for k, v in (main_response.headers or {}).items()}
                        main_ext = self.infer_file_extension(main_response.url, main_headers, b"")
                        if main_ext and main_ext != "html" and self.is_supported_format(main_ext):
                            accepted_responses.insert(0, (main_response.url, main_headers, main_ext))
                    if not accepted_responses and post_load_wait_ms > 0:
                        await page.wait_for_timeout(post_load_wait_ms)

                    browser_user_agent = await page.evaluate("() => navigator.userAgent")
                    browser_cookies = await context.cookies()
                finally:
                    await browser.close()

        cookie_header = "; ".join(
            f"{cookie['name']}={cookie['value']}" for cookie in browser_cookies if cookie.get("name")
        )
        return BrowserSessionState(
            user_agent=browser_user_agent,
            cookie_header=cookie_header,
            cached_at=time.time(),
        ), accepted_responses

    async def _download_via_browser_session(
        self,
        *,
        url: str,
        referer: Optional[str],
        route_options: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Tuple[bytes, Dict[str, str], Dict[str, Any]]:
        options = dict(route_options or {})
        domain_key = self._domain_key(url)
        state = self._browser_session_cache.get(domain_key)
        if state and self._is_browser_session_fresh(state, options) and not force_refresh:
            try:
                return await self._fetch_with_browser_session_state(url=url, referer=referer, state=state)
            except Exception:
                pass

        lock = self._session_lock_for_domain(domain_key)
        async with lock:
            state = self._browser_session_cache.get(domain_key)
            if state and self._is_browser_session_fresh(state, options) and not force_refresh:
                try:
                    return await self._fetch_with_browser_session_state(url=url, referer=referer, state=state)
                except Exception:
                    pass

            state, accepted_responses = await self._bootstrap_browser_session_state(
                url=url,
                referer=referer,
                route_options=options,
            )
            self._browser_session_cache[domain_key] = state
            candidate_url = accepted_responses[0][0] if accepted_responses else url
            body, response_headers, meta = await self._fetch_with_browser_session_state(
                url=candidate_url,
                referer=referer,
                state=state,
            )
            meta.update({
                "candidate_url": candidate_url,
                "session_reused": False,
                "domain_key": domain_key,
            })
            return body, response_headers, meta

    async def _download_browser_route(
        self,
        *,
        row_index: int,
        url: str,
        retry_count: int,
        filename_base: Optional[str],
        referer: Optional[str],
        route_options: Dict[str, Any],
    ) -> Tuple[bool, str, str, str, int]:
        try:
            body, response_headers, meta = await self._download_via_browser_session(
                url=url,
                referer=referer,
                route_options=route_options,
            )
        except Exception as exc:
            error_msg = f"Browser-routed download failed: {exc}"
            self.logger.warning(error_msg)
            return False, "", self._best_effort_url_extension(url), error_msg, retry_count + 1
        return await self._finalize_download_result(
            row_index=row_index,
            url=meta.get("candidate_url") or url,
            resp_headers=response_headers,
            content=body,
            retry_count=retry_count,
            filename_base=filename_base,
            referer=referer,
        )

    async def _preflight_download(
        self,
        *,
        row_index: int,
        url: str,
        retry_count: int,
        filename_base: Optional[str],
        referer: Optional[str],
    ) -> Optional[Tuple[bool, str, str, str, int]]:
        route, route_options = self._resolve_route(url)
        if route != "browser":
            return None
        return await self._download_browser_route(
            row_index=row_index,
            url=url,
            retry_count=retry_count,
            filename_base=filename_base,
            referer=referer,
            route_options=route_options,
        )

    async def _recover_html_interstitial(
        self,
        *,
        row_index: int,
        url: str,
        headers: Dict[str, str],
        content: bytes,
        html_issue: str,
        retry_count: int,
        filename_base: Optional[str],
        referer: Optional[str],
    ) -> Optional[Tuple[bool, str, str, str, int]]:
        route, route_options = self._resolve_route(url)
        if route == "standard":
            return None
        if route == "auto" and not self._should_attempt_browser_recovery(url, html_issue):
            return None

        try:
            body, response_headers, meta = await self._download_via_browser_session(
                url=url,
                referer=referer,
                route_options=route_options,
            )
        except Exception as exc:
            message = f"{html_issue}; browser recovery failed: {exc}"
            self.logger.warning(message)
            return False, "", "html", message, retry_count + 1

        file_ext = self.infer_file_extension(meta["candidate_url"], response_headers, body)
        if file_ext == "html":
            message = (
                f"{html_issue}; browser recovery still returned HTML from {meta['candidate_url']}"
            )
            self.logger.warning(message)
            return False, "", file_ext, message, retry_count + 1
        if not self.is_supported_format(file_ext):
            message = (
                f"{html_issue}; browser recovery returned unsupported format: {file_ext}"
            )
            self.logger.warning(message)
            return False, "", file_ext or "", message, retry_count + 1

        if filename_base and str(filename_base).strip():
            filename = f"{filename_base}.{file_ext}"
        else:
            filename = self.generate_filename(row_index, file_ext)

        await self._write_recovered_file(row_index, filename, body)
        self.logger.info(
            "Recovered browser-gated download via browser mode: %s -> %s",
            url,
            filename,
        )
        return True, filename, file_ext, "", retry_count
