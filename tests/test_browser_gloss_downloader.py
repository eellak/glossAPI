import asyncio

import pandas as pd

from glossapi import Corpus
from glossapi.download_policy import build_download_policy
from glossapi.gloss_browser_downloader import BrowserGlossDownloader, BrowserSessionState
import glossapi.corpus.phase_download as phase_download_mod


def test_browser_downloader_skips_viewer_interstitial(tmp_path, monkeypatch):
    downloader = BrowserGlossDownloader(output_dir=str(tmp_path))
    called = False

    async def _fake_browser_download(**kwargs):
        nonlocal called
        called = True
        return b"%PDF-1.7\n", {"Content-Type": "application/pdf"}, {"candidate_url": kwargs["url"]}

    monkeypatch.setattr(downloader, "_download_via_browser_session", _fake_browser_download)

    result = asyncio.run(
        downloader._recover_html_interstitial(
            row_index=0,
            url="https://freader.ekt.gr/eadd/index.php?doc=60819&lang=el",
            headers={"Content-Type": "text/html"},
            content=b"<html></html>",
            html_issue=(
                "HTML document viewer returned instead of a downloadable file; "
                "a source-specific fetcher with persisted cookies/redirect handling is required"
            ),
            retry_count=0,
            filename_base="AAA_000",
            referer=None,
        )
    )

    assert result is None
    assert called is False


def test_browser_downloader_recovers_challenge_page(tmp_path, monkeypatch):
    downloader = BrowserGlossDownloader(output_dir=str(tmp_path))

    async def _fake_browser_download(**kwargs):
        return (
            b"%PDF-1.7\n%dummy\n",
            {"Content-Type": "application/pdf"},
            {"candidate_url": "https://example.org/file.pdf"},
        )

    monkeypatch.setattr(downloader, "_download_via_browser_session", _fake_browser_download)

    result = asyncio.run(
        downloader._recover_html_interstitial(
            row_index=0,
            url="https://example.org/file.pdf",
            headers={"Content-Type": "text/html"},
            content=b"<html>challenge</html>",
            html_issue=(
                "HTML challenge page returned instead of a document; "
                "browser automation or cookie bootstrap is required"
            ),
            retry_count=1,
            filename_base="AAA_000",
            referer=None,
        )
    )

    assert result == (True, "AAA_000.pdf", "pdf", "", 1)
    assert (tmp_path / "downloads" / "AAA_000.pdf").read_bytes().startswith(b"%PDF-1.7")
    assert not (tmp_path / "downloads" / ".part_browser_0").exists()


def test_browser_downloader_domain_cookie_lookup(tmp_path):
    downloader = BrowserGlossDownloader(
        output_dir=str(tmp_path),
        domain_cookies={"eur-lex.europa.eu": {"token": "abc123"}},
    )

    cookies = downloader._domain_cookies_for_url(
        "https://eur-lex.europa.eu/legal-content/EL/TXT/PDF/?uri=OJ:L_202502360"
    )

    assert cookies == {"token": "abc123"}


def test_browser_downloader_bootstrap_url_uses_base_for_file_endpoints(tmp_path):
    downloader = BrowserGlossDownloader(output_dir=str(tmp_path))

    assert downloader._choose_browser_bootstrap_url(
        "https://eur-lex.europa.eu/legal-content/EL/TXT/PDF/?uri=OJ:L_202502360"
    ) == "https://eur-lex.europa.eu"


def test_browser_downloader_ignores_err_aborted_for_file_navigation(tmp_path):
    downloader = BrowserGlossDownloader(output_dir=str(tmp_path))

    assert downloader._should_ignore_navigation_exception(
        "https://eur-lex.europa.eu/legal-content/EL/TXT/PDF/?uri=OJ:L_202502360",
        RuntimeError("Page.goto: net::ERR_ABORTED"),
    )
    assert not downloader._should_ignore_navigation_exception(
        "https://example.org/article",
        RuntimeError("Page.goto: net::ERR_ABORTED"),
    )


def test_browser_downloader_uses_default_browser_route_for_preflight(tmp_path, monkeypatch):
    downloader = BrowserGlossDownloader(output_dir=str(tmp_path), default_download_route="browser")

    async def _fake_download_browser_route(**kwargs):
        return True, "AAA_000.pdf", "pdf", "", 0

    monkeypatch.setattr(downloader, "_download_browser_route", _fake_download_browser_route)

    result = asyncio.run(
        downloader._preflight_download(
            row_index=0,
            url="https://example.org/file.pdf",
            retry_count=0,
            filename_base="AAA_000",
            referer=None,
        )
    )

    assert result == (True, "AAA_000.pdf", "pdf", "", 0)


def test_browser_downloader_reuses_cached_domain_session(tmp_path, monkeypatch):
    downloader = BrowserGlossDownloader(output_dir=str(tmp_path), default_download_route="auto")
    bootstraps = 0
    fetches = 0

    async def _fake_fetch_with_browser_session_state(**kwargs):
        nonlocal fetches
        fetches += 1
        return b"%PDF-1.7\n", {"Content-Type": "application/pdf"}, {"candidate_url": kwargs["url"]}

    async def _bootstrap(**kwargs):
        nonlocal bootstraps
        bootstraps += 1
        return BrowserSessionState(user_agent="UA", cookie_header="a=b", cached_at=10_000.0), []

    monkeypatch.setattr(downloader, "_bootstrap_browser_session_state", _bootstrap)
    monkeypatch.setattr(downloader, "_fetch_with_browser_session_state", _fake_fetch_with_browser_session_state)
    monkeypatch.setattr("glossapi.gloss_browser_downloader.time.time", lambda: 10_100.0)

    first = asyncio.run(
        downloader._download_via_browser_session(url="https://eur-lex.europa.eu/file.pdf", referer=None)
    )
    second = asyncio.run(
        downloader._download_via_browser_session(url="https://eur-lex.europa.eu/file2.pdf", referer=None)
    )

    assert first[0].startswith(b"%PDF")
    assert second[0].startswith(b"%PDF")
    assert bootstraps == 1
    assert fetches == 2


def test_browser_downloader_policy_routes_domain_to_browser(tmp_path, monkeypatch):
    policy = build_download_policy(
        {
            "default": {"downloader": "standard"},
            "rules": [
                {
                    "match": {"domains": ["eur-lex.europa.eu"]},
                    "downloader": "browser",
                    "browser_timeout_ms": 1234,
                }
            ],
        }
    )
    downloader = BrowserGlossDownloader(
        output_dir=str(tmp_path),
        download_policy=policy,
        default_download_route="standard",
    )

    observed = {}

    async def _fake_download_browser_route(**kwargs):
        observed.update(kwargs)
        return True, "AAA_000.pdf", "pdf", "", 0

    monkeypatch.setattr(downloader, "_download_browser_route", _fake_download_browser_route)

    result = asyncio.run(
        downloader._preflight_download(
            row_index=0,
            url="https://eur-lex.europa.eu/legal-content/EL/TXT/PDF/?uri=OJ:L_202502360",
            retry_count=0,
            filename_base="AAA_000",
            referer=None,
        )
    )

    assert result == (True, "AAA_000.pdf", "pdf", "", 0)
    assert observed["route_options"]["browser_timeout_ms"] == 1234


def test_download_policy_preserves_transport_and_scheduler_options():
    policy = build_download_policy(
        {
            "default": {"downloader": "standard"},
            "rules": [
                {
                    "match": {"domains": ["ikee.lib.auth.gr"]},
                    "downloader": "standard",
                    "request_timeout": 120,
                    "ssl_verify": False,
                    "per_domain_concurrency": 2,
                    "domain_concurrency_floor": 1,
                    "domain_concurrency_ceiling": 3,
                    "skip_failed_after": 5,
                    "domain_cookies": {"sessionid": "abc"},
                }
            ],
        }
    )

    route, options = policy.resolve("https://ikee.lib.auth.gr/record/123/files/file.pdf")

    assert route == "standard"
    assert options["request_timeout"] == 120
    assert options["ssl_verify"] is False
    assert options["per_domain_concurrency"] == 2
    assert options["domain_concurrency_floor"] == 1
    assert options["domain_concurrency_ceiling"] == 3
    assert options["skip_failed_after"] == 5
    assert options["domain_cookies"] == {"sessionid": "abc"}


def test_browser_downloader_route_options_apply_standard_transport_settings(tmp_path):
    policy = build_download_policy(
        {
            "default": {"downloader": "standard"},
            "rules": [
                {
                    "match": {"domains": ["ktisis.cut.ac.cy"]},
                    "downloader": "standard",
                    "request_timeout": 90,
                    "ssl_verify": False,
                    "per_domain_concurrency": 2,
                    "domain_concurrency_floor": 1,
                    "domain_concurrency_ceiling": 2,
                    "skip_failed_after": 4,
                    "domain_cookies": {"sessionid": "abc"},
                }
            ],
        }
    )
    downloader = BrowserGlossDownloader(
        output_dir=str(tmp_path),
        download_policy=policy,
        default_download_route="standard",
    )

    async def _build_connector():
        return downloader._build_session_connector(
            "https://ktisis.cut.ac.cy/items/123/file.pdf",
            route_options=route_options,
        )

    route, route_options = downloader._resolve_route("https://ktisis.cut.ac.cy/items/123/file.pdf")
    timeout = downloader._build_request_timeout(0, route_options=route_options)
    connector = asyncio.run(_build_connector())
    cookies = downloader._resolve_request_cookies(
        "https://ktisis.cut.ac.cy/items/123/file.pdf",
        route_options=route_options,
    )
    floor, ceiling, start, skip_after = downloader._resolve_domain_scheduler_settings(route_options)

    assert route == "standard"
    assert timeout.total == 90
    assert connector is not None
    assert cookies["sessionid"] == "abc"
    assert (floor, ceiling, start, skip_after) == (1, 2, 2, 4)


def test_corpus_download_mode_selects_browser_downloader(tmp_path, monkeypatch):
    input_df = pd.DataFrame({"url": ["https://example.org/file.pdf"]})
    input_parquet = tmp_path / "urls.parquet"
    input_df.to_parquet(input_parquet, index=False)

    observed = {}

    class DummyBrowserDownloader:
        def __init__(self, *args, **kwargs):
            observed["cls"] = "browser"
            observed["kwargs"] = kwargs

        def download_files(self, input_parquet: str, **kwargs):
            return pd.DataFrame(
                {
                    "url": ["https://example.org/file.pdf"],
                    "filename": ["AAA_000.pdf"],
                    "download_success": [True],
                    "download_error": [""],
                }
            )

    monkeypatch.setattr(phase_download_mod, "BrowserGlossDownloader", DummyBrowserDownloader)

    corpus = Corpus(input_dir=tmp_path, output_dir=tmp_path)
    result = corpus.download(input_parquet=input_parquet, download_mode="browser")

    assert observed["cls"] == "browser"
    assert observed["kwargs"]["default_download_route"] == "browser"
    assert bool(result["download_success"].iloc[0]) is True
    assert (tmp_path / "download_results" / f"download_results_{input_parquet.name}").exists()


def test_corpus_browser_mode_alias_selects_browser_downloader(tmp_path, monkeypatch):
    input_df = pd.DataFrame({"url": ["https://example.org/file.pdf"]})
    input_parquet = tmp_path / "urls.parquet"
    input_df.to_parquet(input_parquet, index=False)

    observed = {}

    class DummyBrowserDownloader:
        def __init__(self, *args, **kwargs):
            observed["cls"] = "browser"

        def download_files(self, input_parquet: str, **kwargs):
            return pd.DataFrame(
                {
                    "url": ["https://example.org/file.pdf"],
                    "filename": ["AAA_000.pdf"],
                    "download_success": [True],
                    "download_error": [""],
                }
            )

    monkeypatch.setattr(phase_download_mod, "BrowserGlossDownloader", DummyBrowserDownloader)

    corpus = Corpus(input_dir=tmp_path, output_dir=tmp_path)
    corpus.download(input_parquet=input_parquet, browser_mode=True)

    assert observed["cls"] == "browser"


def test_corpus_policy_file_selects_browser_router(tmp_path, monkeypatch):
    input_df = pd.DataFrame({"url": ["https://eur-lex.europa.eu/file.pdf"]})
    input_parquet = tmp_path / "urls.parquet"
    input_df.to_parquet(input_parquet, index=False)
    policy_path = tmp_path / "download_policy.yml"
    policy_path.write_text(
        "default:\n  downloader: standard\nrules:\n  - match:\n      domains: [eur-lex.europa.eu]\n    downloader: browser\n",
        encoding="utf-8",
    )

    observed = {}

    class DummyBrowserDownloader:
        def __init__(self, *args, **kwargs):
            observed["kwargs"] = kwargs

        def download_files(self, input_parquet: str, **kwargs):
            return pd.DataFrame(
                {
                    "url": ["https://eur-lex.europa.eu/file.pdf"],
                    "filename": ["AAA_000.pdf"],
                    "download_success": [True],
                    "download_error": [""],
                }
            )

    monkeypatch.setattr(phase_download_mod, "BrowserGlossDownloader", DummyBrowserDownloader)

    corpus = Corpus(input_dir=tmp_path, output_dir=tmp_path)
    corpus.download(input_parquet=input_parquet, download_policy_file=policy_path)

    assert observed["kwargs"]["download_policy_file"] == policy_path.resolve()
    assert observed["kwargs"]["default_download_route"] == "standard"
