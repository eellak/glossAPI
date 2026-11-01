"""CP-CRM probes (reviewer file review_scratch/CP-CRM/test_reviewer_entry.py, ported verbatim: 17 fail and 4 pass on
f545cf3, all pass on the fix), then fix-round tests for the same findings. All secrets are synthetic markers; sockets
are denied, HTTP boundaries intercepted."""
import asyncio
import json
import socket
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from glossapi import Corpus
from glossapi.cli import plan_documents
from glossapi.ledger.urls import CredentialsUnsupportedError
from glossapi.gloss_downloader import GlossDownloader, RateLimiter

MARK = "CRM_REVIEW_SYNTHETIC_SECRET"
BAD = "https://example.invalid/a.pdf?access_token=" + MARK
GOOD = "https://example.invalid/a.pdf?id=7"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOSSAPI_CONVERSION_LOG_FILE", str(tmp_path / "conversion.log"))
    def deny(*args, **kwargs):
        raise AssertionError("review probe forbids socket connections")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def hits(out):
    found = [str(p.relative_to(out)) for p in Path(out).rglob("*")
             if p.is_file() and not p.is_symlink() and MARK.encode() in p.read_bytes()]
    database = Path(out) / "corpus.sqlite3"
    if database.exists():
        from glossapi.ledger.writer import open_readonly
        with open_readonly(database) as connection:
            for row in connection.execute("SELECT kind,json FROM snapshots"):
                if MARK in row[1]:
                    found.append("ledger:snapshots:" + row[0])
    return sorted(set(found))


@pytest.mark.parametrize("selector", [BAD, "HTTPS://user:" + MARK + "@example.invalid/a.pdf",
                                      " https://example.invalid/a.pdf?token=" + MARK,
                                      "url:v1:" + BAD])
def test_plan_all_selector_spellings_refuse_without_echo(tmp_path, selector):
    try:
        result = plan_documents(tmp_path / "out", files=[selector], stages=["extract"])
    except Exception as error:
        assert MARK not in str(error) + json.dumps(getattr(error, "details", {}))
        assert isinstance(error, CredentialsUnsupportedError), repr(error)
    else:
        assert MARK not in json.dumps(result), "secret in returned plan"
        pytest.fail("credentialed input was accepted")


def test_unselected_manifest_url_cannot_escape_into_plan(tmp_path):
    manifest = {"manifest_version": 1, "documents": [
        {"url": GOOD, "selection_id": "good"}, {"url": BAD, "selection_id": "bad"}], "selection": ["good"]}
    with pytest.raises(CredentialsUnsupportedError):
        result = plan_documents(tmp_path / "out", manifest=manifest, stages=["extract"])
        print("PLAN_SECRET", MARK in json.dumps(result), "EXCLUSIONS", result["exclusions"])


def test_nested_download_policy_refused_before_plan(tmp_path):
    settings = {"download": {"download_policy": {"default": {"domain_cookies": {"s": MARK}}}}}
    with pytest.raises(CredentialsUnsupportedError):
        result = plan_documents(tmp_path / "out", files=[GOOD], stages=["extract"], settings=settings)
        print("PLAN_SECRET", MARK in json.dumps(result))


def test_nested_policy_refused_before_run_records(monkeypatch, tmp_path):
    from glossapi.stages.scheduler import StreamingScheduler
    def stop_before_workers(*args, **kwargs):
        raise RuntimeError("review stopped before dispatch")
    monkeypatch.setattr(StreamingScheduler, "execute", stop_before_workers)
    out = tmp_path / "out"
    error = None
    try:
        Corpus(tmp_path / "empty", out).run([GOOD], stages=("extract",),
            settings={"download": {"download_policy": {"default": {"domain_cookies": {"s": MARK}}}}})
    except Exception as caught:
        error = caught
    import test_k4_credential_refusal as shipped
    monkeypatch.setattr(shipped, "MARKER", MARK)
    print("SHIPPED_MARKER_SCANNER", shipped.marker_hits(out))
    print("ERROR", repr(error), "HITS", hits(out))
    assert hits(out) == [], "credential download settings persisted before refusal"
    assert isinstance(error, CredentialsUnsupportedError)


@pytest.mark.parametrize("cell", [json.dumps([GOOD, BAD]), {"href": BAD},
                                    [{"url": BAD}], GOOD + ";" + BAD])
def test_legacy_expanded_cells_refused_before_transport(monkeypatch, tmp_path, cell):
    from glossapi.stages.profile import _public_adapter_scope
    out = tmp_path / "out"
    source = tmp_path / "source.parquet"
    pd.DataFrame({"url": [cell]}).to_parquet(source, index=False)
    seen = []
    async def fake_download(self, row_index, url, *args, **kwargs):
        seen.append(url)
        return False, "", ".pdf", "review fake transport", 99
    monkeypatch.setattr(GlossDownloader, "download_file", fake_download)
    error = None
    try:
        Corpus(tmp_path / "empty", out).download(source, rate_limit=0, max_retries=1, retry_interval=0, sleep=0,
                                                  pre_ping_domains=False)
    except Exception as caught:
        error = caught
    print("ERROR", repr(error), "TRANSPORT", seen, "HITS", hits(out))
    assert seen == [], "credentialed URL reached per-file acquisition"
    assert hits(out) == [], "credentialed source persisted"
    assert isinstance(error, CredentialsUnsupportedError)


@pytest.mark.parametrize("url", [BAD, "https://user:" + MARK + "@example.invalid/a.pdf"])
def test_direct_download_file_refuses_before_http(monkeypatch, tmp_path, url):
    import glossapi.gloss_downloader as module
    seen = []
    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda **kwargs: FakeSession())
    monkeypatch.setattr(GlossDownloader, "_build_session_connector", lambda *args, **kwargs: None)
    def boundary(session, method, target, **kwargs):
        seen.append(target)
        raise RuntimeError("review intercepted HTTP request")
    monkeypatch.setattr(module, "recorded_request", boundary)
    dl = GlossDownloader(output_dir=str(tmp_path / "out"), sleep=0, max_retries=1, rate_limit=0, request_timeout=1)
    error = None
    try:
        asyncio.run(dl.download_file(0, url, None, RateLimiter(1000000)))
    except Exception as caught:
        error = caught
    finally:
        dl.close_logs()
    print("ERROR", repr(error), "HTTP_BOUNDARY", seen, "HITS", hits(tmp_path / "out"))
    assert seen == [], "credentialed URL reached recorded_request"
    assert isinstance(error, CredentialsUnsupportedError)


def test_amended_plan_is_revalidated_before_recording(monkeypatch, tmp_path):
    from glossapi.ledger.encoding import snapshot_hash
    result = plan_documents(tmp_path / "out", files=[GOOD], stages=["extract"])
    result["documents"][0]["source_observation"]["source_uri"] = BAD
    result["plan_hash"] = snapshot_hash({k: v for k, v in result.items() if k != "plan_hash"})
    with pytest.raises(CredentialsUnsupportedError):
        Corpus(tmp_path / "empty", tmp_path / "out").run(plan=result)
    assert hits(tmp_path / "out") == []


def test_scheduler_freeze_and_reload_refuse_credentialed_operation(tmp_path):
    from glossapi.ledger.writer import LedgerWriter
    from glossapi.stages.scheduler import StreamingScheduler, ScheduledOperation
    from glossapi.stages.contracts import DocumentSnapshot, StageSpecification
    out = tmp_path / "out"
    out.mkdir()
    with LedgerWriter(out / "corpus.sqlite3") as writer:
        engine = StreamingScheduler(writer, out)
        operation = ScheduledOperation("download", "download", DocumentSnapshot("synthetic-doc", None, BAD),
            StageSpecification("download.acquire", settings={"url": BAD, "filename_base": "AAA_000"}))
        error = None
        try:
            run = engine.freeze([operation])
            loaded, _, _ = engine._load(run)
            print("FROZEN_RUN", run, "RELOADED_URL", loaded[0].specification.settings["url"])
        except Exception as caught:
            error = caught
    print("ERROR", repr(error), "HITS", hits(out))
    assert hits(out) == [], "scheduler recorded credentialed operation"
    assert isinstance(error, CredentialsUnsupportedError)


def test_planning_by_known_document_id_revalidates_its_url(tmp_path):
    from glossapi.ledger.writer import LedgerWriter
    from glossapi.ledger.identity import source_key
    out = tmp_path / "out"
    out.mkdir()
    with LedgerWriter(out / "corpus.sqlite3") as writer:
        with writer.transaction():
            identifier = writer.document(source_key(BAD))
    with pytest.raises(CredentialsUnsupportedError):
        result = plan_documents(out, files=[identifier], stages=["extract"])
        print("PLAN_SECRET", MARK in json.dumps(result), "SOURCE", result["documents"][0]["source_key"])


@pytest.mark.parametrize("field", ["cookie", "access_key", "accessToken"])
def test_obvious_credential_field_spellings_are_refused(tmp_path, field):
    with pytest.raises(CredentialsUnsupportedError):
        result = plan_documents(tmp_path / "out", files=["https://example.invalid/a.pdf?" + field + "=" + MARK], stages=["extract"])
        print("PLAN_SECRET", MARK in json.dumps(result), "FIELD", field)


def test_cli_exclusion_and_nested_policy_are_refused_before_plan(tmp_path):
    import contextlib
    import io
    from glossapi.cli import main
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "documents": [
        {"url": GOOD, "selection_id": "good"}, {"url": BAD, "selection_id": "bad"}], "selection": ["good"]}))
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"download": {"download_policy": {"default": {"domain_cookies": {"s": MARK}}}}}))
    observed = []
    for arguments in (["--manifest", str(manifest)], ["--files", GOOD, "--settings", str(settings)]):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["plan", "--corpus", str(tmp_path / "out"), "--stages", "extract", "--json", *arguments])
        envelope = json.loads(buffer.getvalue())
        observed.append({"exit": code, "secret_in_json": MARK in buffer.getvalue(), "errors": envelope["errors"]})
    print("CLI", observed)
    assert all(v["exit"] == 2 and not v["secret_in_json"] for v in observed)


def test_cli_saved_amended_plan_refuses_source_secret(tmp_path):
    import contextlib
    import io
    from glossapi.cli import main
    from glossapi.ledger.encoding import snapshot_hash
    out = tmp_path / "out"
    result = plan_documents(out, files=[GOOD], stages=["extract"])
    result["documents"][0]["source_observation"]["source_uri"] = BAD
    result["plan_hash"] = snapshot_hash({k: v for k, v in result.items() if k != "plan_hash"})
    saved = tmp_path / "plan.json"
    saved.write_text(json.dumps(result))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(["run", "--corpus", str(out), "--plan", str(saved), "--json"])
    envelope = json.loads(buffer.getvalue())
    assert code == 2 and envelope["errors"][0]["code"] == "credentials_unsupported", envelope
    assert MARK not in buffer.getvalue() and hits(out) == []


# ===================================================== part 2: CP-CRM fix round 1 (one rule, few choke points) ===

ALIASES = ["cookie", "Cookie", "cookies", "access_key", "accessKey", "AccessKey", "accessToken", "access-token", "ACCESS_TOKEN",
           "accesstoken", "api_key", "apiKey", "x-api-key", "secret", "clientSecret", "client_secret", "signature", "Signature",
           "X-Amz-Signature", "sig", "password", "Password", "passwd", "pwd", "token", "Token", "auth_token", "authToken",
           "refreshToken", "id_token", "session_id", "sessionId", "secret_key", "privateKey", "jwt", "key", "Authorization",
           "X-Goog-Credential", "credential"]


@pytest.mark.parametrize("field", ALIASES)
@pytest.mark.parametrize("separator", ["?", "#", "?id=7&"])
def test_credential_field_aliases_are_refused_in_query_and_fragment(field, separator):
    from glossapi.ledger.urls import refuse_credentials
    with pytest.raises(CredentialsUnsupportedError) as raised:
        refuse_credentials(urls=["https://example.invalid/a.pdf" + separator + field + "=" + MARK])
    assert MARK not in str(raised.value) + json.dumps(raised.value.details)


@pytest.mark.parametrize("field", ["id", "q", "sequence", "isAllowed", "page", "lang", "session", "keyword", "monkey", "format"])
def test_public_field_names_pass(field):
    from glossapi.ledger.urls import refuse_credentials
    refuse_credentials(urls=["https://example.invalid/a.pdf?" + field + "=1"])


def test_download_policy_file_and_object_with_cookies_are_refused_before_plan(tmp_path):
    from glossapi.download_policy import DownloadPolicy
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("default:\n  downloader: standard\nrules:\n  - match: {domains: [example.invalid]}\n    Cookie: " + MARK + "\n")
    for settings in ({"download": {"download_policy_file": str(policy_file)}},
                     {"download": {"download_policy": DownloadPolicy(default_options={"domain_cookies": {"s": MARK}})}},
                     {"download": {"download_policy": {"rules": [{"match": {}, "accessToken": MARK}]}}},
                     {"download": {"Cookies": {"s": MARK}}}):
        with pytest.raises(CredentialsUnsupportedError) as raised:
            plan_documents(tmp_path / "out", files=[GOOD], stages=["extract"], settings=settings)
        assert MARK not in str(raised.value) + json.dumps(raised.value.details)
    assert hits(tmp_path / "out") == []


def test_direct_download_files_refuses_the_expanded_frame_before_any_request(monkeypatch, tmp_path):
    seen = []
    async def fake_download(self, row_index, url, *args, **kwargs):
        seen.append(url)
        return False, "", ".pdf", "fake", 0
    monkeypatch.setattr(GlossDownloader, "download_file", fake_download)
    for index, frame in enumerate([pd.DataFrame({"url": [GOOD, json.dumps([GOOD, BAD])]}),
                                   pd.DataFrame({"url": ["user:" + MARK + "@example.invalid/a.pdf"]}),  # fetched as https://
                                   pd.DataFrame({"url": [GOOD], "page": ["https://example.invalid/p?token=" + MARK]})]):
        out = tmp_path / f"out{index}"
        dl = GlossDownloader(output_dir=str(out), sleep=0, rate_limit=0, referer_column="page", pre_ping_domains=False)
        try:
            with pytest.raises(CredentialsUnsupportedError):
                dl.download_files(str(tmp_path / "unused.parquet"), input_frame=frame)
        finally:
            dl.close_logs()
        assert hits(out) == [], index
    assert seen == []


def test_legacy_download_checks_fallback_url_columns(monkeypatch, tmp_path):
    seen = []
    async def fake_download(self, row_index, url, *args, **kwargs):
        seen.append(url)
        return False, "", ".pdf", "fake", 0
    monkeypatch.setattr(GlossDownloader, "download_file", fake_download)
    source = tmp_path / "source.parquet"
    pd.DataFrame({"pdf_link": [GOOD, BAD]}).to_parquet(source, index=False)
    with pytest.raises(CredentialsUnsupportedError):
        Corpus(tmp_path / "empty", tmp_path / "out").download(source, rate_limit=0, sleep=0, pre_ping_domains=False)
    assert seen == [] and hits(tmp_path / "out") == []


def test_a_frozen_run_is_checked_again_on_reload(monkeypatch, tmp_path):
    import glossapi.stages.scheduler as scheduler
    from glossapi.ledger.writer import LedgerWriter
    from glossapi.stages.contracts import DocumentSnapshot, StageSpecification
    out = tmp_path / "out"
    out.mkdir()
    with LedgerWriter(out / "corpus.sqlite3") as writer:
        engine = scheduler.StreamingScheduler(writer, out)
        operation = scheduler.ScheduledOperation("download", "download", DocumentSnapshot("synthetic-doc", None, BAD),
            StageSpecification("download.acquire", settings={"url": BAD, "filename_base": "AAA_000"}))
        guard = scheduler._refuse_credentialed
        monkeypatch.setattr(scheduler, "_refuse_credentialed", lambda *args: None)  # a ledger written before the rule
        run = engine.freeze([operation])
        monkeypatch.setattr(scheduler, "_refuse_credentialed", guard)
        with pytest.raises(CredentialsUnsupportedError) as raised:
            scheduler.StreamingScheduler(writer, out)._load(run)
    assert MARK not in str(raised.value)


def test_cli_run_with_policy_cookies_and_unselected_credentialed_row_is_refused(tmp_path):
    import contextlib
    import io
    from glossapi.cli import main
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 1, "documents": [
        {"url": GOOD, "selection_id": "good"}, {"url": " HTTPS://u:" + MARK + "@example.invalid/b.pdf", "selection_id": "bad"}],
        "selection": ["good"]}))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(["run", "--corpus", str(tmp_path / "out"), "--stages", "extract", "--json", "--manifest", str(manifest)])
    assert code == 2 and json.loads(buffer.getvalue())["errors"][0]["code"] == "credentials_unsupported"
    assert MARK not in buffer.getvalue() and hits(tmp_path / "out") == []
