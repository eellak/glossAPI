"""Adapter contracts and real publication, coverage, HTTP and process fences."""

from glossapi.ledger.compact import connect as ledger_connect
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

from glossapi.stages.contracts import ArtifactReference, DocumentSnapshot, OutputArtifact, StageRequest, StageResult, StageSpecification
from glossapi.stages.download import DownloadStage
from glossapi.stages.execution import ReceiptCollector, checked_path, outputs_in, run_fenced, sha256_file
from glossapi.stages.extract import ExtractStage


def request(tmp_path, source=None, **settings):
    inputs = () if source is None else (ArtifactReference(str(uuid.uuid4()), "source_payload", source,
        sha256_file(source), source.stat().st_size),)
    return StageRequest(*(str(uuid.uuid4()) for _ in range(5)), settings.pop("operation", "extract.document"),
        inputs, settings, {}, "cpu", tmp_path/"attempt", time.monotonic_ns()+60_000_000_000)


def simple_pdf(path, *, blank=False):
    stream = b"" if blank else b"BT /F1 12 Tf 20 100 Td (A small text layer) Tj ET"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "+str(len(stream)).encode()+b" >>\nstream\n"+stream+b"\nendstream"]
    payload = b"%PDF-1.4\n"; offsets = [0]
    for number, value in enumerate(objects, 1):
        offsets.append(len(payload)); payload += str(number).encode()+b" 0 obj\n"+value+b"\nendobj\n"
    xref = len(payload)
    payload += b"xref\n0 6\n0000000000 65535 f \n"+b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    payload += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"+str(xref).encode()+b"\n%%EOF\n"
    path.write_bytes(payload)
    return path


def prepared_extractor():
    from glossapi.gloss_extract import GlossExtract
    extractor = GlossExtract()
    extractor.use_pypdfium_backend = True
    extractor.repair_fonts = False
    extractor.pdf_backend_name = "pypdfium"
    return extractor


def test_contracts_are_frozen_serializable_and_metadata_imports_are_light():
    script = """
import sys
from glossapi.stages.contracts import StageRequest
from glossapi.stages.download import DownloadStage
from glossapi.stages.extract import ExtractStage
assert not any(x in sys.modules for x in ('pandas','numpy','docling','torch','aiohttp','pypdfium2'))
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    specification = StageSpecification("extract.document", settings={"phase1_backend": "docling"})
    assert pickle.loads(pickle.dumps(specification)) == specification
    assert ExtractStage().packages(specification)[1].distributions == ("docling==2.81.0",)
    assert DownloadStage().packages(StageSpecification("download.acquire", settings={"download_mode": "browser"}))[1].package_group == "browser"
    with pytest.raises(AttributeError):
        specification.operation = "changed"


def test_plans_declare_inputs_output_requirements_and_naming_barrier():
    document = DocumentSnapshot("document", "version", "https://example.org/paper.pdf")
    assert DownloadStage().plan([document], settings={}).barriers == ("source_manifest_and_naming_sealed",)
    planned = ExtractStage().plan([document], settings={}).operations[0]
    assert planned.skip_reason == "no_matching_inputs"
    assert planned.specification.input_roles == ("source_payload",)
    assert planned.specification.required_output_kinds == ("raw_markdown",)


@pytest.mark.parametrize("blank", [False, True])
def test_real_safe_adapter_preserves_text_and_records_blank_coverage(tmp_path, blank):
    source = simple_pdf(tmp_path/"source.pdf", blank=blank)
    original = source.read_bytes()
    task = request(tmp_path, source, repair_fonts=False)
    result = ExtractStage().run_document(task, reporter=ReceiptCollector())
    assert result.status == "ok", asdict(result)
    assert source.read_bytes() == original
    assert (task.work_directory/"markdown/source.md").read_text() == ("" if blank else "A small text layer")
    decision = result.decisions[0]
    assert decision["coverage"]["attempted"] == [1]
    assert decision["coverage"]["blank"] == ([1] if blank else [])
    assert decision["coverage_complete"] and decision["coverage_allows_source_release"]
    assert decision["retention_authority"] is False
    assert result.metrics["tools"]["packages"]


def test_safe_page_failure_is_warning_with_incomplete_coverage(tmp_path, monkeypatch):
    import pypdfium2
    class BrokenPage:
        def get_textpage(self):
            raise OSError("page decoder failed")
        def close(self):
            pass
    class FakePdf:
        def __init__(self, *args):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return BrokenPage()
        def close(self):
            pass
    monkeypatch.setattr(pypdfium2, "PdfDocument", FakePdf)
    source = simple_pdf(tmp_path/"source.pdf")
    task = request(tmp_path, source); task.work_directory.mkdir()
    result = ExtractStage().run_prepared(task, extractor=prepared_extractor(), reporter=ReceiptCollector())
    assert result.status == "ok"
    assert result.decisions[0]["coverage"]["failed"] == [1]
    assert not result.decisions[0]["coverage_allows_source_release"]
    warning = next(item for item in result.decisions[0]["backend_observations"] if item["reason_code"] == "safe_page_error")
    assert warning["error_type"] == "builtins.OSError" and "page decoder failed" in warning["traceback"]


def test_missing_required_json_fails_but_optional_json_does_not(tmp_path):
    source = simple_pdf(tmp_path/"source.pdf")
    task = request(tmp_path, source, required_output_kinds=["raw_markdown", "docling_json"])
    task.work_directory.mkdir()
    result = ExtractStage().run_prepared(task, extractor=prepared_extractor(), reporter=ReceiptCollector())
    assert result.status == "failed" and result.reason_code == "required_extract_artifact_missing"
    assert result.decisions[0]["missing_required_kinds"] == ["docling_json"]
    assert (task.work_directory/"markdown/source.md").is_file()
    assert all(output.retention_class == "failed_attempt" for output in result.outputs)


def test_partial_chunk_is_not_success_even_when_canonical_markdown_exists(tmp_path, monkeypatch):
    source = simple_pdf(tmp_path/"source.pdf")
    task = request(tmp_path, source); task.work_directory.mkdir()
    extractor = prepared_extractor(); extractor.use_pypdfium_backend = False
    extractor.long_pdf_page_threshold = 0
    def partial(source, output_dir, timeout_dir):
        (output_dir/"source.md").write_text("partial")
        (output_dir/"source.chunks.json").write_text(json.dumps({"entries": [
            {"page_start": 1, "page_end": 1, "status": "partial"}]}))
        return True
    monkeypatch.setattr(extractor, "_process_file_chunked", partial)
    result = ExtractStage().run_prepared(task, extractor=extractor, reporter=ReceiptCollector())
    assert result.status == "failed" and result.reason_code == "chunk_assembly_failed"
    assert not result.decisions[0]["coverage_complete"]
    assert not result.decisions[0]["coverage_allows_source_release"]


def test_changed_input_is_rejected_before_kernel(tmp_path, monkeypatch):
    source = simple_pdf(tmp_path/"source.pdf")
    task = request(tmp_path, source); task.work_directory.mkdir()
    source.write_bytes(b"changed")
    extractor = prepared_extractor()
    monkeypatch.setattr(extractor, "_process_single_document", lambda *a, **kw: pytest.fail("Changed input reached backend"))
    result = ExtractStage().run_prepared(task, extractor=extractor, reporter=ReceiptCollector())
    assert result.status == "failed" and result.reason_code == "input_hash_changed"


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "nested/../../outside"])
def test_output_escape_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        checked_path(tmp_path, relative)


def test_output_symlink_rejected(tmp_path):
    outside = tmp_path/"outside"; outside.write_text("protected")
    root = tmp_path/"attempt"; root.mkdir(); (root/"link").symlink_to(outside)
    with pytest.raises(ValueError):
        checked_path(root, "link")
    with pytest.raises(ValueError):
        outputs_in(root)


class SlowExportStage:
    name = "extract"
    def _run_document(self, task, *, reporter):
        # Conversion completed; an export subprocess hangs and later writes.
        (task.work_directory/"partial.md").write_text("converted")
        child_code = "import pathlib,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(3);pathlib.Path('late.txt').write_text('stale')"
        subprocess.Popen([sys.executable, "-c", child_code], cwd=task.work_directory,
            start_new_session=bool(task.settings.get("detached_exporter", False)))
        time.sleep(20)


@pytest.mark.parametrize("detached", [False, True])
def test_timeout_reaps_export_worker_and_descendants_before_return(tmp_path, detached):
    if detached and sys.platform != "linux":
        pytest.skip("Detached process-family fencing is qualified on the Linux node")
    task = replace(request(tmp_path, detached_exporter=detached), deadline_monotonic_ns=time.monotonic_ns()+1_500_000_000)
    result = run_fenced(SlowExportStage(), task, reporter=ReceiptCollector())
    assert result.status == "failed" and result.reason_code == "extract_timeout"
    assert (task.work_directory/"partial.md").read_text() == "converted"
    assert result.error.retryable
    time.sleep(3.1)
    assert not (task.work_directory/"late.txt").exists()


def test_sequential_publisher_commits_verified_immutable_outputs(tmp_path):
    from glossapi.ledger import LedgerWriter
    from glossapi.ledger.identity import source_key
    from glossapi.stages.sequential import SequentialStageRunner
    source = simple_pdf(tmp_path/"source.pdf")
    corpus = tmp_path/"corpus"; corpus.mkdir()
    with LedgerWriter(corpus/"corpus.sqlite3") as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision="test")
            document = writer.document(source_key(source.as_uri()))
            version = writer.version(document, source_sha256=sha256_file(source), stored_sha256=sha256_file(source))
            file_id = writer.artifact(run, "extract", "source_payload", source.as_uri(), version_id=version,
                sha256=sha256_file(source), bytes=source.stat().st_size, ownership="initial_external", retention_class="initial")
        snapshot = DocumentSnapshot(document, version, source.as_uri(), artifacts=(ArtifactReference(file_id, "source_payload", source, sha256_file(source), source.stat().st_size),))
        result, destination = SequentialStageRunner(writer, corpus).execute(ExtractStage(), snapshot,
            StageSpecification("extract.document", ("source_payload",), ("raw_markdown",), settings={"repair_fonts": False}))
        assert result.status == "ok"
        assert "/.glossapi/artifacts/" in str(destination)
        assert writer.connection.execute("SELECT COUNT(*) FROM current_task_state WHERE status='ok'").fetchone()[0] == 1
        facts = writer.connection.execute("SELECT * FROM files WHERE producer_attempt_id=?", (result.attempt_id,)).fetchall()
        assert {row["kind"] for row in facts} >= {"raw_markdown", "source_snapshot"}
        for row in facts:
            from urllib.parse import urlparse, unquote
            path = Path(unquote(urlparse(row["uri"]).path))
            assert path.is_file() and sha256_file(path) == row["sha256"]
        assert not list((corpus/".glossapi/staging").iterdir())
        failed, _ = SequentialStageRunner(writer, corpus).execute(FailedExportStage(), snapshot,
            StageSpecification("extract.document", ("source_payload",), ("raw_markdown",)))
        failure = writer.connection.execute("SELECT status,error_type,message,traceback FROM events WHERE attempt_id=? AND event_kind='lifecycle' AND status='failed'", (failed.attempt_id,)).fetchone()
        assert tuple(failure[:3]) == ("failed", "builtins.OSError", "Cannot export document")
        assert "Cannot export document" in failure[3]
    assert source.is_file()


class FailedExportStage(ExtractStage):
    def _run_document(self, task, *, reporter):
        from glossapi.stages.execution import failed_result
        try:
            raise OSError("Cannot export document")
        except OSError as error:
            return failed_result(task, "extract_exception", error)


def test_page_quality_adapter_binds_input_and_effective_profile(tmp_path):
    source = simple_pdf(tmp_path/"source.pdf")
    task = request(tmp_path, source, operation="extract.page_quality", repair_fonts=False,
        page_quality={"rule_profile": "v3", "fold_validity": False})
    result = ExtractStage().run_document(task, reporter=ReceiptCollector())
    assert result.status == "ok", asdict(result)
    assert result.decisions[0]["input_sha256"] == task.inputs[0].sha256
    assert result.decisions[0]["effective_settings"]["rule_profile"] == "v3"
    assert result.decisions[0]["dictionary_sha256"] is None
    assert any(output.kind == "page_quality" for output in result.outputs)


@pytest.mark.skipif(sys.platform != "linux", reason="Node-local HTTP replay gate")
def test_http_redirect_inner_retry_changed_bytes_and_source_stored_hashes(tmp_path):
    pytest.importorskip("aiohttp"); pytest.importorskip("aiofiles")
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    state = {"calls": 0, "body": b"%PDF-1.4\n"+b"x"*70000+b"first"}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200); self.end_headers(); return
            if self.path == "/redirect":
                self.send_response(302); self.send_header("Location", "/payload.pdf"); self.end_headers(); return
            state["calls"] += 1
            if state["calls"] == 1:
                self.send_response(503); self.send_header("Retry-After", "1"); self.end_headers(); return
            self.send_response(200); self.send_header("Content-Type", "application/pdf")
            self.send_header("ETag", '"version"'); self.send_header("Set-Cookie", "secret")
            self.send_header("Content-Length", str(len(state["body"]))); self.end_headers(); self.wfile.write(state["body"])
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/redirect"
        first = request(tmp_path/"first", operation="download.acquire", url=url, filename_base="ABC_123",
            sleep=0, max_retries=2, request_timeout=10)
        a = DownloadStage().run_document(first, reporter=ReceiptCollector())
        assert a.status == "ok", asdict(a)
        acquisition = a.decisions[0]["acquisitions"][-1]
        assert acquisition["source_sha256"] == acquisition["stored_sha256"] == hashlib.sha256(state["body"]).hexdigest()
        assert acquisition["stored_bytes"] == acquisition["source_entity_bytes"] == len(state["body"])
        assert a.decisions[0]["assigned_filename"] == "ABC_123.pdf"
        requests = a.decisions[0]["http"]["requests"]
        payload_requests = [item for item in requests if item["requested_url"] == url]
        assert [item["retry_index"] for item in payload_requests] == [0, 1]
        assert [item["http_status"] for item in payload_requests] == [503, 200]
        assert payload_requests[0]["error"]["error_type"].endswith("ClientResponseError")
        assert payload_requests[0]["retry_after_ms"] == 1000
        assert payload_requests[1]["backoff_ms"] == 1000
        assert payload_requests[1]["redirects"][0]["status"] == 302
        assert payload_requests[1]["final_url"].endswith("/payload.pdf")
        assert all(item["started_at"] and item["finished_at"] for item in requests)
        assert "set-cookie" not in json.dumps(requests).lower() and "secret" not in json.dumps(requests)
        state["body"] = state["body"][:-5]+b"other"
        second = replace(first, work_directory=tmp_path/"second/attempt")
        b = DownloadStage().run_document(second, reporter=ReceiptCollector())
        assert b.status == "ok"
        assert b.decisions[0]["acquisitions"][-1]["stored_sha256"] != acquisition["stored_sha256"]
        assert (first.work_directory/"downloads/ABC_123.pdf").read_bytes() != (second.work_directory/"downloads/ABC_123.pdf").read_bytes()
        # The unchanged finite-manifest expansion/naming and compatibility
        # serializer retain source columns. Its assigned name equals the adapter.
        import pandas as pd
        from glossapi.gloss_downloader import GlossDownloader
        legacy_root = tmp_path/"legacy"
        manifest = tmp_path/"sources.parquet"
        pd.DataFrame({"url": [url], "filename_base": ["ABC_123"], "collection_slug": ["fixture"],
            "owner_source_id": ["source-7"], "title": ["Original source title"]}).to_parquet(manifest)
        legacy = GlossDownloader(output_dir=legacy_root, sleep=0, pre_ping_domains=False, max_retries=2)
        returned = legacy.download_files(str(manifest))
        assert returned.iloc[0]["filename"] == b.decisions[0]["assigned_filename"]
        assert returned.iloc[0]["owner_source_id"] == "source-7"
        assert returned.iloc[0]["collection_slug"] == "fixture"
        assert returned.iloc[0]["title"] == "Original source title"
        assert (legacy_root/"downloads/ABC_123.pdf").read_bytes() == (second.work_directory/"downloads/ABC_123.pdf").read_bytes()
        # Actual reacquisitions preserve the same URL document and append
        # versions when its bytes change. Source/stored hashes commit with ok.
        from glossapi.ledger import LedgerWriter
        from glossapi.ledger.identity import source_key
        from glossapi.stages.sequential import SequentialStageRunner
        corpus = tmp_path/"download-corpus"; corpus.mkdir()
        with LedgerWriter(corpus/"corpus.sqlite3") as writer:
            with writer.transaction():
                document = writer.document(source_key(url))
            snapshot = DocumentSnapshot(document, None, url, ({"owner_source_id": "source-7", "title": "Original source title"},))
            specification = StageSpecification("download.acquire", (), ("download_payload",), settings={"url": url, "filename_base": "ABC_123", "sleep": 0})
            runner = SequentialStageRunner(writer, corpus)
            c, _ = runner.execute(DownloadStage(), snapshot, specification)
            state["body"] += b"new-revision"
            d, _ = runner.execute(DownloadStage(), snapshot, specification)
            assert c.status == d.status == "ok"
            versions = writer.connection.execute("SELECT source_sha256,stored_sha256 FROM document_versions WHERE document_id=? ORDER BY revision", (document,)).fetchall()
            assert len(versions) == 2 and versions[0][0] == versions[0][1] and versions[1][0] == versions[1][1]
            assert versions[0][1] != versions[1][1]
            assert writer.connection.execute("SELECT COUNT(*) FROM source_records").fetchone()[0] == 2
    finally:
        server.shutdown(); server.server_close(); worker.join()


def test_browser_acquisition_marks_unobservable_source_and_keeps_stored_hash(tmp_path):
    from glossapi.stages.receipts import capture_transport, acquired_receipt
    payload = tmp_path/"browser.pdf"; payload.write_bytes(b"%PDF-browser-synthesized")
    with capture_transport("browser", {}) as captured:
        acquired_receipt("https://example.org/viewer", payload, False)
    acquisition = captured["acquisitions"][0]
    assert acquisition["source_sha256"] is None and acquisition["source_entity_bytes"] is None
    assert acquisition["stored_sha256"] == sha256_file(payload)
    assert acquisition["source_hash_unavailable_reason"] == "original_entity_not_observable"


@pytest.mark.parametrize("outcome,reason", [("exception", "browser_recovery_failed"), ("html", "browser_returned_html"), ("unsupported", "browser_unsupported_format"), ("pdf", "browser_download_completed")])
def test_browser_adapter_preserves_errors_and_transformation_receipts(tmp_path, monkeypatch, outcome, reason):
    pytest.importorskip("aiofiles")
    from glossapi.gloss_browser_downloader import BrowserGlossDownloader
    async def recover_source(self, **kwargs):
        return None
    async def browser(self, **kwargs):
        if outcome == "exception":
            raise RuntimeError("Browser navigation failed")
        body, kind = {"html": (b"<html>blocked</html>", "text/html"),
            "unsupported": (b"PK\\003\\004binary", "application/zip"), "pdf": (b"%PDF-1.4 recovered", "application/pdf")}[outcome]
        suffix = "zip" if outcome == "unsupported" else "pdf"
        return body, {"Content-Type": kind}, {"candidate_url": "https://example.org/recovered."+suffix}
    async def preflight(self, **kwargs):
        return await self._recover_html_interstitial(**kwargs, headers={"Content-Type": "text/html"},
            content=b"<html>challenge</html>", html_issue="challenge page")
    monkeypatch.setattr(BrowserGlossDownloader, "_recover_source_specific_html_interstitial", recover_source)
    monkeypatch.setattr(BrowserGlossDownloader, "_download_via_browser_session", browser)
    monkeypatch.setattr(BrowserGlossDownloader, "_preflight_download", preflight)
    task = request(tmp_path, operation="download.acquire", url="https://example.org/viewer", filename_base="ABC_123", download_mode="browser", sleep=0)
    task.work_directory.mkdir()
    result = DownloadStage()._run_document(task, reporter=ReceiptCollector())
    assert result.reason_code == reason, asdict(result)
    assert result.status == ("ok" if outcome == "pdf" else "failed")
    observations = result.decisions[0]["backend_observations"]
    if outcome == "exception":
        assert observations[-1]["error_type"] == "builtins.RuntimeError"
        assert "Browser navigation failed" in observations[-1]["traceback"]
    if outcome == "pdf":
        assert result.decisions[0]["acquisitions"][-1]["source_sha256"] is None


def test_malformed_required_json_never_becomes_success(tmp_path, monkeypatch):
    source = simple_pdf(tmp_path/"source.pdf")
    task = request(tmp_path, source, required_output_kinds=["raw_markdown", "docling_json"])
    task.work_directory.mkdir()
    extractor = prepared_extractor()
    original = extractor._process_single_document
    def malformed(*args, **kwargs):
        result = original(*args, **kwargs)
        directory = task.work_directory/"json"; directory.mkdir()
        (directory/"source.docling.json").write_text("{invalid")
        return result
    monkeypatch.setattr(extractor, "_process_single_document", malformed)
    result = ExtractStage().run_prepared(task, extractor=extractor, reporter=ReceiptCollector())
    assert result.status == "failed" and result.reason_code == "output_schema_invalid"
    assert result.error and all(output.retention_class == "failed_attempt" for output in result.outputs)


@pytest.mark.parametrize("settings,device", [({"phase1_backend": "auto"}, "cpu"), ({"phase1_backend": "safe"}, "cuda:0"), ({"phase1_backend": "safe", "page_range": [1, 1]}, "cpu")])
def test_adapter_does_not_silently_change_backend_device_or_page_scope(tmp_path, settings, device):
    source = simple_pdf(tmp_path/"source.pdf")
    task = replace(request(tmp_path, source, **settings), device_id=device)
    result = ExtractStage().run_document(task, reporter=ReceiptCollector())
    assert result.status == "failed" and result.reason_code == "extract_configuration_invalid"
    assert not (task.work_directory/"markdown/source.md").exists()


def test_compatibility_profile_uses_single_writer_and_keeps_registered_generation(tmp_path):
    from glossapi.ledger.journal import shadow_stage
    from glossapi.stages.compatibility import extract_compatibility_document
    import sqlite3
    source = simple_pdf(tmp_path/"source.pdf")
    output = tmp_path/"corpus"; output.mkdir()
    corpus = SimpleNamespace(output_dir=output, input_dir=tmp_path, downloads_dir=output/"downloads",
        metadata_path=None, url_column="url", _shadow_ledger_enabled=True)
    @shadow_stage("extract", "extract.wrapper")
    def extract_one(self, file_paths):
        extractor = prepared_extractor()
        return extract_compatibility_document(extractor, file_paths[0], output/"markdown")
    assert extract_one(corpus, [source]) is True
    assert (output/"markdown/source.md").read_text() == "A small text layer"
    with ledger_connect(output/"corpus.sqlite3") as database:
        assert database.execute("SELECT status FROM current_task_state WHERE operation='extract.document'").fetchone()[0] == "ok"
        facts = database.execute("SELECT uri,sha256 FROM files WHERE uri LIKE '%/.glossapi/artifacts/%'").fetchall()
    from urllib.parse import unquote, urlparse
    assert facts
    for uri, digest in facts:
        assert sha256_file(Path(unquote(urlparse(uri).path))) == digest


def test_swallowed_export_exception_retains_type_and_traceback(tmp_path, monkeypatch):
    source = simple_pdf(tmp_path/"source.pdf")
    task = request(tmp_path, source); task.work_directory.mkdir()
    extractor = prepared_extractor(); extractor.use_pypdfium_backend = False
    class BrokenExport:
        def export_to_markdown(self):
            raise OSError("Export writer failed")
    conversion = SimpleNamespace(status="success", input=SimpleNamespace(file=source), document=BrokenExport(), pages=[])
    monkeypatch.setattr(extractor, "_convert_all_with_timeout", lambda *args, **kwargs: [conversion])
    result = ExtractStage().run_prepared(task, extractor=extractor, reporter=ReceiptCollector())
    assert result.status == "failed"
    errors = [item for item in result.decisions[0]["backend_observations"] if item["error_type"] == "builtins.OSError"]
    assert len(errors) == 1 and "Export writer failed" in errors[0]["traceback"]


def test_explicit_python_repair_does_not_load_native_backend(monkeypatch):
    from glossapi.pdf_repair import selector
    monkeypatch.setattr(selector, "import_module", lambda *args: pytest.fail("Explicit Python repair loaded native code"))
    assert selector.get_backend("python").name == "python"
    with pytest.raises(ValueError):
        selector.get_backend("unknown")


@pytest.mark.parametrize("options", [{"accel_type": "CUDA"}, {"accel_type": "CPU", "batched": True}, {"accel_type": "CPU", "use_gpus": "multi"}])
def test_sequential_profile_rejects_unsupported_routes_before_backend_setup(monkeypatch, options):
    from glossapi.corpus.phase_extract import ExtractPhaseMixin
    monkeypatch.setenv("GLOSSAPI_STAGE_PROFILE", "sequential-adapters")
    with pytest.raises(ValueError, match="Sequential adapter profile requires CPU"):
        ExtractPhaseMixin.extract.__wrapped__(object(), **options)


class WorkerWithoutResult:
    name = "extract"
    def _run_document(self, task, *, reporter):
        import os
        os._exit(0)


def test_worker_exit_without_a_result_is_failed_and_reaped(tmp_path):
    result = run_fenced(WorkerWithoutResult(), request(tmp_path), reporter=ReceiptCollector())
    assert result.status == "failed" and result.reason_code == "worker_process_crashed"


class TemporaryFileStage:
    name = "extract"
    def _run_document(self, task, *, reporter):
        import tempfile
        scratch = Path(tempfile.gettempdir())
        assert scratch == task.work_directory/"scratch"
        with tempfile.NamedTemporaryFile(delete=False) as stream:
            stream.write(b"native intermediate")
        return StageResult(task.task_id, task.attempt_id, task.lease_token, "ok", "extract_completed",
            outputs_in(task.work_directory), (), {})


def test_native_temporary_files_stay_inside_attempt_and_are_returned(tmp_path):
    task = request(tmp_path)
    result = run_fenced(TemporaryFileStage(), task, reporter=ReceiptCollector())
    assert result.status == "ok"
    native = [output for output in result.outputs if output.relative_path.startswith("scratch/")]
    assert len(native) == 1
    assert (task.work_directory/native[0].relative_path).read_bytes() == b"native intermediate"
