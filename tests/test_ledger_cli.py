"""Read commands never discover legacy stores or open document payloads."""

from glossapi.ledger.compact import connect as ledger_connect
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tomllib

import pytest

from glossapi.cli import main
from glossapi.ledger import LedgerWriter, open_readonly
from glossapi.ledger.encoding import snapshot_hash


def seed(root):
    with LedgerWriter(root/"corpus.sqlite3") as writer, writer.transaction():
        run = writer.run({}, {}, {}, code_revision="test")
        document = writer.document("local_path:v1:file:///gone/source.pdf")
        version = writer.version(document, source_sha256="a"*64, stored_sha256="b"*64)
        writer.source_record(document, run, "file:///gone/source.pdf", dict(original="retained", needs_ocr=True))
        task = writer.task(run, "extract", "extract.test", {}, version_id=version)
        attempt = writer.attempt(task, worker_id="worker", device_id="cpu", fingerprint={}, inputs={}, settings={}, tools={})
        writer.finish(run, task, attempt, status="failed", reason_code="pdf_open_failed", error_type="ValueError", message="bad PDF", traceback="original traceback")
        writer.artifact(run, "extract", "markdown", "file:///gone/source.md", sha256="c"*64, bytes=29, version_id=version, ownership="legacy_external", presence="missing")
    return run, document, version


def invoke(capsys, *args):
    code = main([*args, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("command", ["plan", "status", "report"])
def test_readonly_commands_create_no_files_even_on_uninitialized_corpus(tmp_path, capsys, command):
    nonexistent = tmp_path/"absent"
    code, result = invoke(capsys, command, "--corpus", str(nonexistent))
    assert code == (0 if command == "plan" else 2)
    assert result["corpus_id"] is None
    assert list(tmp_path.iterdir()) == []
    if command != "plan":
        assert result["errors"][0]["code"] == "ledger_not_initialized"


@pytest.mark.parametrize("command", ["plan", "status", "report"])
def test_closed_wal_ledger_reads_leave_entire_directory_unchanged(tmp_path, capsys, command):
    seed(tmp_path)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    code, result = invoke(capsys, command, "--corpus", str(tmp_path))
    assert code == (1 if command == "status" else 0)
    # status exit 1 names the seeded failed task as a typed error (INTEG-G2)
    assert [error["code"] for error in result["errors"]] == (["pdf_open_failed"] if command == "status" else [])
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}


def test_why_with_all_payloads_absent_recovers_hashes_errors_and_source_fields(tmp_path, capsys):
    run, document, version = seed(tmp_path)
    code, result = invoke(capsys, "report", "--corpus", str(tmp_path), "--why", "source.pdf")
    assert code == 0 and result["run_id"] == run
    data = result["data"]
    assert data["document"]["document_id"] == document
    assert data["document"]["version_id"] == version
    assert data["files"][0]["sha256"] == "c"*64 and data["files"][0]["presence"] == "missing"
    attempt = data["stages"][0]["attempts"][0]
    assert attempt["error"]["traceback"] == "original traceback"
    assert "retained" in json.dumps(data["document"]["source_records"])
    assert data["unknowns"]


def test_ambiguous_filename_returns_all_candidates_and_unknown_selector_is_structured(tmp_path, capsys):
    _, document, _ = seed(tmp_path)
    with LedgerWriter(tmp_path/"corpus.sqlite3") as writer, writer.transaction():
        run = writer.run({}, {}, {}, code_revision="test")
        other = writer.document("local_path:v1:file:///other/source.pdf")
        writer.version(other)
        writer.source_record(other, run, "file:///other/source.pdf", {})
    code, result = invoke(capsys, "report", "--corpus", str(tmp_path), "--why", "source.pdf")
    assert code == 2
    assert result["errors"][0]["details"]["candidate_document_ids"] == sorted([document, other])
    assert invoke(capsys, "report", "--corpus", str(tmp_path), "--why", document)[0] == 0
    code, result = invoke(capsys, "report", "--corpus", str(tmp_path), "--why", "unknown")
    assert code == 2 and result["errors"][0]["code"] == "file_not_found"


def test_plan_hash_prerequisite_closure_and_unknown_routes_without_reuse(tmp_path, capsys):
    seed(tmp_path)
    # K3: an unsupported device for a stage now fails the plan (test_k3_interface); extract runs on cpu only.
    code, result = invoke(capsys, "plan", "--corpus", str(tmp_path), "--stages", "annotate", "--workers", "clean=3", "--device", "extract=cpu")
    assert code == 0
    body = result["data"]; digest = body.pop("plan_hash")
    assert snapshot_hash(body) == digest
    assert [s["stage"] for s in body["documents"][0]["steps"]] == ["extract", "clean", "sections", "annotate"]
    assert all(s["action"] == "conditional" for s in body["documents"][0]["steps"])
    assert next(p for p in body["pools"] if p["stage"] == "clean")["queue_capacity"] == 6
    assert body["disk"]["estimated_peak_bytes"] is None
    assert body["execution"]["available"] is True and body["execution"]["profile"] == "streaming_document_dag"
    assert "execution_available" not in body and "compatibility_profile" not in body
    assert all(step["executable_by_run"] for step in body["documents"][0]["steps"])


def test_new_plan_ids_are_proposals_and_settings_precedence_is_hashed(tmp_path, capsys):
    settings = tmp_path/"settings.json"; settings.write_text('{"debug":"off","disk_budget_bytes":123}')
    code, result = invoke(capsys, "plan", "--corpus", str(tmp_path/"absent"), "--files", "/absent.pdf", "https://EXAMPLE.org/A#fragment", "--settings", str(settings), "--debug", "on")
    assert code == 0
    assert result["data"]["settings"]["debug"] == "on"
    assert len(result["data"]["documents"]) == 2
    assert not (tmp_path/"absent").exists()


@pytest.mark.parametrize("arguments,exit_code", [(["plan", "--workers", "clean=0"], 2), (["plan", "--stages", "bogus"], 2),
    (["plan", "--device", "clean=bogus"], 2), (["plan", "--memory-saving", "extreme", "--stages", "ocr"], 3),
    (["report", "--why", "x", "--format", "parquet"], 2), (["report", "--format", "parquet"], 2), (["status", "--nonsense"], 2)])
def test_invalid_requests_are_versioned_json(tmp_path, capsys, arguments, exit_code):
    seed(tmp_path)
    code, result = invoke(capsys, *arguments, "--corpus", str(tmp_path))
    assert code == exit_code and result["errors"] and result["schema_version"] == 1


def test_only_explicit_output_writes_and_parquet_is_flattened(tmp_path, capsys):
    seed(tmp_path)
    target = tmp_path/"report.parquet"
    code, result = invoke(capsys, "report", "--corpus", str(tmp_path), "--format", "parquet", "--output", str(target))
    assert code == 0 and result["data"]["rows"] == 1
    import pyarrow.parquet as pq
    assert pq.read_table(target).to_pylist()[0]["status"] == "failed"
    assert not list(tmp_path.glob(".report*"))


def test_no_optional_imports_and_console_entry(tmp_path):
    seed(tmp_path)
    code = "import sys; from glossapi.cli import main; main(['report','--corpus',sys.argv[1],'--json']); assert not any(n in sys.modules for n in ('pandas','numpy','pyarrow','torch','docling','sklearn','pypdfium2','glossapi.corpus.corpus_orchestrator'))"
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    project = Path(__file__).resolve().parents[1]/"pyproject.toml"
    assert tomllib.loads(project.read_text())["project"]["scripts"]["glossapi"] == "glossapi.cli:main"


def test_live_wal_ledger_snapshot_reads_committed_events_without_new_files(tmp_path, capsys):
    seed(tmp_path)
    with LedgerWriter(tmp_path/"corpus.sqlite3") as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision="live")
        before = {p.name for p in tmp_path.iterdir()}
        code, result = invoke(capsys, "status", "--corpus", str(tmp_path))
        assert code == 0 and result["run_id"] == run
        assert {p.name for p in tmp_path.iterdir()} == before


def test_unsupported_schema_fails_closed(tmp_path, capsys):
    seed(tmp_path)
    with ledger_connect(tmp_path/"corpus.sqlite3") as connection:
        connection.execute("PRAGMA user_version=99")
    code, result = invoke(capsys, "status", "--corpus", str(tmp_path))
    assert code == 2 and result["errors"][0]["code"] == "unsupported_schema"


def test_why_historical_run_excludes_later_versions_and_presence_changes(tmp_path, capsys):
    run, document, version = seed(tmp_path)
    with LedgerWriter(tmp_path/"corpus.sqlite3") as writer, writer.transaction():
        later = writer.run({}, {}, {}, code_revision="later")
        new_version = writer.version(document, source_sha256="d"*64, stored_sha256="e"*64)
        writer.artifact(later, "extract", "markdown", "file:///gone/new.md", sha256="f"*64, bytes=8, version_id=new_version, ownership="legacy_external")
    code, result = invoke(capsys, "report", "--corpus", str(tmp_path), "--why", document, "--run", run)
    assert code == 0 and result["data"]["document"]["version_id"] == version
    assert all(f["version_id"] == version for f in result["data"]["files"])


def test_orphaned_wal_fails_without_creating_shm(tmp_path, capsys):
    seed(tmp_path)
    wal = tmp_path/"corpus.sqlite3-wal"; wal.write_bytes(b"orphan")
    before = {p.name for p in tmp_path.iterdir()}
    code, result = invoke(capsys, "report", "--corpus", str(tmp_path))
    assert code == 2 and result["errors"]
    assert before == {p.name for p in tmp_path.iterdir()}


def test_why_retains_decision_error_type_and_traceback_after_success(tmp_path, capsys):
    run, document, version=seed(tmp_path)
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer, writer.transaction():
        task=writer.task(run,'extract','extract.fallback',{},version_id=version)
        attempt=writer.attempt(task,worker_id='worker',device_id='cpu',fingerprint={},inputs={},settings={},tools={})
        writer.event(run,event_kind='decision',reason_code='font_repair_error',task_id=task,version_id=version,attempt_id=attempt,
                     error_type='ValueError',message='encrypted PDF',traceback='original repair traceback',details=dict(fallback='original_pdf'))
        writer.finish(run,task,attempt,status='ok',reason_code='extract_completed')
    code,result=invoke(capsys,'report','--corpus',str(tmp_path),'--why',document)
    assert code==0
    stage=next(s for s in result['data']['stages'] if s['operation']=='extract.fallback')
    decision=stage['attempts'][0]['decisions'][0]
    assert decision['error_type']=='ValueError' and decision['traceback']=='original repair traceback'
    assert stage['status']=='ok' and decision['details']['fallback']=='original_pdf'


def test_new_remote_plan_includes_acquisition_prerequisite_but_local_plan_does_not(tmp_path, capsys):
    code,result=invoke(capsys,'plan','--corpus',str(tmp_path),'--stages','extract','--files','https://example.org/a','/explicit/local.pdf')
    assert code==0
    remote,local=result['data']['documents']
    assert [s['stage'] for s in remote['steps']]==['download','extract']
    assert remote['steps'][1]['input_roles']==[dict(producer='download',role='stage_input')]
    assert [s['stage'] for s in local['steps']]==['extract']
    assert [b['name'] for b in result['data']['barriers']]==['source_manifest','download_publication','extract_publication']  # every executed barrier (W18 M13)


def test_imported_remote_source_version_is_visible_without_stage_or_artifact_facts(tmp_path, capsys):
    from glossapi.ledger import import_corpus
    (tmp_path/'source.json').write_text('[{"url":"https://example.org/not-yet-downloaded"}]')
    import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        document,version=connection.execute('SELECT document_id,version_id FROM document_versions').fetchone()
    code,result=invoke(capsys,'report','--corpus',str(tmp_path),'--why',document)
    assert code==0 and result['data']['document']['version_id']==version
    assert result['data']['document']['versions'][0]['stored_sha256'] is None


def test_canonical_url_selectors_reuse_identity_and_preserve_explicit_record_ambiguity(tmp_path, capsys):
    from glossapi.ledger.identity import source_key
    uri='HTTPS://EXAMPLE.org/A?b=2&a=1#old'
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer, writer.transaction():
        run=writer.run({}, {}, {}, code_revision='test')
        # CP-LEDGER cycle 2: a query/fragment URL is keyed under the corpus key,
        # so its identity is computed while the corpus writer is open.
        key=source_key(uri)
        document=writer.document(key);version=writer.version(document)
        writer.source_record(document,run,uri,{},source_kind='url')
        writer.event(run,event_kind='import',reason_code='legacy_observed',version_id=version)
    selector='https://example.org/A?b=2&a=1#new'
    code,result=invoke(capsys,'plan','--corpus',str(tmp_path),'--files',selector)
    assert code==0 and result['data']['documents'][0]['document_id']==document
    assert invoke(capsys,'report','--corpus',str(tmp_path),'--why',selector)[0]==0
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer, writer.transaction():
        other=writer.document(source_key(uri,dataset_id='dataset',source_id='independent'))
        writer.version(other);writer.source_record(other,run,selector,{},source_kind='url')
    code,result=invoke(capsys,'report','--corpus',str(tmp_path),'--why',uri)
    assert code==2 and result['errors'][0]['details']['candidate_document_ids']==sorted([document,other])
    assert invoke(capsys,'report','--corpus',str(tmp_path),'--why',key)[0]==0
