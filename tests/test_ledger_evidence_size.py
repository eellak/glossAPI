"""Storage regressions: delta replay must retain evidence without corpus copies."""
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from glossapi.ledger import LedgerWriter
from glossapi.ledger.encoding import untyped_value
from glossapi.ledger.journal import ResultJournal, shadow_stage, observe_store
from glossapi.ledger.stores import store_states


def context(tmp_path):
    output = tmp_path/"out"; output.mkdir()
    return SimpleNamespace(input_dir=tmp_path/"empty", output_dir=output,
                           downloads_dir=output/"downloads", _shadow_ledger_enabled=True)


def deltas(connection):
    return [json.loads(row[0]) for row in connection.execute(
        "SELECT json FROM snapshots WHERE kind='legacy_store' ORDER BY rowid")]


def test_changed_rows_survive_reopen_deletion_reordering_and_schema_change(tmp_path):
    corpus = context(tmp_path)
    path = corpus.output_dir/"metadata.parquet"
    pd.DataFrame({"filename": ["a.pdf", "b.pdf"], "score": [1, 2]}).to_parquet(path)

    @shadow_stage("clean", "clean.delta_fixture")
    def wrapper(self, table):
        pd.DataFrame(table).to_parquet(path)
        observe_store(path)
        observe_store(path)  # No second evidence copy for unchanged bytes.

    wrapper(corpus, {"filename": ["a.pdf", "b.pdf"], "score": [1, 3]})
    wrapper(corpus, {"filename": ["b.pdf", "c.pdf"], "score": [3, 4], "new": [True, False]})
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        history = deltas(writer.connection)
        assert history[0]["mode"] == "first_import" and len(history[0]["changed"]) == 2
        assert len(history) == 3
        assert len(history[1]["changed"]) == 1 and not history[1]["deleted"]
        assert len(history[2]["changed"]) == 2 and len(history[2]["deleted"]) == 1
        assert history[2]["previous_sha256"] == writer.connection.execute(
            "SELECT sha256 FROM snapshots WHERE kind='legacy_store' ORDER BY rowid LIMIT 1 OFFSET 1").fetchone()[0]
        state = store_states(writer)[path.as_uri()]
        assert len(state["rows"]) == history[-1]["row_count"] == 2
        assert state["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert history[-1]["bytes"] == path.stat().st_size
        assert sorted(row["values"] for row in state["rows"].values()) == [["b.pdf", 3, True], ["c.pdf", 4, False]]


def test_200_incremental_writes_store_200_rows_not_20100(tmp_path):
    corpus = context(tmp_path); path = corpus.output_dir/"state.parquet"
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        journal = ResultJournal(writer, corpus, "extract", "extract.scale_fixture", {})
        for size in range(1, 201):
            pd.DataFrame({"filename": [f"{i}.pdf" for i in range(size)], "flag": [True]*size}).to_parquet(path)
            journal.store(path, kind="metadata")
        history = deltas(writer.connection)
        assert len(history) == 200
        assert sum(len(delta["changed"]) for delta in history) == 200
        assert all(delta["mode"] == "delta" and not delta["deleted"] for delta in history)
        assert [delta["row_count"] for delta in history] == list(range(1, 201))


def test_duplicate_row_keys_text_hashes_and_explanatory_values(tmp_path):
    corpus = context(tmp_path); path = corpus.output_dir/"sections.parquet"
    from decimal import Decimal
    frame = pd.DataFrame({"filename": ["same", "same"], "id": [1, 1],
        "text": ["very private body", "other private body"], "title": ["private heading"]*2,
        "model_label": ["header", "text"], "reason": ["rule_1", "fallback"],
        "large_id": [2**60+1]*2, "decimal": [Decimal("1.25")]*2})
    frame["section"] = "unabridged section body"
    frame["json_section"] = '{"content":"duplicated section body"}'
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        journal = ResultJournal(writer, corpus, "sections", "sections.duplicates", {})
        frame.to_parquet(path); journal.store(path, kind="sections")
        first = deltas(writer.connection)[0]
        assert len(first["changed"]) == 2
        assert "private body" not in json.dumps(first) and "private heading" not in json.dumps(first)
        assert "section body" not in json.dumps(first)
        rows = list(first["changed"].values())
        assert {row["values"][4] for row in rows} == {"header", "text"}
        header = next(row for row in rows if row["values"][4] == "header")
        assert header["values"][4:7] == ["header", "rule_1", 2**60+1]
        assert untyped_value(header["values"][7]["typed"]) == Decimal("1.25")
        frame.loc[1, "reason"] = "changed_rule"
        frame.to_parquet(path); journal.store(path, kind="sections")
        assert len(deltas(writer.connection)[1]["changed"]) == 1


def test_json_key_deltas_retain_measurements_after_payload_deletion(tmp_path):
    corpus = context(tmp_path); path = corpus.output_dir/"report.json"
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        journal = ResultJournal(writer, corpus, "clean", "clean.json", {})
        path.write_text(json.dumps({"a.pdf": {"score": 1, "text": "payload"}}))
        journal.store(path, kind="report")
        path.write_text(json.dumps({"a.pdf": {"score": 1, "text": "payload"}, "b.pdf": {"score": 2}}))
        journal.store(path, kind="report"); path.unlink()
        history = deltas(writer.connection)
        assert list(history[1]["changed"]) == ['"b.pdf"']
        assert store_states(writer)[path.as_uri()]["rows"]['"a.pdf"']["score"] == 1
        assert "payload" not in json.dumps(history)


def test_unreadable_store_keeps_prior_evidence_and_reports_unknown_count(tmp_path):
    corpus = context(tmp_path); path = corpus.output_dir/"report.json"
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        journal = ResultJournal(writer, corpus, "clean", "clean.bad_json", {})
        path.write_text('{"score":3}'); journal.store(path, kind="report")
        path.write_text('{'); journal.store(path, kind="report")
        last = deltas(writer.connection)[-1]
        assert last["row_count"] is None and last["row_count_unavailable_reason"] == "JSONDecodeError"
        assert not last["changed"] and not last["deleted"]
        assert store_states(writer)[path.as_uri()]["rows"] == {'"score"': 3}


def test_rolled_back_observation_never_becomes_a_delta_base(tmp_path, monkeypatch):
    corpus = context(tmp_path); path = corpus.output_dir/"report.json"
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        journal = ResultJournal(writer, corpus, "clean", "clean.rollback", {})
        path.write_text('{"score":3}')
        original = writer.event
        def fail_import(*args, **kwargs):
            if kwargs.get("event_kind") == "import":
                raise RuntimeError("rollback fixture")
            return original(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(writer, "event", fail_import)
            with pytest.raises(RuntimeError, match="rollback fixture"):
                journal.store(path, kind="report")
        assert not deltas(writer.connection) and not store_states(writer)
        journal.store(path, kind="report")
        assert deltas(writer.connection)[0]["previous_sha256"] is None


def test_source_manifest_observes_original_metadata_once_without_second_row_copy(tmp_path):
    corpus = context(tmp_path); path = tmp_path/"sources.parquet"
    frame = pd.DataFrame({"url": ["https://example/a"]*2, "body": ["original source metadata"]*2})
    frame.to_parquet(path)
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        for _ in range(2):
            journal = ResultJournal(writer, corpus, "download", "download.manifest", {})
            journal.source_rows(frame, path, url_column="url")
            # The original in-memory frame can outlive an overwritten manifest.
            # Preserve its original binding without copying source bodies again.
            pd.DataFrame({"changed_legacy_file": [True]}).to_parquet(path)
        assert writer.connection.execute("SELECT COUNT(*) FROM source_records").fetchone()[0] == 2
        manifest = json.loads(writer.connection.execute("SELECT json FROM snapshots WHERE kind='source_manifest'").fetchone()[0])
        assert manifest["row_count"] == 2 and "rows" not in manifest
        assert "original source metadata" not in json.dumps(manifest)
        assert all(untyped_value(json.loads(row[0])["fields"])["body"] == "original source metadata"
                   for row in writer.connection.execute("SELECT metadata_json FROM source_records"))
        journal.source_rows(frame, path, url_column="body")
        assert writer.connection.execute("SELECT COUNT(*) FROM source_records").fetchone()[0] == 4


def test_whole_corpus_comparator_rejects_return_only_and_product_mutations():
    from copy import deepcopy
    from tools.harness.corpus import compare_arms
    from tools.harness.common import STAGES
    baseline = {"stages": {stage: dict(status="ok", error=None, wrapper_return=None,
        products={"x": {"sha256": "a"*64}}, removed_files=[]) for stage in STAGES}}
    for field, mutation in (("wrapper_return", "changed"), ("products", {}), ("removed_files", ["x"])):
        candidate = deepcopy(baseline); candidate["stages"]["extract"][field] = mutation
        assert not compare_arms(baseline, candidate)["passes"]


def test_repeated_wrapper_inputs_share_facts_but_each_attempt_binds_inputs(tmp_path):
    corpus = context(tmp_path)
    sources = [tmp_path/f"{i}.pdf" for i in range(3)]
    for path in sources: path.write_bytes(b"source")
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        for _ in range(5):
            journal = ResultJournal(writer, corpus, "extract", "extract.shared_inputs", {"file_paths": sources})
            journal.finish(None, None)
        assert writer.connection.execute("SELECT COUNT(*) FROM files WHERE kind='initial_input'").fetchone()[0] == 3
        assert writer.connection.execute("SELECT COUNT(*) FROM source_records").fetchone()[0] == 3
        assert writer.connection.execute("SELECT COUNT(*) FROM attempt_inputs").fetchone()[0] == 15
        sources[0].write_bytes(b"changed source")
        journal = ResultJournal(writer, corpus, "extract", "extract.shared_inputs", {"file_paths": sources})
        journal.finish(None, None)
        assert writer.connection.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 4
        assert writer.connection.execute("SELECT COUNT(*) FROM files WHERE kind='initial_input'").fetchone()[0] == 4
        assert writer.connection.execute("SELECT COUNT(*) FROM attempt_inputs").fetchone()[0] == 18


def test_diagnostic_tables_preserve_every_field_and_escape_marker_collisions():
    from glossapi.ledger.encoding import pack_evidence, unpack_evidence
    original = {"kind": "branch_evidence", "font_analysis": {"fonts": [
        dict(object_id=i, name=f"font-{i}", reason="An exact explanatory reason repeated for every font",
             codes=i % 17, repair_possible=False, error=None,
             nested={"_table": {"_object": [None, True, 2, "literal"]}}) for i in range(1000)]},
        "typed": {"type": "mapping", "value": [[{"type": "str", "value": "x"}, {"type": "int", "value": "9007199254740993"}]]}}
    packed = pack_evidence(original)
    assert packed["encoding"] == "columnar_json_v1"
    assert unpack_evidence(packed) == original
    assert len(json.dumps(packed)) < len(json.dumps(original)) * 0.65
    small = {"kind": "branch_evidence", "message": "small unchanged object"}
    assert unpack_evidence(pack_evidence(small)) == small


def test_worker_store_rows_are_sealed_as_deltas_even_after_payload_disappears(tmp_path, monkeypatch):
    from glossapi.ledger import journal as hooks
    corpus = context(tmp_path); path = corpus.output_dir/"worker.json"
    path.write_text('{"a.pdf":{"score":1}}')
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        parent = ResultJournal(writer, corpus, "clean", "clean.worker_stores", {})
        collector = hooks.WorkerObservations()
        monkeypatch.setattr(hooks, "LedgerWriter", lambda *a, **k: pytest.fail("worker opened SQLite"))
        path.write_text('{"a.pdf":{"score":1},"b.pdf":{"score":2}}')
        collector.store(path, kind="worker_report")
        path.write_text('{"a.pdf":{"score":3},"b.pdf":{"score":2}}')
        collector.store(path, kind="worker_report")
        path.unlink()
        forwarded = json.loads(json.dumps(collector.receipts))
        token = hooks._active_journal.set(parent)
        try:
            hooks.accept_worker_observations(forwarded)
        finally:
            hooks._active_journal.reset(token)
        parent.finish(None, None)
        history = deltas(writer.connection)
        assert [len(d["changed"]) for d in history] == [1, 1, 1]
        assert history[0]["mode"] == "first_import"
        assert store_states(writer)[path.as_uri()]["rows"] == {'"a.pdf"': {"score": 3}, '"b.pdf"': {"score": 2}}
        assert writer.connection.execute("SELECT presence FROM current_file_state WHERE kind='worker_report' ORDER BY event_id DESC LIMIT 1").fetchone()[0] == "missing"


def test_final_inventory_does_not_duplicate_atomic_output_registration(tmp_path):
    from glossapi.ledger import open_readonly
    corpus = context(tmp_path)
    @shadow_stage("clean", "clean.final_outputs")
    def wrapper(self):
        (self.output_dir/"result.md").write_text("unchanged output")
        (self.output_dir/"report.json").write_text('{"score":3}')
    wrapper(corpus)
    with open_readonly(corpus.output_dir/"corpus.sqlite3") as read:
        facts = read.execute("SELECT file_id,producer_attempt_id FROM files").fetchall()
        assert len(facts) == 2 and all(row[1] for row in facts)
        assert read.execute("SELECT COUNT(*) FROM events WHERE event_kind='artifact'").fetchone()[0] == 2
        terminal = read.execute("SELECT attempt_id,output_manifest_sha256 FROM events WHERE status='ok'").fetchone()
        outputs = json.loads(read.execute("SELECT json FROM snapshots WHERE sha256=?", (terminal[1],)).fetchone()[0])["files"]
        assert {row[0] for row in facts} == {output["file_id"] for output in outputs}
        assert {row[1] for row in facts} == {terminal[0]}
        assert len(deltas(read)) == 1
    wrapper(corpus)
    with open_readonly(corpus.output_dir/"corpus.sqlite3") as read:
        assert read.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 2
        details = json.loads(read.execute("SELECT details_json FROM events WHERE status='ok' ORDER BY event_id DESC LIMIT 1").fetchone()[0])
        assert len(details["observed_files"]) == 2
        assert all(set(ref) == {"file_id"} for ref in details["observed_files"])


@pytest.mark.parametrize("format", ["nested-json", "fatal-lines"])
def test_nested_and_line_stores_grow_by_changed_rows_only(tmp_path, format):
    from glossapi.ledger.journal import payload_evidence
    from glossapi.ledger.stores import restore_json_store
    corpus = context(tmp_path)
    path = corpus.output_dir/("pages.json" if format == "nested-json" else "fatal_skip.txt")
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        journal = ResultJournal(writer, corpus, "ocr", "ocr.collection_growth", {})
        for size in range(1, 201):
            if format == "nested-json":
                data = {"page_count": 200, "pages": {str(i): {"text": "private page text", "run_id": str(i)} for i in range(size)}}
                path.write_text(json.dumps(data))
            else:
                lines = [f"stem_{i:03}" for i in range(200-size, 200)]
                path.write_text("\n".join(lines)+"\n")
            journal.store(path, kind="ocr_page_state" if format == "nested-json" else "fatal_skiplist")
        history = deltas(writer.connection)
        assert len(history) == 200
        assert sum(len(d["changed"]) for d in history) == (201 if format == "nested-json" else 200)
        state = store_states(writer)[path.as_uri()]
        schema = json.loads(writer.connection.execute("SELECT json FROM snapshots WHERE sha256=?", (state["schema_sha256"],)).fetchone()[0])
        if format == "nested-json":
            assert restore_json_store(schema, state["rows"]) == payload_evidence(data)
            assert "private page text" not in json.dumps(history)
            data["pages"]["42"]["run_id"] = "updated"
            del data["pages"]["43"]
            path.write_text(json.dumps(data)); journal.store(path, kind="ocr_page_state")
            last = deltas(writer.connection)[-1]
            assert len(last["changed"]) == len(last["deleted"]) == 1
            assert restore_json_store(schema, store_states(writer)[path.as_uri()]["rows"]) == payload_evidence(data)
        else:
            assert schema["order"] == "sorted"
            assert sorted(state["rows"].values()) == lines


def test_fatal_membership_branch_evidence_is_delta_recoverable_and_rollback_safe(tmp_path):
    from glossapi.ledger.stores import membership_snapshot, membership_values
    corpus = context(tmp_path); path = corpus.output_dir/"fatal_skip.txt"
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        journal = ResultJournal(writer, corpus, "extract", "extract.fatal_members", {})
        for size in range(1, 201):
            values = [f"stem_{i:03}" for i in range(size)]
            journal.observe("legacy_observed", path=path, store="fatal_skiplist", stems=values)
        history = [json.loads(row[0]) for row in writer.connection.execute("SELECT json FROM snapshots WHERE kind='membership_delta' ORDER BY rowid")]
        assert len(history) == sum(len(d["added"]) for d in history) == 200
        assert all(not d["removed"] for d in history)
        last = json.loads(writer.connection.execute("SELECT json FROM snapshots WHERE kind='branch_evidence' ORDER BY rowid DESC LIMIT 1").fetchone()[0])
        assert "stems" not in last and membership_values(writer.connection, last["shared_fields"]["stems"]) == values
        # Reobserving an earlier set reuses its value snapshot rather than
        # writing all its members again or forking the replay chain.
        for count in (100, 150, 100, 200):
            journal.observe("legacy_observed", path=path, store="fatal_skiplist", stems=values[:count])
        assert writer.connection.execute("SELECT COUNT(*) FROM snapshots WHERE kind='membership_delta'").fetchone()[0] == 200
        with pytest.raises(RuntimeError):
            with writer.transaction():
                membership_snapshot(writer, "rollback", ["x"])
                raise RuntimeError("rollback")
        with writer.transaction():
            digest = membership_snapshot(writer, "rollback", ["x"])
        delta = json.loads(writer.connection.execute("SELECT json FROM snapshots WHERE sha256=?", (digest,)).fetchone()[0])
        assert delta["previous_sha256"] is None
        assert membership_values(writer.connection, digest) == ["x"]


def test_sorted_metadata_insertions_do_not_rewrite_shifted_positional_indices(tmp_path):
    corpus = context(tmp_path); path = corpus.output_dir/"sorted.parquet"
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        journal = ResultJournal(writer, corpus, "download", "download.sorted_rows", {})
        for size in range(1, 201):
            names = [f"file_{i:03}.pdf" for i in range(200-size, 200)]
            # Materialize the unnamed positional index too: Arrow may restore
            # it as an integer Index rather than the RangeIndex subclass.
            pd.DataFrame({"filename": names, "flag": [True]*size}).to_parquet(path, index=True)
            journal.store(path, kind="metadata")
        history = deltas(writer.connection)
        assert len(history) == sum(len(d["changed"]) for d in history) == 200
        state = store_states(writer)[path.as_uri()]
        assert all("index" not in row for row in state["rows"].values())
        assert sorted(row["values"][0] for row in state["rows"].values()) == names
        # A genuine index label remains explanatory data and must survive.
        pd.DataFrame({"filename": ["labelled.pdf"], "flag": [True]}, index=pd.Index([123], name="source_row")).to_parquet(path)
        journal.store(path, kind="metadata")
        row = next(iter(store_states(writer)[path.as_uri()]["rows"].values()))
        assert row["index"] == 123


@pytest.mark.parametrize("identity", ["base", "url", "content", "duplicate"])
def test_pending_download_rows_never_share_a_null_or_positional_identity(tmp_path, identity):
    corpus = context(tmp_path); path = corpus.output_dir/"pending.parquet"
    frame = pd.DataFrame({"filename": pd.Series([pd.NA]*200, dtype="string"),
        "source_row": pd.Series([pd.NA]*200, dtype="Int64"),
        "row_id": pd.Series([pd.NA]*200, dtype="string"),
        "slot": list(range(200)), "flag": [False]*200})
    if identity == "base": frame["filename_base"] = [f"doc_{i:03}" for i in range(200)]
    if identity == "url": frame["url"] = [f"https://example/{i:03}" for i in range(200)]
    if identity == "duplicate": frame["id"] = [0]*200
    with LedgerWriter(corpus.output_dir/"corpus.sqlite3") as writer:
        journal = ResultJournal(writer, corpus, "download", "download.pending", {})
        frame.to_parquet(path, index=False); journal.store(path, kind="metadata")
        order = reversed(range(200)) if identity in ("content", "duplicate") else range(200)
        for slot in order:
            mask = frame["slot"] == slot
            frame.loc[mask, "flag"] = True
            if identity in ("base", "url"): frame.loc[mask, "filename"] = f"doc_{slot:03}.pdf"
            frame = frame.sort_values(["flag", "slot"], ascending=[False, True]).reset_index(drop=True)
            frame.to_parquet(path, index=False); journal.store(path, kind="metadata")
        history = deltas(writer.connection)
        assert len(history) == 201
        assert sum(len(d["changed"]) for d in history) == 400
        assert sum(len(d["deleted"]) for d in history) == (200 if identity in ("content", "duplicate") else 0)
        assert len(store_states(writer)[path.as_uri()]["rows"]) == 200
