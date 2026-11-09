"""Legacy inventory, conservative identity and atomic shadow import contracts."""
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from glossapi.ledger import LedgerError, open_readonly
from glossapi.ledger.encoding import snapshot_hash, untyped_value
from glossapi.ledger.identity import source_key
from glossapi.ledger.importer import import_corpus, inventory_corpus
from glossapi.ledger.reader import snapshot


def rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def test_url_identity_preserves_meaningful_bytes_and_explicit_namespaces():
    # Public URLs keep url:v1 with their query (the url:v2 keyed form was withdrawn with the credential channel, owner
    # 1 Nov 2026). Imported legacy URLs are caller-owned rows, recorded as given.
    assert source_key("HTTPS://User:Pass@EXAMPLE.COM/A%2fb?b=2&a=1#x") == "url:v1:https://User:Pass@example.com/A%2fb?b=2&a=1"
    assert source_key("https://EXAMPLE.com/a?") != source_key("https://example.com/a")
    assert source_key("https://example.com/A") != source_key("https://example.com/a")
    assert source_key("https://example.com/a?b=1&a=2") != source_key("https://example.com/a?a=2&b=1")
    assert source_key("https://x", dataset_id="a", source_id="1") != source_key("https://x", dataset_id="b", source_id="1")


def test_all_parquets_duplicate_occurrences_lost_columns_and_conflicting_flags(tmp_path):
    original = [dict(url="https://EXAMPLE.org/A#x", source_row=73, filename="a.pdf", download_success=True,
                     raw_bytes=b"\x00\xff", wide_integer=2**62+19, scraping="original")]*2
    rows(tmp_path/"input.parquet", original)
    rows(tmp_path/"download_results/named.parquet", [dict(url="https://example.org/A", filename="a.pdf", download_success=False, needs_ocr=True)])
    rows(tmp_path/"download_results/final.parquet", [dict(url="https://example.org/A", filename="a.pdf", processing_stage="extract,clean")])
    rows(tmp_path/"download_results/partial.parquet", [dict(filename="a.pdf", filter="ok")])
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.parquet")}
    first = import_corpus(tmp_path)
    second = import_corpus(tmp_path)
    assert second["already_imported"] and first["run_id"] == second["run_id"]
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        records = connection.execute("SELECT source_record_id,metadata_json FROM source_records").fetchall()
        originals = [untyped_value(untyped_value(json.loads(metadata)["fields"])["original_fields"]) for _, metadata in records if 'scraping' in metadata]
        assert originals == original
        assert len({identifier for identifier, _ in records}) == len(records)
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    assert any(c["code"] == "conflicting_store_claims" and c["field"] == "download_success" for c in first["conflicts"])
    assert any(c["code"] == "source_identity_unresolved" for c in first["conflicts"])
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*.parquet")}


def test_rename_duplicate_content_and_changed_source_version(tmp_path):
    old = tmp_path/"old.pdf"; old.write_bytes(b"%PDF same")
    first = import_corpus(tmp_path)
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        document_id, key = connection.execute("SELECT document_id,source_key FROM documents").fetchone()
    old.rename(tmp_path/"new.pdf")
    import_corpus(tmp_path, mapping={"new.pdf": dict(document_id=document_id)})
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert connection.execute("SELECT source_key FROM documents").fetchone()[0] == key
        assert connection.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 1
    (tmp_path/"copy.pdf").write_bytes(b"%PDF same")
    import_corpus(tmp_path, mapping={"new.pdf": dict(document_id=document_id)})
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM events WHERE reason_code='duplicate_content'").fetchone()[0] >= 1
    (tmp_path/"new.pdf").write_bytes(b"%PDF changed")
    import_corpus(tmp_path, mapping={"new.pdf": dict(document_id=document_id)})
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT MAX(revision) FROM document_versions WHERE document_id=?", (document_id,)).fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM document_versions WHERE source_sha256 IS NOT NULL").fetchone()[0] == 0


def test_unlinked_rename_is_new_document_and_basename_is_not_a_join(tmp_path):
    (tmp_path/"a").mkdir(); (tmp_path/"b").mkdir()
    (tmp_path/"a/same.pdf").write_bytes(b"a")
    (tmp_path/"b/same.pdf").write_bytes(b"b")
    rows(tmp_path/"sources.parquet", [dict(source_path="a/same.pdf"), dict(source_path="b/same.pdf")])
    import_corpus(tmp_path)
    (tmp_path/"a/same.pdf").rename(tmp_path/"renamed.pdf")
    result = import_corpus(tmp_path)
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM current_file_state WHERE presence='missing'").fetchone()[0] >= 1


def test_chunk_spelling_is_not_identity_and_mapping_requires_parent_hash(tmp_path):
    (tmp_path/"notes__p0001-0010.pdf").write_bytes(b"real named source")
    (tmp_path/"parent.pdf").write_bytes(b"parent")
    (tmp_path/"markdown").mkdir()
    chunk = tmp_path/"markdown/parent__p00001-00010.md"; chunk.write_text("chunk")
    result = import_corpus(tmp_path)
    assert any(c["code"] == "chunk_parent_unresolved" for c in result["conflicts"])
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert connection.execute("SELECT page_first FROM files WHERE uri=?", (chunk.as_uri(),)).fetchone()[0] is None
        parent = connection.execute("SELECT d.document_id,v.stored_sha256 FROM documents d JOIN document_versions v USING(document_id) WHERE d.source_key=?", (source_key((tmp_path/"parent.pdf").as_uri()),)).fetchone()
    mapping = {str(chunk.relative_to(tmp_path)): dict(document_id=parent[0], parent_input_sha256=parent[1], page_first=1, page_last=10)}
    import_corpus(tmp_path, mapping=mapping)
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT page_first,page_last FROM files WHERE uri=? ORDER BY rowid DESC", (chunk.as_uri(),)).fetchone() == (1, 10)


def test_live_queue_backup_includes_committed_wal_and_is_observation_only(tmp_path):
    path = tmp_path/"ocr_queue.sqlite3"
    queue = sqlite3.connect(path)
    queue.execute("PRAGMA journal_mode=WAL"); queue.execute("PRAGMA wal_autocheckpoint=0")
    queue.execute("CREATE TABLE queue (source TEXT,status TEXT,retries INTEGER,error TEXT)")
    queue.commit(); queue.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    queue.execute("INSERT INTO queue VALUES ('https://x','running',3,'old error')"); queue.commit()
    original = path.read_bytes()
    result = import_corpus(tmp_path)
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        inventory = snapshot(connection, result["inventory_sha256"])
        entry = inventory["files"][0]
        assert entry["wal"]["bytes"] > 0
        row = untyped_value(entry["tables"][0]["rows"][0])
        assert row == dict(source='https://x', status='running', retries=3, error='old error')
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
    assert path.read_bytes() == original
    queue.close()
    # Frozen evidence remains usable when every queue byte has gone.
    path.unlink()
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert snapshot(connection, result["inventory_sha256"])["files"][0]["tables"][0]["rows"]


def test_malformed_missing_pickle_fatal_and_unknown_provenance(tmp_path):
    rows(tmp_path/"source.parquet", [dict(source_path="missing.pdf")])
    (tmp_path/"bad.parquet").write_bytes(b"not parquet")
    (tmp_path/"bad.json").write_text("{")
    (tmp_path/".processing_state.pkl").write_bytes(b"must never unpickle")
    (tmp_path/"fatal_skip.txt").write_text("missing\nunresolved\n")
    result = import_corpus(tmp_path, trust_legacy_outputs=True)
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM current_file_state WHERE presence='corrupt'").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM current_file_state WHERE presence='missing'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM events WHERE reason_code='fatal_skiplist'").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM events WHERE event_kind='lifecycle'").fetchone()[0] == 0
    assert any(c["code"] == "fatal_stem_unresolved" for c in result["conflicts"])
    assert result["unknowns"]


def test_inventory_change_rejected_and_import_failure_rolls_back(tmp_path, monkeypatch):
    (tmp_path/"a.pdf").write_bytes(b"a")
    import glossapi.ledger.importer as importer
    real_stamp = importer._stamp
    calls = []
    def changing(root):
        calls.append(1)
        if len(calls) == 2:
            (root/"a.pdf").write_bytes(b"changed")
        return real_stamp(root)
    monkeypatch.setattr(importer, "_stamp", changing)
    with pytest.raises(LedgerError, match="changed"):
        import_corpus(tmp_path)
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    monkeypatch.setattr(importer, "_stamp", real_stamp)
    first = import_corpus(tmp_path)
    with pytest.raises(LedgerError):
        import_corpus(tmp_path, mapping={"a.pdf": dict(document_id="not-a-uuid")})
    assert import_corpus(tmp_path)["run_id"] == first["run_id"]


def test_explicit_source_records_with_same_url_keep_separate_histories(tmp_path):
    rows(tmp_path/"input.parquet", [dict(dataset_id="dataset", source_id="one", url="https://example.org/a"),
                                    dict(dataset_id="dataset", source_id="two", url="https://example.org/a")])
    import_corpus(tmp_path)
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM events WHERE reason_code='duplicate_url'").fetchone()[0] == 2


def test_manifest_links_artifacts_and_diagnostics_survive_payload_removal(tmp_path):
    (tmp_path/"native.json").write_text('{"pages":[{"number":1,"status":"dropped","text":"secret body"}],"model":null}')
    manifest = [dict(source_uri="https://example.org/source", artifacts=[dict(path="native.json"), dict(path="missing.md")])]
    (tmp_path/"source.json").write_text(json.dumps(manifest))
    result = import_corpus(tmp_path)
    (tmp_path/"native.json").unlink(); (tmp_path/"source.json").unlink()
    from glossapi.ledger.reader import report_why, select_run
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        document_id = connection.execute("SELECT document_id FROM documents").fetchone()[0]
        why = report_why(connection, document_id, run=select_run(connection))
        assert len(why["files"]) == 2
        native = next(f for f in why["files"] if f["path"].endswith("native.json"))
        assert native["legacy_evidence"][0]["evidence"]["pages"][0]["status"] == "dropped"
        assert 'secret body' not in json.dumps(why)
        missing = next(f for f in why["files"] if f["path"].endswith("missing.md"))
        assert missing["presence"] == "missing"
        manifest_id = why["document"]["source_records"][0]["manifest_sha256"]
        original = snapshot(connection, manifest_id)
        assert original["original_file_id"] and original["original_file_sha256"] != manifest_id


def test_invalid_ranges_and_conflicting_manifest_owners_stay_unresolved(tmp_path):
    (tmp_path/"range.json").write_text('{"page_first":0,"page_last":2}')
    (tmp_path/"shared.md").write_text("shared")
    (tmp_path/"sources.json").write_text(json.dumps([
        dict(source_uri="https://a", artifacts=[dict(path="shared.md")]),
        dict(source_uri="https://b", artifacts=[dict(path="shared.md")])]))
    result = import_corpus(tmp_path)
    assert any(c["code"] == "manifest_artifact_ownership_ambiguous" for c in result["conflicts"])
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT version_id FROM files WHERE uri=?", ((tmp_path/"shared.md").as_uri(),)).fetchone()[0] is None
        assert connection.execute("SELECT presence FROM current_file_state WHERE uri=?", ((tmp_path/"range.json").as_uri(),)).fetchone()[0] == "corrupt"


def test_product_parquets_store_body_hashes_without_losing_scalar_evidence(tmp_path):
    rows(tmp_path/"sections.parquet", [dict(filename="a", id=2**62+17, row_id="row_big", section="private body", json_section="private body", section_length=1, section_propo=1000, header="private header", has_text=True)])
    result = import_corpus(tmp_path)
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        inventory = snapshot(connection, result["inventory_sha256"])
        assert "private body" not in json.dumps(inventory) and "private header" not in json.dumps(inventory)
        row = untyped_value(inventory["files"][0]["rows"][0]["fields"])
        assert row["id"] == 2**62+17 and row["has_text"] is True
        assert len(row["section"]["sha256"]) == 64
        assert connection.execute("SELECT COUNT(*) FROM source_records").fetchone()[0] == 0


def test_changed_inventory_reconciles_without_duplicate_unchanged_file_facts(tmp_path):
    (tmp_path/"source.pdf").write_bytes(b"source")
    first = import_corpus(tmp_path)
    (tmp_path/"new.json").write_text('{"known":0}')
    second = import_corpus(tmp_path)
    assert first["run_id"] != second["run_id"]
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM files WHERE uri=?", ((tmp_path/"source.pdf").as_uri(),)).fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
        assert any("explicit_inventory_reconciliation" in row[0] for row in connection.execute("SELECT details_json FROM events"))


def test_expanded_resources_preserve_supplied_resource_indices(tmp_path):
    rows(tmp_path/"input.parquet", [dict(dataset_id="d", source_id="source", url="https://x/a", url_index=3),
                                    dict(dataset_id="d", source_id="source", url="https://x/b", url_index=4)])
    import_corpus(tmp_path)
    with open_readonly(tmp_path/"corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert connection.execute("SELECT url_index FROM source_records ORDER BY url_index").fetchall() == [(3,), (4,)]


@pytest.mark.parametrize('format', ['jsonl', 'parquet'])
def test_export_records_are_payloads_not_new_sources_and_document_bodies_stay_out(tmp_path, format):
    record=dict(doc_id='compatibility-id', chunk_id=0, document='exported private document body', filename='sample', url='https://example.org/a')
    path=tmp_path/('export.'+format)
    if format=='parquet':
        rows(path,[record])
    else:
        path.write_text(json.dumps(record)+'\n')
    result=import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] == 0
        inventory=snapshot(connection,result['inventory_sha256'])
        assert 'exported private document body' not in json.dumps(inventory)
        assert connection.execute('SELECT COUNT(*) FROM source_records').fetchone()[0] == 0


def test_original_source_type_tags_retain_nanoseconds_and_array_dtype(tmp_path):
    import numpy as np
    rows(tmp_path/'input.parquet',[dict(url='https://example.org/a', timestamp=pd.Timestamp('2026-01-01T00:00:00.123456789'), array=np.array([2**62+19],dtype='int64'))])
    result=import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        original=snapshot(connection,result['inventory_sha256'])['files'][0]['rows'][0]['fields']
        metadata=untyped_value(json.loads(connection.execute('SELECT metadata_json FROM source_records').fetchone()[0])['fields'])
        assert metadata['original_fields']==original
        assert '123456789' in json.dumps(metadata['original_fields'])
        assert 'int64' in json.dumps(metadata['original_fields'])


def test_trust_reconciliation_reuses_original_source_occurrence(tmp_path):
    (tmp_path/'source.json').write_text('[{"url":"https://example.org/a","filename":"a.pdf"}]')
    first=import_corpus(tmp_path)
    second=import_corpus(tmp_path,trust_legacy_outputs=True)
    assert second['run_id']!=first['run_id']
    assert import_corpus(tmp_path,trust_legacy_outputs=True)['already_imported']
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM source_records').fetchone()[0]==1
        assert connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0]==2
        assert any('source_occurrence_reobserved' in row[0] for row in connection.execute('SELECT details_json FROM events'))


def test_changed_source_mapping_appends_reconciliation_and_preserves_original_record(tmp_path):
    (tmp_path/'source.json').write_text('[{"url":"https://example.org/a"},{"url":"https://example.org/b"}]')
    import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        original=connection.execute('SELECT source_record_id,document_id,metadata_json FROM source_records ORDER BY source_uri').fetchall()
    mapping={'source.json#0:0':dict(document_id=original[1][1])}
    second=import_corpus(tmp_path,mapping=mapping)
    assert any(c['code']=='explicit_source_mapping_changed' and c['resolution']=='explicit_mapping' for c in second['conflicts'])
    assert import_corpus(tmp_path,mapping=mapping)['already_imported']
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        for row in original:
            assert connection.execute('SELECT source_record_id,document_id,metadata_json FROM source_records WHERE source_record_id=?',(row[0],)).fetchone()==row
        assert connection.execute('SELECT COUNT(*) FROM source_records').fetchone()[0]==3
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0]==2


@pytest.mark.parametrize('chunk_name',['a-part__p0001-0010.pdf','a-part__p00001-00010.pdf'])
def test_explicit_pdf_chunks_use_parent_version_without_changing_source_bytes(tmp_path,chunk_name):
    (tmp_path/'z-parent.pdf').write_bytes(b'full parent')
    import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        document,version,digest=connection.execute('SELECT document_id,version_id,stored_sha256 FROM document_versions').fetchone()
    (tmp_path/'downloads').mkdir();(tmp_path/'downloads'/chunk_name).write_bytes(b'page subset')
    mapping={'downloads/'+chunk_name:dict(document_id=document,parent_input_sha256=digest,page_first=1,page_last=10)}
    import_corpus(tmp_path,mapping=mapping)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM document_versions').fetchone()[0]==1
        assert connection.execute('SELECT stored_sha256 FROM document_versions WHERE version_id=?',(version,)).fetchone()[0]==digest
        assert connection.execute('SELECT version_id,page_first,page_last FROM files WHERE uri=?',((tmp_path/'downloads'/chunk_name).as_uri(),)).fetchone()==(version,1,10)
    invalid={key:dict(value,page_first=True) for key,value in mapping.items()}
    with pytest.raises(LedgerError,match='ranges'):
        import_corpus(tmp_path,mapping=invalid)


def test_overlapping_explicit_chunk_ranges_are_preserved_as_conflicting_observations(tmp_path):
    (tmp_path/'parent.pdf').write_bytes(b'parent')
    import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        document,version,digest=connection.execute('SELECT document_id,version_id,stored_sha256 FROM document_versions').fetchone()
    (tmp_path/'first.md').write_text('first');(tmp_path/'second.md').write_text('second')
    mapping={name:dict(document_id=document,parent_input_sha256=digest,page_first=first,page_last=last)
             for name,first,last in [('first.md',1,10),('second.md',5,15)]}
    result=import_corpus(tmp_path,mapping=mapping,trust_legacy_outputs=True)
    assert any(c['code']=='overlapping_chunk_observations' for c in result['conflicts'])
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM files WHERE page_first IS NOT NULL').fetchone()[0]==2
        assert connection.execute('SELECT COUNT(*) FROM document_versions').fetchone()[0]==1


def test_requested_url_identity_is_independent_of_recorded_local_source_location(tmp_path):
    (tmp_path/'nested').mkdir();source=tmp_path/'nested/original.pdf';source.write_bytes(b'stored bytes')
    manifest=tmp_path/'source.json'
    manifest.write_text(json.dumps([dict(url='HTTPS://EXAMPLE.org/Resource#one',source_path='nested/original.pdf')]))
    import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        document,key=connection.execute('SELECT document_id,source_key FROM documents').fetchone()
        version,digest=connection.execute('SELECT version_id,stored_sha256 FROM document_versions').fetchone()
        assert key=='url:v1:https://example.org/Resource' and digest is not None
        assert connection.execute('SELECT COUNT(*) FROM source_records WHERE document_id=?',(document,)).fetchone()[0]==2
    source.rename(tmp_path/'nested/renamed.pdf')
    manifest.write_text(json.dumps([dict(url='https://example.org/Resource#two',source_path='nested/renamed.pdf')]))
    import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        assert connection.execute('SELECT document_id,source_key FROM documents').fetchone()==(document,key)
        assert connection.execute('SELECT version_id,stored_sha256 FROM document_versions').fetchone()==(version,digest)
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0]==1


def test_explicit_source_ids_do_not_alias_after_nanosecond_decoding(tmp_path):
    identities=[pd.Timestamp('2026-01-01T00:00:00.123456781'),pd.Timestamp('2026-01-01T00:00:00.123456789')]
    rows(tmp_path/'source.parquet',[dict(dataset_id='dataset',source_id=identifier,url='https://example.org/a') for identifier in identities])
    import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        keys={row[0] for row in connection.execute('SELECT source_key FROM documents')}
        assert keys=={source_key('https://example.org/a',dataset_id='dataset',source_id=identifier) for identifier in identities}
        assert len(keys)==2


@pytest.mark.parametrize('format',['parquet','jsonl'])
def test_sidecar_page_rows_remain_payload_evidence_with_bodies_hashed(tmp_path,format):
    folder=tmp_path/'sidecars';folder.mkdir()
    record=dict(filename='source.pdf',page=1,text='private page body',status='dropped')
    if format=='parquet':
        rows(folder/'pages.parquet',[record])
    else:
        (folder/'pages.jsonl').write_text(json.dumps(record)+'\n')
    result=import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM source_records').fetchone()[0]==0
        assert 'private page body' not in json.dumps(snapshot(connection,result['inventory_sha256']))


def test_queue_binary_identity_metadata_keeps_its_type_and_bytes(tmp_path):
    path=tmp_path/'queue.sqlite3'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE queue (source_sha256 BLOB, status TEXT, body BLOB)')
        connection.execute('INSERT INTO queue VALUES (?, ?, ?)',(b'\x00\xff'*16,'failed',b'private body bytes'))
    result=import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        entry=snapshot(connection,result['inventory_sha256'])['files'][0]
        row=untyped_value(entry['tables'][0]['rows'][0])
        assert row['source_sha256']==b'\x00\xff'*16
        assert row['status']=='failed' and row['body']['bytes']==18
        assert 'private body bytes' not in json.dumps(entry)


def test_source_metadata_columns_named_like_diagnostics_remain_original_fields(tmp_path):
    original=dict(url='https://example.org/a',section='source category',json_section='source JSON label',header='original header',title='scraped title')
    rows(tmp_path/'input.parquet',[original])
    import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        metadata=untyped_value(json.loads(connection.execute('SELECT metadata_json FROM source_records').fetchone()[0])['fields'])
        assert untyped_value(metadata['original_fields'])==original


def test_bad_source_uri_is_a_preserved_conflict_without_losing_other_rows(tmp_path):
    (tmp_path/'input.json').write_text('[{"url":"https://[invalid"},{"url":"https://example.org/good"}]')
    result=import_corpus(tmp_path)
    assert any(c['code']=='invalid_source_uri' for c in result['conflicts'])
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0]==1
        assert len(snapshot(connection,result['inventory_sha256'])['files'][0]['rows'])==2


def test_unknown_json_ranges_are_not_corruption_and_bad_ranges_keep_diagnostics(tmp_path):
    (tmp_path/'unknown.json').write_text('{"page_first":null,"page_last":null,"status":"unreported"}')
    (tmp_path/'bad.json').write_text('{"page_first":0,"page_last":2,"error":"original diagnosis"}')
    result=import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        entries={e['path']:e for e in snapshot(connection,result['inventory_sha256'])['files']}
        assert entries['unknown.json']['format']=='json'
        assert entries['bad.json']['format']=='malformed'
        assert entries['bad.json']['evidence']['error']=='original diagnosis'


def test_bad_legacy_manifest_chunk_range_is_a_conflict_not_a_failed_import(tmp_path):
    from glossapi.ledger.encoding import file_hash
    (tmp_path/'parent.pdf').write_bytes(b'parent');(tmp_path/'part.md').write_text('part')
    (tmp_path/'input.json').write_text(json.dumps([dict(url='https://example.org/a',source_path='parent.pdf',artifacts=[
        dict(path='part.md',parent_input_sha256=file_hash(tmp_path/'parent.pdf'),page_first=0,page_last=10)])]))
    result=import_corpus(tmp_path)
    assert any(c['code']=='invalid_legacy_artifact_range' for c in result['conflicts'])
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        assert connection.execute('SELECT page_first,page_last FROM files WHERE uri=?',((tmp_path/'part.md').as_uri(),)).fetchone()==(None,None)


def test_nested_parquet_diagnostic_arrays_hash_bodies_and_keep_measurements(tmp_path):
    folder=tmp_path/'sidecars';folder.mkdir()
    rows(folder/'pages.parquet',[dict(filename='a.pdf',pages=[dict(number=1,text='nested private body',score=0.25)])])
    result=import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        entry=snapshot(connection,result['inventory_sha256'])['files'][0]
        assert 'nested private body' not in json.dumps(entry)
        assert '0x1.0000000000000p-2' in json.dumps(entry)
        assert connection.execute('SELECT COUNT(*) FROM source_records').fetchone()[0]==0


def test_docling_document_is_a_registered_payload_not_copied_text(tmp_path):
    (tmp_path/'document.json').write_text(json.dumps(dict(schema_name='DoclingDocument',version='1.7.0',texts=[dict(orig='private original text',text='private normalized text')])))
    result=import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        entry=snapshot(connection,result['inventory_sha256'])['files'][0]
        assert entry['format']=='document_payload_json' and entry['document_schema_version']=='1.7.0'
        assert 'private original text' not in json.dumps(entry) and 'private normalized text' not in json.dumps(entry)
        assert connection.execute('SELECT COUNT(*) FROM files').fetchone()[0]==1


def test_json_source_decimal_lexemes_are_preserved_without_float_rounding(tmp_path):
    import decimal
    (tmp_path/'input.json').write_text('[{"url":"https://example.org/a","amount":0.12345678901234567890123456789}]')
    import_corpus(tmp_path)
    with open_readonly(tmp_path/'corpus.sqlite3') as connection:
        metadata=untyped_value(json.loads(connection.execute('SELECT metadata_json FROM source_records').fetchone()[0])['fields'])
        original=untyped_value(metadata['original_fields'])
        assert original['amount']==decimal.Decimal('0.12345678901234567890123456789')


def test_frozen_queue_main_and_wal_without_shm_import_without_touching_originals(tmp_path):
    import shutil
    live=tmp_path/'live';live.mkdir();corpus=tmp_path/'corpus';corpus.mkdir()
    producer=sqlite3.connect(live/'queue.sqlite3')
    producer.execute('PRAGMA journal_mode=WAL');producer.execute('PRAGMA wal_autocheckpoint=0')
    producer.execute('CREATE TABLE queue (status TEXT,retries INTEGER)');producer.commit()
    producer.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    producer.execute("INSERT INTO queue VALUES ('running',3)");producer.commit()
    for name in ('queue.sqlite3','queue.sqlite3-wal'):
        shutil.copyfile(live/name,corpus/name)
    producer.close()
    original={p.name:p.read_bytes() for p in corpus.iterdir()}
    result=import_corpus(corpus)
    with open_readonly(corpus/'corpus.sqlite3') as connection:
        entry=snapshot(connection,result['inventory_sha256'])['files'][0]
        assert entry['snapshot_method']=='backup_from_frozen_private_copy'
        assert untyped_value(entry['tables'][0]['rows'][0])==dict(status='running',retries=3)
    assert not (corpus/'queue.sqlite3-shm').exists()
    assert all((corpus/name).read_bytes()==body for name,body in original.items())
