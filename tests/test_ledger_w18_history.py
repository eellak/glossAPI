"""W18 M6: an as-of report of an older run is unchanged after a later run releases or replaces its files (K4(b))."""
import json

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def views(root, source, run_id):
    from glossapi.ledger.reader import report_why, select_run, status_view
    with open_readonly(root/'corpus.sqlite3') as connection:
        run = select_run(connection, run_id)
        why = report_why(connection, str(source), run=run)
        status = status_view(connection, run=run)
    files = sorted((item['path'], item['kind'], item['presence'], item['sha256'],
                    json.dumps(item['retention'], sort_keys=True, default=str)) for item in why['files'])
    stages = sorted((item['operation'], item['status'], item['run_id']) for item in why['stages'])
    return dict(files=files, stages=stages, state=status['state'], counts=status['counts'])


def test_as_of_report_after_a_later_run_releases_files(tmp_path):
    from test_ledger_fixset_lg import pdf
    source = pdf(tmp_path/'in/doc.pdf', 'History body text')
    root = tmp_path/'out'
    first = Corpus(tmp_path/'empty', root).run([source], stages=('extract', 'clean', 'export'),
                                               settings={'debug': 'on', 'extract': {'repair_fonts': False}})
    assert first['state'] == 'completed'
    before = views(root, source, first['run_id'])
    assert any(presence == 'present' and kind == 'raw_markdown' for _, kind, presence, _, _ in before['files'])
    second = Corpus(tmp_path/'empty', root).run([source], stages=('extract', 'clean', 'export'),
                                                settings={'debug': 'off', 'extract': {'repair_fonts': False}})
    assert second['state'] == 'completed'
    after = views(root, source, first['run_id'])
    assert after == before, 'a later run changed the as-of view of run 1'
    later = views(root, source, second['run_id'])
    released = [row for row in later['files'] if row[2] == 'deleted']
    assert released, 'run 2 (debug off) released nothing'
    unexplained = [row[:2] for row in released if not json.loads(row[4])['deletions']]
    assert not unexplained, unexplained
    consumed = [row for row in released if row[1] == 'raw_markdown']
    assert consumed and all(json.loads(row[4])['consumer_ok_event_ids'] for row in consumed), [row[:2] for row in consumed]


def test_as_of_queries_use_indexes_on_a_v4_ledger(tmp_path):
    """The as-of retention and history queries never scan the event tables (K4C-G05 after the v4 index drop)."""
    from test_ledger_fixset_lg import pdf
    from glossapi.ledger.compact import connect
    source = pdf(tmp_path/'in/doc.pdf', 'History body text')
    root = tmp_path/'out'
    Corpus(tmp_path/'empty', root).run([source], stages=('extract', 'clean', 'export'), settings={'extract': {'repair_fonts': False}})
    with connect(root/'corpus.sqlite3') as connection:
        assert connection.execute('PRAGMA user_version').fetchone()[0] >= 4
        file_id = connection.execute("SELECT file_id FROM files LIMIT 1").fetchone()[0]
        statements = [
            ("SELECT event_id,details_json FROM event_records WHERE file_id=? AND event_kind='artifact' AND reason_code='deleted' "
             "ORDER BY event_id DESC LIMIT 1", (file_id,)),
            ("SELECT event_id,reason_code,details_json FROM retention_events WHERE file_id=? AND event_id<=? ORDER BY event_id", (file_id, 1 << 62)),
        ]
        for sql, parameters in statements:
            plan = ' '.join(row[-1] for row in connection.execute('EXPLAIN QUERY PLAN '+sql, parameters))
            assert 'SCAN event_records' not in plan and 'SCAN retention_events' not in plan, (sql, plan)
