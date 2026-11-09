"""W18 M8 + K3A-M3: an amended plan after a debug-off release re-derives from kept inputs, never rewrites past
deletions, and the ledger records the approved plan --json hash (K4(b))."""
import json

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def deletion_records(root, run_id):
    with open_readonly(root/'corpus.sqlite3') as connection:
        rows = [list(row) for row in connection.execute('SELECT * FROM completed_deletions WHERE run_id=? ORDER BY event_id', (run_id,))]
        rows += [list(row) for row in connection.execute(
            f"SELECT event_id,file_id,event_kind,reason_code,details_json FROM {connection.event_table} WHERE run_id=? AND "
            "(event_kind='retention' OR (event_kind='artifact' AND reason_code='deleted')) ORDER BY event_id", (run_id,))]
    return json.loads(json.dumps(rows, default=str))


def test_amended_plan_revalidates_released_inputs_and_keeps_past_deletions(tmp_path):
    from glossapi.cli import plan_documents
    from test_ledger_fixset_lg import pdf
    source = pdf(tmp_path/'in/doc.pdf', 'A heading line and a body for the amended plan test')
    root = tmp_path/'out'
    first = Corpus(tmp_path/'empty', root).run([source], stages=('extract', 'clean', 'export'),
                                               settings={'debug': 'off', 'extract': {'repair_fonts': False}})
    assert first['state'] == 'completed'
    before = deletion_records(root, first['run_id'])
    assert before, 'run A released nothing under debug=off'
    with open_readonly(root/'corpus.sqlite3') as connection:
        released = {row[0] for row in connection.execute("SELECT uri FROM current_file_state WHERE kind='raw_markdown' AND presence='deleted'")}
    assert released
    plan = plan_documents(root, files=[str(source)], stages=['extract', 'clean', 'sections', 'export'],
                          settings={'debug': 'off', 'extract': {'repair_fonts': False}})
    steps = [step['stage'] for step in plan['documents'][0]['steps']]
    # The released raw markdown is not an available input: plan B re-derives it from the kept supplied original.
    assert steps[:2] == ['extract', 'clean'] and 'sections' in steps
    second = Corpus(tmp_path/'empty', root).run(plan=plan)
    assert second['state'] == 'completed', second
    assert deletion_records(root, first['run_id']) == before, 'an amended run changed run A deletion records'
    with open_readonly(root/'corpus.sqlite3') as connection:
        settings = json.loads(connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.settings_sha256 WHERE r.run_id=?',
                                                 (second['run_id'],)).fetchone()[0])
    assert settings['approved_plan_hash'] == plan['plan_hash']


def test_cli_run_envelope_and_ledger_name_the_same_approved_plan(tmp_path):
    from test_k3_interface import cli
    from test_ledger_fixset_lg import pdf
    source = pdf(tmp_path/'in/doc.pdf', 'Plan identity body text')
    corpus = tmp_path/'out'
    code, planned = cli('plan', '--corpus', str(corpus), '--files', str(source), '--stages', 'extract,clean')
    assert code == 0, planned
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(planned))
    code, envelope = cli('run', '--corpus', str(corpus), '--plan', str(plan_path))
    assert code == 0, envelope
    with open_readonly(corpus/'corpus.sqlite3') as connection:
        settings = json.loads(connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.settings_sha256 WHERE r.run_id=?',
                                                 (envelope['data']['run_id'],)).fetchone()[0])
    assert settings['approved_plan_hash'] == envelope['data']['bound_plan_hash'] == planned['data']['plan_hash']
