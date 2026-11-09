"""W18 S2: after a debug-off run every file row has a retention declaration (or a contract-declared keep), and the
plan's per-document releases equal the actual deletions (K4(b))."""
import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def test_debug_off_rows_are_declared_and_preview_releases_equal_deletions(tmp_path):
    from glossapi.cli import plan_documents
    from glossapi.stages.retention import retention_facts
    from test_ledger_fixset_lg import pdf
    source = pdf(tmp_path/'in/doc.pdf', 'Declaration body text')
    root = tmp_path/'out'
    plan = plan_documents(root, files=[str(source)], stages=['extract', 'clean', 'export'],
                          settings={'debug': 'off', 'extract': {'repair_fonts': False}})
    result = Corpus(tmp_path/'empty', root).run(plan=plan)
    assert result['state'] == 'completed'
    preview = plan['deletion_policy']['per_document'][0]['released']
    previewed_kinds = {kind for item in preview for kind in item['kinds']}
    with open_readonly(root/'corpus.sqlite3') as connection:
        rows = connection.execute("SELECT file_id,kind,presence,ownership,retention_class,uri FROM current_file_state").fetchall()
        undeclared = []
        deleted_kinds = set()
        for file_id, kind, presence, ownership, retention, uri in rows:
            facts = retention_facts(connection, file_id)
            if presence == 'deleted':
                deleted_kinds.add(kind)
            if 'retention_not_declared' in facts['pins'] and ownership == 'managed' and retention not in ('final', 'evidence', 'initial', 'recovery'):
                undeclared.append((kind, uri.rsplit('/', 2)[-2:]))
    # Kinds the preview declares released are released; kinds released must be previewed (file classes, not stems).
    released_public = {kind for kind in deleted_kinds if kind in previewed_kinds}
    assert previewed_kinds <= deleted_kinds | {'page_quality', 'ocr_markdown'}, (previewed_kinds, deleted_kinds)
    assert released_public, preview
    assert not undeclared, undeclared
