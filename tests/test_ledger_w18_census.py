"""W18 M4: the annotated-section export is a separate final product with its own census (PLAN 2.3, K4(b))."""
import hashlib
import json

import pytest

from glossapi import Corpus
from glossapi.ledger.reader import evidence, report_why, select_run
from glossapi.ledger.writer import open_readonly

pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def test_annotated_section_export_accounts_for_every_planned_version(tmp_path):
    from test_ledger_fixset_lg import pdf
    from test_ledger_stages import simple_pdf
    text = pdf(tmp_path/'in/text.pdf', 'A section heading and a long enough body text for one section')
    blank = simple_pdf(tmp_path/'in/blank.pdf', blank=True)
    corrupt = tmp_path/'in/corrupt.pdf'; corrupt.write_bytes(b'%PDF-1.4\nnot a real pdf\n')
    corpus = Corpus(tmp_path/'empty', tmp_path/'out')
    result = corpus.run([text, blank, corrupt], stages=('extract', 'clean', 'sections', 'annotate', 'export'),
                        settings={'extract': {'repair_fonts': False}})
    assert result['state'] in ('completed', 'completed_with_failures'), result
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
        table = connection.event_table
        planned = {row[0] for row in connection.execute(
            "SELECT DISTINCT version_id FROM tasks WHERE run_id=? AND version_id IS NOT NULL", (result['run_id'],))}
        decisions = {}
        for version, reason, encoded in connection.execute(
                f"SELECT version_id,reason_code,details_json FROM {table} WHERE run_id=? AND event_kind='decision'", (result['run_id'],)):
            details = evidence(connection, {'details_json': encoded})
            if details.get('product') == 'annotated_sections':
                decisions.setdefault(version, []).append((reason, details))
        assert set(decisions) == planned and all(len(value) == 1 for value in decisions.values()), decisions
        by_source = {}
        for path in (text, blank, corrupt):
            why = report_why(connection, str(path), run=select_run(connection, result['run_id']))
            disposition = why['occurrences'][0]['disposition']
            by_source[path.name] = disposition
            assert 'annotated_sections' in disposition.get('products', {}), disposition
    sections = {name: value['products']['annotated_sections'] for name, value in by_source.items()}
    assert sections['text.pdf']['decision'] == 'include' and sections['text.pdf']['section_count'] >= 1
    assert sections['blank.pdf']['decision'] == 'omit' and sections['blank.pdf']['cause'] in ('no_sections', 'upstream_failed')
    assert sections['corrupt.pdf']['decision'] == 'omit' and sections['corrupt.pdf']['cause'] == 'upstream_failed'
    assert 'document_jsonl' in by_source['text.pdf']['products']
    outputs = {item['kind']: item for item in result['outputs']}
    for kind, name in (('classified_sections_parquet', 'classified_sections.parquet'), ('annotated_sections_parquet', 'fully_annotated_sections.parquet')):
        path = corpus.output_dir/name
        assert outputs[kind]['path'] == str(path)
        assert outputs[kind]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest() and outputs[kind]['bytes'] == path.stat().st_size
    assert result['section_counts']['selected'] == len(planned)
    json.dumps(result['outputs'])
