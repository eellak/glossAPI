"""K4g fix round 1 (CP-K4g, reviewer probes ported). Each test fails on a8a3495."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FAULT = '''
import os
from glossapi.stages import extract as E
original=E.ExtractStage._run_document
def fault(self,request,*,reporter):
    blocked=os.environ.get('PROBE_BLOCKED','').split(',')
    if any(name and any(name in str(p.path) or (name+' original text').encode() in p.path.read_bytes() for p in request.inputs if p.path.is_file()) for name in blocked):
        raise OSError(28,'injected persistent worker ENOSPC')
    return original(self,request,reporter=reporter)
E.ExtractStage._run_document=fault
'''


def child(root, mode, blocked='', allow_failure=False, **extra):
    root.mkdir(parents=True, exist_ok=True)
    stub = root/'fault'
    stub.mkdir(exist_ok=True)
    (stub/'sitecustomize.py').write_text(FAULT)
    env = {**os.environ, **extra, 'PROBE_BLOCKED': blocked, 'GLOSSAPI_CONVERSION_LOG_FILE': str(root/'conversion.log'),
           'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': os.pathsep.join([str(stub), *filter(None, [os.environ.get('PYTHONPATH')]), str(REPO/'src'), str(HERE)])}
    env.pop('GLOSSAPI_STAGE_PROFILE', None)
    done = subprocess.run([sys.executable, '-B', str(HERE/'k4g_fix1_child.py'), str(root), mode], cwd=root, env=env,
                          capture_output=True, text=True, timeout=600)
    if not allow_failure:
        assert done.returncode == 0, done.stderr[-5000:]
    return json.loads(done.stdout.splitlines()[-1])


# ================================================================================ F1 ===

@pytest.fixture(scope='module')
def failed_charge(tmp_path_factory):
    """Three documents, one with a persistent worker ENOSPC, extreme, debug off, sections + annotate + export."""
    return child(tmp_path_factory.mktemp('failed-charge')/'run', 'start', 'gamma', PROBE_EXTREME='1', PROBE_ANNOTATE='1',
                 PROBE_PARAGRAPHS='2500')


def test_f1_failed_run_charge_covers_every_later_footprint(failed_charge):
    """CP-K4g F1 (reviewer probe): the charge credited releases that retention suppresses while a document is failed."""
    assert failed_charge['state'] == 'completed_with_failures'
    for index, entry in enumerate(failed_charge['measured']):
        later = max(item['after'] for item in failed_charge['measured'][index:])
        assert entry['need'] >= later - entry['before'], (entry, later - entry['before'])


def test_f1_no_credit_while_document_releases_are_suspended(failed_charge):
    """One predicate decides both: a failed document suspends the between-publication releases, so nothing is credited."""
    exact = [call for call in failed_charge['credits'] if call['failed']]
    assert exact and all(call['credits'] == {} for call in exact)


def test_f1_a_room_between_the_old_bound_and_the_real_need_is_refused(failed_charge):
    """A filesystem with room between the erroneous bound (suspended releases credited) and the real footprint: the
    annotate check must refuse it (on a8a3495 it passed, and the publications would have exhausted the disk)."""
    from glossapi.stages import final_products as FP
    measured = {entry['stage']: entry for entry in failed_charge['measured']}
    annotate = measured['annotate']
    actual = max(item['after'] for item in failed_charge['measured'] if item['stage'] in ('annotate', 'export')) - annotate['before']
    call = next(call for call in failed_charge['credits'] if call.get('charges') and set(call['charges']) == {'annotate', 'export'})
    erroneous = FP.remaining_need(call['charges'], call['unsuspended'])
    assert erroneous < actual  # the defect this guards
    room = (erroneous + actual)//2
    assert annotate['need'] > room  # refused: need > room (require_room's test)


# ================================================================================ F2 ===

def test_f2_public_extreme_resume_with_a_released_input_is_refused_typed_before_writing(tmp_path):
    """CP-K4g F2 (reviewer probe): a kept private raw markdown is gone and its download was retired. Scheduler recovery
    used to invalidate the extract and re-run it (input_hash_changed, completed_with_failures, new publications)."""
    root = tmp_path/'missing'
    options = dict(PROBE_EXTREME='1', PROBE_REMOTE='1')
    start = child(root, 'start', 'gamma', **options)
    assert start['state'] == 'completed_with_failures'
    raw = [path for path in (root/'output/.glossapi/artifacts').rglob('*.md') if path.parent.name == 'markdown']
    assert raw
    raw[0].unlink()
    export_before = (root/'output/export.jsonl').read_bytes()
    resumed = child(root, 'retry-missing', '', allow_failure=True, **options)
    assert resumed.get('error_code') == 'republish_inputs_released', resumed
    assert (root/'output/export.jsonl').read_bytes() == export_before
    from glossapi.ledger.writer import open_readonly
    with open_readonly(root/'output/corpus.sqlite3') as connection:
        attempts = [row[0] for row in connection.execute(
            "SELECT (SELECT COUNT(*) FROM attempts a WHERE a.task_id=t.task_id) FROM tasks t WHERE t.operation LIKE '%.compatibility_publish'")]
    assert attempts and all(count == 1 for count in attempts)  # nothing was published again


# ================================================================================ F3 ===

def test_f3_large_integer_resource_indices_stay_exact():
    """CP-K4g F3: url_index went through float, so 2**53 and 2**53+1 collided."""
    from glossapi.stages.run import row_resource_index
    indices = [2**53, 2**53 + 1, 2**63 - 1]
    assert [row_resource_index({'url_index': value}) for value in indices] == indices


@pytest.mark.parametrize('entry', ['plan', 'run'])
def test_f3_two_large_indices_are_two_resources_in_plan_and_run(tmp_path, entry):
    from test_k4g_b1 import _plan_keys, _run_keys
    documents = [dict(url=f'https://a.example/{i}.pdf', source_id='S1', url_index=value) for i, value in enumerate((2**53, 2**53 + 1))]
    keys = _plan_keys(tmp_path, documents)[0] if entry == 'plan' else _run_keys(tmp_path, documents)
    assert len(set(keys)) == 2


@pytest.mark.parametrize('value', [1.0, 2.5, float('nan'), float('inf'), True, '3', -1, 2**63, 2.0**64])
def test_f3_non_integer_values_keep_their_typed_outcome(value):
    """Integral floats are accepted as their integer; anything else, or a value above the ledger's signed 64-bit
    url_index, is the typed invalid_manifest (bool and strings included). Integers are never routed through float."""
    from glossapi.input_manifest import ManifestError
    from glossapi.stages.run import row_resource_index
    if isinstance(value, float) and value == value and value not in (float('inf'),) and value.is_integer() and value < 2**63:
        assert row_resource_index({'url_index': value}) == int(value)
    elif isinstance(value, float) and value != value:
        assert row_resource_index({'url_index': value}) == 0  # NaN is a missing cell (pandas), as before
    else:
        with pytest.raises(ManifestError) as caught:
            row_resource_index({'url_index': value})
        assert caught.value.code == 'invalid_manifest' and caught.value.details['field'] == 'url_index'


# ================================================================================ F4 ===
# K4BP-I1: targeted tests for the `invalidation` event kind and the five active snapshot kinds that had none.

def _snapshots(connection, kind):
    return [json.loads(row[0]) for row in connection.execute('SELECT json FROM snapshots WHERE kind=?', (kind,))]


@pytest.fixture(scope='module')
def completed_run(tmp_path_factory):
    from glossapi import Corpus
    from test_ledger_fixset_lg import pdf
    root = tmp_path_factory.mktemp('snapshot-kinds')
    os.environ.setdefault('GLOSSAPI_CONVERSION_LOG_FILE', str(root/'conversion.log'))
    result = Corpus(root/'empty', root/'out').run([pdf(root/'doc.pdf', 'Snapshot kinds document text')], stages=('extract', 'clean', 'export'),
                                                  settings={'extract': {'repair_fonts': False}, 'debug': 'off'})
    assert result['state'] == 'completed'
    return root/'out', result


def test_f4_product_projection_names_the_product_its_immutable_generation_and_hash(completed_run):
    from glossapi.ledger.writer import open_readonly
    root, _ = completed_run
    with open_readonly(root/'corpus.sqlite3') as connection:
        projections = _snapshots(connection, 'product_projection')
    export = next(item for item in projections if item['product_uri'].endswith('/export.jsonl'))
    from glossapi.stages.execution import sha256_file
    assert export['sha256'] == sha256_file(root/'export.jsonl') and '/.glossapi/artifacts/' in export['immutable_uri']
    assert export['producer_attempt_id']


def test_f4_retained_run_result_binds_the_result_and_the_kept_files(completed_run):
    from glossapi.ledger.writer import open_readonly
    root, result = completed_run
    with open_readonly(root/'corpus.sqlite3') as connection:
        (retained,) = [item for item in _snapshots(connection, 'retained_run_result') if item['result']['run_id'] == result['run_id']]
        present = {row[0] for row in connection.execute("SELECT file_id FROM current_file_state WHERE presence='present'")}
    assert retained['result']['state'] == 'completed' and set(retained['retained_file_ids']) <= present
    assert any(output['kind'] == 'jsonl' for output in retained['result']['outputs'])


def test_f4_private_release_manifest_lists_released_files(completed_run):
    from glossapi.ledger.writer import open_readonly
    root, _ = completed_run
    with open_readonly(root/'corpus.sqlite3') as connection:
        manifests = _snapshots(connection, 'private_release_manifest')
        assert manifests
        for manifest in manifests:
            presences = {connection.execute('SELECT presence FROM current_file_state WHERE file_id=?', (file_id,)).fetchone()[0]
                         for file_id in manifest['file_ids']}
            assert presences == {'deleted'}, presences


def test_f4_store_schema_is_the_schema_each_store_delta_names(tmp_path, monkeypatch):
    """The legacy control profile observes metadata stores: every `legacy_store` delta names a `store_schema` snapshot,
    and the download table's schema lists its columns."""
    from glossapi import Corpus
    from glossapi.ledger.writer import open_readonly
    from test_ledger_fixset_lg import pdf
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'legacy')
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))
    root = tmp_path/'out'
    corpus = Corpus(tmp_path/'empty', root)
    corpus.extract(file_paths=[pdf(root/'downloads'/'doc.pdf', 'Store schema text')], repair_fonts=False, accel_type='CPU', num_threads=1)
    corpus.clean(drop_bad=False, num_threads=1)
    with open_readonly(root/'corpus.sqlite3') as connection:
        schemas = {row[0]: json.loads(row[1]) for row in connection.execute("SELECT sha256,json FROM snapshots WHERE kind='store_schema'")}
        deltas = _snapshots(connection, 'legacy_store')
        reasons = {row[0] for row in connection.execute(f'SELECT DISTINCT reason_code FROM {connection.event_table}')}
    assert deltas and {delta['schema_sha256'] for delta in deltas} <= set(schemas)
    columns = [{column['value'] for column in schema.get('columns', [])} for schema in schemas.values()]
    assert any({'url', 'filename'} <= names for names in columns)
    assert 'repair_disabled_or_non_pdf' in reasons  # K4BP-I1 reason-code coverage: repair_fonts=False is recorded


def test_f4_worker_source_rows_snapshot_keeps_the_forwarded_rows(tmp_path):
    from types import SimpleNamespace
    from glossapi.ledger import journal as J
    from glossapi.ledger.writer import LedgerWriter
    output = tmp_path/'output'
    output.mkdir()
    corpus = SimpleNamespace(input_dir=tmp_path, output_dir=output, downloads_dir=output/'downloads')
    rows = [{'index': 0, 'fields': {'url': 'https://a.example/x.pdf', 'title': 'X'}}]
    with LedgerWriter(output/'corpus.sqlite3') as writer:
        journal = J.ResultJournal(writer, corpus, 'download', 'download.rows_fixture', {})
        token = J._active_journal.set(journal)
        try:
            J.accept_worker_observations([dict(path=str(tmp_path/'input.parquet'), status='ok', reason_code='legacy_observed', message='',
                                               error_type=None, traceback=None, details={'source_rows': rows})])
        finally:
            J._active_journal.reset(token)
        journal.finish(None, None)
        (stored,) = _snapshots(writer.connection, 'worker_source_rows')
    assert stored['rows'] == rows and stored['uri'] == str(tmp_path/'input.parquet')


def test_f4_resume_invalidates_a_committed_attempt_whose_output_vanished(tmp_path):
    """A failed run keeps its document products (B1c); one vanishes outside retention while its local source stays: the
    resume records an `invalidation` (output_hash_mismatch) for that extract and runs it again."""
    from glossapi.ledger.writer import open_readonly
    root = tmp_path/'run'
    start = child(root, 'start', 'gamma')
    assert start['state'] == 'completed_with_failures'
    raw = [path for path in (root/'output/.glossapi/artifacts').rglob('alpha.md') if path.parent.name == 'markdown']
    assert raw
    raw[0].unlink()
    resumed = child(root, 'retry-complete', '')
    assert resumed['state'] == 'completed' and sorted(resumed['records']) == ['alpha', 'beta', 'gamma']
    with open_readonly(root/'output/corpus.sqlite3') as connection:
        events = connection.execute(f"SELECT e.reason_code,t.operation FROM {connection.event_table} e JOIN tasks t USING(task_id) "
                                    "WHERE e.event_kind='invalidation'").fetchall()
        tables = ('event_records', 'retention_events') if connection.compact else ('events',)
        reasons = {row[0] for table in tables for row in connection.execute(f'SELECT DISTINCT reason_code FROM {table}')}
    assert ('output_hash_mismatch', 'extract.document') in [tuple(row) for row in events]
    assert 'retention_pinned' in reasons  # K4BP-I1 reason-code coverage: retention declarations pin consumed inputs
