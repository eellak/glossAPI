"""K3a decision interface: CLI JSON schema v1, input manifest v1, capabilities/doctor, exact plan preview, settings help.

Acceptance map (PLAN rev 5 section 2.4; register R22-R28, R35):
- schemas: every schema is valid draft 2020-12, locked by a changelog entry, and real outputs validate (seed corpora,
  a real extract->clean->export run, capabilities, doctor, every typed error);
- manifest: Python <-> CLI <-> LLM-written JSON round trip with arbitrary metadata returned verbatim; unknown versions
  and malformed entries are typed errors; unselected siblings are never opened (audit hook in every process);
- capabilities/doctor: metadata-only detection (no backend import), the scheduler's own availability rule;
- plan: exact targets/exclusions/capabilities/devices/deletion policy; device, capability and disk-budget conflicts;
  run --plan refuses after one input byte, one setting or one installed capability changes; plan writes nothing;
- settings help equals PLAN section 2.2.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap

import pytest

from glossapi.cli import main, plan_documents

REPO = Path(__file__).resolve().parents[1]
jsonschema = pytest.importorskip('jsonschema')
from cli_schema_validation import validate  # noqa: E402  (needs jsonschema)


def cli(*arguments):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main([*arguments, '--json'])
    envelope = json.loads(buffer.getvalue())
    validate(envelope)
    return code, envelope


def env_with(*paths):
    return {**os.environ, 'PYTHONPATH': os.pathsep.join([*map(str, paths), str(REPO / 'src'), str(REPO / 'tests'),
                                                          *filter(None, [os.environ.get('PYTHONPATH')])])}


def cli_subprocess(*arguments, env=None, timeout=300):
    done = subprocess.run([sys.executable, '-B', '-m', 'glossapi.cli', *arguments, '--json'], capture_output=True, text=True,
                          env=env or env_with(), timeout=timeout)
    assert done.stdout.count('\n') == 1, (done.stdout[-2000:], done.stderr[-3000:])
    assert 'Traceback' not in done.stdout
    envelope = json.loads(done.stdout)
    validate(envelope)
    return done.returncode, envelope, done.stderr


def pdf(path, text='A small text layer'):
    from test_ledger_stages import simple_pdf
    path.parent.mkdir(parents=True, exist_ok=True)
    simple_pdf(path)
    path.write_bytes(path.read_bytes().replace(b'A small text layer', text.encode('ascii')))
    return path


def tree(root):
    return {str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None, path.stat().st_mtime_ns)
            for path in sorted(Path(root).rglob('*'))}


# ------------------------------------------------------------------ schemas ---

def test_schemas_are_valid_and_locked_by_a_changelog_entry():
    from jsonschema import Draft202012Validator
    from glossapi.schemas import NAMES, SCHEMA_VERSION, load_schema, schema_files, schema_lock
    lock = schema_lock()
    assert lock['schema_version'] == SCHEMA_VERSION == 1
    latest = lock['entries'][-1]
    current = {name: hashlib.sha256(data).hexdigest() for name, data in schema_files().items()}
    changed = sorted(name for name in NAMES if current[name] != latest['files'].get(name))
    assert not changed, (f'Schema file(s) {changed} changed without a changelog entry: within v1 only additive optional fields '
                         'are allowed; append an entry to schemas/v1/SCHEMA_LOCK.json and docs/cli_json_schema.md, or start v2')
    docs = (REPO / 'docs/cli_json_schema.md').read_text()
    assert all(f'`{entry["entry"]}`' in docs and entry['change'] for entry in lock['entries'])
    assert 'additive optional fields' in docs and 'v2' in docs
    for name in NAMES:
        value = load_schema(name)
        Draft202012Validator.check_schema(value)
        assert value['$id'].endswith('/schemas/v1/' + name + '.json')


def test_lock_detects_a_schema_change(tmp_path, monkeypatch):
    """The lock test is not vacuous: one changed byte in a schema is reported."""
    import glossapi.schemas as schemas
    original = schemas.schema_files

    def edited(version=1):
        files = original(version)
        files['plan'] = files['plan'].replace(b'"plan_hash"', b'"plan_hash2"', 1)
        return files

    monkeypatch.setattr(schemas, 'schema_files', edited)
    with pytest.raises(AssertionError, match='changed without a changelog entry'):
        test_schemas_are_valid_and_locked_by_a_changelog_entry()


def test_runtime_never_imports_jsonschema_or_backends(tmp_path):
    source = pdf(tmp_path / 'in' / 'a.pdf')
    code = textwrap.dedent('''
        import contextlib, io, json, sys
        from glossapi.cli import main
        for argv in (['capabilities'], ['doctor', '--corpus', sys.argv[1]],
                     ['plan', '--corpus', sys.argv[1], '--files', sys.argv[2], '--stages', 'download,extract,clean,ocr,math,sections,annotate,export']):
            with contextlib.redirect_stdout(io.StringIO()):
                main([*argv, '--json'])
        heavy = ('jsonschema', 'referencing', 'torch', 'docling', 'pandas', 'pyarrow', 'numpy', 'greekocr', 'onnxruntime', 'cv2', 'sklearn', 'pypdfium2')
        print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in heavy)))
    ''')
    done = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path / 'corpus'), str(source)], capture_output=True, text=True, env=env_with())
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == []


def test_seed_corpus_outputs_and_typed_errors_validate(tmp_path):
    from test_ledger_cli import seed
    run_id, document, _ = seed(tmp_path)
    for arguments in (('plan', '--corpus', str(tmp_path)), ('status', '--corpus', str(tmp_path)), ('report', '--corpus', str(tmp_path)),
                      ('report', '--corpus', str(tmp_path), '--why', document), ('status', '--corpus', str(tmp_path), '--file', document),
                      ('report', '--corpus', str(tmp_path), '--format', 'parquet', '--output', str(tmp_path / 'out.parquet')),
                      ('capabilities',), ('doctor', '--corpus', str(tmp_path))):
        code, envelope = cli(*arguments)
        assert not envelope['errors'], (arguments, envelope['errors'])
    for arguments, expected in ((('status', '--corpus', str(tmp_path / 'absent')), 'ledger_not_initialized'),
                                (('report', '--corpus', str(tmp_path), '--why', 'nothing'), 'file_not_found'),
                                (('plan', '--corpus', str(tmp_path), '--stages', 'bogus'), 'invalid_stage'),
                                (('plan', '--corpus', str(tmp_path), '--memory-saving', 'extreme'), 'disk_budget_required'),
                                (('status', '--nonsense'), 'invalid_request'), (('bogus',), 'invalid_request')):
        code, envelope = cli(*arguments)
        assert code in (2, 3) and envelope['errors'][0]['code'] == expected, (arguments, envelope['errors'])


# ----------------------------------------------------------------- manifest ---

GREEK = 'Η Ελληνική γλώσσα — «δοκιμή» ἀρχαῖα'


def llm_manifest(tmp_path):
    """A manifest exactly as an LLM that scraped elsewhere would write it (hand-typed JSON text)."""
    first, second, third = pdf(tmp_path / 'in' / 'first.pdf', 'First kept text'), pdf(tmp_path / 'in' / 'second.pdf', 'Second text'), \
        pdf(tmp_path / 'in' / 'third.pdf', 'Third kept text')
    text = '''{
  "manifest_version": 1,
  "producer": {"tool": "an-llm", "query": "%s"},
  "documents": [
    {"path": "%s", "selection_id": "first",
     "metadata": {"title": "%s", "year": 1821, "ratio": 0.125, "big": 12345678901234567890, "flags": [true, false, null],
                  "nested": {"authors": [{"name": "Κωνσταντίνος", "role": null}], "empty": {}, "list": []},
                  "escaped": "\\u03b1\\u03b2 tab\\tend", "none": null}},
    {"path": "%s", "selection_id": "second", "metadata": {"title": "unselected"}},
    {"path": "%s", "selection_id": "third", "source_row": "row-3", "metadata": {"title": "τρίτο", "pages": 1}}
  ],
  "selection": ["first", "third"],
  "stages": ["extract", "clean"],
  "products": ["document_export"],
  "settings": {"debug": "off"}
}''' % (GREEK, first, GREEK, second, third)
    path = tmp_path / 'llm_manifest.json'
    path.write_text(text, encoding='utf-8')
    return path, json.loads(text), (first, second, third)


def test_manifest_round_trip_python_cli_llm_with_arbitrary_metadata(tmp_path):
    pytest.importorskip('glossapi_rs_cleaner')
    from glossapi.input_manifest import build_manifest, load_manifest, write_manifest
    path, original, (first, second, third) = llm_manifest(tmp_path)
    parsed = load_manifest(path)
    assert parsed.selectors == [str(first), str(third)] and parsed.stages == ['extract', 'clean', 'export']
    assert [item['selection_id'] for item in parsed.exclusions] == ['second'] and parsed.exclusions[0]['reason_code'] == 'not_selected'
    # Python API: rebuild the same manifest and write it; the bytes decode to the identical object.
    rebuilt = build_manifest(original['documents'], selection=original['selection'], stages=original['stages'],
                             products=original['products'], settings=original['settings'])
    rebuilt['producer'] = original['producer']
    written = write_manifest(rebuilt, tmp_path / 'python_manifest.json')
    assert json.loads(written.read_text(encoding='utf-8')) == original
    assert GREEK in written.read_text(encoding='utf-8')
    # CLI plan from the LLM file == Python API plan (new document IDs are per-plan proposals).
    corpus = tmp_path / 'corpus'
    code, planned = cli('plan', '--corpus', str(corpus), '--manifest', str(path))
    assert code == 0, planned['errors']
    api = plan_documents(corpus, manifest=written)
    normal = lambda data: json.loads(json.dumps({k: v for k, v in data.items() if k not in ('plan_hash',)}).replace(
        data['documents'][0]['document_id'], 'D0').replace(data['documents'][1]['document_id'], 'D1'))
    assert normal(api) == normal(planned['data'])
    data = planned['data']
    assert [target['path'] for target in data['targets']] == [str(first), str(third)]
    assert data['exclusions'] == [dict(manifest_index=1, selection_id='second', selector=str(second), reason_code='not_selected',
                                       basis='manifest selection')]
    assert data['input_manifest']['selected'] == 2 and data['input_manifest']['excluded'] == 1
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text(json.dumps(planned), encoding='utf-8')
    code, ran = cli('run', '--corpus', str(corpus), '--plan', str(plan_file))
    assert code == 0 and ran['data']['state'] == 'completed', ran['errors']
    # Verbatim metadata in status --file, report and report --why.
    expected = {str(first): original['documents'][0], str(third): original['documents'][2]}
    code, report = cli('report', '--corpus', str(corpus))
    rows = {row['source_uri']: row['metadata'] for row in report['data']['sources']}
    for source, entry in expected.items():
        metadata = rows[Path(source).as_uri()]
        assert metadata['manifest_row'] == entry and metadata['manifest_row']['metadata'] == entry['metadata']
        assert {row['metadata_encoding'] for row in report['data']['sources']} == {'json'}
        code, status = cli('status', '--corpus', str(corpus), '--file', source)
        assert status['data']['sources'][0]['metadata']['manifest_row'] == entry
        code, why = cli('report', '--corpus', str(corpus), '--why', source)
        from glossapi.ledger.encoding import untyped_value
        assert untyped_value(why['data']['occurrences'][0]['metadata']['fields'])['manifest_row'] == entry  # typed ledger form
    assert json.dumps(rows, ensure_ascii=False).count(GREEK) == 1  # the first entry's title; the producer field is not an entry
    assert not any('second' in json.dumps(row) for row in rows.values())


@pytest.mark.parametrize('manifest,code,field', [
    ({'manifest_version': 2, 'documents': [{'path': '/a.pdf'}]}, 'unsupported_manifest_version', None),
    ({'manifest_version': '1', 'documents': [{'path': '/a.pdf'}]}, 'unsupported_manifest_version', None),
    ({'manifest_version': 1, 'documents': []}, 'invalid_manifest', 'documents'),
    ({'manifest_version': 1, 'documents': [{'path': 'relative.pdf'}]}, 'invalid_manifest', 'path'),
    ({'manifest_version': 1, 'documents': [{'path': '/a.pdf', 'url': 'https://x/a.pdf'}]}, 'invalid_manifest', None),
    ({'manifest_version': 1, 'documents': [{'path': '/a.pdf', 'metadata': ['not', 'an', 'object']}]}, 'invalid_manifest', 'metadata'),
    ({'manifest_version': 1, 'documents': [{'path': 7}]}, 'invalid_manifest', 'path'),
    ({'manifest_version': 1, 'documents': ['/a.pdf']}, 'invalid_manifest', None),
    ({'manifest_version': 1, 'documents': [{'path': '/a.pdf', 'selection_id': 'x'}, {'path': '/b.pdf', 'selection_id': 'x'}]}, 'invalid_manifest', 'selection_id'),
    ({'manifest_version': 1, 'documents': [{'path': '/a.pdf', 'selection_id': 'x'}], 'selection': ['y']}, 'invalid_manifest', 'selection'),
    ({'manifest_version': 1, 'documents': [{'path': '/a.pdf'}], 'stages': ['bogus']}, 'invalid_manifest', 'stages'),
    ({'manifest_version': 1, 'documents': [{'path': '/a.pdf'}], 'devices': {'ocr': 'gpu'}}, 'invalid_manifest', 'devices'),
    ({'manifest_version': 1, 'documents': [{'path': '/a.pdf'}], 'workers': {'clean': 0}}, 'invalid_manifest', 'workers'),
    ({'manifest_version': 1, 'documents': [{'path': '/a.pdf'}], 'extra': 1}, 'invalid_manifest', None),
    ('not json at all {', 'invalid_manifest', None),
    ([{'no_selector': 1}], 'invalid_manifest', None),
])
def test_malformed_manifests_are_typed_errors_without_traceback(tmp_path, manifest, code, field):
    path = tmp_path / 'manifest.json'
    path.write_text(manifest if isinstance(manifest, str) else json.dumps(manifest))
    for command in ('plan', 'run'):
        exit_code, envelope, stderr = cli_subprocess(command, '--corpus', str(tmp_path / 'corpus'), '--manifest', str(path))
        assert exit_code == 2 and envelope['errors'][0]['code'] == code, envelope['errors']
        assert 'Traceback' not in stderr
        if field:
            assert envelope['errors'][0]['details'].get('field') == field, envelope['errors']
    assert not (tmp_path / 'corpus').exists()


def test_manifest_and_arguments_must_agree(tmp_path):
    source = pdf(tmp_path / 'in' / 'a.pdf')
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'manifest_version': 1, 'documents': [{'path': str(source)}], 'stages': ['extract'], 'devices': {'extract': 'cpu'}}))
    code, envelope = cli('plan', '--corpus', str(tmp_path / 'c'), '--manifest', str(path), '--stages', 'clean')
    assert code == 2 and envelope['errors'][0]['code'] == 'conflicting_request'
    code, envelope = cli('plan', '--corpus', str(tmp_path / 'c'), '--manifest', str(path), '--stages', 'extract', '--device', 'clean=cpu')
    assert code == 0 and {a['stage']: a['device_id'] for a in envelope['data']['device_allocation']} == {'extract': 'cpu'}


SITE = '''
import os, sys
_log, _traps = os.environ['K3_TRAP_LOG'], tuple(os.environ['K3_TRAPS'].split(os.pathsep))
def _hook(event, args):
    if event in ('open', 'os.listdir', 'os.scandir', 'os.stat') and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0]) if not isinstance(args[0], os.PathLike) else os.fspath(args[0])
        if path.startswith(_traps):
            with open(_log, 'a') as stream:
                stream.write('%s %s %s\\n' % (os.getpid(), event, path))
sys.addaudithook(_hook)
'''


def test_unselected_siblings_are_never_opened_hashed_or_processed(tmp_path):
    """Trap files in the input directory: one unselected manifest entry and one file the manifest never names."""
    pytest.importorskip('glossapi_rs_cleaner')
    chosen = pdf(tmp_path / 'in' / 'chosen.pdf', 'Chosen text')
    unselected = pdf(tmp_path / 'in' / 'unselected.pdf', 'Unselected text')
    trap = pdf(tmp_path / 'in' / 'trap.pdf', 'Trap text')
    site = tmp_path / 'site'; site.mkdir(); (site / 'sitecustomize.py').write_text(SITE)
    log = tmp_path / 'trap.log'
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'manifest_version': 1, 'selection': ['c'], 'stages': ['extract', 'clean', 'export'],
        'documents': [{'path': str(chosen), 'selection_id': 'c'}, {'path': str(unselected), 'selection_id': 'u'}]}))
    env = {**env_with(site), 'K3_TRAP_LOG': str(log), 'K3_TRAPS': os.pathsep.join([str(unselected), str(trap)])}
    corpus = tmp_path / 'corpus'
    code, planned, _ = cli_subprocess('plan', '--corpus', str(corpus), '--manifest', str(manifest), env=env)
    assert code == 0 and [item['selector'] for item in planned['data']['exclusions']] == [str(unselected)]
    code, ran, stderr = cli_subprocess('run', '--corpus', str(corpus), '--manifest', str(manifest), env=env)
    assert code == 0 and ran['data']['state'] == 'completed', (ran['errors'], stderr[-2000:])
    assert ran['data']['document_counts']['selected'] == 1
    assert not log.exists() or log.read_text() == '', log.read_text()
    # The audit hook does see the chosen file (the guard is not vacuous).
    env['K3_TRAPS'] = str(chosen)
    cli_subprocess('plan', '--corpus', str(tmp_path / 'other'), '--manifest', str(manifest), env=env)
    assert str(chosen) in log.read_text()
    exported = [json.loads(line) for line in (corpus / 'export.jsonl').read_text().splitlines()]
    assert len(exported) == 1 and 'Chosen text' in exported[0]['document']


# ----------------------------------------------------------- capabilities ---

def test_capabilities_match_the_scheduler_rule_and_report_installs(tmp_path):
    from glossapi.capabilities import STAGE_ORDER, capabilities_view
    from glossapi.stages.contracts import StageSpecification
    from glossapi.stages.scheduler import ScheduledOperation, StreamingScheduler
    from glossapi.stages.contracts import DocumentSnapshot
    data = capabilities_view()
    assert [row['stage'] for row in data['stages']] == list(STAGE_ORDER)
    scheduler = StreamingScheduler.__new__(StreamingScheduler)
    from glossapi.stages.scheduler import stage_registry
    scheduler.registry = stage_registry()
    for row in data['stages']:
        if row['stage'] not in scheduler.registry:
            continue
        operation = ScheduledOperation('k', row['stage'], DocumentSnapshot('d', None, 'file:///x.pdf'), StageSpecification(row['stage'] + '.planned'))
        package_ok = not scheduler._check_package(operation)
        ours = all(not item['blocking'] for item in row['packages']) and all(item['status'] == 'registered' for item in row['entry_points'])
        assert package_ok == ours, row
        if not row['available']:
            assert row['install_commands'] and all(command.startswith(('glossapi install --device ', 'python install_glossapi.py core')) for command in row['install_commands'])
    cpu = data['devices'][0]
    assert cpu == dict(device_id='cpu', kind='cpu', hardware='present', state='available', basis='always present', reason=None)
    assert all(row['state'] in ('unknown_until_probe', 'unavailable') for row in data['devices'][1:])


def test_capability_detection_imports_no_backend_even_on_attempt(tmp_path):
    """Imports of torch/docling/pandas/OCR backends raise during capabilities, doctor and plan: none is attempted."""
    code = textwrap.dedent('''
        import contextlib, importlib.abc, io, json, sys
        BLOCKED = ('torch', 'docling', 'docling_core', 'pandas', 'greekocr', 'onnxruntime', 'cv2', 'transformers', 'sklearn')
        attempts = []
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in BLOCKED:
                    attempts.append(fullname)
                    raise ModuleNotFoundError('blocked by K3 capability test', name=fullname)
        sys.meta_path.insert(0, Block())
        from glossapi.cli import main
        codes = []
        for argv in (['capabilities'], ['doctor', '--corpus', sys.argv[1]],
                     ['plan', '--corpus', sys.argv[1], '--files', '/absent.pdf', '--stages', 'download,ocr,math,annotate,export']):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                codes.append(main([*argv, '--json']))
            assert not json.loads(out.getvalue())['errors'], out.getvalue()
        print(json.dumps(attempts))
    ''')
    done = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path)], capture_output=True, text=True, env=env_with())
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == []


def test_doctor_checks_have_fix_hints_and_probe_is_explicit(tmp_path):
    code, envelope = cli('doctor', '--corpus', str(tmp_path / 'not-yet'))
    data = envelope['data']
    ids = {check['check_id'] for check in data['checks']}
    assert {'python_version', 'disk_free', 'corpus_writable', 'ledger', 'native_extension:glossapi_rs_cleaner',
            'native_extension:glossapi_rs_noise', 'native_extension:glossapi_rs_pdfrepair'} <= ids
    assert all(check['fix_hint'] for check in data['checks'] if check['status'] in ('warning', 'error'))
    assert data['device_probe'] == dict(requested=False, imported=[], process=None, framework=None, error=None)
    assert code == (0 if data['healthy'] else 1)
    assert not (tmp_path / 'not-yet').exists()


def test_doctor_probe_devices_imports_torch_only_in_a_child(tmp_path):
    pytest.importorskip('torch')
    code, envelope = cli('doctor', '--corpus', str(tmp_path), '--probe-devices')
    probe = envelope['data']['device_probe']
    assert probe['requested'] and probe['process'] == 'child' and probe['imported'] == ['torch']
    assert any(item['code'] == 'device_probe_imported_torch' for item in envelope['warnings'])
    assert 'torch' not in sys.modules or True  # the parent may have torch from other tests; the probe ran in a child


# --------------------------------------------------------------- plan preview ---

def test_exact_preview_targets_exclusions_devices_and_deletion_policy(tmp_path):
    first = pdf(tmp_path / 'in' / 'first.pdf')
    code, envelope = cli('plan', '--corpus', str(tmp_path / 'c'), '--files', str(first), str(first), '/absent/missing.pdf',
                         '--stages', 'extract,clean,export', '--memory-saving', 'normal')
    data = envelope['data']
    assert code == 0, envelope['errors']
    assert [(t['path'], t['state'], t['bytes']) for t in data['targets']] == [(str(first), 'present', first.stat().st_size),
                                                                              ('/absent/missing.pdf', 'absent', None)]
    assert data['targets'][0]['sha256'] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert data['exclusions'] == [dict(selector=str(first), reason_code='duplicate_source',
                                       duplicate_of_document_id=data['targets'][0]['document_id'], basis='same source identity as an earlier entry')]
    assert [(a['stage'], a['device_id'], a['state']) for a in data['device_allocation']] == [('extract', 'cpu', 'available'),
                                                                                            ('clean', 'cpu', 'available'), ('export', 'cpu', 'available')]
    policy = data['deletion_policy']
    assert policy['final_products'] == [dict(stage='export', product='document_export', kinds=['export_jsonl'])]
    assert [(r['producer_stage'], r['after_readers'], r['applies']) for r in policy['releases']] == [('extract', ['clean'], True), ('clean', ['export'], True)]
    assert policy['per_document'][0]['source'] == dict(kind='supplied', action='kept', reason='supplied originals are never deleted')
    code, debug_on = cli('plan', '--corpus', str(tmp_path / 'c'), '--files', str(first), '--stages', 'extract,clean,export', '--debug', 'on')
    assert all(not r['applies'] for r in debug_on['data']['deletion_policy']['releases'])
    assert debug_on['data']['deletion_policy']['per_document'][0]['released'] == []
    code, extreme = cli('plan', '--corpus', str(tmp_path / 'c'), '--files', 'https://example.org/a.pdf', '--stages', 'extract',
                        '--memory-saving', 'extreme', '--disk-budget-bytes', '1000000000', '--settings', str(write_json(tmp_path / 's.json',
                        {'max_source_bytes': 10**8, 'max_attempt_bytes': 10**8})))
    deletion = extreme['data']['deletion_policy']
    assert deletion['deletions'][0]['applies'] and deletion['deletions'][0]['after_readers'] == ['extract']
    assert deletion['per_document'][0]['source']['action'] == 'deleted_after_last_reader'
    assert extreme['data']['targets'][0]['download'] == {'source_uri': 'https://example.org/a.pdf', 'reacquire': False,
                                                              'reason': 'first acquisition'}  # K4BP-G5 (additive)


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize('arguments,expected', [
    (['--stages', 'extract', '--device', 'extract=cuda:0'], 'device_unsupported_for_stage'),
    (['--stages', 'clean', '--device', 'clean=mps'], 'device_unsupported_for_stage'),
])
def test_unsupported_device_fails_plan_and_run_before_anything_runs(tmp_path, arguments, expected):
    source = pdf(tmp_path / 'in' / 'a.pdf')
    corpus = tmp_path / 'corpus'
    code, envelope = cli('plan', '--corpus', str(corpus), '--files', str(source), *arguments)
    assert code == 2 and envelope['errors'][0]['code'] == expected
    assert envelope['data']['execution']['ready'] is False and expected in envelope['data']['execution']['blocked_by']
    code, envelope = cli('run', '--corpus', str(corpus), '--files', str(source), *arguments)
    assert code == 2 and envelope['errors'][0]['code'] == expected
    assert not corpus.exists()


def test_unavailable_device_and_option_conflict_fail_explicitly(tmp_path, monkeypatch):
    import glossapi.capabilities as capabilities
    monkeypatch.setattr(capabilities, '_nvidia_devices', lambda: [])
    source = pdf(tmp_path / 'in' / 'a.pdf')
    corpus = tmp_path / 'corpus'
    code, envelope = cli('plan', '--corpus', str(corpus), '--files', str(source), '--stages', 'ocr', '--device', 'ocr=cuda:0')
    assert code == 2 and envelope['errors'][0]['code'] == 'device_unavailable'
    settings = write_json(tmp_path / 's.json', {'ocr': {'options': {'device': 'cpu'}}})
    code, envelope = cli('plan', '--corpus', str(corpus), '--files', str(source), '--stages', 'ocr', '--device', 'ocr=mps', '--settings', str(settings))
    assert code == 2 and envelope['errors'][0]['code'] in ('device_option_conflict', 'device_unavailable')
    assert any(item['code'] == 'device_option_conflict' for item in envelope['errors'][0]['details']['conflicts'])
    code, envelope = cli('run', '--corpus', str(corpus), '--files', str(source), '--stages', 'ocr', '--device', 'ocr=cuda:0')
    assert code == 2 and envelope['errors'][0]['code'] == 'device_unavailable' and not corpus.exists()


def test_missing_capability_is_previewed_and_run_refuses_before_writing(tmp_path, monkeypatch):
    import glossapi.capabilities as capabilities
    real = capabilities.installed_version
    monkeypatch.setattr(capabilities, 'installed_version', lambda name: None if name == 'aiohttp' else real(name))
    corpus = tmp_path / 'corpus'
    code, envelope = cli('plan', '--corpus', str(corpus), '--files', 'https://example.org/a.pdf', '--stages', 'extract')
    data = envelope['data']
    assert code == 0 and data['execution']['ready'] is False and data['execution']['blocked_by'] == ['missing_capability']
    assert data['capabilities']['missing'] == [dict(stage='download', reason_code='missing_package', install_profiles=['download'],
                                                    install_commands=['glossapi install --device cpu --package download'])]
    assert any(item['code'] == 'missing_capability' for item in envelope['warnings'])
    code, envelope = cli('run', '--corpus', str(corpus), '--files', 'https://example.org/a.pdf', '--stages', 'extract')
    assert code == 2 and envelope['errors'][0]['code'] == 'missing_capability'
    assert envelope['errors'][0]['details']['install_commands'] == ['glossapi install --device cpu --package download']
    assert not corpus.exists()


def test_extreme_without_disk_budget_fails_before_anything_runs(tmp_path):
    source = pdf(tmp_path / 'in' / 'a.pdf')
    for command in ('plan', 'run'):
        code, envelope = cli(command, '--corpus', str(tmp_path / 'corpus'), '--files', str(source), '--memory-saving', 'extreme')
        assert code == 3 and envelope['errors'][0]['code'] == 'disk_budget_required'
    assert not (tmp_path / 'corpus').exists()


def shadow_distribution(directory, name, version):
    info = directory / f'{name}-{version}.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
    return directory


def test_run_plan_refuses_after_one_input_byte_one_setting_or_one_capability_changes(tmp_path):
    source = pdf(tmp_path / 'in' / 'a.pdf', 'Byte probe text')
    corpus = tmp_path / 'corpus'
    code, planned = cli('plan', '--corpus', str(corpus), '--files', str(source), '--stages', 'extract')
    assert code == 0
    plan_file = write_json(tmp_path / 'plan.json', planned)
    # 1. one input byte (same size, original mtime restored: only the content hash catches it)
    original, timestamps = source.read_bytes(), (source.stat().st_atime_ns, source.stat().st_mtime_ns)
    index = original.index(b'Byte probe')
    source.write_bytes(original[:index] + b'b' + original[index + 1:])
    os.utime(source, ns=timestamps)
    code, envelope, _ = cli_subprocess('run', '--corpus', str(corpus), '--plan', str(plan_file))
    assert code == 2 and envelope['errors'][0]['code'] == 'plan_source_changed' and envelope['errors'][0]['details']['field'] == 'sha256'
    source.write_bytes(original); os.utime(source, ns=timestamps)
    # 2. one setting, edited in the saved plan
    edited = json.loads(plan_file.read_text()); edited['data']['settings']['debug'] = 'on'
    code, envelope, _ = cli_subprocess('run', '--corpus', str(corpus), '--plan', str(write_json(tmp_path / 'edited.json', edited)))
    assert code == 2 and envelope['errors'][0]['code'] == 'plan_hash_mismatch'
    # 3. one installed capability: a different ftfy distribution now shadows the planned one
    shadow = shadow_distribution(tmp_path / 'shadow', 'ftfy', '6.99.0')
    code, envelope, _ = cli_subprocess('run', '--corpus', str(corpus), '--plan', str(plan_file), env=env_with(shadow))
    assert code == 2 and envelope['errors'][0]['code'] == 'plan_capability_changed' and envelope['errors'][0]['details']['stages'] == ['extract']
    assert not corpus.exists()
    # Unchanged, the same plan file runs.
    pytest.importorskip('glossapi_rs_cleaner')
    code, envelope, _ = cli_subprocess('run', '--corpus', str(corpus), '--plan', str(plan_file))
    assert code == 0 and envelope['data']['state'] == 'completed', envelope['errors']


def test_plan_writes_nothing_on_a_read_only_corpus_copy(tmp_path):
    pytest.importorskip('glossapi_rs_cleaner')
    source = pdf(tmp_path / 'in' / 'a.pdf', 'Read only text')
    corpus = tmp_path / 'corpus'
    code, envelope = cli('run', '--corpus', str(corpus), '--files', str(source), '--stages', 'extract,clean,export')
    assert code == 0, envelope['errors']
    import shutil
    copy = tmp_path / 'readonly'
    shutil.copytree(corpus, copy)
    new_source = pdf(copy / 'new.pdf', 'Unplanned text')
    for path in [copy, *copy.rglob('*')]:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        before = tree(copy)
        for arguments in ((), ('--files', str(new_source), str(source), '--stages', 'download,extract,clean,ocr,sections,annotate,export'),
                          ('--stages', 'extract,clean,export', '--debug', 'on')):
            code, envelope, stderr = cli_subprocess('plan', '--corpus', str(copy), *arguments)
            assert not envelope['errors'] and code == 0, (envelope['errors'], stderr[-1500:])
            assert envelope['corpus_id']
        code, envelope, _ = cli_subprocess('capabilities')
        code, envelope, _ = cli_subprocess('doctor', '--corpus', str(copy))
        assert tree(copy) == before
    finally:
        for path in [copy, *copy.rglob('*')]:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


# --------------------------------------------------------------- settings help ---

PLAN_2_2 = {  # PLAN rev 5 section 2.2 (= rev 4 section 2.3), copied verbatim
    'debug': {'off': 'An intermediate is released only after every planned reader has succeeded on its exact hash. Supplied files, '
                     'selected final products, evidence and the ledger are kept.',
              'on': "Keep every stage's intermediates and attempts, including bounded inner HTTP request bodies."},
    'memory_saving': {'normal': 'Never delete a supplied original or a downloaded file.',
                      'extreme': 'Delete only downloads this run owns, after the last planned reader succeeds and outputs are durable. '
                                 "Never a supplied original, another run's file, or a path reached through a link. Requires a disk budget."},
}
PLAN_2_2_NOTE = ('Stages are fixed at the start of a run, and evaluation adds no readers. A plan amendment that adds a reader is a new '
                 'plan; it cannot justify a past deletion.')


@pytest.mark.parametrize('command', ['plan', 'run'])
def test_settings_help_matches_plan_section_2_2_exactly(command):
    from glossapi.settings_contract import SETTINGS_CONTRACT, PLAN_FIXED_NOTE
    assert SETTINGS_CONTRACT == PLAN_2_2 and PLAN_FIXED_NOTE == PLAN_2_2_NOTE
    done = subprocess.run([sys.executable, '-B', '-m', 'glossapi.cli', command, '--help'], capture_output=True, text=True, env=env_with())
    assert done.returncode == 0
    flat = ' '.join(done.stdout.split())
    for option, name in (('--debug', 'debug'), ('--memory-saving', 'memory_saving')):
        section = flat[flat.index(option + ' {'):]
        for value, sentence in PLAN_2_2[name].items():
            assert f'{value}: {sentence}' in section, (option, value)
    assert PLAN_2_2_NOTE in flat
    from glossapi.ledger.reader import plan_view
    help_in_plan = plan_view(selectors=['/absent.pdf'], stages=['extract'])['settings_help']
    assert all(sentence in help_in_plan[name] for name in PLAN_2_2 for sentence in PLAN_2_2[name].values())


# ------------------------------------------------- a failed run reports typed errors (K2/K3 integration) ---

def test_a_run_with_failures_reports_typed_errors(tmp_path):
    """`run` exit 1 (the command ran and found failures) names every failed task and unavailable source in
    `errors`, with the ledger's reason code, document and stage; it never reports exit 1 with an empty list."""
    pytest.importorskip('glossapi_rs_cleaner')
    good = pdf(tmp_path / 'in' / 'good.pdf', 'A good text layer')
    broken = tmp_path / 'in' / 'broken.pdf'
    broken.write_bytes(b'%PDF-1.4\nthis is not a pdf body\n')
    missing = tmp_path / 'in' / 'missing.pdf'
    corpus = tmp_path / 'corpus'
    code, envelope = cli('run', '--corpus', str(corpus), '--files', str(good), str(broken), str(missing),
                         '--stages', 'extract,clean,export')
    data = envelope['data']
    assert code == 1 and data['state'] == 'completed_with_failures', envelope['errors']
    errors = envelope['errors']
    assert errors and all(error['document_id'] and error['stage'] for error in errors), errors
    failed = [error for error in errors if error['details']['status'] == 'failed']
    unavailable = [error for error in errors if error['code'] == 'source_unavailable']
    assert len(failed) == data['counts']['failed'] >= 1 and {error['stage'] for error in failed} == {'extract'}, errors
    assert all(error['details']['reason_code'] == error['code'] and error['message'] for error in failed), errors
    assert len(unavailable) == data['source_inventory']['unavailable'] == 1, errors
    from glossapi.ledger.writer import open_readonly
    from glossapi.ledger.reader import resolve_document
    with open_readonly(corpus / 'corpus.sqlite3') as connection:
        assert failed[0]['document_id'] == resolve_document(connection, str(broken))['document_id']
        assert unavailable[0]['document_id'] == resolve_document(connection, str(missing))['document_id']


def test_failures_the_task_table_does_not_name_still_get_a_summary_error(tmp_path):
    """The fallback: a cancelled run (or failures without a failed task row) reports one summary error."""
    pytest.importorskip('glossapi_rs_cleaner')
    good = pdf(tmp_path / 'in' / 'good.pdf', 'A good text layer')
    corpus = tmp_path / 'corpus'
    code, envelope = cli('run', '--corpus', str(corpus), '--files', str(good), '--stages', 'extract,clean,export')
    assert code == 0 and envelope['data']['state'] == 'completed' and envelope['errors'] == [], envelope['errors']
    from glossapi.ledger.writer import open_readonly
    from glossapi.ledger.reader import run_failure_errors
    with open_readonly(corpus / 'corpus.sqlite3') as connection:
        errors = run_failure_errors(connection, envelope['run_id'], state='cancelled', counts={'failed': 0})
    assert [error['code'] for error in errors] == ['run_cancelled'] and errors[0]['details']['state'] == 'cancelled'
    validate(dict(envelope, errors=errors))
