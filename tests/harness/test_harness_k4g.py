"""K4g B4-N harness items: LS1-G01 (wrapper returns in golden receipts), LS5-G05 (diagnostics role), DW2-G03 (venv
footprint basis), T5-G07 (PDFium build and substitute-font identity). CPU only; no models or network."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.harness.common import STAGES, diagnostics, file_role, inventory, sha256, write_json
from tools.harness.compare import compare


def fake_golden(monkeypatch, returns):
    from tools.harness import golden

    class PDF:
        def __init__(self, path): pass
        def __len__(self): return 2
        def close(self): pass
    monkeypatch.setitem(sys.modules, 'pypdfium2', SimpleNamespace(PdfDocument=PDF))

    class FakeCorpus:
        def __init__(self, i, o, **kw):
            self.output_dir = o; self.sections_parquet = o/'sections.parquet'; self.fully_annotated_parquet = o/'annotated.parquet'
        def extract(self, **kw):
            (self.output_dir/'markdown').mkdir(); (self.output_dir/'markdown/a.md').write_text('raw')
            (self.output_dir/'logs').mkdir(); (self.output_dir/'logs/download_request.logs').write_text('request 1\n')
            return returns['extract']
        def clean(self, **kw):
            (self.output_dir/'clean_markdown').mkdir(); (self.output_dir/'clean_markdown/a.md').write_text('clean')
            return returns['clean']
        def section(self): self.sections_parquet.write_bytes(b's'); return returns['sections']
        def annotate(self): self.fully_annotated_parquet.write_bytes(b'a'); return returns['annotate']
        def jsonl(self, path): path.write_text('{}\n'); return returns['export'](path)

    class FakeRSS:
        peak = own_peak = 1
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(golden, 'PeakRSS', FakeRSS)
    monkeypatch.setattr(golden, 'Corpus', FakeCorpus)
    monkeypatch.setattr(golden, 'IMPORTS', {'pid': 1})
    return golden


def golden_run(tmp_path, monkeypatch, name, returns):
    golden = fake_golden(monkeypatch, returns)
    data = tmp_path/'data'; data.mkdir(exist_ok=True); pdf = data/'a.pdf'; pdf.write_bytes(b'tiny')
    root = (tmp_path/name).resolve()
    document = golden.document_job(dict(filename='a.pdf', sha256=sha256(pdf)), data, root, 'text-layer')
    manifest = [dict(filename='a.pdf', sha256=sha256(pdf))]
    write_json(root/'run.json', dict(schema=golden.SCHEMA, complete=True, profile='text-layer', manifest_sha256='b'*64,
                                     manifest=manifest, documents=[document]))
    return root, document


RETURNS = dict(extract=None, clean={'documents': 1}, sections=None, annotate=None, export=lambda path: path)


def test_golden_records_typed_wrapper_returns_with_only_the_run_root_normalised(tmp_path, monkeypatch):
    root, document = golden_run(tmp_path, monkeypatch, 'run-a', RETURNS)
    stages = document['stages']
    assert all(state['status'] == 'ok' and 'return' in state for state in stages.values())
    assert stages['extract']['return'] == {'type': 'null', 'value': None}
    assert stages['clean']['return']['type'] == 'mapping'
    # A returned path under the run root is recorded relative to it; nothing else is rewritten.
    assert stages['export']['return'] == {'type': 'path', 'value': '<run>/documents/a/work/export.jsonl'}
    assert json.loads((root/'run.json').read_text())['schema'] == 2


def test_return_only_mutation_fails_the_comparison(tmp_path, monkeypatch):
    a, _ = golden_run(tmp_path, monkeypatch, 'run-a', RETURNS)
    b, _ = golden_run(tmp_path, monkeypatch, 'run-b', RETURNS)
    same = compare(a, b)
    assert same['identical'] and same['passes'] and same['wrapper_returns'] == 'compared'
    c, _ = golden_run(tmp_path, monkeypatch, 'run-c', dict(RETURNS, clean={'documents': 2}))
    changed = compare(a, c)
    assert not changed['identical'] and not changed['passes'] and changed['undeclared_differences'] == 1
    (row,) = [r for r in changed['results'] if r['differences']]
    assert row['stage'] == 'clean' and row['differences'][0]['path'] == '@return'
    # A return that escapes the run root keeps its absolute path, so moving it is visible too.
    d, _ = golden_run(tmp_path, monkeypatch, 'run-d', dict(RETURNS, export=lambda path: Path('/elsewhere/export.jsonl')))
    assert not compare(a, d)['passes']


def test_schema_one_receipts_stay_comparable_and_say_returns_were_not_compared(tmp_path, monkeypatch):
    a, _ = golden_run(tmp_path, monkeypatch, 'run-a', RETURNS)
    old, _ = golden_run(tmp_path, monkeypatch, 'run-old', dict(RETURNS, clean={'documents': 9}))
    value = json.loads((old/'run.json').read_text())
    value['schema'] = 1
    for state in value['documents'][0]['stages'].values():
        state.pop('return'); state.pop('diagnostics')
    write_json(old/'run.json', value)
    result = compare(old, a)
    assert result['passes'] and result['wrapper_returns'].startswith('not compared')
    # A schema-2 receipt that lost a stage's return is an integrity failure, not a silent pass.
    value = json.loads((a/'run.json').read_text()); value['documents'][0]['stages']['annotate'].pop('return')
    write_json(a/'run.json', value)
    broken = compare(a, a)
    assert not broken['passes'] and any('without a wrapper return' in e for e in broken['integrity_errors'])


def test_downloader_logs_are_diagnostics_not_products(tmp_path, monkeypatch):
    root = tmp_path/'out'
    for name, text in (('logs/download_request.logs', 'r'), ('logs/download_progress.logs', 'p'), ('conversion.log', 'c'),
                       ('markdown/a.md', 'm'), ('notes.logs', 'n'), ('state.lock', 'l'), ('downloads/.harness-direct-paths', '')):
        (root/name).parent.mkdir(parents=True, exist_ok=True); (root/name).write_text(text)
    assert sorted(inventory(root)) == ['markdown/a.md', 'notes.logs']  # no broad .logs waiver outside logs/
    assert sorted(diagnostics(root)) == ['conversion.log', 'logs/download_progress.logs', 'logs/download_request.logs']
    assert diagnostics(root)['logs/download_request.logs']['sha256'] == sha256(root/'logs/download_request.logs')
    assert [file_role(n) for n in ('a/logs/x.logs', 'x.logs', 'x.lock')] == ['diagnostic', 'product', 'ignored']
    # The golden receipt keeps the downloader log as a hashed diagnostic of the stage that wrote it, not a product.
    _, document = golden_run(tmp_path, monkeypatch, 'run-a', RETURNS)
    extract = document['stages']['extract']
    assert [f['path'] for f in extract['diagnostics']] == ['logs/download_request.logs']
    assert 'logs/download_request.logs' not in {f['path'] for f in extract['files']}


def test_venv_footprint_separates_own_and_resolved_bytes(tmp_path, monkeypatch):
    from tools.harness import common
    shared = tmp_path/'shared'; shared.mkdir(); (shared/'big.bin').write_bytes(b'\0' * (512 * 1024))
    prefix = tmp_path/'venv'; prefix.mkdir(); (prefix/'lib').symlink_to(shared, target_is_directory=True)
    monkeypatch.setattr(common.sys, 'prefix', str(prefix))
    footprint = common.venv_footprint()
    assert footprint['venv_resolved_bytes'] >= 512 * 1024 > footprint['venv_own_bytes']
    assert 'du -skL' in footprint['venv_bytes_basis']
    assert 'venv_disk_bytes' not in footprint


def test_environment_binds_the_pdfium_build_and_its_substitute_font():
    pytest.importorskip('pypdfium2')
    from tools.harness.common import pdfium_identity
    identity = pdfium_identity()
    assert identity['available'] and identity['pypdfium2'] and identity.get('pdfium_info')
    font = identity['unembedded_arial_substitute']
    assert not font['embedded'] and font['bytes'] > 0 and len(font['sha256']) == 64
    assert identity['native_library'] and all(len(v) == 64 for v in identity['native_library'].values())
