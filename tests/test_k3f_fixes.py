"""CP-K3 fix set K3F: the reviewer's counterexamples (review_scratch/CP-K3/probes.py, test_adversarial.py) as live tests.

- F1: an explicitly empty selection selects nothing; plan previews zero targets with ``execution.ready`` false and the
  blocking conflict ``empty_selection``; run (manifest or saved plan) refuses with that typed error before the ledger is
  touched. An omitted selection still means every corpus document.
- F2: a non-metadata writer (``write_manifest``, the CLI exporter) refuses a hard-linked destination
  (``MetadataAliasError``) and a destination that resolves to metadata through a symlink or ``..``
  (``MetadataDestinationError``) before any byte is written.
- F3: a fresh plan binds the currently observed bytes of every present local source, ledger-known ones included, so
  plan -> run of an edited, previously processed PDF succeeds; a change after planning is still refused.
- F4: the guide's plan/run and manifest error rows come from the owners' registries, and the consistency check reads
  the codes the owners' live functions actually raise; mutating an owner is caught.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from test_k3_interface import cli, cli_subprocess, pdf, write_json

pytest.importorskip('jsonschema')


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runs(corpus):
    with sqlite3.connect(f'file:{corpus / "corpus.sqlite3"}?mode=ro', uri=True) as connection:
        return connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0]


def _seeded(tmp_path, *names):
    sources = [pdf(tmp_path / 'in' / name, 'Seed text ' + name[0]) for name in names]
    corpus = tmp_path / 'corpus'
    code, envelope, stderr = cli_subprocess('run', '--corpus', str(corpus), '--files', *map(str, sources), '--stages', 'extract')
    assert code == 0 and envelope['data']['state'] == 'completed', (envelope['errors'], stderr[-2000:])
    return corpus, sources


# --------------------------------------------------------------------- F1 ---

@pytest.mark.parametrize('shape', ['v1-selection-empty', 'pre-v1-empty-list'])
def test_explicit_empty_selection_selects_nothing_and_run_refuses(tmp_path, shape):
    corpus, (a, b) = _seeded(tmp_path, 'a.pdf', 'b.pdf')
    runs_before = _runs(corpus)
    if shape == 'v1-selection-empty':
        value = {'manifest_version': 1, 'documents': [{'path': str(a), 'selection_id': 'a'}, {'path': str(b), 'selection_id': 'b'}],
                 'selection': [], 'stages': ['extract']}
    else:
        value = []
    manifest = write_json(tmp_path / 'empty.json', value)
    arguments = ['--corpus', str(corpus), '--manifest', str(manifest)] + ([] if shape.startswith('v1') else ['--stages', 'extract'])
    code, planned = cli('plan', *arguments)
    data = planned['data']
    assert code == 0
    assert data['targets'] == [] and data['documents'] == []
    assert data['execution']['ready'] is False and 'empty_selection' in data['execution']['blocked_by']
    assert [item['code'] for item in data['conflicts'] if item['code'] == 'empty_selection'] == ['empty_selection']
    assert 'empty_selection' in [warning['code'] for warning in planned['warnings']]
    if shape.startswith('v1'):
        assert data['input_manifest']['selected'] == 0
        assert sorted(item['selector'] for item in data['exclusions']) == sorted([str(a), str(b)])
        assert all(item['reason_code'] == 'not_selected' for item in data['exclusions'])
    # run with the manifest: typed refusal before the ledger is touched
    code, ran, stderr = cli_subprocess('run', *arguments)
    assert code == 2 and ran['errors'][0]['code'] == 'empty_selection' and ran['data'] is None, (ran, stderr[-2000:])
    # run with the saved plan: the same refusal
    plan_file = write_json(tmp_path / 'plan.json', planned)
    code, ran, _ = cli_subprocess('run', '--corpus', str(corpus), '--plan', str(plan_file))
    assert code == 2 and ran['errors'][0]['code'] == 'empty_selection'
    assert _runs(corpus) == runs_before


def test_python_run_refuses_an_empty_selection_plan(tmp_path):
    from glossapi import Corpus
    from glossapi.cli import plan_documents
    from glossapi.plan_preview import PlanBindingError
    corpus, (a,) = _seeded(tmp_path, 'a.pdf')
    data = plan_documents(corpus, manifest={'manifest_version': 1, 'documents': [{'path': str(a), 'selection_id': 'a'}],
                                            'selection': [], 'stages': ['extract']})
    assert data['targets'] == [] and data['execution']['ready'] is False
    runs_before = _runs(corpus)
    with pytest.raises(PlanBindingError) as error:
        Corpus(corpus, corpus).run(plan=data)
    assert error.value.code == 'empty_selection'
    assert _runs(corpus) == runs_before


def test_omitted_selection_still_means_every_corpus_document(tmp_path):
    corpus, sources = _seeded(tmp_path, 'a.pdf', 'b.pdf')
    code, planned = cli('plan', '--corpus', str(corpus), '--stages', 'extract')
    assert code == 0 and sorted(target['path'] for target in planned['data']['targets']) == sorted(map(str, sources))
    assert 'empty_selection' not in planned['data']['execution']['blocked_by']
    # A v1 manifest without "selection" selects every entry.
    manifest = write_json(tmp_path / 'all.json', {'manifest_version': 1, 'stages': ['extract'],
                                                  'documents': [{'path': str(source)} for source in sources]})
    code, planned = cli('plan', '--corpus', str(corpus), '--manifest', str(manifest))
    assert code == 0 and len(planned['data']['targets']) == 2 and planned['data']['execution']['ready'] is True


# --------------------------------------------------------------------- F2 ---

def _metadata(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    target = tmp_path / 'corpus' / 'download_results' / 'download_results.parquet'
    target.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{'filename': 'a.pdf', 'title': 'committed'}]), target)
    return target


def _destination(tmp_path, target, spelling):
    if spelling == 'hard-link':
        alias = tmp_path / 'manifest.json'
        os.link(target, alias)
        return alias
    if spelling == 'symlink':
        alias = tmp_path / 'manifest.json'
        alias.symlink_to(target)
        return alias
    if spelling == 'dot-dot':
        (tmp_path / 'corpus' / 'other').mkdir()
        return tmp_path / 'corpus' / 'other' / '..' / 'download_results' / 'download_results.parquet'
    # '..' after a directory symlink climbs from the link's target (traversal order, as the OS does)
    real = tmp_path / 'corpus' / 'download_results' / 'nested'
    real.mkdir()
    (tmp_path / 'link').symlink_to(real, target_is_directory=True)
    return tmp_path / 'link' / '..' / 'download_results.parquet'


def _expected(spelling):
    from glossapi.parquet_schema import MetadataAliasError, MetadataDestinationError
    return MetadataAliasError if spelling == 'hard-link' else MetadataDestinationError


SPELLINGS = ['hard-link', 'symlink', 'dot-dot', 'dir-symlink-dot-dot']


@pytest.mark.parametrize('spelling', SPELLINGS)
def test_write_manifest_refuses_metadata_aliases_before_truncation(tmp_path, spelling):
    from glossapi.input_manifest import build_manifest, write_manifest
    target = _metadata(tmp_path)
    before = target.read_bytes()
    destination = _destination(tmp_path, target, spelling)
    with pytest.raises(_expected(spelling)):
        write_manifest(build_manifest([{'path': '/absolute/a.pdf'}]), destination)
    assert target.read_bytes() == before
    if spelling == 'hard-link':
        assert target.stat().st_nlink == 2 and destination.read_bytes() == before
    if spelling == 'symlink':
        assert destination.is_symlink()


@pytest.mark.parametrize('spelling', SPELLINGS)
def test_cli_export_refuses_metadata_aliases_before_writing(tmp_path, spelling):
    from glossapi.cli import _export
    target = _metadata(tmp_path)
    before = target.read_bytes()
    destination = _destination(tmp_path, target, spelling)
    listing = sorted(os.listdir(destination.parent))
    with pytest.raises(_expected(spelling)):
        _export(destination, {'rows': [{'filename': 'new.pdf'}]}, 'json')
    assert target.read_bytes() == before and sorted(os.listdir(destination.parent)) == listing


def test_any_hard_linked_destination_is_refused_and_plain_destinations_still_work(tmp_path):
    """A hard-linked name cannot be told apart from a metadata alias without searching the filesystem, so every
    hard-linked destination is refused; an ordinary (or new) destination is written normally."""
    from glossapi.cli import _export
    from glossapi.input_manifest import build_manifest, write_manifest
    from glossapi.parquet_schema import MetadataAliasError
    other = tmp_path / 'notes.json'; other.write_text('keep')
    alias = tmp_path / 'out.json'; os.link(other, alias)
    with pytest.raises(MetadataAliasError):
        write_manifest(build_manifest([{'path': '/absolute/a.pdf'}]), alias)
    with pytest.raises(MetadataAliasError):
        _export(alias, {'rows': []}, 'json')
    assert other.read_text() == 'keep'
    written = write_manifest(build_manifest([{'path': '/absolute/a.pdf'}]), tmp_path / 'fresh.json')
    assert json.loads(written.read_text())['manifest_version'] == 1
    existing = tmp_path / 'existing.json'; existing.write_text('old')
    write_manifest(build_manifest([{'path': '/absolute/a.pdf'}]), existing)
    assert json.loads(existing.read_text())['manifest_version'] == 1
    assert _export(tmp_path / 'report.json', {'rows': []}, 'json')['bytes'] > 0


# --------------------------------------------------------------------- F3 ---

def test_replanning_an_edited_known_source_binds_current_bytes_and_runs(tmp_path):
    pytest.importorskip('glossapi_rs_cleaner')
    corpus, (a,) = _seeded(tmp_path, 'a.pdf')
    old = _digest(a)
    a.write_bytes(a.read_bytes().replace(b'Seed text a', b'Edited text', 1))
    actual = _digest(a)
    assert actual != old
    code, planned = cli('plan', '--corpus', str(corpus), '--files', str(a), '--stages', 'extract')
    target = planned['data']['targets'][0]
    assert code == 0 and target['known_version_id'] is not None
    assert target['sha256'] == actual, {'preview': target['sha256'], 'actual': actual, 'previous': old}
    observation = planned['data']['documents'][0]['source_observation']
    assert observation['sha256'] == actual and observation['stored_sha256'] == old  # history kept separately
    plan_file = write_json(tmp_path / 'plan.json', planned)
    code, ran, stderr = cli_subprocess('run', '--corpus', str(corpus), '--plan', str(plan_file))
    assert code == 0 and ran['data']['state'] == 'completed', (ran['errors'], stderr[-2000:])
    code, status = cli('status', '--corpus', str(corpus), '--file', str(a))
    assert status['data']['documents'][0]['version_id'] != target['known_version_id']
    # The next plan reports the new identity as the stored one.
    code, again = cli('plan', '--corpus', str(corpus), '--files', str(a), '--stages', 'extract')
    assert again['data']['documents'][0]['source_observation']['stored_sha256'] == actual


def test_a_known_source_changed_after_planning_is_still_refused(tmp_path):
    corpus, (a,) = _seeded(tmp_path, 'a.pdf')
    code, planned = cli('plan', '--corpus', str(corpus), '--files', str(a), '--stages', 'extract')
    assert code == 0 and planned['data']['targets'][0]['sha256'] == _digest(a)
    plan_file = write_json(tmp_path / 'plan.json', planned)
    original, timestamps = a.read_bytes(), (a.stat().st_atime_ns, a.stat().st_mtime_ns)
    index = original.index(b'Seed text')
    a.write_bytes(original[:index] + b's' + original[index + 1:])  # same size, mtime restored: only the hash sees it
    os.utime(a, ns=timestamps)
    runs_before = _runs(corpus)
    code, ran, _ = cli_subprocess('run', '--corpus', str(corpus), '--plan', str(plan_file))
    assert code == 2 and ran['errors'][0]['code'] == 'plan_source_changed' and ran['errors'][0]['details']['field'] == 'sha256'
    assert _runs(corpus) == runs_before


def test_python_run_handoff_checks_the_observed_hash(tmp_path):
    """run_corpus (the Python handoff) compares against the hash observed at planning, not the historical one."""
    from glossapi import Corpus
    from glossapi.cli import plan_documents
    pytest.importorskip('glossapi_rs_cleaner')
    corpus, (a,) = _seeded(tmp_path, 'a.pdf')
    a.write_bytes(a.read_bytes().replace(b'Seed text a', b'Edited text', 1))
    data = plan_documents(corpus, files=[str(a)], stages=['extract'])
    outcome = Corpus(corpus, corpus).run(plan=data)
    assert outcome['state'] == 'completed'


# --------------------------------------------------------------------- F4 ---

def _recompiled(owner, old, new):
    source = Path(owner.__file__).read_text()
    changed = source.replace(old, new)
    assert changed != source, old
    namespace = {'__name__': owner.__name__, '__package__': 'glossapi'}
    exec(compile(changed, owner.__file__, 'exec'), namespace)
    return namespace


def test_guide_error_rows_come_from_the_owner_registries():
    from glossapi.input_manifest import ERRORS as MANIFEST_ERRORS, SUPPORTED_VERSIONS
    from glossapi.llm_guide import guide_view
    from glossapi.plan_preview import ERRORS as PLAN_RUN_ERRORS
    from test_k3b_interface import guide_disagreements, raised_codes
    import glossapi.input_manifest as manifest_owner
    import glossapi.plan_preview as plan_owner
    guide = guide_view()
    assert guide_disagreements(guide) == []
    for registry in (PLAN_RUN_ERRORS, MANIFEST_ERRORS):
        for code, (commands, exit_code, _meaning) in registry.items():
            assert guide['errors'][code]['command'] == commands and guide['errors'][code]['exit_code'] == exit_code
    # every code the owners raise is registered, and the registries list nothing the owners never raise
    assert raised_codes(plan_owner) <= set(PLAN_RUN_ERRORS) and raised_codes(manifest_owner) == set(MANIFEST_ERRORS)
    assert {'invalid_plan', 'empty_selection', 'plan_source_changed'} <= set(PLAN_RUN_ERRORS)
    assert ', '.join(map(str, SUPPORTED_VERSIONS)) in guide['errors']['unsupported_manifest_version']['meaning']


def test_reviewer_owner_mutation_of_a_binding_code_is_caught(monkeypatch):
    """CP-K3 F4 witness: plan_preview recompiled with its emitted plan_source_changed renamed."""
    import glossapi.plan_preview as owner
    from test_k3b_interface import guide_disagreements
    from glossapi.llm_guide import guide_view
    namespace = _recompiled(owner, "PlanBindingError('plan_source_changed'", "PlanBindingError('plan_source_altered'")
    monkeypatch.setattr(owner, 'verify_plan_binding', namespace['verify_plan_binding'])
    problems = guide_disagreements(guide_view())
    assert any('plan_source_altered' in problem for problem in problems), problems


@pytest.mark.parametrize('mutation', ['manifest-code', 'manifest-version', 'registry-entry', 'plan-blocker'])
def test_owner_mutations_are_caught(monkeypatch, mutation):
    import glossapi.input_manifest as manifest_owner
    import glossapi.plan_preview as plan_owner
    from glossapi.llm_guide import guide_view
    from test_k3b_interface import guide_disagreements
    if mutation == 'manifest-code':
        namespace = _recompiled(manifest_owner, "ManifestError('unsupported_manifest_version'", "ManifestError('manifest_version_unsupported'")
        monkeypatch.setattr(manifest_owner, 'parse_manifest', namespace['parse_manifest'])
        guide = guide_view()
    elif mutation == 'manifest-version':
        guide = guide_view()
        monkeypatch.setattr(manifest_owner, 'SUPPORTED_VERSIONS', (1, 2))
    elif mutation == 'registry-entry':
        guide = guide_view()
        registry = dict(plan_owner.ERRORS)
        registry.pop('plan_hash_mismatch')
        monkeypatch.setattr(plan_owner, 'ERRORS', registry)
    else:
        guide = guide_view()
        monkeypatch.setattr(plan_owner, 'RUN_BLOCKERS', plan_owner.RUN_BLOCKERS | {'new_blocker'})
    assert guide_disagreements(guide)
