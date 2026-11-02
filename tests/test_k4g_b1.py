"""K4g batch B1: ledger/stages/CLI gap fixes. Each test names its GAPS id; each fails on 7c9bd37 unless marked
"(behaviour kept)"."""
import os

import pytest

from glossapi.ledger import LedgerWriter
from glossapi.stages.execution import sha256_file


# ======================================================= K4F1-G1 case (d): by content ===

SOURCE = b'{"a": 1}\n'*1000


def _published(writer, root, *, before_barrier=None, earlier=None, hook=None):
    """A retained managed source and its user-visible byte copy (as ``test_k4c2_fixes.published_copy``).

    ``before_barrier(copy)`` changes the copy after registration and before the run-close barrier observes its
    identity: the barrier then records the changed copy's own (mtime, ctime), exactly what a write within the
    barrier's clock tick leaves on a coarse-clock filesystem. ``earlier``: bytes an earlier attempt registered at the
    same path. ``hook(writer, copy)`` runs inside the barrier, right after the identity observation."""
    source = root/'.glossapi/artifacts/corpus/export/attempt/export.jsonl'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(SOURCE)
    copy = root/'export.jsonl'
    with writer.transaction():
        run = writer.run({}, {}, {}, code_revision='k4g-b1-test')
        manifest = writer.snapshot('inputs', {'kind': 'inputs', 'files': []})
        task = writer.task(run, 'export', 'export.jsonl', {}, scope_key='corpus:'+manifest)
        if earlier is not None:
            old_task = writer.task(run, 'export', 'earlier.jsonl', {}, scope_key='corpus:'+manifest)
            old_attempt = writer.attempt(old_task, worker_id='w', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
            copy.write_bytes(earlier)
            writer.artifact(run, 'export', 'document_jsonl', copy.as_uri(), producer_attempt_id=old_attempt,
                            sha256=sha256_file(copy), bytes=len(earlier), retention_class='final')
        attempt = writer.attempt(task, worker_id='w', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        copy.write_bytes(SOURCE)
        digest, size = sha256_file(source), source.stat().st_size
        writer.artifact(run, 'export', 'document_jsonl', source.as_uri(), producer_attempt_id=attempt, sha256=digest,
                        bytes=size, retention_class='final')
        copy_id = writer.artifact(run, 'export', 'document_jsonl', copy.as_uri(), producer_attempt_id=attempt, sha256=digest,
                                  bytes=size, retention_class='final')
    if before_barrier:
        before_barrier(copy)
    writer.barrier_context = {'run_id': run, 'debug': 'off'}
    if hook:
        original = writer.event
        state = {'done': False}
        def event(*args, **kwargs):
            if not state['done']:
                state['done'] = True
                hook(writer, copy)
            return original(*args, **kwargs)
        writer.event = event
    writer.durable_barrier()
    return copy, source, copy_id


def _start(path):
    with LedgerWriter(path) as writer:
        restored = writer.verify_excluded_copies()
        return restored, [dict(entry) for entry in writer.ambiguous_copies]


def _rewrite(data):
    def write(copy):
        copy.write_bytes(data)
    return write


@pytest.mark.parametrize('edit', [
    b'MY EDIT, a complete file\n',                     # shorter, different
    SOURCE.replace(b'"a"', b'"b"'),                    # same size, different bytes
    SOURCE + b'{"mine": true}\n',                      # the managed bytes plus a user's line
    b'X' + SOURCE[1:],                                 # one byte changed in place
])
def test_k4f1g1_same_tick_user_edit_is_kept_and_refused(tmp_path, edit):
    """K4F1-G1 (d): an edit whose timestamps equal the barrier's identity was overwritten on 7c9bd37."""
    ledger = tmp_path/'corpus.sqlite3'
    with LedgerWriter(ledger) as writer:
        copy, source, copy_id = _published(writer, tmp_path, before_barrier=_rewrite(edit))
    restored, ambiguous = _start(ledger)
    assert restored == []
    assert [(entry['file_id'], entry['reason']) for entry in ambiguous] == [(copy_id, 'complete_different_content')]
    assert copy.read_bytes() == edit and source.read_bytes() == SOURCE  # both kept


def test_k4f1g1_same_tick_edit_refuses_every_start_until_resolved(tmp_path):
    from glossapi.ledger.writer import PostCloseEditError
    ledger = tmp_path/'corpus.sqlite3'
    with LedgerWriter(ledger) as writer:
        copy, source, copy_id = _published(writer, tmp_path, before_barrier=_rewrite(b'mine\n'))
    for _ in range(2):
        with LedgerWriter(ledger) as writer:
            writer.verify_excluded_copies()
            with pytest.raises(PostCloseEditError, match='post_close_edit'):
                writer.refuse_ambiguous_copies()
    copy.unlink()
    assert _start(ledger) == ([copy_id], [])
    assert copy.read_bytes() == SOURCE


@pytest.mark.parametrize('torn', [
    b'',                                                             # nothing written back
    SOURCE[:100],                                                    # a prefix
    SOURCE[:4096],                                                   # a page-aligned prefix
    bytes(4096) + SOURCE[4096:],                                     # first page never written back
    SOURCE[:512] + bytes(1024) + SOURCE[1536:6000],                  # a hole, then a prefix
    bytes(len(SOURCE)),                                              # all zeros, full size
])
def test_k4f1g1_torn_images_are_still_republished(tmp_path, torn):
    """(behaviour kept) every torn image of the managed bytes is restored from the managed source."""
    ledger = tmp_path/'corpus.sqlite3'
    with LedgerWriter(ledger) as writer:
        copy, source, copy_id = _published(writer, tmp_path, before_barrier=_rewrite(torn))
    assert _start(ledger) == ([copy_id], [])
    assert copy.read_bytes() == SOURCE


def test_k4f1g1_rolled_back_replace_restores_the_earlier_registered_copy(tmp_path):
    """(behaviour kept) an undurable rename leaves the previous registered copy at the path: restored."""
    ledger = tmp_path/'corpus.sqlite3'
    earlier = b'{"old": 1}\n'*50
    with LedgerWriter(ledger) as writer:
        copy, source, copy_id = _published(writer, tmp_path, earlier=earlier, before_barrier=_rewrite(earlier))
    assert _start(ledger) == ([copy_id], [])
    assert copy.read_bytes() == SOURCE


def test_k4f1g1_a_symlink_at_the_copy_path_is_never_replaced(tmp_path):
    ledger = tmp_path/'corpus.sqlite3'
    target = tmp_path/'mine.jsonl'
    target.write_bytes(b'mine\n')
    def link(copy):
        copy.unlink()
        copy.symlink_to(target)
    with LedgerWriter(ledger) as writer:
        copy, source, copy_id = _published(writer, tmp_path, before_barrier=link)
    restored, ambiguous = _start(ledger)
    assert restored == [] and [entry['file_id'] for entry in ambiguous] == [copy_id]
    assert copy.is_symlink() and target.read_bytes() == b'mine\n'


def test_k4f1g1_coarse_clock_same_tick_edit_is_kept(tmp_path):
    """Real clock: the user's edit lands in the barrier's own timestamp tick (Linux coarse clock). It skips where the
    filesystem stamps distinct times for every write (APFS on the Mac); on the node it runs."""
    attempts = 0
    for attempt in range(40):
        root = tmp_path/f'try{attempt}'
        root.mkdir()
        ledger = root/'corpus.sqlite3'
        edit = b'USER EDIT IN THE SAME TICK\n'
        def same_tick(writer, copy):
            copy.write_bytes(edit)  # in place, right after the barrier observed the copy's identity
        with LedgerWriter(ledger) as writer:
            copy, source, copy_id = _published(writer, root, hook=same_tick)
            import json
            details = json.loads(writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} "
                                                           "WHERE message='barrier_summary'").fetchone()[0])
        recorded = details['run_close_excluded_copy_identities'][0]
        info = copy.stat()
        attempts += 1
        if [info.st_mtime_ns, info.st_ctime_ns] != recorded:
            continue  # the write crossed a tick (or the clock is fine-grained): try again
        restored, ambiguous = _start(ledger)
        assert restored == [] and [(entry['file_id'], entry['reason']) for entry in ambiguous] == [(copy_id, 'complete_different_content')]
        assert copy.read_bytes() == edit
        return
    pytest.skip(f'no same-tick write in {attempts} attempts: this filesystem clock is fine-grained')


# ================================================== K4F1-G1 case (c): racy-git rule ===

def test_k4f1g1_fresh_original_is_never_bound_from_the_cache(tmp_path, monkeypatch):
    """K4F1-G1 (c): an original written within the margin of its observation is not cached, so a later rewrite in the
    same tick (same size, same mtime and ctime) cannot be bound to the old hash by a non-reading phase."""
    import hashlib
    from glossapi.ledger import journal
    monkeypatch.setattr(journal, '_original_hashes', {})
    from glossapi.ledger import encoding
    original = tmp_path/'source.pdf'
    original.write_bytes(b'%PDF one')
    # The same-tick rewrite below keeps the stat identity: freeze it (a coarse clock does this for real).
    frozen = encoding.own_hash_key(os.stat(original))
    monkeypatch.setattr(encoding, 'own_hash_key', lambda info: frozen)
    assert journal.observed_original_hash('extract', original) == hashlib.sha256(b'%PDF one').hexdigest()
    assert journal._original_hashes == {}
    original.write_bytes(b'%PDF two')
    assert journal.observed_original_hash('clean', original) == hashlib.sha256(b'%PDF two').hexdigest()


def test_k4f1g1_an_original_older_than_the_margin_is_still_cached(tmp_path, monkeypatch):
    """(behaviour kept) user originals are older than the margin: non-reading phases bind without reading."""
    import time
    from glossapi.ledger import journal
    monkeypatch.setattr(journal, '_original_hashes', {})
    original = tmp_path/'source.pdf'
    original.write_bytes(b'%PDF old')
    time.sleep(getattr(journal, 'RACY_MARGIN_NS', 0)/1e9 + 0.1)
    digest = journal.observed_original_hash('extract', original)
    reads = []
    monkeypatch.setattr(journal, 'file_hash', lambda path, **k: reads.append(path) or 'x')
    assert journal.observed_original_hash('clean', original) == digest and reads == []


# ================================================================== CRED-F8 url_index ===

def _plan_keys(tmp_path, documents):
    from glossapi.cli import plan_documents
    data = plan_documents(tmp_path/'corpus', manifest={'documents': documents}, stages=['extract'])
    return [target['source_key'] for target in data['documents']], data


def _run_keys(tmp_path, documents):
    from glossapi.stages.run import _remote_inventory
    _, _, identities, _ = _remote_inventory([dict(row) for row in documents], tmp_path/'corpus')
    return [identity for identity in identities if identity]


TWO_URLS_ONE_SOURCE = [{'url': 'https://a.example/x.pdf', 'source_id': 'S1', 'url_index': 0},
                       {'url': 'https://a.example/y.pdf', 'source_id': 'S1', 'url_index': 1}]


def test_credf8_url_index_keys_two_targets_in_plan_and_run_alike(tmp_path):
    """CRED-F8: on 7c9bd37 plan showed 1 target (ready) and run raised an untyped ValueError."""
    planned, data = _plan_keys(tmp_path, TWO_URLS_ONE_SOURCE)
    assert len(planned) == 2 and len(set(planned)) == 2 and data['execution']['ready']
    assert _run_keys(tmp_path, TWO_URLS_ONE_SOURCE) == planned


def test_credf8_expanded_list_row_and_declared_rows_key_alike(tmp_path):
    """(behaviour kept for the list cell) run keys a list cell by position; the same resources declared one per row
    with url_index get the same identities, and plan agrees."""
    listed = _run_keys(tmp_path, [{'url': ['https://a.example/x.pdf', 'https://a.example/y.pdf'], 'source_id': 'S1'}])
    planned, _ = _plan_keys(tmp_path, TWO_URLS_ONE_SOURCE)
    assert listed == planned == _run_keys(tmp_path, TWO_URLS_ONE_SOURCE)


@pytest.mark.parametrize('documents', [
    [{'url': 'https://a.example/x.pdf', 'source_id': 'S1'}, {'url': 'https://a.example/y.pdf', 'source_id': 'S1'}],
    [{'url': 'https://a.example/x.pdf', 'source_id': 'S1', 'url_index': 1}, {'url': 'https://a.example/y.pdf', 'source_id': 'S1', 'url_index': 1}],
], ids=['no-index', 'same-index'])
def test_credf8_two_urls_on_one_identity_refused_typed_in_plan_and_run(tmp_path, documents):
    from glossapi.input_manifest import ManifestError
    for function in (_plan_keys, _run_keys):
        with pytest.raises(ManifestError) as caught:
            function(tmp_path, documents)
        assert caught.value.code == 'invalid_manifest' and caught.value.details['reason'] == 'ambiguous_source_identity'
        assert caught.value.details['index'] == 1


@pytest.mark.parametrize('value', [-1, 1.5, 'one', True])
def test_credf8_invalid_url_index_refused_typed_in_plan_and_run(tmp_path, value):
    from glossapi.input_manifest import ManifestError
    documents = [{'url': 'https://a.example/x.pdf', 'source_id': 'S1', 'url_index': value}]
    for function in (_plan_keys, _run_keys):
        with pytest.raises(ManifestError) as caught:
            function(tmp_path, documents)
        assert caught.value.code == 'invalid_manifest' and caught.value.details['field'] == 'url_index'


def test_credf8_same_url_twice_is_still_one_target(tmp_path):
    """(behaviour kept) the same URL twice under one identity is a duplicate, not an ambiguity."""
    documents = [{'url': 'https://a.example/x.pdf', 'source_id': 'S1'}, {'url': 'https://a.example/x.pdf', 'source_id': 'S1'}]
    planned, data = _plan_keys(tmp_path, documents)
    assert len(planned) == 1 and [item['reason_code'] for item in data['exclusions']] == ['duplicate_source']
    assert set(_run_keys(tmp_path, documents)) == set(planned)


def test_credf8_cli_reports_the_typed_code(tmp_path, capsys):
    import json
    from glossapi.cli import main
    manifest = tmp_path/'m.json'
    manifest.write_text(json.dumps({'documents': [{'url': 'https://a.example/x.pdf', 'source_id': 'S1'},
                                                  {'url': 'https://a.example/y.pdf', 'source_id': 'S1'}]}))
    assert main(['plan', '--corpus', str(tmp_path/'corpus'), '--manifest', str(manifest), '--json']) == 2
    envelope = json.loads(capsys.readouterr().out)
    assert envelope['errors'][0]['code'] == 'invalid_manifest'
    assert envelope['errors'][0]['details']['reason'] == 'ambiguous_source_identity'


# ================================================================== small items ===

def test_integ_g2_status_exit_1_names_its_failures(tmp_path, capsys):
    """INTEG-G2: status exited 1 with an empty errors list on 7c9bd37."""
    import json
    from glossapi.cli import main
    from test_ledger_retention import consumer
    root = tmp_path/'corpus'; root.mkdir()
    with LedgerWriter(root/'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='integ-g2')
            document = writer.document('file:///original'); version = writer.version(document)
            path = root/'a.md'; path.write_text('x')
            file = writer.artifact(run, 'extract', 'raw_markdown', path.as_uri(), version_id=version, sha256=sha256_file(path), bytes=1)
        consumer(writer, run, version, file, status='failed')
    assert main(['status', '--corpus', str(root), '--json']) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert [(error['code'], error['stage'], error['document_id']) for error in envelope['errors']] == [('clean_process_failed', 'clean', document)]
    assert main(['status', '--corpus', str(root), '--stage', 'extract', '--json']) == 0  # the filter applies
    assert json.loads(capsys.readouterr().out)['errors'] == []


def test_ls3_g04_loaded_transport_reference_and_ocr_modules_are_inventoried(monkeypatch):
    import sys, types
    from glossapi.ledger import journal
    for name in ('aiohttp', 'playwright', 'PIL', 'yaml', 'pypdf', 'fontTools', 'onnxruntime', 'greekocr'):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.SimpleNamespace(__file__=None, __version__='0-test'))
    names = {entry['name'] for entry in journal.loaded_tools()['packages']}
    assert {'aiohttp', 'playwright', 'PIL', 'yaml', 'pypdf', 'fontTools', 'onnxruntime', 'greekocr'} <= names


def test_ls3_g04_inventory_never_imports(monkeypatch):
    """(behaviour kept) a module that is not loaded is not imported to identify it."""
    import sys
    from glossapi.ledger import journal
    monkeypatch.delitem(sys.modules, 'greekocr', raising=False)
    assert 'greekocr' not in {entry['name'] for entry in journal.loaded_tools()['packages']}
    assert 'greekocr' not in sys.modules


def test_ls4_g02_overwritten_mirror_keeps_one_present_observation(tmp_path, monkeypatch):
    """LS4-G02: a re-published fixed-path mirror (raw and clean markdown) left the older row present on 7c9bd37."""
    from glossapi import Corpus
    from glossapi.ledger.writer import open_readonly
    from test_ledger_fixset_lg import pdf
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))
    root = tmp_path/'out'
    source = root/'downloads'/'doc.pdf'
    for text in ('First version of the document text', 'Second, different version of the text'):
        pdf(source, text)
        corpus = Corpus(tmp_path/'empty', root)
        corpus.extract(file_paths=[source], repair_fonts=False, accel_type='CPU', num_threads=1, skip_existing=False)
        corpus.clean(drop_bad=False, num_threads=1)
    with open_readonly(root/'corpus.sqlite3') as connection:
        rows = connection.execute("SELECT uri,COUNT(DISTINCT sha256) FROM current_file_state WHERE presence='present' AND ownership='managed' "
                                  "AND uri NOT LIKE '%/.glossapi/%' GROUP BY uri").fetchall()
        assert rows and all(count == 1 for _, count in rows), rows
        superseded = connection.execute("SELECT COUNT(*) FROM current_file_state WHERE presence='missing' AND uri LIKE '%/markdown/doc.md'").fetchone()[0]
        assert superseded >= 1


def test_ls8_g03_metadata_predicates_match_through_the_compact_adapter(tmp_path):
    import hashlib
    from glossapi.ledger.compact import connect
    from glossapi.ledger.schema import SCHEMA_SQL
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        corpus_id = writer.connection.execute('SELECT corpus_id FROM ledger_info').fetchone()[0]
    connection = connect(tmp_path/'corpus.sqlite3')
    try:
        assert connection.execute('SELECT COUNT(*) FROM ledger_info WHERE corpus_id=?', (corpus_id,)).fetchone()[0] == 1
        digest = hashlib.sha256(SCHEMA_SQL.encode()).hexdigest()
        assert connection.execute('SELECT version FROM schema_migrations WHERE migration_sha256=?', (digest,)).fetchone()[0] == 1
        # Other tables keep compact binding: a run id still matches its BLOB.
        assert connection.compact
    finally:
        connection.close()


def test_ls9_g04_already_missing_file_in_a_batch_is_not_a_policy_deletion(tmp_path):
    import hashlib
    from glossapi.stages.retention import RetentionController
    from test_ledger_retention import consumer
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='ls9-g04')
        files = []
        for index in range(2):
            path = tmp_path/'products'/f'{index}.md'
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(f'exact {index}')
            with writer.transaction():
                document = writer.document(f'file:///original-{index}'); version = writer.version(document)
                file = writer.artifact(run, 'extract', 'raw_markdown', path.as_uri(), version_id=version,
                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
            task, _ = consumer(writer, run, version, file)
            files.append((file, task, path))
        controller = RetentionController(writer, tmp_path, run)
        with writer.batch():
            for file, task, _ in files:
                controller.declare(file, [task])
        writer.durable_barrier()
        files[0][2].unlink()  # gone before the batch
        controller.delete_many([file for file, _, _ in files])
        tables = ('event_records', 'retention_events') if writer.connection.compact else ('events',)
        reasons = lambda file: [row[1] for row in sorted(tuple(row) for table in tables for row in writer.connection.execute(
            f'SELECT event_id,reason_code FROM {table} WHERE file_id=?', (file,)))]
        presence = lambda file: writer.connection.execute('SELECT presence FROM current_file_state WHERE file_id=?', (file,)).fetchone()[0]
        assert not {'delete_planned', 'delete_recovered', 'deleted'} & set(reasons(files[0][0]))
        assert reasons(files[0][0])[-1] == 'missing' and presence(files[0][0]) == 'missing'
        assert presence(files[1][0]) == 'deleted' and not files[1][2].exists()  # the present file is still deleted


def test_fg_new_1_receipt_without_commit_names_none(tmp_path, monkeypatch):
    """FG-NEW-1 (optional hardening): a shipped receipt without `commit` raised KeyError in every ledger run."""
    import json
    from glossapi.ledger import journal
    (tmp_path/'.code-receipt.json').write_text(json.dumps({'tree_sha256': 'ab'*32}))
    monkeypatch.setattr(journal, '__file__', str(tmp_path/'src'/'glossapi'/'ledger'/'journal.py'))
    journal._code_identity.cache_clear()
    try:
        assert journal._code_identity() == ('unknown_not_recorded', 'ab'*32)
    finally:
        monkeypatch.undo()
        journal._code_identity.cache_clear()


# ================================================================ K4F3-G1 charge ===

def _publication_checks(tmp_path, monkeypatch, stages, n):
    from glossapi import Corpus
    from glossapi.stages import run as RUN
    from test_k4fix2 import _extreme, _unique_bytes
    from test_ledger_fixset_lg import pdf
    checks, settings = _extreme(tmp_path, monkeypatch, lambda: 1 << 30)
    settings = {**settings, 'max_source_bytes': 1 << 22, 'max_attempt_bytes': 1 << 25}
    root = tmp_path/'out'
    measured = []
    real = RUN.publish_barrier
    def barrier(corpus, run, stage, callback, **kwargs):
        before = _unique_bytes(root)
        need = checks[-1]['need']
        try:
            return real(corpus, run, stage, callback, **kwargs)
        finally:
            measured.append(dict(stage=stage, before=before, after=_unique_bytes(root), need=need))
    monkeypatch.setattr(RUN, 'publish_barrier', barrier)
    text = ' '.join(f'Paragraph {i} of a longer document text for native cleaning and sections.' for i in range(int(os.environ.get('K4G_PARAGRAPHS', 3000))))
    sources = [pdf(tmp_path/f'doc{i}.pdf', f'Document {i}. {text}') for i in range(n)]
    assert Corpus(tmp_path/'empty', root).run(sources, stages=stages, settings=settings)['state'] == 'completed'
    return checks, measured


@pytest.mark.parametrize('stages', [('extract', 'clean', 'export'), ('extract', 'clean', 'sections', 'annotate', 'export')], ids=['export', 'both'])
def test_k4f3g1_every_check_covers_what_remains_after_releases(tmp_path, monkeypatch, stages):
    """K4F3-G1: the calibrated charge credits only releases that really happen before a later publication (a reader
    that is a later publication, such as the export reading the cleaned texts, keeps a file), so every check still
    covers the largest footprint of the publications that remain."""
    checks, measured = _publication_checks(tmp_path, monkeypatch, stages, 6)
    for index, entry in enumerate(measured):
        later = max(item['after'] for item in measured[index:])
        assert entry['need'] >= later - entry['before'], (entry['stage'], entry['need'], later - entry['before'])
    pre = next(check['need'] for check in checks if check.get('when') == '(exact, before the publications)')
    assert pre >= max(item['after'] for item in measured) - measured[0]['before']

