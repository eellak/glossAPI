"""K2 native batch, batch-3 gap (MEDIUM): a configured but missing GLOSSAPI_GREEK_WORDLIST.

Coordinator decision: extraction proceeds without the word list (dictionary rates null, as before the ledger) and the
identity records the fixed marker ``missing:<path as given>``; a present file keeps its existing identity bytes
(its sha256). The end-to-end nullable-rates case is ``test_extract_font_repair.py::test_bad_wordlist_is_nullable``.
"""
import hashlib
import json
from pathlib import Path

from glossapi.stages.extract import wordlist_identity


def _decisions(root):
    from glossapi.ledger import open_readonly
    from glossapi.ledger.encoding import unpack_evidence
    with open_readonly(Path(root) / 'corpus.sqlite3') as connection:
        evidence = [unpack_evidence(json.loads(raw)) for raw, in connection.execute("SELECT json FROM snapshots WHERE kind='stage_evidence'")]
    return [item for item in evidence if isinstance(item, dict) and 'dictionary_sha256' in item]


def _extract(tmp_path, monkeypatch, wordlist):
    from glossapi import Corpus
    from test_ledger_stages import simple_pdf
    downloads = tmp_path / 'downloads'
    downloads.mkdir(parents=True)
    simple_pdf(downloads / 'doc.pdf')
    if wordlist is None:
        monkeypatch.delenv('GLOSSAPI_GREEK_WORDLIST', raising=False)
    else:
        monkeypatch.setenv('GLOSSAPI_GREEK_WORDLIST', wordlist)
    corpus = Corpus(tmp_path, tmp_path)
    corpus.extract(phase1_backend='safe', accel_type='CPU')
    return _decisions(tmp_path)


def test_wordlist_identity_values(tmp_path):
    present = tmp_path / 'words.txt'
    present.write_text('λόγος\nσοφία\n', encoding='utf-8')
    assert wordlist_identity(None) is None and wordlist_identity('') is None
    # A present file: the same bytes as before this change (sha256 of the file).
    assert wordlist_identity(str(present)) == hashlib.sha256(present.read_bytes()).hexdigest()
    # A missing file: the fixed marker with the path exactly as configured (no resolution).
    assert wordlist_identity('relative/missing words.txt') == 'missing:relative/missing words.txt'
    assert wordlist_identity(str(tmp_path / 'missing.txt')) == 'missing:' + str(tmp_path / 'missing.txt')
    assert wordlist_identity(str(tmp_path)) == 'missing:' + str(tmp_path)  # a directory is not a word list


def test_present_wordlist_identity_is_unchanged_in_the_extract_decision(tmp_path, monkeypatch):
    words = tmp_path / 'words.txt'
    words.write_text('λόγος\n', encoding='utf-8')
    decisions = _extract(tmp_path / 'corpus', monkeypatch, str(words))
    assert decisions and {item['dictionary_sha256'] for item in decisions} == {hashlib.sha256(words.read_bytes()).hexdigest()}


def test_missing_wordlist_extracts_and_records_the_marker(tmp_path, monkeypatch):
    missing = str(tmp_path / 'missing.txt')
    decisions = _extract(tmp_path / 'corpus', monkeypatch, missing)
    assert decisions and {item['dictionary_sha256'] for item in decisions} == {'missing:' + missing}


def test_unset_wordlist_records_none(tmp_path, monkeypatch):
    decisions = _extract(tmp_path / 'corpus', monkeypatch, None)
    assert decisions and {item['dictionary_sha256'] for item in decisions} == {None}


def test_run_identity_semantic_files(tmp_path, monkeypatch):
    """corpus.run's run identity: a present list keeps its existing entry bytes; a missing one adds the marker."""
    from glossapi import Corpus
    from glossapi.ledger import open_readonly
    from test_ledger_stages import simple_pdf

    def semantic_files(root, wordlist):
        source = tmp_path / 'source.pdf'
        if not source.exists():
            simple_pdf(source)
        monkeypatch.setenv('GLOSSAPI_GREEK_WORDLIST', wordlist)
        corpus = Corpus(root, root)
        result = corpus.run([str(source)], stages=('extract',))
        assert result['state'] == 'completed', result  # the scheduler accepts the frozen marker
        with open_readonly(Path(root) / 'corpus.sqlite3') as connection:
            digest, = connection.execute('SELECT settings_sha256 FROM runs').fetchone()
            settings = json.loads(connection.execute('SELECT json FROM snapshots WHERE sha256=?', (digest,)).fetchone()[0])
        return settings['semantic_files']

    words = tmp_path / 'words.txt'
    words.write_text('λόγος\n', encoding='utf-8')
    present = semantic_files(tmp_path / 'present', str(words))
    assert present[-1] == {'path': str(words.resolve()), 'sha256': hashlib.sha256(words.read_bytes()).hexdigest()}
    missing_path = str(tmp_path / 'nope' / 'missing.txt')
    missing = semantic_files(tmp_path / 'missing', missing_path)
    assert missing[-1] == {'path': missing_path, 'sha256': 'missing:' + missing_path}
    assert missing[:-1] == present[:-1]  # the other identity entries are unaffected


def _frozen_wordlist_run(tmp_path, words):
    """A one-document streaming run frozen while the configured word list ``words`` is missing (marker binding)."""
    from test_ledger_scheduler import graph, scheduler
    from glossapi.ledger import LedgerWriter
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    engine = scheduler(writer, tmp_path)
    binding = {'path': str(words), 'sha256': wordlist_identity(str(words))}
    assert binding['sha256'] == 'missing:' + str(words)
    run = engine.freeze(graph(writer, tmp_path, 1), settings={'semantic_files': [binding]})
    return writer, engine, run


def test_scheduler_refuses_a_missing_wordlist_that_appeared(tmp_path):
    """CP-K2a F5: the actual ``StreamingScheduler.execute`` refuses, before any dispatch, a run frozen with the
    ``missing:`` marker once the file exists. Fails if the scheduler's semantic-file check is removed."""
    import pytest
    words = tmp_path / 'later.txt'
    writer, engine, run = _frozen_wordlist_run(tmp_path, words)
    with writer:
        words.write_text('λόγος\n', encoding='utf-8')
        with pytest.raises(ValueError, match='Frozen model/dictionary input changed; create an amended run'):
            engine.execute(run)
        assert writer.connection.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 0  # nothing dispatched


def test_scheduler_runs_a_wordlist_that_is_still_missing(tmp_path):
    """The other half: the marker still matches, so the frozen run executes. Fails if the scheduler's ``missing:``
    branch is removed (the ordinary present-file check would refuse a resumable run)."""
    words = tmp_path / 'never.txt'
    writer, engine, run = _frozen_wordlist_run(tmp_path, words)
    with writer:
        result = engine.execute(run)
        assert result['state'] == 'completed' and result['counts']['ok'] == 2
