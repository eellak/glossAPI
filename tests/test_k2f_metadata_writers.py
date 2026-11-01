"""CP-K2a F1: every writer of a parquet that ``find_metadata_parquet`` can select publishes under the shared metadata
lock, re-reading the current file and merging only its own rows/fields (network work stays outside the lock).

Each interleaving test commits a second writer's rows through ``update_metadata_parquet`` at the point where the
writer under test has already read the file but not yet written it; on 9d5d372 that commit was lost. The no-race
tests pin the published bytes to the serializer each writer used before, so nothing changes without concurrency.
"""
import threading

import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from glossapi.parquet_schema import update_metadata_parquet

CONCURRENT = {'url': ['https://example.invalid/concurrent.pdf'], 'filename': ['concurrent.pdf'],
              'download_success': [True], 'extract_status': ['success']}


def _commit_concurrent_row(target):
    update_metadata_parquet(target, lambda current: pd.concat([current, pd.DataFrame(CONCURRENT)], ignore_index=True))


# ------------------------------------------------------------------ Corpus.download publication (phase_download)

def _download_case(tmp_path, monkeypatch, *, concurrent):
    """The reviewer's probe: old.pdf recorded, new.pdf acquired; optionally a commit during acquisition."""
    from glossapi import Corpus
    from glossapi.gloss_downloader import GlossDownloader
    source = tmp_path / 'urls.parquet'
    pd.DataFrame({'url': ['https://example.invalid/new.pdf']}).to_parquet(source, index=False)
    target = tmp_path / 'download_results/download_results_urls.parquet'
    target.parent.mkdir()
    pd.DataFrame({'url': ['https://example.invalid/old.pdf'], 'filename': ['old.pdf'],
                  'download_success': [False]}).to_parquet(target, index=False)
    acquired = pd.DataFrame({'url': ['https://example.invalid/new.pdf'], 'filename': ['new.pdf'], 'download_success': [True]})

    def transport(self, **kwargs):
        if concurrent:
            _commit_concurrent_row(target)
            assert 'concurrent.pdf' in pd.read_parquet(target).filename.tolist()
        return acquired.copy()

    monkeypatch.setattr(GlossDownloader, 'download_files', transport)
    returned = Corpus(tmp_path, tmp_path).download(input_parquet=source)
    return target, returned, acquired


def test_download_publication_keeps_a_row_committed_during_acquisition(tmp_path, monkeypatch):
    target, returned, _ = _download_case(tmp_path, monkeypatch, concurrent=True)
    final = pd.read_parquet(target)
    assert set(final.filename) == {'old.pdf', 'new.pdf', 'concurrent.pdf'}  # 9d5d372: {'old.pdf', 'new.pdf'}
    assert final.loc[final.filename == 'concurrent.pdf', 'extract_status'].tolist() == ['success']
    assert set(returned.filename) == {'old.pdf', 'new.pdf', 'concurrent.pdf'}


def test_download_publication_without_concurrency_writes_the_previous_bytes(tmp_path, monkeypatch):
    from glossapi.gloss_downloader import persisted_download_frame
    existing = pd.DataFrame({'url': ['https://example.invalid/old.pdf'], 'filename': ['old.pdf'], 'download_success': [False]})
    target, returned, acquired = _download_case(tmp_path, monkeypatch, concurrent=False)
    expected = tmp_path / 'expected.parquet'
    persisted_download_frame(pd.concat([existing, acquired], ignore_index=True), 'url').to_parquet(expected, index=False)
    assert target.read_bytes() == expected.read_bytes()
    assert not list(target.parent.glob('*.tmp')) and not list(target.parent.glob('*.lock'))


def test_download_publication_replaces_rows_of_the_same_resource_only(tmp_path, monkeypatch):
    """A re-acquired resource replaces its earlier row; a concurrent row and column survive."""
    from glossapi import Corpus
    from glossapi.gloss_downloader import GlossDownloader
    source = tmp_path / 'urls.parquet'
    pd.DataFrame({'url': ['https://example.invalid/a.pdf', 'https://example.invalid/b.pdf']}).to_parquet(source, index=False)
    target = tmp_path / 'download_results/download_results_urls.parquet'
    target.parent.mkdir()
    pd.DataFrame({'url': ['https://example.invalid/a.pdf', 'https://example.invalid/b.pdf'], 'filename': ['a.pdf', 'b.pdf'],
                  'download_success': [True, False]}).to_parquet(target, index=False)

    def transport(self, **kwargs):
        _commit_concurrent_row(target)
        return pd.DataFrame({'url': ['https://example.invalid/b.pdf'], 'filename': ['b_retry.pdf'], 'download_success': [True]})

    monkeypatch.setattr(GlossDownloader, 'download_files', transport)
    Corpus(tmp_path, tmp_path).download(input_parquet=source)
    final = pd.read_parquet(target)
    assert final.filename.tolist() == ['a.pdf', 'concurrent.pdf', 'b_retry.pdf']
    assert 'extract_status' in final.columns


def test_identity_rewrite_keeps_a_row_committed_after_the_read(tmp_path, monkeypatch):
    """All URLs done + an older results file without ``resource_identity``: the rewrite that adds it (line 269 on
    9d5d372, then a redaction rewrite) is applied to the file as it is now."""
    from glossapi import Corpus
    import glossapi.gloss_downloader as downloader
    source = tmp_path / 'urls.parquet'
    pd.DataFrame({'url': ['https://example.invalid/old.pdf']}).to_parquet(source, index=False)
    target = tmp_path / 'download_results/download_results_urls.parquet'
    target.parent.mkdir()
    pd.DataFrame({'url': ['https://example.invalid/old.pdf'], 'filename': ['old.pdf'],
                  'download_success': [True]}).to_parquet(target, index=False)
    real = downloader.persisted_download_frame
    calls = []

    def persisted(frame, url_column='url'):
        if not calls:  # the first call computes the rewrite from the earlier read; then another writer commits
            _commit_concurrent_row(target)
        calls.append(len(frame))
        return real(frame, url_column)

    monkeypatch.setattr(downloader, 'persisted_download_frame', persisted)
    Corpus(tmp_path, tmp_path).download(input_parquet=source)
    final = pd.read_parquet(target)
    assert set(final.filename) == {'old.pdf', 'concurrent.pdf'}  # 9d5d372: {'old.pdf'}
    assert 'resource_identity' in final.columns and final.resource_identity.notna().all()


def test_identity_rewrite_without_concurrency_writes_the_previous_bytes(tmp_path, monkeypatch):
    from glossapi import Corpus
    from glossapi.gloss_downloader import persisted_download_frame
    source = tmp_path / 'urls.parquet'
    pd.DataFrame({'url': ['https://example.invalid/old.pdf']}).to_parquet(source, index=False)
    target = tmp_path / 'download_results/download_results_urls.parquet'
    target.parent.mkdir()
    existing = pd.DataFrame({'url': ['https://example.invalid/old.pdf'], 'filename': ['old.pdf'], 'download_success': [True]})
    existing.to_parquet(target, index=False)
    Corpus(tmp_path, tmp_path).download(input_parquet=source)
    expected = tmp_path / 'expected.parquet'
    persisted_download_frame(existing, 'url').to_parquet(expected, index=False)
    assert target.read_bytes() == expected.read_bytes()


# ------------------------------------------------------------------ create-if-missing fallback seed (parquet_schema)

def _markdown(tmp_path):
    markdown = tmp_path / 'markdown'
    markdown.mkdir()
    (markdown / 'a.md').write_text('α')
    return markdown


def test_basic_metadata_seed_never_replaces_a_file_created_meanwhile(tmp_path, monkeypatch):
    from glossapi.parquet_schema import ParquetSchema
    schema = ParquetSchema()
    target = tmp_path / 'download_results/download_results.parquet'

    def ensure(self, base_dir):  # the check finds nothing; another writer creates the file right after it
        _commit_concurrent_row(target)
        return None

    monkeypatch.setattr(ParquetSchema, 'ensure_metadata_parquet', ensure)
    assert schema.create_basic_metadata_parquet(_markdown(tmp_path), tmp_path) == target
    assert pd.read_parquet(target).filename.tolist() == ['concurrent.pdf']  # 9d5d372: ['a.md'] (overwritten)


def test_basic_metadata_seed_without_concurrency_writes_the_previous_bytes(tmp_path, monkeypatch):
    from glossapi.parquet_schema import ParquetSchema
    schema = ParquetSchema()
    monkeypatch.setattr(ParquetSchema, 'ensure_metadata_parquet', lambda self, base_dir: None)
    target = schema.create_basic_metadata_parquet(_markdown(tmp_path), tmp_path)
    expected = tmp_path / 'expected.parquet'
    pq.write_table(pa.Table.from_pandas(pd.DataFrame([{'filename': 'a.md', schema.url_column: ''}])), expected)
    assert target.read_bytes() == expected.read_bytes()


# ------------------------------------------------------------------ OpenArchives post-download publication

def _freeze(tmp_path, monkeypatch, download):
    import glossapi.scripts.openarchives_download_freeze as freeze
    source = tmp_path / 'input.parquet'
    pd.DataFrame([{'filename': 'ABC_001.pdf', 'pdf_url': 'https://example.com/a.pdf', 'needs_ocr': True},
                  {'filename': 'ABC_002.pdf', 'pdf_url': 'https://example.com/b.pdf', 'needs_ocr': True}]).to_parquet(source, index=False)

    class DummyCorpus:
        def __init__(self, *args, **kwargs):
            pass

        def download(self, **kwargs):
            return download()

    monkeypatch.setattr(freeze, 'Corpus', DummyCorpus)
    work = tmp_path / 'work'
    assert freeze.main(['--input-parquet', str(source), '--work-root', str(work)]) == 0
    return work / 'download_results/download_results.parquet'


def _downloaded():
    return pd.DataFrame([{'url': 'https://example.com/a.pdf', 'filename': 'ABC_001.pdf', 'download_success': True,
                          'download_error': '', 'file_ext': 'pdf'},
                         {'url': 'https://example.com/b.pdf', 'filename': 'ABC_002.pdf', 'download_success': False,
                          'download_error': 'HTTP 404', 'file_ext': 'pdf'}])


def test_openarchives_publication_keeps_rows_and_columns_committed_during_download(tmp_path, monkeypatch):
    work_holder = {}

    def download():
        canonical = work_holder['canonical']
        update_metadata_parquet(canonical, lambda current: pd.concat([
            current.assign(triage=['kept'] * len(current)),
            pd.DataFrame({'filename': ['OTHER_009.pdf'], 'url': ['https://example.com/z.pdf']})], ignore_index=True))
        return _downloaded()

    work_holder['canonical'] = tmp_path / 'work/download_results/download_results.parquet'
    canonical = _freeze(tmp_path, monkeypatch, download)
    final = pd.read_parquet(canonical)
    assert final.filename.tolist() == ['ABC_001.pdf', 'ABC_002.pdf', 'OTHER_009.pdf']  # 9d5d372: OTHER_009 lost
    assert final.triage.tolist()[:2] == ['kept', 'kept']  # 9d5d372: column lost
    assert final.download_success.tolist()[:2] == [True, False] and final.download_error.tolist()[1] == 'HTTP 404'


def test_openarchives_publication_without_concurrency_writes_the_previous_bytes(tmp_path, monkeypatch):
    from glossapi.parquet_schema import ParquetSchema
    from glossapi.scripts.openarchives_manifest import _normalize_download_results, _prepare_download_input
    canonical = _freeze(tmp_path, monkeypatch, _downloaded)
    manifest = _prepare_download_input(pd.read_parquet(tmp_path / 'input.parquet'))
    expected = tmp_path / 'expected.parquet'
    schema = ParquetSchema({'url_column': 'url'})
    schema.write_metadata_parquet(schema.normalize_metadata_frame(
        _normalize_download_results(shard_df=manifest, download_results_df=_downloaded(), url_column='url')), expected)
    assert canonical.read_bytes() == expected.read_bytes()


# ------------------------------------------------------------------ in-place filename_base precompute

def _alias(target, kind):
    """``target`` under another spelling of the same file (CP-K2c F1)."""
    if kind == 'literal':
        return target
    if kind == 'dotdot':
        return target.parent / '..' / target.parent.name / target.name
    if kind == 'dir-symlink':
        link = target.parent.parent / 'linked_results'
        link.symlink_to(target.parent, target_is_directory=True)
        return link / target.name
    if kind == 'file-symlink':
        link = target.parent / 'manifest-link.parquet'
        link.symlink_to(target)
        return link
    raise ValueError(kind)


@pytest.mark.parametrize('spelling', ['literal', 'dotdot', 'dir-symlink'])
@pytest.mark.parametrize('side', ['output', 'input'])
def test_in_place_filename_base_precompute_serializes_with_a_concurrent_writer(tmp_path, monkeypatch, spelling, side):
    """In place under any spelling of the same file (CP-K2a F1; CP-K2c F1: on 7ad4962 the ``..`` and symlinked-directory
    spellings took the unlocked direct write and lost the concurrent row and column)."""
    from glossapi.gloss_downloader import GlossDownloader
    from glossapi.parquet_schema import ParquetSchema
    target = tmp_path / 'download_results/manifest.parquet'
    target.parent.mkdir()
    pd.DataFrame({'url': ['https://example.invalid/old.pdf'], 'filename': ['old.pdf']}).to_parquet(target, index=False)
    alias = _alias(target, spelling)
    assert alias.samefile(target) and (spelling == 'literal' or alias.absolute() != target.absolute())
    assert ParquetSchema().find_metadata_parquet(tmp_path / 'download_results') == target
    source, output = (target, alias) if side == 'output' else (alias, target)
    downloader = GlossDownloader(url_column='url', output_dir=str(tmp_path / 'out'))
    real = GlossDownloader.ensure_filename_base
    writers, errors = [], []

    def commit():
        try:
            update_metadata_parquet(target, lambda current: pd.concat([current, pd.DataFrame({
                'url': ['https://example.invalid/concurrent.pdf'], 'filename': ['concurrent.pdf'], 'triage': ['committed']})],
                ignore_index=True))
        except BaseException as error:
            errors.append(error)

    def ensure(self, frame):
        # The transform has read the file; a second writer tries to commit now.
        thread = threading.Thread(target=commit)
        writers.append(thread); thread.start(); thread.join(0.3)
        return real(self, frame)

    monkeypatch.setattr(GlossDownloader, 'ensure_filename_base', ensure)
    try:
        downloader.precompute_filename_base_to_parquet(source, None if spelling == 'literal' and side == 'output' else output)
    finally:
        for thread in writers:
            thread.join(5)
        downloader.close_logs()
    assert not errors and not any(thread.is_alive() for thread in writers)
    final = pd.read_parquet(target)
    assert set(final.filename) == {'old.pdf', 'concurrent.pdf'}  # 9d5d372 (literal) / 7ad4962 (aliases): ['old.pdf']
    assert final.loc[final.filename == 'concurrent.pdf', 'triage'].tolist() == ['committed']
    assert final.filename_base.notna().any()
    assert not target.is_symlink() and not list(target.parent.glob('*.tmp')) and not list(target.parent.glob('*.lock'))


def test_distinct_precompute_output_is_an_explicit_new_artifact(tmp_path):
    """A genuinely distinct output: the input is never rewritten; the output is replaced whole (atomic, locked)."""
    from glossapi.gloss_downloader import GlossDownloader
    source = tmp_path / 'download_results/manifest.parquet'
    source.parent.mkdir()
    pd.DataFrame({'url': ['https://example.invalid/a.pdf']}).to_parquet(source, index=False)
    before = source.read_bytes()
    output = tmp_path / 'derived/manifest.parquet'
    output.parent.mkdir()
    pd.DataFrame({'stale': [1]}).to_parquet(output, index=False)
    downloader = GlossDownloader(url_column='url', output_dir=str(tmp_path / 'out'))
    try:
        frame = downloader.precompute_filename_base_to_parquet(source, output)
    finally:
        downloader.close_logs()
    assert source.read_bytes() == before
    written = pd.read_parquet(output)
    assert 'stale' not in written.columns and written.filename_base.notna().all() and len(frame) == len(written)
    expected = tmp_path / 'expected.parquet'
    frame.to_parquet(expected, index=False)
    assert output.read_bytes() == expected.read_bytes()  # the previous direct ``to_parquet`` bytes


@pytest.mark.parametrize('spelling', ['dotdot', 'dir-symlink', 'file-symlink'])
def test_every_spelling_takes_the_same_lock_and_names_the_same_file(tmp_path, spelling):
    from filelock import FileLock, Timeout
    from glossapi.parquet_schema import _parquet_lock, metadata_path_identity, same_metadata_file
    target = tmp_path / 'download_results/download_results.parquet'
    target.parent.mkdir()
    pd.DataFrame({'filename': ['a.pdf']}).to_parquet(target, index=False)
    alias = _alias(target, spelling)
    assert same_metadata_file(alias, target) and metadata_path_identity(alias) == metadata_path_identity(target)
    assert not same_metadata_file(target, tmp_path / 'download_results/other.parquet')
    assert same_metadata_file(target.parent / '..' / 'download_results' / 'missing.parquet', target.parent / 'missing.parquet')
    with _parquet_lock(alias):
        with pytest.raises(Timeout):  # the literal spelling's lock is the one held
            FileLock(str(metadata_path_identity(target)) + '.lock').acquire(timeout=0.2)
    # a writer through the alias replaces the real file, never the link
    update_metadata_parquet(alias, lambda current: pd.concat([current, pd.DataFrame({'filename': ['b.pdf']})], ignore_index=True))
    assert pd.read_parquet(target).filename.tolist() == ['a.pdf', 'b.pdf'] and not target.is_symlink()
    if spelling == 'file-symlink':
        assert alias.is_symlink()


# ------------------------------------------------------------------ merge rule: never drop an unmatched row/column

def test_openarchives_merge_keeps_foreign_rows_when_current_lacks_filename(tmp_path):
    """CP-K2c F2 (the reviewer's probe): the keyless downloader shape is selectable metadata; on 7ad4962 the merge
    treated it as empty and deleted the foreign row and its committed column."""
    from glossapi.parquet_schema import ParquetSchema
    from glossapi.scripts.openarchives_manifest import _write_canonical_metadata
    target = tmp_path / 'download_results' / 'download_results.parquet'
    update_metadata_parquet(target, lambda current: pd.DataFrame({
        'url': ['https://example.invalid/foreign.pdf'], 'download_success': [True],
        'download_error': [''], 'filename_base': ['FOREIGN_001'], 'triage': ['committed']}))
    assert ParquetSchema().find_metadata_parquet(tmp_path) == target
    rows = pd.DataFrame({'url': ['https://example.invalid/new.pdf'], 'filename': ['NEW_001.pdf'], 'download_success': [True]})
    _write_canonical_metadata(tmp_path, rows, replaces_filenames=['NEW_001.pdf'])
    final = pd.read_parquet(target)
    assert final.url.tolist() == ['https://example.invalid/new.pdf', 'https://example.invalid/foreign.pdf']
    assert final.loc[final.url.str.contains('foreign'), 'triage'].tolist() == ['committed']
    assert {'download_error', 'filename_base', 'triage'} <= set(final.columns)


def test_merge_owned_rows_never_drops_an_unmatched_current_row_or_column():
    from glossapi.parquet_schema import merge_owned_rows
    rows = pd.DataFrame({'filename': ['a.pdf', 'b.pdf'], 'download_success': [True, False]})
    keyless = pd.DataFrame({'download_success': [True], 'download_error': [''], 'filename_base': ['X_001'], 'triage': ['t']})
    merged = merge_owned_rows(keyless, rows, 'filename', owned={'a.pdf'})
    assert len(merged) == 3 and merged.iloc[:2].filename.tolist() == ['a.pdf', 'b.pdf']
    assert merged.iloc[2][['download_error', 'filename_base', 'triage']].tolist() == ['', 'X_001', 't']
    # rows without the key: nothing matchable either; every current row is kept
    current = pd.DataFrame({'filename': ['c.pdf'], 'extract_status': ['ok']})
    assert merge_owned_rows(current, rows.drop(columns=['filename']), 'filename').filename.tolist()[-1] == 'c.pdf'
    # keyed: null keys and unowned keys are kept, owned keys absent from rows are dropped, foreign columns joined
    current = pd.DataFrame({'filename': ['a.pdf', 'old.pdf', None, 'z.pdf'], 'triage': ['ta', 'to', 'tn', 'tz']})
    merged = merge_owned_rows(current, rows, 'filename', owned={'old.pdf'})
    assert merged.filename.tolist()[:2] == ['a.pdf', 'b.pdf'] and merged.triage.tolist()[0] == 'ta'
    assert set(merged.triage.dropna()) == {'ta', 'tn', 'tz'} and 'old.pdf' not in merged.filename.tolist()
    # nothing concurrent: the rows themselves
    assert merge_owned_rows(pd.DataFrame(), rows, 'filename') is rows
