"""CP-K2a F1: every writer of a parquet that ``find_metadata_parquet`` can select publishes under the shared metadata
lock, re-reading the current file and merging only its own rows/fields (network work stays outside the lock).

Each interleaving test commits a second writer's rows through ``update_metadata_parquet`` at the point where the
writer under test has already read the file but not yet written it; on 9d5d372 that commit was lost. The no-race
tests pin the published bytes to the serializer each writer used before, so nothing changes without concurrency.
"""
import threading

import pandas as pd
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


def test_redaction_rewrite_keeps_a_row_committed_after_the_read(tmp_path, monkeypatch):
    """All URLs done + an older results file without ``resource_identity``: the redacted rewrite (line 269 on
    9d5d372) is applied to the file as it is now."""
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


def test_redaction_rewrite_without_concurrency_writes_the_previous_bytes(tmp_path, monkeypatch):
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

def test_in_place_filename_base_precompute_serializes_with_a_concurrent_writer(tmp_path, monkeypatch):
    from glossapi.gloss_downloader import GlossDownloader
    target = tmp_path / 'download_results/manifest.parquet'
    target.parent.mkdir()
    pd.DataFrame({'url': ['https://example.invalid/a.pdf']}).to_parquet(target, index=False)
    downloader = GlossDownloader(url_column='url', output_dir=str(tmp_path / 'out'))
    real = GlossDownloader.ensure_filename_base
    started, writer = threading.Event(), []

    def ensure(self, frame):
        # The transform has read the file; a second writer tries to commit now.
        thread = threading.Thread(target=lambda: (started.set(), _commit_concurrent_row(target)))
        thread.start(); writer.append(thread); started.wait(2)
        thread.join(0.3)
        return real(self, frame)

    monkeypatch.setattr(GlossDownloader, 'ensure_filename_base', ensure)
    downloader.precompute_filename_base_to_parquet(target)
    writer[0].join(5)
    final = pd.read_parquet(target)
    assert 'concurrent.pdf' in final.filename.tolist()  # 9d5d372: overwritten by the transform of the older read
    assert final.filename_base.notna().any()
