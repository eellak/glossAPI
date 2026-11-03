"""K2F round 3: the one metadata-target policy (CP-K2d F1–F5).

Policy: every metadata parquet writer goes through ``parquet_schema.metadata_target`` — the physical file
(``resolve_metadata_target`` = ``os.path.realpath(os.path.expanduser(path))``, traversal order, no lexical folding) is
locked, read and atomically replaced; the caller's spelling is only used for messages, observations and return values;
a hard-linked target is refused (``MetadataAliasError``) before anything is read or written; same-file decisions
compare resolved targets (and samefile of existing ones). Merges are lossless or refuse (``MetadataSchemaError``).

The CP-K2d probes are reproduced here (hard-link probes adapted: the coordinator's policy refuses hard links instead of
sharing a lock between their names), plus the combined symlink-then-``..`` case with a sentinel, hard-link refusal in
both directions, the ensure writer through a file symlink, exact large integers after parquet readback, the empty
current schema union, and a static test that fails when a writer bypasses the choke point.
"""
import ast
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock, Timeout

import glossapi.parquet_schema as _schema
from glossapi.parquet_schema import merge_owned_rows, same_metadata_file, update_metadata_parquet

# Resolved per name so that this file also runs (and fails test by test) against 01f53d4, which lacks the policy API.
class _Missing(Exception):
    pass
MetadataAliasError = getattr(_schema, 'MetadataAliasError', _Missing)
MetadataSchemaError = getattr(_schema, 'MetadataSchemaError', _Missing)
resolve_metadata_target = getattr(_schema, 'resolve_metadata_target', getattr(_schema, 'metadata_path_identity', None))
metadata_target = getattr(_schema, 'metadata_target', getattr(_schema, '_parquet_lock', None))

BIG = 9007199254740993  # 2**53 + 1: not representable as float64


def aliases(tmp_path, kind):
    """The CP-K2d alias fixture: ``download_results/Manifest.parquet`` and another spelling of it."""
    target = tmp_path / 'download_results' / 'Manifest.parquet'
    target.parent.mkdir()
    pd.DataFrame({'url': ['https://example.invalid/old.pdf'], 'filename': ['old.pdf']}).to_parquet(target, index=False)
    if kind == 'relative':
        alias = Path(os.path.relpath(target, Path.cwd()))
    elif kind == 'dotdot':
        alias = target.parent / '..' / target.parent.name / target.name
    elif kind == 'dir-symlink':
        link = tmp_path / 'linked'
        link.symlink_to(target.parent, target_is_directory=True)
        alias = link / target.name
    elif kind == 'file-symlink':
        alias = tmp_path / 'file-link.parquet'
        alias.symlink_to(target)
    elif kind == 'case-file':
        alias = target.with_name(target.name.swapcase())
    elif kind == 'case-dir':
        alias = target.parent.with_name(target.parent.name.swapcase()) / target.name
    elif kind == 'hardlink':
        alias = tmp_path / 'hardlink.parquet'
        os.link(target, alias)
    elif kind == 'symlink-dotdot':
        inner = target.parent / 'inner'
        inner.mkdir()
        link = tmp_path / 'linked-inner'
        link.symlink_to(inner, target_is_directory=True)
        alias = link / '..' / target.name
    else:
        raise ValueError(kind)
    if kind.startswith('case-') and not alias.exists():
        pytest.skip('scratch filesystem is case-sensitive')
    assert target.samefile(alias)
    return target, alias


SHARED = ['relative', 'dotdot', 'dir-symlink', 'file-symlink', 'case-file', 'case-dir', 'symlink-dotdot']


# ------------------------------------------------------------------ one lock, one physical file (K2D-F1, F3)

@pytest.mark.parametrize('kind', SHARED)
def test_aliases_really_share_one_lock(tmp_path, kind):
    target, alias = aliases(tmp_path, kind)
    assert same_metadata_file(target, alias)
    with metadata_target(alias) as physical:
        assert physical.samefile(target)
        with pytest.raises(Timeout):
            with FileLock(str(resolve_metadata_target(target)) + '.lock', timeout=0.05):
                pass


def test_hard_links_are_refused_before_any_lock_read_or_write(tmp_path):
    """K2D-F2 policy: no shared lock for hard links; the choke point refuses them (both names)."""
    target, alias = aliases(tmp_path, 'hardlink')
    before = target.read_bytes()
    for name in (target, alias):
        with pytest.raises(MetadataAliasError, match='hard-linked'):
            with metadata_target(name):
                pass
        with pytest.raises(MetadataAliasError):
            update_metadata_parquet(name, lambda current: current.assign(triage='changed'))
    assert target.read_bytes() == before and alias.read_bytes() == before and target.samefile(alias)
    assert not list(target.parent.glob('*.lock')) and not list(tmp_path.glob('*.lock'))


@pytest.mark.parametrize('kind', SHARED)
@pytest.mark.parametrize('side', ['input', 'output'])
def test_precompute_alias_updates_requested_output(tmp_path, kind, side):
    from glossapi.gloss_downloader import GlossDownloader
    target, alias = aliases(tmp_path, kind)
    source, output = (alias, target) if side == 'input' else (target, alias)
    downloader = GlossDownloader(url_column='url', output_dir=str(tmp_path / 'out'))
    try:
        returned = downloader.precompute_filename_base_to_parquet(source, output)
    finally:
        downloader.close_logs()
    final = pd.read_parquet(output)
    pd.testing.assert_frame_equal(final, returned)
    assert final.filename_base.notna().all()
    if kind == 'file-symlink':
        assert alias.is_symlink()


@pytest.mark.parametrize('side', ['input', 'output'])
def test_precompute_refuses_hard_linked_input_and_output(tmp_path, side):
    """Hard-link refusal in both directions; neither name changes."""
    from glossapi.gloss_downloader import GlossDownloader
    target, alias = aliases(tmp_path, 'hardlink')
    before = target.read_bytes()
    source, output = (alias, target) if side == 'input' else (target, alias)
    downloader = GlossDownloader(url_column='url', output_dir=str(tmp_path / 'out'))
    try:
        with pytest.raises(MetadataAliasError, match='hard-linked'):
            downloader.precompute_filename_base_to_parquet(source, output)
    finally:
        downloader.close_logs()
    assert target.read_bytes() == before and alias.read_bytes() == before and target.samefile(alias)


def test_precompute_into_a_hard_linked_distinct_output_is_refused(tmp_path):
    """A distinct output whose resolved target is hard-linked elsewhere is a write target too: refused."""
    from glossapi.gloss_downloader import GlossDownloader
    source, _ = aliases(tmp_path, 'relative')
    output = tmp_path / 'derived.parquet'
    pd.DataFrame({'x': [1]}).to_parquet(output, index=False)
    os.link(output, tmp_path / 'derived-other-name.parquet')
    before = output.read_bytes()
    downloader = GlossDownloader(url_column='url', output_dir=str(tmp_path / 'out'))
    try:
        with pytest.raises(MetadataAliasError):
            downloader.precompute_filename_base_to_parquet(source, output)
    finally:
        downloader.close_logs()
    assert output.read_bytes() == before


def test_symlink_then_dotdot_never_rewrites_an_unrelated_file(tmp_path):
    """K2D-F1: ``linked-inner/../Manifest.parquet`` names ``download_results/Manifest.parquet``; the unrelated sibling
    ``<tmp>/Manifest.parquet`` (what lexical folding chose on 01f53d4) stays byte-identical."""
    target, alias = aliases(tmp_path, 'symlink-dotdot')
    sentinel = tmp_path / target.name
    pd.DataFrame({'filename': ['unrelated.pdf'], 'triage': ['untouched']}).to_parquet(sentinel, index=False)
    before = sentinel.read_bytes()
    assert same_metadata_file(target, alias) and not same_metadata_file(sentinel, alias)
    assert resolve_metadata_target(alias) == resolve_metadata_target(target)
    update_metadata_parquet(alias, lambda current: current.assign(triage='changed'))
    assert sentinel.read_bytes() == before
    assert pd.read_parquet(target).triage.tolist() == ['changed']
    with metadata_target(alias):
        with pytest.raises(Timeout):
            with FileLock(str(resolve_metadata_target(target)) + '.lock', timeout=0.05):
                pass
        with FileLock(str(resolve_metadata_target(sentinel)) + '.lock', timeout=0.05):
            pass  # the sentinel's lock is a different lock


def test_precompute_through_symlink_then_dotdot_input_reads_the_real_file(tmp_path):
    """01f53d4 with no sibling: the helper read a nonexistent folded path and raised a URL-column error."""
    from glossapi.gloss_downloader import GlossDownloader
    target, alias = aliases(tmp_path, 'symlink-dotdot')
    downloader = GlossDownloader(url_column='url', output_dir=str(tmp_path / 'out'))
    try:
        downloader.precompute_filename_base_to_parquet(alias)
    finally:
        downloader.close_logs()
    assert pd.read_parquet(target).filename_base.notna().all()
    assert not (tmp_path / target.name).exists()


def test_callback_failure_is_atomic_and_releases_alias_lock(tmp_path):
    target, alias = aliases(tmp_path, 'file-symlink')
    before = target.read_bytes()
    def fail(frame):
        raise RuntimeError('independent callback failure')
    with pytest.raises(RuntimeError, match='independent callback failure'):
        update_metadata_parquet(alias, fail)
    assert target.read_bytes() == before and alias.is_symlink()
    with FileLock(str(resolve_metadata_target(target)) + '.lock', timeout=0):
        pass


def test_missing_precompute_source_does_not_create_output(tmp_path):
    from glossapi.parquet_schema import rewrite_metadata_parquet
    missing, output = tmp_path / 'missing.parquet', tmp_path / 'output.parquet'
    with pytest.raises(FileNotFoundError):
        rewrite_metadata_parquet(missing, output, lambda frame: frame)
    assert not output.exists()


# ------------------------------------------------------------------ the ensure writer (K2D-F3)

def _ensure_case(tmp_path, link):
    from glossapi.parquet_schema import ParquetSchema
    physical = tmp_path / 'physical' / 'live.parquet'
    physical.parent.mkdir()
    pd.DataFrame({'filename': ['old.pdf'], 'url': ['https://example.invalid/old.pdf']}).to_parquet(physical, index=False)
    alias = tmp_path / 'download_results' / 'alias.parquet'
    alias.parent.mkdir()
    link(physical, alias)
    markdown = tmp_path / 'markdown'
    markdown.mkdir()
    (markdown / 'new.md').write_text('new document')
    return ParquetSchema(), physical, alias


def test_ensure_writer_preserves_file_symlink_identity(tmp_path):
    schema, physical, alias = _ensure_case(tmp_path, lambda physical, alias: alias.symlink_to(physical))
    ensured = schema.ensure_metadata_parquet(tmp_path)
    assert ensured == alias  # the caller's spelling is returned
    assert alias.is_symlink() and same_metadata_file(physical, alias)  # 01f53d4: the link was replaced
    assert set(pd.read_parquet(physical).filename) == {'old.pdf', 'new.pdf'}


def test_ensure_writer_refuses_a_hard_linked_candidate(tmp_path):
    schema, physical, alias = _ensure_case(tmp_path, lambda physical, alias: os.link(physical, alias))
    before = physical.read_bytes()
    with pytest.raises(MetadataAliasError):
        schema.ensure_metadata_parquet(tmp_path)
    assert physical.read_bytes() == before and alias.samefile(physical)


# ------------------------------------------------------------------ lossless merges (K2D-F4, F5)

@pytest.mark.parametrize('shape', ['keyless-current', 'keyless-rows', 'partial-keys'])
def test_merge_preserves_exact_integer_values_and_dtype(tmp_path, shape):
    from glossapi.parquet_schema import pandas_parquet
    from glossapi.scripts.openarchives_manifest import _write_canonical_metadata
    current = pd.DataFrame({'url': ['https://example.invalid/foreign.pdf'], 'download_success': [True],
                            'download_error': [''], 'filename_base': ['FOREIGN'],
                            'foreign_id': pd.Series([BIG], dtype='int64')})
    rows = pd.DataFrame({'url': ['https://example.invalid/new.pdf'], 'filename': ['NEW.pdf'], 'download_success': [True]})
    if shape == 'keyless-rows':
        current['filename'] = 'FOREIGN.pdf'
        rows = rows.drop(columns='filename')
    elif shape == 'partial-keys':
        current['filename'] = pd.Series([None], dtype='str')
    merged = merge_owned_rows(current, rows, 'filename', owned={'NEW.pdf'})
    kept = merged.loc[merged.url == 'https://example.invalid/foreign.pdf', 'foreign_id']
    assert int(kept.iloc[0]) == BIG and pd.api.types.is_integer_dtype(kept.dtype)  # 01f53d4: float64 ...992.0
    # the public publisher, then parquet readback: exact
    target = tmp_path / 'download_results' / 'download_results.parquet'
    update_metadata_parquet(target, lambda _: current, serialize=pandas_parquet)
    if shape == 'keyless-current':
        _write_canonical_metadata(tmp_path, rows, replaces_filenames=['NEW.pdf'])
    else:
        update_metadata_parquet(target, lambda current_: merge_owned_rows(current_, rows, 'filename', owned={'NEW.pdf'}))
    stored = pd.read_parquet(target)
    value = stored.loc[stored.url == 'https://example.invalid/foreign.pdf', 'foreign_id'].iloc[0]
    assert int(value) == BIG and pd.api.types.is_integer_dtype(stored.foreign_id.dtype)
    assert stored.loc[stored.url == 'https://example.invalid/new.pdf', 'foreign_id'].isna().all()


def test_join_keeps_exact_integers_and_booleans_for_unmatched_rows():
    current = pd.DataFrame({'filename': ['a.pdf'], 'foreign_id': pd.Series([BIG], dtype='int64'), 'flag': [True]})
    rows = pd.DataFrame({'filename': ['a.pdf', 'b.pdf'], 'download_success': [True, False]})
    merged = merge_owned_rows(current, rows, 'filename')
    assert merged.foreign_id.tolist()[0] == BIG and pd.isna(merged.foreign_id.tolist()[1])
    assert str(merged.foreign_id.dtype) == 'Int64' and str(merged.flag.dtype) == 'boolean'


def test_mixed_schema_nullable_fields_survive():
    current = pd.DataFrame({
        'filename': ['same.pdf', None, 'foreign.pdf', 'owned.pdf'],
        'triage': ['same', 'keyless', 'foreign', 'old'],
        'foreign_int': pd.Series([1, 2, 3, 4], dtype='Int64'),
        'foreign_bool': pd.Series([True, False, None, True], dtype='boolean'),
        'when': pd.date_range('2026-10-01', periods=4, tz='UTC'),
    })
    rows = pd.DataFrame({'filename': ['same.pdf', 'new.pdf'], 'download_success': [True, False]})
    merged = merge_owned_rows(current, rows, 'filename', owned={'owned.pdf'})
    assert merged.filename.tolist()[:2] == ['same.pdf', 'new.pdf']
    assert merged.triage.dropna().tolist() == ['same', 'keyless', 'foreign']
    for column in ['foreign_int', 'foreign_bool', 'when']:
        assert merged[column].dtype == current[column].dtype
    for tag in ['same', 'keyless', 'foreign']:
        expected = current.loc[current.triage == tag].reset_index(drop=True)
        actual = merged.loc[merged.triage == tag, current.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)


def test_empty_current_columns_survive():
    current = pd.DataFrame({'filename': pd.Series([], dtype='str'), 'triage': pd.Series([], dtype='str'),
                            'foreign_id': pd.Series([], dtype='Int64'), 'count': pd.Series([], dtype='int64')})
    rows = pd.DataFrame({'filename': ['new.pdf']})
    result = merge_owned_rows(current, rows, 'filename')
    assert set(current.columns) <= set(result.columns)  # 01f53d4: ['filename']
    assert str(result.foreign_id.dtype) == 'Int64' and str(result['count'].dtype) == 'Int64'
    assert merge_owned_rows(pd.DataFrame(), rows, 'filename') is rows


def test_an_inexact_union_is_refused_before_anything_is_replaced(tmp_path):
    """Both sides have the column with dtypes whose union rounds (int64 2**53+1 vs float64): typed refusal."""
    from glossapi.parquet_schema import lossless_concat, pandas_parquet
    current = pd.DataFrame({'download_success': [True], 'size': pd.Series([BIG], dtype='int64')})
    rows = pd.DataFrame({'filename': ['n.pdf'], 'size': [1.5]})
    with pytest.raises(MetadataSchemaError, match="'size'"):
        lossless_concat([rows, current])
    target = tmp_path / 'download_results' / 'download_results.parquet'
    update_metadata_parquet(target, lambda _: current, serialize=pandas_parquet)
    before = target.read_bytes()
    with pytest.raises(MetadataSchemaError):
        update_metadata_parquet(target, lambda current_: merge_owned_rows(current_, rows, 'filename'))
    assert target.read_bytes() == before


def test_download_publication_keeps_exact_integers_of_rows_it_does_not_replace(tmp_path, monkeypatch):
    from glossapi import Corpus
    from glossapi.gloss_downloader import GlossDownloader
    source = tmp_path / 'urls.parquet'
    pd.DataFrame({'url': ['https://example.invalid/new.pdf']}).to_parquet(source, index=False)
    target = tmp_path / 'download_results/download_results_urls.parquet'
    target.parent.mkdir()
    pd.DataFrame({'url': ['https://example.invalid/old.pdf'], 'filename': ['old.pdf'], 'download_success': [True],
                  'foreign_id': pd.Series([BIG], dtype='int64')}).to_parquet(target, index=False)
    monkeypatch.setattr(GlossDownloader, 'download_files', lambda self, **kw: pd.DataFrame(
        {'url': ['https://example.invalid/new.pdf'], 'filename': ['new.pdf'], 'download_success': [True]}))
    Corpus(tmp_path, tmp_path).download(input_parquet=source)
    stored = pd.read_parquet(target)
    assert stored.loc[stored.filename == 'old.pdf', 'foreign_id'].tolist() == [BIG]


# ------------------------------------------------------------------ round 4: generic publishers (CP-K2e F1, F5)

MetadataStaleGenerationError = getattr(_schema, 'MetadataStaleGenerationError', _Missing)
MetadataDestinationError = getattr(_schema, 'MetadataDestinationError', _Missing)


def _seed_metadata(root):
    path = root / 'download_results/download_results.parquet'
    path.parent.mkdir(parents=True)
    pd.DataFrame({'filename': ['old.pdf'], 'url': ['u'], 'foreign_id': pd.Series([BIG], dtype='int64')}).to_parquet(path, index=False)
    return path


def test_caller_kernel_republication_preserves_a_metadata_file_symlink(tmp_path):
    """CP-K2e F1 (reviewer probe 1): the callback updates a symlinked metadata file; the adapter's republication goes
    through metadata_target, so the link survives and the physical file holds the callback's bytes. The callback runs
    with no metadata lock held (no nested acquisition)."""
    from glossapi import Corpus
    from glossapi.stages.caller_kernel import execute_caller_kernel
    corpus = Corpus(tmp_path, tmp_path)
    alias = _seed_metadata(tmp_path)
    physical = tmp_path / 'physical.parquet'
    alias.rename(physical)
    alias.symlink_to(physical)
    free = []

    def callback():
        with FileLock(str(resolve_metadata_target(alias)) + '.lock', timeout=0):
            free.append(True)  # nobody holds the metadata lock while the callback runs
        update_metadata_parquet(alias, lambda current: current.assign(triage='callback'))
        return 'done'

    assert execute_caller_kernel(corpus, 'clean', 'clean', callback) == 'done'
    assert free and alias.is_symlink() and alias.samefile(physical)  # 6bb0faf: link replaced by a second file
    final = pd.read_parquet(physical)
    assert final.triage.tolist() == ['callback'] and final.foreign_id.tolist() == [BIG]


def test_caller_kernel_republication_refuses_to_revert_a_commit_after_capture(tmp_path, monkeypatch):
    """CP-K2e F1 (reviewer probe 2): a locked commit lands after the generation captured the file and before its
    republication. 6bb0faf republished the stale capture and lost ``concurrent.pdf``; now the republication is refused
    under the lock with MetadataStaleGenerationError and the live file keeps the commit."""
    from glossapi import Corpus
    import glossapi.stages.caller_kernel as caller_kernel
    corpus = Corpus(tmp_path, tmp_path)
    target = _seed_metadata(tmp_path)
    real = caller_kernel.execute_aggregate

    def interleave(*args, **kwargs):
        result = real(*args, **kwargs)  # the callback's bytes are in the generation now
        update_metadata_parquet(target, lambda current: _schema.lossless_concat([
            current, pd.DataFrame({'filename': ['concurrent.pdf'], 'url': ['v']})]))
        return result

    monkeypatch.setattr(caller_kernel, 'execute_aggregate', interleave)
    with pytest.raises(MetadataStaleGenerationError, match='changed after its generation was captured'):
        caller_kernel.execute_caller_kernel(corpus, 'clean', 'clean', lambda: update_metadata_parquet(
            target, lambda current: current.assign(triage='callback')))
    final = pd.read_parquet(target)
    assert final.filename.tolist() == ['old.pdf', 'concurrent.pdf'] and final.triage.tolist()[0] == 'callback'


def test_caller_kernel_republication_without_concurrency_publishes(tmp_path):
    from glossapi import Corpus
    from glossapi.stages.caller_kernel import execute_caller_kernel
    corpus = Corpus(tmp_path, tmp_path)
    target = _seed_metadata(tmp_path)
    execute_caller_kernel(corpus, 'clean', 'clean', lambda: update_metadata_parquet(target, lambda c: c.assign(triage='x')))
    assert pd.read_parquet(target).triage.tolist() == ['x'] and not list(target.parent.glob('*.lock'))


@pytest.mark.parametrize('spelling', ['literal', 'file-symlink', 'new-download-results-name'])
def test_cli_export_refuses_a_metadata_destination(tmp_path, spelling):
    """CP-K2e F1 (CLI): the exporter never writes metadata; a destination resolving to an existing metadata parquet,
    or to a ``download_results/*.parquet`` name, is refused before anything is written."""
    from glossapi.cli import _export
    target = _seed_metadata(tmp_path)
    before = target.read_bytes()
    if spelling == 'literal':
        destination = target
    elif spelling == 'file-symlink':
        destination = tmp_path / 'export-link.parquet'
        destination.symlink_to(target)
    else:
        destination = target.parent / 'download_results_new.parquet'
    with pytest.raises(MetadataDestinationError):
        _export(destination, {'rows': [{'filename': 'new.pdf', 'url': 'v'}]}, 'parquet')
    assert target.read_bytes() == before
    if spelling == 'file-symlink':
        assert destination.is_symlink()
    if spelling == 'new-download-results-name':
        assert not destination.exists()


def test_cli_export_to_an_ordinary_path_still_works(tmp_path):
    from glossapi.cli import _export
    receipt = _export(tmp_path / 'report.parquet', {'rows': [{'filename': 'new.pdf', 'url': 'v'}]}, 'parquet')
    assert receipt['rows'] == 1 and pd.read_parquet(tmp_path / 'report.parquet').filename.tolist() == ['new.pdf']


def test_triage_propagates_hard_link_refusal(tmp_path):
    """CP-K2e F5 (reviewer probe): ``update_download_results_parquet`` returned None for a hard-linked target."""
    from glossapi.ocr.utils.triage import update_download_results_parquet
    path = _seed_metadata(tmp_path)
    os.link(path, tmp_path / 'alias.parquet')
    before = path.read_bytes()
    with pytest.raises(MetadataAliasError):
        update_download_results_parquet(tmp_path, 'new', {}, 'stop')
    assert path.read_bytes() == before


def test_processing_state_save_propagates_hard_link_refusal(tmp_path):
    """The other broad legacy handlers re-raise policy errors too (corpus_state shown; the static test checks all)."""
    from glossapi.corpus.corpus_state import _ProcessingStateManager
    path = _seed_metadata(tmp_path)
    os.link(path, tmp_path / 'alias.parquet')
    manager = _ProcessingStateManager(tmp_path / 'state.pkl')
    manager.base_dir = tmp_path
    with pytest.raises(MetadataAliasError):
        manager.save({'new'}, set())
    assert path.samefile(tmp_path / 'alias.parquet')


# ------------------------------------------------------------------ static: no writer bypasses the choke point

SRC = Path(_schema.__file__).resolve().parents[1]  # the imported package's src/ (so a red run scans the old tree)
CHOKE_WRITERS = {'_write_metadata_parquet_locked', '_write_serialized_locked', '_copy_file_locked'}
HELPERS = {'update_metadata_parquet', 'replace_metadata_parquet', 'rewrite_metadata_parquet'}
WRITE_FUNCTIONS = {'os.replace', 'os.rename', 'shutil.copyfile', 'shutil.copy', 'shutil.copy2', 'shutil.move',
                   'pyarrow.parquet.write_table', 'pyarrow.parquet.ParquetWriter'}
WRITE_METHODS = {'to_parquet': '.to_parquet', 'write_table': '.write_table', 'ParquetWriter': 'ParquetWriter'}
CHOKE = 'choke point: the replacement of the target yielded by metadata_target'
SERIALIZER = 'serializer passed as serialize= to a metadata helper; writes only to its path parameter (AST-checked)'
SECTIONS = 'section-level parquet (section column): find_metadata_parquet skips it; not a metadata target'
DOCUMENTS = 'document / PDF / binary payload files, not parquet metadata'
STAGEWORK = 'copy into a stage work directory or ledger area (immutable generation input / recovery), not a live target'
# Every concrete direct file write in src/: (module, enclosing scope, call) -> (count of call sites, reason). Counts are
# exact, so a second write in an allowlisted scope fails (CP-K2e F6); calls are matched after import-alias resolution.
EXPECTED = {
    ('glossapi/parquet_schema.py', '_write_metadata_parquet_locked', '.write_table'): (1, CHOKE),
    # Fix 7 (R1): the three choke writers' atomic replace is one helper that first records the uncommitted write.
    ('glossapi/parquet_schema.py', '_replace_in_place', 'os.replace'): (1, CHOKE),
    ('glossapi/parquet_schema.py', '_copy_file_locked', 'shutil.copyfile'): (1, CHOKE),
    ('glossapi/parquet_schema.py', 'pandas_parquet', '.to_parquet'): (1, SERIALIZER),
    ('glossapi/parquet_schema.py', 'ParquetSchema.create_basic_metadata_parquet', '.write_table'): (1, SERIALIZER),
    ('glossapi/parquet_schema.py', 'ParquetSchema.write_parquet', '.write_table'): (1, SERIALIZER),
    ('glossapi/stages/execution.py', 'link_published', 'os.replace'):
        (1, 'K4(c) decision 3: atomic rename of a fresh hard link of a private intermediate onto its per-document '
            'compatibility product; never a .parquet or symlinked destination (it returns None and the caller copies)'),
    ('glossapi/stages/execution.py', 'copy_reused', 'shutil.copyfile'):
        (1, 'K4(c) part 3 (K4C-G08): freeze_inputs copies a producer-bound input whose recorded hash this run already '
            'holds into the attempt\'s private inputs/ directory (a fresh name below the attempt); never corpus metadata'),
    ('glossapi/ledger/writer.py', '_restore_copy', 'shutil.copyfile'):
        (1, 'K4C-M2: republish an unflushed non-metadata byte copy (export.jsonl, markdown) from its intact managed '
            'source at the next run start; a .parquet copy goes through metadata_target + _copy_file_locked instead'),
    ('glossapi/ledger/writer.py', '_restore_copy', 'os.replace'):
        (1, 'K4C-M2: atomic replace of that temporary copy; never a .parquet destination (see the copyfile entry)'),
    ('glossapi/stages/products.py', '_publish_products_locked', 'os.replace'):
        (1, 'product publication; metadata-shaped products are written to the metadata_target-resolved physical '
            'path held by publish_products (or by a caller passing physical=), checked below'),
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async._write_checkpoint', '.to_parquet'):
        (1, 'download_results_<input>.partial.parquet checkpoint: excluded by find_metadata_parquet (.partial.), single owner'),
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async._write_checkpoint', 'os.replace'):
        (1, 'atomic replace of that .partial. checkpoint'),
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async_per_domain._write_checkpoint', '.to_parquet'):
        (1, 'download_results_<input>.partial.parquet checkpoint: excluded by find_metadata_parquet (.partial.), single owner'),
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async_per_domain._write_checkpoint', 'os.replace'):
        (1, 'atomic replace of that .partial. checkpoint'),
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async_per_domain.maybe_log_and_write_progress', 'os.replace'):
        (1, 'domain_progress.json'),
    ('glossapi/gloss_downloader.py', 'GlossDownloader._mark_domain_ssl_insecure', 'os.replace'): (1, 'downloader TLS policy state file'),
    ('glossapi/gloss_downloader.py', 'GlossDownloader._move_temp_file_to_final', 'os.rename'): (1, DOCUMENTS),
    ('glossapi/gloss_downloader.py', 'GlossDownloader._move_temp_file_to_final', 'os.replace'): (1, DOCUMENTS),
    ('glossapi/gloss_downloader.py', 'GlossDownloader._retain_superseded_request_body', 'os.replace'): (1, DOCUMENTS),
    ('glossapi/gloss_browser_downloader.py', 'BrowserGlossDownloader._write_recovered_file', 'os.replace'): (1, DOCUMENTS),
    ('glossapi/scripts/openarchives_ocr_merge.py', '_copy_once', 'shutil.copy2'): (1, DOCUMENTS),
    ('glossapi/annotation_partitions.py', 'predict_partitions', '.write_table'): (1, SECTIONS),
    ('glossapi/annotation_partitions.py', 'predict_partitions.flush', '.write_table'): (2, SECTIONS),
    ('glossapi/annotation_partitions.py', 'predict_partitions.flush', 'ParquetWriter'): (1, SECTIONS),
    ('glossapi/corpus/phase_annotate.py', 'AnnotatePhaseMixin._add_document_types', '.to_parquet'): (1, SECTIONS),
    ('glossapi/corpus/phase_annotate.py', 'AnnotatePhaseMixin.annotate', '.to_parquet'): (3, SECTIONS),
    ('glossapi/corpus/phase_sections.py', 'SectionPhaseMixin.section', '.to_parquet'): (1, SECTIONS),
    ('glossapi/gloss_section_classifier.py', 'GlossSectionClassifier._process_predictions', '.to_parquet'): (1, SECTIONS),
    ('glossapi/gloss_section_classifier.py', 'GlossSectionClassifier.fully_annotate', '.to_parquet'): (1, SECTIONS),
    ('glossapi/gloss_section.py', 'GlossSection.to_parquet', '.write_table'): (1, SECTIONS),
    ('glossapi/gloss_section.py', 'GlossSection.to_parquet', 'ParquetWriter'): (1, SECTIONS),
    ('glossapi/stages/sections.py', 'SectionsStage.finalize', '.write_table'): (1, SECTIONS),
    ('glossapi/stages/sections.py', 'SectionsStage.finalize', 'ParquetWriter'): (1, SECTIONS),
    ('glossapi/stages/run.py', 'prepare_streaming', '.to_parquet'):
        (1, '.glossapi/source-manifests (excluded from find_metadata_parquet), new unique directory'),
    ('glossapi/cli.py', '_export', '.write_table'):
        (1, 'CLI export to a caller path; a metadata destination is refused with MetadataDestinationError (checked below)'),
    ('glossapi/cli.py', '_export', 'os.replace'):
        (1, 'CLI export to a caller path; a metadata destination is refused with MetadataDestinationError (checked below)'),
    ('glossapi/corpus/corpus_skiplist.py', '_SkiplistManager.add', 'os.replace'): (1, 'skiplist text file'),
    ('glossapi/corpus/phase_extract.py', 'ExtractPhaseMixin.extract', 'shutil.copy2'): (1, DOCUMENTS),
    ('glossapi/page_quality.py', 'write_page_quality', 'os.replace'): (1, 'json/metrics/*.page_quality.json'),
    ('glossapi/pdf_repair/extract.py', 'repair_for_extract', 'os.replace'): (1, DOCUMENTS),
    ('glossapi/pdf_repair/tounicode.py', 'repair_pdf', 'os.replace'): (1, DOCUMENTS),
    ('glossapi/pdf_repair/tounicode.py', 'repair_pdf', 'shutil.copyfile'): (1, DOCUMENTS),
    ('glossapi/scripts/openarchives_pdf_stage_pull.py', 'run', 'os.replace'): (1, DOCUMENTS),
    ('glossapi/scripts/openarchives_pdf_stage_pull.py', 'write_json', 'os.replace'): (1, 'JSON summary'),
    ('glossapi/scripts/pdf_font_repair.py', '_batch', 'os.replace'): (1, DOCUMENTS),
    ('glossapi/scripts/pdf_font_repair.py', '_write_record', 'os.replace'): (1, 'JSON repair record'),
    ('glossapi/ledger/importer.py', '_sqlite_snapshot', 'shutil.copyfile'): (2, STAGEWORK),
    ('glossapi/stages/caller_kernel.py', 'CallerKernelStage.finalize', 'shutil.copyfile'):
        (1, STAGEWORK + '; its republication onto live files goes through publish_products(captured=True)'),
    ('glossapi/stages/export.py', 'ExportStage._finalize', 'shutil.copyfile'): (1, STAGEWORK),
    ('glossapi/stages/metadata.py', 'MetadataProjectionStage.finalize', 'shutil.copyfile'): (1, STAGEWORK),
    ('glossapi/stages/ocr.py', 'OcrStage._run_document', 'shutil.copy2'): (2, DOCUMENTS),
    ('glossapi/stages/scheduler.py', 'StreamingScheduler._recover', 'os.rename'): (1, STAGEWORK),
    # found by import-alias resolution (`from shutil import copy2`, `import shutil as _shutil`): source-document copies
    ('glossapi/corpus/batched_extract.py', 'DoclingAdapter.failure', 'shutil.copy2'): (1, DOCUMENTS + ' (failed PDF to problematic dir)'),
    ('glossapi/corpus/phase_ocr_math.py', '_gpu_math_worker._quarantine_items', 'shutil.copy2'): (2, DOCUMENTS + ' (quarantined PDFs)'),
    ('glossapi/gloss_extract.py', 'GlossExtract._process_file_chunked', 'shutil.copy2'): (1, DOCUMENTS + ' (timed-out PDF)'),
    ('glossapi/gloss_extract.py', 'GlossExtract._process_single_document', 'shutil.copy2'): (1, DOCUMENTS + ' (timed-out PDF)'),
    ('glossapi/gloss_extract.py', 'GlossExtract._process_single_document_pypdfium', 'shutil.copy2'): (2, DOCUMENTS + ' (timed-out PDF)'),
    ('glossapi/gloss_extract.py', 'GlossExtract.extract_path', 'shutil.copy2'): (1, DOCUMENTS + ' (problematic PDF)'),
    ('glossapi/stages/sequential.py', 'SequentialStageRunner.prepare.complete', 'os.rename'): (1, STAGEWORK),
}
# Metadata writers whose policy errors legacy broad `except` blocks must re-raise (CP-K2e F5).
POLICY_WRITERS = {'update_metadata_parquet', 'replace_metadata_parquet', 'rewrite_metadata_parquet', 'write_metadata_parquet',
                  'ensure_metadata_parquet', 'update_download_results_parquet', 'metadata_target'}
# Every call of the generic publisher, with how its metadata destinations are handled.
PUBLISHERS = {
    ('glossapi/stages/metadata.py', 'seal_metadata_projections'): 'physical= from an enclosing `with metadata_target(...)`',
    ('glossapi/stages/caller_kernel.py', 'execute_caller_kernel'): 'captured=True: guarded inside publish_products',
    ('glossapi/stages/annotate.py', '<call>'): 'section-level products; publish_products guards any metadata-shaped one',
    ('glossapi/stages/export.py', '<call>'): 'export products; publish_products guards any metadata-shaped one',
    ('glossapi/stages/sections.py', '<call>'): 'sections parquet; publish_products guards any metadata-shaped one',
}


def _aliases(tree):
    """Local name -> canonical dotted name, from the module's imports (``from shutil import copyfile as publish``,
    ``import shutil as sh``, ``from pyarrow import parquet as pq``, ...)."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                names[alias.asname or alias.name] = node.module + '.' + alias.name
    return names


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return base + '.' + node.attr if base else None
    return None


def _write_name(node, aliases):
    """The canonical write this call performs, or None."""
    dotted = _dotted(node.func)
    if dotted:
        head, _, rest = dotted.partition('.')
        canonical = aliases.get(head, head) + ('.' + rest if rest else '')
        canonical = canonical.replace('pyarrow.parquet.', 'pyarrow.parquet.')
        if canonical in WRITE_FUNCTIONS:
            return {'pyarrow.parquet.write_table': '.write_table', 'pyarrow.parquet.ParquetWriter': 'ParquetWriter'}.get(canonical, canonical)
    if isinstance(node.func, ast.Attribute) and node.func.attr in WRITE_METHODS:
        return WRITE_METHODS[node.func.attr]
    if isinstance(node.func, ast.Name) and node.func.id in WRITE_METHODS:
        return WRITE_METHODS[node.func.id]
    return None


def _call_name(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else None


def _walk(tree):
    """(node, enclosing scope names, ancestors) for every node."""
    def visit(node, scope, ancestors):
        named = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        inner = scope + [node.name] if named else scope
        yield node, scope, ancestors
        for child in ast.iter_child_nodes(node):
            yield from visit(child, inner, ancestors + [node])
    yield from visit(tree, [], [])


def _inside_metadata_target(ancestors):
    return any(isinstance(node, (ast.With, ast.AsyncWith)) and any(
        isinstance(item.context_expr, ast.Call) and _call_name(item.context_expr) == 'metadata_target'
        for item in node.items) for node in ancestors)


def _serializer_argument(ancestors):
    """Inside a lambda (or def) that is the serialize= argument of a metadata helper."""
    for parent, child in zip(ancestors, ancestors[1:]):
        if isinstance(parent, ast.Call) and _call_name(parent) in HELPERS and isinstance(child, ast.keyword) \
                and child.arg == 'serialize':
            return True
    return False


def _writes_only_its_path_parameter(node, ancestors, name):
    """A serializer writes only to the path parameter of its enclosing lambda/def (not a literal or other path)."""
    function = next((a for a in reversed(ancestors) if isinstance(a, (ast.Lambda, ast.FunctionDef))), None)
    if function is None:
        return False
    parameters = {argument.arg for argument in function.args.args}
    position = 0 if name == '.to_parquet' else 1
    destination = node.args[position] if len(node.args) > position else next(
        (keyword.value for keyword in node.keywords if keyword.arg in ('path', 'where')), None)
    return isinstance(destination, ast.Name) and destination.id in parameters


def static_problems(src):
    found, problems = {}, []
    for path in sorted(Path(src).rglob('*.py')):
        module = str(path.relative_to(src))
        tree = ast.parse(path.read_text(), str(path))
        aliases = _aliases(tree)
        for node, scope, ancestors in _walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'publish_products' and module == 'glossapi/stages/products.py':
                if not any(isinstance(call, ast.Call) and _call_name(call) == 'metadata_target' for call in ast.walk(node)):
                    problems.append('publish_products no longer routes metadata-shaped products through metadata_target')
            if not isinstance(node, ast.Call):
                continue
            where = '.'.join(scope) or '<module>'
            name = _write_name(node, aliases)
            if name:
                key = (module, where, name)
                found[key] = found.get(key, 0) + 1
                if key in EXPECTED and EXPECTED[key][1] == SERIALIZER:
                    if where != 'pandas_parquet' and not _serializer_argument(ancestors):
                        problems.append(f'{key} is listed as a serializer but is not a serialize= argument of {sorted(HELPERS)}')
                    if not _writes_only_its_path_parameter(node, ancestors, name):
                        problems.append(f'{key}:{node.lineno} serializer writes to something other than its path parameter')
            called = _call_name(node)
            # the choke-point writers run only inside `with metadata_target(...)` (or in the ensure merge it calls)
            if called in CHOKE_WRITERS and not _inside_metadata_target(ancestors) \
                    and where not in ('ParquetSchema._merge_metadata_locked',):
                problems.append(f'{module}:{node.lineno} {called} outside `with metadata_target(...)` ({where})')
            if called == '_merge_metadata_locked' and not _inside_metadata_target(ancestors):
                problems.append(f'{module}:{node.lineno} _merge_metadata_locked outside `with metadata_target(...)`')
            if called == 'publish_products' and module != 'glossapi/stages/products.py':
                listed = PUBLISHERS.get((module, where)) or PUBLISHERS.get((module, '<call>'))
                if listed is None:
                    problems.append(f'{module}:{node.lineno} unlisted publish_products call ({where})')
                if any(keyword.arg == 'physical' for keyword in node.keywords) and not _inside_metadata_target(ancestors):
                    problems.append(f'{module}:{node.lineno} physical= publication outside metadata_target')
                if module == 'glossapi/stages/caller_kernel.py' and not any(
                        keyword.arg == 'captured' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                        for keyword in node.keywords):
                    problems.append(f'{module}:{node.lineno} caller-kernel republication without captured=True')
            if called in POLICY_WRITERS and module != 'glossapi/parquet_schema.py':
                for ancestor, child in zip(ancestors, ancestors[1:] + [node]):
                    if isinstance(ancestor, ast.Try) and child in ancestor.body:
                        types = [ast.unparse(handler.type) if handler.type else 'BARE' for handler in ancestor.handlers]
                        broad = [t for t in types if t == 'BARE' or 'Exception' in t or 'ValueError' in t]
                        if broad and not (types and types[0] == 'MetadataPolicyError'):
                            problems.append(f'{module}:{node.lineno} {called} inside `try` whose handler {broad[0]} can '
                                            'swallow MetadataPolicyError (add `except MetadataPolicyError: raise` first)')
            if called == '_parquet_lock' and module != 'glossapi/parquet_schema.py':
                problems.append(f'{module}:{node.lineno} uses the raw lock primitive; use metadata_target')
    for key, count in sorted(found.items()):
        if key not in EXPECTED:
            problems.append(f'unlisted direct write {key} x{count}: route it through parquet_schema.metadata_target '
                            'helpers, or add it here with the reason it is not a metadata target')
        elif EXPECTED[key][0] != count:
            problems.append(f'{key}: {count} call sites, {EXPECTED[key][0]} listed (a new write in a listed scope?)')
    stale = sorted(set(EXPECTED) - set(found))
    if stale:
        problems.append(f'listed writers no longer present (update the list): {stale}')
    return problems


def test_no_metadata_writer_bypasses_the_choke_point():
    problems = static_problems(SRC)
    assert not problems, '\n'.join(problems)


def _shadow(tmp_path, edit=None, extra=None):
    """A shadow src/ of the imported package (symlinked files), optionally with one file edited and one added."""
    shadow = tmp_path / 'src'
    for original in SRC.rglob('*.py'):
        target = shadow / original.relative_to(SRC)
        target.parent.mkdir(parents=True, exist_ok=True)
        if edit and original == Path(_schema.__file__).resolve() and edit[0] == 'parquet_schema':
            text = original.read_text().replace(*edit[1])
            assert text != original.read_text()
            target.write_text(text)
        else:
            target.symlink_to(original)
    for name, text in (extra or {}).items():
        (shadow / name).write_text(text)
    return shadow


def test_the_static_check_catches_a_bypass(tmp_path):
    """Non-vacuous: a direct to_parquet in a new module and a choke writer outside the lock both fail."""
    shadow = _shadow(tmp_path, extra={
        'glossapi/bypass.py': 'def publish(frame, path):\n    frame.to_parquet(path, index=False)\n',
        'glossapi/raw.py': 'from .parquet_schema import _write_serialized_locked\n'
                           'def publish(frame, path):\n    _write_serialized_locked(frame, path, None)\n',
        'glossapi/swallow.py': 'from .parquet_schema import update_metadata_parquet\n'
                               'def quiet(path):\n    try:\n        update_metadata_parquet(path, lambda c: c)\n'
                               '    except Exception:\n        return None\n'})
    problems = '\n'.join(static_problems(shadow))
    assert 'unlisted direct write' in problems and 'outside `with metadata_target' in problems
    assert 'can swallow MetadataPolicyError' in problems


def test_static_check_catches_an_import_aliased_write(tmp_path):
    """CP-K2e F6 mutation 1 (the reviewer's control): ``from shutil import copyfile as publish``."""
    shadow = _shadow(tmp_path, extra={'hidden.py': 'from shutil import copyfile as publish\n'
                                                  'def overwrite(source, metadata_path):\n'
                                                  '    publish(source, metadata_path)\n'})
    problems = static_problems(shadow)
    assert any('unlisted direct write' in problem and 'shutil.copyfile' in problem for problem in problems), problems


def test_static_check_catches_a_second_write_in_an_allowlisted_scope(tmp_path):
    """CP-K2e F6 mutation 2 (the reviewer's control): a second to_parquet inside ``pandas_parquet``."""
    shadow = _shadow(tmp_path, edit=('parquet_schema', (
        '    frame.to_parquet(path, index=False)\n',
        '    frame.to_parquet(path, index=False)\n'
        '    frame.to_parquet("download_results/download_results.parquet", index=False)\n')))
    problems = '\n'.join(static_problems(shadow))
    assert '2 call sites, 1 listed' in problems and 'other than its path parameter' in problems


# ------------------------------------------------------------------ static: the K3 interface modules (integration)
# deps/k3-interface added the decision-interface modules and CLI commands. EXPECTED above inventories only parquet and
# replace/copy writes, so these modules (and the CLI they share with K2) are also scanned with a broader detector:
# text/byte writes, write-mode opens, deletes, directories, temporary files and child processes. Every such call is
# listed with its reason; none writes metadata parquet, and the two caller-path writers refuse a metadata destination
# (``refuse_metadata_destination``) before their first write.
K3_MODULES = ('glossapi/capabilities.py', 'glossapi/doctor.py', 'glossapi/input_manifest.py', 'glossapi/install_plan.py',
              'glossapi/llm_guide.py', 'glossapi/package_catalog.py', 'glossapi/plan_preview.py',
              'glossapi/settings_contract.py', 'glossapi/schemas/__init__.py', 'glossapi/cli.py')
BROAD_FUNCTIONS = WRITE_FUNCTIONS | {
    'os.open', 'os.unlink', 'os.remove', 'os.rmdir', 'os.mkdir', 'os.makedirs', 'os.system', 'shutil.rmtree',
    'tempfile.mkstemp', 'tempfile.mkdtemp', 'tempfile.NamedTemporaryFile', 'tempfile.TemporaryDirectory',
    'subprocess.run', 'subprocess.Popen', 'subprocess.call', 'subprocess.check_call', 'subprocess.check_output'}
BROAD_METHODS = {'write_text', 'write_bytes', 'unlink', 'mkdir', 'rmdir', 'touch', 'rename', 'to_parquet', 'write_table',
                 'to_csv', 'to_json'}
EXPORT = 'CLI export to a caller path (report --output); refuse_metadata_destination runs first (checked below)'
K3_EXPECTED = {
    ('glossapi/doctor.py', '_child', 'subprocess.run'):
        (1, 'doctor --probe-devices: a child interpreter (-I) imports the device framework and prints JSON; no file'),
    ('glossapi/input_manifest.py', 'write_manifest', '.write_text'):
        (1, 'caller-chosen input manifest JSON (Python API); refuse_metadata_destination runs first (checked below)'),
    ('glossapi/install_plan.py', '_snapshot', 'subprocess.run'):
        (1, "reads the environment's installed distributions in a fresh interpreter; no file"),
    ('glossapi/install_plan.py', 'execute_install', 'subprocess.run'):
        (2, 'install --yes only: the planned pip/uv commands (the Python environment, never corpus files) and the '
            "installer's read-only dependency check"),
    ('glossapi/cli.py', '_export', 'tempfile.mkstemp'): (1, EXPORT),
    ('glossapi/cli.py', '_export', '.write_table'): (1, EXPORT),
    ('glossapi/cli.py', '_export', '.write_text'): (1, EXPORT),
    ('glossapi/cli.py', '_export', 'os.replace'): (1, EXPORT),
    ('glossapi/cli.py', '_export', '.unlink'): (1, "removes _export's own temporary file when the export fails"),
    ('glossapi/cli.py', 'main', '.mkdir'): (1, '`run`: creates the corpus root after the plan checks (plan never reaches it)'),
}
REFUSING_WRITERS = {('glossapi/cli.py', '_export'), ('glossapi/input_manifest.py', 'write_manifest')}


def _broad_write_name(node, aliases):
    dotted = _dotted(node.func)
    if dotted:
        head, _, rest = dotted.partition('.')
        canonical = aliases.get(head, head) + ('.' + rest if rest else '')
        if canonical in BROAD_FUNCTIONS:
            return {'pyarrow.parquet.write_table': '.write_table', 'pyarrow.parquet.ParquetWriter': 'ParquetWriter'}.get(canonical, canonical)
    if isinstance(node.func, ast.Attribute) and node.func.attr in BROAD_METHODS:
        return '.' + node.func.attr
    if _call_name(node) == 'open':
        if isinstance(node.func, ast.Name):
            mode = node.args[1] if len(node.args) > 1 else None
        else:
            mode = node.args[0] if node.args else None
        mode = mode or next((keyword.value for keyword in node.keywords if keyword.arg == 'mode'), None)
        value = 'r' if mode is None else mode.value if isinstance(mode, ast.Constant) else '?'
        if any(flag in str(value) for flag in 'wax+?'):
            return f'open({value})'
    return None


def k3_static_problems(src, modules=K3_MODULES):
    found, first_write, refusal, problems = {}, {}, {}, []
    for module in modules:
        tree = ast.parse((Path(src) / module).read_text())
        aliases = _aliases(tree)
        for node, scope, _ancestors in _walk(tree):
            if not isinstance(node, ast.Call):
                continue
            where = '.'.join(scope) or '<module>'
            if _call_name(node) == 'refuse_metadata_destination':
                refusal[(module, where)] = min(refusal.get((module, where), node.lineno), node.lineno)
            name = _broad_write_name(node, aliases)
            if name:
                key = (module, where, name)
                found[key] = found.get(key, 0) + 1
                first_write[(module, where)] = min(first_write.get((module, where), node.lineno), node.lineno)
    for key, count in sorted(found.items()):
        if key not in K3_EXPECTED:
            problems.append(f'unlisted K3 write {key} x{count}: list it with its reason, or route metadata through '
                            'parquet_schema.metadata_target')
        elif K3_EXPECTED[key][0] != count:
            problems.append(f'{key}: {count} call sites, {K3_EXPECTED[key][0]} listed')
    stale = sorted(key for key in set(K3_EXPECTED) - set(found) if key[0] in modules)
    if stale:
        problems.append(f'listed K3 writes no longer present (update the list): {stale}')
    for writer in sorted(REFUSING_WRITERS):
        if writer[0] in modules and not refusal.get(writer, 10 ** 9) < first_write.get(writer, 0):
            problems.append(f'{writer} writes before refuse_metadata_destination (a metadata destination must be refused first)')
    return problems


def test_k3_interface_modules_write_only_declared_files():
    problems = k3_static_problems(SRC)
    assert not problems, '\n'.join(problems)


def test_k3_static_check_is_not_vacuous(tmp_path):
    """A planted text write, a write-mode open and a manifest writer that refuses only after writing all fail."""
    shadow = tmp_path / 'src'
    for module in K3_MODULES:
        (shadow / module).parent.mkdir(parents=True, exist_ok=True)
        text = (SRC / module).read_text()
        if module == 'glossapi/input_manifest.py':
            refusal = "    refuse_metadata_destination(path, writer='write a manifest')\n"
            assert refusal in text
            text = text.replace(refusal, '').replace('    return target\n', '    ' + refusal.strip() + '\n    return target\n', 1)
        (shadow / module).write_text(text)
    (shadow / 'glossapi/k3_bypass.py').write_text(
        'from pathlib import Path\n'
        'def save(path, text):\n    Path(path).write_text(text)\n'
        'def append(path):\n    with open(path, "a") as stream:\n        stream.write("x")\n')
    problems = '\n'.join(k3_static_problems(shadow, K3_MODULES + ('glossapi/k3_bypass.py',)))
    assert "('glossapi/k3_bypass.py', 'save', '.write_text')" in problems
    assert "('glossapi/k3_bypass.py', 'append', 'open(a)')" in problems
    assert "('glossapi/input_manifest.py', 'write_manifest') writes before refuse_metadata_destination" in problems


@pytest.mark.parametrize('spelling', ['literal', 'file-symlink', 'new-download-results-name'])
def test_write_manifest_refuses_a_metadata_destination(tmp_path, spelling):
    """The K3 manifest writer is not a metadata writer: like the CLI exporter it refuses a metadata destination
    before anything is written."""
    from glossapi.input_manifest import build_manifest, write_manifest
    target = _seed_metadata(tmp_path)
    before = target.read_bytes()
    if spelling == 'literal':
        destination = target
    elif spelling == 'file-symlink':
        destination = tmp_path / 'manifest-link.json'
        destination.symlink_to(target)
    else:
        destination = target.parent / 'download_results_manifest.parquet'
    manifest = build_manifest([{'path': str(tmp_path / 'a.pdf'), 'metadata': {'title': 'α'}}])
    with pytest.raises(MetadataDestinationError):
        write_manifest(manifest, destination)
    assert target.read_bytes() == before
    if spelling == 'file-symlink':
        assert destination.is_symlink()
    if spelling == 'new-download-results-name':
        assert not destination.exists()
    written = write_manifest(manifest, tmp_path / 'manifest.json')
    assert json.loads(written.read_text(encoding='utf-8')) == manifest

# ------------------------------------------------------------------ static: every write in src/ (INTEG-G3)
# EXPECTED inventories parquet and replace/copy writes (metadata), K3_EXPECTED the interface modules call by call. This
# per-module allowlist covers every module of src/ with the broad detector (text/byte writes, write-mode opens,
# deletes, directories, temporary files, child processes): a module may perform only the write kinds listed for it,
# with the reason. A new module that writes, or a new kind of write in a module, fails until it is listed here.
# Metadata parquet is never among the reasons: those writes go through parquet_schema.metadata_target (EXPECTED above).
PIPELINE = ('legacy pipeline products in the corpus output tree (downloads, markdown, sidecars, logs, worker markers, '
            'section parquet, export), the corpus skiplist/state, and probe child processes (nvidia-smi, the native '
            'build); metadata parquet only through the parquet_schema helpers')
PIPELINE_FP = PIPELINE + '; `open(?)` here is not a file open (PIL Image.open / the batched adapter.open)'
OCR = 'OCR and math products and sidecars in the corpus output tree (page artifacts, JSON helpers, route files)'
LEDGER = 'the ledger database, its migration backup and sidecar files, and git provenance probes (read-only child)'
STAGES = ('stage attempt workspaces, private generations and published products (publish_products, retention '
          'deletes of owned files, interruption markers); metadata only through metadata_target')
SCRIPTS = 'offline operator and benchmark scripts writing to caller-chosen output directories'
K3 = 'decision-interface modules: inventoried call by call in K3_EXPECTED above'
MODEL = 'copies the neutral section-classifier files named by a hash-checked manifest into the frozen model directory'
CHOKE_MODULE = 'the metadata choke point itself: lock files, atomic replacement, staged copies (checked by EXPECTED)'
ALL_WRITES = {
    'glossapi/annotation_partitions.py': (['.write_table', 'ParquetWriter', 'open(wb)', 'os.makedirs'], PIPELINE),
    'glossapi/cli.py': (['.mkdir', '.unlink', '.write_table', '.write_text', 'os.replace', 'tempfile.mkstemp'], K3),
    'glossapi/corpus/batched_extract.py': (['.mkdir', '.unlink', 'open(?)', 'shutil.copy2'], PIPELINE_FP),
    'glossapi/corpus/corpus_orchestrator.py': (['.mkdir', '.unlink', '.write_text', 'open(a)', 'os.makedirs', 'subprocess.run'], PIPELINE),
    'glossapi/corpus/corpus_skiplist.py': (['.mkdir', '.write_text', 'os.replace'], PIPELINE),
    'glossapi/corpus/ocr/dispatch.py': (['.mkdir', '.write_text'], PIPELINE),
    'glossapi/corpus/phase_annotate.py': (['.to_parquet'], PIPELINE),
    'glossapi/corpus/phase_clean.py': (['.mkdir', '.rename', '.unlink', '.write_text', 'open(w)', 'shutil.rmtree', 'subprocess.Popen', 'subprocess.run'], PIPELINE),
    'glossapi/corpus/phase_download.py': (['os.makedirs', 'os.remove'], PIPELINE),
    'glossapi/corpus/phase_export.py': (['.mkdir', '.write_text', 'open(w)', 'open(wb)'], PIPELINE),
    'glossapi/corpus/phase_extract.py': (['.mkdir', '.unlink', 'os.makedirs', 'shutil.copy2', 'subprocess.run'], PIPELINE),
    'glossapi/corpus/phase_ocr_math.py': (['.mkdir', '.unlink', '.write_text', 'shutil.copy2', 'subprocess.run'], PIPELINE),
    'glossapi/corpus/phase_sections.py': (['.to_parquet', 'os.makedirs'], PIPELINE),
    'glossapi/doctor.py': (['subprocess.run'], K3),
    'glossapi/gloss_browser_downloader.py': (['open(?)', 'os.replace'], PIPELINE_FP),
    'glossapi/gloss_downloader.py': (['.mkdir', '.to_parquet', '.write_text', 'open(w)', 'os.makedirs', 'os.remove', 'os.rename', 'os.replace'], PIPELINE),
    'glossapi/gloss_extract.py': (['.mkdir', '.write_text', 'open(w)', 'shutil.copy2'], PIPELINE),
    'glossapi/gloss_section.py': (['.write_table', 'ParquetWriter', 'os.makedirs'], PIPELINE),
    'glossapi/gloss_section_classifier.py': (['.to_parquet', 'os.makedirs'], PIPELINE),
    'glossapi/input_manifest.py': (['.write_text'], K3),
    'glossapi/install_plan.py': (['subprocess.run'], K3),
    'glossapi/ledger/importer.py': (['shutil.copyfile', 'tempfile.TemporaryDirectory'], LEDGER),
    'glossapi/ledger/journal.py': (['subprocess.check_output'], LEDGER),
    'glossapi/ledger/writer.py': (['os.open', 'os.replace', 'os.unlink', 'shutil.copyfile'], LEDGER),  # os.unlink: the fix 7 rollback removes a failed attempt's own uncommitted file, fix 8 a verified abandoned temporary (_recover_locked)
    'glossapi/ocr/defaults.py': (['.write_text'], OCR),
    'glossapi/ocr/math/enrich.py': (['.mkdir', '.write_text', 'open(a)', 'open(w)'], OCR),
    'glossapi/ocr/page_artifacts.py': (['.mkdir', '.unlink', '.write_text', 'open(xb)'], OCR),
    'glossapi/ocr/utils/json_io.py': (['.write_text', 'open(wb)'], OCR),
    'glossapi/ocr/utils/triage.py': (['.mkdir', '.write_text'], OCR),
    'glossapi/page_quality.py': (['.mkdir', '.unlink', 'os.replace', 'tempfile.NamedTemporaryFile'], PIPELINE),
    'glossapi/parquet_schema.py': (['.mkdir', '.to_parquet', '.unlink', '.write_table', 'os.makedirs', 'os.open', 'os.replace', 'os.unlink', 'shutil.copyfile'], CHOKE_MODULE),
    'glossapi/pdf_repair/extract.py': (['.mkdir', 'os.replace', 'tempfile.TemporaryDirectory'], PIPELINE),
    'glossapi/pdf_repair/tounicode.py': (['.mkdir', '.unlink', 'os.replace', 'shutil.copyfile', 'tempfile.mkstemp'], PIPELINE),
    'glossapi/scripts/batched_extract_benchmark.py': (['.mkdir', '.write_text'], SCRIPTS),
    'glossapi/scripts/batched_stability_benchmark.py': (['.mkdir', '.touch', '.unlink', '.write_text', 'open(a)', 'subprocess.Popen'], SCRIPTS),
    'glossapi/scripts/build_ocr_golden_pages.py': (['.mkdir', '.unlink', '.write_text', 'open(w)'], SCRIPTS),
    'glossapi/scripts/docling_cost_benchmark.py': (['.mkdir', '.touch', '.write_text', 'open(a)', 'open(w)', 'subprocess.run'], SCRIPTS),
    'glossapi/scripts/extract_checkpoint_benchmark.py': (['.mkdir', '.write_text', 'shutil.rmtree'], SCRIPTS),
    'glossapi/scripts/full_pipeline_checkpoint.py': (['.mkdir', '.unlink', '.write_text', 'shutil.rmtree'], SCRIPTS),
    'glossapi/scripts/install_glossapi.py': (['subprocess.run'], SCRIPTS),
    'glossapi/scripts/ocr_gpu_batch.py': (['.mkdir'], SCRIPTS),
    'glossapi/scripts/openarchives_download_freeze.py': (['.mkdir'], SCRIPTS),
    'glossapi/scripts/openarchives_download_probe.py': (['.mkdir', '.write_text'], SCRIPTS),
    'glossapi/scripts/openarchives_hf_refresh.py': (['.mkdir', '.write_text', 'open(wb)'], SCRIPTS),
    'glossapi/scripts/openarchives_manifest.py': (['.mkdir'], SCRIPTS),
    'glossapi/scripts/openarchives_ocr_cutoff_shards.py': (['.mkdir', '.write_text'], SCRIPTS),
    'glossapi/scripts/openarchives_ocr_enrich.py': (['.mkdir'], SCRIPTS),
    'glossapi/scripts/openarchives_ocr_merge.py': (['.mkdir', '.write_text', 'shutil.copy2'], SCRIPTS),
    'glossapi/scripts/openarchives_ocr_shards.py': (['.mkdir', '.write_text'], SCRIPTS),
    'glossapi/scripts/openarchives_pdf_stage_pull.py': (['.mkdir', '.unlink', '.write_text', 'open(a)', 'os.replace', 'subprocess.run'], SCRIPTS),
    'glossapi/scripts/page_quality_batch.py': (['.mkdir', '.write_text', 'open(w)', 'tempfile.TemporaryDirectory'], SCRIPTS),
    'glossapi/scripts/pdf_font_repair.py': (['.mkdir', '.write_text', 'open(w)', 'os.replace', 'tempfile.TemporaryDirectory'], SCRIPTS),
    'glossapi/scripts/review_manifest_materialize.py': (['.mkdir', '.unlink', '.write_text', 'open(w)'], SCRIPTS),
    'glossapi/scripts/sample_report.py': (['.write_text'], SCRIPTS),
    'glossapi/scripts/table_markdown_audit.py': (['.mkdir', '.unlink', '.write_text', 'open(w)'], SCRIPTS),
    'glossapi/scripts/table_sentence_context_review.py': (['.mkdir', '.unlink', '.write_text', 'open(w)'], SCRIPTS),
    'glossapi/section_model.py': (['.write_bytes'], MODEL),
    'glossapi/stages/acquisition_pool.py': (['.mkdir'], STAGES),
    'glossapi/stages/annotate.py': (['.mkdir'], STAGES),
    'glossapi/stages/caller_kernel.py': (['.mkdir', 'shutil.copyfile'], STAGES),
    'glossapi/stages/clean.py': (['.mkdir', '.write_bytes', 'shutil.rmtree', 'tempfile.mkdtemp'], STAGES),
    'glossapi/stages/compatibility.py': (['.mkdir', '.write_text', 'open(w)', 'shutil.rmtree', 'tempfile.TemporaryDirectory', 'tempfile.mkdtemp'], STAGES),
    'glossapi/stages/download.py': (['.mkdir'], STAGES),
    'glossapi/stages/download_compatibility.py': (['shutil.rmtree', 'tempfile.mkdtemp'], STAGES),
    'glossapi/stages/execution.py': (['.mkdir', 'open(wb)', 'os.replace', 'os.unlink', 'shutil.copyfile'], STAGES),
    'glossapi/stages/export.py': (['.mkdir', 'shutil.copyfile', 'shutil.rmtree'], STAGES),
    'glossapi/stages/extract.py': (['.mkdir'], STAGES),
    'glossapi/stages/interruption.py': (['.mkdir', 'os.open'], STAGES),
    'glossapi/stages/math.py': (['.mkdir', '.write_bytes', '.write_text'], STAGES),
    'glossapi/stages/metadata.py': (['.mkdir', 'shutil.copyfile'], STAGES),
    'glossapi/stages/ocr.py': (['.mkdir', '.write_bytes', '.write_text', 'shutil.copy2'], STAGES),
    'glossapi/stages/products.py': (['.mkdir', '.unlink', 'os.open', 'os.replace', 'shutil.rmtree', 'tempfile.mkdtemp'], STAGES),  # os.open: fix 8 publication temporary, named before its intent record binds it (O_EXCL, 0600)
    'glossapi/stages/retention.py': (['os.open', 'os.unlink'], STAGES),
    'glossapi/stages/run.py': (['.mkdir', '.to_parquet'], STAGES),
    'glossapi/stages/scheduler.py': (['.mkdir', 'os.rename'], STAGES),
    'glossapi/stages/sections.py': (['.mkdir', '.write_table', 'ParquetWriter', 'shutil.rmtree'], STAGES),
    'glossapi/stages/sequential.py': (['.mkdir', '.rmdir', 'os.open', 'os.rename'], STAGES),
    'glossapi/stages/workers.py': (['.mkdir'], STAGES),
}


def all_write_problems(src):
    found = {}
    for path in sorted(Path(src).rglob('*.py')):
        module = str(path.relative_to(src))
        tree = ast.parse(path.read_text(), str(path))
        aliases = _aliases(tree)
        for node, _scope, _ancestors in _walk(tree):
            if isinstance(node, ast.Call):
                name = _broad_write_name(node, aliases)
                if name:
                    found.setdefault(module, set()).add(name)
    problems = [f'unlisted write kind(s) {sorted(kinds - set(ALL_WRITES.get(module, ((), ""))[0]))} in {module}: add them '
                'to ALL_WRITES with the reason, or route metadata through parquet_schema.metadata_target'
                for module, kinds in sorted(found.items()) if kinds - set(ALL_WRITES.get(module, ((), ''))[0])]
    stale = sorted((module, kind) for module, (kinds, _reason) in ALL_WRITES.items() for kind in kinds
                   if kind not in found.get(module, set()))
    return problems, stale


def test_every_module_writes_only_its_listed_kinds():
    problems, _stale = all_write_problems(SRC)
    assert not problems, '\n'.join(problems)


def test_the_all_writes_inventory_has_no_stale_entries():
    _problems, stale = all_write_problems(SRC)
    assert not stale, f'listed writes no longer present (update ALL_WRITES): {stale}'


def test_the_all_writes_check_is_not_vacuous(tmp_path):
    shadow = _shadow(tmp_path, extra={'glossapi/new_writer.py': 'from pathlib import Path\n'
                                                                'def save(path):\n    Path(path).write_text("x")\n'})
    problems, _stale = all_write_problems(shadow)
    assert any('new_writer.py' in problem and '.write_text' in problem for problem in problems), problems
