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


# ------------------------------------------------------------------ static: no writer bypasses the choke point

SRC = Path(_schema.__file__).resolve().parents[1]  # the imported package's src/ (so a red run scans the old tree)
CHOKE_WRITERS = {'_write_metadata_parquet_locked', '_write_serialized_locked', '_copy_file_locked'}
HELPERS = {'update_metadata_parquet', 'replace_metadata_parquet', 'rewrite_metadata_parquet'}
WRITES = {('os', 'replace'), ('os', 'rename'), ('shutil', 'copyfile'), ('shutil', 'copy'), ('shutil', 'copy2'), ('shutil', 'move')}
# Every direct file write in src/, by (module, enclosing scope, call): how it relates to the metadata policy.
CHOKE = 'choke point: the replacement of the target yielded by metadata_target'
SERIALIZER = 'serializer passed as serialize= to a metadata helper (writes only the helper temporary file)'
SECTIONS = 'section-level parquet (section column): find_metadata_parquet skips it; not a metadata target'
DOCUMENTS = 'document / PDF / binary payload files, not parquet metadata'
STAGEWORK = 'copy into a stage work directory or ledger area (immutable generation input / recovery), not a live target'
EXPECTED = {
    ('glossapi/parquet_schema.py', '_write_metadata_parquet_locked', '.write_table'): CHOKE,
    ('glossapi/parquet_schema.py', '_write_metadata_parquet_locked', 'os.replace'): CHOKE,
    ('glossapi/parquet_schema.py', '_write_serialized_locked', 'os.replace'): CHOKE,
    ('glossapi/parquet_schema.py', '_copy_file_locked', 'os.replace'): CHOKE,
    ('glossapi/parquet_schema.py', '_copy_file_locked', 'shutil.copyfile'): CHOKE,
    ('glossapi/parquet_schema.py', 'pandas_parquet', '.to_parquet'): SERIALIZER,
    ('glossapi/parquet_schema.py', 'ParquetSchema.create_basic_metadata_parquet', '.write_table'): SERIALIZER,
    ('glossapi/parquet_schema.py', 'ParquetSchema.write_parquet', '.write_table'): SERIALIZER,
    ('glossapi/stages/products.py', 'publish_products', 'os.replace'):
        'product mirror publication; the metadata projection passes physical= from metadata_target (checked below)',
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async._write_checkpoint', '.to_parquet'):
        'download_results_<input>.partial.parquet checkpoint: excluded by find_metadata_parquet (.partial.), single owner',
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async._write_checkpoint', 'os.replace'):
        'atomic replace of that .partial. checkpoint',
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async_per_domain._write_checkpoint', '.to_parquet'):
        'download_results_<input>.partial.parquet checkpoint: excluded by find_metadata_parquet (.partial.), single owner',
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async_per_domain._write_checkpoint', 'os.replace'):
        'atomic replace of that .partial. checkpoint',
    ('glossapi/gloss_downloader.py', 'GlossDownloader._download_files_async_per_domain.maybe_log_and_write_progress', 'os.replace'):
        'domain_progress.json',
    ('glossapi/gloss_downloader.py', 'GlossDownloader._mark_domain_ssl_insecure', 'os.replace'): 'downloader TLS policy state file',
    ('glossapi/gloss_downloader.py', 'GlossDownloader._move_temp_file_to_final', 'os.rename'): DOCUMENTS,
    ('glossapi/gloss_downloader.py', 'GlossDownloader._move_temp_file_to_final', 'os.replace'): DOCUMENTS,
    ('glossapi/gloss_downloader.py', 'GlossDownloader._retain_superseded_request_body', 'os.replace'): DOCUMENTS,
    ('glossapi/gloss_browser_downloader.py', 'BrowserGlossDownloader._write_recovered_file', 'os.replace'): DOCUMENTS,
    ('glossapi/scripts/openarchives_ocr_merge.py', 'main', '.to_parquet'):
        'DECLARED EXCEPTION (GAPS K2F-G2): offline operator merge; output = master would be an unlocked in-place rewrite',
    ('glossapi/scripts/openarchives_ocr_merge.py', '_copy_once', 'shutil.copy2'): DOCUMENTS,
    ('glossapi/annotation_partitions.py', 'predict_partitions', '.write_table'): SECTIONS,
    ('glossapi/annotation_partitions.py', 'predict_partitions.flush', '.write_table'): SECTIONS,
    ('glossapi/annotation_partitions.py', 'predict_partitions.flush', 'ParquetWriter'): SECTIONS,
    ('glossapi/corpus/phase_annotate.py', 'AnnotatePhaseMixin._add_document_types', '.to_parquet'): SECTIONS,
    ('glossapi/corpus/phase_annotate.py', 'AnnotatePhaseMixin.annotate', '.to_parquet'): SECTIONS,
    ('glossapi/corpus/phase_sections.py', 'SectionPhaseMixin.section', '.to_parquet'): SECTIONS,
    ('glossapi/gloss_section_classifier.py', 'GlossSectionClassifier._process_predictions', '.to_parquet'): SECTIONS,
    ('glossapi/gloss_section_classifier.py', 'GlossSectionClassifier.fully_annotate', '.to_parquet'): SECTIONS,
    ('glossapi/gloss_section.py', 'GlossSection.to_parquet', '.write_table'): SECTIONS,
    ('glossapi/gloss_section.py', 'GlossSection.to_parquet', 'ParquetWriter'): SECTIONS,
    ('glossapi/stages/sections.py', 'SectionsStage.finalize', '.write_table'): SECTIONS,
    ('glossapi/stages/sections.py', 'SectionsStage.finalize', 'ParquetWriter'): SECTIONS,
    ('glossapi/stages/run.py', 'prepare_streaming', '.to_parquet'):
        '.glossapi/source-manifests (excluded from find_metadata_parquet), new unique directory',
    ('glossapi/cli.py', '_export', '.write_table'): 'explicit CLI export to a caller path (temporary + atomic replace), no read',
    ('glossapi/cli.py', '_export', 'os.replace'): 'explicit CLI export to a caller path (temporary + atomic replace), no read',
    ('glossapi/corpus/corpus_skiplist.py', '_SkiplistManager.add', 'os.replace'): 'skiplist text file',
    ('glossapi/corpus/phase_extract.py', 'ExtractPhaseMixin.extract', 'shutil.copy2'): DOCUMENTS,
    ('glossapi/page_quality.py', 'write_page_quality', 'os.replace'): 'json/metrics/*.page_quality.json',
    ('glossapi/pdf_repair/extract.py', 'repair_for_extract', 'os.replace'): DOCUMENTS,
    ('glossapi/pdf_repair/tounicode.py', 'repair_pdf', 'os.replace'): DOCUMENTS,
    ('glossapi/pdf_repair/tounicode.py', 'repair_pdf', 'shutil.copyfile'): DOCUMENTS,
    ('glossapi/scripts/openarchives_pdf_stage_pull.py', 'run', 'os.replace'): DOCUMENTS,
    ('glossapi/scripts/openarchives_pdf_stage_pull.py', 'write_json', 'os.replace'): 'JSON summary',
    ('glossapi/scripts/pdf_font_repair.py', '_batch', 'os.replace'): DOCUMENTS,
    ('glossapi/scripts/pdf_font_repair.py', '_write_record', 'os.replace'): 'JSON repair record',
    ('glossapi/ledger/importer.py', '_sqlite_snapshot', 'shutil.copyfile'): STAGEWORK,
    ('glossapi/stages/caller_kernel.py', 'CallerKernelStage.finalize', 'shutil.copyfile'): STAGEWORK,
    ('glossapi/stages/export.py', 'ExportStage.finalize', 'shutil.copyfile'): STAGEWORK,
    ('glossapi/stages/metadata.py', 'MetadataProjectionStage.finalize', 'shutil.copyfile'): STAGEWORK,
    ('glossapi/stages/ocr.py', 'OcrStage._run_document', 'shutil.copy2'): DOCUMENTS,
    ('glossapi/stages/scheduler.py', 'StreamingScheduler._recover', 'os.rename'): STAGEWORK,
    ('glossapi/stages/sequential.py', 'SequentialStageRunner.prepare.complete', 'os.rename'): STAGEWORK,
}


def _call_name(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and (func.value.id, func.attr) in WRITES:
            return func.value.id + '.' + func.attr
        if func.attr in ('to_parquet', 'write_table'):
            return '.' + func.attr
        if func.attr == 'ParquetWriter':
            return 'ParquetWriter'
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


def test_no_metadata_writer_bypasses_the_choke_point():
    found, problems = {}, []
    for path in sorted(SRC.rglob('*.py')):
        module = str(path.relative_to(SRC))
        for node, scope, ancestors in _walk(ast.parse(path.read_text(), str(path))):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            where = '.'.join(scope) or '<module>'
            if name in ('.to_parquet', '.write_table', 'ParquetWriter') or name in {a + '.' + b for a, b in WRITES}:
                key = (module, where, name)
                found[key] = EXPECTED.get(key, 'UNLISTED')
                if key not in EXPECTED:
                    problems.append(f'unlisted direct write {key}: route it through parquet_schema.metadata_target '
                                    'helpers, or add it here with the reason it is not a metadata target')
                elif EXPECTED[key] == SERIALIZER and where != 'pandas_parquet' and not _serializer_argument(ancestors):
                    problems.append(f'{key} is listed as a serializer but is not a serialize= argument of {sorted(HELPERS)}')
            # the choke-point writers run only inside `with metadata_target(...)` (or in the ensure merge it calls)
            if name in CHOKE_WRITERS and not _inside_metadata_target(ancestors) \
                    and where not in ('ParquetSchema._merge_metadata_locked',):
                problems.append(f'{module}:{node.lineno} {name} outside `with metadata_target(...)` ({where})')
            if name == '_merge_metadata_locked' and not _inside_metadata_target(ancestors):
                problems.append(f'{module}:{node.lineno} _merge_metadata_locked outside `with metadata_target(...)`')
            if name == 'publish_products' and any(keyword.arg == 'physical' for keyword in node.keywords) \
                    and not _inside_metadata_target(ancestors):
                problems.append(f'{module}:{node.lineno} metadata projection publication outside metadata_target')
            if name == '_parquet_lock' and module != 'glossapi/parquet_schema.py':
                problems.append(f'{module}:{node.lineno} uses the raw lock primitive; use metadata_target')
    stale = sorted(set(EXPECTED) - set(found))
    assert not problems, '\n'.join(problems)
    assert not stale, f'listed writers no longer present (update the list): {stale}'


def test_the_static_check_catches_a_bypass(tmp_path):
    """The check is not vacuous: a direct to_parquet in a metadata module, or a choke writer outside the lock, fails."""
    global SRC
    package = tmp_path / 'glossapi'
    package.mkdir()
    (package / 'bypass.py').write_text('def publish(frame, path):\n    frame.to_parquet(path, index=False)\n')
    (package / 'raw.py').write_text('from .parquet_schema import _write_serialized_locked\n'
                                    'def publish(frame, path):\n    _write_serialized_locked(frame, path, None)\n')
    saved, SRC = SRC, tmp_path
    try:
        with pytest.raises(AssertionError) as caught:
            test_no_metadata_writer_bypasses_the_choke_point()
    finally:
        SRC = saved
    assert 'unlisted direct write' in str(caught.value) and 'outside `with metadata_target' in str(caught.value)
