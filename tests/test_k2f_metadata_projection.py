"""W18 S1 / K4BI-G1: metadata merges are an exact union (K4(b) round 1).

Columns of an existing results table that glossapi does not know (the caller's own, the HTTP downloader's
selection_id, resource_identity, base_domain) are kept with their types; an int64 above 2**53 survives a merge that adds
rows the table does not have. Declared export change: HTTP export records now carry those columns.
"""
from pathlib import Path

import pandas as pd

BIG = 2**53 + 1


def existing(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'url': ['http://x/a.pdf', 'http://x/b.pdf'], 'filename': ['a.pdf', 'b.pdf'],
                  'big': pd.Series([BIG, 7], dtype='int64'), 'selection_id': ['A', 'B'],
                  'category': pd.Categorical(['u', 'v'])}).to_parquet(path, index=False)


def test_ensure_metadata_parquet_keeps_every_existing_column_exactly(tmp_path):
    from glossapi.parquet_schema import ParquetSchema
    existing(tmp_path/'download_results/download_results_x.parquet')
    (tmp_path/'markdown').mkdir()
    (tmp_path/'markdown/a.md').write_text('# a\n\nbody'); (tmp_path/'markdown/c.md').write_text('# c\n\nbody')
    path = ParquetSchema({'url_column': 'url'}).ensure_metadata_parquet(tmp_path)
    frame = pd.read_parquet(path).set_index('filename')
    assert int(frame.loc['a.pdf', 'big']) == BIG and int(frame.loc['b.pdf', 'big']) == 7 and pd.isna(frame.loc['c.pdf', 'big'])
    assert frame.loc['a.pdf', 'selection_id'] == 'A' and frame.loc['b.pdf', 'selection_id'] == 'B'
    assert str(frame.loc['b.pdf', 'category']) == 'v'


def test_triage_row_append_keeps_existing_integers_exact(tmp_path):
    from glossapi.ocr.utils.triage import update_download_results_parquet
    existing(tmp_path/'download_results/download_results.parquet')
    update_download_results_parquet(tmp_path, 'new_doc', {'formula_total': 3}, 'stop')
    frame = pd.read_parquet(tmp_path/'download_results/download_results.parquet').set_index('filename')
    assert int(frame.loc['a.pdf', 'big']) == BIG
    assert 'new_doc.pdf' in frame.index and frame.loc['b.pdf', 'selection_id'] == 'B'
