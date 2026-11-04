"""CP1 reproductions copied from review_scratch/CP1/test_cp1_extra.py, plus fixes.

The original overlapping-shard check now requires loud rejection at reassembly;
the dispatch check separately verifies that stale shards are removed before OCR.
"""

import json
import multiprocessing as mp
from pathlib import Path
import signal
import time
from pypdf import PdfWriter
import pytest

from pdf_fixtures import FixturePdf

from glossapi.page_quality import compress_page_ranges
from glossapi.scripts import page_quality_batch as cli

def make_pdf(path, n=2):
    d=FixturePdf()
    for i in range(n):
        d.insert_text(d.new_page(), (72,72), 'Native heading\nA B\n1 2' if i==0 else f'Old scanned appendix {i+1}')
    d.save(path)

def test_crafted_ranges():
    assert compress_page_ranges([22,3,4,5,6,7,8,9,14,20,21,3,1,99,100,102])=='1,3-9,14,20-22,99-100,102'

def test_unreadable_encrypted_corrupt_and_continue(tmp_path):
    good=tmp_path/'good.pdf';make_pdf(good)
    corrupt=tmp_path/'bad.pdf';corrupt.write_bytes(b'not a PDF')
    encrypted=tmp_path/'locked.pdf'
    w=PdfWriter(clone_from=good);w.encrypt(user_password='secret',owner_password='owner');w.write(encrypted)
    records=list(cli.batch([encrypted,corrupt,good],workers=1,timeout=5))
    print('ENCRYPTED/CORRUPT RECORDS',json.dumps(records))
    assert len(records)==3
    assert records[0].get('error') and records[1].get('error')
    assert not records[2].get('error')

def stall_or_run(path,wordlist,result_path,fold_validity=False,rule_profile="v4",
                 image_dominant_min_valid_word_rate=0.9,image_dominant_max_junk_share=0.1):
    """Stand-in for cli._worker: same seven arguments, forwarded unchanged to the real worker."""
    if Path(path).name=='stall.pdf':
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        Path(path).with_suffix('.started').write_text('started')
        time.sleep(10)
        Path(path).with_suffix('.late').write_text('incorrect late completion')
    else:
        cli._worker(path,wordlist,result_path,fold_validity,rule_profile,
                    image_dominant_min_valid_word_rate,image_dominant_max_junk_share)

def test_actual_stall_killed_and_next_file_continues(tmp_path,monkeypatch):
    original_context=mp.get_context('fork')
    class Context:
        def Process(self,**kwargs):
            return original_context.Process(target=stall_or_run,args=kwargs['args'])
    monkeypatch.setattr(cli.mp,'get_context',lambda _:Context())
    good=tmp_path/'good.pdf';make_pdf(good)
    stall=tmp_path/'stall.pdf'
    # Load the bundled v4 wordlist in the parent so the forked good-file worker does not spend
    # the 0.2 s budget on a cold load (earlier tests warm it when the whole file runs).
    from glossapi import page_quality
    page_quality._default_words()
    previous={p.pid for p in mp.active_children()}
    started=time.monotonic();records=list(cli.batch([stall,good],workers=1,timeout=.2));elapsed=time.monotonic()-started
    print('STALL RECORDS',json.dumps(records),'elapsed',elapsed)
    assert stall.with_suffix('.started').exists()
    assert records[0]['error']=='Timeout after 0.2 seconds'
    assert not records[1].get('error')
    assert records[1]['filename']=='good.pdf' and len(records[1]['pages'])==2
    assert elapsed<1.8
    assert not stall.with_suffix('.late').exists()
    assert {p.pid for p in mp.active_children()}==previous

def test_partial_ocr_preserves_existing_unselected_markdown(tmp_path):
    from glossapi.ocr import page_artifacts as runner
    from glossapi.ocr.page_artifacts import _write_outputs
    source=tmp_path/'source.pdf';make_pdf(source)
    out=tmp_path/'out'
    original='# Native heading\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nOld scanned appendix'
    _write_outputs(output_dir=out,stem='source',markdown=original,page_count=2)
    _write_outputs(output_dir=out,stem='source__p00002-00002',markdown='New OCR appendix',page_count=1,
                   extra_metrics={'source_start_page':2,'source_end_page':2})
    runner._ensure_canonical_outputs(out_root=out,pdf_root=tmp_path,file_list=['source.pdf'],page_ranges=['source.pdf:2:2'])
    actual=(out/'markdown/source.md').read_text()
    print('BEFORE MARKDOWN',repr(original),'AFTER MARKDOWN',repr(actual))
    assert '| A | B |' in actual and '# Native heading' in actual, 'Partial OCR erased the kept page heading/table markdown'

def test_partial_reassembly_rejects_overlapping_stale_shards(tmp_path):
    from glossapi.ocr import page_artifacts as runner
    from glossapi.ocr.page_artifacts import _write_outputs, _join_page_outputs, _split_page_outputs
    source=tmp_path/'source.pdf';make_pdf(source,3)
    out=tmp_path/'out'
    _write_outputs(output_dir=out,stem='source',markdown=_join_page_outputs(['kept','old 2','old 3']),page_count=3)
    # An old interrupted run used a different shard size; current run writes two one-page shards.
    _write_outputs(output_dir=out,stem='source__p00002-00003',markdown=_join_page_outputs(['STALE PAGE 2','STALE PAGE 3']),page_count=2,
                   extra_metrics={'source_start_page':2,'source_end_page':3})
    _write_outputs(output_dir=out,stem='source__p00002-00002',markdown='FRESH PAGE 2',page_count=1,
                   extra_metrics={'source_start_page':2,'source_end_page':2})
    _write_outputs(output_dir=out,stem='source__p00003-00003',markdown='FRESH PAGE 3',page_count=1,
                   extra_metrics={'source_start_page':3,'source_end_page':3})
    original = (out/'markdown/source.md').read_text()
    with pytest.raises(RuntimeError, match='Overlapping OCR shards'):
        runner._ensure_canonical_outputs(out_root=out,pdf_root=tmp_path,file_list=['source.pdf'],page_ranges=['source.pdf:2:3'])
    actual=_split_page_outputs((out/'markdown/source.md').read_text())
    assert actual == ['kept', 'old 2', 'old 3']
    assert (out/'markdown/source.md').read_text() == original


@pytest.mark.parametrize('json_mode', ['missing', 'plain', 'zstd', 'unmatchable', 'extra_text', 'extra_cell', 'crossing_page', 'corrupt_zstd'])
def test_docling_heading_table_and_pages_2_3_survive(tmp_path, json_mode):
    from glossapi.ocr import page_artifacts as runner
    from glossapi.ocr.page_artifacts import _write_outputs, _join_page_outputs, _split_page_outputs
    import zstandard

    make_pdf(tmp_path / 'source.pdf', 3)
    out = tmp_path / 'out'
    original = '# Native heading\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nOld page 2\n\nOld page 3'
    _write_outputs(output_dir=out, stem='source', markdown=original, page_count=3)
    _write_outputs(output_dir=out, stem='source__p00002-00003',
                   markdown=_join_page_outputs(['FRESH PAGE 2', 'FRESH PAGE 3']), page_count=2,
                   extra_metrics={'source_start_page': 2, 'source_end_page': 3})
    document = {
        'body': {'children': [{'$ref': '#/groups/0'}, {'$ref': '#/texts/1'}, {'$ref': '#/texts/2'}]},
        'groups': [{'children': [{'$ref': '#/texts/0'}, {'$ref': '#/tables/0'}]}],
        'texts': [{'text': 'Native heading', 'prov': [{'page_no': 1}]},
                  {'text': 'Old page 2', 'prov': [{'page_no': 2}]},
                  {'text': 'Old page 3', 'prov': [{'page_no': 3}]}],
        'tables': [{'prov': [{'page_no': 1}], 'data': {'table_cells': [
            {'text': cell} for cell in ['A', 'B', '1', '2']
        ]}}],
    }
    if json_mode == 'unmatchable':
        document['texts'][0]['text'] = 'A different document heading'
    if json_mode == 'extra_text':
        document['texts'][0]['text'] = 'Native'
    if json_mode == 'extra_cell':
        document['tables'][0]['data']['table_cells'].pop()
    if json_mode == 'crossing_page':
        document['tables'][0]['prov'].append({'page_no': 2})
    if json_mode != 'missing':
        payload = json.dumps(document).encode()
        compressed = json_mode in {'zstd', 'corrupt_zstd'}
        if compressed:
            payload = zstandard.ZstdCompressor().compress(payload) if json_mode == 'zstd' else b'not zstd'
        (out / ('json/source.docling.json.zst' if compressed else 'json/source.docling.json')).write_bytes(payload)

    runner._ensure_canonical_outputs(out_root=out, pdf_root=tmp_path, file_list=['source.pdf'],
                                     page_ranges=['source.pdf:2:3'])
    actual = (out / 'markdown/source.md').read_text()
    metrics = json.loads((out / 'json/metrics/source.metrics.json').read_text())
    assert '# Native heading' in actual and '| A | B |\n|---|---|\n| 1 | 2 |' in actual
    assert 'FRESH PAGE 2' in actual and 'FRESH PAGE 3' in actual
    if json_mode in {'plain', 'zstd'}:
        assert metrics['kept_markdown_source'] == 'docling_json_provenance'
        assert actual.endswith('FRESH PAGE 2\n\nFRESH PAGE 3\n')
        assert '<!-- page:' not in actual
        assert 'Old page' not in actual
    else:
        assert metrics['kept_markdown_source'] == 'existing_markdown_whole_plus_ocr'
        assert actual.startswith(original + '\n')
        assert '<!-- ocr-pages: 2-3 -->' in actual


def test_comment_only_page_boundaries_keep_heading_and_table(tmp_path):
    from glossapi.ocr import page_artifacts as runner
    from glossapi.ocr.page_artifacts import _write_outputs, _split_page_outputs

    make_pdf(tmp_path / 'source.pdf', 3)
    _write_outputs(output_dir=tmp_path, stem='source', page_count=3,
                   markdown='# Keep\n\n| A | B |\n<!-- page:2 -->\nold 2\n<!-- page:3 -->\nold 3')
    _write_outputs(output_dir=tmp_path, stem='source__p00002-00002', page_count=1, markdown='FRESH PAGE 2',
                   extra_metrics={'source_start_page': 2, 'source_end_page': 2})
    runner._ensure_canonical_outputs(out_root=tmp_path, pdf_root=tmp_path, file_list=['source.pdf'],
                                     page_ranges=['source.pdf:2:2'])
    actual = (tmp_path / 'markdown/source.md').read_text()
    assert actual == '# Keep\n\n| A | B |\n\nFRESH PAGE 2\n\nold 3\n'
    assert json.loads((tmp_path / 'json/metrics/source.metrics.json').read_text())['kept_markdown_source'] == 'page_markers'
