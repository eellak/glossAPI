"""Page replacement, language-aware validity and bounded OCR attempt budgets."""
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from glossapi.ocr import page_artifacts as artifacts
from glossapi.ocr.page_replacement import native_pages, text_hash
from glossapi.markdown_layout import markup_free
from glossapi.markdown_layout.render import strip_record
from glossapi.page_quality import page_quality, replay_page_selection
from pdf_fixtures import FixturePdf
from test_markdown_layout import _extract


def _source(tmp_path, monkeypatch, mode):
    pdf = tmp_path/'source.pdf'
    doc = FixturePdf()
    for text in ('Native first page\nwith paragraphs', '', 'Old overlapping page three'):
        page = doc.new_page()
        if text:
            doc.insert_text(page, (72,72), text)
    doc.save(pdf)
    markdown, sidecar = _extract(pdf, tmp_path/'out', monkeypatch, mode)
    return pdf, tmp_path/'out', markdown, sidecar


def _replace(root, pdf, pages):
    for page, text in pages.items():
        artifacts._write_outputs(output_dir=root, stem=f'source__p{page:05}-{page:05}',
            markdown=text, page_count=1, extra_metrics={'source_start_page':page,'source_end_page':page})
    artifacts._ensure_canonical_outputs(out_root=root,pdf_root=pdf.parent,file_list=[pdf.name],
        page_ranges=[f'{pdf.name}:{page}:{page}' for page in pages])
    return (root/'markdown/source.md').read_text()


@pytest.mark.parametrize('mode', ['layout_v2','plain'])
def test_partial_pages_replace_in_place_with_bound_metric_strip(tmp_path, monkeypatch, mode):
    pdf, root, before, sidecar = _source(tmp_path, monkeypatch, mode)
    after = _replace(root, pdf, {2:'OCR second page',3:'OCR page three'})
    assert 'Old overlapping' not in after
    assert after.count('OCR page three') == 1
    assert after.index('Native first') < after.index('OCR second') < after.index('OCR page three')
    assert '<!-- page:' not in after and '<--- Page Split --->' not in after
    layout = json.loads((root/'sidecars/ocr_layout/source.json').read_text())['markdown_layout']
    plain = markup_free(after, layout)
    assert plain is not None and 'OCR page three' in plain and 'Native first' in plain
    assert layout['strip']['markdown_sha256'] == text_hash(after)
    assert json.loads((root/'json/metrics/source.metrics.json').read_text())['kept_markdown_source']=='page_offsets'
    # Restart-style replay from the immutable native snapshot and durable page state.
    assert _replace(root, pdf, {3:'OCR page three'}) == after
    # Changing another page keeps prior replacements and their physical position.
    final = _replace(root, pdf, {2:'Updated OCR second page'})
    assert final.count('OCR page three') == 1 and 'Old overlapping' not in final
    assert (root/'sidecars/ocr_native/source.md').read_text() == before


@pytest.mark.parametrize('damage,reason', [('stale','page_offsets_hash_mismatch'),
    ('missing','page_offsets_missing'),('overlap','page_offsets_unusable'),('foreign_strip','page_offsets_unusable'),('invalid_sidecar','page_offsets_unusable')])
def test_offsets_refuse_stale_or_unusable_records_and_append(tmp_path, monkeypatch, damage, reason):
    pdf, root, before, sidecar = _source(tmp_path, monkeypatch, 'layout_v2')
    if damage == 'stale':
        sidecar['markdown_layout']['strip']['markdown_sha256']='0'*64
    elif damage == 'missing':
        sidecar.pop('markdown_layout')
    elif damage == 'overlap':
        sidecar['markdown_layout']['pages'][1][1]=0
    elif damage == 'foreign_strip':
        sidecar['markdown_layout']['strip']['plain_sha256']='0'*64
    else:
        sidecar=[]
    (root/'sidecars/extract/source.json').write_text(json.dumps(sidecar))
    after = _replace(root, pdf, {3:'Replacement'})
    assert after.startswith(before) and '<!-- ocr-pages: 3 -->' in after
    metric=json.loads((root/'json/metrics/source.metrics.json').read_text())
    assert metric['page_replacement_fallback_reason']==reason


def test_page_offsets_never_guess_from_paragraph_breaks(tmp_path):
    text='Page one\n\nparagraph two\n\nPage three'
    root=tmp_path;path=root/'sidecars/extract/source.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'plain_pages':{'pages':[[1,0,23],[3,25,len(text)]],
        'markdown_chars':len(text),'markdown_sha256':text_hash(text)}}))
    pages,_,reason=native_pages(root,'source',text,3)
    assert pages==['Page one\n\nparagraph two','','Page three'] and reason is None


@pytest.mark.parametrize('text,expected', [
    ('xqzvkj pxxdqn zrjkwv qzvpxn lkqzwv jxxvqr',True),
    ('I am in it as we go to do so he is on an at by or if no up us',False),
    ('The structure of the human body is described in this chapter of the book',False),
])
def test_v5_latin_garbage_and_real_short_words(tmp_path,text,expected):
    path=tmp_path/'layer.pdf';doc=FixturePdf();page=doc.new_page()
    doc.insert_image(page,(0,0,595,842),(20,20,(200,200,200)))
    for i in range(18):
        doc.insert_text(page,(25,35+30*i),text,fontsize=8)
    doc.save(path)
    old=page_quality(path);new=page_quality(path,rule_profile='v5')
    row=new.pages[0]
    assert not old.pages[0].page_needs_ocr
    assert row.page_needs_ocr is expected
    assert replay_page_selection(asdict(row)) is expected
    assert row.valid_latin_word_rate is not None
    assert new.latin_dictionary_identity['license']=='MIT AND BSD'
    assert new.latin_dictionary_identity['sha256']
    assert page_quality(path).rule_profile=='v4'  # qualification is explicit


def test_v5_keeps_clean_non_image_latin_layers(tmp_path,monkeypatch):
    pdf,_,_,_=_source(tmp_path,monkeypatch,'plain')
    v4=page_quality(pdf);v5=page_quality(pdf,rule_profile='v5')
    assert [p.page_needs_ocr for p in v4.pages]==[p.page_needs_ocr for p in v5.pages]


def test_english_wordlist_integrity_and_short_words(monkeypatch):
    import importlib
    pq=importlib.import_module('glossapi.page_quality')
    assert {'i','a','in','it','as','go','to'} <= pq._latin_words()
    pq._latin_words.cache_clear()
    monkeypatch.setattr(pq.gzip,'decompress',lambda _:b'1\ncorrupt\n')
    with pytest.raises(ValueError,match='SHA256'):
        pq._latin_words()
    pq._latin_words.cache_clear()


def test_ocr_deadline_scales_selected_pages_and_respects_explicit_cap(tmp_path):
    from glossapi.ocr.deadline import attempt_budget
    from glossapi.stages.execution import sha256_file
    quality=tmp_path/'quality.json'
    quality.write_text(json.dumps({'pages':[{'page':i,'page_needs_ocr':i<=10} for i in range(1,21)]}))
    ref=SimpleNamespace(kind='page_quality',path=quality,sha256=sha256_file(quality))
    seconds,receipt=attempt_budget([ref],{})
    assert seconds==600 and receipt['pages']==10
    assert attempt_budget([ref],{'max_pages':2})[0]==600
    assert attempt_budget([ref],{'timeout_seconds':.05})[0]==.05
    assert attempt_budget([ref],{},.03)[0]==.03
    assert attempt_budget([],{'page_ranges':[[1,23]]})[0]==1155
    assert attempt_budget([],{'page_ranges':[[1,1000]]})[0]==7200
    quality.write_text('{}')  # stale sidecar cannot inflate the budget
    assert attempt_budget([ref],{})[0]==600
    for value in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):attempt_budget([],{'timeout_seconds':value})


def test_replacement_strip_retains_native_heading_edits_and_ocr_literal_text(tmp_path):
    from glossapi.ocr.page_replacement import replacement_layout
    pages=['# Native title\nBody','OCR # literal heading','Old page']
    markdown,layout=replacement_layout(pages,[[[0,2,'']],[],[]],{2})
    assert markdown=='# Native title\nBody\n\nOCR # literal heading\n\nOld page\n'
    assert markup_free(markdown,layout)=='Native title\nBody\n\nOCR # literal heading\n\nOld page\n'
    assert markup_free(markdown+'changed',layout).endswith('changed')
    assert markup_free('changed'+markdown,layout) is None


def test_v5_routes_bad_non_image_latin_and_records_replay(tmp_path):
    doc=FixturePdf();page=doc.new_page()
    for i in range(12):
        doc.insert_text(page,(20,30+30*i),'xqzvkj pxxdqn zrjkwv qzvpxn lkqzwv jxxvqr',fontsize=9)
    pdf=tmp_path/'bad-native.pdf';doc.save(pdf)
    assert not page_quality(pdf).pages[0].page_needs_ocr
    row=page_quality(pdf,rule_profile='v5').pages[0]
    assert row.page_needs_ocr and row.latin_validity_applies
    assert row.valid_latin_word_rate==0 and replay_page_selection(row)


def test_clean_stage_accepts_ocr_produced_strip_as_its_bound_input(tmp_path):
    from dataclasses import replace
    from glossapi.stages.clean import CleanStage
    from glossapi.stages.contracts import ArtifactReference
    from glossapi.stages.execution import ReceiptCollector, sha256_file
    from test_ledger_stages import request
    text='# A readable heading\n\nThe text of a retained native page.'
    raw=tmp_path/'source.md';raw.write_text(text)
    sidecar=tmp_path/'source.json'
    sidecar.write_text(json.dumps({'markdown_layout':{'source':'ocr_page_replacement',
        'strip':strip_record(text,[[0,2,'']],text[2:])}}))
    refs=tuple(ArtifactReference(str(i),kind,path,sha256_file(path),path.stat().st_size)
        for i,(kind,path) in enumerate([('ocr_markdown',raw),('ocr_layout',sidecar)]))
    given=replace(request(tmp_path),operation='clean.initial.artifact:after_ocr',inputs=refs)
    result=CleanStage().run_document(given,reporter=ReceiptCollector())
    assert result.status=='ok',result.error
    assert result.decisions[0]['measurements']['metric_surface']['basis']=='markup_free_text'


def test_markerless_ocr_product_never_becomes_a_missing_native_snapshot(tmp_path,monkeypatch):
    pdf,root,_,_=_source(tmp_path,monkeypatch,'layout_v2')
    canonical=_replace(root,pdf,{3:'OCR page three'})
    assert '<!-- ocr-pages:' not in canonical
    (root/'sidecars/ocr_native/source.md').unlink()
    with pytest.raises(FileNotFoundError,match='retained native Markdown'):
        _replace(root,pdf,{2:'OCR second page'})
    assert (root/'markdown/source.md').read_text()==canonical
    assert not (root/'sidecars/ocr_native/source.md').exists()
