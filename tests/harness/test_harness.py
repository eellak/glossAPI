"""Tiny CPU-only harness contract tests; no models or network."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from tools.harness.common import STAGES, load_manifest, sha256, write_json
from tools.harness.compare import compare, markdown_diff
from tools.harness.speed import union_seconds


def receipt(tmp_path, name, text='one\ntwo\n'):
    root = tmp_path/name; root.mkdir()
    f = root/'sample.md'; f.write_text(text)
    doc = dict(filename='sample.pdf',input_sha256='a'*64,stages={stage: dict(status='ok',error=None,files=[dict(path='sample.md',stored_path='sample.md',sha256=sha256(f),bytes=f.stat().st_size)]) for stage in STAGES})
    write_json(root/'run.json',dict(complete=True,profile='text-layer',manifest_sha256='b'*64,documents=[doc]))
    return root


def test_compare_exact_and_normalized_and_bound_declarations(tmp_path):
    a,b=receipt(tmp_path,'a'),receipt(tmp_path,'b','one \nthree\n')
    result=compare(a,b)
    assert not result['passes'] and result['undeclared_differences']==5
    diff=result['results'][0]['differences'][0]
    assert diff['markdown']['lines_changed']=={'removed':1,'added':1}
    assert diff['markdown']['words_changed']=={'removed':1,'added':1}
    declared=[dict(d,document=r['document'],stage=r['stage'],reason='test replacement') for r in result['results'] for d in r['differences']]
    assert compare(a,b,declared)['passes']
    assert not compare(a,b,declared)['identical']
    receipt_path=b/'run.json'; data=json.loads(receipt_path.read_text());data['documents'][0]['stages']['extract']['files'][0]['sha256']='c'*64;write_json(receipt_path,data)
    assert not compare(a,b,declared)['passes']


def test_fail_closed_missing_document_incomplete_and_status(tmp_path):
    a,b=receipt(tmp_path,'a'),receipt(tmp_path,'b')
    assert compare(a,b)['identical']
    value=json.loads((b/'run.json').read_text());value['complete']=False;value['documents']=[];write_json(b/'run.json',value)
    r=compare(a,b)
    assert not r['passes'] and r['issues']==['incomplete run'] and r['undeclared_differences']==5


def test_normalization_does_not_discard_words():
    assert markdown_diff('a  b\r\n\n','a b\n')['normalized_identical']
    assert not markdown_diff('a b','a c')['normalized_identical']


def test_manifest_and_interval_validation(tmp_path):
    p=tmp_path/'m.json';write_json(p,[dict(filename='../bad.pdf',sha256='a'*64)])
    with pytest.raises(ValueError):load_manifest(p)
    write_json(p,[dict(filename='a.pdf',sha256='a'*64)]*2)
    with pytest.raises(ValueError):load_manifest(p)
    assert union_seconds([(1,4),(2,3),(6,8),(7,9)])==6


def test_golden_fresh_root_hash_failure_is_durable(tmp_path):
    from tools.harness import golden
    golden.IMPORTS={'pid':1}
    data=tmp_path/'data';data.mkdir();(data/'tiny.pdf').write_bytes(b'not a PDF')
    row=dict(filename='tiny.pdf',sha256='a'*64)
    result=golden.document_job(row,data,tmp_path/'out','text-layer')
    assert all(s['status']=='failed' and 'sha256 mismatch' in s['error'] for s in result['stages'].values())
    assert json.loads((tmp_path/'out/documents/tiny/document.json').read_text())==result
    with pytest.raises(FileExistsError):golden.document_job(row,data,tmp_path/'out','text-layer')


def test_golden_calls_public_api_preserves_overwritten_products(tmp_path,monkeypatch):
    from tools.harness import golden
    class PDF:
        def __init__(self,p):pass
        def __len__(self):return 4
        def close(self):pass
    monkeypatch.setitem(sys.modules,'pypdfium2',SimpleNamespace(PdfDocument=PDF))
    calls=[]
    class FakeCorpus:
        def __init__(self,i,o,**kw):
            self.output_dir=o;self.sections_parquet=o/'sections.parquet';self.fully_annotated_parquet=o/'annotated.parquet'
        def extract(self,**kw):
            calls.append(('extract',kw));p=self.output_dir/'markdown/a.md';p.parent.mkdir();p.write_text('raw')
        def clean(self,**kw):
            calls.append(('clean',kw));p=self.output_dir/'clean_markdown/a.md';p.parent.mkdir();p.write_text('clean')
            (self.output_dir/'markdown/a.md').write_text('modified')
        def section(self):calls.append(('sections',{}));self.sections_parquet.write_bytes(b'sections')
        def annotate(self):calls.append(('annotate',{}));self.fully_annotated_parquet.write_bytes(b'annotated')
        def jsonl(self,p):calls.append(('export',{}));p.write_text('{}\n')
    class FakeRSS:
        peak=own_peak=1024
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(golden,'PeakRSS',FakeRSS)
    monkeypatch.setattr(golden,'Corpus',FakeCorpus);monkeypatch.setattr(golden,'IMPORTS',{'pid':1})
    data=tmp_path/'data';data.mkdir();pdf=data/'a.pdf';pdf.write_bytes(b'tiny')
    out=tmp_path/'out';r=golden.document_job(dict(filename='a.pdf',sha256=sha256(pdf),page_range=[2,4]),data,out,'docling-cpu-pages')
    assert [s for s,_ in calls]==list(STAGES)
    assert calls[0][1]['page_range']==(2,4) and calls[0][1]['phase1_backend']=='docling'
    assert r['pages']==3 and all(s['status']=='ok' for s in r['stages'].values())
    assert (out/'documents/a/snapshots/extract/markdown/a.md').read_text()=='raw'
    assert (out/'documents/a/snapshots/clean/markdown/a.md').read_text()=='modified'


def test_direct_cli_entrypoints_work_without_pipeline_imports():
    import subprocess
    root=Path(__file__).resolve().parents[2]
    for name in ('golden','compare','speed','freeze','ship','baseline'):
        result=subprocess.run([sys.executable,str(root/f'tools/harness/{name}.py'),'--help'],capture_output=True,text=True)
        assert result.returncode==0,result.stderr


@pytest.mark.skipif(sys.platform!='linux',reason='measurement node uses Linux /proc')
def test_rss_samples_process_memory_without_extra_packages():
    from tools.harness.common import PeakRSS
    with PeakRSS() as rss:
        payload=bytearray(4*1024*1024)
        rss.sample()
        assert len(payload)>0 and rss.peak>=rss.own_peak>4*1024*1024
