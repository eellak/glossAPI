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
    write_json(root/'run.json',dict(complete=True,profile='text-layer',manifest_sha256='b'*64,manifest=[dict(filename='sample.pdf',sha256='a'*64)],documents=[doc]))
    return root


def test_compare_exact_and_normalized_and_bound_declarations(tmp_path):
    a,b=receipt(tmp_path,'a'),receipt(tmp_path,'b','one \nthree\n')
    result=compare(a,b)
    assert not result['passes'] and result['undeclared_differences']==5
    diff=result['results'][0]['differences'][0]
    assert diff['markdown']['lines_changed']=={'removed':1,'added':1}
    assert diff['markdown']['words_changed']=={'removed':1,'added':1}
    # Product changes cannot be waived by a free-text reason.
    declared=[dict(d,document=r['document'],stage=r['stage'],reason='runtime timing only') for r in result['results'] for d in r['differences']]
    with pytest.raises(ValueError, match='timing leaf'):
        compare(a,b,declared)



def test_fail_closed_missing_document_incomplete_and_status(tmp_path):
    a,b=receipt(tmp_path,'a'),receipt(tmp_path,'b')
    assert compare(a,b)['identical']
    value=json.loads((b/'run.json').read_text());value['complete']=False;value['documents']=[];write_json(b/'run.json',value)
    r=compare(a,b)
    assert not r['passes'] and any('coverage' in e for e in r['integrity_errors'])


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


@pytest.mark.parametrize('mutation', ['markdown-byte', 'deleted-directory', 'export-reorder'])
def test_stale_receipts_never_hide_physical_product_mutations(tmp_path, mutation):
    import shutil
    a, b = receipt(tmp_path, 'a'), receipt(tmp_path, 'b')
    for root in (a, b):
        value=json.loads((root/'run.json').read_text())
        docdir=root/'documents/sample'; docdir.mkdir(parents=True)
        f=docdir/('export.jsonl' if mutation=='export-reorder' else 'sample.md')
        f.write_text('{"id":1}\n{"id":2}\n' if mutation=='export-reorder' else 'one\ntwo\n')
        for state in value['documents'][0]['stages'].values():
            state['files']=[dict(path=f.name,stored_path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha256(f))]
        write_json(root/'run.json',value)
    assert compare(a,b)['passes']
    if mutation=='deleted-directory': shutil.rmtree(b/'documents')
    elif mutation=='export-reorder': (b/'documents/sample/export.jsonl').write_text('{"id":2}\n{"id":1}\n')
    else: (b/'documents/sample/sample.md').write_text('One\ntwo\n')
    result=compare(a,b)
    assert not result['passes'] and not result['identical'] and result['integrity_errors']


def metric_pair(tmp_path):
    a,b=receipt(tmp_path,'a'),receipt(tmp_path,'b')
    for root,seconds in ((a,1.0),(b,2.0)):
        f=root/'sample.metrics.json';write_json(f,dict(timings={'layout':{'total_sec':seconds}},product='same'))
        value=json.loads((root/'run.json').read_text())
        for state in value['documents'][0]['stages'].values():state['files']=[]
        value['documents'][0]['stages']['extract']['files']=[dict(path=f.name,stored_path=f.name,bytes=f.stat().st_size,sha256=sha256(f))]
        write_json(root/'run.json',value)
    diff=compare(a,b)['results'][0]['differences'][0]
    declaration=dict(document='sample.pdf',stage='extract',path=diff['path'],a_sha256=diff['a_sha256'],b_sha256=diff['b_sha256'],fields=['/timings/layout/total_sec'],reason='runtime timing only')
    return a,b,declaration


@pytest.mark.parametrize('rehash', [False, True])
def test_product_mutation_under_existing_timing_declaration_fails(tmp_path,rehash):
    a,b,d=metric_pair(tmp_path)
    assert compare(a,b,[d])['passes']
    f=b/'sample.metrics.json';value=json.loads(f.read_text());value['review_injected_product']='changed';write_json(f,value)
    if rehash:
        run=json.loads((b/'run.json').read_text());r=run['documents'][0]['stages']['extract']['files'][0]
        r.update(sha256=sha256(f),bytes=f.stat().st_size);write_json(b/'run.json',run)
        # Even a fresh hash pair may waive only the explicitly named timing leaves.
        d['b_sha256']=r['sha256']
    result=compare(a,b,[d])
    assert not result['passes']
    if rehash: assert result['undeclared_differences']==1 and result['unused_declarations']
    else: assert result['integrity_errors']


@pytest.mark.parametrize('failed', ['all', 'determinism', 'pages-determinism', None])
def test_campaign_exit_propagates_required_failures_without_subprocesses(tmp_path,monkeypatch,failed):
    from tools.harness import baseline
    def execute(command,root,name,timeout=None):
        if 'tools.harness.golden' in command:
            write_json(Path(command[command.index('--out')+1])/'run.json',{})
        return dict(exit_code=int(failed=='all' or name=={'determinism':'text-determinism','pages-determinism':'docling-pages-determinism'}.get(failed,failed)),timed_out=False,wall_seconds=1)
    monkeypatch.setattr(baseline,'execute',execute)
    monkeypatch.setattr(baseline,'audit_page_timings',lambda a,b,out:out/'declared.json')
    monkeypatch.setattr(baseline.os,'nice',lambda n:None)
    def forbidden(*a,**kw):raise AssertionError('must not spawn')
    monkeypatch.setattr(baseline.subprocess,'Popen',forbidden)
    monkeypatch.setattr(sys,'argv',['baseline','--out',str(tmp_path/'campaign'),'--data','unused','--cohort','unused','--docling-python',sys.executable,'--gating-only'])
    assert baseline.main()==(0 if failed is None else 1)
    c=json.loads((tmp_path/'campaign/campaign.json').read_text())
    assert c['complete'] and c['text_complete'] and c['gates_passed']==(failed is None)


def test_mac_rss_ps_sampler_and_permission_denial(monkeypatch):
    from tools.harness import common
    monkeypatch.setattr(common.sys,'platform','darwin')
    pid=common.os.getpid()
    monkeypatch.setattr(common.subprocess,'check_output',lambda *a,**kw:f'{pid} 1 1000\n987654 {pid} 200\n')
    with common.PeakRSS() as rss:
        assert rss.available and rss.own_peak==1024000 and rss.peak==1228800
    def denied(*a,**kw):raise PermissionError('sandbox')
    monkeypatch.setattr(common.subprocess,'check_output',denied)
    with common.PeakRSS() as rss:
        assert not rss.available and rss.peak==0 and 'PermissionError' in rss.unavailable_reason
