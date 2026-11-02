"""Child process for tests/test_k4g_fix1.py (ported from the CP-K4g reviewer's adversarial_child.py): one run or resume
of three generated PDFs with an injected persistent worker ENOSPC for the documents named in PROBE_BLOCKED; prints one
JSON line (state, census, export records, barriers, every disk check with its measured footprint and release credits)."""
import json
import os
from pathlib import Path
import sys

def wide_pdf(path,text):
    chunks=[text[i:i+20000] for i in range(0,len(text),20000)]
    kids=' '.join(str(4+2*i)+' 0 R' for i in range(len(chunks)))
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>',f'<< /Type /Pages /Kids [{kids}] /Count {len(chunks)} >>'.encode(),
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    for i,chunk in enumerate(chunks):
        stream=('BT /F1 12 Tf 20 100 Td ('+chunk+') Tj ET').encode('ascii')
        objects.extend([f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {10*len(chunk)+100} 200] /Resources << /Font << /F1 3 0 R >> >> /Contents {5+2*i} 0 R >>'.encode(),
            b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream'])
    payload=b'%PDF-1.4\n';offsets=[]
    for i,obj in enumerate(objects,1):
        offsets.append(len(payload));payload+=str(i).encode()+b' 0 obj\n'+obj+b'\nendobj\n'
    xref=len(payload)
    payload+=f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode()+b''.join(f'{v:010} 00000 n \n'.encode() for v in offsets)
    payload+=f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n'.encode()+str(xref).encode()+b'\n%%EOF\n'
    path.write_bytes(payload);return path

def main():
    import pytest
    from glossapi import Corpus
    from glossapi.ledger.writer import open_readonly
    from glossapi.stages import run as RUN, final_products as FP
    from test_k4fix2 import _extreme, _unique_bytes
    from test_ledger_fixset_lg import pdf
    root=Path(sys.argv[1]); root.mkdir(exist_ok=True,parents=True)
    mode=sys.argv[2]
    debug=os.environ.get('PROBE_DEBUG','off')
    patch=pytest.MonkeyPatch()
    checks=[]; measured=[]; calls=[]
    if os.environ.get('PROBE_EXTREME')=='1':
        checks,settings=_extreme(root,patch,lambda:1<<32)
        settings.update(debug=debug,max_source_bytes=1<<24,max_attempt_bytes=1<<26)
    else:
        settings={'debug':debug,'extract':{'repair_fonts':False}}
    real_credits=FP.release_credits
    def credits(corpus,result):
        value=real_credits(corpus,result)
        # What a credit computation that ignores retention's suspension would subtract (the CP-K4g F1 defect): the
        # bound a fixed tree must stay above.
        try:unsuspended=real_credits(corpus,result,honour_suspension=False)
        except TypeError:unsuspended=value
        calls.append(dict(credits=value,unsuspended=unsuspended,failed=RUN._has_failed_documents(corpus._scheduler_writer,result['run_id'])))
        return value
    FP.release_credits=credits
    real_need=FP.remaining_need
    def need(charges,credits):
        value=real_need(charges,credits)
        if calls:calls[-1].update(charges=dict(charges),need=value,unsuspended_need=real_need(charges,calls[-1]['unsuspended']))
        return value
    FP.remaining_need=need
    real=RUN.publish_barrier
    def barrier(corpus,run,stage,callback,**kw):
        before=_unique_bytes(root/'output')
        try:return real(corpus,run,stage,callback,**kw)
        finally:measured.append(dict(stage=stage,before=before,after=_unique_bytes(root/'output'),need=checks[-1]['need'] if checks else None))
    RUN.publish_barrier=barrier
    if os.environ.get('PROBE_NO_COVERAGE')=='1':
        RUN._publication_coverage=lambda corpus:None
    corpus=Corpus(root/'empty',root/'output')
    if mode=='start':
        n=int(os.environ.get('PROBE_PARAGRAPHS','0'))
        text=' '.join(f'Paragraph {i} has a different longer sentence about native cleaning and sections in document.' for i in range(n))
        sources=[(wide_pdf if n else pdf)(root/(name+'.pdf'),name+' original text '+text) for name in ('alpha','beta','gamma')]
        stages=('extract','clean','sections','annotate','export') if os.environ.get('PROBE_ANNOTATE')=='1' else ('extract','clean','export')
        if os.environ.get('PROBE_REMOTE')=='1':
            from test_k4fix1 import transport
            bodies=transport.__wrapped__(patch)
            urls=['http://fixture.invalid/'+s.name for s in sources]
            bodies.update({url:s.read_bytes() for url,s in zip(urls,sources)})
            sources=urls
        result=corpus.run(sources,stages=stages,settings=settings)
        (root/'runid').write_text(result['run_id'])
    else:
        if os.environ.get('PROBE_REMOTE')=='1':
            from test_k4fix1 import transport
            bodies=transport.__wrapped__(patch)
            bodies.update({'http://fixture.invalid/'+p.name:p.read_bytes() for p in root.glob('*.pdf')})
        result=corpus.run(resume=(root/'runid').read_text(),retry_failed=(mode!='plain'))
    with open_readonly(root/'output/corpus.sqlite3') as c:
        barriers=[list(row) for row in c.execute("SELECT t.operation,s.status,(SELECT COUNT(*) FROM attempts a WHERE a.task_id=t.task_id) FROM tasks t JOIN current_task_state s USING(task_id) WHERE t.operation LIKE '%.compatibility_publish'")]
        downloads=[dict(zip(('uri','presence'),row)) for row in c.execute("SELECT uri,presence FROM current_file_state WHERE kind='download_payload'")]
    path=root/'output/export.jsonl'
    records=[json.loads(line)['filename'] for line in path.read_text().splitlines()] if path.exists() else []
    data=dict(state=result['state'],counts=result.get('document_counts'),records=records,barriers=barriers,
              checks=checks,measured=measured,credits=calls,downloads=downloads,
              private_md=[str(p.relative_to(root)) for p in (root/'output/.glossapi/artifacts').rglob('*.md')],
              good=getattr(corpus,'good_files',None))
    (root/(mode+'.json')).write_text(json.dumps(data,indent=2))
    print(json.dumps(data))

if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(error_code=getattr(error,'code',None),error_type=type(error).__name__,message=str(error))))
        raise
