"""Detached node campaign. Text gates finish before bounded lower-priority Docling."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ''):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'tools.harness'
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import REPO, load_manifest, write_json


def execute(command, root, name, timeout=None):
    began=time.monotonic()
    with (root/f'{name}.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timed_out=False
        try: code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=process.returncode
    row=dict(command=command,exit_code=code,timed_out=timed_out,wall_seconds=time.monotonic()-began)
    write_json(root/f'{name}.exit.json',row)
    return row


def audit_page_timings(left, right, output):
    """Generate pair-bound declarations only after auditing every changed JSON leaf."""
    from .compare import compare
    strict = compare(left, right)
    write_json(output/'docling-pages-strict.json', strict)
    declarations = []
    for row in strict['results']:
        for diff in row['differences']:
            fields = diff.get('json_fields', [])
            if fields and fields == diff.get('timing_fields'):
                declarations.append(dict(document=row['document'], stage=row['stage'],
                    path=diff['path'], a_sha256=diff['a_sha256'], b_sha256=diff['b_sha256'],
                    fields=fields, reason='Audited numeric runtime timing leaves only'))
    write_json(output/'docling-pages-declared-timings.json', declarations)
    return output/'docling-pages-declared-timings.json'


def gate_status(campaign):
    required = ('subset30-text-1', 'subset30-text-2', 'cohort242-text', 'determinism', 'speed',
                'subset30-docling-pages-1', 'subset30-docling-pages-2', 'pages-determinism')
    campaign['gates_passed'] = all(campaign['gating'].get(k, {}).get('exit_code') == 0
                                   and not campaign['gating'][k].get('timed_out', False) for k in required)
    return 0 if campaign['gates_passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--data',required=True);p.add_argument('--cohort',required=True)
    p.add_argument('--docling-python',required=True);p.add_argument('--docling-only',action='store_true')
    p.add_argument('--gating-only',action='store_true',help='finish after required text/speed/fixed-page gates')
    p.add_argument('--pilot-limit-seconds',type=int,default=600)
    a=p.parse_args();root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=a.docling_only)
    subset=REPO/'tools/harness/manifests/subset30.json'
    campaign={'schema':1,'gating':{},'docling':{},'complete':False}
    if (root/'campaign.json').exists():campaign=json.loads((root/'campaign.json').read_text())
    def save():write_json(root/'campaign.json',campaign)
    def golden(python,manifest,name,profile,timeout=None):
        return execute([python,'-m','tools.harness.golden',str(manifest),'--data',a.data,'--out',str(root/name),'--profile',profile],root,name,timeout)
    if not a.docling_only:
        for name,manifest in [('subset30-text-1',subset),('subset30-text-2',subset),('cohort242-text',Path(a.cohort))]:
            campaign['gating'][name]=golden(sys.executable,manifest,name,'text-layer');save()
        campaign['gating']['determinism']=execute([sys.executable,'-m','tools.harness.compare',str(root/'subset30-text-1'),str(root/'subset30-text-2'),'--out',str(root/'text-determinism.json')],root,'text-determinism');save()
        campaign['gating']['speed']=execute([sys.executable,'-m','tools.harness.speed',str(subset),'--data',a.data,'--out',str(root/'speed-text')],root,'speed-text');save()
        # Required bounded page sample is the CPU Docling equivalence gate.
        # Run its distinct interpreter at lower priority after all text work.
        campaign['text_complete']=True;gate_status(campaign);save()
    os.nice(10)
    for index in (1,2):
        name=f'subset30-docling-pages-{index}'
        if name not in campaign['docling']:
            campaign['docling'][name]=golden(a.docling_python,subset,name,'docling-cpu-pages',2700);save()
        campaign['gating'][name]=campaign['docling'][name];save()
    if all((root/f'subset30-docling-pages-{i}/run.json').exists() for i in (1,2)):
        declared=audit_page_timings(root/'subset30-docling-pages-1', root/'subset30-docling-pages-2', root)
        campaign['docling']['pages-determinism']=execute([sys.executable,'-m','tools.harness.compare',str(root/'subset30-docling-pages-1'),str(root/'subset30-docling-pages-2'),'--declared',str(declared),'--out',str(root/'docling-pages-determinism.json')],root,'docling-pages-determinism');save()
        campaign['gating']['pages-determinism']=campaign['docling']['pages-determinism'];save()
    gate_status(campaign);save()
    if a.gating_only:
        campaign['complete']=True;save()
        return gate_status(campaign)
    docs=load_manifest(subset)
    by_pages=sorted(docs,key=lambda d:(d['pages'],d['filename']))
    pilot=[by_pages[0],by_pages[len(by_pages)//2],by_pages[-1]]
    pilot_rows=[]
    for index,row in enumerate(pilot):
        manifest=root/f'docling-pilot-{index}.json';write_json(manifest,{'documents':[row]})
        name=f'docling-pilot-{index}'
        if name not in campaign['docling']:
            campaign['docling'][name]=golden(a.docling_python,manifest,name,'docling-cpu',a.pilot_limit_seconds);save()
        path=root/name/'run.json'
        result=json.loads(path.read_text()) if path.exists() else None
        pilot_rows.append(dict(filename=row['filename'],pages=row['pages'],run=result,
                               execution=campaign['docling'][name]))
    # Never extrapolate a completed-run rate from a killed/incomplete pilot.
    successful=[r for r in pilot_rows if r['run'] and r['run'].get('complete') and r['run']['summary']['extract']['ok']==1]
    measured_seconds=sum(r['execution']['wall_seconds'] for r in successful)
    measured_pages=sum(r['pages'] for r in successful)
    censored=len(successful)!=3
    rate=measured_seconds/measured_pages if measured_pages else None
    conservative=max([r['execution']['wall_seconds']/r['pages'] for r in successful]+[
        r['execution']['wall_seconds']/r['pages'] for r in pilot_rows if r['execution']['timed_out']]) if pilot_rows else None
    projection_rate=conservative if censored else rate
    cohort_pages=sum(r['pages'] for r in json.loads((REPO/'tools/harness/manifests/cohort_inventory.json').read_text()))
    projection={'pilot_files':[r['filename'] for r in pilot_rows], 'pilot_complete':not censored,
        'measured_seconds_per_page':rate,'projection_seconds_per_page':projection_rate,
        'censored':censored,'basis':'whole public pipeline wall per input page, max observed/lower-bound rate when censored; no full-corpus eligibility from censored pilots',
        'subset30_seconds':projection_rate*sum(d['pages'] for d in docs) if projection_rate else None,
        'cohort242_seconds':projection_rate*cohort_pages if projection_rate else None}
    campaign['docling']['projection']=projection;save()
    # Stability uses the30 source documents, capped to three pages each;
    # the external process-group timeout bounds cooperative native timeouts.
    # No outer crash-loop here: that launcher creates a separate session which
    # would escape a process-group timeout. Signal resume is covered by tests;
    # this bounded sample records actual internal export/parser recovery.
    stability=root/'stability-input';stability.mkdir(exist_ok=True)
    for d in docs:
        (stability/d['filename']).symlink_to(Path(a.data)/d['filename'])
    campaign['docling']['stability']=execute([a.docling_python,'-m','glossapi.scripts.batched_stability_benchmark',
        '--pdfs',str(stability),'--out',str(root/'stability'),'--device','cpu','--mode','both',
        '--max-pages','3','--workers','2','--document-timeout','120','--artifacts-path',os.environ.get('DOCLING_ARTIFACTS_PATH',''),
        ],root,'stability',1800);save()
    full_eligible=not censored and projection['subset30_seconds']<=2700
    budget=2700/max(projection_rate or 1e30,1e-9)
    selected=[];used=0
    for d in sorted(docs,key=lambda d:d['filename']):
        if full_eligible or used+d['pages']<=budget:
            selected.append(d);used+=d['pages']
    bounded=root/'docling-bounded-subset.json';write_json(bounded,dict(documents=selected,projection=projection,budget_seconds=2700))
    campaign['docling']['selected_full_documents']=[d['filename'] for d in selected];save()
    if selected:
        for i in (1,2):
            name=f'docling-bounded-{i}'
            campaign['docling'][name]=golden(a.docling_python,bounded,name,'docling-cpu',2700);save()
        campaign['docling']['bounded-determinism']=execute([sys.executable,'-m','tools.harness.compare',str(root/'docling-bounded-1'),str(root/'docling-bounded-2'),'--out',str(root/'docling-bounded-determinism.json')],root,'docling-bounded-determinism');save()
    else:campaign['docling']['full_subset_skipped']='no complete document projected within 45 minutes'
    # A full cohort is opt-in only when the completed pilot projects <=3h.
    if not censored and projection['cohort242_seconds']<=10800:
        command=[a.docling_python,'-m','tools.harness.golden',a.cohort,'--data',a.data,'--out',str(root/'cohort242-docling'),'--profile','docling-cpu']
        log=(root/'cohort242-docling.log').open('w')
        process=subprocess.Popen(['timeout','--signal=TERM','--kill-after=10','10800',*command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();campaign['docling']['cohort_background']={'pid':process.pid,'command':command,'budget_seconds':10800}
    else:campaign['docling']['cohort_skipped']='projection >3h or pilot censored; defer full Docling baseline to GPU'
    campaign['complete']=True
    code=gate_status(campaign);save()
    return code

if __name__=='__main__':raise SystemExit(main())
