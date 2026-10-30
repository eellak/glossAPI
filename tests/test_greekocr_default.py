"""Default engine, package identity, routing and device reservations."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tomllib
import uuid

import pytest

from glossapi.ocr import defaults
from glossapi.scripts import install_glossapi as installer
from glossapi.stages.contracts import ArtifactReference
from glossapi.stages.execution import ReceiptCollector, sha256_file
from glossapi.stages.ocr import OcrStage
from test_ledger_stages import request, simple_pdf
from test_ledger_scheduler import Producer, graph

REPO = Path(__file__).resolve().parents[1]


class DeviceProbe(Producer):
    name='ocr'


def test_one_toolkit_pin_supplies_installer_and_extra():
    configuration = tomllib.loads((REPO/'pyproject.toml').read_text())
    assert configuration['tool']['setuptools']['dynamic']['optional-dependencies']['ocr']['file'] == ['src/glossapi/ocr_package.txt']
    requirements = defaults.package_requirements()
    assert len([row for row in requirements if row.startswith('greek-ocr-toolkit')]) == 1
    plan = installer.build_install_plan(packages=['ocr'])
    commands = installer.build_pip_commands(plan, REPO)
    assert all(row in commands[1] for row in requirements)
    assert commands[0][4].startswith('torch==2.14.1')
    assert not any('onnxruntime-gpu' in row for row in requirements)


@pytest.mark.parametrize('machine', ['x86_64','aarch64'])
@pytest.mark.parametrize('device', ['cpu','cuda126','cuda132'])
def test_linux_ocr_plan_pins_torch_and_supports_both_architectures(monkeypatch,machine,device):
    monkeypatch.setattr(installer.platform,'system',lambda:'Linux')
    monkeypatch.setattr(installer.platform,'machine',lambda:machine)
    plan=installer.build_install_plan(packages=['ocr'],device=device,
        ocr_toolkit_source='file:///scratch/toolkit')
    commands=installer.build_pip_commands(plan,REPO)
    assert 'greek-ocr-toolkit[docling] @ file:///scratch/toolkit' in commands[1]
    assert not any('github.com' in item for item in commands[1])
    assert commands[0][-1].endswith('/cpu' if device=='cpu' else '/cu'+device[4:])


def test_plan_lists_default_package_and_honors_device():
    from glossapi.ledger.reader import plan_view
    plan=plan_view(selectors=[str(REPO/'tests/fixtures/greekocr/polytonic_scan.pdf')],stages=['ocr'],devices={'ocr':'cuda:1'})
    package=next(row for row in plan['requirements'] if row['package_group']=='ocr')
    pool=next(row for row in plan['pools'] if row['stage']=='ocr')
    assert package['entry_point']=='docling:glossapi' and package['estimated_bytes']>0
    assert 'install_glossapi.py ocr' in package['install_command']
    assert pool['device_id']=='cuda:1' and pool['cpu_tokens']>=1


def test_accelerators_are_exclusive_and_detector_reserves_mps():
    assert defaults.accelerator_claims('ocr','cuda:1',{'options':{'detector_device':'cpu'}})=={'cuda:1'}
    assert defaults.accelerator_claims('ocr','cpu',{'options':{'detector_device':'coreml'}})=={'mps'}
    defaults.validate_device('ocr','cuda:1',{'options':{'device':'cuda:1'}})
    with pytest.raises(ValueError,match='match its scheduler'):
        defaults.validate_device('ocr','cuda:1',{'options':{'device':'cuda:0'}})
    with pytest.raises(ValueError,match='not qualified'):
        defaults.validate_device('extract','cuda:0',{})


def test_option_device_cannot_escape_its_cpu_pool(tmp_path):
    given=request(tmp_path,operation='ocr.document',options={'device':'cuda:0'})
    result=OcrStage().run_document(given,reporter=ReceiptCollector())
    assert result.status=='failed' and 'match its scheduler' in result.error.message


def test_scheduler_serializes_one_accelerator_and_records_cpu_reservation(tmp_path):
    from glossapi.ledger import LedgerWriter
    from glossapi.stages.scheduler import StreamingScheduler
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        operations=[replace(op,stage='ocr',specification=replace(op.specification,operation='ocr.test',device_id='cuda:0',
            settings={'delay':.03,'options':{'device':'cuda:0','detector_device':'cpu'}}))
            for op in graph(writer,tmp_path,3) if op.stage=='extract']
        engine=StreamingScheduler(writer,tmp_path,workers={'ocr':2},cpu_tokens=2,registry={'ocr':DeviceProbe})
        run=engine.freeze(operations)
        result=engine.execute(run)
        assert result['state']=='completed'
        active=0
        for row in result['trace']:
            active += 1 if row['event']=='dispatch' else -1 if row['event']=='complete' else 0
            assert active<=1
        rows=[json.loads(row[0]) for row in writer.connection.execute(
            f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='heartbeat'",(run,))]
        reservations=[row for row in rows if 'cpu_tokens' in row]
        assert len(reservations)==3
        assert all(row['cpu_tokens']==1 and row['accelerator_devices']==['cuda:0'] for row in reservations)


def test_unflagged_pages_pass_through_without_loading_model(tmp_path,monkeypatch):
    from glossapi.corpus.ocr import dispatch
    monkeypatch.setattr(dispatch,'entry_points',lambda **kw:[SimpleNamespace(name='glossapi')])
    monkeypatch.setattr(dispatch,'_build_converter',lambda _:pytest.fail('unflagged pages must not enter OCR'))
    source=simple_pdf(tmp_path/'source.pdf')
    native=tmp_path/'source.md';native.write_bytes(b'Native text must remain byte-identical\n')
    quality=tmp_path/'source.page_quality.json'
    quality.write_text(json.dumps({'pages_total':1,'pages':[{'page':1,'page_needs_ocr':False}]}))
    given=request(tmp_path,source,operation='ocr.document')
    refs=tuple(ArtifactReference(str(uuid.uuid4()),kind,path,sha256_file(path),path.stat().st_size)
        for kind,path in [('native_markdown',native),('page_quality',quality)])
    result=OcrStage().run_document(replace(given,inputs=(*given.inputs,*refs)),reporter=ReceiptCollector())
    assert result.status=='ok' and result.reason_code=='ocr_not_routed'
    output=next(item for item in result.outputs if item.kind=='ocr_markdown')
    assert (given.work_directory/output.relative_path).read_bytes()==native.read_bytes()


def test_pinned_pdfium_routes_the_real_scan():
    from glossapi.page_quality import page_quality
    report=page_quality(REPO/'tests/fixtures/greekocr/polytonic_scan.pdf',dpi_probe=True)
    assert [page.page_needs_ocr for page in report.pages]==[True,True]
    assert all(page.image_area_ratio>0.99 for page in report.pages)
    assert all(page.min_image_dpi>0 for page in report.pages)


@pytest.mark.greekocr_integration
@pytest.mark.skipif(os.environ.get('GLOSSAPI_OCR_INTEGRATION')!='1',reason='requires installed plugin and warm caches')
def test_real_default_stage_cpu_and_provenance(tmp_path,monkeypatch):
    import socket
    attempts=[]
    def blocked(*args,**kwargs):
        attempts.append(True)
        raise AssertionError('offline OCR attempted socket access')
    for name in ('connect','connect_ex'):
        monkeypatch.setattr(socket.socket,name,blocked)
    for name in ('create_connection','getaddrinfo'):
        monkeypatch.setattr(socket,name,blocked)
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    source=REPO/'tests/fixtures/greekocr/polytonic_scan.pdf'
    given=request(tmp_path,source,operation='ocr.document',cache_root=os.environ['GREEKOCR_CACHE_DIR'],
                  options={'detector_device':'cpu'})
    result=OcrStage().run_document(given,reporter=ReceiptCollector())
    assert result.status=='ok',result.error
    plugin=result.decisions[0]['plugin']
    assert plugin['device']==plugin['detector_device']=='cpu'
    assert plugin['options']['segmenter_preset']=='clean_scan'
    assert plugin['models'][0]['resolved_commit']==defaults.MODEL['model_revision']
    assert plugin['models'][0]['manifest_sha256']==defaults.MODEL['model_manifest_sha256']
    assert not attempts
    calls=[json.loads(line) for line in (given.work_directory/'sidecars/ocr_runtime/calls.jsonl').read_text().splitlines()]
    expected=json.loads((REPO/'tests/fixtures/greekocr/expected.json').read_text())
    assert [[line['text'] for line in call['lines']] for call in sorted(calls,key=lambda x:x['page'])]==expected['pages']


@pytest.mark.greekocr_integration
@pytest.mark.skipif(os.environ.get('GLOSSAPI_OCR_RUN_INTEGRATION')!='1',reason='requires current native cleaner and installed OCR')
def test_public_run_routing_preserves_text_layer_and_publishes_ocr(tmp_path):
    from fpdf import FPDF
    from pypdf import PdfReader, PdfWriter
    from glossapi import Corpus
    from glossapi.ocr.page_artifacts import _split_page_outputs
    from glossapi.ledger.compact import connect

    native=tmp_path/'native.pdf'
    pdf=FPDF();pdf.add_page();pdf.set_font('Helvetica',size=12)
    text=('This readable page describes a library collection and its native text layer. '
        'The archive stores printed books, reports, articles and historical documents. '
        'Readers search the catalogue by author, subject, title and publication year. '
        'Each record connects a source file with its stable identifier and descriptive metadata. '
        'A careful extraction process keeps paragraphs in their original reading order. '
        'Quality inspection checks the available characters before choosing which pages require recognition. '
        'This paragraph has varied sentences and ordinary words without a repeating line pattern. '
        'The following page contains a synthetic Greek scan for a separate recognition check. '
        'Preserving this passage allows the test to distinguish retained native content from newly recognized text.')
    pdf.multi_cell(w=180,h=7,text=text);pdf.output(str(native))
    writer=PdfWriter();writer.add_page(PdfReader(native).pages[0])
    writer.add_page(PdfReader(REPO/'tests/fixtures/greekocr/polytonic_scan.pdf').pages[0])
    source=tmp_path/'mixed.pdf';writer.write(source)
    corpus=Corpus(tmp_path/'input',tmp_path/'output')
    outcome=corpus.run([source],stages=('ocr','export'),workers={'extract':1,'clean':1,'ocr':1},
        settings={'cpu_tokens':2,'ocr':{'device_id':os.environ.get('GLOSSAPI_OCR_DEVICE','cpu'),
            'cache_root':os.environ['GREEKOCR_CACHE_DIR'],'options':{'detector_device':'cpu'}}})
    assert outcome['state']=='completed',outcome
    canonical=(corpus.output_dir/'markdown/mixed.md').read_text()
    native_saved=(corpus.output_dir/'sidecars/ocr_native/mixed.md').read_text()
    assert ' '.join(native_saved.split())==' '.join(text.split())
    assert canonical.startswith(native_saved.rstrip())
    expected=json.loads((REPO/'tests/fixtures/greekocr/expected.json').read_text())['pages'][0]
    assert all(line in canonical for line in expected)
    with connect(corpus.output_dir/'corpus.sqlite3') as connection:
        stages=connection.execute("SELECT a.device_id,s.status FROM attempts a JOIN tasks t USING(task_id) JOIN current_task_state s USING(task_id) WHERE t.run_id=? AND t.operation='ocr.document'",(outcome['run_id'],)).fetchall()
        assert len(stages)==1 and stages[0][1]=='ok'
    records=[json.loads(line) for line in (corpus.output_dir/'export.jsonl').read_text().splitlines()]
    assert len(records)==1 and expected[0] in records[0]['document']
    assert records[0]['ocr_success'] and records[0]['extraction_mode']=='docling'
    before=(corpus.output_dir/'export.jsonl').read_bytes()
    resumed=corpus.run(resume=outcome['run_id'])
    assert resumed['state']=='completed'
    assert (corpus.output_dir/'export.jsonl').read_bytes()==before


@pytest.mark.greekocr_integration
@pytest.mark.skipif(os.environ.get('GLOSSAPI_OCR_RUN_INTEGRATION')!='1',reason='requires current native cleaner and installed OCR')
def test_public_corpus_ocr_uses_default_engine(tmp_path,monkeypatch):
    import shutil
    import pandas as pd
    from glossapi import Corpus
    monkeypatch.setenv('GLOSSAPI_CACHE_DIR',os.environ['GREEKOCR_CACHE_DIR'])
    source=tmp_path/'input';source.mkdir()
    shutil.copy2(REPO/'tests/fixtures/greekocr/polytonic_scan.pdf',source/'scan.pdf')
    corpus=Corpus(source,tmp_path/'output')
    metadata=corpus.output_dir/'download_results/download_results.parquet'
    metadata.parent.mkdir(exist_ok=True)
    pd.DataFrame([{'filename':'scan.pdf','needs_ocr':True,'ocr_pages':'1-2'}]).to_parquet(metadata)
    corpus.ocr(mode='ocr_bad',device='cpu')
    text=(corpus.output_dir/'markdown/scan.md').read_text()
    expected=json.loads((REPO/'tests/fixtures/greekocr/expected.json').read_text())['pages']
    assert all(line in text for page in expected for line in page)
    row=pd.read_parquet(metadata).iloc[0]
    assert row.ocr_success and not row.needs_ocr
