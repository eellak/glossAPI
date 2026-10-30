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


def assert_pinned_detector(plugin):
    """The recorded detector is the pinned upstream file, with no override, path and hash observed."""
    from greekocr.segmenters.assets import SMALL_DETECTOR
    detector=plugin['detector']
    assert detector['override'] is False and detector['source']=='pinned_upstream', detector
    assert detector['expected']['sha256']==SMALL_DETECTOR['sha256']
    assert detector['sha256']==SMALL_DETECTOR['sha256'] and detector['observed_sha256']==[SMALL_DETECTOR['sha256']]
    assert detector['matches_expected'] is True and detector['paths']
    assert all(row['sha256']==SMALL_DETECTOR['sha256'] for row in detector['observations'])


def offline_stage(tmp_path,monkeypatch,**settings):
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
                  options={'detector_device':'cpu'},**settings)
    return OcrStage().run_document(given,reporter=ReceiptCollector()),attempts


@pytest.mark.greekocr_integration
@pytest.mark.skipif(os.environ.get('GLOSSAPI_OCR_INTEGRATION')!='1',reason='requires installed plugin and warm caches')
def test_real_tampered_detector_override_is_recorded_and_fails_the_pinned_check(tmp_path,monkeypatch):
    """Reviewer case (F4): one ONNX metadata property added to a scratch detector copy via the override."""
    onnx=pytest.importorskip('onnx')
    from greekocr.segmenters.assets import SMALL_DETECTOR, fetch_small_detector
    monkeypatch.setenv('GREEKOCR_CACHE_DIR',os.environ['GREEKOCR_CACHE_DIR'])
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    model=onnx.load(str(fetch_small_detector()))
    entry=model.metadata_props.add(); entry.key='cp_ocrdef_review'; entry.value='tampered'
    override=tmp_path/'detector'; override.mkdir()
    onnx.save(model,str(override/'PP-OCRv6_small_det.onnx'))
    monkeypatch.setenv('GREEKOCR_SEGMENTER_MODEL_DIR',str(override))
    result,attempts=offline_stage(tmp_path,monkeypatch)
    assert result.status=='ok',result.error
    detector=result.decisions[0]['plugin']['detector']
    assert detector['override'] is True and detector['source']=='environment_override'
    assert detector['override_identity']['directory']==str(override.resolve())
    assert detector['paths']==[str(override/'PP-OCRv6_small_det.onnx')]
    assert detector['sha256']!=SMALL_DETECTOR['sha256'] and detector['matches_expected'] is False
    with pytest.raises(AssertionError):
        assert_pinned_detector(result.decisions[0]['plugin'])
    assert not attempts


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
    assert_pinned_detector(plugin)
    assert plugin['requested_device']=='cpu'
    assert plugin['toolkit']['admission'] in ('pinned_vcs_commit','pinned_code_fingerprint')
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


# ---- CP-OCRDEF fix round 1: red on e0bd5ff, green after the fix ----

PIN = defaults.toolkit_pin() if hasattr(defaults, 'toolkit_pin') else {
    'url':'https://github.com/fffoivos/greek-ocr-toolkit', 'commit':'bb5aa6fadf933a00a8cba50bc560ce723a157962'}
OTHER_COMMIT = 'f'*40


def fake_toolkit(root, *, commit=None, direct=None, files=None):
    """A real on-disk installed distribution: same version, chosen source metadata."""
    info = root/'greek_ocr_toolkit-0.1.0.dist-info'
    info.mkdir(parents=True)
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: greek-ocr-toolkit\nVersion: 0.1.0\n')
    direct = direct or {'url':PIN['url'], 'vcs_info':{'vcs':'git', 'commit_id':commit, 'requested_revision':commit}}
    (info/'direct_url.json').write_text(json.dumps(direct))
    for name, text in (files or {'__init__.py':'VERSION = "0.1.0"\n', 'segmenters/api.py':'X = 1\n'}).items():
        path = root/'greekocr'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def use_site(monkeypatch, root):
    import sys
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delenv('GLOSSAPI_OCR_TOOLKIT_SOURCE', raising=False)
    import importlib.metadata
    assert importlib.metadata.distribution('greek-ocr-toolkit').read_text('direct_url.json')
    assert str(root) == sys.path[0]


def test_package_gate_rejects_a_different_toolkit_commit(tmp_path, monkeypatch):
    """Reviewer probe (F1): all versions and the entry point present; only the commit differs."""
    import importlib.metadata as metadata
    from packaging.requirements import Requirement
    from glossapi.ledger import LedgerWriter
    from glossapi.stages.contracts import DocumentSnapshot, StageSpecification
    from glossapi.stages.scheduler import ScheduledOperation, StreamingScheduler
    versions = {Requirement(r).name: next(iter(Requirement(r).specifier)).version
                if Requirement(r).specifier else '0.1.0' for r in defaults.package_requirements()
                if not Requirement(r).marker or Requirement(r).marker.evaluate()}
    monkeypatch.setattr(metadata, 'version', lambda n: versions[n])
    monkeypatch.setattr(metadata, 'entry_points', lambda **kw: [SimpleNamespace(name='glossapi')])
    monkeypatch.delenv('GLOSSAPI_OCR_TOOLKIT_SOURCE', raising=False)
    results = {}
    for commit in ('0'*40, PIN['commit']):
        monkeypatch.setattr(metadata, 'distribution', lambda n, commit=commit: SimpleNamespace(version='0.1.0',
            read_text=lambda _: json.dumps({'url':PIN['url'], 'vcs_info':{'vcs':'git', 'commit_id':commit}})))
        op = ScheduledOperation('ocr', 'ocr', DocumentSnapshot('d', 'v', 'file:///test.pdf'),
                                StageSpecification('ocr.document', settings={'entry_point_name':'glossapi'}))
        root = tmp_path/commit[:4]; root.mkdir()
        with LedgerWriter(root/'corpus.sqlite3') as writer:
            results[commit] = StreamingScheduler(writer, root)._check_package(op)
    assert results['0'*40], 'A different installed toolkit commit passed the pinned package gate'
    assert '0'*40 in results['0'*40] and PIN['commit'] in results['0'*40]
    assert results[PIN['commit']] is False


def test_same_version_from_different_commits_admission_and_identity(tmp_path, monkeypatch):
    """F1: same version 0.1.0 installed from the pinned and from another commit."""
    import importlib.metadata as metadata
    from glossapi.ledger import LedgerWriter
    from glossapi.ledger.encoding import snapshot_hash
    from glossapi.stages.contracts import DocumentSnapshot, StageSpecification
    from glossapi.stages.scheduler import ScheduledOperation, StreamingScheduler, execution_environment
    from packaging.requirements import Requirement
    # Every other pinned dependency reports its pinned version; the toolkit is read from disk.
    versions = {Requirement(r).name: next(iter(Requirement(r).specifier)).version
                for r in defaults.package_requirements() if Requirement(r).specifier
                and (not Requirement(r).marker or Requirement(r).marker.evaluate())}
    real_version = metadata.version
    monkeypatch.setattr(metadata, 'version', lambda n: versions.get(n) or real_version(n))
    monkeypatch.setattr(metadata, 'entry_points', lambda **kw: [SimpleNamespace(name='glossapi')])
    op = ScheduledOperation('ocr', 'ocr', DocumentSnapshot('d', 'v', 'file:///test.pdf'),
                            StageSpecification('ocr.document', settings={'entry_point_name':'glossapi'}))
    sites = {label: fake_toolkit(tmp_path/label, commit=commit) for label, commit in
             (('pinned', PIN['commit']), ('other', OTHER_COMMIT))}
    gates, environments = {}, {}
    for label, site in sites.items():
        # Behaviour first (gate + frozen identity), using only APIs that existed at e0bd5ff.
        with monkeypatch.context() as scoped:
            use_site(scoped, site)
            root = tmp_path/(label+'-ledger'); root.mkdir()
            with LedgerWriter(root/'corpus.sqlite3') as writer:
                gates[label] = StreamingScheduler(writer, root)._check_package(op)
            environments[label] = snapshot_hash(execution_environment())
    assert gates['pinned'] is False
    assert gates['other'], 'same version from a different commit passed the package gate'
    assert environments['pinned'] != environments['other'], 'execution identity ignores the installed source'
    seen = {}
    for label, site in sites.items():
        with monkeypatch.context() as scoped:
            use_site(scoped, site)
            gate = gates[label]
            status = defaults.toolkit_source_status()
            environment = execution_environment()
            seen[label] = (gate, status, environment)
            if label == 'other':
                # An explicitly declared development override describing the installed source is admitted.
                scoped.setenv('GLOSSAPI_OCR_TOOLKIT_SOURCE', 'git+'+PIN['url']+'@'+OTHER_COMMIT)
                override = defaults.toolkit_source_status()
                assert override['admitted'] and override['status'] == 'development_override'
                assert snapshot_hash(execution_environment()) != snapshot_hash(environment)
                scoped.setenv('GLOSSAPI_OCR_TOOLKIT_SOURCE', 'git+'+PIN['url']+'@'+'e'*40)
                assert defaults.toolkit_source_status()['status'] == 'override_mismatch'
    assert seen['pinned'][0] is False and seen['pinned'][1]['status'] == 'pinned_vcs_commit'
    gate, status, _ = seen['other']
    assert isinstance(gate, str) and OTHER_COMMIT in gate and PIN['commit'] in gate and 'GLOSSAPI_OCR_TOOLKIT_SOURCE' in gate
    assert not status['admitted'] and status['status'] == 'commit_mismatch' and status['version'] == '0.1.0'
    pinned_ocr, other_ocr = seen['pinned'][2]['ocr']['toolkit'], seen['other'][2]['ocr']['toolkit']
    assert pinned_ocr['installed_commit'] == PIN['commit'] and other_ocr['installed_commit'] == OTHER_COMMIT
    assert snapshot_hash(seen['pinned'][2]) != snapshot_hash(seen['other'][2])


def test_install_without_vcs_metadata_uses_recorded_code_fingerprint(tmp_path, monkeypatch):
    """F1 fallback: archive/dir installs are admitted only when the code fingerprint matches the pin's."""
    site = fake_toolkit(tmp_path/'site', direct={'url':'file:///scratch/toolkit.tar.gz', 'archive_info':{}})
    digest = defaults.toolkit_code_fingerprint(site/'greekocr')
    package = tmp_path/'ocr_package.txt'
    package.write_text('\n'.join(line if not line.startswith('# greekocr-source-fingerprint ') else
        f'# greekocr-source-fingerprint {PIN["commit"]} {digest["sha256"]} {digest["python_files"]}'
        for line in (REPO/'src/glossapi/ocr_package.txt').read_text().splitlines())+'\n')
    monkeypatch.setattr(defaults, '_package_file', lambda: package)
    use_site(monkeypatch, site)
    status = defaults.toolkit_source_status()
    assert status['admitted'] and status['status'] == 'pinned_code_fingerprint'
    (site/'greekocr/segmenters/api.py').write_text('X = 2\n')
    status = defaults.toolkit_source_status()
    assert not status['admitted'] and status['status'] == 'fingerprint_mismatch' and PIN['commit'] in status['reason']


def test_recorded_fingerprint_belongs_to_the_pin_and_matches_a_pinned_install():
    pin = defaults.toolkit_pin()
    assert pin['fingerprint']['commit'] == pin['commit'] and pin['fingerprint']['python_files'] > 0
    status = defaults.toolkit_source_status(fingerprint=True)
    if status['status'] != 'pinned_vcs_commit':
        pytest.skip('no toolkit installed from the pinned VCS commit: '+status['status'])
    assert status['installed_code_fingerprint'] == {k:pin['fingerprint'][k] for k in ('sha256','python_files')}


def test_default_detector_accelerator_is_reserved(monkeypatch):
    """Reviewer probe (F2): the omitted macOS detector default (CoreML) claims the Apple accelerator."""
    monkeypatch.setattr(defaults.sys, 'platform', 'darwin')
    assert defaults.default_values('cpu')['detector_device'] == 'coreml'
    assert defaults.accelerator_claims('ocr', 'cpu', {}) == {'mps'}
    assert defaults.accelerator_claims('ocr', 'cpu', {'options':{}}) == {'mps'}
    assert defaults.accelerator_claims('ocr', 'cpu', {'options':{'detector_device':'cpu'}}) == set()


def _peak(outcome):
    active = peak = 0
    for row in outcome['trace']:
        active += 1 if row['event'] == 'dispatch' else -1 if row['event'] == 'complete' else 0
        peak = max(peak, active)
    return peak


@pytest.mark.parametrize('devices', [('cpu','cpu','cpu'), ('mps','cpu','mps','cpu')], ids=['coreml-coreml', 'mps-coreml'])
def test_omitted_detector_default_serializes_coreml_and_mps(tmp_path, monkeypatch, devices):
    """Reviewer probe (F2): omitted detector settings; CoreML/CoreML and MPS/CoreML must not overlap."""
    from glossapi.ledger import LedgerWriter
    from glossapi.stages.scheduler import StreamingScheduler
    monkeypatch.setattr(defaults.sys, 'platform', 'darwin')
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        extracts = [op for op in graph(writer, tmp_path, len(devices)) if op.stage == 'extract']
        operations = [replace(op, stage='ocr', specification=replace(op.specification, operation='ocr.test',
            device_id=device, settings={'delay':.03, 'options':{}})) for op, device in zip(extracts, devices)]
        engine = StreamingScheduler(writer, tmp_path, workers={'ocr':2}, cpu_tokens=4, registry={'ocr':DeviceProbe})
        run = engine.freeze(operations)
        outcome = engine.execute(run)
        rows = [json.loads(row[0]) for row in writer.connection.execute(
            f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='heartbeat'", (run,))]
    assert outcome['state'] == 'completed'
    assert _peak(outcome) <= 1, 'Operations using the implicit CoreML detector were admitted together'
    reservations = [row for row in rows if 'cpu_tokens' in row]
    assert len(reservations) == len(devices)
    assert all(row['detector_device'] == 'coreml' and 'mps' in row['accelerator_devices'] for row in reservations)


def test_corpus_device_mismatch_fails_before_adapter(tmp_path, monkeypatch):
    """Reviewer probe (F3): explicit cpu request vs ocr_options.device='cuda:1' is rejected before the adapter."""
    from glossapi.stages import compatibility as c
    from glossapi.corpus.ocr.config import OcrRequest
    source = simple_pdf(tmp_path/'source.pdf')
    opts = SimpleNamespace(kind='glossapi', device='cuda:1', model_dump=lambda **kw: {'device':'cuda:1'})
    req = OcrRequest('ocr_bad', 'docling', opts, 'cpu', None, False, None, 8, 220, 'single', None, False, False)
    seen = []
    monkeypatch.setattr(c, 'execute_compatibility_stage', lambda context, stage, inputs, specification: (
        seen.append(specification.device_id), (SimpleNamespace(outputs=()), tmp_path))[1])
    monkeypatch.setattr(c, 'materialize_stage_outputs', lambda *a: None)
    context = SimpleNamespace(output_dir=tmp_path, input_dir=tmp_path)
    with pytest.raises(ValueError, match='disagrees'):
        c.ocr_compatibility_documents(context, [source.name], request=req)
    assert seen == [], 'Explicit CPU request was silently replaced by the option device'


@pytest.mark.parametrize('requested,option,resolved', [('cpu','cpu','cpu'), (None,'cuda:1','cuda:1'), ('cuda','cuda:0','cuda:0')])
def test_corpus_records_requested_and_resolved_devices(tmp_path, monkeypatch, requested, option, resolved):
    from glossapi.stages import compatibility as c
    from glossapi.corpus.ocr.config import OcrRequest
    source = simple_pdf(tmp_path/'source.pdf')
    opts = SimpleNamespace(kind='glossapi', device=option, model_dump=lambda **kw: {'device':option})
    req = OcrRequest('ocr_bad', 'docling', opts, requested, None, False, None, 8, 220, 'single', None, False, False)
    seen = []
    monkeypatch.setattr(c, 'execute_compatibility_stage', lambda context, stage, inputs, specification: (
        seen.append(specification), (SimpleNamespace(outputs=()), tmp_path))[1])
    monkeypatch.setattr(c, 'materialize_stage_outputs', lambda *a: None)
    c.ocr_compatibility_documents(SimpleNamespace(output_dir=tmp_path, input_dir=tmp_path), [source.name], request=req)
    assert seen[0].device_id == resolved and seen[0].settings['requested_device'] == requested


def test_direct_dispatch_device_mismatch_fails_before_converter(tmp_path, monkeypatch):
    """F3, direct path: corpus/ocr/dispatch.py must not prefer the option device over the request."""
    from glossapi.corpus.ocr import dispatch
    from glossapi.corpus.ocr.config import OcrRequest
    source = simple_pdf(tmp_path/'source.pdf')
    monkeypatch.setattr(dispatch, 'adapters_enabled', lambda *a: False)
    built = []
    def build(request):
        built.append(request.device)
        raise RuntimeError('converter constructed')
    monkeypatch.setattr(dispatch, '_build_converter', build)
    opts = SimpleNamespace(kind='external', device='cuda:1')
    req = OcrRequest('ocr_bad', 'docling', opts, 'cpu', None, False, None, 8, 220, 'single', None, False, False)
    context = SimpleNamespace(output_dir=tmp_path/'out', input_dir=tmp_path)
    with pytest.raises(ValueError, match='disagrees'):
        dispatch.run_plugin_ocr(context, [source.name], request=req)
    assert built == [], 'converter was constructed for the substituted option device'


def test_detector_override_is_recorded_and_bound_into_execution_identity(tmp_path, monkeypatch):
    """F4: actual detector path/hash + override status; the override changes execution identity."""
    import sys
    import types
    from glossapi.ledger.encoding import snapshot_hash
    from glossapi.stages.scheduler import execution_environment
    pinned = {'repo_id':'PaddlePaddle/PP-OCRv6_small_det_onnx', 'filename':'inference.onnx',
              'sha256':'d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e', 'bytes':9880512}
    try:
        from greekocr.segmenters.assets import SMALL_DETECTOR as pinned
    except ImportError:
        for name in ('greekocr', 'greekocr.segmenters'):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, 'greekocr.segmenters.assets', types.SimpleNamespace(SMALL_DETECTOR=pinned))
    monkeypatch.delenv('GREEKOCR_SEGMENTER_MODEL_DIR', raising=False)
    baseline = snapshot_hash(execution_environment())
    override = tmp_path/'detector'; override.mkdir()
    model = override/'PP-OCRv6_small_det.onnx'; model.write_bytes(b'tampered detector metadata')
    monkeypatch.setenv('GREEKOCR_SEGMENTER_MODEL_DIR', str(override))
    first = snapshot_hash(execution_environment())
    model.write_bytes(b'another tampered detector')
    second = snapshot_hash(execution_environment())
    assert len({baseline, first, second}) == 3, 'detector override does not change execution identity'
    model.write_bytes(b'tampered detector metadata')
    observed = {'path':str(model), 'sha256':sha256_file(model), 'requested_device':'cpu', 'providers':['CPUExecutionProvider']}
    record = defaults.detector_provenance([observed])
    assert record['override'] is True and record['source'] == 'environment_override'
    assert record['override_identity']['directory'] == str(override.resolve())
    assert record['paths'] == [str(model)] and record['sha256'] == observed['sha256'] != pinned['sha256']
    assert record['matches_expected'] is False and record['expected']['sha256'] == pinned['sha256']
    monkeypatch.delenv('GREEKOCR_SEGMENTER_MODEL_DIR')
    clean = defaults.detector_provenance([{**observed, 'sha256':pinned['sha256']}])
    assert clean['override'] is False and clean['source'] == 'pinned_upstream' and clean['matches_expected'] is True


@pytest.mark.parametrize('routed', [False, True], ids=['unrouted', 'routed'])
def test_wrong_engine_is_rejected_on_routed_and_unrouted_documents(tmp_path, monkeypatch, routed):
    """Reviewer probe (F6): engine validation precedes the ocr_not_routed early return; no model loads."""
    from glossapi.corpus.ocr import dispatch
    import glossapi.page_quality as quality_module
    monkeypatch.setattr(dispatch, 'entry_points', lambda **kw: [SimpleNamespace(name='glossapi')])
    monkeypatch.setattr(dispatch, '_build_converter', lambda _: pytest.fail('a model was loaded'))
    monkeypatch.setattr(quality_module, 'page_quality', lambda *a, **k: pytest.fail('routing ran before validation'))
    source = simple_pdf(tmp_path/'source.pdf')
    native = tmp_path/'native.md'; native.write_text('Native')
    quality = tmp_path/'quality.json'
    quality.write_text(json.dumps({'pages_total':1, 'pages':[{'page':1, 'page_needs_ocr':routed}]}))
    given = request(tmp_path, source, operation='ocr.document', options={'kind':'another-engine'})
    refs = tuple(ArtifactReference(str(uuid.uuid4()), kind, p, sha256_file(p), p.stat().st_size)
                 for kind, p in [('native_markdown', native), ('page_quality', quality)])
    result = OcrStage().run_document(replace(given, inputs=(*given.inputs, *refs)), reporter=ReceiptCollector())
    assert result.status == 'failed', 'A mismatched explicit engine was accepted'
    assert result.reason_code == 'ocr_configuration_invalid' and 'does not match the Greek OCR engine' in result.error.message
    assert not (given.work_directory/'json/metrics').exists()
