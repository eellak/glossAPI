"""Opt-in equality on owned cohort PDFs, never the owner's private scans.

This is a correctness test, with no speed assertions or benchmark loop.
"""
from dataclasses import replace
from contextlib import closing
import io
import json
import math
import os
from pathlib import Path
import uuid

import pytest


@pytest.mark.greekocr_integration
@pytest.mark.skipif(not os.environ.get('GLOSSAPI_OCR_COHORT_PDF'), reason='supply an owned cohort PDF')
def test_routed_cohort_matches_direct_pipeline(tmp_path):
    import cv2
    import numpy as np
    import pypdfium2 as pdfium
    import torch
    from greekocr.segmenters import get
    from greekocr.deploy.docling_plugin.assets import fetch_model
    from greekocr.deploy.docling_plugin.recognizer.inference import Recognizer
    from greekocr.deploy.docling_plugin.recognizer.strips import extract_strip
    from greekocr.deploy.docling_plugin.segmenter import native_crop
    from glossapi.ocr.defaults import MODEL
    from glossapi.ocr.page_artifacts import _join_page_outputs, _split_page_outputs
    from glossapi.page_quality import page_quality
    from glossapi.stages.contracts import ArtifactReference
    from glossapi.stages.execution import ReceiptCollector, sha256_file
    from glossapi.stages.ocr import OcrStage
    from test_ledger_stages import request

    torch.set_num_threads(2)
    source=Path(os.environ['GLOSSAPI_OCR_COHORT_PDF'])
    device=os.environ.get('GLOSSAPI_OCR_DEVICE','cpu')
    if device.startswith('cuda'):
        assert torch.cuda.is_available()
    report=page_quality(source)
    selected=[p.page for p in report.pages if p.page_needs_ocr][:2]
    assert selected, 'Selected cohort has no needs_ocr pages; choose a routed source'
    quality=tmp_path/'route.page_quality.json'
    payload=report.to_dict()
    for page in payload['pages']:
        page['page_needs_ocr']=page['page'] in selected
    quality.write_text(json.dumps(payload))
    with closing(pdfium.PdfDocument(source)) as pdf:
        native=[]
        for page in pdf:
            textpage=page.get_textpage(); native.append(textpage.get_text_range());textpage.close();page.close()
    native_path=tmp_path/'native.md';native_path.write_text(_join_page_outputs(native))
    given=request(tmp_path,source,operation='ocr.document',cache_root=os.environ['GREEKOCR_CACHE_DIR'],
                  options={'detector_device':'cpu'},num_threads=2)
    references=tuple(ArtifactReference(str(uuid.uuid4()),kind,path,sha256_file(path),path.stat().st_size)
        for kind,path in [('native_markdown',native_path),('page_quality',quality)])
    given=replace(given,device_id=device,inputs=(*given.inputs,*references),
                  deadline_monotonic_ns=__import__('time').monotonic_ns()+600_000_000_000)
    from glossapi.ledger import LedgerWriter
    from glossapi.stages.contracts import DocumentSnapshot, StageSpecification
    from glossapi.stages.scheduler import ScheduledOperation, StreamingScheduler
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with writer.transaction():
            observation=writer.run({}, {}, {},code_revision='cohort-equality')
            document=writer.document(source.resolve().as_uri())
            version=writer.version(document,source_sha256=sha256_file(source),stored_sha256=sha256_file(source))
            inputs=[]
            for artifact in given.inputs:
                file=writer.artifact(observation,'ocr',artifact.kind,artifact.path.resolve().as_uri(),version_id=version,
                    sha256=artifact.sha256,bytes=artifact.bytes,ownership='initial_external',retention_class='initial')
                inputs.append(replace(artifact,file_id=file))
        specification=StageSpecification('ocr.document',tuple(item.kind for item in inputs),
            ('ocr_markdown','ocr_metrics'),device,{**given.settings,'entry_point_name':'glossapi'})
        engine=StreamingScheduler(writer,tmp_path,workers={'ocr':1},cpu_tokens=2)
        operation=ScheduledOperation('cohort:ocr','ocr',DocumentSnapshot(document,version,source.as_uri(),artifacts=tuple(inputs)),
            specification,cpu_tokens=2)
        # Retain the plugin's candidate rows for the direct comparison consumer.
        outcome=engine.execute(engine.freeze([operation],settings={'debug':'on'}))
        assert outcome['state']=='completed',outcome
        attempt=outcome['attempts']['cohort:ocr']
        result=engine.result(attempt)
        work=next(item for item in engine._outputs(attempt) if item.kind=='worker_receipt').path.parent
    assert result.status=='ok',result.error
    calls=[json.loads(line) for line in (work/'sidecars/ocr_runtime/calls.jsonl').read_text().splitlines()]
    assert sorted(call['page'] for call in calls)==selected
    assert result.decisions[0]['plugin']['device']==device
    canonical=next(item for item in result.outputs if item.kind=='ocr_markdown')
    canonical_text=(work/canonical.relative_path).read_text()
    layout=json.loads((work/'sidecars/ocr_layout'/f'{source.stem}.json').read_text())['markdown_layout']
    if layout.get('pages'):
        actual_pages=['']*len(_split_page_outputs(native_path.read_text()))
        for page,start,end,_ in layout['pages']:
            actual_pages[page-1]=canonical_text[start:end]
    else:
        actual_pages=_split_page_outputs(canonical_text)
    # Compare to the actual retained Markdown surface (including its newline
    # convention), rather than PDFium's pre-serialization CRLF strings.
    for page,text in enumerate(_split_page_outputs(native_path.read_text()),1):
        if page not in selected:
            assert actual_pages[page-1].strip()==text.strip()

    release,receipt=fetch_model(repo_id=MODEL['model_repo'],variant=MODEL['model_variant'],
        revision=MODEL['model_revision'],manifest_sha256=MODEL['model_manifest_sha256'])
    segmenter=get('paddle_small_heuristics_r8',detector='small',detector_device='cpu',preset='clean_scan',image_threads=0)
    recognizer=Recognizer(str(release),device=device)
    comparisons=[]
    with closing(pdfium.PdfDocument(source)) as pdf:
        for number in selected:
            page=pdf[number-1];size=page.get_size()
            # Match Docling 2.81 PyPdfiumPageBackend's physical pixel contract:
            # 1.5x oversampling followed by Pillow's default resize to 3x.
            image=page.render(scale=4.5,rotation=0,crop=(0,0,0,0)).to_pil().resize(
                (round(size[0]*3),round(size[1]*3))).convert('RGB')
            page.close()
            segmentation=segmenter.segment(np.asarray(image),page_size=size)
            lines=[line for line in segmentation.lines if line.status=='kept']+[line for line in segmentation.lines if line.status=='dropped']
            encoded=io.BytesIO();image.save(encoded,format='PNG')
            gray=cv2.imdecode(np.frombuffer(encoded.getvalue(),np.uint8),cv2.IMREAD_GRAYSCALE)
            strips=[]
            for line in lines:
                if line.status!='kept':continue
                crop=np.asarray(native_crop(gray,line),dtype=np.float32)/255.
                pad=max(0,math.ceil(crop.shape[0]/6)-crop.shape[1])
                if pad:crop=np.pad(crop,((0,0),(pad//2,pad-pad//2)),constant_values=1.)
                strips.append(extract_strip(crop,(0,0,crop.shape[1],crop.shape[0])))
            predictions=iter(recognizer.recognize(strips))
            direct=[{'polygon':line.polygon,'bbox':list(line.bbox),'status':line.status,
                     'drop_reason':line.reason,'recognised_text':next(predictions)[0] if line.status=='kept' else ''}
                    for line in lines]
            actual=next(call for call in calls if call['page']==number)['direct_rows']
            actual=[{key:line[key] for key in direct[0]} for line in actual] if direct else actual
            assert actual==direct
            comparisons.append({'page':number,'candidates':len(direct),'kept':len(strips)})
    proof={'source_sha256':sha256_file(source),'source_filename':source.name,'device':device,
        'detector_device':'cpu','selected_needs_ocr_pages':selected,'pages':comparisons,
        'model':receipt,'direct_scope':'independent matching Docling 2.81 PDFium 4.5x-to-3x render, segmenter and recognizer invocation; shared official crop/strip primitives',
        'equal':True,'scheduler_run_id':outcome['run_id'],'scheduler_device':device,'cpu_tokens':2}
    (tmp_path/'cohort_equality.json').write_text(json.dumps(proof,indent=2))
