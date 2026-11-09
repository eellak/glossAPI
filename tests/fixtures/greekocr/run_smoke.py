"""Run the real registered plugin through Docling. No private scans or seeds needed."""
import os
os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
import argparse, hashlib, importlib.metadata, json, time, unicodedata
from pathlib import Path

def normalized(pages):
    # Ordered page and line structure retained; NFC and whitespace within each line only.
    return json.dumps([[unicodedata.normalize('NFC', ' '.join(t.split())) for t in page]
                       for page in pages], ensure_ascii=False, separators=(',', ':')).encode('utf-8')

def run(layout_dir, output):
    import torch
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON
    from docling.datamodel.pipeline_options import PdfPipelineOptions, LayoutOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from greekocr.deploy.docling_plugin.options import GlossapiOcrOptions
    assert importlib.metadata.version('docling') == '2.81.0'
    # Exercise both acquisition APIs, with explicit paths/overrides absent.
    assert not os.environ.get('GREEKOCR_SEGMENTER_MODEL_DIR'), 'Unset detector override'
    assert os.environ.get('GREEKOCR_CACHE_DIR'), 'Set glossAPI cache root'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    log = output/'plugin_calls.jsonl'
    if log.exists():
        raise FileExistsError('Use a fresh output directory: '+str(output))
    fixture = Path(__file__).resolve().parent/'polytonic_scan.pdf'
    opts = GlossapiOcrOptions(model_repo='fffoivos/greek-ocr', model_variant='x1',
        model_revision='2a6b019be728e5be70cdbd981414c630b63fca08',
        model_manifest_sha256='85d455d2cae2b756ab17e7114a3de13d7d41426d1fa5093028a179cf0c7d1345', device='cpu',
        segmenter='paddle_small_heuristics_r8', segmenter_detector='small',
        detector_device='cpu', segmenter_preset='clean_scan', segmenter_rules={},
        recognition_mode='detector', crop_scale=3.0, replace_policy='all',
        force_full_page_ocr=True, call_log=log)
    pipeline = PdfPipelineOptions(do_ocr=True, ocr_options=opts, allow_external_plugins=True,
        accelerator_options=AcceleratorOptions(device='cpu', num_threads=2),
        do_table_structure=False, images_scale=3.0, artifacts_path=Path(layout_dir),
        layout_options=LayoutOptions(model_spec=DOCLING_LAYOUT_HERON.model_copy(
            update={'revision':'8f39ad3c0b4c58e9c2d2c84a38465abf757272d8'})),
        ocr_batch_size=1, layout_batch_size=1, table_batch_size=1)
    start = time.perf_counter()
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={InputFormat.PDF:
        PdfFormatOption(pipeline_options=pipeline, backend=PyPdfiumDocumentBackend)})
    result = converter.convert(fixture)
    wall = time.perf_counter()-start
    calls = sorted([json.loads(s) for s in log.read_text().splitlines()], key=lambda r:r['page'])
    assert len(result.pages) == 2 and [r['page'] for r in calls] == [1,2]
    assert all(r['status'] == 'replaced' for r in calls)
    pages = [[r['text'] for r in c['lines']] for c in calls]
    rows = [r for c in calls for r in c['direct_rows']]
    from greekocr.segmenters.assets import fetch_small_detector, SMALL_DETECTOR
    from greekocr.deploy.docling_plugin.assets import fetch_model
    detector = fetch_small_detector()
    release, receipt = fetch_model(repo_id=opts.model_repo, variant=opts.model_variant,
        revision=opts.model_revision, manifest_sha256=opts.model_manifest_sha256)
    toolkit = importlib.metadata.distribution('greek-ocr-toolkit')
    installed_source = json.loads(toolkit.read_text('direct_url.json') or 'null')
    vcs_commit = (installed_source or {}).get('vcs_info', {}).get('commit_id')
    payload = dict(source_commit=vcs_commit, installed_toolkit_source=installed_source,
        source_commit_unavailable_reason=None if vcs_commit else 'installed_archive_has_no_vcs_metadata',
        detector=dict(SMALL_DETECTOR, cached_path=str(detector)),
        model_acquisition=receipt, model_release=str(release),
        options=opts.model_dump(mode='json'), schema='greekocr-docling-synthetic-smoke-v1', pages=pages,
        kept_lines=sum(r['status']=='kept' for r in rows),
        dropped_lines=sum(r['status']=='dropped' for r in rows),
        normalized_sha256=hashlib.sha256(normalized(pages)).hexdigest(),
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        wall_seconds=wall, seconds_per_page=wall/2,
        timings_s=[c['timings_s'] for c in calls],
        versions={n:importlib.metadata.version(n) for n in ['docling','docling-core','torch','onnxruntime','numpy','Pillow']})
    (output/'actual.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--layout-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); run(a.layout_dir.resolve(), a.output)
