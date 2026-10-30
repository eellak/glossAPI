"""Greek OCR package identity and lazy runtime defaults (no model downloads)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

MODEL = dict(model_repo='fffoivos/greek-ocr', model_variant='x1',
    model_revision='2a6b019be728e5be70cdbd981414c630b63fca08',
    model_manifest_sha256='85d455d2cae2b756ab17e7114a3de13d7d41426d1fa5093028a179cf0c7d1345')
INSTALL_MESSAGE = 'Install an OCR plugin with `python install_glossapi.py ocr --device cpu` (or mps/cuda126/cuda132).'


def package_requirements():
    return tuple(line.split(' #',1)[0].strip() for line in (Path(__file__).parents[1]/'ocr_package.txt').read_text().splitlines()
                 if line.strip() and not line.startswith('#'))


def cache_root(value=None):
    # GlossAPI owns the namespace; the plugin owns all acquisition/verification.
    return Path(value or os.environ.get('GLOSSAPI_CACHE_DIR') or
                Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'glossapi').expanduser().resolve()


def select_device(device=None):
    if device:
        return device
    try:
        import torch
    except ImportError:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda:0'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def planned_device(device=None):
    """Read-only hint; execution checks the real backend before dispatch."""
    import platform
    if device:
        return device
    if sys.platform == 'darwin' and platform.machine() == 'arm64':
        return 'mps'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        return 'cpu' if visible.lower() in ('','-1','none','nodevfiles') else 'cuda:0'
    if Path('/dev/nvidia0').exists():
        return 'cuda:0'
    return 'cpu'


def package_entry(device='cpu'):
    darwin = sys.platform == 'darwin'
    # Historical installed payload, not a download or complete Linux/CUDA lock.
    measured = 879630674 if darwin else 850327717
    return {'package_group':'ocr', 'distributions':list(package_requirements()),
        'entry_point':'docling:glossapi', 'install_command':f'python install_glossapi.py ocr --device {"mps" if device == "mps" else "cuda126" if device.startswith("cuda") else "cpu"}',
        'device':device, 'detector_device':'coreml' if darwin else 'cpu',
        'estimated_bytes':measured + 15152294 + 9880512,
        'size_scope':'historical Mac inference payload plus x1+detector; excludes transitive dependencies, layout and CUDA libraries',
        'size_is_estimate':True, 'installed':None, 'reason_code':'capabilities_not_recorded',
        'release_status':'PENDING-PUBLIC-SHA' if 'PENDING-PUBLIC-SHA' in (Path(__file__).parents[1]/'ocr_package.txt').read_text() else 'pinned_public_source'}


def validate_device(stage, device, settings):
    values = settings.get('options', {})
    if stage == 'ocr' and values.get('device',device) != device:
        raise ValueError('OCR option device must match its scheduler device pool')
    if device == 'cpu':
        return
    if stage != 'ocr' or settings.get('entry_point_name', 'glossapi') != 'glossapi':
        raise ValueError('Device is not qualified: '+device)
    if device != 'mps' and device != 'cuda' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('Unsupported OCR device: '+device)
    if values.get('device', device) != device:
        raise ValueError('OCR option device must match its scheduler device pool')


def accelerator_claims(stage, device, settings):
    claims = set()
    if device != 'cpu':
        claims.add('cuda:0' if device == 'cuda' else device)
    if stage == 'ocr' and settings.get('options', {}).get('detector_device') == 'coreml':
        claims.add('mps')
    return claims


def default_values(device='cpu'):
    return {**MODEL, 'segmenter':'paddle_small_heuristics_r8', 'segmenter_detector':'small',
        'segmenter_preset':'clean_scan', 'segmenter_rules':{}, 'device':device,
        'detector_device':'coreml' if sys.platform == 'darwin' else 'cpu',
        'recognition_mode':'detector', 'crop_scale':3.0, 'replace_policy':'routed_pages'}


def greek_options(values, *, device, routing_sidecar, call_log=None):
    if (values or {}).get('kind','glossapi') != 'glossapi':
        raise ValueError('Configured OCR kind does not match the Greek OCR engine; pass external options to Corpus.ocr()')
    try:
        from greekocr.deploy.docling_plugin.options import GlossapiOcrOptions
    except ImportError as error:
        raise RuntimeError(INSTALL_MESSAGE) from error
    options = {**default_values(device), **(values or {})}
    options['routing_sidecar'] = routing_sidecar
    if call_log is not None:
        options['call_log'] = call_log
    return GlossapiOcrOptions(**options)


def serialize_options(options):
    return options.model_dump(mode='json') if hasattr(options, 'model_dump') else {
        'kind':getattr(options, 'kind', None), 'options_unavailable_reason':'options_do_not_expose_json'}


def model_provenance(converter):
    """Read the plugin's verified receipt from the actual converter instance."""
    receipts = []
    for pipeline in getattr(converter, 'initialized_pipelines', {}).values():
        model = getattr(pipeline, 'ocr_model', None)
        receipt = getattr(model, 'model_acquisition', None)
        if receipt and receipt not in receipts:
            receipts.append(receipt)
    return receipts


def write_routes(path, jobs):
    rows = []
    for name, source, ranges in jobs:
        selected = {p for first,last in ranges for p in range(first,last+1)}
        rows.append({'filename':name, 'pages':[{'page':p, 'page_needs_ocr':True} for p in sorted(selected)]})
    Path(path).write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
