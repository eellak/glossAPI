"""Greek OCR package identity and lazy runtime defaults (no model downloads)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

MODEL = dict(model_repo='fffoivos/greek-ocr', model_variant='x1',
    model_revision='2a6b019be728e5be70cdbd981414c630b63fca08',
    model_manifest_sha256='85d455d2cae2b756ab17e7114a3de13d7d41426d1fa5093028a179cf0c7d1345')
INSTALL_MESSAGE = 'Install an OCR plugin with `python install_glossapi.py ocr --device cpu` (or mps/cuda126/cuda132).'


TOOLKIT_DISTRIBUTION = 'greek-ocr-toolkit'
TOOLKIT_OVERRIDE_ENV = 'GLOSSAPI_OCR_TOOLKIT_SOURCE'
DETECTOR_OVERRIDE_ENV = 'GREEKOCR_SEGMENTER_MODEL_DIR'
_FINGERPRINT_MARK = '# greekocr-source-fingerprint '


def _package_file():
    return Path(__file__).parents[1]/'ocr_package.txt'


def package_requirements():
    return tuple(line.split(' #',1)[0].strip() for line in _package_file().read_text().splitlines()
                 if line.strip() and not line.startswith('#'))


def _split_source(value):
    """Return (url, revision) for a pip source URL.

    As in pip, a VCS revision follows the last '@' of the URL *path*, so user
    information in the authority (``git+ssh://git@host/...``) stays in the URL and
    branch/tag names may contain '/' (``...@dev/glossapi-ocr-default``). The URL
    fragment (``#egg=``/``#subdirectory=``) is not part of the source URL.
    """
    from urllib.parse import urlsplit, urlunsplit
    value = (value or '').strip()
    if value.startswith('git+'):
        value = value[4:]
        parts = urlsplit(value)
        if parts.scheme and parts.netloc:
            path, separator, revision = parts.path.rpartition('@')
            if not separator:
                path, revision = parts.path, None
            return _normal_url(urlunsplit((parts.scheme, parts.netloc, path, parts.query, ''))), revision
    return _normal_url(value), None


def _normal_url(url):
    url = (url or '').rstrip('/')
    return url[:-4] if url.endswith('.git') else url


def toolkit_pin():
    """The pinned toolkit source and its recorded code fingerprint, from ocr_package.txt only."""
    requirement = next(row for row in package_requirements() if row.startswith('greek-ocr-toolkit'))
    url, commit = _split_source(requirement.split(' @ ',1)[1] if ' @ ' in requirement else '')
    fingerprint = None
    for line in _package_file().read_text().splitlines():
        if line.startswith(_FINGERPRINT_MARK):
            recorded_commit, digest, count = line[len(_FINGERPRINT_MARK):].split()
            fingerprint = {'commit':recorded_commit, 'sha256':digest, 'python_files':int(count)}
    return {'requirement':requirement, 'url':url, 'commit':commit, 'fingerprint':fingerprint}


def toolkit_code_fingerprint(package_directory):
    """SHA256 over sorted (relative path, file SHA256) rows of the installed greekocr/**/*.py files."""
    root = Path(package_directory)
    rows = {'greekocr/'+path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py')) if '__pycache__' not in path.parts}
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    return {'sha256':digest, 'python_files':len(rows)}


def toolkit_source_status(*, fingerprint=False, environ=None):
    """Admission verdict for the installed toolkit source against the single pin.

    Admitted: PEP 610 VCS commit equal to the pin; no VCS metadata but the installed
    code fingerprint equals the recorded fingerprint of the pinned commit; or an
    explicitly declared development override (GLOSSAPI_OCR_TOOLKIT_SOURCE) that
    describes the installed source. Everything else is rejected with a reason.
    """
    import importlib.metadata
    environ = os.environ if environ is None else environ
    pin = toolkit_pin()
    override = environ.get(TOOLKIT_OVERRIDE_ENV) or None
    status = {'distribution':TOOLKIT_DISTRIBUTION, 'pinned_url':pin['url'], 'pinned_commit':pin['commit'],
              'override_declared':override}
    try:
        distribution = importlib.metadata.distribution(TOOLKIT_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return {**status, 'admitted':False, 'status':'not_installed', 'reason':INSTALL_MESSAGE}
    try:
        direct = json.loads(distribution.read_text('direct_url.json') or 'null') or {}
    except (TypeError, ValueError):
        direct = {}
    vcs = direct.get('vcs_info') or {}
    commit = vcs.get('commit_id')
    installed_url = _normal_url(direct.get('url'))
    status.update(version=getattr(distribution,'version',None) or 'of unknown version', installed_url=installed_url or None,
                  installed_commit=commit, installed_direct_url=direct or None)
    code = None
    def installed_code():
        locate = getattr(distribution, 'locate_file', None)
        directory = Path(locate('greekocr')) if locate else None
        return toolkit_code_fingerprint(directory) if directory and directory.is_dir() else None
    if fingerprint:
        code = installed_code()
        status['installed_code_fingerprint'] = code
    if override:
        url, revision = _split_source(override)
        if installed_url == url and (revision is None or revision in (commit, vcs.get('requested_revision'))):
            return {**status, 'admitted':True, 'status':'development_override',
                    'reason':f'{TOOLKIT_OVERRIDE_ENV} declares this installed source as a development override'}
        return {**status, 'admitted':False, 'status':'override_mismatch',
                'reason':(f'{TOOLKIT_OVERRIDE_ENV}={override} does not describe the installed greek-ocr-toolkit '
                          f'source ({installed_url or "unknown"}@{commit or "no-vcs"}).')}
    if commit:
        if commit == pin['commit'] and installed_url == pin['url']:
            return {**status, 'admitted':True, 'status':'pinned_vcs_commit'}
        return {**status, 'admitted':False, 'status':'commit_mismatch',
                'reason':(f'Installed greek-ocr-toolkit {status["version"]} comes from {installed_url}@{commit}, but glossAPI '
                          f'pins {pin["url"]}@{pin["commit"]} in ocr_package.txt. Reinstall with `python install_glossapi.py ocr`, '
                          f'or declare a development override with {TOOLKIT_OVERRIDE_ENV}=<installed source>.')}
    recorded = pin['fingerprint']
    if code is None:
        code = installed_code()
        status['installed_code_fingerprint'] = code
    if recorded and recorded['commit'] == pin['commit'] and code and code['sha256'] == recorded['sha256']:
        return {**status, 'admitted':True, 'status':'pinned_code_fingerprint'}
    return {**status, 'admitted':False, 'status':'fingerprint_mismatch',
            'reason':(f'Installed greek-ocr-toolkit {status["version"]} has no VCS metadata and its code fingerprint '
                      f'{(code or {}).get("sha256")} does not match the recorded fingerprint of pinned commit {pin["commit"]}. '
                      f'Reinstall with `python install_glossapi.py ocr`, or declare a development override with '
                      f'{TOOLKIT_OVERRIDE_ENV}=<installed source>.')}


def detector_override(environ=None):
    """Identity of a GREEKOCR_SEGMENTER_MODEL_DIR detector override (None when absent)."""
    environ = os.environ if environ is None else environ
    directory = environ.get(DETECTOR_OVERRIDE_ENV)
    if not directory:
        return None
    path = Path(directory).expanduser()
    files = {item.name:hashlib.sha256(item.read_bytes()).hexdigest()
             for item in sorted(path.glob('*.onnx')) if item.is_file()} if path.is_dir() else {}
    return {'environment':DETECTOR_OVERRIDE_ENV, 'directory':str(path.resolve()), 'onnx_sha256':files}


def detector_provenance(observations, *, segmenter_detector='small', environ=None):
    """Actual detector path/hash per call, override status, and the expected upstream identity."""
    try:
        from greekocr.segmenters.assets import SMALL_DETECTOR
        expected = dict(SMALL_DETECTOR) if segmenter_detector == 'small' else None
    except ImportError:
        expected = None
    override = detector_override(environ)
    observed = sorted({row['sha256'] for row in observations if row.get('sha256')})
    return {'expected':expected, 'override':override is not None, 'override_identity':override,
        'source':'environment_override' if override else 'pinned_upstream',
        'paths':sorted({row['path'] for row in observations if row.get('path')}),
        'sha256':observed[0] if len(observed) == 1 else None, 'observed_sha256':observed,
        'matches_expected':bool(expected) and observed == [expected['sha256']],
        'observations':observations}


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
        return 'cpu' if visible.strip().lower() in ('','-1','none','nodevfiles') else 'cuda:0'
    if any(path.name[6:].isdigit() for path in Path('/dev').glob('nvidia[0-9]*')):
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


def effective_ocr_settings(device, settings):
    """Resolve Greek OCR plugin defaults (including detector_device) before freezing an operation."""
    if settings.get('entry_point_name', 'glossapi') != 'glossapi':
        return settings
    return {**settings, 'options':{**default_values(device), **(settings.get('options') or {})}}


def accelerator_claims(stage, device, settings):
    """Claims derive from effective settings: an omitted detector_device uses its platform default."""
    claims = set()
    if device != 'cpu':
        claims.add('cuda:0' if device == 'cuda' else device)
    if stage == 'ocr':
        values = settings.get('options') or {}
        detector = values.get('detector_device')
        if detector is None and settings.get('entry_point_name', 'glossapi') == 'glossapi':
            detector = default_values(device)['detector_device']
        if detector == 'coreml':
            claims.add('mps')
    return claims


def _normal_device(device):
    return 'cuda:0' if device == 'cuda' else device


def check_requested_device(requested, options):
    """Reject a requested device that disagrees with ocr_options.device, before any pool/converter choice."""
    option_device = getattr(options, 'device', None) if options is not None else None
    if requested and option_device and _normal_device(str(requested)) != _normal_device(str(option_device)):
        raise ValueError(f'Requested OCR device {requested!r} disagrees with ocr_options.device {option_device!r}; '
                         'make them equal or set only one')
    return option_device


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
