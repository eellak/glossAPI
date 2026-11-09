"""Dependency-light receipts shared by the node harness."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import threading
import time

STAGES = ('extract', 'clean', 'sections', 'annotate', 'export')
REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def load_manifest(path):
    value = json.loads(Path(path).read_text())
    rows = value if isinstance(value, list) else value['documents']
    if not rows or len({r['filename'] for r in rows}) != len(rows):
        raise ValueError('manifest must contain unique nonempty filenames')
    for row in rows:
        if Path(row['filename']).name != row['filename'] or not row['filename'].endswith('.pdf'):
            raise ValueError('manifest filenames must be PDF basenames')
        if len(row['sha256']) != 64:
            raise ValueError('manifest requires sha256 for every PDF')
    return rows


def file_role(relative):
    """Role of one file under a pipeline output root (LS5-G05): 'product', 'diagnostic' or 'ignored'.

    Diagnostics are named explicitly, not waived by a broad suffix rule: runtime ``*.log`` files (conversion.log,
    attempt logs) and the downloader's ``logs/*.logs`` request/progress files (``gloss_downloader.py``). They are
    hashed into a separate diagnostics map; they are never compared as products. Lock files and the harness's own
    direct-path marker are ignored.
    """
    path = PurePosixPath(Path(relative).as_posix())
    if path.suffix == '.lock' or path.name == '.harness-direct-paths':
        return 'ignored'
    if path.suffix == '.log' or (path.suffix == '.logs' and len(path.parts) >= 2 and path.parts[-2] == 'logs'):
        return 'diagnostic'
    return 'product'


def _facts(root, role):
    root = Path(root)
    return {str(p.relative_to(root)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file() and file_role(p.relative_to(root)) == role}


def inventory(root):
    """All pipeline product files, including state. Diagnostics are kept separately (diagnostics()), never products."""
    return _facts(root, 'product')


def diagnostics(root):
    """Diagnostic files (runtime logs, downloader .logs) with their hashes: retained evidence, not products."""
    return _facts(root, 'diagnostic')


def code_receipt():
    shipped = REPO / '.code-receipt.json'
    if shipped.exists():
        receipt = json.loads(shipped.read_text())
        for name, digest in receipt['files'].items():
            if sha256(REPO / name) != digest:
                raise ValueError(f'shipped tree drift: {name}')
        return receipt
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=REPO).decode().split('\0')
    files = {name: sha256(REPO / name) for name in sorted(set(names)) if name and (REPO / name).is_file()}
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'tree_sha256': hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'files': files, 'kind': 'tracked-and-new-working-tree'}


def _du_bytes(path, *flags):
    try:
        return int(subprocess.check_output(['du', '-sk', *flags, str(path)], text=True, stderr=subprocess.DEVNULL).split()[0]) * 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def venv_footprint():
    """DW2-G03: own bytes under sys.prefix and the bytes it resolves to through symlinked shared dependencies."""
    return {'venv_own_bytes': _du_bytes(sys.prefix), 'venv_resolved_bytes': _du_bytes(sys.prefix, '-L'),
            'venv_bytes_basis': 'du -sk sys.prefix (own: symlinks counted as links) and du -skL sys.prefix (resolved: '
                                'symlinked shared dependency directories followed); 1 KiB blocks; hard links counted once'}


def _probe_pdf():
    """A one-page PDF whose only text uses an unembedded TrueType Arial, so PDFium must substitute a font."""
    content = b'BT /F1 12 Tf 10 50 Td (Ab 4,12%) Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>',
               b'<< /Length %d >>\nstream\n' % len(content) + content + b'\nendstream',
               b'<< /Type /Font /Subtype /TrueType /BaseFont /Arial /FirstChar 32 /LastChar 126 /Encoding /WinAnsiEncoding >>']
    body, offsets = b'%PDF-1.4\n', []
    for number, value in enumerate(objects, 1):
        offsets.append(len(body))
        body += b'%d 0 obj\n' % number + value + b'\nendobj\n'
    xref = len(body)
    body += b'xref\n0 %d\n0000000000 65535 f \n' % (len(objects) + 1) + b''.join(b'%010d 00000 n \n' % o for o in offsets)
    return body + b'trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (len(objects) + 1, xref)


def pdfium_identity():
    """T5-G07: the PDFium build and the font it substitutes for an unembedded Arial.

    Equal package versions do not promise equal products across machines: PDFium resolves unembedded fonts from the
    OS (macOS system Arial) or its bundled fallback (the node's Chrom Sans). Exact baselines stay per machine.
    """
    try:
        import ctypes
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
    except Exception as error:  # an environment without pypdfium2 records why
        return {'available': False, 'error': f'{type(error).__name__}: {error}'}
    identity = {'available': True, 'pypdfium2': importlib.metadata.version('pypdfium2')}
    version = getattr(pdfium, 'version', None)
    for name in ('PDFIUM_INFO', 'PYPDFIUM_INFO'):
        value = getattr(version, name, None) if version is not None else getattr(pdfium, name, None)
        if value is not None:
            identity[name.lower()] = str(value)
    try:
        import pypdfium2_raw
        identity['native_library'] = {p.name: sha256(p) for p in sorted(Path(pypdfium2_raw.__file__).parent.iterdir())
                                      if p.is_file() and 'pdfium' in p.name and p.suffix in ('.so', '.dylib', '.dll')}
    except Exception as error:
        identity['native_library'] = {'error': f'{type(error).__name__}: {error}'}
    try:
        document = pdfium.PdfDocument(_probe_pdf())
        try:
            page = document[0]
            fonts = []
            for item in page.get_objects(filter=[raw.FPDF_PAGEOBJ_TEXT]):
                font = raw.FPDFTextObj_GetFont(item)
                name = ctypes.create_string_buffer(512)
                (getattr(raw, 'FPDFFont_GetBaseFontName', None) or raw.FPDFFont_GetFontName)(font, name, 512)
                size = ctypes.c_size_t()
                raw.FPDFFont_GetFontData(font, None, 0, ctypes.byref(size))
                data = (ctypes.c_ubyte * size.value)()
                raw.FPDFFont_GetFontData(font, data, size.value, ctypes.byref(size))
                fonts.append({'requested': 'Arial (unembedded TrueType, WinAnsi)', 'name': name.value.decode('latin-1'),
                              'embedded': bool(raw.FPDFFont_GetIsEmbedded(font)), 'bytes': size.value,
                              'sha256': hashlib.sha256(bytes(data)).hexdigest()})
            page.close()
        finally:
            document.close()
        identity['unembedded_arial_substitute'] = fonts[0] if len(fonts) == 1 else {'error': f'{len(fonts)} text objects'}
    except Exception as error:
        identity['unembedded_arial_substitute'] = {'error': f'{type(error).__name__}: {error}'}
    return identity


def environment():
    packages = sorted(({'name': d.metadata['Name'], 'version': d.version} for d in importlib.metadata.distributions()), key=lambda d: d['name'].lower())
    return {'code': code_receipt(), 'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'packages': packages,
            'package_count': len(packages), **venv_footprint(), 'pdfium': pdfium_identity(),
            'environment': {k: v for k, v in os.environ.items() if k in (
                'DOCLING_ARTIFACTS_PATH', 'HF_HUB_OFFLINE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'GLOSSAPI_IMAGES_SCALE', 'GLOSSAPI_STAGE_PROFILE')}}


class PeakRSS:
    """Sample current RSS of this process and its descendants every 10ms.

    Absolute stage peak, not lifetime ru_maxrss or allocated-byte delta. Shared
    pages can be counted twice in the process-tree total. Very short peaks may
    be missed; the interval and measurement basis are always in the receipt.
    """
    def __enter__(self):
        self.pid = os.getpid()
        self.peak = self.own_peak = 0
        self.available = True
        self.basis = 'process + descendants current RSS'
        self.unavailable_reason = None
        self.stop = threading.Event()
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        if sys.platform == 'linux':
            seen = set()
            def rss(pid):
                if pid in seen: return 0
                seen.add(pid)
                try:
                    own = int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
                    children = set()
                    for task in Path(f'/proc/{pid}/task').glob('*/children'):
                        try: children.update(map(int, task.read_text().split()))
                        except (OSError, ValueError): pass
                    return own + sum(rss(child) for child in children)
                except (OSError, ValueError, IndexError): return 0
            try:
                own = int(Path(f'/proc/{self.pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            except (OSError, ValueError, IndexError): own = 0
            total = rss(self.pid)
        else:
            # BSD ps is present on macOS; no optional Python package or sysctl
            # process enumeration is needed. A restricted host may deny ps.
            try:
                rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='],
                                               text=True, stderr=subprocess.DEVNULL)
                processes = {int(pid): (int(parent), int(kib)*1024)
                             for pid, parent, kib in (line.split() for line in rows.splitlines())}
                own = processes[self.pid][1]
                descendants = {self.pid}
                while True:
                    found = {pid for pid, (parent, _) in processes.items() if parent in descendants}
                    if found <= descendants: break
                    descendants.update(found)
                total = sum(processes[pid][1] for pid in descendants)
                self.basis = 'BSD ps current process + descendants RSS (KiB to bytes)'
            except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
                self.available = False
                self.unavailable_reason = f'{type(exc).__name__}: {exc}'
                self.basis = 'unavailable; zero is not a memory measurement'
                return
        self.peak = max(self.peak, total)
        self.own_peak = max(self.own_peak, own)

    def run(self):
        while not self.stop.wait(.01):
            self.sample()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
