"""Dependency-light receipts shared by the node harness."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
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


def inventory(root):
    """All pipeline files, including state; logs are retained separately, not products."""
    root = Path(root)
    return {str(p.relative_to(root)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()
            and p.suffix not in ('.log', '.lock') and p.name != '.harness-direct-paths'}


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


def environment():
    packages = sorted(({'name': d.metadata['Name'], 'version': d.version} for d in importlib.metadata.distributions()), key=lambda d: d['name'].lower())
    size = subprocess.check_output(['du', '-sk', sys.prefix], text=True).split()[0]
    return {'code': code_receipt(), 'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'packages': packages,
            'package_count': len(packages), 'venv_disk_bytes': int(size) * 1024,
            'environment': {k: v for k, v in os.environ.items() if k in (
                'DOCLING_ARTIFACTS_PATH', 'HF_HUB_OFFLINE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'GLOSSAPI_IMAGES_SCALE')}}


class PeakRSS:
    """Sample current RSS of this process and its descendants every 10ms.

    Absolute stage peak, not lifetime ru_maxrss or allocated-byte delta. Shared
    pages can be counted twice in the process-tree total. Very short peaks may
    be missed; the interval and measurement basis are always in the receipt.
    """
    def __enter__(self):
        self.pid = os.getpid()
        self.peak = self.own_peak = 0
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
            import psutil  # optional, only needed outside the Linux measurement node
            process = psutil.Process(self.pid)
            own = process.memory_info().rss
            total = own
            for child in process.children(recursive=True):
                try: total += child.memory_info().rss
                except psutil.Error: pass
        self.peak = max(self.peak, total)
        self.own_peak = max(self.own_peak, own)

    def run(self):
        while not self.stop.wait(.01):
            self.sample()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
