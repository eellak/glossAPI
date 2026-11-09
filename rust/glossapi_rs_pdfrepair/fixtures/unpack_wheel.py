"""Unpack a built wheel locally using atomic replacements for native libraries."""
from pathlib import Path
import os
import tempfile
import zipfile

root = Path(__file__).resolve().parents[1] / 'target/python'
wheel = max(root.parent.joinpath('wheels').glob('*.whl'), key=lambda p: p.stat().st_mtime)
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.endswith('/'):
            continue
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(archive.read(name))
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
print(root)
