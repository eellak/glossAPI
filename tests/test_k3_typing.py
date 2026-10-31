"""W19 (R65): pyright strict on the new K3 modules, staged baseline (no new errors) for touched cli.py/reader.py/retention.py.

Pyright is a dev/test tool only (dev group); the test skips where it is not installed.
"""
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_typing_gate_holds():
    pytest.importorskip('pyright')
    done = subprocess.run([sys.executable, str(REPO / 'tools/typing/check_typing.py')], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
    assert '"strict_errors": 0' in done.stdout and '"new_errors": 0' in done.stdout
