"""Copy this directory; supply a warm cache and local layout path and run with pytest."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from run_smoke import normalized

ROOT = Path(__file__).resolve().parent

@pytest.mark.greekocr_integration
@pytest.mark.skipif(not all(os.environ.get(n) for n in ('GREEKOCR_CACHE_DIR','OCR_LAYOUT_DIR')),
                    reason='Greek OCR smoke requires prewarmed plugin and layout caches')
def test_docling_greekocr_cpu(tmp_path):
    from pypdf import PdfReader
    expected = json.loads((ROOT/'expected.json').read_text())
    pdf = ROOT/'polytonic_scan.pdf'
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == expected['fixture_sha256']
    reader = PdfReader(pdf)
    assert len(reader.pages) == 2
    assert all(not p.extract_text() and len(p.images) == 1 for p in reader.pages)
    names = ['GREEKOCR_CACHE_DIR', 'OCR_LAYOUT_DIR']
    missing = [n for n in names if not os.environ.get(n)]
    assert not missing, 'Provide local model paths: '+', '.join(missing)
    output = tmp_path/'run'
    command = [sys.executable, str(ROOT/'run_smoke.py'),
        '--layout-dir', os.environ['OCR_LAYOUT_DIR'], '--output', str(output)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout+'\n'+proc.stderr
    actual = json.loads((output/'actual.json').read_text())
    assert actual['kept_lines'] == expected['kept_lines'] == 6
    assert actual['dropped_lines'] == expected['dropped_lines'] == 0
    assert actual['pages'] == expected['pages']
    assert hashlib.sha256(normalized(actual['pages'])).hexdigest() == expected['normalized_sha256']
    assert actual['normalized_sha256'] == expected['normalized_sha256']
    assert actual['versions']['docling'] == '2.81.0'
    assert actual['detector']['sha256'] == 'd73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e'
    assert actual['detector']['repo_id'] == 'PaddlePaddle/PP-OCRv6_small_det_onnx'
    assert actual['model_acquisition']['resolved_commit'] == '2a6b019be728e5be70cdbd981414c630b63fca08'
    assert actual['model_acquisition']['manifest_sha256'] == '85d455d2cae2b756ab17e7114a3de13d7d41426d1fa5093028a179cf0c7d1345'
    assert actual['options']['release_dir'] is None and actual['options']['checkpoint'] is None
    assert actual['options']['device'] == actual['options']['detector_device'] == 'cpu'
    calls = [json.loads(s) for s in (output/'plugin_calls.jsonl').read_text().splitlines()]
    assert [c['page'] for c in calls] == [1, 2]
    assert all(c['status'] == 'replaced' and c['detector']['requested_device'] == 'cpu' for c in calls)
