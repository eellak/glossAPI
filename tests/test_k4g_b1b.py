"""K4g batch B1b: K3F-G1 and the LS7-G01 guide row (one registry for the extreme-mode disk-guard refusals).
The OCR census fix is covered in tests/test_ocr_same_basename.py."""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_k3f_g1_guide_rows_come_from_the_disk_guard_registry():
    from glossapi.llm_guide import _cli_errors
    from glossapi.stages import disk_budget
    rows = _cli_errors()
    for code, row in disk_budget.error_contracts().items():
        assert rows[code] == row
    meaning = rows['disk_budget_required']['meaning']
    assert all(field in meaning for field in ('disk_budget_bytes', 'max_source_bytes', 'max_attempt_bytes'))
    assert rows['disk_guard_unavailable']['exit_code'] == 3


def test_k3f_g1_every_registered_code_is_raised_and_exits_3(tmp_path, capsys):
    from glossapi.cli import main
    from glossapi.stages import disk_budget
    source = '\n'.join(path.read_text() for path in (REPO/'src/glossapi').rglob('*.py') if path.name != 'llm_guide.py')
    for code in disk_budget.ERRORS:
        assert re.search(r"Error\(\s*['\"]" + code + r"['\"]", source), code  # constructed by a typed error
    assert main(['plan', '--corpus', str(tmp_path/'corpus'), '--files', str(tmp_path/'a.pdf'), '--memory-saving', 'extreme', '--json']) == 3
    assert json.loads(capsys.readouterr().out)['errors'][0]['code'] == 'disk_budget_required'
