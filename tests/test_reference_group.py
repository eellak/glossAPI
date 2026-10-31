"""The `reference` dependency group: pins, inclusion in test/dev, and fail-not-skip when required."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET

import pytest

import reference_oracles
from reference_oracles import ENV, PINS, problem, required

REPO = Path(__file__).resolve().parents[1]


def _groups():
    return tomllib.loads((REPO / 'pyproject.toml').read_text())['dependency-groups']


def test_reference_pins_match_pyproject_and_default_groups_include_them():
    groups = _groups()
    pinned = {}
    for requirement in groups['reference']:
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)(\[[^\]]+\])?==([0-9][0-9A-Za-z.]*)', requirement)
        assert match, f'reference requirements must be exact pins: {requirement}'
        pinned[match[1].lower()] = match[3]
    assert pinned == {distribution.lower(): version for distribution, version in PINS.values()}
    assert {'include-group': 'reference'} in groups['test']
    assert {'include-group': 'test'} in groups['dev']
    flat = [entry for entry in groups['test'] if isinstance(entry, str)]
    # The oracles are pinned only once, in the reference group.
    assert not any(re.match(r'(tenacity|aiofiles|dask|pypdf|fonttools|scikit-learn)\b', entry, re.I) for entry in flat)
    assert not any(entry.lower().startswith('psutil') for entry in flat), 'psutil has no consumer'


def test_installed_reference_group_matches_the_pins_when_required():
    if not required():
        pytest.skip(f'{ENV} is not set; the reference group is optional in this environment')
    assert {module: problem(module) for module in PINS} == {module: None for module in PINS}


def _run_probe(tmp_path, env_value, body):
    probe = tmp_path / 'test_probe.py'
    probe.write_text('import importlib.abc, sys\n'
                     'class Absent(importlib.abc.MetaPathFinder):\n'
                     '    def find_spec(self, fullname, path=None, target=None):\n'
                     "        if fullname.split('.')[0] == 'tenacity':\n"
                     "            raise ModuleNotFoundError('injected absent oracle', name='tenacity')\n"
                     'import reference_oracles\n'
                     f'{body}\n')
    env = {key: value for key, value in os.environ.items() if key != ENV}
    if env_value is not None:
        env[ENV] = env_value
    env['PYTHONPATH'] = os.pathsep.join([str(Path(reference_oracles.__file__).parent), env.get('PYTHONPATH', '')])
    report = tmp_path / 'report.xml'
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--rootdir', str(tmp_path),
                    '-c', os.devnull, str(probe), f'--junitxml={report}'], cwd=tmp_path, env=env,
                   capture_output=True, text=True)
    outcomes = {}
    for case in ET.parse(report).getroot().iter('testcase'):
        kinds = [child.tag for child in case if child.tag in ('failure', 'error', 'skipped')]
        message = ' '.join((child.get('message') or '') + (child.text or '') for child in case)
        outcomes[case.get('name')] = (kinds[0] if kinds else 'passed', message)
    return outcomes


MISSING = ('def test_missing():\n'
           '    sys.meta_path.insert(0, Absent()); sys.modules.pop("tenacity", None)\n'
           '    reference_oracles.oracle("tenacity")\n')
WRONG_VERSION = ('def test_wrong_version(monkeypatch):\n'
                 '    monkeypatch.setitem(reference_oracles.PINS, "pypdf", ("pypdf", "0.0.1"))\n'
                 '    reference_oracles.oracle("pypdf")\n')


@pytest.mark.parametrize('flag', ['1', 'true'])
def test_missing_or_wrong_oracle_fails_when_required(tmp_path, flag):
    pytest.importorskip('pypdf')
    outcomes = _run_probe(tmp_path, flag, MISSING + WRONG_VERSION)
    assert outcomes['test_missing'][0] == 'failure' and 'reference' in outcomes['test_missing'][1]
    assert outcomes['test_wrong_version'][0] == 'failure' and 'pins 0.0.1' in outcomes['test_wrong_version'][1]


@pytest.mark.parametrize('flag', [None, '0', ''])
def test_missing_oracle_skips_when_optional(tmp_path, flag):
    outcomes = _run_probe(tmp_path, flag, MISSING)
    assert outcomes['test_missing'][0] == 'skipped'
