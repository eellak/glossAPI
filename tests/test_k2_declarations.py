"""K2 batch 2a small declarations (P1, P3, P6, P7, P8) stay consistent across the files that repeat them."""
import re
from pathlib import Path
import tomllib

REPO = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((REPO / 'pyproject.toml').read_text())


def test_p1_pandas_floor_is_the_qualified_line_everywhere():
    assert 'pandas>=3.0.0' in PROJECT['project']['dependencies']
    for stage in ('clean', 'export', 'sections'):
        text = (REPO / 'src/glossapi/stages' / f'{stage}.py').read_text()
        assert re.findall(r"'pandas[^']*'", text) == ["'pandas>=3.0.0'"], stage


def test_p3_filelock_floor():
    assert 'filelock>=4.2.4' in PROJECT['project']['dependencies']


def test_p6_docs_profile_lock():
    lock = (REPO / 'dependency_setup/requirements-docs.lock').read_text()
    pins = dict(re.findall(r'^([a-z0-9][a-z0-9_.-]*)==([^\s\\]+)', lock, re.M))
    assert pins['mkdocs'] == '1.6.1' and pins['mkdocs-material'] == '9.8.0'
    blocks = re.split(r'\n(?=[a-z0-9])', lock.split('\n', 3)[-1].strip())
    assert all('--hash=sha256:' in block for block in blocks if '==' in block)
    assert (REPO / 'dependency_setup/extra-docs.txt').read_text().split() == ['mkdocs>=1.6.1', 'mkdocs-material>=9.8.0']


def test_p7_phase_clean_maturin_fallback_matches_the_crates():
    fallback = set(re.findall(r'"(maturin[<>=][^"]*)"', (REPO / 'src/glossapi/corpus/phase_clean.py').read_text()))
    crates = {tomllib.loads(path.read_text())['build-system']['requires'][0]
              for path in (REPO / 'rust').glob('*/pyproject.toml')}
    assert fallback == crates == {'maturin>=1.15.0,<2.0'}
    assert 'maturin==1.15.0' in (REPO / 'dependency_setup/requirements-tooling.txt').read_text().split()


def test_p8_pip_is_pinned_in_the_build_profile():
    lines = (REPO / 'dependency_setup/requirements-tooling.txt').read_text().split()
    assert 'pip==26.2.2' in lines
