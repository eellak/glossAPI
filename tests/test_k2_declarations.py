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


def test_p13_setuptools_rust_is_pinned_and_the_build_floor_is_the_tested_version():
    """K2 native (P13): the tooling pins setuptools-rust 1.13.0 and the root build-system floor is that version."""
    assert 'setuptools-rust==1.13.0' in (REPO / 'dependency_setup/requirements-tooling.txt').read_text().split()
    assert 'setuptools-rust>=1.13.0,<2' in PROJECT['build-system']['requires']


def test_u2a_native_declarations():
    """K2 native (U2a): PyO3 0.30.1 with abi3-py311 in all three extensions, a declared MSRV and the pinned
    toolchain; every extension pyproject builds with maturin >=1.15.0,<2 (P7)."""
    toolchain = tomllib.loads((REPO / 'rust-toolchain.toml').read_text())['toolchain']
    assert toolchain['channel'] == '1.99.1'
    for crate in ('glossapi_rs_cleaner', 'glossapi_rs_noise', 'glossapi_rs_pdfrepair', 'glossapi_rs_common'):
        manifest = tomllib.loads((REPO / 'rust' / crate / 'Cargo.toml').read_text())
        assert manifest['package']['rust-version'] == '1.85' and manifest['package']['edition'] == '2021'
        dependencies = manifest.get('dependencies', {})
        assert 'lazy_static' not in dependencies and 'once_cell' not in dependencies
        if crate != 'glossapi_rs_common':
            pyo3 = dependencies['pyo3']
            assert pyo3['version'] == '0.30.1' and 'abi3-py311' in pyo3['features'] and 'abi3-py38' not in pyo3['features']
            assert manifest['build-dependencies']['pyo3-build-config'] == '=0.30.1'
            requires = tomllib.loads((REPO / 'rust' / crate / 'pyproject.toml').read_text())['build-system']['requires']
            assert requires == ['maturin>=1.15.0,<2.0']

