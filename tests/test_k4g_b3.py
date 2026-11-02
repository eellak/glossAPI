"""K4g batch B3: native API marker (INTEG-G1), canonical requirement text (K3B-L2), live install output (K3B-L5)."""
from __future__ import annotations

import importlib.machinery
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
REQUIRED_NATIVES = ('glossapi_rs_cleaner', 'glossapi_rs_noise')


def _child(code: str, *paths: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([*map(str, paths), str(REPO / 'src')]), 'PYTHONDONTWRITEBYTECODE': '1'}
    return subprocess.run([sys.executable, '-B', '-c', textwrap.dedent(code)], capture_output=True, text=True, env=env, timeout=120)


def _fake_native(root: Path, name: str, payload: bytes) -> Path:
    """A maturin-style package whose extension file holds ``payload`` (never imported: detection only reads it)."""
    package = root / name
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('raise ImportError("a fake native module must never be imported")\n')
    binary = package / (name + importlib.machinery.EXTENSION_SUFFIXES[0])
    binary.write_bytes(b'\x7fELF fake binary ' + payload + b' trailing bytes')
    return binary


# ------------------------------------------------------------------------------------------------- INTEG-G1 ---

def test_rust_markers_match_the_levels_capabilities_require():
    """Raising a level on one side only is caught: the crate marker and NATIVE_API_LEVELS agree."""
    from glossapi.capabilities import NATIVE_API_LEVELS
    assert set(NATIVE_API_LEVELS) == set(REQUIRED_NATIVES)
    for name, level in NATIVE_API_LEVELS.items():
        source = (REPO / 'rust' / name / 'src' / 'lib.rs').read_text(encoding='utf-8')
        assert re.findall(r'NATIVE_API_MARKER: &str = "GLOSSAPI_NATIVE_API:([a-z_]+):(\d+);"', source) == [(name, str(level))]
        assert 'm.add("__glossapi_native_api__", NATIVE_API_MARKER)' in source


def test_marker_level_parsing_is_exact():
    from glossapi.capabilities import _marker_level
    assert _marker_level(b'xx GLOSSAPI_NATIVE_API:glossapi_rs_cleaner:7; yy', 'glossapi_rs_cleaner') == 7
    assert _marker_level(b'GLOSSAPI_NATIVE_API:glossapi_rs_noise:1;', 'glossapi_rs_cleaner') is None
    assert _marker_level(b'GLOSSAPI_NATIVE_API:glossapi_rs_cleaner:1', 'glossapi_rs_cleaner') is None  # unterminated
    assert _marker_level(b'GLOSSAPI_NATIVE_API:glossapi_rs_cleaner:1;GLOSSAPI_NATIVE_API:glossapi_rs_cleaner:2;',
                         'glossapi_rs_cleaner') is None  # ambiguous


def test_installed_required_natives_carry_the_required_level():
    """The natives this test session loads were built from this source (a stale overlay fails here)."""
    import importlib
    import importlib.util
    from glossapi.capabilities import NATIVE_API_LEVELS, native_api_level, native_module_status
    for name in REQUIRED_NATIVES:
        if importlib.util.find_spec(name) is None:
            pytest.skip(f'{name} is not installed in this environment')
        assert native_api_level(name) == NATIVE_API_LEVELS[name], native_module_status(name, True)
        module = importlib.import_module(name)
        extension = getattr(module, name, module)  # maturin package: the extension is the same-named submodule
        assert extension.__glossapi_native_api__ == f'GLOSSAPI_NATIVE_API:{name}:{NATIVE_API_LEVELS[name]};'


@pytest.mark.parametrize('payloads,available', [
    ({'glossapi_rs_cleaner': b'', 'glossapi_rs_noise': b''}, False),  # builds that predate the marker
    ({'glossapi_rs_cleaner': b'GLOSSAPI_NATIVE_API:glossapi_rs_cleaner:0;',
      'glossapi_rs_noise': b'GLOSSAPI_NATIVE_API:glossapi_rs_noise:1;'}, False),  # one older build
    ({'glossapi_rs_cleaner': b'GLOSSAPI_NATIVE_API:glossapi_rs_cleaner:1;',
      'glossapi_rs_noise': b'GLOSSAPI_NATIVE_API:glossapi_rs_noise:1;'}, True),
])
def test_stale_native_makes_clean_unavailable_in_capabilities_plan_and_doctor(tmp_path, payloads, available):
    for name, payload in payloads.items():
        _fake_native(tmp_path / 'site', name, payload)
    done = _child('''
        import json, sys
        from glossapi.capabilities import stage_capability
        from glossapi.doctor import doctor_view
        clean = stage_capability('clean')
        doctor = next(item for item in doctor_view(%r)['checks'] if item['check_id'] == 'stage:clean')
        print(json.dumps(dict(clean=clean, doctor=doctor,
              imported=sorted(name for name in sys.modules if name.startswith('glossapi_rs_')))))
    ''' % str(tmp_path / 'corpus'), tmp_path / 'site')
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.splitlines()[-1])
    clean, doctor = result['clean'], result['doctor']
    assert result['imported'] == [], 'detection imported a native module'
    assert clean['available'] is available
    rows = {row['module']: row for row in clean['native_modules']}
    stale = [name for name, payload in payloads.items() if payload != f'GLOSSAPI_NATIVE_API:{name}:1;'.encode()]
    for name in REQUIRED_NATIVES:
        row = rows[name]
        assert row['required_api_level'] == 1
        assert row['status'] == ('missing' if name in stale else 'present')
        assert (row['detail'] or '').startswith('stale native module') is (name in stale)
    if available:
        assert clean['reason_code'] is None and doctor['status'] == 'ok'
    else:
        assert clean['reason_code'] == 'missing_native_module'
        assert clean['install_commands'] == ['python install_glossapi.py core']
        assert doctor['status'] == 'error' and all(name in doctor['message'] for name in stale)


def test_capability_fingerprint_changes_when_a_native_goes_stale(tmp_path):
    for label, level in (('current', 1), ('stale', 0)):
        for name in REQUIRED_NATIVES:
            _fake_native(tmp_path / label, name, f'GLOSSAPI_NATIVE_API:{name}:{level};'.encode())
    code = '''
        from glossapi.capabilities import capability_fingerprint, stage_capability
        print(capability_fingerprint([stage_capability('clean')]))
    '''
    prints = [_child(code, tmp_path / label) for label in ('current', 'stale')]
    assert all(done.returncode == 0 for done in prints), [done.stderr[-2000:] for done in prints]
    assert prints[0].stdout != prints[1].stdout  # a plan made against a stale build is refused after the rebuild


# -------------------------------------------------------------------------------------------------- K3B-L2 ---

def test_installed_metadata_lines_equal_the_source_tree_lines(monkeypatch):
    """A wheel's Requires-Dist (setuptools/packaging normal form) yields the same requirement text, hence the same lock
    input hash and install commands, as the source tree."""
    packaging_requirements = pytest.importorskip('packaging.requirements')
    from glossapi import package_catalog as catalog
    tree = catalog._from_source_tree(catalog.source_tree())
    requires = [str(packaging_requirements.Requirement(line)) for line in tree.core]
    for extra, lines in tree.extras.items():
        for line in lines:
            requirement, _, marker = line.partition(';')
            clause = f'extra == "{extra}"'
            separator = ' ; ' if '@' in requirement else '; '  # PEP 508: a URL needs whitespace before ';'
            requires.append(str(packaging_requirements.Requirement(
                requirement.strip() + separator + (f'({marker.strip()}) and {clause}' if marker.strip() else clause))))
    assert any(re.search(r'<[^,]*,>=', line) for line in requires), 'the emulated metadata must reorder some specifiers'

    class Metadata:
        def get_all(self, key):
            return list(tree.extras) if key == 'Provides-Extra' else None

    class Distribution:
        metadata = Metadata()
        def __init__(self):
            self.requires = requires
        def locate_file(self, _):
            return Path('/site-packages')

    monkeypatch.setattr(catalog.importlib.metadata, 'distribution', lambda name: Distribution())
    installed = catalog._from_metadata()
    assert installed.core == tree.core
    assert installed.extras == tree.extras
    targets = (catalog.measured() or {}).get('targets') or {}
    for target in targets.values():
        for profile in ('core', 'download', 'browser'):
            assert catalog.inputs_sha256(installed, profile, None, target) == catalog.inputs_sha256(tree, profile, None, target)


def test_canonical_requirement_keeps_source_lines_and_orders_specifiers():
    from glossapi import package_catalog as catalog
    tree = catalog._from_source_tree(catalog.source_tree())
    with open(REPO / 'pyproject.toml', 'rb') as handle:
        import tomllib
        written = tomllib.load(handle)['project']['dependencies']
    assert list(tree.core) == written  # the source tree's own text passes through unchanged
    assert catalog.canonical_requirement('aiohttp<4,>=3.14.5') == 'aiohttp>=3.14.5,<4'
    assert catalog.canonical_requirement('x[a]!=2,<3,>=1; sys_platform == "darwin"') == 'x[a]>=1,<3,!=2; sys_platform == "darwin"'
    assert catalog.canonical_requirement('pkg @ https://example.org/pkg.whl') == 'pkg @ https://example.org/pkg.whl'


# -------------------------------------------------------------------------------------------------- K3B-L5 ---

def test_install_command_output_reaches_stderr_while_the_command_runs(tmp_path, monkeypatch):
    """The first line is on stderr before the command exits: the child waits for a file the parent writes only after
    it has seen that line (a captured-then-written run would time out and fail)."""
    from glossapi import install_plan
    seen = tmp_path / 'seen'
    lines: list[str] = []

    class Stderr:
        def write(self, text):
            lines.append(text)
            if 'first line' in text:
                seen.write_text('1')
            return len(text)

        def flush(self):
            pass

    monkeypatch.setattr(install_plan.sys, 'stderr', Stderr())
    child = textwrap.dedent(f'''
        import pathlib, sys, time
        print("first line", flush=True)
        deadline = time.monotonic() + 20
        while not pathlib.Path({str(seen)!r}).exists():
            if time.monotonic() > deadline:
                print("parent never saw the first line", file=sys.stderr, flush=True)
                sys.exit(3)
            time.sleep(0.02)
        print("error line", file=sys.stderr, flush=True)
        for index in range({install_plan.OUTPUT_TAIL_LINES + 5}):
            print("line", index)
    ''')
    snapshot = dict(distributions={}, fingerprint='a' * 64, stages={})
    monkeypatch.setattr(install_plan, '_snapshot', lambda python: snapshot)
    monkeypatch.setattr(install_plan, '_installer', lambda choice, python: dict(
        available=True, kind='pip', check=[sys.executable, '-c', 'pass']))

    def plan(*argvs):
        return dict(installer=dict(python=sys.executable), warnings=[], commands=[dict(argv=list(argv)) for argv in argvs])

    started = time.monotonic()
    data, failure = install_plan.execute_install(plan([sys.executable, '-c', child]))
    assert failure is None, ''.join(lines)[-2000:]
    assert time.monotonic() - started < 20
    assert ''.join(lines).startswith('first line\nerror line\n')
    tail = data['executed']['steps'][0]['output_tail']
    assert len(tail) == install_plan.OUTPUT_TAIL_LINES and tail[-1] == f'line {install_plan.OUTPUT_TAIL_LINES + 4}'
    data, failure = install_plan.execute_install(plan([sys.executable, '-c', 'import sys; print("bye"); sys.exit(7)'],
                                                      [sys.executable, '-c', 'print("never")']))
    assert failure is not None and failure.code == 'install_command_failed'
    assert [(step['returncode'], step['output_tail']) for step in data['executed']['steps']] == [(7, ['bye'])]
