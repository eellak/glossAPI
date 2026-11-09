"""K3b decision interface: the package catalog (`capabilities --catalog`), `install --device`, the LLM guide.

Acceptance map (PLAN rev 5 section 2.4; K3b brief items 1-4; register R25, R27, R28):
- catalog: every profile (core + every pyproject extra) with what it adds (from the stage adapters), devices, measured
  install size and package count (from the locks, current for today's requirements), licences, installed state;
- install: dry run by default (nothing runs); the command list equals exactly the requested profiles' requirements,
  computed here independently from pyproject and the extra files; conflicts fail before anything runs; --yes runs
  exactly those commands and reports the capability re-check;
- guide: schema-valid, equal to docs/llm_guide.md, and consistent with the schemas, settings contract, catalog,
  stage capabilities, manifest rules, CLI parser and measured costs; a disagreement fails;
- typed errors and exit codes documented in docs/cli_json_schema.md.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap
import tomllib

import pytest

from glossapi.cli import main

REPO = Path(__file__).resolve().parents[1]
jsonschema = pytest.importorskip('jsonschema')
from cli_schema_validation import validate  # noqa: E402  (needs jsonschema)

PYPI = 'https://pypi.org/simple'


def cli(*arguments):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main([*arguments, '--json'])
    envelope = json.loads(buffer.getvalue())
    validate(envelope)
    return code, envelope


def env_with(*paths):
    return {**os.environ, 'PYTHONPATH': os.pathsep.join([*map(str, paths), str(REPO / 'src'), str(REPO / 'tests'),
                                                          *filter(None, [os.environ.get('PYTHONPATH')])])}


# ------------------------------------------------- independent requirement reading (not the catalog's code) ---

def _file_lines(path):
    return [line.split(' #', 1)[0].strip() for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#') and line.split(' #', 1)[0].strip()]


def declared():
    project = tomllib.loads((REPO / 'pyproject.toml').read_text())
    extras = {name: [line for file in entry['file'] for line in _file_lines(REPO / file)]
              for name, entry in project['tool']['setuptools']['dynamic']['optional-dependencies'].items()}
    return project['project']['dependencies'], extras


def expand(extras, name, seen=None):
    seen = seen if seen is not None else []
    lines = []
    for line in extras[name]:
        if line.startswith('glossapi['):
            for child in line[len('glossapi['):-1].split(','):
                if child not in seen:
                    seen.append(child)
                    lines += [item for item in expand(extras, child, seen) if item not in lines]
        elif line not in lines:
            lines.append(line)
    return lines


def expected_commands(packages, device, system, machine, python=sys.executable):
    from glossapi.scripts import install_glossapi as installer
    core, extras = declared()
    lines = []
    for name in packages:
        for line in (core if name == 'core' else expand(extras, name)):
            if line not in lines:
                lines.append(line)
    bound = any(name in installer.TORCH_PACKAGES or any(line.startswith('glossapi[layout]') for line in extras.get(name, []))
                for name in packages)
    prefix = [python, '-m', 'pip', 'install']
    commands = []
    if bound:
        index, suffix = installer.torch_build(device, system, machine)
        pins = [f'torch=={installer.TORCH_VERSION}{suffix}', f'torchvision=={installer.TORCHVISION_VERSION}{suffix}']
        commands.append([*prefix, *pins, '--index-url', index])
        lines = pins + [line for line in lines if not line.split('=')[0].split('[')[0].strip().lower() in ('torch', 'torchvision')]
    if lines:
        commands.append([*prefix, '--index-url', PYPI, *lines])
    return commands


@pytest.fixture
def no_subprocess(monkeypatch):
    import glossapi.install_plan as plan
    monkeypatch.setattr(plan.subprocess, 'run', lambda *a, **k: pytest.fail('a command ran: ' + repr(a[:1])))


@pytest.fixture
def pip_present(monkeypatch):
    import glossapi.install_plan as plan
    real = plan.module_present
    monkeypatch.setattr(plan, 'module_present', lambda name: True if name == 'pip' else real(name))


# ---------------------------------------------------------------------------------------------- catalog ---

def test_catalog_lists_every_profile_with_current_measured_locks():
    code, envelope = cli('capabilities', '--catalog')
    assert code == 0 and not envelope['warnings'], envelope['warnings']
    catalog = envelope['data']['catalog']
    _, extras = declared()
    assert [row['name'] for row in catalog['profiles']] == ['core', *extras]
    from glossapi.package_catalog import lock_packages, measured
    data = measured()
    for row in catalog['profiles']:
        assert row['purpose'], row['name']
        assert row['measurement']['state'] == 'current', (row['name'], row['measurement'],
                                                          'requirements changed: re-run tools/catalog/measure_profiles.py')
        assert row['install_size'] and all(size['status'] == 'resolved' for size in row['install_size'])
        targets = {size['target'] for size in row['install_size']}
        assert targets == {'linux-x86_64', 'macos-arm64'}, row['name']
        for size in row['install_size']:
            lock = data['locks'][size['lock']]
            packages = lock_packages(data, lock)
            assert size['packages'] == len(packages) and size['lock_sha256'] == lock['lock_sha256']
            assert size['download_bytes'] == sum(item['download_bytes'] or 0 for item in packages)
            core = {item['name'] for item in lock_packages(data, data['locks'][f"core@any@{size['target']}"])}
            assert size['added_names'] == sorted(item['name'] for item in packages if row['name'] == 'core' or item['name'] not in core)
            assert size['unmeasured'] == sorted(item['name'] for item in packages if item['installed_bytes'] is None or item['download_bytes'] is None)
        if row['device_bound']:
            assert {size['device'] for size in row['install_size']} == {'cpu', 'cuda126', 'cuda132', 'mps'}
            assert all(device['supported'] for device in row['devices'])
        assert 'installed' in row and row['installed']['state'] in ('installed', 'partial', 'missing')
    by_name = {row['name']: row for row in catalog['profiles']}
    # Measured facts that matter to the owner: the OCR toolkit has no wheel and declares no licence at the pin.
    assert all(size['unmeasured'] == ['greek-ocr-toolkit'] for size in by_name['ocr']['install_size'])
    assert any('not declared' in item for item in by_name['ocr']['licence']['roots'])
    assert by_name['annotate']['install_size'][0]['added_packages'] == 0
    assert by_name['docling']['alias_of'] == 'layout' and by_name['docling']['install_size'] == by_name['layout']['install_size']


def test_catalog_requirements_are_the_declared_extra_files_expanded():
    from glossapi.package_catalog import catalog_view
    core, extras = declared()
    catalog = catalog_view()
    assert catalog['requirements_source']['kind'] == 'source_tree'
    for row in catalog['profiles']:
        assert row['requirements'] == (list(core) if row['name'] == 'core' else expand(extras, row['name'])), row['name']
    browser = next(row for row in catalog['profiles'] if row['name'] == 'browser')
    assert browser['includes'] == ['download'] and 'aiohttp>=3.14.5,<4' in browser['requirements']


def test_installed_metadata_source_matches_the_files(monkeypatch, tmp_path):
    """A wheel install reads Requires-Dist; built from the same files, it yields the same profiles."""
    import glossapi.package_catalog as catalog
    core, extras = declared()
    lines = [f'Requires-Dist: {line}' for line in core]
    for name, items in extras.items():
        for item in items:
            requirement, _, marker = item.partition(';')
            lines.append(f'Requires-Dist: {requirement.strip()}; ' + (f'{marker.strip()} and ' if marker else '') + f'extra == "{name}"')
    info = tmp_path / 'glossapi-9.9.9.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: glossapi\nVersion: 9.9.9\n'
                                   + ''.join(f'Provides-Extra: {name}\n' for name in extras) + '\n'.join(lines) + '\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(catalog, 'source_tree', lambda: None)
    source = catalog.requirement_source()
    assert source.kind == 'installed_metadata' and list(source.core) == list(core) and list(source.extras) == list(extras)
    for name in extras:
        assert catalog.expanded_requirements(source, name) == expand(extras, name), name


def test_catalog_enables_come_from_the_stage_adapters():
    from glossapi.capabilities import declared_requirements
    from glossapi.package_catalog import STAGE_VARIANTS, catalog_view
    rows = {row['name']: row for row in catalog_view(installed=False)['profiles']}
    for stage, pairs in STAGE_VARIANTS:
        groups = {requirement.package_group for requirement in declared_requirements(stage, dict(pairs))} or {'core'}
        for group in groups:
            assert any(item['stage'] == stage and item['settings'] == dict(pairs) for item in rows[group]['enables']), (group, stage, pairs)
    assert {(item['stage'], tuple(item['settings'].items())) for item in rows['layout']['enables']} == {
        ('extract', (('phase1_backend', 'docling'),)), ('ocr', ()), ('math', ())}
    assert [item['settings'] for item in rows['browser']['enables']] == [{'download_mode': 'browser'}]
    assert [item['settings'] for item in rows['annotate-legacy']['enables']] == [{'classifier_model_format': 'joblib'}]


def test_catalog_installed_state_follows_distribution_metadata(monkeypatch, tmp_path):
    from glossapi.package_catalog import catalog_view
    before = {row['name']: row['installed']['state'] for row in catalog_view()['profiles']}
    assert before['docs'] == 'missing'
    info = tmp_path / 'mkdocs-1.6.1.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: mkdocs\nVersion: 1.6.1\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    docs = next(row for row in catalog_view()['profiles'] if row['name'] == 'docs')
    assert docs['installed']['state'] == 'partial'
    assert [row['status'] for row in docs['installed']['requirements']] == ['satisfied', 'missing']


def test_changed_requirements_make_the_measurement_stale_and_warn(monkeypatch):
    import glossapi.package_catalog as catalog
    real = catalog.requirement_source()
    changed = catalog.RequirementSource(real.kind, real.location, real.core, {**real.extras, 'download': (*real.extras['download'], 'rich>=13')})
    monkeypatch.setattr(catalog, 'requirement_source', lambda: changed)
    code, envelope = cli('capabilities', '--catalog')
    rows = {row['name']: row for row in envelope['data']['catalog']['profiles']}
    assert code == 0 and rows['download']['measurement']['state'] == 'stale' and rows['browser']['measurement']['state'] == 'stale'
    assert rows['layout']['measurement']['state'] == 'current'
    assert [warning['code'] for warning in envelope['warnings']] == ['catalog_measurement_stale']


def test_catalog_guide_and_install_preview_import_no_backend():
    """Imports of torch/docling/pandas/OCR backends raise: none is attempted by catalog, guide or install dry run."""
    code = textwrap.dedent('''
        import contextlib, importlib.abc, io, json, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in ('torch', 'docling', 'docling_core', 'pandas', 'pyarrow', 'numpy', 'greekocr', 'onnxruntime',
                                          'cv2', 'transformers', 'sklearn', 'jsonschema', 'pip'):
                    raise ImportError('blocked ' + name)
        sys.meta_path.insert(0, Block())
        from glossapi.cli import main
        codes = []
        for argv in (['capabilities', '--catalog'], ['guide'], ['install', '--device', 'cpu', '--package', 'layout,ocr']):
            with contextlib.redirect_stdout(io.StringIO()):
                codes.append(main([*argv, '--json']))
        print(json.dumps(codes))
    ''')
    done = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, env=env_with())
    assert done.returncode == 0, done.stderr[-3000:]
    assert json.loads(done.stdout.splitlines()[-1]) == [0, 0, 0]


# ---------------------------------------------------------------------------------------------- install ---

CASES = [
    ('Linux', 'x86_64', 'cpu', ['download']),
    ('Linux', 'x86_64', 'cpu', ['browser']),
    ('Linux', 'x86_64', 'cpu', ['layout']),
    ('Linux', 'x86_64', 'cuda126', ['ocr']),
    ('Linux', 'x86_64', 'cuda132', ['layout', 'download']),
    ('Linux', 'x86_64', 'cpu', ['docling']),
    ('Linux', 'x86_64', 'cpu', ['annotate-legacy', 'font-repair']),
    ('Darwin', 'arm64', 'mps', ['ocr']),
    ('Darwin', 'arm64', 'cpu', ['layout', 'ocr']),
    ('Darwin', 'arm64', 'cpu', ['docs']),
    ('Linux', 'x86_64', 'cpu', ['annotate']),
    ('Linux', 'x86_64', 'cpu', []),
]


@pytest.mark.parametrize('system,machine,device,packages', CASES)
def test_install_commands_are_exactly_the_requested_profiles(monkeypatch, no_subprocess, pip_present, system, machine, device, packages):
    import glossapi.install_plan as plan
    monkeypatch.setattr(plan, 'installed_version', lambda name: None)
    data = plan.plan_install([device], packages, system=system, machine=machine)
    commands = [command['argv'] for command in data['commands']]
    assert commands == expected_commands(packages, device, system, machine)
    assert all(command['shell'] == shlex.join(command['argv']) for command in data['commands'])
    flat = [token for command in commands for token in command]
    assert not any(token.startswith('glossapi') or token in ('-e', '--editable', '.') or token.startswith(str(REPO)) for token in flat)
    assert '--extra-index-url' not in flat
    _, extras = declared()
    requested = {line for name in packages for line in (expand(extras, name) if name != 'core' else [])}
    others = {line for name in extras if name not in packages for line in expand(extras, name)} - requested
    assert not (set(flat) & others - {'torch==2.14.1', 'torchvision==0.29.1'}), set(flat) & others
    assert data['mode'] == 'dry_run' and data['executed'] is None and data['changes'] is None


def test_install_dry_run_cli_executes_nothing_and_validates(no_subprocess):
    code, envelope = cli('install', '--device', 'cpu', '--package', 'download')
    assert code == 0 and envelope['command'] == 'install' and not envelope['errors']
    data = envelope['data']
    assert data['mode'] == 'dry_run' and [command['requirements'] for command in data['commands']] == [['aiohttp>=3.14.5,<4', 'pyyaml>=6.0']]
    assert data['would_change']['from_lock']['locks'] and data['requirements'][0]['package'] == 'download'


@pytest.mark.parametrize('argv,code_expected', [
    (('install', '--package', 'download'), 'device_required'),
    (('install', '--device', 'cpu', '--device', 'cuda126', '--package', 'layout'), 'conflicting_device_profiles'),
    (('install', '--device', 'cpu,mps', '--package', 'download'), 'conflicting_device_profiles'),
    (('install', '--device', 'gpu'), 'unknown_device'),
    (('install', '--device', 'cpu', '--package', 'download,vanilla'), 'unknown_package'),
    (('install', '--device', 'cpu', '--installer', 'conda'), 'invalid_request'),
])
def test_conflicting_or_unknown_choices_fail_before_anything_runs(no_subprocess, argv, code_expected):
    code, envelope = cli(*argv)
    assert code == 2 and envelope['errors'][0]['code'] == code_expected and envelope['data'] is None


@pytest.mark.parametrize('system,machine,device', [('Linux', 'x86_64', 'mps'), ('Darwin', 'arm64', 'cuda126'), ('Darwin', 'x86_64', 'cpu')])
def test_device_profile_this_platform_lacks_fails(monkeypatch, no_subprocess, system, machine, device):
    import glossapi.install_plan as plan
    monkeypatch.setattr(plan.platform, 'system', lambda: system)
    monkeypatch.setattr(plan.platform, 'machine', lambda: machine)
    code, envelope = cli('install', '--device', device, '--package', 'download')
    assert code == 2 and envelope['errors'][0]['code'] == 'device_unsupported_on_platform'


def test_package_that_does_not_support_the_device_fails(monkeypatch, no_subprocess):
    """A device profile whose lock did not resolve for a package is refused (the catalog says why)."""
    import copy
    import glossapi.package_catalog as catalog
    data = copy.deepcopy(catalog.measured())
    data['locks']['ocr@cuda132@linux-x86_64']['status'] = 'failed'
    monkeypatch.setattr(catalog, 'measured', lambda: data)
    import glossapi.install_plan as plan
    monkeypatch.setattr(plan, 'measured', lambda: data)
    monkeypatch.setattr(plan.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(plan.platform, 'machine', lambda: 'x86_64')
    code, envelope = cli('install', '--device', 'cuda132', '--package', 'download,ocr')
    assert code == 2 and envelope['errors'][0]['code'] == 'package_device_unsupported'
    assert envelope['errors'][0]['details']['package'] == 'ocr'
    code, envelope = cli('install', '--device', 'cuda132', '--package', 'download')
    assert code == 0  # download installs the same packages on any device profile


def test_installed_torch_of_another_device_profile_conflicts(monkeypatch, no_subprocess):
    import glossapi.install_plan as plan
    monkeypatch.setattr(plan.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(plan.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(plan, 'installed_version', lambda name: '2.14.1+cpu' if name == 'torch' else None)
    code, envelope = cli('install', '--device', 'cuda126', '--package', 'layout')
    assert code == 2 and envelope['errors'][0]['code'] == 'installed_device_profile_conflict'
    assert envelope['errors'][0]['details']['installed'] == '2.14.1+cpu'
    assert cli('install', '--device', 'cpu', '--package', 'layout')[0] == 0
    assert cli('install', '--device', 'cuda126', '--package', 'download')[0] == 0  # not device-bound: torch untouched


class FakeRun:
    """Records install commands; answers the capability re-check snapshots (before, after)."""

    def __init__(self, fail_at=None):
        self.calls, self.fail_at, self.snapshots = [], fail_at, 0

    def __call__(self, argv, **kwargs):
        if argv[1:2] == ['-c']:
            self.snapshots += 1
            after = self.snapshots > 1
            payload = dict(distributions={'aiohttp': '3.14.5'} if after else {}, fingerprint=('b' if after else 'a') * 64,
                           stages={'download': [after, None if after else 'missing_package'], 'extract': [True, None]})
            return subprocess.CompletedProcess(argv, 0, json.dumps(payload) + '\n', '')
        self.calls.append(argv)
        failed = self.fail_at is not None and len(self.calls) - 1 == self.fail_at
        return subprocess.CompletedProcess(argv, 7 if failed else 0, 'out\n', 'err\n' if failed else '')


def test_yes_runs_exactly_the_previewed_commands_and_reports_the_recheck(monkeypatch, pip_present):
    import glossapi.install_plan as plan
    fake = FakeRun()
    monkeypatch.setattr(plan.subprocess, 'run', fake)
    monkeypatch.setattr(plan, 'installed_version', lambda name: None)
    monkeypatch.setattr(plan.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(plan.platform, 'machine', lambda: 'x86_64')
    code, envelope = cli('install', '--device', 'cpu', '--package', 'layout', '--yes')
    assert code == 0, envelope['errors']
    data = envelope['data']
    previewed = expected_commands(['layout'], 'cpu', 'Linux', 'x86_64')
    assert fake.calls[:-1] == previewed and fake.calls[-1] == [sys.executable, '-m', 'pip', 'check']
    assert data['mode'] == 'executed' and [step['argv'] for step in data['executed']['steps']] == previewed
    assert data['changes']['added'] == ['aiohttp==3.14.5'] and data['changes']['stages'][0]['stage'] == 'download'
    assert data['changes']['fingerprint_before'] != data['changes']['fingerprint_after'] and fake.snapshots == 2


def test_yes_stops_at_the_first_failed_command(monkeypatch, pip_present):
    import glossapi.install_plan as plan
    fake = FakeRun(fail_at=0)
    monkeypatch.setattr(plan.subprocess, 'run', fake)
    monkeypatch.setattr(plan, 'installed_version', lambda name: None)
    monkeypatch.setattr(plan.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(plan.platform, 'machine', lambda: 'x86_64')
    code, envelope = cli('install', '--device', 'cpu', '--package', 'ocr', '--yes')
    assert code == 2 and envelope['errors'][0]['code'] == 'install_command_failed'
    assert len(fake.calls) == 1 and 'torch==2.14.1+cpu' in fake.calls[0]  # nothing after the torch command, no pip check
    assert envelope['data']['executed']['steps'][0]['returncode'] == 7 and envelope['data']['changes'] is not None


def test_yes_without_an_installer_is_a_typed_error(monkeypatch):
    import glossapi.install_plan as plan
    monkeypatch.setattr(plan, 'module_present', lambda name: False)
    monkeypatch.setattr(plan.shutil, 'which', lambda name: None)
    monkeypatch.setattr(plan.subprocess, 'run', lambda *a, **k: pytest.fail('ran'))
    code, envelope = cli('install', '--device', 'cpu', '--package', 'download')
    assert code == 0 and 'installer_unavailable' in [warning['code'] for warning in envelope['warnings']]
    code, envelope = cli('install', '--device', 'cpu', '--package', 'download', '--yes')
    assert code == 2 and envelope['errors'][0]['code'] == 'installer_unavailable'


def test_uv_installer_form(monkeypatch, no_subprocess):
    import glossapi.install_plan as plan
    monkeypatch.setattr(plan.shutil, 'which', lambda name: '/opt/uv' if name == 'uv' else None)
    data = plan.plan_install(['cpu'], ['download'], installer='uv', system='Linux', machine='x86_64')
    assert data['commands'][0]['argv'][:5] == ['/opt/uv', 'pip', 'install', '--python', sys.executable]


@pytest.mark.skipif(not os.environ.get('GLOSSAPI_INSTALL_E2E_PYTHON'), reason='set GLOSSAPI_INSTALL_E2E_PYTHON to a throwaway venv python')
def test_real_install_in_a_throwaway_venv():
    """--yes for real (CPU profile, download) in a disposable venv that has glossapi installed; never a shared venv."""
    python = os.environ['GLOSSAPI_INSTALL_E2E_PYTHON']
    assert 'scratchpad' in python or 'tmp' in python, 'refusing a non-throwaway environment'
    run = lambda *argv: subprocess.run([python, '-m', 'glossapi.cli', *argv, '--json'], capture_output=True, text=True, timeout=900)
    before = json.loads(run('capabilities').stdout)['data']
    assert not next(row for row in before['stages'] if row['stage'] == 'download')['available']
    done = run('install', '--device', 'cpu', '--package', 'download', '--yes')
    envelope = json.loads(done.stdout)
    validate(envelope)
    assert done.returncode == 0, envelope['errors']
    assert any(item.startswith('aiohttp==') for item in envelope['data']['changes']['added'])
    assert {row['stage']: row['available_after'] for row in envelope['data']['changes']['stages']}.get('download') is True
    after = json.loads(run('capabilities').stdout)['data']
    assert next(row for row in after['stages'] if row['stage'] == 'download')['available']


# ------------------------------------------------------------------------------------------------ guide ---

def guide_disagreements(guide):
    """Every way the guide can disagree with its sources; empty when consistent."""
    from glossapi.capabilities import STAGE_DEVICE_KINDS, STAGE_ORDER
    from glossapi.cli import parser
    from glossapi.input_manifest import PRODUCTS, SELECTOR_KEYS
    from glossapi.install_plan import ERRORS
    from glossapi.llm_guide import measured_costs
    from glossapi.package_catalog import catalog_view
    from glossapi.schemas import NAMES, load_schema
    from glossapi.settings_contract import PLAN_FIXED_NOTE, SETTINGS_CONTRACT, SETTINGS_DEFAULTS
    problems = []
    settings = guide['choices']['settings']
    if {name: entry['values'] for name, entry in settings['values'].items()} != SETTINGS_CONTRACT:
        problems.append('settings text differs from settings_contract')
    if {name: entry['default'] for name, entry in settings['values'].items()} != SETTINGS_DEFAULTS or settings['fixed_stages'] != PLAN_FIXED_NOTE:
        problems.append('settings defaults/note differ')
    catalog = catalog_view(installed=False)
    if [row['name'] for row in guide['choices']['packages']] != [row['name'] for row in catalog['profiles']]:
        problems.append('package names differ from the catalog')
    for ours, theirs in zip(guide['choices']['packages'], catalog['profiles']):
        sizes = [(size['target'], size['device'], size['packages'], size['added_packages'], size['added_download_bytes'], size['added_installed_bytes'])
                 for size in theirs['install_size']]
        if [(size['target'], size['device'], size['packages'], size['added_packages'], size['added_download_bytes'], size['added_installed_bytes'])
                for size in ours['install_size']] != sizes:
            problems.append('install sizes differ for ' + ours['name'])
        if ours['devices'] != [item['device'] for item in theirs['devices'] if item['supported']] or ours['purpose'] != theirs['purpose']:
            problems.append('devices/purpose differ for ' + ours['name'])
        if (ours.get('proprietary'), ours.get('proprietary_packages')) != (theirs['licence']['proprietary'], theirs['licence']['proprietary_packages']):
            problems.append('proprietary licence flag differs for ' + ours['name'])
    if [row['name'] for row in guide['schemas']] != list(NAMES) or any(load_schema(row['name'])['$id'] != row['id'] for row in guide['schemas']):
        problems.append('schema list differs')
    if [row['stage'] for row in guide['phases']] != list(STAGE_ORDER) or any(row['device_kinds'] != list(STAGE_DEVICE_KINDS[row['stage']]) for row in guide['phases']):
        problems.append('phases differ from capabilities')
    if {row['name']: row['stage'] for row in guide['products']} != PRODUCTS:
        problems.append('products differ from the manifest')
    if guide['choices']['files']['selector_keys'] != list(SELECTOR_KEYS):
        problems.append('manifest selector keys differ')
    for row in guide['workflow']:
        try:
            parser().parse_args(row['argv'][1:])
        except Exception as error:
            problems.append(f"workflow step {row['step']} does not parse: {error}")
    if any(guide['errors'][code]['exit_code'] != exit_code for code, (exit_code, _) in ERRORS.items()):
        problems.append('install error codes differ')
    # Plan/run and manifest errors (CP-K3 F4, CP-K3c2 C2-F2): the guide rows equal the owners' registries; each
    # registry equals the literal codes the owner's source raises (read from its AST, any argument position), computed
    # codes stand only at declared sites, and the live functions are the ones compiled from that source.
    import glossapi.input_manifest as manifest_owner
    import glossapi.plan_preview as plan_owner
    for owner in (plan_owner, manifest_owner):
        for code, row in owner.error_contracts().items():
            if guide['errors'].get(code) != row:
                problems.append(f'guide error row for {code} differs from {owner.__name__}.ERRORS')
        problems.extend(code_problems(owner.__name__, Path(owner.__file__).read_text(), registry=set(owner.ERRORS),
                                      computed_sites=COMPUTED_CODE_SITES.get(owner.__name__, frozenset())))
        problems.extend(live_source_problems(owner))
        raised = raised_codes(owner)
        dynamic = plan_owner.PLAN_ERRORS if owner is plan_owner else frozenset()
        stale = sorted(set(owner.ERRORS) - raised - dynamic)
        if stale:
            problems.append(f'{owner.__name__}.ERRORS lists codes it never raises: {stale}')
    conflict_codes = sorted((plan_owner.PLAN_ERRORS | plan_owner.RUN_BLOCKERS) - set(plan_owner.ERRORS))
    if conflict_codes:
        problems.append(f'plan/run conflict codes missing from plan_preview.ERRORS: {conflict_codes}')
    docs = (REPO / 'docs/cli_json_schema.md').read_text()
    missing = [code for code in guide['errors'] if f'`{code}`' not in docs]
    if missing:
        problems.append(f'error codes not documented in cli_json_schema.md: {missing}')
    costs = measured_costs()
    if guide['costs']['time_per_page'] != costs['time_per_page'] or guide['costs']['disk'] != costs['disk']:
        problems.append('costs differ from measured_costs.json')
    return problems


ERROR_CLASSES = frozenset(('PlanBindingError', 'ManifestError'))
# Call sites that pass a computed code, each checked another way: refuse_conflicts re-raises a conflict's code, which
# is a PLAN_ERRORS code (the PLAN_ERRORS/RUN_BLOCKERS subset check in guide_disagreements).
COMPUTED_CODE_SITES = {'glossapi.plan_preview': frozenset(('refuse_conflicts',))}


def _error_calls(source):
    """Every ``PlanBindingError(...)``/``ManifestError(...)`` call in ``source``: (enclosing function qualname, line,
    the literal code or None when the code is computed). The code is the first positional argument (a starred literal
    tuple included) or the ``code=`` keyword, wherever it stands among the arguments (CP-K3c2 C2-F2)."""
    import ast
    calls = []

    def literal(node):
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = scope + (child.name,)
            if isinstance(child, ast.Call):
                name = child.func.id if isinstance(child.func, ast.Name) else child.func.attr if isinstance(child.func, ast.Attribute) else None
                if name in ERROR_CLASSES:
                    code = None
                    if child.args:
                        first = child.args[0]
                        if isinstance(first, ast.Starred):
                            items = first.value.elts if isinstance(first.value, (ast.Tuple, ast.List)) else []
                            code = literal(items[0]) if items else None
                        else:
                            code = literal(first)
                    else:
                        keyword = next((item for item in child.keywords if item.arg == 'code'), None)
                        code = literal(keyword.value) if keyword is not None else None
                    calls.append(('.'.join(scope) or '<module>', child.lineno, code))
            visit(child, inner)

    visit(ast.parse(source), ())
    return calls


def code_problems(module_name, source, *, registry, computed_sites=frozenset()):
    """Literal codes missing from ``registry``, and computed codes outside the declared ``computed_sites`` (fail closed)."""
    problems = []
    for where, line, code in _error_calls(source):
        if code is None and where not in computed_sites:
            problems.append(f'{module_name}.{where} line {line} raises a computed error code (not a literal and not a declared computed site)')
        elif code is not None and code not in registry:
            problems.append(f'{module_name}.{where} line {line} raises {code!r}, missing from its ERRORS registry (and the guide)')
    return problems


def raised_codes(module):
    """The literal codes ``module``'s source passes to its typed error classes (positional or ``code=``)."""
    return {code for _, _, code in _error_calls(Path(module.__file__).read_text()) if code is not None}


def live_source_problems(module):
    """Every live function of ``module`` (module level and class methods) must be the one compiled from its source file,
    so that reading the source reads what runs. A function replaced at run time, for example recompiled from edited
    source, is reported with the string constants it adds."""
    import inspect
    import types
    compiled = {}

    def walk(code):
        compiled[code.co_qualname] = code
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                walk(constant)

    walk(compile(Path(module.__file__).read_text(), module.__file__, 'exec'))

    def strings(code):
        out = set()
        for constant in code.co_consts:
            if isinstance(constant, str):
                out.add(constant)
            elif isinstance(constant, tuple):
                out.update(item for item in constant if isinstance(item, str))
            elif isinstance(constant, types.CodeType):
                out |= strings(constant)
        return out

    import dataclasses

    def members(cls):
        for item in vars(cls).values():
            if isinstance(item, (staticmethod, classmethod)):
                item = item.__func__
            if isinstance(item, property):
                yield from (accessor for accessor in (item.fget, item.fset, item.fdel) if accessor is not None)
            elif inspect.isfunction(item):
                yield item

    def generated(code):
        # methods that @dataclass writes (exec of generated text, or its own helpers); nothing else is exempt
        return (code.co_filename == '<string>' and code.co_qualname.startswith('__create_fn__.')) or code.co_filename == dataclasses.__file__

    functions = []
    for value in vars(module).values():
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            functions.append(value)
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            functions.extend(members(value))
    problems = []
    for function in functions:
        if generated(function.__code__):
            continue
        live, source = function.__code__, compiled.get(function.__code__.co_qualname)
        if live != source:
            added = sorted(strings(live) - (strings(source) if source is not None else set()))
            problems.append(f'{module.__name__}.{live.co_qualname} differs from its source file (added constants: {added})')
    return problems


def test_guide_is_schema_valid_consistent_and_equals_the_docs_file():
    from glossapi.llm_guide import render_markdown
    code, envelope = cli('guide')
    assert code == 0 and envelope['command'] == 'guide'
    guide = envelope['data']
    assert guide_disagreements(guide) == []
    assert (REPO / 'docs/llm_guide.md').read_text() == render_markdown(guide), \
        'docs/llm_guide.md is stale: regenerate with `python -m glossapi.llm_guide > docs/llm_guide.md`'
    validate(guide['handover']['example'], 'manifest')
    steps = [row['argv'][1] for row in guide['workflow']]
    assert steps.index('capabilities') < steps.index('install') < steps.index('plan') < steps.index('run') < steps.index('status') < steps.index('report')
    assert guide['workflow'][-1]['argv'][4:6] == ['--why', 'DOCUMENT']
    assert 'does not search or scrape' in guide['choices']['files']['no_scraping']


@pytest.mark.parametrize('mutation', ['settings', 'catalog', 'schemas', 'costs', 'workflow'])
def test_guide_disagreement_is_detected(monkeypatch, mutation):
    """The consistency check is not vacuous: each source drifting from the guide is reported."""
    import copy
    from glossapi.llm_guide import guide_view
    guide = copy.deepcopy(guide_view())
    if mutation == 'settings':
        guide['choices']['settings']['values']['debug']['values']['off'] += ' (edited)'
    elif mutation == 'catalog':
        guide['choices']['packages'][1]['install_size'][0]['added_packages'] += 1
    elif mutation == 'schemas':
        guide['schemas'].pop()
    elif mutation == 'costs':
        guide['costs']['time_per_page'][0]['stages']['extract'] = 0.001
    else:
        guide['workflow'][0]['argv'].append('--no-such-option')
    assert guide_disagreements(guide)


def test_guide_markdown_detects_a_hand_edit(tmp_path):
    from glossapi.llm_guide import guide_view, render_markdown
    text = render_markdown(guide_view())
    assert text.replace('debug=off', 'debug=OFF', 1) != text


def test_install_errors_are_documented_with_their_exit_codes():
    from glossapi.install_plan import ERRORS
    docs = (REPO / 'docs/cli_json_schema.md').read_text()
    for code, (exit_code, _) in ERRORS.items():
        assert f'| `{code}` | {exit_code} |' in docs, code
