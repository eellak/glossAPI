from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from glossapi.scripts import install_glossapi as installer

REPO = Path(__file__).resolve().parents[1]


def test_default_is_core_without_torch():
    plan = installer.build_install_plan()
    commands = installer.build_pip_commands(plan, REPO)
    assert plan.packages == ("core",)
    assert plan.extras == ()
    assert plan.device is None
    assert commands[:2] == [
        [sys.executable, "-m", "pip", "install", "--index-url", installer.PYPI_INDEX, str(REPO)],
        [sys.executable, "-m", "pip", "check"],
    ]


def test_named_packages_and_editable_target():
    plan = installer.build_install_plan(packages=["download", "browser", "annotate", "download"], editable=True)
    command = installer.build_pip_commands(plan, REPO)[0]
    assert plan.extras == ("annotate", "browser", "download")
    assert command[-2:] == ["--editable", str(REPO) + "[annotate,browser,download]"]


@pytest.mark.parametrize("device,suffix,index", [
    ("cpu", "+cpu", "cpu"), ("cuda126", "+cu126", "cu126"), ("cuda132", "+cu132", "cu132"),
])
def test_linux_torch_pair_precedes_layout_and_is_constrained(monkeypatch, device, suffix, index):
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    plan = installer.build_install_plan(packages=["layout"], device=device)
    commands = installer.build_pip_commands(plan, REPO)
    assert commands[0] == [sys.executable, "-m", "pip", "install",
                           "torch==2.14.1" + suffix, "torchvision==0.29.1" + suffix,
                           "--index-url", "https://download.pytorch.org/whl/" + index]
    constraint = Path(commands[1][commands[1].index("--constraint") + 1])
    assert f"torch==2.14.1{suffix}" in constraint.read_text().splitlines()
    assert f"torchvision==0.29.1{suffix}" in constraint.read_text().splitlines()
    assert commands[1][-1] == str(REPO) + "[layout]"
    assert "--extra-index-url" not in sum(commands, [])


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_mac_uses_normal_wheels(monkeypatch, device):
    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installer.platform, "machine", lambda: "arm64")
    plan = installer.build_install_plan(packages=["layout"], device=device)
    commands = installer.build_pip_commands(plan, REPO)
    assert commands[0][-1] == installer.PYPI_INDEX
    assert "torch==2.14.1" in commands[0]
    assert commands[1][commands[1].index("--constraint") + 1].endswith("constraints-torch-macos.txt")


@pytest.mark.parametrize("system,machine,device", [
    ("Darwin", "arm64", "cuda126"), ("Linux", "x86_64", "mps"),
    ("Darwin", "x86_64", "mps"), ("Windows", "AMD64", "cpu"),
])
def test_invalid_platform_device_rejected(monkeypatch, system, machine, device):
    monkeypatch.setattr(installer.platform, "system", lambda: system)
    monkeypatch.setattr(installer.platform, "machine", lambda: machine)
    with pytest.raises(ValueError):
        installer.build_install_plan(packages=["layout"], device=device)


@pytest.mark.parametrize("kwargs", [
    {"packages": ["vanilla"]}, {"device": "cpu"},
    {"packages": ["layout"], "device": "cuda"},
    {"pypdfium2_version": "5.0.0"}, {"pypdfium2_version": "4.29.0"},
    {"pypdfium2_version": "4.30.0rc1"},
])
def test_invalid_selection_rejected(kwargs):
    with pytest.raises(ValueError):
        installer.build_install_plan(**kwargs)


def test_pdfium_4_pin_is_in_final_resolve():
    plan = installer.build_install_plan(pypdfium2_version="4.30.0")
    assert installer.build_pip_commands(plan, REPO)[0][-1] == "pypdfium2==4.30.0"


def test_dry_run_prints_exact_commands_without_execution(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("dry-run executed a command")
    monkeypatch.setattr(installer.subprocess, "run", unexpected)
    assert installer.main(["download", "--dry-run"]) == 0
    output = capsys.readouterr().out.splitlines()
    plan = installer.build_install_plan(packages=["download"])
    assert output[1:] == [shlex.join(command) for command in installer.build_pip_commands(plan, REPO)]


def test_failed_torch_install_stops_before_docling(monkeypatch):
    monkeypatch.setattr(installer, "legacy_native_distributions", lambda: [])
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    calls = []
    def failed(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 23)
    monkeypatch.setattr(installer.subprocess, "run", failed)
    assert installer.main(["layout"]) == 23
    assert len(calls) == 1
    assert "torch==2.14.1+cpu" in calls[0]


def test_shell_dry_run_does_not_create_venv(tmp_path):
    target = tmp_path / "environment with spaces"
    completed = subprocess.run(["bash", str(REPO / "dependency_setup/setup_glossapi.sh"),
                                "--python", sys.executable, "--venv", str(target),
                                "download", "--dry-run"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert not target.exists()
    commands = [shlex.split(line) for line in completed.stdout.splitlines() if not line.startswith("Packages:")]
    assert commands[0] == [sys.executable, "-m", "venv", str(target)]
    assert commands[1][0] == str(target / "bin/python")
    assert commands[1][-1] == str(REPO) + "[download]"


@pytest.mark.parametrize('name', installer.NATIVE_MODULES)
def test_legacy_package_refused_before_any_install(tmp_path, monkeypatch, capsys, name):
    # Real importlib.metadata detection, including normalized distribution names.
    info = tmp_path / (name + '-0.1.0.dist-info')
    info.mkdir()
    (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name.replace("_", "-")}\nVersion: 0.1.0\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(installer.subprocess, 'run', lambda *a, **kw: pytest.fail('mutated environment'))
    detected = installer.legacy_native_distributions()
    assert name in detected
    assert installer.main([]) == 1
    message = capsys.readouterr().err
    assert 'Refusing installation' in message
    assert shlex.join([sys.executable, '-m', 'pip', 'uninstall', '-y', *detected]) in message


def test_install_verifies_native_in_fresh_process_after_pip_check(monkeypatch):
    monkeypatch.setattr(installer, 'legacy_native_distributions', lambda: [])
    calls = []
    monkeypatch.setattr(installer.subprocess, 'run', lambda cmd, **kw:
                        (calls.append(cmd) or subprocess.CompletedProcess(cmd, 0)))
    assert installer.main([]) == 0
    assert calls[-2] == [sys.executable, '-m', 'pip', 'check']
    assert calls[-1][:2] == [sys.executable, '-c']
    assert 'verify_native_origins' in calls[-1][2]


def test_native_verifier_checks_ownership_hash_and_shadowing(tmp_path, monkeypatch, capsys):
    import base64
    import hashlib
    from importlib.machinery import EXTENSION_SUFFIXES
    from types import SimpleNamespace

    monkeypatch.setattr(installer, 'legacy_native_distributions', lambda: [])
    files = []
    modules = {}
    for name in installer.NATIVE_MODULES:
        path = tmp_path / (name + EXTENSION_SUFFIXES[0])
        path.write_bytes(name.encode())
        record = installer.metadata.PackagePath(path.name)
        record.hash = SimpleNamespace(mode='sha256', value=base64.urlsafe_b64encode(
            hashlib.sha256(path.read_bytes()).digest()).rstrip(b'=').decode())
        files.append(record)
        modules[name] = SimpleNamespace(__file__=str(path))
    dist = SimpleNamespace(files=files, locate_file=lambda f: tmp_path / f)
    monkeypatch.setattr(installer.metadata, 'distributions', lambda **kw: [dist])
    monkeypatch.setattr(installer.importlib, 'import_module', lambda name: modules[name])
    installer.verify_native_origins(REPO)
    output = capsys.readouterr().out
    assert all(name in output for name in installer.NATIVE_MODULES)
    path.write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='RECORD'):
        installer.verify_native_origins(REPO)
    path.write_bytes(name.encode())
    dist.files = []
    with pytest.raises(RuntimeError, match='not owned'):
        installer.verify_native_origins(REPO)
    modules[installer.NATIVE_MODULES[0]].__file__ = str(tmp_path / '__init__.py')
    with pytest.raises(RuntimeError, match='shadowing package'):
        installer.verify_native_origins(REPO)
