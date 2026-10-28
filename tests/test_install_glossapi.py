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
    assert commands == [
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
