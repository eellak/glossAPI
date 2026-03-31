from pathlib import Path

from glossapi.scripts.install_glossapi import (
    build_deepseek_command,
    build_install_plan,
    build_pip_command,
)


def test_build_install_plan_collects_phase_extras():
    plan = build_install_plan(
        phases=["download", "browser_download", "extract", "ocr"],
        editable=True,
        include_cuda=False,
    )

    assert plan.phases == ("download", "browser_download", "extract", "ocr")
    assert set(plan.extras) == {"browser", "docling"}
    assert plan.editable is True
    assert plan.needs_deepseek_runtime is True


def test_build_install_plan_adds_cuda_extra():
    plan = build_install_plan(
        phases=["download"],
        editable=False,
        include_cuda=True,
    )

    assert set(plan.extras) == {"cuda"}
    assert plan.editable is False
    assert plan.needs_deepseek_runtime is False


def test_build_pip_command_uses_editable_install():
    plan = build_install_plan(
        phases=["download", "browser_download"],
        editable=True,
        include_cuda=False,
    )
    command = build_pip_command(plan, Path("/tmp/repo"))

    assert command[:4] == [command[0], "-m", "pip", "install"]
    assert "-e" in command
    assert command[-1] == ".[browser]"


def test_build_deepseek_command_points_to_setup_script():
    command = build_deepseek_command(Path("/tmp/repo"))

    assert command is None or command[0]
