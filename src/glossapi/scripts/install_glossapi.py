"""Install core or explicitly selected GlossAPI packages in the active Python."""
from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PACKAGES = ("core", "download", "browser", "layout", "annotate")
DEVICES = ("cpu", "cuda126", "cuda132", "mps")
TORCH_VERSION = "2.14.1"
TORCHVISION_VERSION = "0.29.1"
PYPI_INDEX = "https://pypi.org/simple"


@dataclass(frozen=True)
class InstallPlan:
    packages: tuple[str, ...]
    extras: tuple[str, ...]
    editable: bool
    device: str | None
    torch_index: str | None
    torch_suffix: str
    pypdfium2_version: str | None


def build_install_plan(
    *, packages: Sequence[str] = (), editable: bool = False,
    device: str | None = None, pypdfium2_version: str | None = None,
) -> InstallPlan:
    selected = tuple(dict.fromkeys(packages or ("core",)))
    unknown = set(selected) - set(PACKAGES)
    if unknown:
        raise ValueError(f"Unknown packages: {', '.join(sorted(unknown))}; choose {', '.join(PACKAGES)}")
    if device is not None and device not in DEVICES:
        raise ValueError(f"Unknown device '{device}'; choose {', '.join(DEVICES)}")
    if device and "layout" not in selected:
        raise ValueError("--device requires the layout package")
    index = None
    suffix = ""
    if "layout" in selected:
        device = device or "cpu"
        system = platform.system()
        if device == "mps":
            if system != "Darwin" or platform.machine() != "arm64":
                raise ValueError("mps requires an Apple Silicon Mac")
            index = PYPI_INDEX
        elif device.startswith("cuda"):
            if system != "Linux":
                raise ValueError("CUDA layout packages require Linux; use cpu or mps on macOS")
            suffix = "+cu" + device[4:]
            index = "https://download.pytorch.org/whl/" + suffix[1:]
        elif system == "Linux":
            suffix = "+cpu"
            index = "https://download.pytorch.org/whl/cpu"
        elif system == "Darwin" and platform.machine() == "arm64":
            index = PYPI_INDEX
        else:
            raise ValueError("layout supports Linux or Apple Silicon macOS")
    if pypdfium2_version:
        parts = pypdfium2_version.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("--pypdfium2-version requires a stable version such as 4.30.0")
        if not (4, 30, 0) <= tuple(map(int, parts)) < (5, 0, 0):
            raise ValueError("pypdfium2 must remain >=4.30,<5")
    return InstallPlan(selected, tuple(sorted(set(selected) - {"core"})), bool(editable),
                       device, index, suffix, pypdfium2_version)


def build_pip_commands(plan: InstallPlan, repo_root: Path) -> list[list[str]]:
    """Return every command, in execution order, with absolute install targets."""
    pip = [sys.executable, "-m", "pip", "install"]
    commands = []
    constraints = []
    if plan.torch_index:
        commands.append([*pip, f"torch=={TORCH_VERSION}{plan.torch_suffix}",
                         f"torchvision=={TORCHVISION_VERSION}{plan.torch_suffix}",
                         "--index-url", plan.torch_index])
        profile = plan.torch_suffix[1:] or "macos"
        constraints = ["--constraint", str(repo_root / "dependency_setup" / f"constraints-torch-{profile}.txt")]
    target = str(repo_root)
    if plan.extras:
        target += f"[{','.join(plan.extras)}]"
    command = [*pip, "--index-url", PYPI_INDEX, *constraints]
    if plan.editable:
        command.append("--editable")
    command.append(target)
    if plan.pypdfium2_version:
        command.append(f"pypdfium2=={plan.pypdfium2_version}")
    commands.append(command)
    commands.append([sys.executable, "-m", "pip", "check"])
    return commands


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="*", choices=None, metavar="PACKAGE",
                        help=f"Default: core. Choices: {', '.join(PACKAGES)}")
    parser.add_argument("--packages", dest="package_list", default="", help="Comma-separated package names")
    parser.add_argument("--device", choices=DEVICES, help="Layout device (default: cpu)")
    parser.add_argument("--pypdfium2-version", help="Pin a qualified 4.x release, e.g. 4.30.0")
    parser.add_argument("--editable", action="store_true", help="Install the local source in editable mode")
    parser.add_argument("--dry-run", action="store_true", help="Print exact commands without executing them")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not (3, 11) <= sys.version_info[:2] <= (3, 14) or sys.version_info.releaselevel != "final":
        parser.error("Use a stable Python 3.11–3.14 interpreter")
    try:
        plan = build_install_plan(packages=[*args.packages, *filter(None, args.package_list.split(','))],
                                  editable=args.editable, device=args.device,
                                  pypdfium2_version=args.pypdfium2_version)
    except ValueError as error:
        parser.error(str(error))
    repo_root = Path(__file__).resolve().parents[3]
    commands = build_pip_commands(plan, repo_root)
    print(f"Packages: {', '.join(plan.packages)}; device: {plan.device or 'none'}", flush=True)
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    for command in commands:
        completed = subprocess.run(command, cwd=repo_root)
        if completed.returncode:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
