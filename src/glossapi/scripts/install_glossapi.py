"""Install core or explicitly selected GlossAPI packages in the active Python."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PACKAGES = ("core", "download", "browser", "layout", "ocr", "annotate", "annotate-legacy")
DEVICES = ("cpu", "cuda126", "cuda132", "mps")
TORCH_VERSION = "2.14.1"
TORCHVISION_VERSION = "0.29.1"
PYPI_INDEX = "https://pypi.org/simple"
NATIVE_MODULES = ("glossapi_rs_cleaner", "glossapi_rs_noise", "glossapi_rs_pdfrepair")


def legacy_native_distributions() -> list[str]:
    """Standalone packages take import precedence over root-wheel extensions."""
    found = []
    for name in NATIVE_MODULES:
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        found.append(name)
    return found


def verify_native_origins(repo_root: Path, *, editable: bool = False) -> None:
    """Check fresh imports against root wheel ownership and RECORD hashes."""
    from importlib.machinery import EXTENSION_SUFFIXES

    legacy = legacy_native_distributions()
    if legacy:
        raise RuntimeError(f"Legacy standalone native distributions still installed: {', '.join(legacy)}")
    modules = {name: importlib.import_module(name) for name in NATIVE_MODULES}
    owned = {}
    if not editable:
        # PYTHONPATH may expose source egg-info ahead of the installed wheel.
        # Look for ownership beside the loaded extensions, not at first metadata match.
        parents = {Path(module.__file__).resolve().parent for module in modules.values()}
        if len(parents) != 1:
            raise RuntimeError("Native modules loaded from different installation directories")
        distributions = list(metadata.distributions(name="glossapi", path=[str(parents.pop())]))
        if len(distributions) != 1:
            raise RuntimeError("Missing or ambiguous glossapi wheel metadata beside native modules")
        distribution = distributions[0]
        owned = {Path(distribution.locate_file(f)).resolve(): f for f in distribution.files or []}
    origins = {}
    for name in NATIVE_MODULES:
        module = modules[name]
        origin = Path(module.__file__).resolve()
        if not any(origin.name == name + suffix for suffix in EXTENSION_SUFFIXES):
            raise RuntimeError(f"{name} loaded from shadowing package: {origin}")
        digest = hashlib.sha256(origin.read_bytes()).digest()
        if editable:
            if origin.parent != (repo_root / "src").resolve():
                raise RuntimeError(f"{name} is outside the editable source: {origin}")
        else:
            record = owned.get(origin)
            if record is None or record.hash is None or record.hash.mode != "sha256":
                raise RuntimeError(f"{name} is not owned and hash-bound by the glossapi wheel: {origin}")
            if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != record.hash.value:
                raise RuntimeError(f"{name} does not match the glossapi wheel RECORD: {origin}")
        origins[name] = {"origin": str(origin), "sha256": digest.hex()}
    print(json.dumps({"native_modules": origins}, indent=2), flush=True)


@dataclass(frozen=True)
class InstallPlan:
    packages: tuple[str, ...]
    extras: tuple[str, ...]
    editable: bool
    device: str | None
    torch_index: str | None
    torch_suffix: str
    pypdfium2_version: str | None
    ocr_toolkit_source: str | None = None


def build_install_plan(
    *, packages: Sequence[str] = (), editable: bool = False,
    device: str | None = None, pypdfium2_version: str | None = None, ocr_toolkit_source: str | None = None,
) -> InstallPlan:
    selected = tuple(dict.fromkeys(packages or ("core",)))
    unknown = set(selected) - set(PACKAGES)
    if unknown:
        raise ValueError(f"Unknown packages: {', '.join(sorted(unknown))}; choose {', '.join(PACKAGES)}")
    if device is not None and device not in DEVICES:
        raise ValueError(f"Unknown device '{device}'; choose {', '.join(DEVICES)}")
    if device and not set(selected) & {'layout', 'ocr'}:
        raise ValueError("--device requires the layout or ocr package")
    index = None
    suffix = ""
    if set(selected) & {'layout', 'ocr'}:
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
                       device, index, suffix, pypdfium2_version, ocr_toolkit_source)


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
    # The OCR extra and installer consume the same file. Passing its entries
    # directly lets development override the unpublished source without also
    # resolving the public URL through wheel metadata.
    extras = tuple(extra for extra in plan.extras if extra != 'ocr')
    if extras:
        target += f"[{','.join(extras)}]"
    command = [*pip, "--index-url", PYPI_INDEX, *constraints]
    if plan.editable:
        command.append("--editable")
    command.append(target)
    if 'ocr' in plan.packages:
        requirements = (repo_root/'src/glossapi/ocr_package.txt').read_text().splitlines()
        for requirement in requirements:
            if not requirement.strip() or requirement.startswith('#'):
                continue
            requirement = requirement.split(' #',1)[0].strip()
            if requirement.startswith('greek-ocr-toolkit') and plan.ocr_toolkit_source:
                requirement = 'greek-ocr-toolkit[docling] @ ' + plan.ocr_toolkit_source
            command.append(requirement)
    if plan.pypdfium2_version:
        command.append(f"pypdfium2=={plan.pypdfium2_version}")
    commands.append(command)
    commands.append([sys.executable, "-m", "pip", "check"])
    # A new process avoids certifying modules cached before pip replaced files.
    verify = ("import runpy; from pathlib import Path; "
              f"ns=runpy.run_path({str(Path(__file__).resolve())!r}); "
              f"ns['verify_native_origins'](Path({str(repo_root)!r}), editable={plan.editable!r})")
    commands.append([sys.executable, "-c", verify])
    return commands


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="*", choices=None, metavar="PACKAGE",
                        help=f"Default: core. Choices: {', '.join(PACKAGES)}")
    parser.add_argument("--packages", dest="package_list", default="", help="Comma-separated package names")
    parser.add_argument("--device", choices=DEVICES, help="Layout/OCR device (default: cpu)")
    parser.add_argument('--ocr-toolkit-source', default=os.environ.get('GLOSSAPI_OCR_TOOLKIT_SOURCE'),
                        help='Development-only source URL (git+file://… or file:///archive-directory); public pin remains in package metadata')
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
                                  pypdfium2_version=args.pypdfium2_version, ocr_toolkit_source=args.ocr_toolkit_source)
    except ValueError as error:
        parser.error(str(error))
    repo_root = Path(__file__).resolve().parents[3]
    commands = build_pip_commands(plan, repo_root)
    print(f"Packages: {', '.join(plan.packages)}; device: {plan.device or 'none'}", flush=True)
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    legacy = legacy_native_distributions()
    if legacy:
        print("Refusing installation: legacy standalone native packages shadow the bundled extensions: "
              + ", ".join(legacy) + ".\nMigrate this interpreter first, then rerun the installer:\n"
              + shlex.join([sys.executable, "-m", "pip", "uninstall", "-y", *legacy]),
              file=sys.stderr, flush=True)
        return 1
    for command in commands:
        completed = subprocess.run(command, cwd=repo_root)
        if completed.returncode:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
