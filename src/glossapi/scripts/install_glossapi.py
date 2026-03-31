"""Guided installer for GlossAPI extras."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


PHASE_TO_EXTRAS: Dict[str, Set[str]] = {
    "download": set(),
    "browser_download": {"browser"},
    "extract": {"docling"},
    "ocr": set(),
    "docs": {"docs"},
}


@dataclass(frozen=True)
class InstallPlan:
    phases: tuple[str, ...]
    extras: tuple[str, ...]
    editable: bool
    include_cuda: bool
    needs_deepseek_runtime: bool


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") not in {"", "dumb", None}


def _style(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{question} {suffix} ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def _resolve_phase_selection(tokens: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    seen: Set[str] = set()
    for token in tokens:
        phase = str(token).strip().lower()
        if not phase:
            continue
        if phase not in PHASE_TO_EXTRAS:
            raise ValueError(f"Unsupported phase '{token}'. Valid phases: {', '.join(sorted(PHASE_TO_EXTRAS))}")
        if phase not in seen:
            seen.add(phase)
            resolved.append(phase)
    return resolved


def build_install_plan(
    *,
    phases: Sequence[str],
    editable: bool,
    include_cuda: bool,
) -> InstallPlan:
    selected = _resolve_phase_selection(phases)
    extras: Set[str] = set()
    for phase in selected:
        extras.update(PHASE_TO_EXTRAS[phase])
    if include_cuda:
        extras.add("cuda")
    return InstallPlan(
        phases=tuple(selected),
        extras=tuple(sorted(extras)),
        editable=bool(editable),
        include_cuda=bool(include_cuda),
        needs_deepseek_runtime=("ocr" in selected),
    )


def build_pip_command(plan: InstallPlan, repo_root: Path) -> List[str]:
    target = "."
    if plan.extras:
        target = f".[{','.join(plan.extras)}]"
    cmd = [sys.executable, "-m", "pip", "install"]
    if plan.editable:
        cmd.append("-e")
    cmd.append(target)
    return cmd


def build_deepseek_command(repo_root: Path) -> Optional[List[str]]:
    script = repo_root / "dependency_setup" / "setup_deepseek_uv.sh"
    if not script.exists():
        return None
    shell = shutil.which("bash") or shutil.which("sh")
    if not shell:
        return None
    return [shell, str(script)]


def _interactive_plan(default_editable: bool) -> InstallPlan:
    print(_style("GlossAPI Installer", "1;36"))
    print("Select only the phases you plan to use so optional dependencies stay minimal.\n")

    selected: List[str] = ["download"]
    print(_style("Core", "1;37"))
    print("  download: base downloader/data pipeline dependencies")
    if _prompt_yes_no("Add browser-gated download support?", default=False):
        selected.append("browser_download")
    if _prompt_yes_no("Add extraction support (Docling)?", default=False):
        selected.append("extract")
    if _prompt_yes_no("Add OCR support (DeepSeek backend)?", default=False):
        selected.append("ocr")
    if _prompt_yes_no("Add docs tooling?", default=False):
        selected.append("docs")
    include_cuda = _prompt_yes_no("Include CUDA extras where relevant?", default=False)
    editable = _prompt_yes_no("Install in editable mode?", default=default_editable)
    return build_install_plan(phases=selected, editable=editable, include_cuda=include_cuda)


def _plan_summary(plan: InstallPlan, command: Sequence[str]) -> str:
    extras = ", ".join(plan.extras) if plan.extras else "(none)"
    phases = ", ".join(plan.phases) if plan.phases else "(none)"
    return "\n".join(
        [
            _style("Install plan", "1;32"),
            f"  phases: {phases}",
            f"  extras: {extras}",
            f"  editable: {'yes' if plan.editable else 'no'}",
            f"  command: {shlex.join(command)}",
            f"  deepseek runtime: {'separate setup required' if plan.needs_deepseek_runtime else 'not requested'}",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python install_glossapi.py",
        description="Guided installer for GlossAPI optional dependency groups.",
    )
    parser.add_argument(
        "--phases",
        default="",
        help=(
            "Comma-separated phases to install. Valid values: "
            + ", ".join(sorted(PHASE_TO_EXTRAS))
            + ". If omitted, an interactive wizard is shown."
        ),
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Include the CUDA extra.",
    )
    parser.add_argument(
        "--editable",
        dest="editable",
        action="store_true",
        help="Install in editable mode.",
    )
    parser.add_argument(
        "--no-editable",
        dest="editable",
        action="store_false",
        help="Install as a regular package.",
    )
    parser.set_defaults(editable=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed pip command without running it.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts in non-interactive mode.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]

    if args.phases.strip():
        plan = build_install_plan(
            phases=[token for token in args.phases.split(",") if token.strip()],
            editable=args.editable,
            include_cuda=bool(args.cuda),
        )
    else:
        plan = _interactive_plan(default_editable=bool(args.editable))

    command = build_pip_command(plan, repo_root)
    print(_plan_summary(plan, command))
    deepseek_command = build_deepseek_command(repo_root) if plan.needs_deepseek_runtime else None
    if deepseek_command:
        print(f"  deepseek command: {shlex.join(deepseek_command)}")

    if args.dry_run:
        return 0
    if not args.yes and not args.phases.strip():
        if not _prompt_yes_no("Run this install command now?", default=True):
            print("Aborted.")
            return 1

    completed = subprocess.run(command, cwd=repo_root)
    if completed.returncode != 0:
        return int(completed.returncode)
    if plan.needs_deepseek_runtime and deepseek_command:
        print(_style("Provisioning dedicated DeepSeek runtime…", "1;33"))
        completed = subprocess.run(deepseek_command, cwd=repo_root)
    return int(completed.returncode)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
