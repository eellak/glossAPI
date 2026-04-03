"""Resolve DeepSeek runtime paths for split-runtime GlossAPI installs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]


def _runtime_sort_key(candidate: Path) -> tuple[int, int, str]:
    name = candidate.parent.parent.name
    if name == "deepseek":
        return (1, 0, name)
    if name.startswith("deepseek"):
        suffix = name[len("deepseek") :]
        if suffix.isdigit():
            return (0, -int(suffix), name)
    return (2, 0, name)


def _candidate_deepseek_pythons(
    *,
    explicit_python: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
    repo_root: Optional[Path] = None,
) -> List[Path]:
    resolved_env = dict(env or os.environ)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT

    candidates: List[Path] = []

    def _append(candidate: Optional[Path | str]) -> None:
        if not candidate:
            return
        path = Path(candidate).expanduser()
        if path not in candidates:
            candidates.append(path)

    _append(explicit_python)
    _append(resolved_env.get("GLOSSAPI_DEEPSEEK_PYTHON"))
    _append(resolved_env.get("GLOSSAPI_DEEPSEEK_TEST_PYTHON"))

    venv_root = root / "dependency_setup" / ".venvs"
    if venv_root.exists():
        for candidate in sorted(venv_root.glob("deepseek*/bin/python"), key=_runtime_sort_key):
            _append(candidate)

    _append(sys.executable)
    return candidates


def resolve_deepseek_python(
    *,
    explicit_python: Optional[Path | str] = None,
    env: Optional[Dict[str, str]] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """Return the best available DeepSeek Python interpreter path.

    Preference order:
    1. explicit function argument
    2. explicit environment override
    3. validated repo-local DeepSeek venv(s)
    4. current process interpreter
    """

    resolved_env = dict(env or os.environ)
    explicit_candidate = Path(explicit_python).expanduser() if explicit_python else None
    if explicit_candidate is not None:
        return explicit_candidate

    for key in ("GLOSSAPI_DEEPSEEK_PYTHON", "GLOSSAPI_DEEPSEEK_TEST_PYTHON"):
        raw = resolved_env.get(key)
        if raw:
            return Path(raw).expanduser()

    candidates = _candidate_deepseek_pythons(
        explicit_python=None,
        env={},
        repo_root=repo_root,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


__all__ = ["resolve_deepseek_python"]
