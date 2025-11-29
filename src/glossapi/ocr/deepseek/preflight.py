"""Preflight checks for the DeepSeek OCR CLI environment."""

from __future__ import annotations

import dataclasses
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_SCRIPT = Path.cwd() / "deepseek-ocr" / "run_pdf_ocr_vllm.py"
DEFAULT_MODEL_DIR = Path.cwd() / "deepseek-ocr" / "DeepSeek-OCR"
DEFAULT_LIB_DIR = Path.cwd() / "deepseek-ocr" / "libjpeg-turbo" / "lib"


@dataclasses.dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    errors: List[CheckResult]
    warnings: List[CheckResult]
    infos: List[CheckResult]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summarize(self) -> str:
        lines: List[str] = []
        if self.errors:
            lines.append("Errors:")
            lines += [f"  - {c.name}: {c.message}" for c in self.errors]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  - {c.name}: {c.message}" for c in self.warnings]
        if self.infos:
            lines.append("Info:")
            lines += [f"  - {c.name}: {c.message}" for c in self.infos]
        return "\n".join(lines)


def _ensure_path(path: Path, label: str, errors: List[CheckResult]) -> Optional[Path]:
    if not path:
        errors.append(CheckResult(label, False, "Not provided"))
        return None
    if not path.exists():
        errors.append(CheckResult(label, False, f"Missing at {path}"))
        return None
    return path


def check_deepseek_env(
    env: Optional[Dict[str, str]] = None,
    *,
    check_flashinfer: bool = True,
) -> PreflightReport:
    """Validate DeepSeek CLI prerequisites without running the model."""

    env = dict(env or os.environ)
    errors: List[CheckResult] = []
    warnings: List[CheckResult] = []
    infos: List[CheckResult] = []

    allow_cli = env.get("GLOSSAPI_DEEPSEEK_ALLOW_CLI", "0") == "1"
    allow_stub = env.get("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "1") == "1"
    if not allow_cli:
        warnings.append(
            CheckResult(
                "allow_cli",
                False,
                "Set GLOSSAPI_DEEPSEEK_ALLOW_CLI=1 to force the real CLI.",
            )
        )
    if allow_stub:
        warnings.append(
            CheckResult(
                "allow_stub",
                False,
                "Set GLOSSAPI_DEEPSEEK_ALLOW_STUB=0 to fail instead of falling back to stub output.",
            )
        )

    script = Path(env.get("GLOSSAPI_DEEPSEEK_VLLM_SCRIPT") or DEFAULT_SCRIPT)
    _ensure_path(script, "vllm_script", errors)

    python_bin = Path(env.get("GLOSSAPI_DEEPSEEK_TEST_PYTHON") or sys.executable)
    _ensure_path(python_bin, "deepseek_python", errors)

    model_dir = Path(
        env.get("GLOSSAPI_DEEPSEEK_TEST_MODEL_DIR")
        or env.get("GLOSSAPI_DEEPSEEK_MODEL_DIR")
        or DEFAULT_MODEL_DIR
    )
    model_dir = _ensure_path(model_dir, "model_dir", errors)
    if model_dir:
        has_weights = any(model_dir.glob("*.safetensors")) or (model_dir / "model-00001-of-000001.safetensors").exists()
        has_config = (model_dir / "config.json").exists()
        if not has_weights or not has_config:
            errors.append(
                CheckResult(
                    "model_contents",
                    False,
                    f"Model dir {model_dir} is missing weights/config.json",
                )
            )

    ld_path_env = env.get("GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH")
    lib_dir = Path(ld_path_env) if ld_path_env else DEFAULT_LIB_DIR
    _ensure_path(lib_dir, "ld_library_path", errors)

    cc1plus_path = shutil.which("cc1plus", path=env.get("PATH", ""))
    if not cc1plus_path:
        errors.append(
            CheckResult(
                "cc1plus",
                False,
                "C++ toolchain missing (cc1plus not on PATH); install g++ and ensure PATH includes gcc's cc1plus.",
            )
        )
    else:
        infos.append(CheckResult("cc1plus", True, f"Found at {cc1plus_path}"))

    if check_flashinfer:
        try:
            import flashinfer  # type: ignore

            infos.append(CheckResult("flashinfer", True, f"flashinfer {flashinfer.__version__} import ok"))
        except Exception as exc:  # pragma: no cover - depends on env
            errors.append(CheckResult("flashinfer", False, f"flashinfer import failed: {exc}"))

    return PreflightReport(errors=errors, warnings=warnings, infos=infos)


def main(argv: Optional[Iterable[str]] = None) -> int:
    report = check_deepseek_env()
    summary = report.summarize()
    if summary:
        print(summary)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
