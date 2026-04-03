"""Preflight checks for the DeepSeek OCR environment."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .runtime_paths import resolve_deepseek_python

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCRIPT = REPO_ROOT / "src" / "glossapi" / "ocr" / "deepseek" / "run_pdf_ocr_transformers.py"
DEFAULT_MODEL_DIR = REPO_ROOT / "deepseek-ocr-2-model" / "DeepSeek-OCR-2"


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
    if not path.exists():
        errors.append(CheckResult(label, False, f"Missing at {path}"))
        return None
    return path


def check_deepseek_env(
    env: Optional[Dict[str, str]] = None,
    *,
    check_torch: bool = True,
) -> PreflightReport:
    """Validate DeepSeek OCR prerequisites without running the model."""

    env = dict(env or os.environ)
    errors: List[CheckResult] = []
    warnings: List[CheckResult] = []
    infos: List[CheckResult] = []

    allow_cli = env.get("GLOSSAPI_DEEPSEEK_ALLOW_CLI", "1") == "1"
    allow_stub = env.get("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "0") == "1"
    if not allow_cli:
        errors.append(
            CheckResult(
                "allow_cli",
                False,
                "DeepSeek OCR requires the real CLI/runtime. Set GLOSSAPI_DEEPSEEK_ALLOW_CLI=1.",
            )
        )
    if allow_stub:
        errors.append(
            CheckResult(
                "allow_stub",
                False,
                "Stub execution is no longer supported. Set GLOSSAPI_DEEPSEEK_ALLOW_STUB=0.",
            )
        )

    script = Path(
        env.get("GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT")
        or DEFAULT_SCRIPT
    )
    _ensure_path(script, "runner_script", errors)

    python_bin = resolve_deepseek_python(env=env)
    _ensure_path(python_bin, "deepseek_python", errors)

    model_dir = Path(
        env.get("GLOSSAPI_DEEPSEEK_TEST_MODEL_DIR")
        or env.get("GLOSSAPI_DEEPSEEK_MODEL_DIR")
        or DEFAULT_MODEL_DIR
    )
    model_dir = _ensure_path(model_dir, "model_dir", errors)
    if model_dir:
        has_weights = any(model_dir.glob("*.safetensors"))
        has_config = (model_dir / "config.json").exists()
        if not has_weights or not has_config:
            errors.append(
                CheckResult(
                    "model_contents",
                    False,
                    f"Model dir {model_dir} is missing weights/config.json",
                )
            )

    if check_torch:
        try:
            import torch  # type: ignore

            infos.append(CheckResult("torch", True, f"torch {torch.__version__} import ok"))
            if not torch.cuda.is_available():
                warnings.append(CheckResult("cuda", False, "Torch CUDA is not available."))
        except Exception as exc:  # pragma: no cover - depends on env
            errors.append(CheckResult("torch", False, f"torch import failed: {exc}"))

    return PreflightReport(errors=errors, warnings=warnings, infos=infos)


def main(argv: Optional[Iterable[str]] = None) -> int:
    del argv
    report = check_deepseek_env()
    summary = report.summarize()
    if summary:
        print(summary)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
