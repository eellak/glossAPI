"""DeepSeek OCR runner."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pypdfium2 as _pypdfium2
except Exception:  # pragma: no cover - optional dependency
    _pypdfium2 = None

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCRIPT = REPO_ROOT / "src" / "glossapi" / "ocr" / "deepseek" / "run_pdf_ocr_transformers.py"


def _page_count(pdf_path: Path) -> int:
    if _pypdfium2 is None:
        return 0
    try:
        return len(_pypdfium2.PdfDocument(str(pdf_path)))
    except Exception:
        return 0


def _run_cli(
    input_dir: Path,
    output_dir: Path,
    *,
    files: List[str],
    model_dir: Path,
    python_bin: Optional[Path],
    script: Path,
    max_pages: Optional[int],
    content_debug: bool,
    device: Optional[str],
) -> None:
    python_exe = Path(python_bin) if python_bin else Path(sys.executable)
    cmd: List[str] = [
        str(python_exe),
        str(script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--model-dir",
        str(model_dir),
    ]
    if files:
        cmd += ["--files", *files]
    if max_pages is not None:
        cmd += ["--max-pages", str(max_pages)]
    if content_debug:
        cmd.append("--content-debug")
    if device:
        cmd += ["--device", str(device)]

    env = os.environ.copy()
    if shutil.which("cc1plus", path=env.get("PATH", "")) is None:
        for candidate in sorted(Path("/usr/lib/gcc/x86_64-linux-gnu").glob("*/cc1plus")):
            env["PATH"] = f"{candidate.parent}:{env.get('PATH', '')}"
            break
    ld_path = env.get("GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH")
    if ld_path:
        env["LD_LIBRARY_PATH"] = f"{ld_path}:{env.get('LD_LIBRARY_PATH', '')}"

    LOGGER.info("Running DeepSeek OCR CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # nosec: controlled arguments


def run_for_files(
    self_ref: Any,
    files: Iterable[str],
    *,
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,  # kept for API compatibility
    max_pages: Optional[int] = None,
    allow_stub: bool = False,  # ignored after stub removal; kept for compatibility
    allow_cli: bool = True,  # ignored after stub removal; kept for compatibility
    python_bin: Optional[Path] = None,
    vllm_script: Optional[Path] = None,
    content_debug: bool = False,
    persist_engine: bool = True,  # placeholder for future session reuse
    precision: Optional[str] = None,  # reserved
    device: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,  # reserved
    disable_fp8_kv: bool = False,  # reserved
    **_: Any,
) -> Dict[str, Any]:
    """Run DeepSeek OCR for the provided files."""

    requested_stub = bool(allow_stub)
    del log_dir, allow_stub, allow_cli, persist_engine, precision
    del gpu_memory_utilization, disable_fp8_kv

    if requested_stub or os.environ.get("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "0") == "1":
        raise RuntimeError(
            "DeepSeek stub execution has been removed. "
            "Unset GLOSSAPI_DEEPSEEK_ALLOW_STUB and configure the real DeepSeek runtime."
        )

    file_list = [str(f) for f in files or []]
    if not file_list:
        return {}

    input_root = Path(getattr(self_ref, "input_dir", ".")).resolve()
    out_root = Path(output_dir) if output_dir else Path(getattr(self_ref, "output_dir", input_root))
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_root = Path(
        model_dir
        or os.environ.get("GLOSSAPI_DEEPSEEK_MODEL_DIR", "")
        or (REPO_ROOT / "deepseek-ocr-2-model" / "DeepSeek-OCR-2")
    )
    if not model_root.exists():
        raise FileNotFoundError(
            "DeepSeek model directory not found. Set model_dir or GLOSSAPI_DEEPSEEK_MODEL_DIR."
        )

    script_path = Path(
        vllm_script
        or os.environ.get("GLOSSAPI_DEEPSEEK_RUNNER_SCRIPT", "")
        or DEFAULT_SCRIPT
    )
    if not script_path.exists():
        raise FileNotFoundError(f"DeepSeek OCR runner script not found: {script_path}")

    python_exe = Path(
        python_bin
        or os.environ.get("GLOSSAPI_DEEPSEEK_PYTHON", "")
        or os.environ.get("GLOSSAPI_DEEPSEEK_TEST_PYTHON", "")
        or sys.executable
    )
    if not python_exe.exists():
        raise FileNotFoundError(f"DeepSeek Python interpreter not found: {python_exe}")

    _run_cli(
        input_dir=input_root,
        output_dir=out_root,
        files=file_list,
        model_dir=model_root,
        python_bin=python_exe,
        script=script_path,
        max_pages=max_pages,
        content_debug=content_debug,
        device=device,
    )

    results: Dict[str, Any] = {}
    for name in file_list:
        pdf_path = (input_root / name).resolve()
        stem = Path(name).stem
        md_path = md_dir / f"{stem}.md"
        metrics_path = metrics_dir / f"{stem}.metrics.json"
        if not md_path.exists():
            raise FileNotFoundError(f"DeepSeek OCR did not produce markdown for {name}: {md_path}")
        if not md_path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"DeepSeek OCR produced empty markdown for {name}: {md_path}")
        page_count = _page_count(pdf_path)
        if metrics_path.exists():
            try:
                results[stem] = json.loads(metrics_path.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        results[stem] = {"page_count": page_count}
        metrics_path.write_text(json.dumps(results[stem], indent=2), encoding="utf-8")

    return results
