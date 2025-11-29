"""DeepSeek OCR runner with stub and optional CLI dispatch."""

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
    python_bin: Optional[Path],
    script: Path,
    max_pages: Optional[int],
    content_debug: bool,
    gpu_memory_utilization: Optional[float] = None,
    disable_fp8_kv: bool = False,
) -> None:
    python_exe = Path(python_bin) if python_bin else Path(sys.executable)
    cmd: List[str] = [
        str(python_exe),
        str(script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]
    if max_pages is not None:
        cmd += ["--max-pages", str(max_pages)]
    if content_debug:
        cmd.append("--content-debug")
    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if disable_fp8_kv:
        cmd.append("--no-fp8-kv")

    env = os.environ.copy()
    if shutil.which("cc1plus", path=env.get("PATH", "")) is None:
        # FlashInfer JIT (via vLLM) needs a C++ toolchain; add a known cc1plus location if missing.
        for candidate in sorted(Path("/usr/lib/gcc/x86_64-linux-gnu").glob("*/cc1plus")):
            env["PATH"] = f"{candidate.parent}:{env.get('PATH','')}"
            break
    ld_path = env.get("GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH")
    if ld_path:
        env["LD_LIBRARY_PATH"] = f"{ld_path}:{env.get('LD_LIBRARY_PATH','')}"

    LOGGER.info("Running DeepSeek CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # nosec: controlled arguments


def _run_one_pdf(pdf_path: Path, md_out: Path, metrics_out: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stub processor for a single PDF."""
    page_count = _page_count(pdf_path)
    max_pages = cfg.get("max_pages")
    if max_pages is not None and page_count:
        page_count = min(page_count, max_pages)

    md_lines = [
        f"# DeepSeek OCR (stub) — {pdf_path.name}",
        "",
        f"Pages: {page_count if page_count else 'unknown'}",
    ]
    if cfg.get("content_debug"):
        md_lines.append("")
        md_lines.append("<!-- content_debug: stub output -->")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    metrics = {"page_count": page_count}
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_for_files(
    self_ref: Any,
    files: Iterable[str],
    *,
    model_dir: Optional[Path] = None,  # kept for API compatibility
    output_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,  # unused placeholder to mirror rapidocr
    max_pages: Optional[int] = None,
    allow_stub: bool = True,
    allow_cli: bool = False,
    python_bin: Optional[Path] = None,
    vllm_script: Optional[Path] = None,
    content_debug: bool = False,
    persist_engine: bool = True,  # placeholder for future session reuse
    precision: Optional[str] = None,  # reserved
    device: Optional[str] = None,  # reserved
    gpu_memory_utilization: Optional[float] = None,
    disable_fp8_kv: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    """Run DeepSeek OCR for the provided files.

    Returns a mapping of stem -> minimal metadata (page_count).
    """

    file_list = [str(f) for f in files or []]
    if not file_list:
        return {}

    input_root = Path(getattr(self_ref, "input_dir", ".")).resolve()
    out_root = Path(output_dir) if output_dir else Path(getattr(self_ref, "output_dir", input_root))
    md_dir = out_root / "markdown"
    metrics_dir = out_root / "json" / "metrics"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env_allow_stub = os.environ.get("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "1") == "1"
    env_allow_cli = os.environ.get("GLOSSAPI_DEEPSEEK_ALLOW_CLI", "0") == "1"

    use_cli = allow_cli or env_allow_cli
    use_stub = allow_stub and env_allow_stub

    script_path = Path(vllm_script) if vllm_script else Path.cwd() / "deepseek-ocr" / "run_pdf_ocr_vllm.py"
    # Optional GPU memory utilization override (env wins over kwarg)
    env_gpu_mem = os.environ.get("GLOSSAPI_DEEPSEEK_GPU_MEMORY_UTILIZATION")
    gpu_mem_fraction = gpu_memory_utilization
    if env_gpu_mem:
        try:
            gpu_mem_fraction = float(env_gpu_mem)
        except Exception:
            gpu_mem_fraction = gpu_memory_utilization
        disable_fp8_kv = disable_fp8_kv or os.environ.get("GLOSSAPI_DEEPSEEK_NO_FP8_KV") == "1"

    if use_cli and script_path.exists():
        try:
            _run_cli(
                input_root,
                out_root,
                python_bin=python_bin,
                script=script_path,
                max_pages=max_pages,
                content_debug=content_debug,
                gpu_memory_utilization=gpu_mem_fraction,
                disable_fp8_kv=disable_fp8_kv,
            )
            results: Dict[str, Any] = {}
            for name in file_list:
                pdf_path = (input_root / name).resolve()
                stem = Path(name).stem
                md_path = md_dir / f"{stem}.md"
                metrics_path = metrics_dir / f"{stem}.metrics.json"
                if not md_path.exists() or not md_path.read_text(encoding="utf-8").strip():
                    placeholder = [
                        f"# DeepSeek OCR — {pdf_path.name}",
                        "",
                        "[[Blank page]]",
                    ]
                    md_path.parent.mkdir(parents=True, exist_ok=True)
                    md_path.write_text("\n".join(placeholder) + "\n", encoding="utf-8")
                page_count = _page_count(pdf_path)
                if not metrics_path.exists():
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    metrics_path.write_text(json.dumps({"page_count": page_count}, indent=2), encoding="utf-8")
                results[stem] = {"page_count": page_count}
            return results
        except Exception as exc:
            if not use_stub:
                raise
            LOGGER.warning("DeepSeek CLI failed (%s); falling back to stub output", exc)

    cfg = {"max_pages": max_pages, "content_debug": content_debug}
    results: Dict[str, Any] = {}
    for name in file_list:
        pdf_path = (input_root / name).resolve()
        stem = Path(name).stem
        md_path = md_dir / f"{stem}.md"
        metrics_path = metrics_dir / f"{stem}.metrics.json"
        results[stem] = _run_one_pdf(pdf_path, md_path, metrics_path, cfg)

    return results
