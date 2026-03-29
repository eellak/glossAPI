"""DeepSeek OCR runner."""

from __future__ import annotations

from contextlib import ExitStack
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


def _build_cli_command(
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
) -> List[str]:
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
    return cmd


def _build_env(*, python_bin: Optional[Path], visible_device: Optional[int] = None) -> Dict[str, str]:
    env = os.environ.copy()
    if python_bin:
        python_path = Path(python_bin).expanduser()
        venv_bin = str(python_path.parent)
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(python_path.parent.parent)
    env.pop("PYTHONHOME", None)
    if visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    if shutil.which("cc1plus", path=env.get("PATH", "")) is None:
        for candidate in sorted(Path("/usr/lib/gcc/x86_64-linux-gnu").glob("*/cc1plus")):
            env["PATH"] = f"{candidate.parent}:{env.get('PATH', '')}"
            break
    ld_path = env.get("GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH")
    if ld_path:
        env["LD_LIBRARY_PATH"] = f"{ld_path}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


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
    visible_device: Optional[int] = None,
) -> None:
    cmd = _build_cli_command(
        input_dir=input_dir,
        output_dir=output_dir,
        files=files,
        model_dir=model_dir,
        python_bin=python_bin,
        script=script,
        max_pages=max_pages,
        content_debug=content_debug,
        device=device,
    )
    env = _build_env(python_bin=python_bin, visible_device=visible_device)

    LOGGER.info("Running DeepSeek OCR CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # nosec: controlled arguments


def _parse_device_index(device: Optional[str]) -> Optional[int]:
    if not device:
        return None
    value = str(device).strip().lower()
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _detect_visible_gpus() -> List[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        parsed = [piece.strip() for piece in visible.split(",") if piece.strip()]
        if parsed and all(piece.isdigit() for piece in parsed):
            return [int(piece) for piece in parsed]
    torch_mod = None
    try:  # pragma: no cover - best effort
        import torch as torch_mod  # type: ignore
    except Exception:  # pragma: no cover - optional import
        torch_mod = None
    if torch_mod is not None:
        try:
            if torch_mod.cuda.is_available():
                return list(range(torch_mod.cuda.device_count()))
        except Exception:
            pass
    try:  # pragma: no cover - shell fallback
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        devices: List[int] = []
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("GPU "):
                    prefix = line.split(":", 1)[0]
                    idx = prefix.split()[1]
                    if idx.isdigit():
                        devices.append(int(idx))
        return devices
    except Exception:
        return []


def _resolve_lane_devices(
    *,
    use_gpus: Optional[str],
    devices: Optional[List[int]],
    workers_per_gpu: int,
    device: Optional[str],
) -> List[int]:
    if devices:
        resolved = [int(dev) for dev in devices]
        if resolved:
            return resolved
    if str(use_gpus or "single").strip().lower() == "multi":
        resolved = _detect_visible_gpus()
        if resolved:
            return resolved
    if workers_per_gpu > 1:
        from_device = _parse_device_index(device)
        if from_device is not None:
            return [from_device]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            first = visible.split(",", 1)[0].strip()
            if first.isdigit():
                return [int(first)]
        return [0]
    return []


def _effective_page_count(pdf_path: Path, max_pages: Optional[int]) -> int:
    count = _page_count(pdf_path)
    if max_pages is not None and count > 0:
        return min(count, int(max_pages))
    return max(1, count)


def _plan_lanes(
    *,
    file_list: List[str],
    input_root: Path,
    lane_devices: List[int],
    workers_per_gpu: int,
    max_pages: Optional[int],
) -> List[Dict[str, Any]]:
    lanes: List[Dict[str, Any]] = []
    lane_id = 0
    for visible_device in lane_devices:
        for _ in range(max(1, int(workers_per_gpu))):
            lanes.append(
                {
                    "lane_id": lane_id,
                    "visible_device": int(visible_device),
                    "files": [],
                    "weight": 0,
                }
            )
            lane_id += 1
    if not lanes:
        return []

    weighted_files = []
    for name in file_list:
        pdf_path = (input_root / name).resolve()
        weighted_files.append((name, _effective_page_count(pdf_path, max_pages)))
    weighted_files.sort(key=lambda item: (-item[1], item[0]))

    for name, weight in weighted_files:
        lane = min(lanes, key=lambda item: int(item["weight"]))
        lane["files"].append(name)
        lane["weight"] = int(lane["weight"]) + int(weight)
    return lanes


def _run_multi_cli(
    *,
    input_root: Path,
    out_root: Path,
    file_list: List[str],
    lane_devices: List[int],
    workers_per_gpu: int,
    model_root: Path,
    python_exe: Path,
    script_path: Path,
    max_pages: Optional[int],
    content_debug: bool,
    log_dir: Path,
) -> None:
    lanes = _plan_lanes(
        file_list=file_list,
        input_root=input_root,
        lane_devices=lane_devices,
        workers_per_gpu=workers_per_gpu,
        max_pages=max_pages,
    )
    if not lanes:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    with ExitStack() as stack:
        procs = []
        for lane in lanes:
            lane_files = list(lane["files"])
            if not lane_files:
                continue
            visible_device = int(lane["visible_device"])
            log_path = log_dir / f"lane_{lane['lane_id']}_gpu{visible_device}.log"
            fh = stack.enter_context(log_path.open("w", encoding="utf-8"))
            cmd = _build_cli_command(
                input_dir=input_root,
                output_dir=out_root,
                files=lane_files,
                model_dir=model_root,
                python_bin=python_exe,
                script=script_path,
                max_pages=max_pages,
                content_debug=content_debug,
                device="cuda",
            )
            env = _build_env(python_bin=python_exe, visible_device=visible_device)
            LOGGER.info(
                "Running DeepSeek OCR lane=%s visible_gpu=%s files=%d weight=%d: %s",
                lane["lane_id"],
                visible_device,
                len(lane_files),
                lane["weight"],
                " ".join(cmd),
            )
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)  # nosec: controlled args
            procs.append((lane, log_path, proc))

        for lane, log_path, proc in procs:
            rc = proc.wait()
            if rc != 0:
                failures.append(
                    f"lane={lane['lane_id']} gpu={lane['visible_device']} rc={rc} log={log_path}"
                )
    if failures:
        raise RuntimeError("DeepSeek OCR multi-worker failure(s): " + "; ".join(failures))


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
    use_gpus: Optional[str] = None,
    devices: Optional[List[int]] = None,
    workers_per_gpu: int = 1,
    gpu_memory_utilization: Optional[float] = None,  # reserved
    disable_fp8_kv: bool = False,  # reserved
    **_: Any,
) -> Dict[str, Any]:
    """Run DeepSeek OCR for the provided files."""

    requested_stub = bool(allow_stub)
    del allow_stub, allow_cli, persist_engine, precision
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

    lane_devices = _resolve_lane_devices(
        use_gpus=use_gpus,
        devices=devices,
        workers_per_gpu=int(max(1, workers_per_gpu)),
        device=device,
    )
    multi_requested = str(use_gpus or "single").strip().lower() == "multi" or int(max(1, workers_per_gpu)) > 1
    if multi_requested and lane_devices:
        _run_multi_cli(
            input_root=input_root,
            out_root=out_root,
            file_list=file_list,
            lane_devices=lane_devices,
            workers_per_gpu=int(max(1, workers_per_gpu)),
            model_root=model_root,
            python_exe=python_exe,
            script_path=script_path,
            max_pages=max_pages,
            content_debug=content_debug,
            log_dir=Path(log_dir) if log_dir else (out_root / "logs" / "deepseek_workers"),
        )
    else:
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
