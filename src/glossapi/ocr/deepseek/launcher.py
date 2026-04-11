"""Worker-launch helpers for DeepSeek OCR subprocess execution."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from .defaults import DEFAULT_MAX_NEW_TOKENS

LOGGER = logging.getLogger(__name__)


def _iter_venv_library_dirs(venv_root: Path) -> List[str]:
    """Return runtime library directories exposed by a DeepSeek virtualenv."""

    lib_dirs: List[str] = []
    for site_packages in sorted((venv_root / "lib").glob("python*/site-packages")):
        torch_lib = site_packages / "torch" / "lib"
        if torch_lib.is_dir():
            lib_dirs.append(str(torch_lib))
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))
    return lib_dirs


def _build_cli_command(
    input_dir: Path,
    output_dir: Path,
    *,
    files: List[str],
    page_ranges: Optional[List[str]],
    model_dir: Path,
    python_bin: Optional[Path],
    script: Path,
    max_pages: Optional[int],
    content_debug: bool,
    device: Optional[str],
    ocr_profile: str,
    prompt_override: Optional[str],
    attn_backend: str,
    base_size: Optional[int],
    image_size: Optional[int],
    crop_mode: Optional[bool],
    render_dpi: Optional[int],
    max_new_tokens: Optional[int],
    repetition_penalty: Optional[float],
    no_repeat_ngram_size: Optional[int],
    runtime_backend: str,
    vllm_batch_size: Optional[int],
    gpu_memory_utilization: Optional[float],
    disable_fp8_kv: bool,
    repair_mode: Optional[str],
    repair_exec_batch_target_pages: Optional[int] = None,
    repair_exec_batch_target_items: Optional[int] = None,
    work_db: Optional[Path] = None,
    worker_id: Optional[str] = None,
    worker_runtime_file: Optional[Path] = None,
    work_stale_after_sec: Optional[float] = None,
    work_heartbeat_sec: Optional[float] = None,
    work_max_attempts: Optional[int] = None,
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
    if page_ranges:
        cmd += ["--page-ranges", *page_ranges]
    if max_pages is not None:
        cmd += ["--max-pages", str(max_pages)]
    if content_debug:
        cmd.append("--content-debug")
    if device:
        cmd += ["--device", str(device)]
    if ocr_profile:
        cmd += ["--ocr-profile", str(ocr_profile)]
    if prompt_override:
        cmd += ["--prompt-override", str(prompt_override)]
    if attn_backend:
        cmd += ["--attn-backend", str(attn_backend)]
    if base_size is not None:
        cmd += ["--base-size", str(int(base_size))]
    if image_size is not None:
        cmd += ["--image-size", str(int(image_size))]
    if crop_mode is True:
        cmd.append("--crop-mode")
    elif crop_mode is False:
        cmd.append("--no-crop-mode")
    if render_dpi is not None:
        cmd += ["--render-dpi", str(int(render_dpi))]
    if max_new_tokens is not None:
        cmd += ["--max-new-tokens", str(int(max_new_tokens))]
    if work_db is not None:
        cmd += ["--work-db", str(work_db)]
    if worker_id:
        cmd += ["--worker-id", str(worker_id)]
    if worker_runtime_file is not None:
        cmd += ["--worker-runtime-file", str(worker_runtime_file)]
    if work_stale_after_sec is not None:
        cmd += ["--work-stale-after-sec", str(float(work_stale_after_sec))]
    if work_heartbeat_sec is not None:
        cmd += ["--work-heartbeat-sec", str(float(work_heartbeat_sec))]
    if work_max_attempts is not None:
        cmd += ["--work-max-attempts", str(int(work_max_attempts))]
    if repetition_penalty is not None:
        cmd += ["--repetition-penalty", str(float(repetition_penalty))]
    if no_repeat_ngram_size is not None:
        cmd += ["--no-repeat-ngram-size", str(int(no_repeat_ngram_size))]
    runtime_backend_norm = str(runtime_backend or "transformers").strip().lower()
    if runtime_backend_norm == "vllm":
        if vllm_batch_size is not None:
            cmd += ["--batch-size", str(int(vllm_batch_size))]
        if gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(float(gpu_memory_utilization))]
        if disable_fp8_kv:
            cmd.append("--disable-fp8-kv")
        if repair_mode:
            cmd += ["--repair-mode", str(repair_mode)]
        if repair_exec_batch_target_pages is not None:
            cmd += ["--repair-exec-batch-target-pages", str(int(repair_exec_batch_target_pages))]
        if repair_exec_batch_target_items is not None:
            cmd += ["--repair-exec-batch-target-items", str(int(repair_exec_batch_target_items))]
    return cmd


def _build_env(
    *,
    python_bin: Optional[Path],
    visible_device: Optional[int] = None,
    script: Optional[Path] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    if python_bin:
        python_path = Path(python_bin).expanduser()
        venv_bin = str(python_path.parent)
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(python_path.parent.parent)
    if script is not None:
        script_path = Path(script).expanduser().resolve()
        src_root = next((parent for parent in script_path.parents if (parent / "glossapi").is_dir()), None)
        if src_root is not None:
            src_root_str = str(src_root)
            existing_pythonpath = str(env.get("PYTHONPATH", "")).strip()
            pythonpath_entries = [src_root_str]
            if existing_pythonpath:
                pythonpath_entries.extend(
                    entry
                    for entry in existing_pythonpath.split(os.pathsep)
                    if entry and entry != src_root_str
                )
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.pop("PYTHONHOME", None)
    if visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    if shutil.which("cc1plus", path=env.get("PATH", "")) is None:
        for candidate in sorted(Path("/usr/lib/gcc/x86_64-linux-gnu").glob("*/cc1plus")):
            env["PATH"] = f"{candidate.parent}:{env.get('PATH', '')}"
            break
    ld_entries: List[str] = []
    if python_bin:
        venv_root = Path(python_bin).expanduser().parent.parent
        # vLLM's compiled extension depends on both the wheel-managed CUDA libs
        # and the PyTorch shared objects shipped in torch/lib.
        ld_entries.extend(_iter_venv_library_dirs(venv_root))
    ld_path = env.get("GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH")
    if ld_path:
        ld_entries.extend(entry for entry in str(ld_path).split(os.pathsep) if entry)
    existing_ld = str(env.get("LD_LIBRARY_PATH", "")).strip()
    if existing_ld:
        ld_entries.extend(entry for entry in existing_ld.split(os.pathsep) if entry)
    if ld_entries:
        deduped: List[str] = []
        seen: Set[str] = set()
        for entry in ld_entries:
            if entry and entry not in seen:
                seen.add(entry)
                deduped.append(entry)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(deduped)
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
    ocr_profile: str,
    prompt_override: Optional[str],
    attn_backend: str,
    base_size: Optional[int],
    image_size: Optional[int],
    crop_mode: Optional[bool],
    render_dpi: Optional[int],
    max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS,
    repetition_penalty: Optional[float],
    no_repeat_ngram_size: Optional[int],
    runtime_backend: str,
    vllm_batch_size: Optional[int],
    gpu_memory_utilization: Optional[float],
    disable_fp8_kv: bool,
    repair_mode: Optional[str],
    repair_exec_batch_target_pages: Optional[int],
    repair_exec_batch_target_items: Optional[int],
    visible_device: Optional[int] = None,
) -> None:
    cmd = _build_cli_command(
        input_dir=input_dir,
        output_dir=output_dir,
        files=files,
        page_ranges=None,
        model_dir=model_dir,
        python_bin=python_bin,
        script=script,
        max_pages=max_pages,
        content_debug=content_debug,
        device=device,
        ocr_profile=ocr_profile,
        prompt_override=prompt_override,
        attn_backend=attn_backend,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        render_dpi=render_dpi,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        runtime_backend=runtime_backend,
        vllm_batch_size=vllm_batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_fp8_kv=disable_fp8_kv,
        repair_mode=repair_mode,
        repair_exec_batch_target_pages=repair_exec_batch_target_pages,
        repair_exec_batch_target_items=repair_exec_batch_target_items,
    )
    env = _build_env(python_bin=python_bin, visible_device=visible_device, script=script)

    LOGGER.info("Running DeepSeek OCR CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # nosec: controlled arguments
