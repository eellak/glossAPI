from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PACKAGE_NAMES = (
    "torch",
    "vllm",
    "transformers",
    "nvidia.cuda_runtime",
    "nvidia.cuda_nvrtc",
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.deepseek_runtime_report",
        description="Print a reproducible DeepSeek OCR runtime report for a GlossAPI checkout.",
    )
    p.add_argument("--repo-root", default=".")
    p.add_argument("--python-bin", default="")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def _detect_python_bin(repo_root: Path, explicit: str) -> Path:
    if str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    candidates = (
        repo_root / "dependency_setup" / ".venvs" / "deepseek" / "bin" / "python",
        repo_root / "dependency_setup" / "deepseek_uv" / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def _read_os_release() -> Dict[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value.strip().strip('"')
    return out


def _run_text(*cmd: str) -> str:
    try:
        completed = subprocess.run(
            list(cmd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return ""
    return completed.stdout.strip()


def _gpu_rows() -> List[Dict[str, str]]:
    text = _run_text(
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": parts[3],
            }
        )
    return rows


def _python_json(python_bin: Path, code: str) -> Dict[str, Any]:
    completed = subprocess.run(
        [str(python_bin), "-c", code],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    try:
        return {"ok": True, "data": json.loads(completed.stdout)}
    except json.JSONDecodeError:
        return {
            "ok": False,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }


def _package_report(python_bin: Path) -> Dict[str, Any]:
    code = """
import importlib
import json
import os
import sys

mods = {}
for name in %s:
    try:
        mod = importlib.import_module(name)
        mods[name] = {
            "version": getattr(mod, "__version__", None),
            "file": getattr(mod, "__file__", None),
        }
    except Exception as exc:
        mods[name] = {"error": repr(exc)}

payload = {
    "python_version": sys.version,
    "executable": sys.executable,
    "virtual_env": os.environ.get("VIRTUAL_ENV"),
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    "packages": mods,
}
print(json.dumps(payload))
""" % (repr(PACKAGE_NAMES),)
    return _python_json(python_bin, code)


def _site_package_nvidia_libs(venv_root: Path) -> List[Path]:
    libs: List[Path] = []
    for site_packages in sorted((venv_root / "lib").glob("python*/site-packages")):
        for lib_dir in sorted((site_packages / "nvidia").glob("*/lib")):
            if lib_dir.is_dir():
                libs.append(lib_dir)
    return libs


def _interesting_libs(lib_dir: Path) -> List[str]:
    names = []
    for child in sorted(lib_dir.iterdir()):
        if not child.is_file():
            continue
        name = child.name
        if any(token in name for token in ("libcudart", "libnvrtc", "libcudnn", "libcuda")):
            names.append(name)
    return names


def _venv_root(python_bin: Path) -> Path:
    return python_bin.parent.parent


def _pip_freeze_subset(python_bin: Path) -> List[str]:
    text = _run_text(str(python_bin), "-m", "pip", "freeze")
    prefixes = (
        "torch",
        "vllm",
        "transformers",
        "nvidia-cuda",
        "nvidia-cudnn",
        "xformers",
        "flash-attn",
    )
    lines = []
    for line in text.splitlines():
        normalized = line.strip().lower()
        if any(normalized.startswith(prefix) for prefix in prefixes):
            lines.append(line.strip())
    return lines


def _report(repo_root: Path, python_bin: Path) -> Dict[str, Any]:
    os_release = _read_os_release()
    venv_root = _venv_root(python_bin)
    lib_dirs = _site_package_nvidia_libs(venv_root)
    return {
        "repo_root": str(repo_root),
        "repo_head": _run_text("git", "-C", str(repo_root), "rev-parse", "HEAD"),
        "hostname": platform.node(),
        "os_release": {
            "PRETTY_NAME": os_release.get("PRETTY_NAME"),
            "VERSION_ID": os_release.get("VERSION_ID"),
        },
        "python_bin": str(python_bin),
        "venv_root": str(venv_root),
        "gpus": _gpu_rows(),
        "python_env": _package_report(python_bin),
        "nvidia_lib_dirs": [
            {
                "path": str(lib_dir),
                "interesting_libs": _interesting_libs(lib_dir),
            }
            for lib_dir in lib_dirs
        ],
        "pip_freeze_subset": _pip_freeze_subset(python_bin),
        "selected_env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
        },
    }


def _print_text(report: Dict[str, Any]) -> None:
    print(f"repo_root: {report['repo_root']}")
    print(f"repo_head: {report['repo_head']}")
    print(f"hostname: {report['hostname']}")
    os_release = report["os_release"]
    print(f"os: {os_release.get('PRETTY_NAME')} (VERSION_ID={os_release.get('VERSION_ID')})")
    print(f"python_bin: {report['python_bin']}")
    print(f"venv_root: {report['venv_root']}")
    print()
    print("gpus:")
    for row in report["gpus"]:
        print(
            f"  - index={row['index']} name={row['name']} "
            f"driver={row['driver_version']} memory_mib={row['memory_total_mib']}"
        )
    print()
    print("python_env:")
    py_env = report["python_env"]
    print(f"  ok: {py_env.get('ok')}")
    if py_env.get("ok"):
        data = py_env["data"]
        print(f"  executable: {data.get('executable')}")
        print(f"  python_version: {data.get('python_version')}")
        print(f"  virtual_env: {data.get('virtual_env')}")
        print(f"  ld_library_path: {data.get('ld_library_path')}")
        for name, package in data.get("packages", {}).items():
            print(f"  {name}: {package}")
    else:
        print(f"  stdout: {py_env.get('stdout')}")
        print(f"  stderr: {py_env.get('stderr')}")
    print()
    print("nvidia_lib_dirs:")
    for item in report["nvidia_lib_dirs"]:
        print(f"  - path: {item['path']}")
        for lib in item["interesting_libs"]:
            print(f"    {lib}")
    print()
    print("pip_freeze_subset:")
    for line in report["pip_freeze_subset"]:
        print(f"  - {line}")
    print()
    print("selected_env:")
    for key, value in report["selected_env"].items():
        print(f"  {key}={value}")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    python_bin = _detect_python_bin(repo_root, str(args.python_bin or ""))
    report = _report(repo_root, python_bin)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_text(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
