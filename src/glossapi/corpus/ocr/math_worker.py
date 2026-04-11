"""Multi-GPU math worker implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def gpu_math_worker(
    device_id: int,
    in_dir: str,
    out_dir: str,
    work_q,
    batch_size: int,
    dpi_base: int,
    device: str,
    targets_map: Dict[str, List[Tuple[int, int]]],
    result_q=None,
    status_map=None,
    marker_dir: str | None = None,
) -> None:
    import os as _os
    from pathlib import Path as _Path
    import sys as _sys

    def _ensure_thread_caps() -> None:
        caps = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        for key, value in caps.items():
            _os.environ.setdefault(key, value)
        try:
            _torch = _sys.modules.get("torch")
            if _torch is not None and hasattr(_torch, "set_num_threads"):
                _torch.set_num_threads(1)
        except Exception:
            pass

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    _ensure_thread_caps()
    _status_proxy = status_map
    _marker_path = None
    if marker_dir:
        try:
            _marker_path = _Path(marker_dir).expanduser() / f"gpu{device_id}.current"
        except Exception:
            _marker_path = None

    try:
        _verbose = str(_os.environ.get("GLOSSAPI_WORKER_LOG_VERBOSE", "1")).strip().lower()
        if _verbose not in ("0", "false", "no", "off", ""):
            try:
                import importlib

                _torch = _sys.modules.get("torch")
                if _torch is None:
                    try:
                        _torch = importlib.import_module("torch")  # type: ignore
                    except Exception:
                        _torch = None
                if _torch is not None:
                    _torch_name = (
                        _torch.cuda.get_device_name(0)
                        if getattr(_torch, "cuda", None) and _torch.cuda.is_available()
                        else "no-cuda"
                    )
                else:
                    _torch_name = "unloaded"
            except Exception:
                _torch_name = "unknown"
            try:
                import onnxruntime as _ort  # type: ignore

                _ort_prov = _ort.get_available_providers()
            except Exception:
                _ort_prov = []
            try:
                import subprocess as _sp

                _nvsmi = _sp.run(
                    ["nvidia-smi", "-L"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                _phys = (
                    _nvsmi.stdout.splitlines()[0].strip()
                    if _nvsmi.returncode == 0 and _nvsmi.stdout
                    else ""
                )
            except Exception:
                _phys = ""
            print(
                f"[MATH GPU{device_id}] bound: CUDA_VISIBLE_DEVICES={_os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
                f"pid={_os.getpid()} torch={_torch_name} ORT={_ort_prov}"
            )
            if _phys:
                print(f"[MATH GPU{device_id}] physical: {_phys}")
    except Exception:
        pass

    try:
        from glossapi import Corpus as _Corpus  # type: ignore
    except Exception:
        try:
            _sys.path.insert(0, str((Path(out_dir).resolve().parents[1] / "src").resolve()))
            from glossapi import Corpus as _Corpus  # type: ignore
        except Exception as exc:
            try:
                print(f"[MATH GPU{device_id}] Cannot import glossapi in worker: {exc}")
            except Exception:
                pass
            if result_q is not None:
                try:
                    result_q.put(
                        {
                            "event": "exit",
                            "worker": device_id,
                            "exitcode": 1,
                            "pid": _os.getpid(),
                        }
                    )
                except Exception:
                    pass
            _sys.exit(1)

    corpus = _Corpus(input_dir=in_dir, output_dir=out_dir)
    batch_size = max(1, int(batch_size))
    exit_code = 0

    import queue as _queue

    def _report_failure(err: Exception, items: List[str], *, fatal: bool = False) -> None:
        nonlocal exit_code
        try:
            print(f"[MATH GPU{device_id}] Batch failed ({len(items)}): {err}")
        except Exception:
            pass
        if result_q is not None:
            try:
                result_q.put(
                    {
                        "event": "math_batch",
                        "worker": device_id,
                        "problematic": list(items),
                        "pid": _os.getpid(),
                        "error": str(err),
                    }
                )
            except Exception:
                pass
        if fatal:
            exit_code = 1

    def _quarantine_items(items: List[str]) -> None:
        if not items:
            return
        try:
            downloads_root = _Path(out_dir) / "downloads"
            if not downloads_root.exists():
                return
            quarantine_root = downloads_root / "problematic_math"
            quarantine_root.mkdir(parents=True, exist_ok=True)
            json_root = _Path(out_dir) / "json"
            json_quarantine = json_root / "problematic_math"
            try:
                json_quarantine.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            import shutil as _shutil

            for stem in items:
                name = str(stem).strip()
                if not name:
                    continue
                pdf_src = downloads_root / f"{name}.pdf"
                if pdf_src.exists():
                    dst = quarantine_root / pdf_src.name
                    if not dst.exists():
                        _shutil.copy2(pdf_src, dst)
                if json_root.exists():
                    for suffix in (
                        ".docling.json.zst",
                        ".docling.json",
                        ".latex_map.jsonl",
                    ):
                        src = json_root / f"{name}{suffix}"
                        if src.exists():
                            dst = json_quarantine / src.name
                            if not dst.exists():
                                _shutil.copy2(src, dst)
        except Exception as exc:
            try:
                print(f"[MATH GPU{device_id}] Quarantine failed: {exc}")
            except Exception:
                pass

    pending: List[str] = []
    try:
        while True:
            try:
                stem = work_q.get(timeout=0.5)
            except _queue.Empty:
                if pending:
                    batch = list(pending)
                    pending.clear()
                else:
                    break
            else:
                if stem is None:
                    if pending:
                        batch = list(pending)
                        pending.clear()
                    else:
                        break
                else:
                    pending.append(str(stem))
                    if len(pending) < batch_size:
                        continue
                    batch = list(pending)
                    pending.clear()

            try:
                if _status_proxy is not None:
                    _status_proxy[device_id] = list(batch)
                if _marker_path is not None:
                    _marker_path.write_text("\n".join(batch), encoding="utf-8")
                picks = {stem: targets_map.get(stem) for stem in batch if stem in targets_map}
                corpus.formula_enrich_from_json(
                    files=batch,
                    device=device,
                    batch_size=batch_size,
                    dpi_base=dpi_base,
                    targets_by_stem=picks or None,
                )
                if result_q is not None:
                    result_q.put(
                        {
                            "event": "math_batch",
                            "worker": device_id,
                            "problematic": [],
                            "processed": list(batch),
                            "pid": _os.getpid(),
                        }
                    )
            except Exception as exc:
                _quarantine_items(batch)
                _report_failure(exc, batch)
            finally:
                if _status_proxy is not None:
                    _status_proxy.pop(device_id, None)
                if _marker_path is not None:
                    try:
                        _marker_path.unlink(missing_ok=True)
                    except Exception:
                        pass
    except Exception as exc:
        _report_failure(exc, list(pending), fatal=True)
    finally:
        if result_q is not None:
            try:
                result_q.put(
                    {
                        "event": "exit",
                        "worker": device_id,
                        "exitcode": exit_code,
                        "pid": _os.getpid(),
                    }
                )
            except Exception:
                pass
    _sys.exit(exit_code)
