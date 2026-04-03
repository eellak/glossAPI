from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from glossapi import Corpus


HEADER_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}\s+\S")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.extract_checkpoint_benchmark",
        description=(
            "Run a strict Phase-1 extraction benchmark on a fixed PDF set and audit "
            "canonical markdown outputs for presence, byte size, header counts, and drift."
        ),
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--report-path", required=True)
    p.add_argument("--baseline-report", default="")
    p.add_argument("--phase1-backend", default="docling", choices=["auto", "safe", "docling"])
    p.add_argument("--accel-type", default="CUDA")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--use-gpus", default="single", choices=["single", "multi"])
    p.add_argument("--devices", nargs="*", type=int, default=None)
    p.add_argument("--workers-per-device", type=int, default=1)
    p.add_argument("--benchmark-mode", action="store_true")
    p.add_argument("--filenames", nargs="*", default=[])
    p.add_argument("--clean-output-dir", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _count_pdf_pages(pdf_path: Path) -> int:
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        return int(len(PdfReader(str(pdf_path)).pages))
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader  # type: ignore

        return int(len(PdfReader(str(pdf_path)).pages))
    except Exception as exc:
        raise RuntimeError(f"Unable to count PDF pages for {pdf_path}: {exc}") from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _markdown_headers(text: str) -> int:
    return int(len(HEADER_RE.findall(text or "")))


def _inventory_markdown(markdown_dir: Path, *, pdf_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    inventory: Dict[str, Dict[str, Any]] = {}
    for pdf_path in pdf_paths:
        stem = pdf_path.stem
        md_path = markdown_dir / f"{stem}.md"
        present = md_path.exists()
        payload = md_path.read_bytes() if present else b""
        text = payload.decode("utf-8") if present else ""
        inventory[stem] = {
            "filename": pdf_path.name,
            "markdown_path": str(md_path),
            "present": bool(present),
            "byte_size": int(len(payload)),
            "header_count": _markdown_headers(text),
            "sha256": _sha256_bytes(payload) if present else None,
        }
    return inventory


def _compare_inventory(
    current_inventory: Dict[str, Dict[str, Any]],
    baseline_inventory: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    added = []
    missing = []
    byte_size_changed = []
    header_count_changed = []
    sha_changed = []
    for stem, current in sorted(current_inventory.items()):
        baseline = baseline_inventory.get(stem)
        if baseline is None:
            added.append(stem)
            continue
        if bool(baseline.get("present")) and not bool(current.get("present")):
            missing.append(stem)
        if int(baseline.get("byte_size", 0)) != int(current.get("byte_size", 0)):
            byte_size_changed.append(stem)
        if int(baseline.get("header_count", 0)) != int(current.get("header_count", 0)):
            header_count_changed.append(stem)
        if baseline.get("sha256") != current.get("sha256"):
            sha_changed.append(stem)
    for stem, baseline in sorted(baseline_inventory.items()):
        if stem in current_inventory:
            continue
        if bool(baseline.get("present")):
            missing.append(stem)
    return {
        "added_markdown": added,
        "missing_markdown": sorted(set(missing)),
        "byte_size_changed": byte_size_changed,
        "header_count_changed": header_count_changed,
        "sha_changed": sha_changed,
    }


def _load_baseline_inventory(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("markdown_inventory") or {})


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if args.filenames:
        selected = {str(name) for name in args.filenames}
        pdf_paths = [path for path in pdf_paths if path.name in selected]
    if not pdf_paths:
        raise SystemExit(f"No PDF files selected under {input_dir}")

    if bool(args.clean_output_dir) and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_pages = int(sum(_count_pdf_pages(path) for path in pdf_paths))
    start_ts = time.time()
    start_perf = time.perf_counter()

    corpus = Corpus(input_dir=input_dir, output_dir=output_dir)
    corpus.extract(
        input_format="pdf",
        accel_type=str(args.accel_type),
        num_threads=int(args.num_threads),
        phase1_backend=str(args.phase1_backend),
        use_gpus=str(args.use_gpus),
        devices=list(args.devices) if args.devices else None,
        workers_per_device=int(args.workers_per_device),
        benchmark_mode=bool(args.benchmark_mode),
        filenames=[path.name for path in pdf_paths],
    )

    elapsed_sec = float(time.perf_counter() - start_perf)
    end_ts = time.time()
    markdown_dir = output_dir / "markdown"
    inventory = _inventory_markdown(markdown_dir, pdf_paths=pdf_paths)
    markdown_present = int(sum(1 for item in inventory.values() if bool(item["present"])))

    report: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "started_at": int(start_ts),
        "finished_at": int(end_ts),
        "elapsed_sec": elapsed_sec,
        "files_total": int(len(pdf_paths)),
        "pages_total": int(total_pages),
        "pages_per_sec": (float(total_pages) / elapsed_sec) if elapsed_sec > 0 else None,
        "phase1_backend": str(args.phase1_backend),
        "accel_type": str(args.accel_type),
        "num_threads": int(args.num_threads),
        "use_gpus": str(args.use_gpus),
        "devices": list(args.devices) if args.devices else [],
        "workers_per_device": int(args.workers_per_device),
        "benchmark_mode": bool(args.benchmark_mode),
        "markdown_present": markdown_present,
        "markdown_missing": int(len(pdf_paths) - markdown_present),
        "markdown_inventory": inventory,
    }

    baseline_raw = str(args.baseline_report or "").strip()
    if baseline_raw:
        baseline_path = Path(baseline_raw).expanduser().resolve()
        if baseline_path.exists():
            report["comparison"] = _compare_inventory(
                inventory,
                _load_baseline_inventory(baseline_path),
            )
        else:
            report["comparison_error"] = f"Baseline report not found: {baseline_path}"

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "files_total": report["files_total"],
        "pages_total": report["pages_total"],
        "elapsed_sec": round(report["elapsed_sec"], 3),
        "pages_per_sec": round(report["pages_per_sec"], 4) if report["pages_per_sec"] is not None else None,
        "markdown_present": report["markdown_present"],
        "markdown_missing": report["markdown_missing"],
        "report_path": str(report_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
