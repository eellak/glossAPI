from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from glossapi import Corpus


def _parse_devices(spec: str | None) -> Optional[List[int]]:
    if spec is None:
        return None
    spec = spec.strip()
    if not spec or spec.lower() == "auto":
        return None
    devices: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            devices.append(int(token))
        except ValueError as exc:  # pragma: no cover - validated in CLI
            raise argparse.ArgumentTypeError(f"Invalid GPU id '{token}' in --devices") from exc
    if not devices:
        raise argparse.ArgumentTypeError("--devices must specify at least one GPU id or 'auto'")
    return devices


def _collect_filenames(download_dir: Path, patterns: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        matches: List[Path] = []
        direct = download_dir / pattern
        if direct.exists():
            matches = [direct]
        else:
            matches = list(download_dir.glob(pattern))
            if not matches:
                candidate = Path(pattern)
                if candidate.is_file():
                    matches = [candidate]
        if not matches:
            print(f"[ocr_gpu_batch] Warning: no files matched '{pattern}'", file=sys.stderr)
            continue
        for path in matches:
            name = path.name
            if name not in seen:
                seen.add(name)
                resolved.append(name)
    return resolved


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.ocr_gpu_batch",
        description="Run GlossAPI OCR/multi-GPU extraction for a batch of PDFs.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF filenames or glob patterns relative to <base-dir>/downloads/",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Corpus directory (expects downloads/, markdown/, etc.). Default: current working directory.",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Comma-separated GPU ids or 'auto' to detect GPUs automatically (default).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for per-worker logs (defaults to <base-dir>/logs/ocr_workers).",
    )
    parser.add_argument(
        "--phase1-backend",
        default="docling",
        choices=["auto", "safe", "docling"],
        help="Phase-1 backend for extraction (default: docling).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files with existing Markdown outputs.",
    )
    parser.add_argument(
        "--emit-formula-index",
        action="store_true",
        help="Request json/<stem>.formula_index.jsonl even when math enrichment is disabled.",
    )
    parser.add_argument(
        "--math",
        dest="math_enrich",
        action="store_true",
        help="Enable math/code enrichment during extraction.",
    )
    parser.add_argument(
        "--no-math",
        dest="math_enrich",
        action="store_false",
        help="Disable math/code enrichment (default).",
    )
    parser.set_defaults(math_enrich=False)
    parser.add_argument(
        "--force-ocr",
        dest="force_ocr",
        action="store_true",
        help="Force GPU OCR during extraction (default).",
    )
    parser.add_argument(
        "--no-force-ocr",
        dest="force_ocr",
        action="store_false",
        help="Skip forced OCR (only run math/layout).",
    )
    parser.set_defaults(force_ocr=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve filenames and exit without launching extraction.",
    )

    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        parser.error(f"Base directory does not exist: {base_dir}")
    download_dir = base_dir / "downloads"
    if not download_dir.exists():
        parser.error(f"Downloads directory not found: {download_dir}")

    filenames = _collect_filenames(download_dir, args.inputs)
    if not filenames:
        parser.error("No matching PDFs found. Nothing to do.")

    log_dir = Path(args.log_dir).expanduser() if args.log_dir else (base_dir / "logs" / "ocr_workers")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    os.environ["GLOSSAPI_WORKER_LOG_DIR"] = str(log_dir)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    device_list = _parse_devices(args.devices)

    emit_formula_index = bool(args.emit_formula_index or args.math_enrich)

    print(f"[ocr_gpu_batch] Base dir: {base_dir}")
    print(f"[ocr_gpu_batch] Downloads dir: {download_dir}")
    print(f"[ocr_gpu_batch] Selected {len(filenames)} file(s): {', '.join(filenames)}")
    if device_list is None:
        print("[ocr_gpu_batch] Devices: auto-detect")
    else:
        print(f"[ocr_gpu_batch] Devices: {', '.join(str(d) for d in device_list)}")
    print(f"[ocr_gpu_batch] Worker logs: {log_dir}")

    if args.dry_run:
        return 0

    corpus = Corpus(input_dir=base_dir, output_dir=base_dir)

    corpus.extract(
        input_format="pdf",
        accel_type="CUDA",
        filenames=filenames,
        use_gpus="multi",
        devices=device_list,
        force_ocr=bool(args.force_ocr),
        formula_enrichment=bool(args.math_enrich),
        code_enrichment=False,
        skip_existing=bool(args.skip_existing),
        export_doc_json=True,
        emit_formula_index=emit_formula_index,
        phase1_backend=args.phase1_backend,
    )

    print("[ocr_gpu_batch] Extraction complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

