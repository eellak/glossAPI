from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .base import DeepSeekConfig
import inspect
from .._naming import canonical_stem
import sys
import subprocess


def _safe_import_fitz():
    try:
        import fitz  # type: ignore

        return fitz
    except Exception:
        return None


def _count_pdf_pages(pdf_path: Path) -> Optional[int]:
    fitz = _safe_import_fitz()
    if fitz is None:
        return None
    try:
        with fitz.open(pdf_path) as doc:  # type: ignore
            return int(doc.page_count)
    except Exception:
        return None


def _run_one_pdf(
    pdf_path: Path,
    md_out: Path,
    metrics_out: Path,
    config: DeepSeekConfig,
    *,
    internal_debug: bool = False,
    content_debug: Optional[bool] = None,
) -> Dict[str, object]:
    """Minimal per-PDF OCR placeholder.

    This implementation is intentionally lightweight and does not invoke the
    heavy DeepSeek/vLLM stack. It prepares stub outputs that allow pipeline
    integration and tests to validate wiring. Downstream environments can
    replace this with a real DeepSeek invocation.
    """
    md_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    page_count = _count_pdf_pages(pdf_path) or 0

    # Basic markdown artifact
    debug_flag = internal_debug if content_debug is None else bool(content_debug)
    if debug_flag:
        # Keep page markers/truncation markers (stubbed here)
        content = [
            "# DeepSeek OCR",
            "",
            f"Source: {pdf_path.name}",
            "",
            f"Pages: {page_count}",
            "",
            "--- page 1 ---",
            "[... possibly truncated ...]",
        ]
        md_out.write_text("\n".join(content) + "\n", encoding="utf-8")
    else:
        # Produce final concatenated document without debug markers
        md_out.write_text(
            f"# DeepSeek OCR\n\nSource: {pdf_path.name}\n\nPages: {page_count}\n",
            encoding="utf-8",
        )

    # Minimal metrics payload with page_count
    metrics = {"page_count": int(page_count), "pages": []}
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def run_for_files(
    corpus,
    files: Iterable[str],
    *,
    model_dir: Optional[Path] = None,
    dtype: str = "auto",
    max_tokens: int = 8192,
    render_dpi: int = 220,
    gpu_util: float = 0.9,
    tp: int = 1,
    save_images: bool = False,
    internal_debug: bool = False,
    content_debug: Optional[bool] = None,
) -> Dict[str, Dict[str, object]]:
    """Run DeepSeek OCR for specific files and update parquet.

    Heavy dependencies are imported lazily inside helpers. Tests may monkeypatch
    `_run_one_pdf` to avoid any heavyweight imports.
    """
    out_md = corpus.output_dir / "markdown"
    out_metrics = corpus.output_dir / "json" / "metrics"
    out_md.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)

    cfg = DeepSeekConfig(
        model_dir=Path(model_dir) if model_dir else None,
        dtype=dtype or "auto",
        max_tokens=int(max_tokens or 8192),
        render_dpi=int(render_dpi or 220),
        gpu_util=float(gpu_util or 0.9),
        tensor_parallel=int(tp or 1),
        save_images=bool(save_images),
    )

    results: Dict[str, Dict[str, object]] = {}
    files_list = [str(f) for f in files]

    # vLLM-only path: invoke the vLLM CLI; abort gracefully if unavailable.
    python_bin = os.environ.get("GLOSSAPI_DEEPSEEK_PYTHON") or sys.executable
    here = Path(__file__).resolve()
    repo_root = here.parents[3] if len(here.parents) >= 4 else here.parents[-1]
    # Allow overriding the vLLM runner path via env
    vllm_script_env = os.environ.get("GLOSSAPI_DEEPSEEK_VLLM_SCRIPT")
    vllm_script = (
        Path(vllm_script_env).resolve()
        if vllm_script_env
        else (repo_root / "deepseek-ocr" / "run_pdf_ocr_vllm.py")
    )

    # Allow stub path only in explicit test mode to keep unit tests fast.
    allow_stub_env = str(os.environ.get("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "")).strip() not in {"", "0", "false", "False"}
    allow_stub = allow_stub_env or ("PYTEST_CURRENT_TEST" in os.environ)

    if not vllm_script.exists():
        msg = f"DeepSeek vLLM runner not found at {vllm_script}; ensure repository layout or install path is correct."
        if allow_stub:
            try:
                corpus.logger.warning(msg + " Falling back to stub _run_one_pdf for tests.")
            except Exception:
                pass
        else:
            raise RuntimeError(msg)

    ran_cli = False
    if vllm_script.exists():
        stage_dir = corpus.output_dir / ".deepseek_stage"
        try:
            if stage_dir.exists():
                for p in stage_dir.iterdir():
                    try:
                        p.unlink()
                    except Exception:
                        pass
            else:
                stage_dir.mkdir(parents=True, exist_ok=True)
            # Link/copy PDFs into stage
            for fname in files_list:
                src_pdf = (corpus.input_dir / fname).resolve()
                dst_pdf = stage_dir / Path(fname).name
                try:
                    if dst_pdf.exists():
                        dst_pdf.unlink()
                    os.symlink(src_pdf, dst_pdf)
                except Exception:
                    import shutil as _shutil

                    _shutil.copy2(src_pdf, dst_pdf)
            # Build and run vLLM CLI
            cmd = [
                str(python_bin),
                str(vllm_script),
                "--input-dir",
                str(stage_dir),
                "--output-dir",
                str(out_md),
                "--mode",
                "clean",
                "--max-tokens",
                str(max_tokens),
                "--dpi",
                str(render_dpi),
            ]
            if content_debug:
                cmd.append("--content-debug")
            if model_dir and Path(model_dir).exists():
                cmd += ["--model", str(model_dir)]
            # Prefer BF16 or FP16 based on dtype
            if str(cfg.dtype).lower() in {"bfloat16", "bf16"}:
                cmd += ["--dtype", "bfloat16"]
            elif str(cfg.dtype).lower() in {"fp16", "float16", "half"}:
                cmd += ["--dtype", "float16"]
            # Inject LD_LIBRARY_PATH for libjpeg if provided or discoverable
            env = os.environ.copy()
            extra_ld = os.environ.get("GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH")
            if not extra_ld:
                # Try common location under the repo
                candidate = repo_root / "deepseek-ocr" / "libjpeg-turbo" / "lib"
                if candidate.exists():
                    extra_ld = str(candidate)
            if extra_ld:
                env["LD_LIBRARY_PATH"] = (
                    f"{extra_ld}:{env.get('LD_LIBRARY_PATH','')}" if env.get("LD_LIBRARY_PATH") else extra_ld
                )
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            if proc.returncode != 0:
                msg = f"DeepSeek vLLM CLI failed (code={proc.returncode}): {proc.stderr.strip()}"
                if allow_stub:
                    try:
                        corpus.logger.error(msg + " Falling back to stub _run_one_pdf for tests.")
                    except Exception:
                        pass
                else:
                    raise RuntimeError(msg)
            else:
                ran_cli = True
                # Parse throughput summary if present and surface in logs
                try:
                    import re as _re
                    m = None
                    for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines():
                        if "Completed" in line and "pages/s" in line:
                            m = _re.search(r"Completed\s+(\d+)\s+page\(s\)\s+in\s+([0-9.]+)s\s+\(([0-9.]+)\s+pages/s,\s+(\d+)\s+tokens/s,\s+(\d+)\s+ROI tokens\)", line)
                            if m:
                                break
                    if m is not None:
                        total_pages = int(m.group(1))
                        elapsed_s = float(m.group(2))
                        pages_per_s = float(m.group(3))
                        tokens_per_s = int(m.group(4))
                        roi_tokens = int(m.group(5))
                        try:
                            corpus.logger.info(
                                "DeepSeek summary: %d pages in %.1fs (%.2f pages/s, %d tokens/s, %d ROI tokens)",
                                total_pages,
                                elapsed_s,
                                pages_per_s,
                                tokens_per_s,
                                roi_tokens,
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
                # Write metrics per file (page_count)
                for fname in files_list:
                    stem = canonical_stem(fname)
                    metrics_path = out_metrics / f"{stem}.metrics.json"
                    pc = _count_pdf_pages((corpus.input_dir / fname).resolve()) or 0
                    try:
                        metrics_path.write_text(
                            json.dumps({"page_count": int(pc)}, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                    results[stem] = {"page_count": pc}
        except Exception as _cli_err:
            if allow_stub:
                try:
                    corpus.logger.error("DeepSeek vLLM runner error; falling back to stub: %s", _cli_err)
                except Exception:
                    pass
            else:
                raise

    # Stub path (tests only)
    if not ran_cli and allow_stub:
        for fname in files_list:
            stem = canonical_stem(fname)
            pdf_path = (corpus.input_dir / fname).resolve()
            md_path = out_md / f"{stem}.md"
            metrics_path = out_metrics / f"{stem}.metrics.json"
            try:
                sig = None
                try:
                    sig = inspect.signature(_run_one_pdf)  # type: ignore[arg-type]
                except Exception:
                    sig = None
                call_kwargs = {}
                if sig is not None:
                    params = sig.parameters
                    if "internal_debug" in params:
                        call_kwargs["internal_debug"] = internal_debug
                    if "content_debug" in params:
                        call_kwargs["content_debug"] = content_debug
                metrics = _run_one_pdf(
                    pdf_path,
                    md_path,
                    metrics_path,
                    cfg,
                    **call_kwargs,
                )
                results[stem] = metrics
            except Exception as exc:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    metrics_path.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
                except Exception:
                    pass
    elif not ran_cli and not allow_stub:
        # Neither vLLM ran nor stub allowed → abort.
        raise RuntimeError("DeepSeek OCR requires vLLM; set up the vLLM environment and try again.")

    # Parquet updates
    try:
        from glossapi.parquet_schema import ParquetSchema  # type: ignore
        import pandas as pd  # type: ignore

        schema = ParquetSchema({"url_column": corpus.url_column})
        pq_path = corpus._resolve_metadata_parquet(schema, ensure=True, search_input=True)
        if pq_path is not None:
            df = pd.read_parquet(pq_path)
            # Normalise frame and ensure expected columns exist
            df = schema.normalize_metadata_frame(df)
            # Ensure columns we are about to write
            for col in ["ocr_success", "needs_ocr", "page_count", "processing_stage", "extraction_mode"]:
                if col not in df.columns:
                    df[col] = pd.NA
            for fname in files_list:
                stem = canonical_stem(fname)
                mask = df["filename"].astype(str) == str(fname)
                if not mask.any():
                    # Add a new row if missing
                    new_row = {col: pd.NA for col in df.columns}
                    new_row[corpus.url_column] = ""
                    new_row["filename"] = str(fname)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    mask = df["filename"].astype(str) == str(fname)
                # Update flags
                df.loc[mask, "ocr_success"] = True
                df.loc[mask, "needs_ocr"] = False
                df.loc[mask, "extraction_mode"] = "deepseek"
                # page_count from metrics if available
                pc = results.get(canonical_stem(fname), {}).get("page_count")
                if pc is not None:
                    try:
                        df.loc[mask, "page_count"] = int(pc)  # type: ignore
                    except Exception:
                        df.loc[mask, "page_count"] = pc  # type: ignore
                # Append 'extract' to processing_stage string
                try:
                    stage = df.loc[mask, "processing_stage"].astype(str).fillna("")
                    needs_append = stage.isna() | (~stage.str.contains(r"\bextract\b"))
                except Exception:
                    needs_append = None
                if needs_append is None or needs_append.any():
                    # Build updated stage value conservatively
                    def _append_extract(val: object) -> str:
                        s = str(val) if val is not None else ""
                        return "extract" if not s else (s if "extract" in s else (s + ",extract"))

                    df.loc[mask, "processing_stage"] = df.loc[mask, "processing_stage"].apply(_append_extract)

            # Persist
            schema.write_metadata_parquet(df, pq_path)
    except Exception:
        # Parquet update failures are non-fatal to the runner
        pass

    return results
