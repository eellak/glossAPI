"""Bounded, in-worker ToUnicode pre-pass for Phase-1 extraction."""
from __future__ import annotations

from ..ledger.journal import observe, observe_store, observe_source_rows

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import threading
import time
import tempfile


class _RepairDeadline(BaseException):
    """Bypass E1's best-effort Exception handlers when the budget expires."""


@contextmanager
def _deadline(seconds: float):
    # Extraction workers run on their main thread. Never run an unbounded
    # pre-pass in a caller's background thread or on a platform without timers.
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise RuntimeError("Font repair requires a main-thread POSIX deadline")
    if seconds <= 0:
        raise _RepairDeadline("Font repair timeout: per-file budget exhausted")
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0] or old_timer[1]:
        raise RuntimeError("Font repair cannot replace an active SIGALRM timer")

    def expire(signum, frame):
        raise _RepairDeadline("Font repair timeout: per-file budget exhausted")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def repair_for_extract(source: Path, root: Path, *, enabled: bool, deadline: float, backend_name: str | None = None):
    """Return (input path, metadata); any pre-pass failure keeps the source.

    Reuse is based on PDF mtime; saved extract metadata supplies the original
    font count/rates without reanalyzing. Missing cache metadata stays nullable.
    """
    data = {"font_repair_status": "skipped", "font_repair_backend": "none", "font_repair_fonts": 0,
            "font_repair_valid_rate_before": None, "font_repair_valid_rate_after": None,
            "font_repair_error": ""}
    if not enabled or source.suffix.lower() != ".pdf":
        observe("repair_disabled_or_non_pdf", path=source)
        return source, data
    target = root / "repaired_pdfs" / f"{source.stem}.pdf"
    try:
        with _deadline(deadline - time.monotonic()):
            from . import text_stats
            from .selector import get_backend
            from .protection import protect_sources

            def rate(path):
                if not os.environ.get("GLOSSAPI_GREEK_WORDLIST"):
                    return None
                try:
                    return text_stats.extract_text_stats(path, max_pages=20)["totals"]["dictionary_valid_rate"]
                except Exception:
                    return None

            # A sidecar directory symlink must never turn this into a download overwrite.
            if target.resolve().is_relative_to((root / "downloads").resolve()):
                raise ValueError("Repaired PDF path resolves inside downloads/")
            protect_sources(target, (source,))
            if target.is_file() and target.stat().st_mtime_ns > source.stat().st_mtime_ns:
                data["font_repair_backend"] = "unknown"
                sidecar = root / "sidecars" / "extract" / f"{source.stem}.json"
                try:
                    saved = json.loads(sidecar.read_text(encoding="utf-8"))
                    if saved.get("font_repair_status") == "repaired":
                        for key in data:
                            data[key] = saved.get(key, data[key])
                except Exception:
                    pass
                data["font_repair_fonts"] = max(0, int(data["font_repair_fonts"] or 0))
                data["font_repair_valid_rate_before"] = rate(source)
                data["font_repair_valid_rate_after"] = rate(target)
                data["font_repair_status"] = "repaired"
                data["font_repair_error"] = ""
                observe("repair_cache_observed", path=source, repair=data, cache_path=str(target))
                return target, data

            backend = get_backend(backend_name) if backend_name is not None else get_backend()
            data["font_repair_backend"] = backend.name
            # Analyse once: a backend with an analyse-once path hands its analysis to repair_pdf, so the
            # repair neither re-inspects fonts nor re-samples text presence.
            analyze_once = getattr(backend, "analyze_for_repair", None)
            if analyze_once is not None:
                report, analysis = analyze_once(source)
            else:
                report, analysis = backend.analyze_pdf(source), None
            from dataclasses import asdict, is_dataclass
            observe("legacy_observed", path=source, font_analysis=asdict(report) if is_dataclass(report) else {"unavailable": "unknown_not_recorded"})
            if report.error:
                raise ValueError(report.error)

            data["font_repair_valid_rate_before"] = rate(source)
            eligible = report.category in {"repairable", "mixed"} and any(
                font.repair_possible for font in report.fonts)
            if eligible:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Publish only after metrics and deadline checks, so a failed
                # pre-pass never leaves a fresh PDF to reuse on the next run.
                with tempfile.TemporaryDirectory(prefix=".extract-font-", dir=target.parent) as staging:
                    staged_pdf = Path(staging) / target.name
                    result = (backend.repair_pdf(source, staged_pdf, analysis=analysis) if analysis is not None
                              else backend.repair_pdf(source, staged_pdf))
                    if result.error or not result.written or result.repaired_fonts < 1:
                        raise ValueError(result.error or "Repair produced no repaired fonts")
                    data["font_repair_fonts"] = int(result.repaired_fonts)
                    data["font_repair_valid_rate_after"] = rate(staged_pdf)
                    if time.monotonic() >= deadline:
                        raise _RepairDeadline("Font repair timeout: per-file budget exhausted")
                    protect_sources(target, (source,))
                    os.replace(staged_pdf, target)
                data["font_repair_status"] = "repaired"
                observe("font_repaired", path=source, repair=data, repaired_path=str(target))
                return target, data
            data["font_repair_status"] = (
                "no_text_layer" if report.category == "no_text_layer" else
                "not_needed" if report.category == "consistent" else "unsupported")
            data["font_repair_valid_rate_after"] = data["font_repair_valid_rate_before"]
    except (_RepairDeadline, Exception) as exc:
        observe("font_repair_error", path=source, error=exc, fallback="original_pdf")
        data.update(font_repair_status="error", font_repair_fonts=0,
                    font_repair_valid_rate_after=None,
                    font_repair_error=f"{type(exc).__name__}: {exc}")
    observe({"not_needed":"font_not_needed", "unsupported":"font_unsupported", "no_text_layer":"font_no_text_layer"}.get(data["font_repair_status"], "legacy_observed"), path=source, repair=data)
    return source, data
