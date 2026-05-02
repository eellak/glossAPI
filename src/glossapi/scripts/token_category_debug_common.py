from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_RUST_EXTENSION_PREBUILD_ATTEMPTED: set[str] = set()


def _latest_tree_mtime(root: Path) -> float:
    latest = 0.0
    if root.is_file():
        try:
            return root.stat().st_mtime
        except OSError:
            return 0.0
    if not root.exists():
        return 0.0
    for path in root.rglob("*"):
        if any(part == "target" for part in path.parts):
            continue
        if not path.is_file():
            continue
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


def load_rust_extension(
    *,
    project_root: Path,
    module_name: str,
    manifest_relative: str,
    required_attrs: Optional[Iterable[str]] = None,
) -> Any:
    required = tuple(required_attrs or ())
    manifest = project_root / manifest_relative
    crate_root = manifest.parent

    def _missing_attrs(module: Any) -> List[str]:
        return [attr for attr in required if not hasattr(module, attr)]

    def _candidate_module_files() -> List[Path]:
        candidates = [module_name]
        if "." not in module_name:
            candidates.append(f"{module_name}.{module_name}")
        module_files: List[Path] = []
        for candidate in candidates:
            try:
                spec = importlib.util.find_spec(candidate)
            except ModuleNotFoundError:
                continue
            origin = None if spec is None else spec.origin
            if not origin:
                continue
            try:
                module_files.append(Path(str(origin)).resolve())
            except Exception:
                continue
        return module_files

    def _module_file(module: Any) -> Optional[Path]:
        module_file = getattr(module, "__file__", None)
        if module_file:
            try:
                return Path(str(module_file)).resolve()
            except Exception:
                return None
        discovered = _candidate_module_files()
        if not discovered:
            return None
        try:
            return discovered[0]
        except Exception:
            return None

    def _sources_newer_than_module(module: Any) -> bool:
        module_file = _module_file(module)
        if module_file is None or not module_file.exists():
            return True
        try:
            module_mtime = module_file.stat().st_mtime
        except OSError:
            return True
        latest_source_mtime = max(
            _latest_tree_mtime(manifest),
            _latest_tree_mtime(crate_root / "src"),
        )
        return latest_source_mtime > (module_mtime + 1e-6)

    def _import_module_with_fallback() -> Any:
        candidates = [module_name]
        if "." not in module_name:
            candidates.append(f"{module_name}.{module_name}")
        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                return importlib.import_module(candidate)
            except Exception as err:
                last_error = err
        if last_error is not None:
            raise last_error
        raise ModuleNotFoundError(module_name)

    def _drop_cached_candidates() -> None:
        candidates = [module_name]
        if "." not in module_name:
            candidates.append(f"{module_name}.{module_name}")
        for candidate in candidates:
            sys.modules.pop(candidate, None)

    def _build_extension_once() -> None:
        if module_name in _RUST_EXTENSION_PREBUILD_ATTEMPTED:
            return
        _RUST_EXTENSION_PREBUILD_ATTEMPTED.add(module_name)
        if not manifest.exists():
            return
        build_env = os.environ.copy()
        if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
            build_env.setdefault("VIRTUAL_ENV", sys.prefix)
            venv_bin = str(Path(sys.prefix) / "bin")
            build_env["PATH"] = (
                f"{venv_bin}:{build_env['PATH']}"
                if build_env.get("PATH")
                else venv_bin
            )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "maturin>=1.5,<2.0"],
            check=True,
            env=build_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "maturin",
                "develop",
                "--release",
                "--manifest-path",
                str(manifest),
            ],
            check=True,
            env=build_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        importlib.invalidate_caches()
        _drop_cached_candidates()

    discovered_module_files = _candidate_module_files()
    if discovered_module_files:
        probe_module = type("_ProbeModule", (), {"__file__": str(discovered_module_files[0])})()
        if _sources_newer_than_module(probe_module):
            needs_build = True
        else:
            needs_build = False
    else:
        needs_build = True

    if needs_build:
        _build_extension_once()
        importlib.invalidate_caches()
        _drop_cached_candidates()
    module = _import_module_with_fallback()
    missing = _missing_attrs(module)
    if not missing:
        return module
    raise RuntimeError(
        f"Rust extension {module_name} is missing required attributes: {', '.join(missing)}"
    )


def utf8_prefix_byte_offsets(text: str) -> List[int]:
    offsets = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8"))
        offsets.append(total)
    return offsets


def build_token_category_page_metric_row(
    page_row: Dict[str, Any],
    matches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    category_counts: Counter[str] = Counter()
    pattern_family_counts: Counter[str] = Counter()
    for match in matches:
        for category in list(match.get("categories") or []):
            category_counts[str(category)] += 1
        for family in list(match.get("pattern_families") or []):
            pattern_family_counts[str(family)] += 1

    page_char_count = int(page_row.get("page_char_count", 0) or 0)
    match_count = int(page_row.get("match_count", 0) or 0)
    return {
        "source_path": str(page_row.get("source_path", "")),
        "source_stem": str(page_row.get("source_stem", "")),
        "base_stem": str(page_row.get("base_stem", "")),
        "debug_output_path": str(page_row.get("output_path", "")),
        "page_kind": str(page_row.get("page_kind", "")),
        "page_number": int(page_row.get("page_number", 0) or 0),
        "page_index_in_file": int(page_row.get("page_index_in_file", 0) or 0),
        "page_char_count": page_char_count,
        "match_count": match_count,
        "match_density_per_1k_chars": (
            float(match_count) * 1000.0 / float(page_char_count)
            if page_char_count > 0
            else 0.0
        ),
        "match_categories": str(page_row.get("match_categories", "")),
        "match_pattern_families": str(page_row.get("match_pattern_families", "")),
        "category_match_counts": dict(category_counts),
        "pattern_family_match_counts": dict(pattern_family_counts),
    }


def build_token_category_match_index_rows(
    page_text: str,
    matches: List[Dict[str, Any]],
    *,
    page_row: Dict[str, Any],
    context_window_chars: int = 240,
) -> List[Dict[str, Any]]:
    if not matches:
        return []

    byte_offsets = utf8_prefix_byte_offsets(page_text)
    rows: List[Dict[str, Any]] = []
    source_stem = str(page_row.get("source_stem", ""))
    page_kind = str(page_row.get("page_kind", ""))
    page_number = int(page_row.get("page_number", 0) or 0)
    page_index_in_file = int(page_row.get("page_index_in_file", 0) or 0)
    page_char_count = int(page_row.get("page_char_count", 0) or 0)
    output_path = str(page_row.get("output_path", ""))
    for fallback_index, match in enumerate(matches, start=1):
        start = int(match.get("start", 0) or 0)
        end = int(match.get("end", 0) or 0)
        if start < 0 or end < start or end > len(page_text):
            continue
        match_index = int(match.get("match_index_in_page", fallback_index) or fallback_index)
        categories = [str(item) for item in list(match.get("categories") or []) if str(item)]
        pattern_families = [
            str(item) for item in list(match.get("pattern_families") or []) if str(item)
        ]
        excerpt_start = max(0, start - int(context_window_chars))
        excerpt_end = min(len(page_text), end + int(context_window_chars))
        rows.append(
            {
                "match_id": f"{source_stem}:{page_kind}:{page_number}:match:{match_index}",
                "source_path": str(page_row.get("source_path", "")),
                "source_stem": source_stem,
                "base_stem": str(page_row.get("base_stem", "")),
                "debug_output_path": output_path,
                "page_kind": page_kind,
                "page_number": page_number,
                "page_index_in_file": page_index_in_file,
                "page_char_count": page_char_count,
                "match_index_in_page": match_index,
                "start_char": start,
                "end_char": end,
                "start_byte": int(byte_offsets[start]),
                "end_byte": int(byte_offsets[end]),
                "match_length_chars": int(end - start),
                "match_length_bytes": int(byte_offsets[end] - byte_offsets[start]),
                "start_line": int(page_text.count("\n", 0, start) + 1),
                "end_line": int(page_text.count("\n", 0, max(start, end - 1)) + 1),
                "categories": categories,
                "category": ",".join(categories),
                "pattern_families": pattern_families,
                "pattern_family": ",".join(pattern_families),
                "matched_text": page_text[start:end],
                "raw_texts": [str(item) for item in list(match.get("raw_texts") or [])],
                "context_before": page_text[excerpt_start:start],
                "context_after": page_text[end:excerpt_end],
                "context_excerpt": page_text[excerpt_start:excerpt_end],
            }
        )
    return rows
