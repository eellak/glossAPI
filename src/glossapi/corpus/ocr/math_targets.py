"""Math-target selection helpers for corpus OCR orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Set

from ..._naming import canonical_stem


def discover_docling_json_stems(output_dir: Path) -> List[str]:
    json_dir = Path(output_dir) / "json"
    if not json_dir.exists():
        return []
    return sorted({canonical_stem(path) for path in json_dir.glob("*.docling.json*")})


def filter_math_only_stems(
    *,
    stems: Sequence[str],
    bad_files: Sequence[str],
    math_done_stems: Set[str],
    reprocess_completed: bool,
    logger,
) -> List[str]:
    kept = list(stems)
    if bad_files:
        before = len(kept)
        bad_set = {canonical_stem(name) for name in bad_files}
        kept = [stem for stem in kept if stem not in bad_set]
        removed = before - len(kept)
        if removed:
            logger.info("Math-only: skipping %d document(s) flagged for OCR", removed)
    if not reprocess_completed and kept and math_done_stems:
        before = len(kept)
        kept = [stem for stem in kept if stem not in math_done_stems]
        removed = before - len(kept)
        if removed:
            logger.info(
                "Math enrichment: skipping %d already enriched document(s) (reprocess_completed=False).",
                removed,
            )
    return kept
