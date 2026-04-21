"""Stratified sampler for the three-counter Gemini review wave.

Input:
  A merged `page_metrics.jsonl` (produced by run_matcher_parallel.py). Each row
  is a page-level record including `page_category_counts` and `page_char_count`.

For each of the three counters we care about:
  - page_font_marker_count        (matcher category: font_name_literal)
  - page_glyph_marker_count       (matcher category: glyph_font_like)
  - page_script_residue_count     (matcher category: script_residue_restricted)

We slice `[min, max]` across the population into 10 zones, allocate ~50 cases
per counter proportionally to zone population (floor 5/zone), draw with a fixed
seed, and write per-case Markdown files.

Filenames:
  <counter>/<counter>_<zero_padded_value>__<source>_p<page>.md

This keeps `ls` sort == metric sort per
`feedback_metric_prefix_in_sample_filenames`.

Each file body contains:
  - `[CONTEXT]` block with the full page text, no inline tags (counts are saturated).
  - `[QUESTIONS]` block with the four locked questions from Task type 2.
  - A small `<!-- case_meta=... -->` HTML comment at the top carrying the
    match_id / counter values / zone info for client-side join after Gemini
    returns verdicts. The `<!-- -->` form means the model will typically
    ignore it; we join by enumeration order for safety.

Run on the laptop after rsyncing `page_metrics.jsonl` back from the worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


COUNTERS = {
    "font_marker": ("font_name_literal", "page_font_marker_count"),
    "glyph_marker": ("glyph_font_like", "page_glyph_marker_count"),
    "script_residue": ("script_residue_restricted", "page_script_residue_count"),
}

PROMPT_QUESTIONS = """\
[QUESTIONS]
1. keep_or_drop       (keep / drop / uncertain)
2. noise_character    (clean / mojibake / glyph_corruption / script_salad /
                       garbled_text_other / mixed / unclear)
3. dominant_signal    (font_names / glyph_tags / script_residue /
                       other_unknown / none / multiple)
4. short_reason       (≤ 40 words, free text)
"""


@dataclass
class PageRow:
    source_path: str
    source_stem: str
    page_kind: str
    page_number: int
    page_char_count: int
    debug_output_path: str
    counters: Dict[str, int]


# Strip `<!-- ... -->` HTML header comments from the debug-page markdown so
# only the page content remains. The debug .md file starts with two such
# comments (source_path + match metadata) before the actual page text.
HEADER_COMMENT_RE = re.compile(r"^<!--[\s\S]*?-->\s*", re.MULTILINE)
# Strip the inline `<match category="…" pattern_family="…">…</match>` wrappers
# from the debug-page passages. The wrappers are matcher-output structure,
# not real page content, and would distract the reviewer on a saturated page.
MATCH_TAG_RE = re.compile(
    r'<match\s+category="[^"]*"\s+pattern_family="[^"]*">([^<]*)</match>'
)


def _read_page_text(debug_md: Path) -> str:
    if not debug_md.exists():
        return ""
    raw = debug_md.read_text(encoding="utf-8", errors="replace")
    # Remove leading `<!-- -->` header block(s)
    body = raw
    while True:
        m = re.match(r"^<!--[\s\S]*?-->\s*", body)
        if not m:
            break
        body = body[m.end():]
    # Unwrap inline <match> tags (saturated-page rule: counts in header
    # carry the numeric signal; passage is for holistic reviewer judgment)
    body = MATCH_TAG_RE.sub(lambda m: m.group(1), body)
    return body.strip()


def _iter_page_metrics(path: Path) -> List[PageRow]:
    out: List[PageRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            category_counts_raw = d.get("category_match_counts") or {}
            if isinstance(category_counts_raw, str):
                try:
                    category_counts_raw = json.loads(category_counts_raw)
                except json.JSONDecodeError:
                    category_counts_raw = {}
            counters = {
                key: int(category_counts_raw.get(category, 0))
                for key, (category, _field_name) in COUNTERS.items()
            }
            out.append(
                PageRow(
                    source_path=str(d.get("source_path", "")),
                    source_stem=str(d.get("source_stem", "")),
                    page_kind=str(d.get("page_kind", "")),
                    page_number=int(d.get("page_number", 0) or 0),
                    page_char_count=int(d.get("page_char_count", 0) or 0),
                    debug_output_path=str(d.get("debug_output_path", "")),
                    counters=counters,
                )
            )
    return out


def _zone_edges(values: List[int], n_zones: int, log_scale: bool) -> List[float]:
    """Return `n_zones+1` edge values spanning min..max (inclusive)."""
    vmin = max(min(values), 1 if log_scale else 0)
    vmax = max(values)
    if vmax <= vmin:
        return [float(vmin), float(vmax) + 1.0]
    if log_scale:
        import math

        lo = math.log10(max(vmin, 1))
        hi = math.log10(max(vmax, vmin + 1))
        step = (hi - lo) / n_zones
        return [10 ** (lo + i * step) for i in range(n_zones + 1)]
    step = (vmax - vmin) / n_zones
    return [vmin + i * step for i in range(n_zones + 1)]


def _zone_for_value(v: int, edges: List[float]) -> int:
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if (i < len(edges) - 2 and lo <= v < hi) or (i == len(edges) - 2 and lo <= v <= hi):
            return i
    return len(edges) - 2


def _allocate_per_zone(
    zone_populations: List[int], total_budget: int, floor: int
) -> List[int]:
    nonempty = [i for i, n in enumerate(zone_populations) if n > 0]
    if not nonempty:
        return [0] * len(zone_populations)
    reserved = floor * len(nonempty)
    remaining = max(total_budget - reserved, 0)
    pop_sum = sum(zone_populations[i] for i in nonempty)
    out = [0] * len(zone_populations)
    for i in nonempty:
        share = int(round(remaining * zone_populations[i] / pop_sum)) if pop_sum else 0
        out[i] = min(floor + share, zone_populations[i])
    # Fix drift so the total matches as closely as possible without exceeding budget.
    current = sum(out)
    while current > total_budget:
        for i in reversed(nonempty):
            if out[i] > floor:
                out[i] -= 1
                current -= 1
                if current <= total_budget:
                    break
        else:
            break
    return out


def _safe_label(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:80] or "page"


def _case_filename(counter_name: str, value: int, row: PageRow, idx_width: int = 6) -> str:
    padded = f"{value:0{idx_width}d}"
    src = _safe_label(row.source_stem or "src")
    return f"{counter_name}_{padded}__{src}_p{row.page_number:05d}.md"


def _build_case_body(row: PageRow, counter_name: str, counter_value: int) -> str:
    meta_comment = json.dumps(
        {
            "source_path": row.source_path,
            "source_stem": row.source_stem,
            "page_number": row.page_number,
            "page_char_count": row.page_char_count,
            "primary_counter": counter_name,
            "counters": row.counters,
        },
        ensure_ascii=False,
    )
    page_text = _read_page_text(Path(row.debug_output_path))
    return (
        f"<!-- case_meta={meta_comment} -->\n\n"
        f"[CONTEXT]\n{page_text}\n\n"
        f"{PROMPT_QUESTIONS}"
    )


def sample_for_counter(
    counter_name: str,
    pages: List[PageRow],
    n_zones: int,
    budget: int,
    floor: int,
    log_scale: bool,
    seed: int,
) -> Tuple[List[Tuple[PageRow, int]], Dict[str, Any]]:
    values = [p.counters[counter_name] for p in pages if p.counters[counter_name] > 0]
    if not values:
        return [], {"zone_edges": [], "per_zone_populations": [], "per_zone_allocated": [], "total": 0}
    edges = _zone_edges(values, n_zones, log_scale)
    zone_to_pages: Dict[int, List[PageRow]] = defaultdict(list)
    for p in pages:
        v = p.counters[counter_name]
        if v <= 0:
            continue
        z = _zone_for_value(v, edges)
        zone_to_pages[z].append(p)
    populations = [len(zone_to_pages.get(i, [])) for i in range(n_zones)]
    allocation = _allocate_per_zone(populations, budget, floor)
    rng = random.Random(seed)
    samples: List[Tuple[PageRow, int]] = []
    for z, quota in enumerate(allocation):
        candidates = list(zone_to_pages.get(z, []))
        rng.shuffle(candidates)
        for row in candidates[:quota]:
            samples.append((row, z))
    meta = {
        "zone_edges": edges,
        "per_zone_populations": populations,
        "per_zone_allocated": allocation,
        "total": sum(allocation),
        "seed": seed,
        "log_scale": log_scale,
    }
    return samples, meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-zones", type=int, default=10)
    parser.add_argument("--budget-per-counter", type=int, default=50)
    parser.add_argument("--floor-per-zone", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--log-scale", action="store_true",
                        help="Slice zones on log scale (use when metric spans orders of magnitude)")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = _iter_page_metrics(args.page_metrics)
    print(f"loaded {len(pages)} page rows from {args.page_metrics}")

    overall_meta: Dict[str, Any] = {
        "n_zones": args.n_zones,
        "budget_per_counter": args.budget_per_counter,
        "floor_per_zone": args.floor_per_zone,
        "seed": args.seed,
        "log_scale": args.log_scale,
        "page_count": len(pages),
        "counters": {},
    }

    for counter_name in COUNTERS:
        subdir = args.output_dir / counter_name
        subdir.mkdir(parents=True, exist_ok=True)
        samples, meta = sample_for_counter(
            counter_name, pages,
            args.n_zones, args.budget_per_counter, args.floor_per_zone,
            args.log_scale, args.seed,
        )
        overall_meta["counters"][counter_name] = meta
        for row, zone in samples:
            value = row.counters[counter_name]
            path = subdir / _case_filename(counter_name, value, row)
            path.write_text(_build_case_body(row, counter_name, value), encoding="utf-8")
        print(f"  {counter_name}: wrote {len(samples)} cases to {subdir}")

    (args.output_dir / "aggregate.json").write_text(
        json.dumps(overall_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
