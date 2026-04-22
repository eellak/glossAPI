"""Analyze Gemini verdicts on line-vs-span drop.

Input: `verdicts_rule_A.jsonl` and `verdicts_rule_B.jsonl` from
`sample_and_review_line_vs_span.py`.

Output: a short markdown report per rule summarizing:
  - N with verdicts
  - distribution of `is_match_noise` / `should_drop_whole_line` /
    `surrounding_prose_is_legitimate_greek`
  - share of cases with `is_match_noise=yes AND should_drop_whole_line=no`
    → that's the "span-drop would be better" signal
  - a handful of short_reason samples for each decision

Decision rule for promotion:
  - if span-drop-preferred share >= 20%: recommend flipping cleaner to
    span drop for that rule set
  - if span-drop-preferred share < 5%: confirm line drop is the right
    choice
  - between 5-20%: flag for manual review
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if "verdict" in d and d["verdict"]:
                out.append(d)
    return out


def _per_rule_summary(rows: List[Dict[str, Any]], rule: str) -> Dict[str, Any]:
    n = len(rows)
    c_noise = Counter(r["verdict"]["is_match_noise"] for r in rows)
    c_line = Counter(r["verdict"]["should_drop_whole_line"] for r in rows)
    c_prose = Counter(r["verdict"]["surrounding_prose_is_legitimate_greek"] for r in rows)
    span_preferred = sum(
        1 for r in rows
        if r["verdict"]["is_match_noise"] == "yes"
        and r["verdict"]["should_drop_whole_line"] == "no"
    )
    span_pct = 100.0 * span_preferred / max(n, 1)
    # Recommendation
    if span_pct >= 20:
        rec = "FLIP TO SPAN DROP"
    elif span_pct < 5:
        rec = "LINE DROP CONFIRMED"
    else:
        rec = "MANUAL REVIEW (5-20% span-preferred)"
    rng = random.Random(20260422)
    return {
        "rule_set": rule,
        "n_verdicts": n,
        "is_match_noise": dict(c_noise),
        "should_drop_whole_line": dict(c_line),
        "surrounding_prose_is_legitimate_greek": dict(c_prose),
        "span_drop_preferred_count": span_preferred,
        "span_drop_preferred_pct": round(span_pct, 2),
        "recommendation": rec,
        "sample_span_preferred": rng.sample(
            [{"source_path": r["source_path"], "matches": r["matches"], "reason": r["verdict"]["short_reason"]}
             for r in rows
             if r["verdict"]["is_match_noise"] == "yes" and r["verdict"]["should_drop_whole_line"] == "no"],
            k=min(5, span_preferred),
        ),
        "sample_line_drop_confirmed": rng.sample(
            [{"source_path": r["source_path"], "matches": r["matches"], "reason": r["verdict"]["short_reason"]}
             for r in rows
             if r["verdict"]["should_drop_whole_line"] == "yes"],
            k=min(5, sum(1 for r in rows if r["verdict"]["should_drop_whole_line"] == "yes")),
        ),
    }


def _format_md(summary_a: Dict[str, Any], summary_b: Dict[str, Any]) -> str:
    def block(s: Dict[str, Any]) -> str:
        out = [f"### Rule {s['rule_set']}"]
        out.append(f"- N verdicts: {s['n_verdicts']}")
        out.append(f"- `is_match_noise`: {s['is_match_noise']}")
        out.append(f"- `should_drop_whole_line`: {s['should_drop_whole_line']}")
        out.append(f"- `surrounding_prose_is_legitimate_greek`: {s['surrounding_prose_is_legitimate_greek']}")
        out.append(f"- **span-drop preferred** (noise=yes AND line_drop=no): "
                   f"{s['span_drop_preferred_count']} = {s['span_drop_preferred_pct']}%")
        out.append(f"- **Recommendation**: {s['recommendation']}")
        out.append("")
        out.append("**Sample span-drop-preferred cases:**")
        for c in s["sample_span_preferred"]:
            out.append(f"- `{c['source_path'][:80]}...`  matches={c['matches']}  — {c['reason']}")
        out.append("")
        out.append("**Sample line-drop-confirmed cases:**")
        for c in s["sample_line_drop_confirmed"]:
            out.append(f"- `{c['source_path'][:80]}...`  matches={c['matches']}  — {c['reason']}")
        return "\n".join(out) + "\n"

    return (
        "# Line-vs-span verdicts — analysis\n\n"
        "Decision contract:\n"
        "- ≥ 20% span-drop preferred → flip cleaner to span drop\n"
        "- < 5% span-drop preferred → line drop confirmed\n"
        "- 5-20% → manual review\n\n"
        + block(summary_a) + "\n" + block(summary_b)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rule_a = _load(args.verdicts_dir / "verdicts_rule_A.jsonl")
    rule_b = _load(args.verdicts_dir / "verdicts_rule_B.jsonl")

    s_a = _per_rule_summary(rule_a, "A")
    s_b = _per_rule_summary(rule_b, "B")

    (args.output_dir / "line_vs_span_summary.json").write_text(
        json.dumps({"A": s_a, "B": s_b}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "line_vs_span_summary.md").write_text(
        _format_md(s_a, s_b), encoding="utf-8",
    )
    print(f"\nRule A: n={s_a['n_verdicts']}  span_preferred={s_a['span_drop_preferred_pct']}%  -> {s_a['recommendation']}")
    print(f"Rule B: n={s_b['n_verdicts']}  span_preferred={s_b['span_drop_preferred_pct']}%  -> {s_b['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
