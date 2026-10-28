"""Read-only T1.6 replay; full native PDF text is required for exact folding.

Run with the experiment Python and PYTHONPATH=src; see T1_6_NOTES.md.
Outcomes use the archived evaluator's classes, independent of routing mode.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import pypdfium2 as pdfium

from glossapi.page_quality import (
    _load_folded_words, _page_text_stats, page_ocr_decision, page_rule_selection,
)


def rate(value):
    number = float(value) if value else None
    return number if number is not None and math.isfinite(number) else None


def outcome(row):
    # Same ordered classification as analysis/page_rule_eval.py.
    if int(row["o_chars"]) < 20:
        return "waste"
    if int(row["n_chars"]) < 20:
        return "gain"
    n, o = rate(row["n_valid"]) or 0, rate(row["o_valid"]) or 0
    if int(row["n_greek"]) >= 10 and n < .5 and o >= .8:
        return "gain"
    if int(row["n_greek"]) >= 10 and n >= .8 and o >= .8:
        return "waste"
    if int(row["n_greek"]) < 10 and int(row["n_latin"]) >= 10:
        return "waste"
    if rate(row["o_valid"]) is not None and o < .5:
        return "waste"
    return "mixed"


def replay(archive, outcomes, originals, wordlist):
    features = {}
    reports = [json.loads(line) for line in (archive / "page_quality.jsonl").open()]
    for report in reports:
        for page in report["pages"]:
            key = (Path(report["filename"]).stem, page["page"])
            if key in features:
                raise ValueError(f"Duplicate archived page {key}")
            features[key] = page
    rows = list(csv.DictReader(outcomes.open(encoding="utf-8", newline="")))
    accepted = [r for r in rows if r["doc_flagged"] == "False"]
    if len({(r["doc"], int(r["page"])) for r in rows}) != len(rows):
        raise ValueError("Duplicate outcome pages")
    groups = defaultdict(list)
    for row in accepted:
        groups[row["doc"]].append(row)
    dictionary = _load_folded_words(str(wordlist.resolve()))
    totals = defaultdict(Counter)
    decisions = defaultdict(Counter)
    checks = Counter()
    details = []
    for stem, group in sorted(groups.items()):
        repaired = archive / "out_pre_ocr/repaired_pdfs" / f"{stem}.pdf"
        source = repaired if repaired.exists() else originals / f"{stem}.pdf"
        with pdfium.PdfDocument(str(source)) as pdf:
            for row in group:
                key = (stem, int(row["page"]))
                feature = features[key]
                if not feature["page_needs_ocr"]:
                    raise ValueError(f"Outcome outside archived v2 selection: {key}")
                page = pdf[key[1] - 1]
                try:
                    textpage = page.get_textpage()
                    try:
                        text = textpage.get_text_range()
                    finally:
                        textpage.close()
                finally:
                    page.close()
                stats = _page_text_stats(text, dictionary, True)
                for name in ("chars", "greek_chars", "latin_chars", "words"):
                    if stats[name] != feature[name]:
                        raise ValueError(f"Native source/sidecar mismatch: {key}, {name}")
                checks["native_source_pages_matched"] += 1
                folded_rate = stats["dictionary_valid_rate"]
                values = {name: feature[name] for name in (
                    "chars", "greek_chars", "latin_chars", "other_chars", "words",
                    "image_area_ratio", "pua_char_count", "control_char_count")}
                modes = {}
                for profile in ("v2", "v3"):
                    modes[f"{profile}_archived"] = page_rule_selection(
                        feature["decision"], rule_profile=profile,
                        valid_greek_word_rate=feature["valid_greek_word_rate"],
                        **{k: values[k] for k in ("words", "pua_char_count", "control_char_count")})
                    fresh, reason = page_ocr_decision(**values,
                        valid_greek_word_rate=feature["valid_greek_word_rate"], rule_profile=profile)
                    if fresh != modes[f"{profile}_archived"]:
                        raise ValueError(f"Unfolded profile/live replay disagreement: {key}")
                    modes[f"{profile}_folded_exact"] = page_ocr_decision(
                        **values, valid_greek_word_rate=folded_rate, rule_profile=profile,
                        candidate_decision=feature["decision"] if profile == "v3" else None)[0]
                # The published T1 count excluded the PUA/control override and
                # imputed unknown validity as zero (page_rule_eval.py fillna(0)).
                modes["published_T1_without_contamination_override"] = page_rule_selection(
                    feature["decision"], rule_profile="v3",
                    valid_greek_word_rate=feature["valid_greek_word_rate"] or 0, words=values["words"])
                modes["v3_folded_archived_labels"] = page_rule_selection(
                    feature["decision"], rule_profile="v3", valid_greek_word_rate=folded_rate,
                    **{k: values[k] for k in ("words", "pua_char_count", "control_char_count")})
                # The published CSV metric counts only >=3-character tokens;
                # disclose this proxy separately from the production metric.
                modes["v3_csv_folded_proxy"] = page_rule_selection(
                    feature["decision"], rule_profile="v3", valid_greek_word_rate=rate(row["n_valid"]),
                    **{k: values[k] for k in ("words", "pua_char_count", "control_char_count")})
                result = outcome(row)
                decisions[feature["decision"]][result] += 1
                if (feature["valid_greek_word_rate"] is not None
                        and feature["valid_greek_word_rate"] < .5
                        and folded_rate is not None and folded_rate >= .5):
                    checks["validity_crossed_0.5"] += 1
                for mode, selected in modes.items():
                    if selected:
                        totals[mode]["pages"] += 1
                        totals[mode][result] += 1
                details.append(dict(doc=stem, page=key[1], source=str(source),
                    archived_decision=feature["decision"], archived_validity=feature["valid_greek_word_rate"],
                    folded_validity=folded_rate, csv_native_validity=rate(row["n_valid"]),
                    outcome=result, selections=modes))
    inputs = [archive / "page_quality.jsonl", outcomes, wordlist]
    summary = dict(reports=len(reports), archived_pages=len(features), outcome_pages=len(rows),
        flagged_outcome_pages=len(rows) - len(accepted), accepted_documents=len(groups),
        accepted_outcome_pages=len(accepted), checks=dict(checks),
        inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        counts={mode: dict(counts) for mode, counts in totals.items()},
        archived_decisions={label: dict(counts) for label, counts in decisions.items()})
    return summary, details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--wordlist", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Reject evidence/output aliases before writing; only --out is mutated.
    sources = [args.outcomes.resolve(), args.wordlist.resolve()]
    roots = [args.archive.resolve(), args.originals.resolve(),
             args.outcomes.resolve().parent, args.wordlist.resolve().parent]
    target = args.out.resolve()
    if target in sources or any(target.is_relative_to(root) for root in roots):
        parser.error("--out must be outside the read-only evidence")
    summary, details = replay(args.archive, args.outcomes, args.originals, args.wordlist)
    target.write_text(json.dumps(dict(summary=summary, pages=details), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
