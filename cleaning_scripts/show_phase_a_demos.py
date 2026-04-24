"""Pull isolated demos of Pilot B's two main transforms from real
corpus docs. For each demo: find a small chunk in the source where
the transform fires meaningfully, run Pilot B on JUST that chunk,
print BEFORE / AFTER side by side.

Run on the laptop against the existing 90-doc sample.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import glossapi_rs_cleaner as c

HDR = re.compile(
    r"^##\s+(?:Raw input \(pre-Phase-A\) — BEFORE|Post-Phase-A output — AFTER)\s*$",
    re.MULTILINE,
)


def extract_body(p: Path) -> str:
    t = p.read_text()
    anc = HDR.search(t)
    rest = t[anc.end():]
    div = re.search(r"^---\s*\n", rest, re.MULTILINE)
    return rest[div.end():].lstrip("\n")


def render_demo(label: str, before: str, after: str) -> str:
    out = []
    out.append("")
    out.append("## " + label)
    out.append("")
    out.append(f"**BEFORE** ({len(before)} chars, {before.count(chr(10)) + 1} lines):")
    out.append("")
    out.append("```")
    out.append(before.replace("\t", "↦").replace("\xa0", "·"))
    out.append("```")
    out.append("")
    out.append(f"**AFTER** ({len(after)} chars, {after.count(chr(10)) + 1} lines):")
    out.append("")
    out.append("```")
    out.append(after.replace("\t", "↦").replace("\xa0", "·"))
    out.append("```")
    return "\n".join(out)


def show(label: str, before: str, after: str):
    print(render_demo(label, before, after))


def find_paragraph_reflow_demo(body: str) -> tuple[str, str] | None:
    """Pick a paragraph between two blank lines where:
    - paragraph has ≥ 4 source lines (real soft-wrap)
    - max line length ≤ 60 chars (PDF column-wrap fragments)
    - paragraph is ≤ 800 chars total (small enough to display)
    """
    paragraphs = body.split("\n\n")
    for para in paragraphs:
        lines = para.split("\n")
        if len(lines) < 4:
            continue
        if len(para) > 800:
            continue
        if any(len(l) > 60 for l in lines):
            continue
        # Skip headings / lists / tables.
        if any(re.match(r"^\s*([#>|*+-]|\d+[.)])", l) for l in lines):
            continue
        # Make a 1-paragraph mini-doc.
        mini = para + "\n"
        out = c.format_surgical_py(mini)
        if out != mini and out.count("\n") < mini.count("\n"):
            return mini, out
    return None


def find_table_demo(body: str) -> tuple[str, str] | None:
    """Find a small GFM table whose delimiter row is long (shows the
    minimization clearly)."""
    lines = body.split("\n")
    for i in range(len(lines) - 2):
        # Header row with pipes
        if "|" not in lines[i]:
            continue
        # Delimiter row with at least 20 dashes and pipes
        delim = lines[i + 1]
        if not (re.search(r"-{20,}", delim) and "|" in delim):
            continue
        # Body row with pipes
        if "|" not in lines[i + 2]:
            continue
        # Take 4 lines: blank, header, delim, first body row, blank.
        chunk_start = i
        chunk_end = min(i + 4, len(lines))
        # Walk forward including any extra body rows up to 3 total
        while chunk_end < len(lines) and "|" in lines[chunk_end] and chunk_end - i < 5:
            chunk_end += 1
        chunk = "\n".join(lines[chunk_start:chunk_end]) + "\n"
        # Make sure mini-doc, surrounded by blank lines for clean parse.
        mini = "before paragraph.\n\n" + chunk + "\nafter paragraph.\n"
        out = c.format_surgical_py(mini)
        if out != mini:
            return chunk, out.split("before paragraph.\n\n", 1)[-1].split("\nafter paragraph.\n", 1)[0] + "\n"
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", required=True, type=Path)
    p.add_argument("--output", type=Path,
                   help="Write demos to this MD file. If omitted, "
                        "print to stdout.")
    p.add_argument("--n-paragraph-demos", type=int, default=4)
    p.add_argument("--n-table-demos", type=int, default=3)
    args = p.parse_args()

    befores = sorted(args.sample_dir.glob("*_BEFORE.md"))

    chunks = []
    chunks.append("# Phase A pilot-B reflow demos\n")
    chunks.append(
        "Real PDF-extracted MD docs from `top100_review`. Each "
        "BEFORE chunk is the raw source extracted from a corpus "
        "doc; each AFTER chunk is what `format_surgical` produces "
        "for that chunk. Inputs and outputs render identically "
        "under cmark-gfm (preview-preserving).\n\n"
        "Legend: `↦` = tab character, `·` = NBSP (U+00A0). The "
        "Docling extractor uses `\\t\\n ` (tab + newline + leading "
        "space) on every soft-wrapped continuation line as a "
        "column-preservation marker. Pilot B unwraps these.\n"
    )

    chunks.append("\n## Paragraph reflow demos\n")
    para_count = 0
    for f in befores:
        body = extract_body(f)
        demo = find_paragraph_reflow_demo(body)
        if demo:
            chunks.append(render_demo(
                f"Demo {para_count + 1} — from `{f.name[:55]}`",
                demo[0],
                demo[1],
            ))
            para_count += 1
            if para_count >= args.n_paragraph_demos:
                break

    chunks.append("\n## GFM table delimiter minimization demos\n")
    chunks.append(
        "The header row passes through byte-exact (no escape "
        "injection on cell content); only the `|---|---|` "
        "delimiter row is canonicalized to `| --- | :--- | ---: |` "
        "form per parsed alignments.\n"
    )
    table_count = 0
    for f in befores:
        body = extract_body(f)
        demo = find_table_demo(body)
        if demo:
            chunks.append(render_demo(
                f"Demo {table_count + 1} — from `{f.name[:55]}`",
                demo[0],
                demo[1],
            ))
            table_count += 1
            if table_count >= args.n_table_demos:
                break

    output_text = "\n".join(chunks) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    main()
