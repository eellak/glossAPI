from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


_SAFE_LABEL_RE = re.compile(r"[^a-z0-9._-]+")


def _slugify_label(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace(" ", "_")
    text = _SAFE_LABEL_RE.sub("_", text)
    text = text.strip("._-")
    return text or "unlabeled"


def _format_metadata_lines(row: Dict[str, object], source_field: str, label_field: str, category_name: str) -> List[str]:
    lines = [
        f"REVIEW_CATEGORY: {category_name}",
        f"REVIEW_LABEL: {row.get(label_field, '')}",
    ]
    for key, value in row.items():
        if key in {source_field, label_field}:
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        lines.append(f"{key.upper()}: {rendered}")
    return lines


def _read_manifest_rows(path: Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_review_copy(
    src: Path,
    dest: Path,
    row: Dict[str, object],
    source_field: str,
    label_field: str,
    category_name: str,
) -> None:
    body = src.read_text(encoding="utf-8", errors="ignore")
    header = "\n".join(_format_metadata_lines(row, source_field, label_field, category_name))
    dest.write_text(f"{header}\n\n=== REVIEW_SOURCE_CONTENT ===\n{body}", encoding="utf-8")


def materialize_manifest_categories(
    manifest_path: Path,
    output_dir: Path,
    *,
    source_field: str = "path",
    label_field: str = "label",
    category_name: str | None = None,
) -> Dict[str, object]:
    rows = _read_manifest_rows(manifest_path)
    category_name = category_name or label_field

    if output_dir.exists():
        for stale in output_dir.rglob("*.txt"):
            stale.unlink()
        for stale in output_dir.rglob("*.json"):
            stale.unlink()
        for stale in output_dir.rglob("*.jsonl"):
            stale.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = output_dir / "by_label"
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_counts: Counter[str] = Counter()
    written_rows: List[Dict[str, object]] = []

    for row in rows:
        if source_field not in row or label_field not in row:
            raise KeyError(f"Manifest row missing required fields: {source_field!r}, {label_field!r}")

        src = Path(str(row[source_field]))
        label = str(row[label_field])
        label_slug = _slugify_label(label)
        dest_dir = labels_dir / label_slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 2
            while True:
                candidate = dest_dir / f"{stem}__dup{counter}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                counter += 1

        _write_review_copy(src, dest, row, source_field, label_field, category_name)
        label_counts[label] += 1
        written_rows.append(
            {
                "label": label,
                "label_slug": label_slug,
                "source_path": str(src),
                "copied_path": str(dest),
            }
        )

    manifest_out = output_dir / "materialized_manifest.jsonl"
    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in written_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "category_name": category_name,
        "source_field": source_field,
        "label_field": label_field,
        "row_count": len(rows),
        "label_counts": dict(label_counts),
        "label_dirs": {
            _slugify_label(label): str(labels_dir / _slugify_label(label))
            for label in sorted(label_counts)
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize categorized review copies from a JSONL manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-field", default="path")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--category-name", default=None)
    args = parser.parse_args()

    materialize_manifest_categories(
        args.manifest,
        args.output_dir,
        source_field=args.source_field,
        label_field=args.label_field,
        category_name=args.category_name,
    )


if __name__ == "__main__":
    main()
