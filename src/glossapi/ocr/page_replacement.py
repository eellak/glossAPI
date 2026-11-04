"""Hash-bound native page boundaries and strip provenance for partial OCR."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..markdown_layout.render import apply_edits, strip_record


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def native_pages(root: Path, stem: str, text: str, count: int):
    """Return validated physical pages, local strip edits, and a fallback reason.

    Missing rows represent blank pages. Never infer boundaries from paragraph
    breaks: the extractor omits blank pages and paragraphs can contain them too.
    """
    path = root / "sidecars/extract" / f"{stem}.json"
    if not path.is_file():
        return None, None, "page_offsets_missing"
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict):
            raise ValueError("page sidecar must be an object")
        if sidecar.get("page_count", count) != count:
            raise ValueError("page count changed")
        layout = sidecar.get("markdown_layout")
        plain = sidecar.get("plain_pages")
        record = layout if isinstance(layout, dict) else plain
        if not isinstance(record, dict) or not isinstance(record.get("pages"), list):
            return None, None, "page_offsets_missing"
        binding = record.get("strip") if layout is not None else record
        if not isinstance(binding, dict):
            return None, None, "page_offsets_unbound"
        if binding.get("markdown_chars") != len(text) or binding.get("markdown_sha256") != text_hash(text):
            return None, None, "page_offsets_hash_mismatch"
        pages, edits = [""] * count, [[] for _ in range(count)]
        cursor = previous = 0
        spans = []
        for row in record["pages"]:
            page, start, end = row[:3]
            if (any(type(v) is not int for v in (page, start, end))
                    or not previous < page <= count or not cursor <= start <= end <= len(text)
                    or text[cursor:start] != ("\n\n" if spans else "")):
                raise ValueError("invalid page span")
            pages[page - 1] = text[start:end]
            spans.append((page, start, end))
            previous, cursor = page, end
        if cursor != len(text):
            raise ValueError("unmapped text")
        if layout is not None:
            from ..markdown_layout import markup_free
            if markup_free(text, layout) is None:
                raise ValueError("invalid strip")
            for start, end, replacement in binding["edits"]:
                owners = [(p, s) for p, s, e in spans if s <= start <= end <= e]
                if len(owners) != 1:
                    raise ValueError("strip crosses a page boundary")
                page, offset = owners[0]
                edits[page - 1].append([start - offset, end - offset, replacement])
        return pages, edits, None
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return None, None, "page_offsets_unusable"


def replacement_layout(pages: list[str], native_edits: list, replaced: set[int]):
    """Compose physical pages without markers; retain only surviving native edits."""
    pieces, plain, edits, spans = [], [], [], []
    length = 0
    for page, text in enumerate(pages, 1):
        if not text:
            continue
        if pieces:
            length += 2
        local = [] if page in replaced else native_edits[page - 1]
        edits.extend([[length + s, length + e, value] for s, e, value in local])
        spans.append([page, length, length + len(text), "ocr" if page in replaced else "native"])
        pieces.append(text)
        plain.append(apply_edits(text, local))
        length += len(text)
    # Match the canonical OCR serializer, while binding its entire output.
    raw = "\n\n".join(pieces)
    markdown = raw.rstrip("\n") + "\n"
    plain_text = "\n\n".join(plain)
    plain_text = plain_text[:len(plain_text) - (len(raw) - len(raw.rstrip("\n")))] if raw.endswith("\n") else plain_text
    plain_text += "\n"
    record = strip_record(markdown, edits, plain_text)
    if record is None:
        raise ValueError("Cannot bind replacement strip")
    return markdown, {"source": "ocr_page_replacement", "pages": spans, "strip": record}
