"""Freeze metadata and hashes only; never render or recognise candidate PDFs."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=None)
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def freeze():
    repo = Path(__file__).resolve().parents[1]
    base = Path.home() / "Projects/glossapi-ocr-local/experiments/compute_reduction_20261006"
    source = base / "integrated_run/page_quality.jsonl"
    build = Path.home() / "Projects/greek-page-ocr/out/o5/build_info.json"
    assignment = json.loads(build.read_text())["document_assignment"]
    excluded = set(assignment["train"]) | set(assignment["val"])
    reserved = set(assignment["reserved"])
    rows = sorted((json.loads(line) for line in source.read_text().splitlines()),
                  key=lambda row: row["filename"])
    rows = [row for row in rows if Path(row["filename"]).stem not in excluded]
    reserved_rows = [row for row in rows if Path(row["filename"]).stem in reserved]
    pages = []

    def add(row, page, role):
        original = base / "sample_pdfs" / row["filename"]
        gt = base / "integrated_run/out/repaired_pdfs" / row["filename"]
        assert original.is_file()
        assert gt.is_file(), f"missing repaired GT: {gt}"
        pages.append(dict(
            document_id=original.stem, pdf=str(original), pdf_sha256=digest(original),
            page=page["page"], role=role, decision=page["decision"],
            o5_partition="reserved" if original.stem in reserved else "outside_o5",
            page_needs_ocr=page["page_needs_ocr"], routing_metrics=page,
            gt_source=dict(pdf=str(gt), sha256=digest(gt),
                           available_at_freeze=True, page=page["page"],
                           method="Docling textline cells by top-left rect; NFC and whitespace collapse"),
        ))

    candidates = [(row, [p for p in sorted(row["pages"], key=lambda p:p["page"])
                         if p["page_needs_ocr"] and p["decision"] in
                         {"garbage_greek_text", "scanned_page"}]) for row in reserved_rows]
    for rank in range(2):
        for row, ps in candidates:
            if len(ps) > rank and sum(p["role"] == "routed" for p in pages) < 24:
                add(row, ps[rank], "routed")
    routed_docs = {p["document_id"] for p in pages}
    for row in sorted(reserved_rows, key=lambda row: (Path(row["filename"]).stem not in routed_docs,
                                           row["filename"])):
        clean = [p for p in row["pages"] if not p["page_needs_ocr"] and
                 p["decision"] == "text_layer" and p["chars"] >= 400 and p["greek_chars"] >= 200]
        if clean and sum(p["role"] == "clean" for p in pages) < 6:
            add(row, clean[0], "clean")
    def english_candidates(pool):
        return [(row, p) for row in pool for p in row["pages"]
                if not p["page_needs_ocr"] and p["decision"] == "text_layer"
                and p["chars"] >= 1000 and p["latin_chars"] >= 800
                and p["greek_chars"] == 0 and p["latin_chars"] / p["chars"] >= .65]
    english = english_candidates(reserved_rows)
    fallback = not english
    if fallback:
        english = english_candidates(rows)
    english_docs = set()
    for row, p in english:
        if row["filename"] not in english_docs and len(english_docs) < 2:
            add(row, p, "english")
            english_docs.add(row["filename"])
    assert Counter(p["role"] for p in pages) == {"routed": 24, "clean": 6, "english": 2}
    assert max(Counter(p["document_id"] for p in pages if p["role"] == "routed").values()) <= 2
    assert not ({p["document_id"] for p in pages} & excluded)
    assert all(p["document_id"] in reserved for p in pages if p["role"] != "english")
    codec = repo / "src/glossapi/ocr/plugin/recognizer/v1.json"
    cohort = dict(
        schema_version=2,
        cohort_rule="exclude o5 train+val; reserved eligible",
        selection="Before inference: reserved docs in lexical order, first eligible garbage/scan "
                  "page per doc then second until 24; first six readable Greek text_layer pages "
                  "prioritising routed docs; first English page in each of two lexical reserved docs "
                  "(chars>=1000, Latin>=800 and >=65%, Greek=0).",
        english_selection=dict(outside_reserved_fallback=fallback,
                               verification="Original textline sample recorded during the frozen run"),
        routing_source=dict(path=str(source), sha256=digest(source)),
        exclusion_source=dict(path=str(build), sha256=digest(build), document_assignment=assignment,
                              excluded_partitions=["train", "val"],
                              excluded_document_count=len(excluded),
                              eligible_reserved_document_count=len(reserved)),
        recognizer=dict(
            checkpoint=str(Path.home() / "Projects/greek-page-ocr/out/o5_node_20261007/runs/real_s0/best.pt"),
            sha256="2e97f3aea523895513f3bad7aaecbc32d159aef9acaf14cfc437baad2b0d46cc",
            codec=str(codec.relative_to(repo)), codec_sha256=digest(codec), arch="compact",
            head="nfd", n_classes=248, device="cpu", crop_scale=3.0),
        pages=pages,
    )
    target = repo / "docs/experiments/glossapi_improvement_20261006/C1_COHORT.json"
    target.write_text(json.dumps(cohort, indent=2, ensure_ascii=False) + "\n")
    print(f"Frozen {len(pages)} pages; {sum(p['gt_source']['available_at_freeze'] for p in pages)} repaired GT files")


if __name__ == "__main__":
    freeze()
