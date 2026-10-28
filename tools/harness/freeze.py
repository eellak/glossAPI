"""Freeze 30 hash-bound documents using deterministic stratified PDF evidence (node only)."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ''):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'tools.harness'
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from .common import load_manifest, sha256, write_json

RULE = '''v1: verify every cohort hash, count pages, analyze fonts using the production repair selector; inspect text and character geometry on pages 1, floor((N+1)/2), N. Choose, without replacement and ties by filename: maximum Greek fraction among native Greek pages with nonrepairable fonts (clean); maximum affected Greek font codes among repairable documents (repair); minimum characters/page among no-text samples (scan); longest document; strongest two-column line geometry; strongest table-like aligned numeric-row geometry; maximum min(Greek,Latin) characters (mixed); shortest. Fill to 30 by round-robin across ascending page-count quartiles, each sorted by sha256. Layout/scan labels are heuristic evidence on sampled pages, not human ground truth. Freeze three contiguous pages per document by maximizing both layout signals in the same three-page window, ties by earliest start; if signals are on distant pages, table gets priority. Never overwrite the manifest.'''


def page_evidence(page, number):
    textpage = page.get_textpage()
    try:
        text = textpage.get_text_range()
        greek = len(re.findall(r'[\u0370-\u03ff\u1f00-\u1fff]', text))
        latin = len(re.findall(r'[A-Za-z]', text))
        lines = {}
        width, height = page.get_size()
        # Bucket character centers by baseline; count separated text bands.
        for index in range(textpage.count_chars()):
            box = textpage.get_charbox(index)
            left, bottom, right, top = box
            if right <= left or top <= bottom:
                continue
            key = round(bottom / 3) * 3
            lines.setdefault(key, []).append((left, right))
        two, tables = 0, 0
        for spans in lines.values():
            spans.sort()
            bands = []
            for left, right in spans:
                if bands and left - bands[-1][1] < 12:
                    bands[-1][1] = max(right, bands[-1][1])
                else:
                    bands.append([left, right])
            if len(bands) >= 2 and any(a[1] < width * .55 and b[0] > width * .45 and b[0] - a[1] > 20 for a, b in zip(bands, bands[1:])):
                two += 1
            if len(bands) >= 3:
                tables += 1
        numbers = len(re.findall(r'\d+(?:[.,]\d+)?', text))
        return {'page': number, 'characters': len(text.strip()), 'greek': greek, 'latin': latin,
                'two_column_rows': two, 'table_rows': tables, 'numeric_tokens': numbers,
                'table_score': tables if numbers >= 15 else 0,
                'text_excerpt': text[:500]}
    finally:
        textpage.close()


def select(inventory):
    selected = []
    categories = [
        ('Greek born-digital clean', lambda d: d['greek'] / max(1, d['characters']) if d['font_category'] in ('consistent',) and d['characters'] > 500 else -1),
        ('broken ToUnicode repair path', lambda d: d['affected_greek_codes'] if d['repairable_fonts'] else -1),
        ('scanned/image-only', lambda d: -d['characters'] if d['characters'] < 30 else -1e9),
        ('longest (>250 if present)', lambda d: d['pages']),
        ('two-column candidate', lambda d: max(p['two_column_rows'] for p in d['sampled_pages'])),
        ('table candidate', lambda d: max(p['table_score'] for p in d['sampled_pages'])),
        ('mixed Greek/Latin', lambda d: min(d['greek'], d['latin'])),
        ('very short', lambda d: -d['pages'])]
    for category, score in categories:
        remaining = [d for d in inventory if d['filename'] not in {s['filename'] for s in selected}]
        chosen = min(remaining, key=lambda d: (-score(d), d['filename']))
        selected.append(dict(chosen, selection_reason=category, selection_score=score(chosen)))
    ordered = sorted(inventory, key=lambda d: (d['pages'], d['filename']))
    buckets = [sorted(ordered[i * len(ordered)//4:(i+1) * len(ordered)//4], key=lambda d: d['sha256']) for i in range(4)]
    while len(selected) < min(30, len(inventory)):
        for index, bucket in enumerate(buckets):
            while bucket and bucket[0]['filename'] in {d['filename'] for d in selected}:
                bucket.pop(0)
            if bucket and len(selected) < 30:
                selected.append(dict(bucket.pop(0), selection_reason=f'page-count quartile {index+1}; sha256 order'))
    for d in selected:
        samples = d['sampled_pages']
        length = min(3, d['pages'])
        starts = {max(1, min(d['pages'] - length + 1, p['page'] - offset)) for p in samples for offset in range(length)}
        def window(start):
            covered = [p for p in samples if start <= p['page'] < start + length]
            table = max((p['table_score'] for p in covered), default=0)
            columns = max((p['two_column_rows'] for p in covered), default=0)
            return (int(table >= 5) + int(columns >= 8), table, columns, -start)
        start = max(starts, key=window)
        d['page_range'] = [start, start + length - 1]
        d['page_selection_evidence'] = [p for p in samples if start <= p['page'] < start + length]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected', required=True); parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--inventory'); parser.add_argument('--review')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise ValueError('frozen manifest exists; refusing overwrite')
    if args.inventory:
        docs = json.loads(Path(args.inventory).read_text())
        chosen = select(docs)
        if args.review:
            review = json.loads(Path(args.review).read_text())
            lookup = {d['filename']: d for d in docs}
            chosen = []
            for anchor in review['anchors']:
                d = dict(lookup[anchor['filename']], selection_reason=anchor['category'], reviewed_evidence=anchor['evidence'])
                d['page_range'] = anchor.get('page_ranges', [[max(1, (d['pages']+1)//2-2), (d['pages']+1)//2]])[0]
                if 'page_ranges' in anchor: d['page_ranges'] = anchor['page_ranges']
                chosen.append(d)
            ordered = sorted(docs, key=lambda d: (d['pages'], d['filename']))
            buckets = [sorted(ordered[i*len(ordered)//4:(i+1)*len(ordered)//4], key=lambda d:d['sha256']) for i in range(4)]
            templates = {d['filename']:d for d in select(docs)}
            while len(chosen)<30:
                for i,bucket in enumerate(buckets):
                    while bucket and bucket[0]['filename'] in {d['filename'] for d in chosen}: bucket.pop(0)
                    if bucket and len(chosen)<30:
                        d = bucket.pop(0)
                        chosen.append(dict(templates.get(d['filename'],d), selection_reason=f'page-count quartile {i+1}; sha256 order'))
            rule = 'v2 reviewed anchors in selection_review.json first; fill to30 by round-robin page-count quartiles, sha256 order; no replacement. Three pages each, inclusive Docling page_ranges, including visually reviewed table103 and two-column206 in YZU; raster-only page68 in BTS. Other ranges use v1 sampled-geometry rule. v1 candidate was rejected before baselines. '+RULE
        else: rule=RULE
        write_json(out, {'schema':1,'frozen':True,'selection_rule':rule,'cohort_manifest_sha256':sha256(args.expected), 'inventory_sha256':sha256(args.inventory), 'review_sha256':sha256(args.review) if args.review else None,'documents':chosen})
        return
    import pypdfium2 as pdfium
    from glossapi.pdf_repair.selector import analyze_pdf
    docs = []
    for row in sorted(load_manifest(args.expected), key=lambda r: r['filename']):
        path = Path(args.data) / row['filename']
        if sha256(path) != row['sha256']: raise ValueError(f'hash mismatch: {path}')
        pdf = pdfium.PdfDocument(path)
        try:
            n = len(pdf)
            samples = []
            for number in sorted({1, (n+1)//2, n}):
                page = pdf[number-1]
                try: samples.append(page_evidence(page, number))
                finally: page.close()
        finally: pdf.close()
        fonts = analyze_pdf(path)
        d = dict(row, pages=n, sampled_pages=samples, font_category=fonts.category,
                 font_backend=fonts.backend, font_error=fonts.error,
                 repairable_fonts=sum(f.repair_possible for f in fonts.fonts),
                 affected_greek_codes=sum(f.affected_greek_codes for f in fonts.fonts),
                 **{key: sum(p[key] for p in samples) for key in ('characters','greek','latin')})
        docs.append(d)
        write_json(out.with_name('cohort_inventory.json'), docs)
        print(f'{len(docs)} {path.name} {n} {fonts.category}', flush=True)
    chosen = select(docs)
    write_json(out, {'schema': 1, 'frozen': True, 'selection_rule': RULE,
                    'cohort_manifest_sha256': sha256(args.expected), 'documents': chosen})

if __name__ == '__main__': main()
