"""Read-only sample dispatch accounting from existing extraction/clean artifacts.

Failure-stratum membership takes priority over accepted_control; otherwise the
first manifest occurrence wins. Each filename contributes once, including in
the total. Source denominators are opened with PDFium, never taken from a
manifest or nullable parquet column. This measures scheduled pages, not OCR
execution, runtime or fidelity. Legacy parquet is enriched in memory only.
"""
from __future__ import annotations

from glossapi.pdfium_guard import pdfium_guard

from contextlib import closing

import argparse
import json
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium

from glossapi.corpus.phase_clean import _attach_page_quality_routes, _enrich_extraction_metrics
from glossapi.page_quality import parse_page_ranges


def build_report(*, manifest: Path, parquet: Path, pdf_dir: Path, output_dir: Path) -> dict:
    """Return exclusive per-stratum and unique total counts; do not mutate inputs."""
    membership = pd.read_csv(manifest)
    if not {'filename', 'stratum'} <= set(membership.columns):
        raise ValueError('Manifest must contain filename and stratum')
    duplicate_names = sorted(membership.loc[membership.filename.duplicated(False), 'filename'].unique())
    membership['_control'] = membership.stratum.eq('accepted_control')
    membership = membership.sort_values('_control', kind='stable').drop_duplicates('filename')
    membership = membership.set_index('filename')
    frame = pd.read_parquet(parquet)
    if frame.filename.duplicated().any():
        raise ValueError('Metadata must contain unique filenames')
    missing = sorted(set(frame.filename) - set(membership.index))
    if missing:
        raise ValueError(f'Metadata filenames absent from manifest: {missing}')
    frame = _enrich_extraction_metrics(frame, output_dir, output_dir / 'markdown')
    frame = _attach_page_quality_routes(frame, output_dir / 'json/metrics')
    records = []
    for _, row in frame.sort_values('filename').iterrows():
        name = row.filename
        with pdfium_guard(), closing(pdfium.PdfDocument(str(pdf_dir / name))) as document:
            pages = len(document)
        ranges = parse_page_ranges(row.ocr_pages)
        if any(end > pages for _, end in ranges):
            raise ValueError(f'Route exceeds actual PDF page count: {name}')
        selected = sum(end - start + 1 for start, end in ranges)
        flagged = bool(row.needs_ocr) if pd.notna(row.get('needs_ocr')) else False
        failed = 'extract_failed' in str(row.get('filter', '')).split(';')
        dispatch_flagged = (selected or pages) if flagged else 0
        dispatch_all = dispatch_flagged if flagged else selected
        repair = row.get('font_repair_status', '')
        records.append(dict(filename=name, stratum=membership.loc[name, 'stratum'],
            source_pages=pages, repaired=int(repair == 'repaired'), no_text=int(repair == 'no_text_layer'),
            unsupported=int(repair == 'unsupported'), errors=int(repair == 'error'),
            flagged=int(flagged), extract_failed=int(failed), extract_failed_pages=pages if failed else 0,
            extract_status=row.extract_status, dispatch_flagged=dispatch_flagged, dispatch_all=dispatch_all,
            whole_document_pages=pages if flagged else 0,
            fallback_docs=int(flagged and not selected), fallback_pages=pages if flagged and not selected else 0,
            ocr_pages_unflagged_count=selected if not flagged else 0,
            pua_char_count=None if pd.isna(row.pua_char_count) else int(row.pua_char_count),
            control_char_count=None if pd.isna(row.control_char_count) else int(row.control_char_count)))
    counts = ['source_pages', 'repaired', 'no_text', 'unsupported', 'errors', 'flagged',
              'extract_failed', 'extract_failed_pages', 'dispatch_flagged', 'dispatch_all',
              'whole_document_pages', 'fallback_docs', 'fallback_pages', 'ocr_pages_unflagged_count']

    def aggregate(rows):
        total = dict(docs=len(rows), **{key: sum(r[key] for r in rows) for key in counts})
        for mode in ('flagged', 'all'):
            total[f'dispatch_{mode}_share'] = total[f'dispatch_{mode}'] / total['source_pages'] if total['source_pages'] else 0
        return total

    strata = [dict(stratum=label, **aggregate([r for r in records if r['stratum'] == label]))
              for label in sorted({r['stratum'] for r in records})]
    return dict(membership_policy='first non-control manifest membership, else first control membership',
                duplicate_memberships=duplicate_names,
                downloaded_duplicate_memberships=sorted(set(duplicate_names) & set(frame.filename)),
                strata=strata, total=aggregate(records), documents=records)


def markdown_table(report: dict) -> str:
    lines = ['| Stratum | Docs | Repaired | No text | Unsupported | Errors | Flagged | Extract failed (pages) | PDF pages | Dispatch flagged | Dispatch all | Unflagged selected |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in report['strata'] + [dict(stratum='TOTAL (unique)', **report['total'])]:
        lines.append(f"| {row['stratum']} | {row['docs']} | {row['repaired']} | {row['no_text']} | {row['unsupported']} | {row['errors']} | {row['flagged']} | {row['extract_failed']} ({row['extract_failed_pages']}) | {row['source_pages']:,} | {row['dispatch_flagged']:,} ({100 * row['dispatch_flagged_share']:.2f}%) | {row['dispatch_all']:,} ({100 * row['dispatch_all_share']:.2f}%) | {row['ocr_pages_unflagged_count']:,} |")
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--pdf-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='Existing pipeline artifact root (read-only)')
    parser.add_argument('--json-out', type=Path, help='Optional report destination; never modifies pipeline artifacts')
    args = parser.parse_args()
    report = build_report(manifest=args.manifest, parquet=args.parquet, pdf_dir=args.pdf_dir, output_dir=args.output_dir)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(markdown_table(report))
    print('\nMembership policy:', report['membership_policy'])
    print('Downloaded overlapping filenames:', ', '.join(report['downloaded_duplicate_memberships']) or 'none')
    failed = [r['filename'] for r in report['documents'] if r['extract_failed']]
    print('Extract-failed filenames:', ', '.join(failed) or 'none')
    print('Scheduling only; no OCR execution or fidelity measurement.')


if __name__ == '__main__':
    main()
