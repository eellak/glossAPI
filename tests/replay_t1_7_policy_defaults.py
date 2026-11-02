"""Read-only replay of T1.7 defaults; write receipts only outside evidence roots.

Historical provenance helper, not part of the test suite (no ``test_`` prefix, never collected) (T0-G05). ``--evidence``
must point at the experiments root that holds ``glossapi_improvement_20261006/node/archive/r_run`` (page_quality.jsonl),
``glossapi_improvement_20261006/analysis/routed_ocr_pages_all.csv`` and
``compute_reduction_20261006/integrated_run/out/download_results/download_results.parquet``. These inputs are not in
this repository; the replay records the sha256 of each input it reads in its receipt (``--out``).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd

from glossapi.corpus.phase_clean import _enrich_extraction_metrics, _readable_non_greek_mask
from glossapi.page_quality import page_rule_selection
from replay_t1_6_page_rules import outcome


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(evidence):
    archive = evidence / 'glossapi_improvement_20261006/node/archive/r_run'
    integrated = evidence / 'compute_reduction_20261006/integrated_run/out'
    metadata = integrated / 'download_results/download_results.parquet'
    reports_path = archive / 'page_quality.jsonl'
    csv = evidence / 'glossapi_improvement_20261006/analysis/routed_ocr_pages_all.csv'
    reports = [json.loads(line) for line in reports_path.read_text().splitlines()]
    pages = {}
    for report in reports:
        for page in report['pages']:
            key = (Path(report['filename']).stem, page['page'])
            if key in pages:
                raise ValueError(f'Duplicate page: {key}')
            pages[key] = page
    outcomes = pd.read_csv(csv, keep_default_na=False).to_dict('records')
    seen, counts = set(), Counter()
    for row in outcomes:
        key = (row['doc'], int(row['page']))
        if key in seen:
            raise ValueError(f'Duplicate outcome: {key}')
        seen.add(key)
        if key not in pages:
            raise ValueError(f'Missing page join: {key}')
        page = pages[key]
        if row['doc_flagged']:
            continue
        assert page['page_needs_ocr']
        if page_rule_selection(page['decision'], valid_greek_word_rate=page['valid_greek_word_rate'],
                               words=page['words'], pua_char_count=page['pua_char_count'],
                               control_char_count=page['control_char_count']):
            counts[outcome(row)] += 1
    df = pd.read_parquet(metadata)
    enriched = _enrich_extraction_metrics(df, integrated, integrated / 'markdown')
    readable = _readable_non_greek_mask(enriched)
    changed, garbage = [], []
    for i, row in enriched.iterrows():
        name = row['filename']
        report = json.loads((integrated / 'json/metrics' / f'{Path(name).stem}.page_quality.json').read_text())
        weak = sum(p['valid_greek_word_rate'] is not None and p['valid_greek_word_rate'] < .5
                   for p in report['pages'])
        total = len(report['pages'])
        if weak > total / 2:
            garbage.append(name)
        other_reasons = set(str(row['filter']).split(';')) - {'ok', '', 'non_greek_text'}
        if readable.loc[i] and row['needs_ocr'] and not other_reasons:
            changed.append(dict(filename=name, pages=total, weak_pages=weak,
                old_flagged_dispatch_pages=report["pages_needing_ocr"] or total,
                latin_percentage=float(row['latin_percentage']),
                percentage_greek=float(row['percentage_greek']),
                mojibake_badness_score=float(row['mojibake_badness_score']),
                char_count_no_comments=int(row['char_count_no_comments']),
                latin_word_shape_rate=float(row['latin_word_shape_rate'])))
    names = {r['filename'] for r in changed}
    lost_garbage = sorted(names & set(garbage))
    assert not lost_garbage, lost_garbage
    pre_path = archive / 'out_pre_ocr/download_results/download_results.parquet'
    post_path = archive / 'out_ocr_flagged/download_results/download_results.parquet'
    pre = pd.read_parquet(pre_path).set_index('filename')
    post = pd.read_parquet(post_path).set_index('filename')
    remaining = post.loc[pre[pre.needs_ocr].index]
    remaining = remaining[remaining.greek_badness_score > 60].index.tolist()
    receipt = dict(documents=len(df), reports=len(reports), archived_pages=len(pages),
        measured_pages=len(outcomes), default_page_selection=dict(pages=sum(counts.values()), **counts),
        old_flagged_documents=int(df.needs_ocr.sum()), new_flagged_documents=int(df.needs_ocr.sum())-len(changed),
        routing_changes=changed, pages_in_exempted_documents=sum(r['pages'] for r in changed),
        old_flagged_dispatch_pages_exempted=sum(r['old_flagged_dispatch_pages'] for r in changed),
        majority_weak_documents=sorted(garbage), majority_weak_losing_ocr=lost_garbage,
        reported_30_post_ocr_high_scores=sorted(remaining),
        reported_30_in_exemption=sorted(names & set(remaining)),
        input_sha256={str(p): digest(p) for p in (metadata, reports_path, csv, pre_path, post_path)},
        sidecar_sha256={str(p): digest(p) for p in sorted((integrated / 'json/metrics').glob('*.page_quality.json'))},
        raw_markdown_sha256={r['filename']: digest(integrated / 'markdown' / f"{Path(r['filename']).stem}.md")
                             for r in changed})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    evidence, target = args.evidence.resolve(), args.out.resolve()
    if target.is_relative_to(evidence):
        parser.error('output must be outside the evidence tree')
    receipt = replay(evidence)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('documents', 'default_page_selection',
        'old_flagged_documents', 'new_flagged_documents', 'pages_in_exempted_documents',
        'old_flagged_dispatch_pages_exempted', 'majority_weak_losing_ocr')}, indent=2))


if __name__ == '__main__':
    main()
