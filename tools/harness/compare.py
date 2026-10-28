"""Compare exact stage product bytes; optional explicit, hash-bound declarations."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ''):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'tools.harness'
import argparse
import difflib
import json
from pathlib import Path
import re
from .common import STAGES, write_json


def markdown_diff(a, b):
    normalize = lambda s: [' '.join(line.split()) for line in s.replace('\r\n', '\n').splitlines() if line.strip()]
    left, right = normalize(a), normalize(b)
    def changed(x, y):
        removed = added = 0
        for tag, i, j, k, l in difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes():
            if tag != 'equal': removed += j-i; added += l-k
        return {'removed': removed, 'added': added}
    lines = changed(left, right)
    words = changed(' '.join(left).split(), ' '.join(right).split())
    first = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if tag != 'equal':
            first.append({'a_line': i+1, 'b_line': k+1, 'a': left[i:min(j,i+3)], 'b': right[k:min(l,k+3)]})
        if len(first) == 3: break
    return {'normalized_identical': left == right, 'lines_changed': lines, 'words_changed': words, 'first_differences': first}


def compare(a, b, declarations=()):
    a, b = Path(a), Path(b)
    a = a / 'run.json' if a.is_dir() else a
    b = b / 'run.json' if b.is_dir() else b
    left, right = json.loads(a.read_text()), json.loads(b.read_text())
    accepted = {}
    for d in declarations:
        required = ('document', 'stage', 'path', 'a_sha256', 'b_sha256', 'reason')
        if any(k not in d for k in required) or not d['reason'].strip():
            raise ValueError('declarations need document, stage, path, a_sha256, b_sha256, reason')
        key = tuple(d[k] for k in required[:-1])
        if key in accepted: raise ValueError('duplicate declaration')
        accepted[key] = d['reason']
    results, used = [], set()
    docs_a = {d['filename']: d for d in left['documents']}
    docs_b = {d['filename']: d for d in right['documents']}
    for doc in sorted(docs_a.keys() | docs_b.keys()):
        da, db = docs_a.get(doc), docs_b.get(doc)
        for stage in STAGES:
            sa = da['stages'].get(stage) if da else None
            sb = db['stages'].get(stage) if db else None
            differences = []
            def difference(path, ah, bh, detail=None):
                key = (doc, stage, path, ah, bh)
                reason = accepted.get(key)
                if reason: used.add(key)
                differences.append({'path': path, 'a_sha256': ah, 'b_sha256': bh, 'accepted_reason': reason, **(detail or {})})
            if sa is None or sb is None:
                difference('@presence', 'present' if sa else None, 'present' if sb else None)
            else:
                if sa['status'] != sb['status']:
                    difference('@status', sa['status'], sb['status'])
                if sa.get('error') != sb.get('error'):
                    difference('@error', sa.get('error'), sb.get('error'))
                if da['input_sha256'] != db['input_sha256']:
                    difference('@input', da['input_sha256'], db['input_sha256'])
                if da.get('page_ranges') != db.get('page_ranges'):
                    difference('@page_ranges', json.dumps(da.get('page_ranges')), json.dumps(db.get('page_ranges')))
                if da.get('page_range') != db.get('page_range'):
                    difference('@page_range', json.dumps(da.get('page_range')), json.dumps(db.get('page_range')))
                fa, fb = {f['path']: f for f in sa['files']}, {f['path']: f for f in sb['files']}
                for path in sorted(fa.keys() | fb.keys()):
                    x, y = fa.get(path), fb.get(path)
                    if x and y and x['sha256'] == y['sha256'] and x['bytes'] == y['bytes']: continue
                    detail = {}
                    if path.endswith('.md') and x and y:
                        detail['markdown'] = markdown_diff((a.parent / x['stored_path']).read_text(), (b.parent / y['stored_path']).read_text())
                    difference(path, x['sha256'] if x else None, y['sha256'] if y else None, detail)
                if sa.get('removed_files', []) != sb.get('removed_files', []):
                    difference('@removed_files', json.dumps(sa.get('removed_files', [])), json.dumps(sb.get('removed_files', [])))
            results.append({'document': doc, 'stage': stage, 'result': 'different' if differences else 'identical', 'differences': differences})
    # An incomplete or differently configured run cannot pass even when its partial products match.
    issues = []
    if not left.get('complete') or not right.get('complete'): issues.append('incomplete run')
    if left['profile'] != right['profile']: issues.append('different profiles')
    if left['manifest_sha256'] != right['manifest_sha256']: issues.append('different manifest hashes')
    unused = [list(k) for k in accepted.keys() - used]
    undeclared = sum(not d['accepted_reason'] for row in results for d in row['differences'])
    return {'identical': not any(r['differences'] for r in results) and not issues,
            'passes': undeclared == 0 and not issues and not unused,
            'undeclared_differences': undeclared, 'issues': issues, 'unused_declarations': unused, 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('A'); parser.add_argument('B'); parser.add_argument('--declared')
    parser.add_argument('--out'); args = parser.parse_args()
    result = compare(args.A, args.B, json.loads(Path(args.declared).read_text()) if args.declared else [])
    if args.out: write_json(args.out, result)
    for row in result['results']:
        print(f"{row['document']} {row['stage']}: {row['result']}")
        for d in row['differences']: print(json.dumps(d, ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}))
    return 0 if result['passes'] else 1

if __name__ == '__main__': raise SystemExit(main())
