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
import math
from pathlib import Path
import re
from .common import STAGES, sha256, write_json


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


def json_changes(a, b, path=''):
    """Exact decoded leaf changes, using JSON pointers (array order is significant)."""
    if type(a) is not type(b):
        return {path}
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return {path}  # schema changes cannot be timing declarations
        return set().union(*(json_changes(a[k], b[k], path+'/'+k.replace('~', '~0').replace('/', '~1')) for k in a))
    if isinstance(a, list):
        if len(a) != len(b):
            return {path}
        return set().union(*(json_changes(x, y, path+'/'+str(i)) for i, (x, y) in enumerate(zip(a, b))))
    return {path} if a != b else set()


def timing_field(path):
    return bool(re.fullmatch(
        r'/timings/[^/]+/(?:avg_sec|p50_sec|p90_sec|total_sec|times_sec/\d+)'
        r'|/pages/\d+/(?:parse_sec|table_sec|layout_sec)'
        r'|/totals/doc_enrich_total_sec', path))


def timing_changes(a, b, fields):
    """Classify only same-type, finite numeric changes at existing timing leaves."""
    def resolve(value, path):
        for token in path.split('/')[1:]:
            key = token.replace('~1', '/').replace('~0', '~')
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value

    def finite_number(value):
        # bool is an int subclass; arbitrarily large JSON integers are finite.
        return type(value) is int or (type(value) is float and math.isfinite(value))

    eligible = []
    for field in fields:
        if not timing_field(field):
            continue
        try:
            left, right = resolve(a, field), resolve(b, field)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if type(left) is type(right) and finite_number(left) and finite_number(right):
            eligible.append(field)
    return sorted(eligible)


def validate_snapshots(run, root):
    """Fail closed before comparing products or considering declarations."""
    issues = []
    if not all(key in run for key in ('documents', 'manifest', 'profile', 'manifest_sha256', 'complete')):
        return ['missing required run/manifest fields']
    if not run['complete']:
        issues.append('incomplete run')
    docs = run['documents']
    if len({d['filename'] for d in docs}) != len(docs):
        issues.append('duplicate documents')
    manifest = run.get('manifest')
    rows = manifest if isinstance(manifest, list) else (manifest or {}).get('documents', [])
    expected = {r['filename']: r for r in rows}
    if not rows or len(expected) != len(rows):
        issues.append('missing or duplicate frozen manifest rows')
    if set(expected) != {d['filename'] for d in docs}:
        issues.append('document coverage differs from frozen manifest')
    checked = {}
    for doc in docs:
        row = expected.get(doc['filename'])
        if row and doc['input_sha256'] != row['sha256']:
            issues.append(f"{doc['filename']}: input differs from frozen manifest")
        if row and run.get('profile') == 'docling-cpu-pages':
            if (doc.get('page_ranges') or [doc.get('page_range')]) != (row.get('page_ranges') or [row.get('page_range')]):
                issues.append(f"{doc['filename']}: page ranges differ from frozen manifest")
        if set(doc['stages']) != set(STAGES):
            issues.append(f"{doc['filename']}: stage coverage differs")
        for stage, state in doc['stages'].items():
            if state['status'] != 'ok':
                issues.append(f"{doc['filename']} {stage}: failed stage")
            elif run.get('schema', 1) >= 2 and 'return' not in state:
                issues.append(f"{doc['filename']} {stage}: schema-2 receipt without a wrapper return")
            files = state['files']
            if len({f['path'] for f in files}) != len(files):
                issues.append(f"{doc['filename']} {stage}: duplicate product paths")
            for f in files:
                path = (root / f['stored_path']).resolve()
                if not path.is_relative_to(root.resolve()):
                    issues.append(f"snapshot outside run: {f['stored_path']}")
                    continue
                try:
                    actual = checked.get(path)
                    if actual is None:
                        actual = (path.stat().st_size, sha256(path))
                        checked[path] = actual
                    if actual != (f['bytes'], f['sha256']):
                        issues.append(f"snapshot size/SHA-256 mismatch: {f['stored_path']}")
                except OSError:
                    issues.append(f"snapshot missing/unreadable: {f['stored_path']}")
    return issues


def compare(a, b, declarations=()):
    a, b = Path(a), Path(b)
    a = a / 'run.json' if a.is_dir() else a
    b = b / 'run.json' if b.is_dir() else b
    left, right = json.loads(a.read_text()), json.loads(b.read_text())
    integrity = [f'{side}: {issue}' for side, run, root in (('A', left, a.parent), ('B', right, b.parent))
                 for issue in validate_snapshots(run, root)]
    if integrity:
        return dict(identical=False, passes=False, undeclared_differences=0,
                    issues=['snapshot/run integrity failed'], integrity_errors=integrity,
                    unused_declarations=[], results=[])
    accepted = {}
    for d in declarations:
        required = ('document', 'stage', 'path', 'a_sha256', 'b_sha256', 'reason')
        if any(k not in d for k in required) or not d['reason'].strip():
            raise ValueError('declarations need document, stage, path, a_sha256, b_sha256, reason')
        key = tuple(d[k] for k in required[:-1])
        if key in accepted: raise ValueError('duplicate declaration')
        fields = d.get('fields', [])
        if not fields or len(set(fields)) != len(fields) or not all(timing_field(f) for f in fields):
            raise ValueError('declarations require exact timing leaf fields as JSON pointers')
        if not d['path'].endswith('.metrics.json'):
            raise ValueError('only timing metric JSON files may be declared')
        accepted[key] = d
    results, used = [], set()
    # LS1-G01: typed wrapper returns are compared when both receipts record them (schema >= 2). A schema-1 receipt
    # (older baselines, left unchanged) has none; the comparison then says so instead of passing them silently.
    schemas = (left.get('schema', 1), right.get('schema', 1))
    returns_compared = all(isinstance(v, int) and v >= 2 for v in schemas)
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
                declaration = accepted.get(key)
                reason = None
                if (declaration and detail
                        and set(detail.get('json_fields', [])) == set(declaration['fields'])
                        and set(detail.get('timing_fields', [])) == set(declaration['fields'])):
                    reason = declaration['reason']
                    used.add(key)
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
                if returns_compared and sa.get('return') != sb.get('return'):
                    # Never declarable: declarations name .metrics.json timing leaves only.
                    difference('@return', json.dumps(sa.get('return'), sort_keys=True), json.dumps(sb.get('return'), sort_keys=True))
                fa, fb = {f['path']: f for f in sa['files']}, {f['path']: f for f in sb['files']}
                for path in sorted(fa.keys() | fb.keys()):
                    x, y = fa.get(path), fb.get(path)
                    if x and y and x['sha256'] == y['sha256'] and x['bytes'] == y['bytes']: continue
                    detail = {}
                    if path.endswith('.md') and x and y:
                        detail['markdown'] = markdown_diff((a.parent / x['stored_path']).read_text(), (b.parent / y['stored_path']).read_text())
                    if path.endswith('.metrics.json') and x and y:
                        try:
                            metrics_a = json.loads((a.parent / x['stored_path']).read_text())
                            metrics_b = json.loads((b.parent / y['stored_path']).read_text())
                            detail['json_fields'] = sorted(json_changes(metrics_a, metrics_b))
                            detail['timing_fields'] = timing_changes(metrics_a, metrics_b, detail['json_fields'])
                        except (ValueError, UnicodeError):
                            pass
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
            'undeclared_differences': undeclared, 'issues': issues, 'unused_declarations': unused,
            'wrapper_returns': 'compared' if returns_compared else f'not compared: receipt schemas {list(schemas)} (returns recorded from schema 2)',
            'results': results}


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
