"""Run identical text-layer workloads at 1/2/4/8 fresh spawned workers."""
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ''):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'tools.harness'
import argparse
from pathlib import Path
from .golden import run
from .common import STAGES, write_json


def union_seconds(intervals):
    total = 0.; end = None
    for left, right in sorted(intervals):
        total += max(0., right - max(left, end if end is not None else left))
        end = max(right, end if end is not None else right)
    return total


def table(receipt):
    workers = {d['worker']['pid']: d['worker'] for d in receipt['documents'] if 'worker' in d}
    rows = {}
    for stage in STAGES:
        docs = receipt['documents']
        states = [d['stages'][stage] for d in docs]
        intervals = []
        for s in states:
            if 'intervals' in s: intervals.extend(s['intervals'])
            elif 'started_monotonic' in s: intervals.append([s['started_monotonic'], s['finished_monotonic']])
        active = union_seconds(intervals)
        pages = sum(d.get('pages', 0) for d in docs)
        rss = {str(pid): max(d['stages'][stage].get('peak_rss_bytes', 0) for d in docs if d.get('worker', {}).get('pid') == pid) for pid in workers}
        rows[stage] = dict(receipt['summary'][stage], active_wall_seconds=active,
            active_seconds_per_page=active / pages if pages else None,
            active_docs_per_hour=3600 * len(docs) / active if active else None,
            rss_per_worker_bytes=rss)
    return {'workers': receipt['workers'], 'wall_seconds': receipt['wall_seconds'],
            'pipeline_docs_per_hour': 3600*len(receipt['documents'])/receipt['wall_seconds'],
            'startup_per_worker': [{**r, 'startup_from_pool_creation_seconds': r['initialized_monotonic']-receipt['pool_created_monotonic']} for r in workers.values()],
            'stages': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest'); p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--workers', nargs='+', type=int, default=[1,2,4,8]); a = p.parse_args()
    if any(n < 1 for n in a.workers): p.error('workers must be positive')
    root = Path(a.out); root.mkdir(parents=True, exist_ok=False)
    result = {'measurement_basis': 'worker_seconds/pages is service cost; active wall is union of measured stage intervals across workers (gaps excluded); pipeline throughput includes startup and all stages', 'runs': []}
    for count in a.workers:
        receipt = run(a.manifest, a.data, root/f'workers-{count}', 'text-layer', count)
        result['runs'].append(table(receipt)); write_json(root/'speed.json', result)
    return 0 if all(not s['failures'] for r in result['runs'] for s in r['stages'].values()) else 1

if __name__ == '__main__': raise SystemExit(main())
