"""Hash-bound, per-document public Corpus pipeline. Run via python -m tools.harness.golden."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ''):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = 'tools.harness'
import argparse
import concurrent.futures
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from .common import STAGES, REPO, PeakRSS, environment, inventory, load_manifest, sha256, write_json

PROFILES = ('text-layer', 'docling-cpu', 'docling-cpu-pages')
IMPORTS = None
Corpus = None


def initialize():
    global IMPORTS, Corpus
    began = time.perf_counter()
    import glossapi
    imported = time.perf_counter()
    from glossapi import Corpus
    finished = time.perf_counter()
    IMPORTS = {'pid': os.getpid(), 'import_glossapi_seconds': imported - began,
               'from_glossapi_import_Corpus_seconds': finished - imported,
               'total_import_seconds': finished - began, 'initialized_monotonic': finished,
               'basis': 'sequential imports in a fresh spawned worker; second import is incremental'}


def expected(corpus, stage):
    return {'extract': list((corpus.output_dir / 'markdown').glob('*.md')),
            'clean': list((corpus.output_dir / 'clean_markdown').glob('*.md')),
            'sections': [corpus.sections_parquet],
            'annotate': [corpus.fully_annotated_parquet],
            'export': [corpus.output_dir / 'export.jsonl']}[stage]


def document_job(row, data, root, profile):
    """Separate Corpus per document: exact attribution, no cross-corpus duplicate fit."""
    if IMPORTS is None:
        initialize()
    root = Path(root)
    ranges = row.get('page_ranges') if profile == 'docling-cpu-pages' else None
    if ranges and len(ranges) > 1:
        segments = [document_job(dict(row, page_ranges=None, page_range=window,
                    task_id=Path(row['filename']).stem + f"__pages{window[0]}-{window[1]}"), data, root, profile)
                    for window in ranges]
        result = {'filename': row['filename'], 'input_sha256': row['sha256'],
                  'worker': dict(IMPORTS), 'pages': sum(d['pages'] for d in segments),
                  'source_pages': segments[0]['source_pages'], 'page_ranges': ranges,
                  'segments': segments, 'stages': {}}
        for stage in STAGES:
            states = [d['stages'][stage] for d in segments]
            wall = sum(s['wall_seconds'] for s in states)
            result['stages'][stage] = dict(status='ok' if all(s['status'] == 'ok' for s in states) else 'failed',
                error='; '.join(s['error'] for s in states if s['error']) or None,
                wall_seconds=wall, seconds_per_page=wall/result['pages'],
                peak_rss_bytes=max(s['peak_rss_bytes'] for s in states),
                own_peak_rss_bytes=max(s['own_peak_rss_bytes'] for s in states),
                rss_sample_interval_seconds=.01,
                rss_available=all(s.get('rss_available', True) for s in states),
                rss_basis=states[0].get('rss_basis'),
                intervals=[[s['started_monotonic'], s['finished_monotonic']] for s in states],
                files=[dict(f, path=f"pages{window[0]}-{window[1]}/"+f['path'])
                       for window,d in zip(ranges,segments) for f in d['stages'][stage]['files']],
                removed_files=[f"pages{window[0]}-{window[1]}/"+p for window,d in zip(ranges,segments)
                               for p in d['stages'][stage]['removed_files']])
        write_json(root/'documents'/Path(row['filename']).stem/'document.json', result)
        return result
    if ranges:
        row = dict(row, page_range=ranges[0])
    identifier = row.get('task_id', Path(row['filename']).stem)
    jobroot = root / 'documents' / identifier
    jobroot.mkdir(parents=True, exist_ok=False)
    out = jobroot / 'work'
    out.mkdir()
    empty = jobroot / 'empty_input'
    empty.mkdir()
    (out / 'downloads').mkdir()
    (out / 'downloads' / '.harness-direct-paths').touch()
    result = {'filename': row['filename'], 'input_sha256': row['sha256'],
              'worker': dict(IMPORTS), 'stages': {}}
    write_json(jobroot / 'document.json', result)
    source = Path(data) / row['filename']
    try:
        if sha256(source) != row['sha256']:
            raise ValueError('PDF sha256 mismatch')
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(source)
        try:
            source_pages = len(pdf)
        finally:
            pdf.close()
        page_range = tuple(row['page_range']) if profile == 'docling-cpu-pages' else None
        pages = page_range[1] - page_range[0] + 1 if page_range else source_pages
        if page_range and not 1 <= page_range[0] <= page_range[1] <= source_pages:
            raise ValueError('page range outside source')
        result.update(pages=pages, source_pages=source_pages, page_range=page_range)
    except Exception:
        error = traceback.format_exc()
        result['stages'] = {stage: {'status': 'failed', 'error': error, 'files': [], 'wall_seconds': 0} for stage in STAGES}
        write_json(jobroot / 'document.json', result)
        return result
    write_json(jobroot / 'document.json', result)
    os.environ['GLOSSAPI_CONVERSION_LOG_FILE'] = str(jobroot / 'conversion.log')
    corpus = Corpus(empty, out, section_classifier_model_path=REPO / 'src/glossapi/models/section_classifier.joblib',
                    extraction_model_path=REPO / 'src/glossapi/models/kmeans_weights.joblib')
    calls = {'extract': lambda: corpus.extract(input_format='pdf', file_paths=[source], phase1_backend='safe' if profile == 'text-layer' else 'docling',
                    accel_type='CPU', num_threads=1, skip_existing=False, batched=False, repair_fonts=True,
                    **({'page_range': page_range} if page_range else {})),
             'clean': lambda: corpus.clean(num_threads=1), 'sections': corpus.section,
             'annotate': corpus.annotate, 'export': lambda: corpus.jsonl(out / 'export.jsonl')}
    for stage in STAGES:
        before = inventory(out)
        status, error = 'ok', None
        began = time.perf_counter()
        with PeakRSS() as rss:
            try:
                calls[stage]()
                required = expected(corpus, stage)
                if not required or not all(p.is_file() for p in required):
                    status, error = 'failed', f'Corpus returned without expected {stage} product (filtered/empty input or swallowed error); see pipeline.log'
            except Exception:
                status, error = 'failed', traceback.format_exc()
        wall = time.perf_counter() - began
        after = inventory(out)
        files = []
        for name, receipt in after.items():
            if before.get(name) != receipt:
                snapshot = jobroot / 'snapshots' / stage / name
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out / name, snapshot)
                files.append(dict(receipt, path=name, stored_path=str(snapshot.relative_to(root))))
        result['stages'][stage] = {'status': status, 'error': error, 'wall_seconds': wall,
            'seconds_per_page': wall / pages if pages else None,
            'started_monotonic': began, 'finished_monotonic': began + wall, 'peak_rss_bytes': rss.peak,
            'own_peak_rss_bytes': rss.own_peak, 'rss_sample_interval_seconds': .01,
            'rss_available': getattr(rss, 'available', True),
            'rss_basis': getattr(rss, 'basis', None),
            'rss_unavailable_reason': getattr(rss, 'unavailable_reason', None),
            'files': files, 'removed_files': sorted(set(before) - set(after))}
        write_json(jobroot / 'document.json', result)
    return result


def logged_job(*args):
    root = Path(args[2]) / 'logs'
    root.mkdir(exist_ok=True)
    with (root / (Path(args[0]['filename']).stem + '.log')).open('w') as log:
        # Native extension/subprocess output is redirected as well.
        saved = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            return document_job(*args)
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            os.close(saved[0]); os.close(saved[1])


def summarize(documents):
    summary = {}
    for stage in STAGES:
        rows = [(d, d['stages'][stage]) for d in documents]
        seconds = sum(s.get('wall_seconds', 0) for _, s in rows)
        pages = sum(d.get('pages', 0) for d, _ in rows)
        summary[stage] = {'documents': len(rows), 'ok': sum(s['status'] == 'ok' for _, s in rows),
            'failures': [{'filename': d['filename'], 'error': s.get('error')} for d, s in rows if s['status'] != 'ok'],
            'pages_attempted': pages, 'worker_seconds': seconds, 'seconds_per_page': seconds / pages if pages else None,
            'docs_per_hour': 3600 * len(rows) / seconds if seconds else None,
            'peak_rss_bytes': max((s.get('peak_rss_bytes', 0) for _, s in rows), default=0)}
    return summary


def run(manifest, data, output, profile='text-layer', workers=1):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = load_manifest(manifest)
    if profile == 'docling-cpu-pages' and any('page_range' not in r for r in rows):
        raise ValueError('docling-cpu-pages requires frozen page_range for every document')
    receipt = {'schema': 1, 'profile': profile, 'workers': workers, 'manifest_sha256': sha256(manifest),
               'manifest': json.loads(Path(manifest).read_text()), 'environment': environment(),
               'scope': 'one fresh Corpus per document; default clean policy; stages called even after earlier failures',
               'rss_basis': 'sampled process + descendants absolute RSS; 10ms interval; shared pages may be counted twice',
               'documents': [], 'complete': False}
    write_json(root / 'run.json', receipt)
    began = time.perf_counter()
    receipt['pool_created_monotonic'] = began
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        futures = {pool.submit(logged_job, r, str(Path(data).resolve()), str(root), profile): r for r in rows}
        for future in concurrent.futures.as_completed(futures):
            try:
                doc = future.result()
            except Exception:
                row = futures[future]
                job = root / 'documents' / Path(row['filename']).stem / 'document.json'
                doc = json.loads(job.read_text()) if job.exists() else {'filename': row['filename'], 'input_sha256': row['sha256'], 'stages': {}}
                for stage in STAGES:
                    doc['stages'].setdefault(stage, {'status': 'failed', 'error': traceback.format_exc(), 'files': [], 'wall_seconds': 0})
            receipt['documents'].append(doc)
            receipt['documents'].sort(key=lambda d: d['filename'])
            receipt['wall_seconds'] = time.perf_counter() - began
            receipt['summary'] = summarize(receipt['documents'])
            write_json(root / 'run.json', receipt)
            print(f"{len(receipt['documents'])}/{len(rows)} {doc['filename']}", flush=True)
    receipt.update(complete=True, wall_seconds=time.perf_counter() - began)
    write_json(root / 'run.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--profile', choices=PROFILES, default='text-layer')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1: parser.error('workers must be positive')
    result = run(args.manifest, args.data, args.out, args.profile, args.workers)
    return 0 if all(s['ok'] == s['documents'] for s in result['summary'].values()) else 1

if __name__ == '__main__':
    raise SystemExit(main())
