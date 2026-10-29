"""Whole-corpus public phase harness and paired shadow-ledger cost measurement.

Run `python -m tools.harness.corpus MANIFEST --data DIR --out DIR` on a
worker. Each arm is a fresh process, ONE Corpus, and one call per public phase.
"""
from __future__ import annotations

from glossapi.ledger.compact import connect as ledger_connect

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

from .common import REPO, STAGES, environment, inventory, load_manifest, sha256, write_json


def audit_ledger(path, documents):
    from glossapi.ledger.writer import integrity
    with ledger_connect(path) as database:
        integrity(database)
        counts = {name: database.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                  for name in ("events", "files", "snapshots", "runs", "documents", "tasks", "attempts", "source_records")}
        kinds = [dict(kind=kind, count=count, stored_json_bytes=size) for kind, count, size in database.execute(
            "SELECT kind,COUNT(*),SUM(LENGTH(CAST(json AS BLOB))) FROM snapshots GROUP BY kind ORDER BY 3 DESC")]
        for item in kinds:
            item['json_bytes']=sum(len(encoded.encode('utf-8')) for encoded, in database.execute('SELECT json FROM snapshots WHERE kind=?',(item['kind'],)))
        pages = [dict(name=name, bytes=size) for name, size in database.execute(
            "SELECT name,SUM(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC")]
        stores = [json.loads(row[0]) for row in database.execute("SELECT json FROM snapshots WHERE kind='legacy_store'")]
        stage_storage = audit_stage_storage(database, documents)

    database.close()
    size = path.stat().st_size
    assert not path.with_name(path.name + "-wal").exists()
    assert not path.with_name(path.name + "-shm").exists()
    return dict(bytes=size, bytes_per_document=size/documents, events_per_document=counts["events"]/documents,
                counts=counts, snapshot_kinds=kinds, sqlite_pages=pages, sha256=sha256(path),
                store_observations=len(stores), full_imports=sum(s.get("mode")=="first_import" for s in stores),
                changed_rows=sum(len(s.get("changed", {})) for s in stores), stage_storage=stage_storage)



def audit_stage_storage(database, documents):
    """Attribute unique snapshot JSON to its referencing stage, including sharing.

    SQLite table/index/free-page allocation remains in sqlite_pages; logical
    UTF-8 bytes here are not falsely described as physical database allocation.
    """
    from collections import defaultdict
    snapshots={digest:(kind,encoded) for digest,kind,encoded in database.execute('SELECT sha256,kind,json FROM snapshots')}
    owners=defaultdict(set)
    for row in database.execute("SELECT stage,specification_sha256 FROM tasks"):
        owners[row[1]].add(row[0])
    for row in database.execute("SELECT t.stage,a.fingerprint_sha256,a.input_manifest_sha256,a.settings_sha256,a.tools_sha256 FROM attempts a JOIN tasks t ON t.task_id=a.task_id"):
        for digest in row[1:]: owners[digest].add(row[0])
    event_bytes=defaultdict(int)
    for stage,manifest,encoded in database.execute("SELECT COALESCE(t.stage,'run'),e.output_manifest_sha256,e.details_json FROM events e LEFT JOIN tasks t ON t.task_id=e.task_id"):
        event_bytes[stage]+=len(encoded.encode('utf-8'))
        if manifest: owners[manifest].add(stage)
        details=json.loads(encoded)
        for key,value in details.items():
            if isinstance(value,str) and value in snapshots: owners[value].add(stage)
    for row in database.execute('SELECT run_id,settings_sha256,plan_sha256,environment_sha256 FROM runs'):
        stages={item[0] for item in database.execute('SELECT DISTINCT stage FROM tasks WHERE run_id=?',(row[0],))} or {'run'}
        for digest in row[1:]: owners[digest].update(stages)
    def references(value):
        if isinstance(value,str):
            if value in snapshots: yield value
        elif isinstance(value,dict):
            for item in value.values(): yield from references(item)
        elif isinstance(value,list):
            for item in value: yield from references(item)
    edges={digest:set(references(json.loads(encoded)))-{digest} for digest,(kind,encoded) in snapshots.items()}
    changed=True
    while changed:
        changed=False
        for digest,children in edges.items():
            for child in children:
                prior=len(owners[child]);owners[child].update(owners[digest]);changed |= len(owners[child])!=prior
    groups=defaultdict(lambda:{'snapshots':0,'json_bytes':0})
    for digest,(kind,encoded) in snapshots.items():
        stage=next(iter(owners[digest])) if len(owners[digest])==1 else 'shared' if owners[digest] else 'unattributed'
        group=groups[(stage,kind)];group['snapshots']+=1;group['json_bytes']+=len(encoded.encode('utf-8'))
    return {'basis':'unique canonical snapshot UTF-8 JSON; shared facts charged once; physical page allocation is separate',
        'snapshot_kinds':[{'stage':stage,'kind':kind,**group,'bytes_per_document':group['json_bytes']/documents}
            for (stage,kind),group in sorted(groups.items())],
        'event_details_bytes':dict(event_bytes)}

def run_arm(manifest, data, root, enabled, *, streaming=False, workers=1, debug='off'):
    from glossapi import Corpus
    from glossapi.ledger.journal import wrapper_return_receipt
    from .golden import expected
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    rows = load_manifest(manifest)
    sources = [Path(data).resolve()/row["filename"] for row in rows]
    for source, row in zip(sources, rows):
        if sha256(source) != row["sha256"]:
            raise ValueError(f"Input hash mismatch: {source}")
    out = root/"work"; out.mkdir()
    empty = root/"empty_input"; empty.mkdir()
    (out/"downloads").mkdir(); (out/"downloads"/".harness-direct-paths").touch()
    # A whole acquired corpus has source rows before extract checkpoints run.
    # Starting with no rows exercises the legacy missing-row pd.NA bug instead
    # of the normal corpus phase flow (recorded separately in GAPS.md).
    import pandas as pd
    from glossapi.parquet_schema import ParquetSchema
    metadata = out/"download_results"/"download_results.parquet"
    ParquetSchema().write_metadata_parquet(pd.DataFrame([
        dict(filename=row["filename"], filename_base=Path(row["filename"]).stem,
             url=row.get("pdf_url", source.as_uri()), download_success=True, file_ext="pdf",
             processing_stage="", extract_status="") for row, source in zip(rows, sources)]), metadata)
    os.environ["GLOSSAPI_CONVERSION_LOG_FILE"] = str(root/"conversion.log")
    receipt = dict(mode="whole-corpus", documents=len(rows), ledger_enabled=enabled,
                   stage_profile=os.environ.get("GLOSSAPI_STAGE_PROFILE", "cutover"),
                   manifest_sha256=sha256(manifest), environment=environment(), stages={}, complete=False)
    write_json(root/"run.json", receipt)
    began = time.perf_counter()
    corpus = Corpus(empty, out, section_classifier_model_path=REPO/"src/glossapi/models/section_classifier.joblib",
                    extraction_model_path=REPO/"src/glossapi/models/kmeans_weights.joblib")
    corpus._shadow_ledger_enabled = enabled
    corpus._retention_settings={'debug':debug,'memory_saving':'normal'}
    if enabled:
        from .retention import RetentionSnapshots
        corpus._retention_observer=RetentionSnapshots(out,root/'retention_snapshots')
    calls = dict(extract=lambda: corpus.extract(input_format="pdf", file_paths=sources, phase1_backend="safe",
        accel_type="CPU", num_threads=1, skip_existing=False, batched=False, repair_fonts=True),
        clean=lambda: corpus.clean(num_threads=1), sections=corpus.section, annotate=corpus.annotate,
        export=lambda: corpus.jsonl(out/"export.jsonl"))
    capture_overhead=[]
    def capture(stage,call):
        capture_began=time.perf_counter()
        before = inventory(out)
        status, error, returned = "ok", None, None
        value=None
        started = time.perf_counter()
        try:
            value = call()
            returned = ({"type": "path", "relative_to_output": str(value.relative_to(out))}
                        if isinstance(value, Path) else wrapper_return_receipt(value))
            required = expected(corpus, stage)
            if not required or not all(p.is_file() for p in required):
                raise RuntimeError(f"Corpus returned without required {stage} product")
        except Exception:
            status, error = "failed", traceback.format_exc()
        wall = time.perf_counter() - started
        after = inventory(out)
        products = {}
        additions = []
        for name, fact in after.items():
            if name == "corpus.sqlite3" or name.startswith(".glossapi/artifacts/"):
                if name.startswith(".glossapi/artifacts/"):
                    with closing(ledger_connect(f"file:{out/'corpus.sqlite3'}?mode=ro"+('' if streaming else '&immutable=1'), uri=True)) as database:
                        recorded = database.execute("SELECT sha256,bytes,presence FROM current_file_state WHERE uri=?", ((out/name).as_uri(),)).fetchone()
                    if recorded != (fact["sha256"], fact["bytes"], "present"):
                        raise AssertionError(f"Unregistered or changed managed addition: {name}")
                additions.append(dict(path=name, **fact)); continue
            if name in ("corpus.sqlite3-wal", "corpus.sqlite3-shm"):
                if streaming:continue
                raise AssertionError("Ledger side files survived wrapper close")
            products[name] = fact
            if before.get(name) != fact:
                snapshot = root/"snapshots"/stage/name
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out/name, snapshot)
        receipt["stages"][stage] = dict(status=status, error=error, wall_seconds=wall,
            wrapper_return=returned, products=products, removed_files=sorted(set(before)-set(after)), ledger_additions=additions)
        write_json(root/"run.json", receipt)
        capture_overhead.append(time.perf_counter()-capture_began-wall)
        if error and streaming:raise RuntimeError(error)
        return value
    if streaming:
        if not enabled:raise ValueError('Streaming execution requires the ledger')
        from functools import wraps
        for stage,method in [('extract','extract'),('clean','clean'),('sections','section'),('annotate','annotate'),('export','jsonl')]:
            original=getattr(corpus,method)
            def traced(*arguments,_stage=stage,_method=original,**options):
                return capture(_stage,lambda:_method(*arguments,**options))
            setattr(corpus,method,wraps(original)(traced))
        result=corpus.run(sources,workers={'extract':workers,'clean':1,'sections':2},
            stages=STAGES,settings={'cpu_tokens':max(2,workers),'debug':debug})
        receipt['streaming']=result
        if result['state']!='completed':raise RuntimeError('Streaming run did not complete')
    else:
        for stage in STAGES:capture(stage,calls[stage])
    receipt.update(complete=True, wall_seconds=time.perf_counter()-began,
                   phase_wall_seconds=(result['total_wall_seconds']-sum(capture_overhead)) if streaming else sum(s["wall_seconds"] for s in receipt["stages"].values()),
                   harness_capture_seconds=sum(capture_overhead))
    if enabled:
        from .retention import verified_snapshots
        retained=verified_snapshots(out,root/'retention_snapshots')
        receipt['retention_deletions']=retained
        receipt['retention_corpus']=str(out)
        receipt['retention_snapshots']=str(root/'retention_snapshots')
        # Streaming wrapper captures precede the parent run's retention. Bring
        # temporary reports to the same post-retention point as sequential
        # captures, only with an exact completed deletion and saved byte proof.
        from glossapi.stages.retention import retention_facts
        retired_reports=[]
        with closing(ledger_connect(f"file:{out/'corpus.sqlite3'}?mode=ro&immutable=1",uri=True)) as database:
            database.row_factory=sqlite3.Row
            for stage,state in receipt['stages'].items():
                fact=state['products'].get('cleaning_report.parquet')
                if not fact or (out/'cleaning_report.parquet').exists():continue
                row=database.execute("SELECT * FROM current_file_state WHERE uri=? AND sha256=? AND bytes=? AND kind='temporary_cleaning_report' AND presence='deleted' ORDER BY event_id DESC LIMIT 1",((out/'cleaning_report.parquet').as_uri(),fact['sha256'],fact['bytes'])).fetchone()
                if row is None:raise AssertionError('Temporary report vanished without exact retention proof')
                proof=retention_facts(database,row['file_id'])
                assert proof['consumer_ok_event_ids'] and not proof['outstanding_consumers'] and not proof['pins']
                assert any(event['reason_code'] in ('delete_completed','delete_recovered') for event in proof['deletions'])
                saved=root/'snapshots'/stage/'cleaning_report.parquet'
                if saved.exists():assert sha256(saved)==fact['sha256'] and saved.stat().st_size==fact['bytes']
                retired_reports.append(dict(stage=stage,file_id=row['file_id'],**fact))
                del state['products']['cleaning_report.parquet']
        receipt['intermediate_retention_point']='after completed run retention'
        receipt['retired_temporary_reports']=retired_reports
    receipt["ledger"] = audit_ledger(out/"corpus.sqlite3", len(rows)) if enabled else None
    write_json(root/"run.json", receipt)
    return receipt


def compare_arms(left, right):
    differences = []
    retained={}
    for label,arm in [('left',left),('right',right)]:
        if arm.get('retention_deletions'):
            from .retention import verified_snapshots
            retained[label]=verified_snapshots(arm['retention_corpus'],arm['retention_snapshots'])
        else:retained[label]=[]
    def surface(state,key,stage,label):
        if key=='products':
            value=dict(state[key])
            for item in retained[label]:
                if list(STAGES).index(stage)>=list(STAGES).index(item['stage']):
                    fact={'sha256':item['sha256'],'bytes':item['bytes']}
                    if item['path'] in value and value[item['path']]!=fact:
                        return value  # Keep the real changed-product difference visible.
                    value[item['path']]=fact
            return value
        if key=='removed_files':return [name for name in state[key] if name not in {item['path'] for item in retained[label]}]
        return state[key]
    for stage in STAGES:
        a, b = left["stages"][stage], right["stages"][stage]
        for key in ("status", "error", "wrapper_return", "products", "removed_files"):
            left_value=surface(a,key,stage,'left');right_value=surface(b,key,stage,'right')
            if left_value != right_value:
                differences.append(dict(stage=stage, field=key, disabled=left_value, enabled=right_value))
    return dict(passes=not differences, differences=differences)


def paired(manifest, data, output, repeats=2):
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    result = dict(scope="one fresh Corpus per arm, all documents, normal public phase calls", pairs=[], complete=False)
    # Reverse ordering to expose warm-cache/order effects; no timing claim is
    # based on the per-document harness or an estimate of observation cost.
    for repeat in range(repeats):
        arms = {}
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            name = f"pair-{repeat+1}-{'enabled' if enabled else 'disabled'}"
            command = [sys.executable, "-m", "tools.harness.corpus", str(Path(manifest).resolve()),
                       "--data", str(Path(data).resolve()), "--out", str(root/name), "--arm",
                       "enabled" if enabled else "disabled"]
            with (root/(name+".log")).open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Corpus arm failed: {name}; exit {completed.returncode}")
            arms[enabled] = json.loads((root/name/"run.json").read_text())
        comparison = compare_arms(arms[False], arms[True])
        disabled, enabled = (arms[key]["phase_wall_seconds"] for key in (False, True))
        result["pairs"].append(dict(disabled_seconds=disabled, enabled_seconds=enabled,
            overhead_seconds=enabled-disabled, overhead_percent=100*(enabled/disabled-1),
            comparison=comparison, ledger=arms[True]["ledger"]))
        write_json(root/"results.json", result)
        if not comparison["passes"]:
            raise AssertionError("Whole-corpus ordinary products/returns/removals differ")
    from glossapi.ledger import LedgerWriter
    with LedgerWriter(root/"empty.sqlite3"):
        pass
    minimum = (root/"empty.sqlite3").stat().st_size
    result.update(complete=True, empty_schema_bytes=minimum,
                  empty_schema_bytes_per_document=minimum/len(load_manifest(manifest)),
                  minimum_basis="one empty schema per corpus, not one schema per document")
    write_json(root/"results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest"); parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--arm", choices=("enabled", "disabled"))
    parser.add_argument('--streaming',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--debug',choices=('off','on'),default='off')
    parser.add_argument("--stage-profile", choices=("legacy", "sequential-adapters", "cutover"), default=os.environ.get("GLOSSAPI_STAGE_PROFILE", "cutover"))
    args = parser.parse_args()
    os.environ["GLOSSAPI_STAGE_PROFILE"] = args.stage_profile
    if args.arm:
        result = run_arm(args.manifest, args.data, args.out, args.arm == "enabled",streaming=args.streaming,workers=args.workers,debug=args.debug)
        return 0 if all(s["status"] == "ok" for s in result["stages"].values()) else 1
    if args.repeats < 1: parser.error("repeats must be positive")
    paired(args.manifest, args.data, args.out, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
