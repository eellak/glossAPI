"""Read-only shadow-ledger CLI. No command imports a pipeline or backend."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

from .ledger.catalogue import STAGES
from .ledger.encoding import canonical_json
from .ledger.reader import ReadRequestError, plan_view, report_view, report_why, select_run, status_view
from .ledger.writer import LedgerError, open_readonly


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ReadRequestError("invalid_request", message)


def parser():
    from .stages.retention import DEBUG_HELP, MEMORY_SAVING_HELP
    result = Parser(prog="glossapi", description=__doc__)
    commands = result.add_subparsers(dest="command", required=True, parser_class=Parser)
    for name in ("plan", "run", "status", "report"):
        command = commands.add_parser(name)
        command.add_argument("--corpus", default=".")
        command.add_argument("--json", action="store_true")
        if name in ("plan", "run"):
            sources = command.add_mutually_exclusive_group()
            sources.add_argument("--files", nargs="+", action="extend", default=[])
            sources.add_argument("--manifest")
            command.add_argument("--stages")
            command.add_argument("--device", action="append", default=[])
            command.add_argument("--workers", action="append", default=[])
            command.add_argument("--settings")
            command.add_argument("--debug", choices=("on", "off"), help="debug = on/off (default off): "+DEBUG_HELP)
            command.add_argument("--memory-saving", choices=("normal", "extreme"), help="memory_saving = normal/extreme (default normal): "+MEMORY_SAVING_HELP)
            command.add_argument("--disk-budget-bytes", type=int)
            if name == 'run':
                command.add_argument('--plan')
                command.add_argument('--resume')
                command.add_argument('--retry-failed', action='store_true')
                command.add_argument('--trust-legacy-outputs', action='store_true')
        else:
            command.add_argument("--run")
            command.add_argument("--stage", choices=sorted(STAGES))
        if name == "status":
            command.add_argument("--file")
        if name == "report":
            command.add_argument("--why")
            command.add_argument("--format", choices=("json", "parquet"), default="json")
            command.add_argument("--output")
    return result


def _assignments(values, *, integer=False):
    output = {}
    for value in values:
        stage, separator, setting = value.partition("=")
        if not separator or stage not in STAGES or stage in output:
            raise ReadRequestError("invalid_request", "Expected one STAGE=VALUE per stage")
        if integer:
            try:
                setting = int(setting)
            except ValueError:
                raise ReadRequestError("invalid_request", "Worker counts must be positive integers") from None
            if setting <= 0:
                raise ReadRequestError("invalid_request", "Worker counts must be positive")
        elif setting not in ("cpu", "mps") and not (setting.startswith("cuda:") and setting[5:].isdigit()):
            raise ReadRequestError("invalid_request", "Device must be cpu, mps or cuda:N")
        output[stage] = setting
    return output


def _json_file(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ReadRequestError("invalid_input", str(error)) from error


def _export(path, data, format):
    """The sole explicit write: atomic export, never a second state store."""
    path = Path(path).absolute()
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    os.close(descriptor)
    try:
        rows = None
        if format == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq
            records = [{key: canonical_json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in data["rows"]]
            table = pa.Table.from_pylist(records)
            pq.write_table(table, temporary)
            rows = len(records)
        else:
            Path(temporary).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        payload = Path(temporary).read_bytes()
        os.replace(temporary, path)
        return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), rows=rows)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    envelope = dict(schema_version=1, command=argv[0] if argv else None, corpus_id=None, run_id=None, data=None, warnings=[], errors=[])
    exit_code = 0
    try:
        args = parser().parse_args(argv)
        envelope["command"] = args.command
        database = Path(args.corpus).absolute() / "corpus.sqlite3"
        if not database.is_file() and args.command not in ("plan", "run"):
            raise ReadRequestError("ledger_not_initialized", "Corpus has no ledger")
        if args.command == "report":
            if args.why and args.format != "json":
                raise ReadRequestError("invalid_request", "--why requires JSON format")
            if args.format == "parquet" and not args.output:
                raise ReadRequestError("invalid_request", "Parquet export requires --output")
            if args.output and Path(args.output).absolute() in (database, database.with_name(database.name+"-wal"), database.with_name(database.name+"-shm")):
                raise ReadRequestError("invalid_request", "Cannot overwrite the ledger")
        with open_readonly(database) if database.is_file() else nullcontext(None) as connection:
            run = None
            if connection:
                envelope["corpus_id"] = connection.execute("SELECT corpus_id FROM ledger_info").fetchone()[0]
                run = select_run(connection, getattr(args, "run", None))
                envelope["run_id"] = run["run_id"] if run else None
            if args.command in ("plan", "run"):
                settings = _json_file(args.settings) if args.settings else {}
                if not isinstance(settings, dict):
                    raise ReadRequestError("invalid_settings", "Settings must be a JSON object")
                for name in ("debug", "memory_saving", "disk_budget_bytes"):
                    if getattr(args, name) is not None:
                        settings[name] = getattr(args, name)
                for name in ("disk_budget_bytes", "max_source_bytes", "max_attempt_bytes"):
                    if settings.get(name) is not None and (type(settings[name]) is not int or settings[name] < 0):
                        raise ReadRequestError("invalid_settings", name + " must be a nonnegative integer")
                manifest = _json_file(args.manifest) if args.manifest else None
                selectors = args.files
                if manifest is not None:
                    records = manifest.get("documents", []) if isinstance(manifest, dict) else manifest
                    if not isinstance(records, list):
                        raise ReadRequestError("invalid_manifest", "Manifest must contain a documents array")
                    selectors = []
                    for row in records:
                        selector = row if isinstance(row, str) else next((row[k] for k in ("document_id", "source_key", "source_uri", "path", "url") if row.get(k)), None) if isinstance(row, dict) else None
                        if not isinstance(selector, str):
                            raise ReadRequestError("invalid_manifest", "Every row requires an explicit source selector")
                        selectors.append(selector)
                data = plan_view(connection, selectors=selectors, stages=args.stages.split(",") if args.stages else None,
                    settings=settings, devices=_assignments(args.device), workers=_assignments(args.workers, integer=True), manifest=manifest)
                if args.command == 'plan':
                    envelope["warnings"].append(dict(code="conditional_plan", message="Run revalidates inputs and capabilities before dispatch"))
            elif args.command == "status":
                data = status_view(connection, run=run, stage=args.stage, selector=args.file)
                exit_code = 1 if data["counts"]["failed"] else 0
            else:
                data = report_why(connection, args.why, run=run, stage=args.stage) if args.why else report_view(connection, run=run, stage=args.stage)
            envelope["data"] = data
        if args.command == "report" and args.output:
            # Export after the short read transaction closes.
            envelope["data"] = _export(args.output, data, args.format)
        if args.command == 'run':
            if args.resume and (args.files or args.manifest or args.plan or args.stages or args.device or args.workers or args.settings or args.debug or args.memory_saving or args.disk_budget_bytes is not None):
                raise ReadRequestError('invalid_request','Resume restores the frozen plan; overrides are forbidden')
            if args.plan and (args.files or args.manifest or args.stages or args.device or args.workers or args.settings or args.debug or args.memory_saving):
                raise ReadRequestError('invalid_request','A supplied plan cannot be combined with planning overrides')
            requested=_json_file(args.plan) if args.plan else data
            if not args.resume:
                for document in requested.get('data',requested)['documents']:
                    from .ocr.defaults import validate_device
                    for step in document['steps']:
                        validate_device(step['stage'],step['device_id'],requested.get('data',requested).get('settings',{}).get(step['stage'],{}))
            root=Path(args.corpus).absolute()
            root.mkdir(parents=True,exist_ok=True)
            if not args.resume:
                from .ledger.importer import import_corpus
                from .ledger.journal import code_revision
                import_corpus(root,trust_legacy_outputs=args.trust_legacy_outputs,code_revision=code_revision())
            from . import Corpus
            corpus=Corpus(root,root)
            # Backends and libraries may print; JSON stdout remains one object.
            from contextlib import redirect_stdout
            with redirect_stdout(sys.stderr):
                saved_stdout=os.dup(1)
                try:
                    sys.stdout.flush()
                    os.dup2(2,1)
                    outcome=corpus.run(resume=args.resume,retry_failed=args.retry_failed,
                        **({'plan':requested} if not args.resume else {}))
                finally:
                    os.dup2(saved_stdout,1)
                    os.close(saved_stdout)
            envelope['data']=outcome
            envelope['run_id']=outcome['run_id']
            with open_readonly(database) as connection:
                envelope['corpus_id']=connection.execute('SELECT corpus_id FROM ledger_info').fetchone()[0]
            exit_code=3 if outcome['state']=='waiting' else 1 if outcome['state'] in ('completed_with_failures','cancelled') else 0
    except (ReadRequestError, LedgerError, sqlite3.Error, OSError, ValueError, TypeError, ImportError, RuntimeError) as error:
        code = getattr(error, "code", "unsupported_schema" if isinstance(error, LedgerError) else "read_error")
        envelope["errors"].append(dict(code=code, message=str(error), document_id=None, stage=None, details=getattr(error, "details", {})))
        exit_code = 3 if code in ("policy_conflict","disk_guard_unavailable","disk_budget_exceeded","waiting_for_disk") else 2
    # Human mode renders the same object and retains every uncertainty.
    print(json.dumps(envelope, ensure_ascii=False, indent=None if "--json" in argv else 2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
