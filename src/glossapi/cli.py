"""glossAPI ledger CLI: plan, run, status, report, capabilities, doctor, install and guide.

plan/status/report/capabilities/doctor/guide are read-only and import no pipeline or
backend; run executes the frozen streaming plan; install previews (default) or, with
--yes, runs the exact package install commands. status/report without --run
select the latest run the user started (a streaming run, not its compatibility
wrapper children). Every command prints one JSON envelope (schema v1:
glossapi/schemas/v1/envelope.json).
"""
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
from typing import Any

from .ledger.catalogue import STAGES
from .ledger.encoding import canonical_json
from .ledger.reader import ReadRequestError, plan_view, report_view, report_why, select_run, status_view
from .ledger.writer import LedgerError, open_readonly
from .package_catalog import CatalogError


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ReadRequestError("invalid_request", message)


def parser():
    from .settings_contract import PLAN_FIXED_NOTE, option_help
    result = Parser(prog="glossapi", description=__doc__)
    commands = result.add_subparsers(dest="command", required=True, parser_class=Parser)
    for name in ("plan", "run", "status", "report", "capabilities", "doctor", "install", "guide"):
        command = commands.add_parser(name, epilog=PLAN_FIXED_NOTE if name in ("plan", "run") else None)
        command.add_argument("--json", action="store_true")
        if name == "capabilities":
            command.add_argument("--catalog", action="store_true",
                help="also list every optional package: what it adds, devices, measured install size and package count, licences, installed state")
            continue
        if name == "guide":
            continue
        if name == "install":
            command.add_argument("--device", action="append", default=[],
                help="the one device profile of this environment: cpu, cuda126, cuda132 or mps (selects the torch build of layout/ocr)")
            command.add_argument("--package", action="append", default=[],
                help="an optional package from `capabilities --catalog` (repeat or comma-separate); only these are installed")
            command.add_argument("--yes", action="store_true", help="execute the previewed commands (default: dry run)")
            command.add_argument("--installer", choices=("auto", "pip", "uv"), default="auto")
            continue
        command.add_argument("--corpus", default=".")
        if name == "doctor":
            command.add_argument("--probe-devices", action="store_true",
                help="also confirm CUDA/MPS with the framework: imports torch in a child process (the result says so)")
            continue
        if name in ("plan", "run"):
            sources = command.add_mutually_exclusive_group()
            sources.add_argument("--files", nargs="+", action="extend", default=[])
            sources.add_argument("--manifest")
            command.add_argument("--stages")
            command.add_argument("--device", action="append", default=[])
            command.add_argument("--workers", action="append", default=[])
            command.add_argument("--settings")
            command.add_argument("--debug", choices=("off", "on"), help=option_help("debug"))
            command.add_argument("--memory-saving", choices=("normal", "extreme"), help=option_help("memory_saving"))
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
    """The sole explicit write: atomic export, never a second state store.

    An export is never a metadata writer (K2F round 4, CP-K2e F1): a destination that resolves to an existing metadata
    parquet (``selectable_metadata_parquet``), or to a ``download_results/*.parquet`` name, is refused with
    ``MetadataDestinationError``, and a hard-linked destination with ``MetadataAliasError``, before anything is
    written (CP-K3 F2)."""
    from .parquet_schema import refuse_metadata_destination
    refuse_metadata_destination(path, writer="export")
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


def _merge(name, given, manifest_value):
    """An argument and the manifest may both name a value only when they agree (per stage for mappings)."""
    if isinstance(given, dict) and isinstance(manifest_value, dict):
        clashes = {key: [given[key], manifest_value[key]] for key in given.keys() & manifest_value.keys() if given[key] != manifest_value[key]}
        if clashes:
            raise ReadRequestError("conflicting_request", f"--{name} disagrees with the manifest", details={"field": name, "clashes": clashes})
        return {**manifest_value, **given}
    if given and manifest_value and given != manifest_value:
        raise ReadRequestError("conflicting_request", f"--{name} disagrees with the manifest's {name}",
                               details={"field": name, "argument": given, "manifest": manifest_value})
    return given or manifest_value


def _manifest_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ReadRequestError("invalid_manifest", "Manifest file is not readable JSON: " + str(error), details={"path": str(path)}) from error


def _plan_data(connection, root, *, files=(), manifest=None, stages=None, devices=None, workers=None, settings=None, flags=None):
    """The exact plan (``plan --json`` data) and the parsed manifest; reads only.

    Settings precedence: manifest settings < --settings file < individual flags.
    Stages/devices/workers given both as arguments and in a v1 manifest must agree.
    """
    from .input_manifest import parse_manifest
    if manifest is not None:
        # Supported url_index cells (numpy/pandas integers, NaN) become plain values once, before both the identities and
        # the plan hash read them (CP-K4gc2 R4).
        from .stages.run import normalize_url_index_cells
        manifest = normalize_url_index_cells(manifest)
    parsed = parse_manifest(manifest) if manifest is not None else None
    settings = {**(parsed.settings if parsed else {}), **(settings or {})}
    for name, value in (flags or {}).items():
        if value is not None:
            settings[name] = value
    for name in ("disk_budget_bytes", "max_source_bytes", "max_attempt_bytes"):
        if settings.get(name) is not None and (type(settings[name]) is not int or settings[name] < 0):
            raise ReadRequestError("invalid_settings", name + " must be a nonnegative integer")
    selectors, rows = list(files or []), [{} for _ in files or []]
    stages = _merge("stages", stages, parsed.stages if parsed else None)
    devices = _merge("device", devices or {}, parsed.devices if parsed else {})
    workers = _merge("workers", workers or {}, parsed.workers if parsed else {})
    if parsed is not None:
        selectors, rows = list(parsed.selectors), list(parsed.rows)
    # Public targets only (PLAN §9): plan_view refuses credentialed selectors, exclusions and download settings.
    identities = None
    if any(str(selector).startswith(("http://", "https://")) for selector in selectors):
        # The plan carries each URL's resource identity (a declared source_id keys its own).
        # A declared source_id is keyed by (dataset_id, source_id, url_index), exactly as run keys it (CRED-F8), and
        # two different URLs on one identity are refused with run's typed error instead of collapsing into one target.
        from .stages.run import _bind, resource_identity, row_resource_index
        identities = [resource_identity(selector, source_id=row.get("source_id"), dataset_id=row.get("dataset_id"),
                                        resource_index=row_resource_index(row, index=index))
                      if str(selector).startswith(("http://", "https://")) else None
                      for index, (selector, row) in enumerate(zip(selectors, rows))]
        bound: dict[str, str] = {}
        for index, (identity, selector) in enumerate(zip(identities, selectors)):
            if identity is not None:
                _bind(bound, identity, selector, index=index)
    # Only an omitted selection (no manifest, no files) means every corpus document;
    # a manifest is an explicit selection even when it selects nothing (CP-K3 F1).
    explicit = parsed is not None or bool(files)
    data = plan_view(connection, selectors=selectors if explicit else None, stages=stages, settings=settings, devices=devices, workers=workers,
        manifest=parsed.selected if parsed else None, identities=identities,
        exclusions=parsed.exclusions if parsed else None, input_manifest=parsed.describe() if parsed else None)
    return data, parsed


def plan_documents(corpus=".", *, files=(), manifest=None, stages=None, devices=None, workers=None, settings=None,
                   debug=None, memory_saving=None, disk_budget_bytes=None):
    """Python API for ``glossapi plan --json``: the same exact plan data the CLI prints.

    ``manifest`` is a decoded manifest (see glossapi.input_manifest) or a path to a
    JSON file. Typed failures raise ReadRequestError, ManifestError or
    PlanBindingError (each has ``code`` and ``details``); a plan with a request
    conflict (unsupported or unavailable device) raises ReadRequestError. A manifest
    that selects nothing returns zero targets with execution.ready false (blocking
    conflict ``empty_selection``); running that plan raises PlanBindingError.
    """
    from .plan_preview import plan_errors
    if isinstance(manifest, (str, os.PathLike)):
        manifest = _manifest_file(manifest)
    database = Path(corpus).absolute() / "corpus.sqlite3"
    with open_readonly(database) if database.is_file() else nullcontext(None) as connection:
        data, _ = _plan_data(connection, Path(corpus).absolute(), files=list(files), manifest=manifest, stages=list(stages) if stages else None,
            devices=dict(devices or {}), workers=dict(workers or {}), settings=dict(settings or {}),
            flags=dict(debug=debug, memory_saving=memory_saving, disk_budget_bytes=disk_budget_bytes))
    errors = plan_errors(data)
    if errors:
        raise ReadRequestError(errors[0]["code"], errors[0]["message"], details={"conflicts": errors})
    return data


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    envelope: dict[str, Any] = dict(schema_version=1, command=argv[0] if argv else None, corpus_id=None, run_id=None, data=None, warnings=[], errors=[])
    exit_code = 0
    parsed = None
    try:
        args = parser().parse_args(argv)
        envelope["command"] = args.command
        if args.command == "capabilities":
            from .capabilities import capabilities_view
            envelope["data"] = data = capabilities_view()
            if args.catalog:
                from .package_catalog import catalog_view
                data["catalog"] = catalog = catalog_view()
                stale = [row["name"] for row in catalog["profiles"] if row["measurement"]["state"] != "current"]
                if stale:
                    envelope["warnings"].append(dict(code="catalog_measurement_stale", message="Install sizes of "
                        + ", ".join(stale) + " were measured from older requirements (tools/catalog/measure_profiles.py)"))
            return _emit(envelope, argv, 0)
        if args.command == "guide":
            from .llm_guide import guide_view
            envelope["data"] = guide_view()
            return _emit(envelope, argv, 0)
        if args.command == "install":
            from .install_plan import execute_install, plan_install
            plan = plan_install(args.device, args.package, installer=args.installer)
            envelope["warnings"].extend(plan["warnings"])
            envelope["data"] = plan
            if args.yes:
                data, failure = execute_install(plan, installer=args.installer)
                envelope["data"] = data
                envelope["warnings"] = list(data["warnings"])
                if failure is not None:
                    raise failure
            return _emit(envelope, argv, 0)
        if args.command == "doctor":
            from .doctor import doctor_view
            envelope["data"] = data = doctor_view(args.corpus, probe=args.probe_devices)
            envelope["corpus_id"] = data["corpus"]["corpus_id"]
            if args.probe_devices:
                envelope["warnings"].append(dict(code="device_probe_imported_torch", message="--probe-devices imported torch in a child process"))
            return _emit(envelope, argv, 0 if data["healthy"] else 1)
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
                data, parsed = _plan_data(connection, Path(args.corpus).absolute(), files=args.files,
                    manifest=_manifest_file(args.manifest) if args.manifest else None, stages=args.stages.split(",") if args.stages else None,
                    devices=_assignments(args.device), workers=_assignments(args.workers, integer=True), settings=settings,
                    flags={name: getattr(args, name) for name in ("debug", "memory_saving", "disk_budget_bytes")})
                if args.command == 'plan':
                    envelope["warnings"].append(dict(code="conditional_plan", message="Run revalidates inputs and capabilities before dispatch"))
                    envelope["warnings"].extend(dict(code=item["code"], message=item["message"], stage=item.get("stage"))
                        for item in data["conflicts"] if item["severity"] in ("warning", "blocking"))
                    from .plan_preview import plan_errors
                    errors = plan_errors(data)
                    if errors:
                        # The preview stays visible next to the typed error.
                        envelope["data"] = data
                        raise ReadRequestError(errors[0]["code"], errors[0]["message"], details={"conflicts": errors})
            elif args.command == "status":
                data = status_view(connection, run=run, stage=args.stage, selector=args.file)
                exit_code = 1 if data["counts"]["failed"] else 0
                if data.get("state") == "unresolved_recovery":
                    # The run's completion gate refused (fix 13): exit 3, typed, as the refused run did.
                    exit_code = 3
                    envelope["errors"].append(dict(code="unresolved_recovery", message="the run's completion gate refused its products",
                        document_id=None, stage=None, details={"conditions": (data.get("completion_gate") or {}).get("failures", [])}))
                if exit_code and run is not None:
                    # Exit 1 names its failures as typed errors, as run does (INTEG-G2), never an empty list.
                    from .ledger.reader import status_failure_errors
                    envelope["errors"].extend(status_failure_errors(connection, run, data))
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
                # Plan identity (hash, inputs, installed capabilities) and conflicts,
                # before the ledger is touched (K3, PLAN 2.4).
                from .plan_preview import refuse_conflicts, verify_plan_binding
                if args.plan:
                    if not isinstance(requested, dict):
                        raise ReadRequestError('invalid_plan', 'A plan file holds a JSON object')
                    verify_plan_binding(requested)
                else:
                    refuse_conflicts(data)
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
            if not args.resume and isinstance(outcome, dict):
                # The CLI plan identity this run was bound to (verified above); the
                # scheduler's own frozen-plan hash stays in plan_hash.
                outcome={**outcome,'bound_plan_hash':requested.get('data',requested).get('plan_hash')}
            envelope['data']=outcome
            envelope['run_id']=outcome['run_id']
            failed=outcome['state'] in ('completed_with_failures','cancelled')
            with open_readonly(database) as connection:
                envelope['corpus_id']=connection.execute('SELECT corpus_id FROM ledger_info').fetchone()[0]
                if failed:
                    # A run that ran and found failures (exit 1) reports them as typed errors, not an empty list.
                    from .ledger.reader import run_failure_errors
                    envelope['errors'].extend(run_failure_errors(connection, outcome['run_id'], state=outcome['state'], counts=outcome.get('counts')))
            exit_code=3 if outcome['state']=='waiting' else 1 if failed else 0
    except (ReadRequestError, CatalogError, LedgerError, sqlite3.Error, OSError, ValueError, TypeError, ImportError, RuntimeError) as error:
        code = getattr(error, "code", "unsupported_schema" if isinstance(error, LedgerError) else "read_error")
        envelope["errors"].append(dict(code=code, message=str(error), document_id=None, stage=None, details=getattr(error, "details", {})))
        if envelope.get("command") == "run" and not isinstance((envelope.get("data") or {}).get("state"), str):
            envelope["data"] = None  # a refused run has no run data (the plan is not its result)
        from .stages.disk_budget import ERRORS as DISK_GUARD_ERRORS
        exit_code = 3 if code in ("policy_conflict","disk_budget_exceeded","waiting_for_disk","disk_full","post_close_edit","unresolved_recovery",*DISK_GUARD_ERRORS) else 2
    return _emit(envelope, argv, exit_code)


def _emit(envelope, argv, exit_code):
    """Print the one JSON envelope. Human mode renders the same object and retains every uncertainty."""
    print(json.dumps(envelope, ensure_ascii=False, indent=None if "--json" in argv else 2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
