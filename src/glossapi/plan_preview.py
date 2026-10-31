"""Exact ``plan --json`` preview fields and the plan-identity check ``run`` applies before anything executes.

``preview_fields`` adds to the plan body (and so to its hash): the exact targets, the exclusions with a reason each,
the required capabilities and missing installs, the device allocation per stage, the deletion policy that follows from
``debug``/``memory_saving``, and every conflict found. ``verify_plan_binding`` refuses a supplied plan whose identity no
longer holds: an edited body (hash), a changed input (size, mtime or bytes), a changed installed capability, or a
blocking conflict. Neither function imports a backend or writes anything.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import unquote, urlsplit

from .capabilities import (capability_fingerprint, device_kind, device_state, devices, install_command,
                           stage_capability)
from .settings_contract import SETTINGS_CONTRACT

PREVIEW_VERSION: Final[int] = 1
# Direct readers of each stage's outputs in the streaming DAG (ledger/reader.plan_view prerequisites; the OCR text
# is read by the post-OCR clean and by export).
READERS: Final[dict[str, tuple[str, ...]]] = {
    'download': ('extract',), 'extract': ('clean', 'math', 'ocr'), 'clean': ('ocr', 'sections', 'export'),
    'ocr': ('clean', 'export'), 'math': (), 'sections': ('annotate',), 'annotate': (), 'export': ()}
# Declared output kinds per stage (stages/run.py streaming operations and the compatibility barriers).
OUTPUT_KINDS: Final[dict[str, tuple[str, ...]]] = {
    'download': ('download_payload',), 'extract': ('raw_markdown', 'page_quality'), 'clean': ('clean_markdown', 'clean_metrics'),
    'ocr': ('ocr_markdown', 'ocr_metrics'), 'math': (), 'sections': ('section_fragment', 'sections_parquet'),
    'annotate': ('classified_parquet',), 'export': ('export_jsonl',)}
PRODUCTS: Final[dict[str, str]] = {'export': 'document_export', 'annotate': 'annotated_sections', 'sections': 'sections_parquet'}
# Conflicts that make ``plan`` itself fail (request errors), and those that only block execution (environment, or an
# explicitly empty selection).
PLAN_ERRORS: Final[frozenset[str]] = frozenset(('device_unsupported_for_stage', 'device_option_conflict', 'device_unavailable'))
RUN_BLOCKERS: Final[frozenset[str]] = frozenset(('empty_selection', 'missing_capability'))
_DEVICE_MEANING: Final[str] = 'A device choice the stage or machine cannot honour; plan keeps its preview in data.'
_REPLAN_MEANING: Final[str] = 'The saved plan no longer matches: re-run plan and show the new preview.'
# THE registry of the typed errors this module raises on the plan/run decision path: code -> (commands, exit code,
# meaning). The LLM guide renders it (llm_guide._cli_errors); tests/test_k3b_interface.py reads the codes the
# functions below actually construct and fails when one is missing here, or listed here and never raised.
ERRORS: Final[dict[str, tuple[str, int, str]]] = {
    'device_unsupported_for_stage': ('plan, run', 2, _DEVICE_MEANING),
    'device_option_conflict': ('plan, run', 2, _DEVICE_MEANING),
    'device_unavailable': ('plan, run', 2, _DEVICE_MEANING),
    'missing_capability': ('run', 2, 'plan exits 0 with execution.ready false and the install commands; run refuses before writing.'),
    'empty_selection': ('run', 2, 'The manifest selects no entry (selection: [] or an empty document list): plan exits 0 with zero '
                        'targets, every entry in exclusions and execution.ready false; run refuses before writing. Omit '
                        'selection to process every entry.'),
    'invalid_plan': ('run --plan', 2, 'The file is not plan --json output (or its data object).'),
    'plan_hash_mismatch': ('run --plan', 2, _REPLAN_MEANING),
    'plan_source_changed': ('run --plan', 2, _REPLAN_MEANING),
    'plan_capability_changed': ('run --plan', 2, _REPLAN_MEANING),
}


def error_contracts() -> dict[str, dict[str, Any]]:
    """The guide rows of ``ERRORS`` (command, exit code, meaning), in registry order."""
    return {code: dict(command=command, exit_code=exit_code, meaning=meaning) for code, (command, exit_code, meaning) in ERRORS.items()}


class PlanBindingError(ValueError):
    """Typed refusal; ``code`` names the broken part of the plan identity."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, Any] = details


def _local_path(uri: str) -> Path | None:
    return Path(unquote(urlsplit(uri).path)) if uri.startswith('file:') else None


def _conflict(code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    return dict(code=code, severity=severity, message=message, **details)


def stage_settings(settings: dict[str, Any], stage: str) -> dict[str, Any]:
    value = settings.get(stage)
    return dict(cast(dict[str, Any], value)) if isinstance(value, dict) else {}


def required_capabilities(stages: list[str], settings: dict[str, Any], pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    device_of = {cast(str, pool['stage']): cast(str, pool['device_id']) for pool in pools}
    rows: list[dict[str, Any]] = []
    toolkit: dict[str, Any] | None = None
    for stage in stages:
        row = stage_capability(stage, stage_settings(settings, stage), device_of.get(stage, 'cpu'), toolkit=toolkit)
        toolkit = cast(dict[str, Any] | None, row['toolkit']) if row['toolkit'] is not None else toolkit
        rows.append(row)
    return rows


def _deletion_policy(documents: list[dict[str, Any]], stages: list[str], settings: dict[str, Any]) -> dict[str, Any]:
    debug, saving = cast(str, settings['debug']), cast(str, settings['memory_saving'])
    readers = {stage: [reader for reader in READERS[stage] if reader in stages] for stage in stages}

    def kinds(stage: str, planned: list[str]) -> list[str]:
        # page_quality is produced only for OCR routing.
        return [kind for kind in OUTPUT_KINDS[stage] if kind != 'page_quality' or 'ocr' in planned]

    final = [dict(stage=stage, product=PRODUCTS.get(stage, stage + '_outputs'), kinds=kinds(stage, stages))
             for stage in stages if not readers[stage] and stage != 'download' and OUTPUT_KINDS[stage]]
    releases = [dict(producer_stage=stage, kinds=kinds(stage, stages), after_readers=readers[stage],
                     when='after every planned reader (' + ', '.join(readers[stage]) + ') succeeded on its exact hash',
                     applies=debug == 'off')
                for stage in stages if stage != 'download' and readers[stage] and OUTPUT_KINDS[stage]]
    download_readers = readers.get('download', [])
    deletions = [dict(file_class='run_owned_download', kinds=['download_payload'], after_readers=download_readers,
                      when='after the last planned reader (' + ', '.join(download_readers) + ') succeeded and its outputs are durable',
                      applies=saving == 'extreme')] if 'download' in stages else []
    per_document: list[dict[str, Any]] = []
    for document in documents:
        steps = [cast(str, step['stage']) for step in cast(list[dict[str, Any]], document['steps'])]
        source_uri = cast(str, cast(dict[str, Any], document['source_observation'])['source_uri'])
        downloaded = 'download' in steps
        source_action = ('deleted_after_last_reader' if downloaded and saving == 'extreme' else 'kept')
        released = [dict(producer_stage=stage, kinds=kinds(stage, steps),
                         after_readers=[reader for reader in READERS[stage] if reader in steps])
                    for stage in steps if debug == 'off' and stage != 'download' and OUTPUT_KINDS[stage]
                    and any(reader in steps for reader in READERS[stage])]
        per_document.append(dict(document_id=document['document_id'], source_uri=source_uri,
                                 source=dict(kind='download' if downloaded else 'supplied', action=source_action,
                                             reason='memory_saving=extreme deletes run-owned downloads' if source_action != 'kept' else
                                             'supplied originals are never deleted' if not downloaded else 'memory_saving=normal keeps downloads'),
                                 released=released))
    return dict(settings=dict(debug=debug, memory_saving=saving),
                contract=dict(debug=SETTINGS_CONTRACT['debug'][debug], memory_saving=SETTINGS_CONTRACT['memory_saving'][saving]),
                never_deleted=['supplied_originals', 'selected_final_products', 'evidence', 'ledger'],
                final_products=final, releases=releases, deletions=deletions, per_document=per_document,
                release_scope='private stage copies under .glossapi/artifacts',
                compatibility_publications=('copies published into markdown/, clean_markdown/, sidecars/ and download_results/ follow the '
                                            'retention declared when they are published; a copy without a declaration is kept '
                                            '(report --why lists each file with its pins)'),
                basis='declared stage output kinds; exact file paths are assigned when the run registers them')


def preview_fields(*, documents: list[dict[str, Any]], stages: list[str], requested: list[str], settings: dict[str, Any],
                   pools: list[dict[str, Any]], explicit_devices: dict[str, str], exclusions: list[dict[str, Any]],
                   input_manifest: dict[str, Any] | None, selection_empty: bool = False) -> dict[str, Any]:
    """``selection_empty``: the caller selected sources explicitly and selected none (CP-K3 F1). The preview then has no
    target and a blocking ``empty_selection`` conflict; it never falls back to every corpus document."""
    visible = devices()
    capabilities = required_capabilities(stages, settings, pools)
    conflicts: list[dict[str, Any]] = []
    if selection_empty:
        conflicts.append(_conflict('empty_selection', 'blocking', 'The selection is explicitly empty: nothing is selected and '
                                   'nothing runs (omit selection to process every entry)', excluded=len(exclusions)))
    allocation: list[dict[str, Any]] = []
    ocr_settings = stage_settings(settings, 'ocr')
    option_device = cast(dict[str, Any], ocr_settings.get('options') or {}).get('device')
    for pool in pools:
        stage, device = cast(str, pool['stage']), cast(str, pool['device_id'])
        explicit = stage in explicit_devices
        kinds = ('cpu', 'mps', 'cuda') if stage == 'ocr' and ocr_settings.get('entry_point_name', 'glossapi') == 'glossapi' else ('cpu',)
        state = device_state(device, visible)
        row = dict(stage=stage, device_id=device, explicit=explicit, supported_device_kinds=list(kinds),
                   state=state['state'] if state else 'unavailable', reason=state['reason'] if state else 'unknown device',
                   workers=pool['workers'], cpu_tokens=pool['cpu_tokens'])
        for key in ('detector_device', 'accelerator_devices'):
            if key in pool:
                row[key] = pool[key]
        allocation.append(row)
        if device_kind(device) not in kinds:
            conflicts.append(_conflict('device_unsupported_for_stage', 'error', f'{stage} cannot run on {device}; it supports ' + ', '.join(kinds),
                                       stage=stage, device_id=device))
            continue
        if stage == 'ocr' and option_device and explicit and option_device != device:
            conflicts.append(_conflict('device_option_conflict', 'error', f'ocr device {device} disagrees with ocr.options.device {option_device}',
                                       stage=stage, device_id=device, option_device=option_device))
        if row['state'] == 'unavailable':
            conflicts.append(_conflict('device_unavailable' if explicit else 'default_device_unavailable', 'error' if explicit else 'warning',
                                       f'{stage} device {device} is not usable here: {row["reason"]}', stage=stage, device_id=device))
        elif row['state'] == 'unknown_until_probe':
            conflicts.append(_conflict('device_unverified', 'warning', f'{stage} device {device} needs a framework probe '
                                       '(glossapi doctor --probe-devices); run checks it before dispatch', stage=stage, device_id=device))
    missing: list[dict[str, Any]] = []
    for row in capabilities:
        if row['available']:
            continue
        missing.append(dict(stage=row['stage'], reason_code=row['reason_code'], install_profiles=row['missing_install_profiles'],
                            install_commands=row['install_commands']))
        conflicts.append(_conflict('missing_capability', 'blocking', row['stage'] + ' is not installed here: ' +
                                   '; '.join(cast(list[str], row['install_commands'])), stage=row['stage'],
                                   reason_code=row['reason_code'], install_commands=row['install_commands']))
    targets: list[dict[str, Any]] = []
    for document in documents:
        observation = cast(dict[str, Any], document['source_observation'])
        uri = cast(str, observation['source_uri'])
        path = _local_path(uri)
        steps = [cast(str, step['stage']) for step in cast(list[dict[str, Any]], document['steps'])]
        targets.append(dict(document_id=document['document_id'], source_key=document['source_key'], source_uri=uri,
                            path=str(path) if path else None,
                            state='remote' if path is None else 'absent' if observation.get('bytes') is None else
                            'unreadable' if observation.get('read_error') else 'present',
                            bytes=observation.get('bytes'), sha256=observation.get('sha256') or observation.get('stored_sha256'),
                            stages=steps, download=dict(source_uri=uri) if 'download' in steps else None,
                            known_version_id=document.get('version_id')))
    install = sorted({command for row in missing for command in cast(list[str], row['install_commands'])})
    return dict(preview_version=PREVIEW_VERSION, requested_stages=list(requested), targets=targets, exclusions=exclusions,
                input_manifest=input_manifest,
                capabilities=dict(stages=capabilities, missing=missing, install_commands=install,
                                  fingerprint=capability_fingerprint(capabilities)),
                device_allocation=allocation, devices=visible,
                deletion_policy=_deletion_policy(documents, stages, settings), conflicts=conflicts)


def plan_errors(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in cast(list[dict[str, Any]], body.get('conflicts', [])) if item['code'] in PLAN_ERRORS and item['severity'] == 'error']


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_plan_binding(plan: dict[str, Any]) -> dict[str, Any]:
    """Refuse a plan whose identity changed since it was made; returns the plan body when it still holds."""
    from .ledger import encoding  # untyped module: read through Any
    snapshot_hash = cast(Any, encoding).snapshot_hash
    body = cast(dict[str, Any], plan.get('data', plan)) if isinstance(plan.get('data'), dict) else plan
    if not isinstance(body.get('plan_hash'), str) or not isinstance(body.get('documents'), list):
        raise PlanBindingError('invalid_plan', 'Not a glossapi plan (expected plan --json output or its data object)')
    if snapshot_hash({key: value for key, value in body.items() if key != 'plan_hash'}) != body['plan_hash']:
        raise PlanBindingError('plan_hash_mismatch', 'Supplied plan hash does not match: the plan was edited after planning '
                               '(settings, targets or stages); plan again', plan_hash=body['plan_hash'])
    for document in cast(list[dict[str, Any]], body['documents']):
        observation = cast(dict[str, Any], document.get('source_observation') or {})
        path = _local_path(cast(str, observation.get('source_uri') or ''))
        if path is None:
            continue
        planned_absent = observation.get('bytes') is None and not observation.get('stored_sha256')
        details = dict(document_id=document.get('document_id'), path=str(path))
        if planned_absent:
            if path.exists():
                raise PlanBindingError('plan_source_changed', 'A source that was absent at planning now exists; plan again',
                                       field='presence', **details)
            continue
        if not path.is_file():
            raise PlanBindingError('plan_source_changed', 'A planned source is missing or not a file; plan again', field='presence', **details)
        stat = path.stat()
        if observation.get('bytes') is not None and (stat.st_size, stat.st_mtime_ns) != (observation['bytes'], observation.get('mtime_ns')):
            raise PlanBindingError('plan_source_changed', 'A planned source changed size or modification time; plan again',
                                   field='bytes' if stat.st_size != observation['bytes'] else 'mtime_ns', **details)
        expected = observation.get('sha256') or observation.get('stored_sha256')
        if expected and _sha256(path) != expected:
            raise PlanBindingError('plan_source_changed', 'A planned source changed content; plan again', field='sha256', **details)
    recorded = cast(dict[str, Any] | None, body.get('capabilities'))
    if recorded is not None:
        stages = [cast(str, row['stage']) for row in cast(list[dict[str, Any]], recorded['stages'])]
        current = required_capabilities(stages, cast(dict[str, Any], body.get('settings') or {}), cast(list[dict[str, Any]], body.get('pools') or []))
        if capability_fingerprint(current) != recorded['fingerprint']:
            before = {cast(str, row['stage']): row for row in cast(list[dict[str, Any]], recorded['stages'])}
            changed = [cast(str, row['stage']) for row in current if capability_fingerprint([row]) != capability_fingerprint([before[cast(str, row['stage'])]])]
            raise PlanBindingError('plan_capability_changed', 'Installed capabilities changed since planning (' + ', '.join(changed) +
                                   '); plan again', stages=changed, planned_fingerprint=recorded['fingerprint'],
                                   current_fingerprint=capability_fingerprint(current))
    refuse_conflicts(body)
    return body


def refuse_conflicts(body: dict[str, Any]) -> None:
    """Typed refusal for a plan that cannot run here (request errors first, then missing installs)."""
    errors = plan_errors(body)
    if errors:
        first = errors[0]
        raise PlanBindingError(cast(str, first['code']), cast(str, first['message']), conflicts=errors)
    blocking = [item for item in cast(list[dict[str, Any]], body.get('conflicts', [])) if item['code'] in RUN_BLOCKERS]
    empty = [item for item in blocking if item['code'] == 'empty_selection']
    if empty:
        raise PlanBindingError('empty_selection', 'The plan selects no document (an explicitly empty selection); nothing runs',
                               conflicts=empty)
    blocking = [item for item in blocking if item['code'] == 'missing_capability']
    if blocking:
        commands = sorted({command for item in blocking for command in cast(list[str], item.get('install_commands', []))})
        raise PlanBindingError('missing_capability', 'Missing capabilities: ' + ', '.join(cast(str, item['stage']) for item in blocking) +
                               '. Install: ' + '; '.join(commands), conflicts=blocking, install_commands=commands)


__all__ = ['ERRORS', 'error_contracts', 'PlanBindingError', 'preview_fields', 'verify_plan_binding', 'refuse_conflicts', 'plan_errors', 'install_command']
