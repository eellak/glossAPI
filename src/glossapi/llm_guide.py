"""The LLM guide (``glossapi guide --json``; the same content rendered as ``docs/llm_guide.md``).

An LLM reads it after installing glossAPI: what glossAPI can do, the choices the user makes and what each costs, the
workflow, and how to hand over files found elsewhere. Every fact is generated from the module that owns it, never
copied: stages and device kinds from ``glossapi.capabilities``, packages and install sizes from
``glossapi.package_catalog`` (its measured locks), settings from ``glossapi.settings_contract``, manifest rules from
``glossapi.input_manifest``, schema files from ``glossapi.schemas``, typed errors from the command modules, and time
per page from ``data/measured_costs.json``. ``tests/test_k3b_interface.py`` fails when the guide or the committed
``docs/llm_guide.md`` disagrees with any of them.

``python -m glossapi.llm_guide`` prints the Markdown rendering.
"""
from __future__ import annotations

from importlib import resources
import json
from typing import Any, Final, cast

from .capabilities import RUN_UNAVAILABLE, STAGE_DEVICE_KINDS, STAGE_ORDER
from . import input_manifest, plan_preview
from .input_manifest import MANIFEST_VERSION, PRODUCTS, SELECTOR_KEYS
from .install_plan import ERRORS as INSTALL_ERRORS
from .package_catalog import DEVICES, PROPRIETARY_BASIS, catalog_view, requirement_source, stage_enablement, torch_build
from .schemas import NAMES, SCHEMA_VERSION, load_schema
from .settings_contract import PLAN_FIXED_NOTE, SETTINGS_CONTRACT, SETTINGS_DEFAULTS

GUIDE_VERSION: Final[int] = 1
# What each stage does, in one sentence (PLAN rev 5 section 2.3). The tests require one entry per stage.
STAGE_PURPOSES: Final[dict[str, str]] = {
    'download': 'Fetch URL sources (HTTP; browser-assisted with the browser package). glossAPI does not search or scrape.',
    'extract': 'Read the PDF text layer with PDFium and repair broken fonts (bundled Rust repair); Docling layout optional.',
    'clean': 'Clean the extracted Markdown (Rust cleaner) and score its quality; route bad pages to OCR.',
    'ocr': 'OCR the routed pages with the Greek OCR toolkit (the default plugin); output is text/Markdown.',
    'math': 'Optional math/code enrichment with Docling (not executable by the streaming run yet).',
    'sections': 'Split documents into sections.',
    'annotate': 'Classify sections with the packaged neutral classifier (product: annotated sections).',
    'export': 'Write the document export (product: document export).',
}
PRODUCT_PURPOSES: Final[dict[str, str]] = {
    'document_export': 'One record per document with its cleaned text and metadata (the export stage).',
    'annotated_sections': 'Sections with their predicted types (the annotate stage). A separate product from the document export.',
}
WORKFLOW: Final[tuple[tuple[str, tuple[str, ...], str], ...]] = (
    ('capabilities', ('capabilities', '--json'),
     'What is installed and runnable now: every stage, its packages, the visible devices. Read-only.'),
    ('catalog', ('capabilities', '--catalog', '--json'),
     'Every optional package with what it adds, devices, measured install size, licences and install state.'),
    ('install (preview)', ('install', '--device', 'cpu', '--package', 'download', '--json'),
     'Dry run: the exact commands and what they would change. Nothing runs. Show it to the user.'),
    ('install (after the user agrees)', ('install', '--device', 'cpu', '--package', 'download', '--yes', '--json'),
     'Runs exactly the previewed commands, then re-checks capabilities and reports what changed.'),
    ('plan (preview)', ('plan', '--corpus', 'CORPUS', '--manifest', 'MANIFEST.json', '--json'),
     'The exact targets, exclusions, required capabilities, device allocation and deletion policy. Writes nothing.'),
    ('run', ('run', '--corpus', 'CORPUS', '--plan', 'PLAN.json', '--json'),
     'Runs the saved plan; refuses if any input byte, setting or installed capability changed since the preview.'),
    ('status', ('status', '--corpus', 'CORPUS', '--json'), 'Progress and per-stage counts of the latest run.'),
    ('report', ('report', '--corpus', 'CORPUS', '--json'), 'The final census: every requested source exactly once.'),
    ('report --why', ('report', '--corpus', 'CORPUS', '--why', 'DOCUMENT', '--json'),
     'Why a document ended as it did, from the ledger alone (also after intermediates are released).'),
)


def measured_costs() -> dict[str, Any]:
    text = resources.files(__package__ or 'glossapi').joinpath('data', 'measured_costs.json').read_text(encoding='utf-8')
    return cast(dict[str, Any], json.loads(text))


def _cli_errors() -> dict[str, dict[str, Any]]:
    """Typed errors an LLM meets on the decision path, with exit codes, each from the registry of the module that raises
    it: install from ``install_plan.ERRORS``, plan/run from ``plan_preview.ERRORS``, manifests from
    ``input_manifest.ERRORS`` (CP-K3 F4)."""
    rows: dict[str, dict[str, Any]] = {code: dict(command='install', exit_code=exit_code, meaning=meaning)
                                       for code, (exit_code, meaning) in INSTALL_ERRORS.items()}
    rows.update(plan_preview.error_contracts())
    rows['disk_budget_required'] = dict(command='plan, run', exit_code=3, meaning='memory_saving=extreme needs --disk-budget-bytes.')
    from .ledger import urls
    rows[urls.CODE] = dict(command='plan, run', exit_code=2, meaning=urls.MEANING)
    rows.update(input_manifest.error_contracts())
    return rows


def _device_profiles(targets: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for device in DEVICES:
        platforms: list[dict[str, Any]] = []
        for target_id, target in targets.items():
            try:
                build = torch_build(device, target['system'], target['machine'])
            except Exception:
                continue
            platforms.append(dict(target=target_id, torch_index=build['index_url'], torch_requirements=build['requirements']))
        rows.append(dict(device=device, platforms=platforms))
    return rows


def _variants(enablement: dict[str, list[dict[str, Any]]], stage: str) -> list[dict[str, Any]]:
    """The stage's settings variants that need other packages than its defaults, each once."""
    seen: dict[str, dict[str, Any]] = {}
    for rows in enablement.values():
        for row in rows:
            if row['stage'] == stage and row['settings']:
                seen.setdefault(json.dumps(row['settings'], sort_keys=True),
                                dict(settings=row['settings'], requires_profiles=row['requires_profiles']))
    return [seen[key] for key in sorted(seen)]


def guide_view() -> dict[str, Any]:
    """``glossapi guide --json`` data (schema guide.json). Environment-independent: no install state."""
    source = requirement_source()
    catalog = catalog_view(source, installed=False)
    enablement = stage_enablement()
    manifest = load_schema('manifest')
    costs = measured_costs()
    phases = [dict(stage=stage, purpose=STAGE_PURPOSES[stage], device_kinds=list(STAGE_DEVICE_KINDS[stage]),
                   executable_by_run=stage not in RUN_UNAVAILABLE,
                   packages=sorted({group for rows in enablement.values() for row in rows if row['stage'] == stage and not row['settings']
                                    for group in cast(list[str], row['requires_profiles'])}),
                   variants=_variants(enablement, stage))
              for stage in STAGE_ORDER]
    packages = [dict(name=row['name'], kind=row['kind'], purpose=row['purpose'], alias_of=row['alias_of'], includes=row['includes'],
                     device_bound=row['device_bound'], devices=[item['device'] for item in row['devices'] if item['supported']],
                     enables=sorted({item['stage'] for item in row['enables']}),
                     install_size=[dict(target=size['target'], device=size['device'], packages=size['packages'],
                                        added_packages=size['added_packages'], added_download_bytes=size['added_download_bytes'],
                                        added_installed_bytes=size['added_installed_bytes'], unmeasured=size['unmeasured'])
                                   for size in row['install_size']],
                     licences=row['licence']['roots'], proprietary=row['licence']['proprietary'],
                     proprietary_packages=row['licence']['proprietary_packages'], install_command=row['install_command'])
                for row in catalog['profiles']]
    return dict(
        guide_version=GUIDE_VERSION, schema_version=SCHEMA_VERSION,
        summary=('glossAPI turns PDFs (local files or URLs you hand over) into clean text: it extracts the text layer, '
                 'repairs fonts, cleans, routes bad pages to Greek OCR, splits sections, annotates them and exports. It '
                 'is driven through a JSON CLI: every command prints one JSON envelope (schema v1).'),
        phases=phases,
        products=[dict(name=name, stage=stage, purpose=PRODUCT_PURPOSES[name]) for name, stage in PRODUCTS.items()],
        choices=dict(
            files=dict(how='A versioned input manifest (manifest_version %d) of local paths or URLs with their original metadata; '
                           'or --files PATH... for a quick plan.' % MANIFEST_VERSION,
                       selector_keys=list(SELECTOR_KEYS), required=list(cast(list[str], manifest.get('required', []))),
                       selection='`selection` names the selection_ids to process; other entries are never opened, hashed or processed. '
                                 'An empty `selection` selects nothing (run refuses with empty_selection); omit it to process every entry.',
                       metadata='Any `metadata` (and any other entry field) is carried verbatim to status --file and report.',
                       no_scraping='glossAPI does not search or scrape. An LLM that found files elsewhere hands them over in a manifest.'),
            devices=dict(run_device_kinds={stage: list(kinds) for stage, kinds in STAGE_DEVICE_KINDS.items()},
                         run_device_syntax='--device STAGE=cpu|mps|cuda:N (plan/run); only the stages listed with mps/cuda can use them.',
                         install_device_profiles=_device_profiles(cast(dict[str, dict[str, str]], catalog['measurement']['targets'])),
                         install_rule='install --device chooses the torch build for device-bound packages (cpu, cuda126, cuda132, mps); '
                                      'one device profile per environment.'),
            packages=packages,
            settings=dict(values={name: dict(values=dict(values), default=SETTINGS_DEFAULTS[name]) for name, values in SETTINGS_CONTRACT.items()},
                          fixed_stages=PLAN_FIXED_NOTE)),
        costs=dict(time_per_page=costs['time_per_page'], disk=costs['disk'], not_measured=costs['not_measured'],
                   install_size_basis=catalog['measurement']['size_basis'], measured_locks_at=catalog['measurement']['measured_at'],
                   rule=costs['rule']),
        workflow=[dict(step=step, argv=['glossapi', *argv], command=' '.join(['glossapi', *argv]), purpose=purpose)
                  for step, argv, purpose in WORKFLOW],
        handover=dict(example={'manifest_version': MANIFEST_VERSION,
                               'documents': [{'url': 'https://example.org/a.pdf', 'selection_id': 'a', 'metadata': {'title': 'Found by the LLM'}},
                                             {'path': '/absolute/path/b.pdf', 'selection_id': 'b'}],
                               'selection': ['a', 'b'], 'products': ['document_export'], 'settings': {'debug': 'off', 'memory_saving': 'normal'}},
                      rules=['Paths are absolute; URLs are fetched by the download stage only when it is planned.',
                             'Show the user `plan --json` before `run`; run the saved plan with `run --plan`.',
                             'Downloads are public targets only: a URL with userinfo or a credential query field is refused (credentials_unsupported).']),
        errors=_cli_errors(),
        exit_codes={'0': 'success', '1': 'ran and found failures (status/run with failed documents; doctor with an error check); run lists each failed '
                                     'task and unavailable source in errors (code = the ledger reason code)',
                    '2': 'a typed error (errors[0].code)', '3': 'a policy or disk conflict, or a run waiting for disk'},
        schemas=[dict(name=name, file=f'glossapi/schemas/v{SCHEMA_VERSION}/{name}.json', id=load_schema(name)['$id']) for name in NAMES],
        sources=dict(stages='glossapi.capabilities', packages='glossapi.package_catalog (data/package_catalog/measured.json)',
                     settings='glossapi.settings_contract', manifest='glossapi.input_manifest', costs='glossapi/data/measured_costs.json',
                     errors='glossapi.install_plan.ERRORS, glossapi.plan_preview.ERRORS, glossapi.input_manifest.ERRORS'))


def _value(value: Any) -> str:
    """A measured value or a measured range (a two-item list)."""
    return '-'.join(str(item) for item in cast(list[Any], value)) if isinstance(value, list) else str(value)


def _mb(value: Any) -> str:
    return '-' if value is None else '%.1f MB' % (cast(int, value) / 1e6)


def render_markdown(guide: dict[str, Any]) -> str:
    """The Markdown form of the guide (docs/llm_guide.md is this output, byte for byte)."""
    out: list[str] = [f"# glossAPI guide for LLMs (guide v{guide['guide_version']}, CLI schema v{guide['schema_version']})", '',
                      '<!-- Generated by `python -m glossapi.llm_guide` from the same sources as `glossapi guide --json`. '
                      'Do not edit by hand; tests/test_k3b_interface.py fails when it drifts. -->', '', guide['summary'], '',
                      '## What glossAPI can do', '', '| stage | what it does | devices | packages |', '| --- | --- | --- | --- |']
    for phase in guide['phases']:
        out.append(f"| `{phase['stage']}` | {phase['purpose']} | {', '.join(phase['device_kinds'])} | {', '.join(phase['packages'])} |")
    out += ['', 'Products (separate):', '']
    out += [f"- `{item['name']}` ({item['stage']} stage): {item['purpose']}" for item in guide['products']]
    choices = guide['choices']
    out += ['', '## The choices the user makes', '', '### Files', '', choices['files']['how'], '',
            f"- Each entry names exactly one of: {', '.join('`%s`' % key for key in choices['files']['selector_keys'])}.",
            f"- {choices['files']['selection']}", f"- {choices['files']['metadata']}", f"- {choices['files']['no_scraping']}", '',
            '### Devices', '', f"- {choices['devices']['run_device_syntax']}", f"- {choices['devices']['install_rule']}", '',
            '| install device profile | platforms (torch index) |', '| --- | --- |']
    for row in choices['devices']['install_device_profiles']:
        out.append(f"| `{row['device']}` | " + '; '.join(f"{item['target']} ({item['torch_index']})" for item in row['platforms']) + ' |')
    out += ['', '### Packages', '', 'Sizes are measured from the locks (added over core; download / installed).', '',
            '| package | kind | adds stages | devices | size per platform | licences of its own requirements |', '| --- | --- | --- | --- | --- | --- |']
    for row in choices['packages']:
        sizes = '; '.join(f"{size['target']}{'/' + size['device'] if size['device'] else ''}: +{size['added_packages']} pkgs, "
                          f"{_mb(size['added_download_bytes'])} / {_mb(size['added_installed_bytes'])}"
                          + (f" (unmeasured: {', '.join(size['unmeasured'])})" if size['unmeasured'] else '')
                          for size in row['install_size'])
        name = f"`{row['name']}`" + (f" (alias of {row['alias_of']})" if row['alias_of'] else '')
        out.append(f"| {name} | {row['kind']} | {', '.join(row['enables']) or '-'} | {', '.join(row['devices'])} | {sizes} | "
                   f"{', '.join(row['licences']) or '-'} |")
    flagged = [row for row in choices['packages'] if row['proprietary']]
    out += ['', 'Proprietary licences (flagged from the locks; ' + PROPRIETARY_BASIS + '):', '']
    out += [f"- `{row['name']}`: proprietary: {', '.join(row['proprietary_packages'])}" for row in flagged] or ['- none']
    out += ['', 'Purposes:', '']
    out += [f"- `{row['name']}`: {row['purpose']}" for row in choices['packages']]
    out += ['', '### Settings', '']
    for name, entry in choices['settings']['values'].items():
        for value, text in entry['values'].items():
            out.append(f"- `{name}={value}`{' (default)' if value == entry['default'] else ''}: {text}")
    out += ['', choices['settings']['fixed_stages'], '', '## Costs (measured only)', '', '| workload | device | machine | per page | date |',
            '| --- | --- | --- | --- | --- |']
    for row in guide['costs']['time_per_page']:
        stages = ', '.join(f'{stage} {_value(value)}' for stage, value in cast(dict[str, Any], row['stages']).items())
        out.append(f"| {row['workload']} | {row['device']} | {row['machine']} | {stages} ({row['unit']}) | {row['date']} |")
    out += ['', 'Disk:', '']
    out += [f"- {row['what']}: {'-'.join(str(item) for item in row['values_kb'])} KB ({row['detail']}; {row['date']})" for row in guide['costs']['disk']]
    out += ['', 'Not measured: ' + '; '.join(guide['costs']['not_measured']) + '.', '', guide['costs']['rule'], '',
            '## Workflow', '']
    out += [f"{index}. **{row['step']}**: `{row['command']}`. {row['purpose']}" for index, row in enumerate(guide['workflow'], 1)]
    out += ['', '## Handing over files found elsewhere', '', '```json', json.dumps(guide['handover']['example'], ensure_ascii=False, indent=2), '```', '']
    out += [f"- {rule}" for rule in guide['handover']['rules']]
    out += ['', '## Typed errors on the decision path', '', '| code | command | exit | meaning |', '| --- | --- | --- | --- |']
    out += [f"| `{code}` | {row['command']} | {row['exit_code']} | {row['meaning']} |" for code, row in guide['errors'].items()]
    out += ['', 'Exit codes: ' + '; '.join(f"`{code}` {meaning}" for code, meaning in guide['exit_codes'].items()) + '.', '',
            '## Schemas', '']
    out += [f"- `{row['name']}`: `{row['file']}`" for row in guide['schemas']]
    return '\n'.join(out) + '\n'


__all__ = ['GUIDE_VERSION', 'guide_view', 'render_markdown', 'measured_costs', 'STAGE_PURPOSES', 'WORKFLOW']

if __name__ == '__main__':  # pragma: no cover
    print(render_markdown(guide_view()), end='')
