# CLI JSON schema v1 and the input manifest v1

Every `glossapi` command (`plan`, `run`, `status`, `report`, `capabilities`, `doctor`, `install`, `guide`) prints
exactly one JSON envelope. The schemas (JSON Schema draft 2020-12) ship with the package in `glossapi/schemas/v1/`:

| file | describes |
| --- | --- |
| `envelope.json` | the envelope: `schema_version` (1), `command`, `corpus_id`, `run_id`, `data`, `warnings`, `errors` |
| `error.json` | one typed error: `code`, `message`, `document_id`, `stage`, `details` |
| `common.json` | shared objects: capability, device, package status, conflict, warning, task counts |
| `plan.json` | `plan --json` data, including the exact preview (below) |
| `run.json` | `run --json` data |
| `status.json` | `status --json` data (`sources` with `--file`) |
| `report.json` | `report --json` data (`sources` = each document's original metadata, verbatim) |
| `report_why.json` | `report --why SELECTOR` data |
| `report_export.json` | the receipt of `report --output` |
| `capabilities.json` | `capabilities --json` data |
| `doctor.json` | `doctor --json` data |
| `manifest.json` | the input manifest v1 (`--manifest`) |
| `install.json` | `install --json` data (dry run or executed) |
| `guide.json` | `guide --json` data (the LLM guide) |

The runtime never validates against the schemas (`jsonschema` is a test-group dependency only); the tests validate
real outputs: the CLI test corpora (`pytest -p cli_schema_validation ...` validates every envelope those tests print),
a real extract → clean → export run, `capabilities`, `doctor` and every typed error.

## Compatibility rule

- Within v1, only **additive optional fields** are allowed: a new key that consumers may ignore. Consumers must
  ignore keys they do not know.
- Anything else is **v2**: removing or renaming a field, making a field required, changing a type or an enum's
  meaning, or changing an exit code's meaning. v2 gets a new `schemas/v2/` directory and `schema_version: 2`.
- Every change to a v1 file needs a changelog entry: append it to `schemas/v1/SCHEMA_LOCK.json` (entry id, date, the
  change, and the new SHA-256 of every file) and list the same entry id below. `tests/test_k3_interface.py` fails when
  a schema file's bytes differ from the latest lock entry.

## Exit codes

`0` success; `1` the command ran and found failures (`status`/`run` with failed documents, `doctor` with an
`error` check); `2` a typed error (`errors[0].code`); `3` a policy or disk conflict (`disk_budget_required`,
`disk_budget_exceeded`, `policy_conflict`, `waiting_for_disk`, `disk_guard_unavailable`) or a run waiting for disk.

## `plan --json`: the exact preview

`plan` executes nothing and writes nothing (it reads the ledger read-only and hashes the selected new local sources).
Its data adds to the hashed plan body:

- `targets`: each document to process, with path, state (`present`/`absent`/`unreadable`/`remote`), bytes, SHA-256,
  stages, and the download request for a remote source;
- `exclusions`: each requested entry that will not be processed, with `reason_code` (`not_selected` for an
  unselected manifest entry, `duplicate_source` for a repeated source);
- `capabilities`: one capability object per planned stage, the missing installs and their install commands, and the
  `fingerprint` of the installed capabilities the plan depends on;
- `device_allocation`: the device, workers and CPU tokens per stage, whether the device was chosen explicitly, and the
  device's state (`available`, `unknown_until_probe`, `unavailable`);
- `deletion_policy`: what `debug`/`memory_saving` keep, release or delete, after which planned readers, per document;
- `conflicts`: every conflict with its severity. `error` conflicts fail `plan` itself (exit 2, the preview stays in
  `data`): `device_unsupported_for_stage`, `device_unavailable` (an explicitly chosen device this machine does not
  have), `device_option_conflict`. `blocking` conflicts (`missing_capability`) keep `plan` at exit 0 with
  `execution.ready: false` and a warning, and make `run` refuse. `memory_saving=extreme` without a disk budget fails
  with `disk_budget_required` (exit 3).

`run --plan FILE` is bound to the plan identity. Before the ledger is touched it refuses with a typed error when
the plan body was edited (`plan_hash_mismatch`, for example a changed setting), a planned input changed size,
modification time or bytes, or appeared/disappeared (`plan_source_changed`), the installed capabilities changed
(`plan_capability_changed`), or a conflict blocks it (`missing_capability`, device codes).

## `capabilities` and `doctor`

`capabilities` lists every stage with the packages it needs (present, missing, version mismatch, not admitted),
the required native modules, the visible devices and the install command for each missing capability. Detection
uses `importlib.metadata` and `importlib.util.find_spec` only: no torch, docling, pandas or OCR backend is imported.
CUDA and MPS are `unknown_until_probe` unless the platform rules them out. A stage is available under the scheduler's
own rule (the OCR stage checks versions, the toolkit source and its entry point).

`doctor` adds environment checks, each with a fix hint: Python version, the Rust extensions (imported in a child
process), standalone native distributions that shadow the root wheel, free disk and writability at the corpus, and the
ledger. `doctor --probe-devices` imports torch in a child process to confirm CUDA/MPS and says so in `device_probe`
and a warning.

## `capabilities --catalog`: the package list

`--catalog` adds `data.catalog` (`common.json#/$defs/catalog`) to the capabilities data. It lists every install profile
(core and every pyproject extra): its purpose, the profiles it includes, its exact requirement lines, the stage
variants it enables (derived from each stage adapter's `packages()`), whether it is device-bound (needs a torch build),
the device profiles it supports per measured platform, the licences of its own requirements and of everything it adds,
its measured install size, and whether it is installed (`installed.state`: installed / partial / missing, per
requirement).

Why a flag on `capabilities` and not a separate `packages` command: the catalog is the same capability surface (the
same requirement detection and the same envelope command), so an LLM learns one read-only command for "what is here and
what could be added", the envelope dispatch stays unchanged, and the plan's `install_commands` point at the same names.

Sizes are measured, not estimated: `tools/catalog/measure_profiles.py` compiles one PEP 751 lock per profile, device
profile and platform (`uv pip compile`, CPython 3.12, linux-x86_64 manylinux_2_28 and macos-arm64 14.0) from exactly the
requirements `install` would request, and records per lock the package count, the download bytes (the lock's wheel
sizes) and the installed bytes (uncompressed sizes from each wheel's ZIP central directory, read with range requests),
plus the packages added over the core lock. Packages without a wheel (the Greek OCR toolkit is a VCS source) are listed
in `unmeasured`. Each lock records the hash of its inputs; when the requirements change, `measurement.state` becomes
`stale`, `capabilities --catalog` warns `catalog_measurement_stale`, and the tests fail until the tool is re-run.

## `install --device DEV [--package NAME ...] [--yes]`

- The default is a **dry run**: `data.commands` holds the exact commands (installer, index URLs, requirement lines),
  `data.requirements` the installed state of each line, and `data.would_change` what would change (the missing or
  mismatched requirements, and an estimate from the measured locks of the packages added and their bytes). Nothing runs.
- `--yes` runs exactly `data.commands` in order and stops at the first failure, then re-runs the capability check in a
  fresh interpreter: `data.executed` (each step's exit code, time and output tail, and the installer's dependency check)
  and `data.changes` (distributions added/removed/changed, stages whose availability changed, capability fingerprints).
- Only what was requested: the commands carry the requested profiles' own requirement lines, with `glossapi[...]` self
  references expanded (browser includes download), and nothing else. glossapi itself is never reinstalled (the source
  installer `install_glossapi.py` does that). A device-bound profile (layout, ocr) first installs the device's
  torch/torchvision build from its index; the second command pins the same build.
- One device profile per environment: `cpu`, `cuda126`, `cuda132` (Linux) or `mps` (Apple Silicon). Without
  `--package` nothing is installed (warning `nothing_selected`).
- `--installer auto|pip|uv`: auto uses pip when this interpreter has it, else `uv pip install --python`.

## `guide --json`

The versioned LLM guide (`guide_version` 1): the phases and the two products, the choices (files, devices, packages,
`debug`, `memory_saving`) with their measured costs, the workflow (capabilities, choose, plan, run, status/report,
`report --why`), how to hand over files found elsewhere (input manifest v1; glossAPI does not scrape), the typed errors
and the schema files. `docs/llm_guide.md` is the same content rendered by `python -m glossapi.llm_guide`. Every fact
comes from the module that owns it; `tests/test_k3b_interface.py` fails when the guide, the Markdown file, the schemas,
the settings contract or the catalog disagree.

## Typed errors and exit codes of `install`, `guide` and `capabilities --catalog`

`guide` and `capabilities --catalog` are read-only and exit 0; their only errors are the generic ones (`invalid_request`
for a bad argument, `read_error`). `install` checks the whole request before anything runs; every refusal below leaves
the environment untouched, except `install_command_failed`, which reports what ran.

| code | exit | when |
| --- | --- | --- |
| `device_required` | 2 | no `--device` |
| `conflicting_device_profiles` | 2 | two device profiles at once (`--device cpu --device cuda126`, `cpu,mps`) |
| `unknown_device` | 2 | not one of cpu, cuda126, cuda132, mps |
| `device_unsupported_on_platform` | 2 | mps off Apple Silicon macOS, cuda126/cuda132 off Linux |
| `unknown_package` | 2 | a package not in the catalog (`details.known` lists them) |
| `package_device_unsupported` | 2 | the package's lock for that device profile did not resolve, or no measured platform has it |
| `installed_device_profile_conflict` | 2 | the environment holds a torch build of another device profile |
| `requirements_unavailable` | 2 | glossapi is neither a source checkout nor an installed distribution |
| `installer_unavailable` | 2 | `--yes` without pip in this interpreter or uv on PATH (the dry run warns instead) |
| `install_command_failed` | 2 | a command exited non-zero; `data` keeps the steps, output tails and the changes so far |
| `invalid_request` | 2 | a malformed argument (for example `--installer conda`) |

Warnings: `nothing_selected`, `installer_unavailable` (dry run), `dependency_check_failed` (after `--yes`),
`catalog_measurement_stale` (`capabilities --catalog`).

## Input manifest v1

```json
{"manifest_version": 1,
 "documents": [{"path": "/abs/a.pdf", "selection_id": "a", "metadata": {"title": "…", "any": ["JSON"]}},
               {"path": "/abs/b.pdf", "selection_id": "b"}],
 "selection": ["a"],
 "stages": ["extract", "clean"], "products": ["document_export"],
 "devices": {"ocr": "mps"}, "workers": {"clean": 2},
 "settings": {"debug": "off", "memory_saving": "normal"}}
```

- Each entry names exactly one source: `path` (absolute), `url` (unchanged URL behaviour), or a ledger reference
  (`document_id`, `source_key`, `source_uri`). `metadata` and any other entry fields are the caller's original
  metadata; `status --file` and `report` return them verbatim (`sources[].metadata.manifest_row`).
- `selection` (optional) names the `selection_id`s to process. Other entries appear in `exclusions` and are never
  opened, hashed or processed.
- `products`: `document_export` adds `export`; `annotated_sections` adds `annotate`.
- Settings precedence: manifest `settings` < `--settings FILE` < individual flags. Stages, devices and workers given
  both in the manifest and as arguments must agree (`conflicting_request`).
- A manifest without `manifest_version` (a list or `{"documents": [...]}`) keeps the pre-v1 behaviour. Other versions
  fail with `unsupported_manifest_version`; malformed entries with `invalid_manifest` and `details.field`/`index`.
- Python: `glossapi.input_manifest.build_manifest`, `load_manifest`, `write_manifest`; `glossapi.cli.plan_documents`
  returns the same plan data as `glossapi plan --json`.

## Changelog

- `1.0` (2026-10-31): initial freeze: envelope, error, plan, run, status, report, report --why, report --output,
  capabilities, doctor, input manifest v1.
- `1.1` (2026-10-31): additive (K3b): new commands `install` and `guide` (`install.json`, `guide.json`, envelope
  dispatch for them); `capabilities.catalog` (with `--catalog`) and the common definitions `catalog`, `package_profile`,
  `lock_size`, `requirement_state`, `device_profile`. No existing field changed.
