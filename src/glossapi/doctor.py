"""``glossapi doctor``: the capability checks plus environment problems, each with a fix hint.

Reads only: package metadata, module locations, the corpus directory's free space and the ledger header. The Rust
extensions are imported in a child process (a broken extension cannot take this process down), and so is torch when
``--probe-devices`` is given (the result says that torch was imported). Nothing is installed or written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Final, cast

from .capabilities import SUPPORTED_PYTHON, capabilities_view, install_command, installed_version

NATIVE_EXTENSIONS: Final[tuple[str, ...]] = ('glossapi_rs_cleaner', 'glossapi_rs_noise', 'glossapi_rs_pdfrepair')
LOW_DISK_BYTES: Final[int] = 2 * 1024**3
PROBE_TIMEOUT_SECONDS: Final[int] = 120
_PROBE = r'''
import json, sys
result = {"framework": None, "imported": []}
try:
    import torch
except Exception as error:
    result["error"] = type(error).__name__ + ": " + str(error)
else:
    result["framework"] = "torch " + torch.__version__
    result["imported"] = ["torch"]
    mps = getattr(torch.backends, "mps", None)
    result["mps"] = {"available": bool(mps and mps.is_available())}
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    result["cuda_count"] = count
    for index in range(max(count, 1)):
        result["cuda:%d" % index] = {"available": index < count}
print(json.dumps(result))
'''


def _check(check_id: str, status: str, message: str, fix_hint: str | None = None, **details: Any) -> dict[str, Any]:
    return dict(check_id=check_id, status=status, message=message, fix_hint=fix_hint, details=details)


def _child(code: str, timeout: int) -> tuple[int | None, str, str]:
    try:
        done = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    except subprocess.TimeoutExpired:
        return None, '', 'timed out after %d s' % timeout
    return done.returncode, done.stdout, done.stderr


def _child_path_code(body: str) -> str:
    # -I drops PYTHONPATH; restore this process's import path so the child sees the same modules.
    return 'import sys; sys.path[:0] = %r\n%s' % ([entry for entry in sys.path if entry], body)


def probe_devices() -> dict[str, Any]:
    """Framework device probe in a child process (imports torch there, never here)."""
    code, out, err = _child(_child_path_code(_PROBE), PROBE_TIMEOUT_SECONDS)
    try:
        result: dict[str, Any] = cast(dict[str, Any], json.loads(out.strip().splitlines()[-1])) if code == 0 and out.strip() else {}
    except ValueError:
        result = {}
    if not result:
        result = {'framework': None, 'imported': [], 'error': (err.strip().splitlines() or ['probe failed'])[-1]}
    result['process'] = 'child'
    return result


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def doctor_view(corpus: str | os.PathLike[str] = '.', *, probe: bool = False) -> dict[str, Any]:
    probe_result = probe_devices() if probe else None
    capabilities = capabilities_view(probe=probe_result)
    checks: list[dict[str, Any]] = []
    low, high = SUPPORTED_PYTHON
    version = sys.version_info[:2]
    checks.append(_check('python_version', 'ok' if low <= version < high else 'error',
                         'Python %d.%d.%d; glossAPI supports >=%d.%d,<%d.%d' % (*sys.version_info[:3], *low, *high),
                         None if low <= version < high else 'Use a Python between %d.%d and %d.%d.' % (*low, high[0], high[1] - 1)))
    for module in NATIVE_EXTENSIONS:
        code, _, err = _child(_child_path_code('import %s' % module), 60)
        ok = code == 0
        message = (err.strip().splitlines() or ['import failed'])[-1]
        checks.append(_check('native_extension:' + module, 'ok' if ok else 'error',
                             module + (' imports' if ok else ' does not import: ' + message),
                             None if ok else 'Reinstall the core package so the Rust extensions are built: ' + install_command('core'),
                             module=module, imported_in='child_process'))
    standalone = [name for name in NATIVE_EXTENSIONS if installed_version(name) is not None]
    checks.append(_check('native_extension_origin', 'warning' if standalone else 'ok',
                         ('Standalone distributions take import precedence over the root wheel: ' + ', '.join(standalone)) if standalone
                         else 'No standalone native distributions shadow the root wheel',
                         ('pip uninstall ' + ' '.join(standalone)) if standalone else None, standalone=standalone))
    for row in capabilities['stages']:
        stage = row['stage']
        if row['available']:
            status, message, hint = 'ok', stage + ' is available', None
        else:
            core = row['missing_install_profiles'] == ['core'] or stage in ('extract', 'clean', 'sections', 'export')
            status = 'error' if core else 'info'
            blocking = [item['name'] + ' (' + item['status'] + ')' for item in row['packages'] if item['blocking']]
            blocking += [item['module'] + ' (missing)' for item in row['native_modules'] if item['required'] and item['status'] == 'missing']
            blocking += [item['group'] + ':' + str(item['name']) + ' (missing entry point)' for item in row['entry_points'] if item['status'] == 'missing']
            message = stage + ' is not available: ' + ', '.join(dict.fromkeys(blocking))
            hint = '; '.join(row['install_commands']) or None
        if not row['executable_by_run']:
            message += ' (glossapi run cannot execute it: ' + str(row['unavailable_reason']) + ')'
        checks.append(_check('stage:' + stage, status, message, hint, reason_code=row['reason_code']))
    root = Path(corpus).absolute()
    target = _existing_parent(root)
    usage = shutil.disk_usage(target)
    checks.append(_check('disk_free', 'warning' if usage.free < LOW_DISK_BYTES else 'ok',
                         '%.1f GB free at %s' % (usage.free / 1e9, target),
                         'Free disk space or choose another --corpus; set disk_budget_bytes for memory_saving=extreme.'
                         if usage.free < LOW_DISK_BYTES else None, path=str(target), free_bytes=usage.free, total_bytes=usage.total))
    writable = os.access(target, os.W_OK)
    checks.append(_check('corpus_writable', 'ok' if writable else 'warning',
                         str(target) + (' is writable' if writable else ' is not writable: plan works, run cannot'),
                         None if writable else 'Choose a writable --corpus for run.', path=str(target)))
    corpus_id: str | None = None
    database = root / 'corpus.sqlite3'
    if database.is_file():
        from .ledger import writer  # untyped module: read through Any
        try:
            with cast(Any, writer).open_readonly(database) as connection:
                corpus_id = cast(str | None, connection.execute('SELECT corpus_id FROM ledger_info').fetchone()[0])
            checks.append(_check('ledger', 'ok', 'Ledger readable', None, corpus_id=corpus_id))
        except Exception as error:  # any failure to read the ledger is reported, never raised
            checks.append(_check('ledger', 'error', 'Ledger unreadable: ' + str(error),
                                 'Restore the ledger or use a new --corpus.', error_type=type(error).__name__))
    else:
        checks.append(_check('ledger', 'ok', 'No ledger yet; run creates one', None))
    for row in capabilities['devices']:
        if row['kind'] == 'cpu' or row['hardware'] == 'absent':
            continue
        checks.append(_check('device:' + row['device_id'], 'ok' if row['state'] == 'available' else 'info',
                             row['device_id'] + ' ' + row['state'] + (': ' + row['reason'] if row['reason'] else ''),
                             'Run glossapi doctor --probe-devices to confirm it.' if row['state'] == 'unknown_until_probe' else
                             (install_command('ocr', row['device_id']) if row['state'] == 'unavailable' else None)))
    counts = {status: sum(check['status'] == status for check in checks) for status in ('ok', 'info', 'warning', 'error')}
    return dict(healthy=counts['error'] == 0, counts=counts, checks=checks, capabilities=capabilities,
                device_probe=dict(requested=probe, imported=(probe_result or {}).get('imported', []),
                                  process=(probe_result or {}).get('process'), framework=(probe_result or {}).get('framework'),
                                  error=(probe_result or {}).get('error')),
                corpus=dict(path=str(root), corpus_id=corpus_id))
