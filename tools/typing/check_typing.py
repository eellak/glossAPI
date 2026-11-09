"""W19 typing gate: strict modules have no pyright error; touched legacy files have no error beyond the staged baseline
(baseline_k3a.json, plus baseline_k4.json for the ledger/ and stages/ modules K4 touched).

Usage: python tools/typing/check_typing.py   (needs pyright==1.1.408 from the dev group; uses this interpreter's packages)
Exit 0 when the gate holds; prints the new diagnostics otherwise. Baseline diagnostics are matched by file, rule and
message (line numbers ignored), so moved code keeps its baseline entry and a fixed one simply disappears.
"""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    config = json.loads((HERE / 'pyrightconfig.json').read_text())
    baseline = json.loads((HERE / 'baseline_k3a.json').read_text())
    environment = {**os.environ, 'PYRIGHT_PYTHON_FORCE_VERSION': baseline['pyright']}
    done = subprocess.run([sys.executable, '-m', 'pyright', '-p', str(HERE / 'pyrightconfig.json'), '--pythonpath', sys.executable,
                           '--outputjson'], capture_output=True, text=True, env=environment, cwd=HERE)
    report = json.loads(done.stdout)
    strict = {str((HERE / path).resolve()).split('/src/')[-1] for path in config['strict']}
    allowed = Counter({(row['file'], row['rule'], row['message']): row['count'] for row in baseline['diagnostics']})
    # CP-K4b F5: the staged gate also covers every ledger/ and stages/ module K4 touched, against their pre-K4 errors.
    extended = HERE / 'baseline_k4.json'
    if extended.exists():
        allowed.update({(row['file'], row['rule'], row['message']): row['count'] for row in json.loads(extended.read_text())['diagnostics']})
    found = Counter()
    strict_errors = []
    for item in report['generalDiagnostics']:
        if item['severity'] != 'error':
            continue
        name = item['file'].split('/src/')[-1]
        key = (name, item.get('rule') or '', item['message'])
        if name in strict:
            strict_errors.append(item)
        else:
            found[key] += 1
    new = found - allowed
    result = dict(pyright=report.get('version'), summary=report['summary'], strict_files=sorted(strict), strict_errors=len(strict_errors),
                  baseline_errors=sum(allowed.values()), touched_errors=sum(found.values()), new_errors=sum(new.values()),
                  fixed_baseline_errors=sum((allowed - found).values()))
    print(json.dumps(result, indent=2))
    for item in strict_errors:
        print('STRICT', item['file'].split('/src/')[-1], item['range']['start']['line'] + 1, item.get('rule'), item['message'].splitlines()[0])
    for (name, rule, message), count in new.items():
        print('NEW', count, name, rule, message.splitlines()[0])
    return 0 if not strict_errors and not new else 1


if __name__ == '__main__':
    raise SystemExit(main())
