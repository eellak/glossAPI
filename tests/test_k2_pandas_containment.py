"""K2 pandas decision: pandas stays a core dependency, and the light entry points never import it.

In a fresh interpreter with every `pandas` import made to fail: `import glossapi`, the CLI, the ledger and its
reader, the stage runner module, and the CLI plan/status/report paths (including `report --why`) must succeed.
A negative control shows the guard trips on a path that does use pandas.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from test_ledger_cli import seed

REPO = Path(__file__).resolve().parents[1]
BLOCK = '''
import importlib.abc, sys
class NoPandas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'pandas':
            raise ModuleNotFoundError('pandas blocked by the containment test', name=fullname)
sys.meta_path.insert(0, NoPandas())
'''


def run(code, *args):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(REPO / 'src'), *filter(None, [os.environ.get('PYTHONPATH')])])}
    return subprocess.run([sys.executable, '-B', '-c', BLOCK + code, *map(str, args)], capture_output=True, text=True, env=env)


def test_light_entry_points_import_and_run_without_pandas(tmp_path):
    run_id, document, _ = seed(tmp_path)
    result = run('''
import contextlib, io, json
import glossapi
import glossapi.cli, glossapi.ledger, glossapi.ledger.reader, glossapi.stages.run
from glossapi.cli import main
corpus, absent, document = sys.argv[1], sys.argv[2], sys.argv[3]
codes = {}
for name, argv in (('plan', ['plan', '--corpus', absent, '--files', '/absent.pdf', '--stages', 'extract,clean,annotate']),
                   ('status', ['status', '--corpus', corpus]),
                   ('report', ['report', '--corpus', corpus]),
                   ('why', ['report', '--corpus', corpus, '--why', document])):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        codes[name] = main([*argv, '--json'])
    envelope = json.loads(out.getvalue())
    assert not envelope['errors'], (name, envelope['errors'])
assert codes == {'plan': 0, 'status': 1, 'report': 0, 'why': 0}, codes
loaded = sorted(n for n in sys.modules if n.split('.')[0] == 'pandas')
assert not loaded, loaded
print(json.dumps(codes))
''', tmp_path, tmp_path / 'absent', document)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {'plan': 0, 'status': 1, 'report': 0, 'why': 0}


def test_the_guard_is_not_vacuous():
    """The same guard fails a path that imports pandas (the Corpus orchestrator)."""
    result = run('from glossapi import Corpus\nCorpus\n')
    assert result.returncode != 0 and 'pandas blocked by the containment test' in result.stderr
