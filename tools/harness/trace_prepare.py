"""Copy frozen source into a scratch directory and instrument it there only (K1/W12).

The production tree is never edited. Usage::

    python -m tools.harness.trace_prepare SOURCE_REPOSITORY TRACE_ROOT

writes TRACE_ROOT/scratch (instrumented copy of src/ and tools/), the hash list
before instrumentation, the changed-file list, and every text anchor result.

Origin: experiments ledger_step11/drivers/trace_prepare.py. K1/W12 repairs:
- LS11-G03: decorators carry class-qualified names (``ledger/writer.py:SyncBatch.close``
  and ``ledger/writer.py:LedgerWriter.close`` are distinct spans). A spec key may
  be a qualified ``Class.method`` or a bare name; a bare name matches every
  function of that name and is still recorded under its qualified name.
- LS11-G02: ``SyncBatch.flush`` and ``LedgerWriter.durable_barrier`` use
  ``writeback_measured`` (per-call span with /proc/meminfo Dirty/Writeback before
  and after); ``diagtrace.install_native`` wraps every ``syncfs``/``fsync`` call
  the same way.
- A text anchor that no longer matches raises instead of silently instrumenting less.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

IMPORT_LINE = 'from tools.harness import diagtrace as _dt\n'
LEDGER = ['snapshot', 'run', 'document', 'version', 'source_record', 'task', 'dependency', 'attempt', 'input', 'event',
          'artifact', 'finish', 'close']
RETENTION = ['retain_completed_run', 'delete_many', '_complete', 'eligibility', 'declare', 'retention_facts', 'seal',
             '_prune_completed_policies', 'release_private_scratch', '_finish_private_release', '_unlink_private_file',
             'retire_completed_scratch', 'recover_private_releases']
# path relative to src/glossapi -> {function key: (category, record, kind)}; kind is 'span' or 'writeback'.
SPECS = {
    'ledger/writer.py': {**{name: ('ledger', False, 'span') for name in LEDGER},
                         'SyncBatch.flush': ('ledger', True, 'writeback'),
                         'LedgerWriter.durable_barrier': ('ledger', True, 'writeback')},
    'ledger/compact.py': {'execute': ('sqlite', False, 'span'), 'executemany': ('sqlite', False, 'span'), 'commit': ('sqlite', False, 'span')},
    'ledger/encoding.py': {'file_hash': ('hashing', False, 'span')},
    'ledger/journal.py': {'inventory': ('publication', True, 'span'), 'finish': ('ledger', True, 'span')},
    'stages/execution.py': {'sha256_file': ('hashing', False, 'span'), 'freeze_inputs': ('verification', False, 'span'),
                            'seal_output_hashes': ('verification', False, 'span')},
    'stages/run.py': {'prepare_streaming': ('orchestrator', True, 'span'), 'publish_barrier': ('publication', True, 'span'),
                      '_run_corpus': ('orchestrator', True, 'span')},
    'stages/scheduler.py': {'freeze': ('scheduler', True, 'span'), 'execute': ('scheduler', True, 'span'),
                            '_outputs': ('ledger', False, 'span'), 'result': ('ledger', False, 'span')},
    'stages/sequential.py': {'prepare': ('publication', True, 'span'), 'verify_projection': ('verification', True, 'span'),
                             'complete': ('publication', True, 'span')},
    'stages/products.py': {'execute_aggregate': ('publication', True, 'span'), 'publish_products': ('publication', True, 'span')},
    'stages/retention.py': {name: ('retention', name in ('retain_completed_run', 'delete_many', '_complete', 'release_private_scratch'), 'span')
                            for name in RETENTION},
    'stages/extract.py': {'_run_document': ('stage', True, 'span'), 'run_prepared': ('stage', True, 'span')},
    'stages/clean.py': {'run_document': ('stage', True, 'span')},
    'stages/sections.py': {'run_document': ('stage', True, 'span'), 'finalize': ('stage', True, 'span')},
    'stages/annotate.py': {'finalize': ('stage', True, 'span')},
    'stages/export.py': {'finalize': ('stage', True, 'span')},
    'gloss_extract.py': {'_process_single_document': ('kernel', True, 'span'), '_process_single_document_pypdfium': ('kernel', True, 'span'),
                         'extract_path': ('kernel', True, 'span')},
    'gloss_section.py': {'rows_for_document': ('kernel', True, 'span'), 'to_parquet': ('kernel', True, 'span')},
    'corpus/phase_annotate.py': {'annotate': ('kernel', True, 'span')},
    'corpus/phase_export.py': {'jsonl': ('kernel', True, 'span')},
}

# (file, anchor, replacement): the step-11 worker/scheduler event hooks.
TEXT_PATCHES = [
    ('stages/workers.py', 'from __future__ import annotations', 'from __future__ import annotations\n' + IMPORT_LINE.rstrip('\n')),
    ('stages/workers.py', '    os.setsid()', '    _dt.install_native()\n    os.setsid()'),
    ('stages/workers.py', '            command = connection.recv()',
     '            with _dt.span("worker_idle", "idle"):\n                command = connection.recv()'),
    ('stages/workers.py', '            stage, request, aggregate = command',
     '            stage, request, aggregate = command\n            _dt.emit("worker_received", task=request.task_id, attempt=request.attempt_id, stage=stage.name, operation=request.operation)'),
    ('stages/workers.py', "            target = request.work_directory/'worker-result.json'",
     '            _dt.emit("worker_ready", task=request.task_id, attempt=request.attempt_id, stage=stage.name, status=result.status)\n            _dt.dump()\n            target = request.work_directory/\'worker-result.json\''),
    ('stages/workers.py', '        self.capacity = workers', '        self.capacity = workers\n        self._diag_stage = None'),
    ('stages/workers.py', "        slot['pipe'].send((stage, request, aggregate))",
     "        self._diag_stage = stage.name\n        slot['pipe'].send((stage, request, aggregate))\n        _dt.emit('submitted', task=request.task_id, attempt=request.attempt_id, stage=stage.name, worker_pid=slot['process'].pid)"),
    ('stages/workers.py', "                    slot['request'] = None",
     "                    _dt.emit('collected', task=request.task_id, attempt=request.attempt_id, stage=self._diag_stage, worker_pid=process.pid)\n                    slot['request'] = None"),
    ('stages/workers.py', '    finally:\n        connection.close()', '    finally:\n        _dt.dump()\n        connection.close()'),
    ('stages/scheduler.py', '                            queue.append(key)',
     "                            queue.append(key)\n                            _dt.emit('enqueued', key=key, stage=operation.stage, task=tasks[key])"),
    ('stages/scheduler.py', '        self.trace.append(row)', "        self.trace.append(row)\n        _dt.emit('scheduler:'+event, **details)"),
]


def qualified_functions(tree):
    """Yield (qualified name, function node) for every function, with class prefixes."""
    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                yield from visit(child, prefix + child.name + '.')
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield prefix + child.name, child
                yield from visit(child, prefix + child.name + '.<locals>.')
            else:
                yield from visit(child, prefix)
    yield from visit(tree, '')


def instrument(path, relative, names):
    text = path.read_text()
    tree = ast.parse(text)
    insertions = []
    matched = []
    for qualified, node in qualified_functions(tree):
        spec = names.get(qualified) or names.get(node.name)
        if spec is None:
            continue
        category, record, kind = spec
        line = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        indent = ' ' * node.col_offset
        label = relative + ':' + qualified
        decorator = (f'@_dt.writeback_measured({label!r}, {category!r})' if kind == 'writeback'
                     else f'@_dt.measured({label!r}, {category!r}, {record!r})')
        insertions.append((line, indent + decorator + '\n'))
        matched.append(label)
    lines = text.splitlines(keepends=True)
    for line, addition in sorted(insertions, reverse=True):
        lines.insert(line, addition)
    import_line = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            import_line = node.end_lineno
    if not import_line and tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) \
            and isinstance(tree.body[0].value.value, str):
        import_line = tree.body[0].end_lineno
    lines.insert(import_line, IMPORT_LINE)
    path.write_text(''.join(lines))
    missing = sorted(key for key in names if not any(label.endswith(':' + key) or label.endswith('.' + key) for label in matched))
    return matched, missing


def digest_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def prepare(source, root):
    source, root = Path(source), Path(root)
    scratch = root / 'scratch'
    scratch.mkdir(parents=True)
    shutil.copytree(source / 'src', scratch / 'src')
    shutil.copytree(source / 'tools', scratch / 'tools')
    for name in ('pyproject.toml', '.code-receipt.json'):
        if (source / name).exists():
            shutil.copy2(source / name, scratch / name)
    before = digest_tree(scratch)
    (root / 'scratch-before.json').write_text(json.dumps(before, indent=2))
    report = {'decorated': {}, 'missing_spec_functions': {}, 'text_patches': []}
    package = scratch / 'src/glossapi'
    for relative, names in SPECS.items():
        path = package / relative
        if not path.exists():
            report['missing_spec_functions'][relative] = ['<file absent>']
            continue
        matched, missing = instrument(path, relative, names)
        report['decorated'][relative] = matched
        if missing:
            report['missing_spec_functions'][relative] = missing
    texts = {}
    for relative, anchor, replacement in TEXT_PATCHES:
        path = package / relative
        text = texts.get(relative, path.read_text())
        if text.count(anchor) < 1:
            raise ValueError(f'trace anchor not found in {relative}: {anchor!r}')
        texts[relative] = text.replace(anchor, replacement, 1)
        report['text_patches'].append(dict(file=relative, anchor=anchor))
    for relative, text in texts.items():
        (package / relative).write_text(text)
    after = digest_tree(scratch)
    changed = {name: digest for name, digest in after.items() if before.get(name) != digest}
    (root / 'instrumented-files.json').write_text(json.dumps(changed, indent=2))
    (root / 'instrumentation.json').write_text(json.dumps(report, indent=2))
    return scratch


if __name__ == '__main__':
    prepare(*map(Path, sys.argv[1:3]))
