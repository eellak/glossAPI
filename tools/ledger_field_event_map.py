"""Ledger field/event map generator (W18 M14), static: it never imports the scanned tree.

    python tools/ledger_field_event_map.py [TREE] [--json OUT.json] [--md OUT.md] [--detail]
        TREE defaults to this repository; the map defaults to docs/ledger_field_event_map.json (the checked-in map that
        tests/test_ledger_field_event_map.py compares with a fresh scan). --detail keeps line numbers and absolute paths.

What it scans (Python stdlib only: ast, sqlite3 in memory, re):
- objects:
  - ``event_kind``, ``reason_code``, ``status``: the frozenset literals EVENT_KINDS / REASON_CODES / STATUSES in
    ``src/glossapi/ledger/catalogue.py``;
  - ``snapshot_kind``: the first string argument of every ``.snapshot('<kind>', ...)`` call in ``src/``;
  - ``field``: ``table.column`` of a fresh ledger: every module-level ``*_SQL`` string of ``ledger/schema.py`` is applied
    in source order to an in-memory SQLite database (the order ``LedgerWriter`` uses for a new ledger), then
    ``PRAGMA table_info`` of every table and view (schema INSERT triggers count as writers).
- references, per object, each as ``path:function:line`` (``tests`` as pytest node ids ``tests/file.py::test``):
  - ``writers`` (``src/`` outside the catalogue/schema definitions and outside report/CLI modules):
    - codes and kinds: a literal in an emitting position (call argument, ``reason_code=``/``event_kind=``/``status=``
      keyword, dict value, return/assignment), never a comparison operand;
    - snapshot kinds: the ``.snapshot`` call;
    - fields: an SQL literal with ``INSERT [OR ...] INTO <table>`` naming the column (or a positional VALUES insert);
  - ``computed_writers``: codes with no literal writer that match a computed pattern found in an emitting position,
    ``<stage> + '<suffix>'`` or ``f'{stage}<suffix>'`` (``stage`` from catalogue STAGES), e.g. ``extract_completed``;
  - ``readers``: a literal used as a comparison operand (``==``, ``!=``, ``in``), inside an SQL literal that reads
    (``SELECT``/``WHERE``) next to ``reason_code``/``event_kind``/``status``/``kind``, or for fields a reading SQL
    literal that names the table (or a view over it) and the column, or a ``row['column']`` subscript;
  - ``report_cli``: any occurrence in a report/CLI module (``cli.py``, ``ledger/reader.py``, ``plan_preview.py``,
    ``llm_guide.py``, ``doctor.py``, ``capabilities.py``, ``stages/receipts.py`` is NOT one) and in ``tools/``;
  - ``tests``: every test function in ``tests/`` whose source names the object (string literal equal to it, or the
    token inside a string literal; fields: the column token and the table name in one test).
- flags: ``no_writer`` (neither literal nor computed), ``no_reader`` (no src reader and no report/CLI reference),
  ``no_test``. ``status`` per W18 M14: ``active`` (has a writer), ``legacy_import_only`` (writers only in
  ``ledger/importer.py``), ``no_writer`` (reserved or dead: a human decides; the map does not guess).

Heuristic limits, stated in the output: generic words that are both codes and other values (``deleted``, ``corrupt``,
``missing``...) over-count references; codes built from data at run time (backend tokens) can be missed; a field's
``row['col']`` reader cannot tell which table the row came from. The output keeps every occurrence so a reviewer can
audit any row. Re-run it on any tree (for example the K4(c) part-2 worktree); the counts and flags are recomputed.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

DEFAULT_TREE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REPORT_MODULES = ('src/glossapi/cli.py', 'src/glossapi/ledger/reader.py', 'src/glossapi/plan_preview.py', 'src/glossapi/llm_guide.py',
                  'src/glossapi/doctor.py')
DEFINITION_MODULES = ('src/glossapi/ledger/catalogue.py', 'src/glossapi/ledger/schema.py')
IMPORTER = 'src/glossapi/ledger/importer.py'
CODE_KEYWORDS = {'reason_code', 'reason', 'code', 'skip_reason', 'event_kind', 'status', 'kind', 'presence', 'cause'}
TOKEN = r'(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])'


def frozenset_literal(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and value.args:
                value = value.args[0]
            if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
                return [element.value for element in value.elts if isinstance(element, ast.Constant) and isinstance(element.value, str)]
    return []


def constant_int(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets) and isinstance(node.value, ast.Constant):
            return node.value.value
    return None


def schema_fields(schema_path):
    """Apply every module-level *_SQL constant (with += continuations) in source order to an in-memory database."""
    tree = ast.parse(schema_path.read_text())
    constants, order = {}, []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id.endswith('_SQL'):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                name = node.targets[0].id
                if name not in constants: order.append(name)
                constants[name] = node.value.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id.endswith('_SQL') and isinstance(node.value, ast.Constant):
            constants[node.target.id] = constants.get(node.target.id, '') + node.value.value
    database = sqlite3.connect(':memory:')
    applied, failures = [], []
    for name in order:
        try:
            database.executescript(constants[name]); applied.append(name)
        except sqlite3.Error as error:
            failures.append(dict(constant=name, error=str(error)))
    fields = {}
    for name, kind, sql in database.execute("SELECT name,type,sql FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"):
        columns = [row[1] for row in database.execute(f'PRAGMA table_info("{name}")')]
        base = None
        if kind == 'view':
            match = re.search(r'\bFROM\s+([A-Za-z_]+)', sql or '', re.I)
            base = match.group(1) if match else None
        fields[name] = dict(type=kind, columns=columns, view_of=base)
    return fields, dict(applied=applied, failures=failures)


class Scanner(ast.NodeVisitor):
    """Collect string-literal occurrences with their context, the enclosing function and computed suffix patterns."""

    def __init__(self, relative, source):
        self.relative, self.source = relative, source
        self.stack, self.parents = [], {}
        self.occurrences = []   # (value, context, function, line)
        self.strings = []       # (value, function, line) every string constant
        self.computed = []      # (prefix_kind, constant, function, line)
        self.snapshots = []     # (kind, function, line)
        self.subscripts = []    # (key, function, line)

    def function(self):
        return self.stack[-1] if self.stack else '<module>'

    def visit_FunctionDef(self, node):
        self.stack.append(node.name if not self.stack else self.stack[-1]+'.'+node.name)
        self.generic_visit(node); self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.stack.append(node.name if not self.stack else self.stack[-1]+'.'+node.name)
        self.generic_visit(node); self.stack.pop()

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
        super().generic_visit(node)

    def context(self, node):
        parent = self.parents.get(node)
        # Climb through container literals.
        holder = node
        while isinstance(parent, (ast.Tuple, ast.List, ast.Set)) or (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                                                                      and parent.func.id in ('frozenset', 'set', 'tuple', 'list') and holder in parent.args):
            holder, parent = parent, self.parents.get(parent)
        if isinstance(parent, ast.Compare):
            return 'compare'
        if isinstance(parent, ast.keyword):
            return 'keyword:'+(parent.arg or '')
        if isinstance(parent, ast.Dict):
            return 'dict_key' if holder in parent.keys else 'dict_value'
        if isinstance(parent, ast.Subscript):
            return 'subscript'
        if isinstance(parent, ast.Call):
            return 'call_arg:'+(parent.func.attr if isinstance(parent.func, ast.Attribute) else getattr(parent.func, 'id', '?'))
        if isinstance(parent, (ast.Return, ast.Assign, ast.AnnAssign, ast.IfExp, ast.BoolOp, ast.Yield)):
            return 'value'
        if isinstance(parent, (ast.Expr,)):
            return 'docstring'
        return 'other:'+type(parent).__name__

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self.strings.append((node.value, self.function(), node.lineno))
            self.occurrences.append((node.value, self.context(node), self.function(), node.lineno))
            parent = self.parents.get(node)
            if isinstance(parent, ast.Subscript):
                self.subscripts.append((node.value, self.function(), node.lineno))
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            for side, other in ((node.right, node.left), (node.left, node.right)):
                if isinstance(side, ast.Constant) and isinstance(side.value, str) and re.fullmatch(r'_[a-z_]+', side.value) and not isinstance(other, ast.Constant):
                    self.computed.append(('suffix', side.value, self.function(), node.lineno, self.context(node)))
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        parts = node.values
        for index, part in enumerate(parts):
            if isinstance(part, ast.Constant) and isinstance(part.value, str) and index > 0 and isinstance(parts[index-1], ast.FormattedValue):
                if re.fullmatch(r'_[a-z_]+', part.value):
                    self.computed.append(('suffix', part.value, self.function(), node.lineno, self.context(node)))
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'snapshot' and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            self.snapshots.append((node.args[0].value, self.function(), node.lineno))
        self.generic_visit(node)


def scan(path, relative):
    source = path.read_text(encoding='utf-8', errors='replace')
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    scanner = Scanner(relative, source)
    for child in ast.iter_child_nodes(tree):
        scanner.parents[child] = tree
    scanner.visit(tree)
    return scanner


def emitting(context):
    return context in ('dict_value', 'value') or context.startswith('call_arg:') or (context.startswith('keyword:') and context[8:] in CODE_KEYWORDS)


SQL_READ = re.compile(r'\b(SELECT|WHERE)\b', re.I)
SQL_INSERT = re.compile(r'\bINSERT\s+(?:OR\s+\w+\s+)?INTO\s+([A-Za-z_]+)\s*(\(([^)]*)\))?', re.I)


def git_head(tree):
    try:
        return subprocess.run(['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build(tree):
    tree = Path(tree).resolve()
    catalogue = ast.parse((tree/'src/glossapi/ledger/catalogue.py').read_text())
    codes = {'event_kind': frozenset_literal(catalogue, 'EVENT_KINDS'), 'reason_code': frozenset_literal(catalogue, 'REASON_CODES'),
             'status': frozenset_literal(catalogue, 'STATUSES')}
    stages = set(frozenset_literal(catalogue, 'STAGES'))
    schema_path = tree/'src/glossapi/ledger/schema.py'
    fields, ddl = schema_fields(schema_path)
    scanners = {}
    for folder in ('src', 'tools', 'tests'):
        for path in sorted((tree/folder).rglob('*.py')):
            relative = str(path.relative_to(tree))
            if '/__pycache__/' in relative: continue
            # The golden/perf measurement harness is test tooling, not a product writer or reader of the ledger; shipped
            # gate trees add measurement-only files there, so it is left out to keep the map a function of the product.
            if relative.startswith(('tools/harness/', 'tests/harness/')): continue
            result = scan(path, relative)
            if result: scanners[relative] = result
    src = {k: v for k, v in scanners.items() if k.startswith('src/') and k not in DEFINITION_MODULES}
    report = {k: v for k, v in scanners.items() if k in REPORT_MODULES or k.startswith('tools/')}
    tests = {k: v for k, v in scanners.items() if k.startswith('tests/')}
    emitters = {k: v for k, v in src.items() if k not in REPORT_MODULES}
    snapshot_kinds = sorted({kind for scanner in src.values() for kind, _, _ in scanner.snapshots})
    computed = [(constant, f'{rel}:{function}:{line}') for rel, scanner in emitters.items()
                for _, constant, function, line, context in scanner.computed]

    def test_ids(predicate):
        hits = set()
        for rel, scanner in tests.items():
            for value, function, line in scanner.strings:
                if predicate(value):
                    hits.add(f'{rel}::{function.split(".")[0]}' if function != '<module>' else rel)
        return sorted(hits)

    entries = []
    all_codes = [(kind, name) for kind in ('event_kind', 'reason_code', 'status') for name in sorted(codes[kind])] + [('snapshot_kind', k) for k in snapshot_kinds]
    for kind, name in all_codes:
        pattern = re.compile(TOKEN.format(re.escape(name)))
        writers, readers, report_refs, computed_writers = [], [], [], []
        for rel, scanner in src.items():
            in_report = rel in REPORT_MODULES
            for value, context, function, line in scanner.occurrences:
                where = f'{rel}:{function}:{line}'
                if value == name:
                    if in_report:
                        report_refs.append(where)
                    elif context == 'compare':
                        readers.append(where)
                    elif kind != 'snapshot_kind' and emitting(context):
                        writers.append(where)
                elif len(value) > len(name) and pattern.search(value) and SQL_READ.search(value) and re.search(r'reason_code|event_kind|status|kind', value):
                    (report_refs if in_report else readers).append(where)
            if kind == 'snapshot_kind':
                writers += [f'{rel}:{function}:{line}' for value, function, line in scanner.snapshots if value == name and not in_report]
        for rel, scanner in report.items():
            if rel.startswith('tools/'):
                report_refs += [f'{rel}:{function}:{line}' for value, function, line in scanner.strings if value == name or (len(value) > len(name) and pattern.search(value))]
        if kind == 'reason_code' and not writers:
            for constant, where in computed:
                if name.endswith(constant) and name[:-len(constant)] in stages:
                    computed_writers.append(f'{where} ({name[:-len(constant)]}+{constant!r})')
        test_hits = test_ids(lambda value: value == name or (len(value) > len(name) and bool(pattern.search(value))))
        has_writer = bool(writers or computed_writers)
        status = ('active' if any(not w.startswith(IMPORTER) for w in writers) or computed_writers else
                  'legacy_import_only' if writers else 'no_writer')
        flags = [flag for flag, bad in (('no_writer', not has_writer), ('no_reader', not readers and not report_refs), ('no_test', not test_hits)) if bad]
        entries.append(dict(object=kind, name=name, status=status, writers=sorted(set(writers)), computed_writers=sorted(set(computed_writers)),
                            readers=sorted(set(readers)), report_cli=sorted(set(report_refs)), tests=test_hits, flags=flags))

    # Fields: table.column.
    views_over = {}
    for name, info in fields.items():
        if info['type'] == 'view' and info['view_of']:
            views_over.setdefault(info['view_of'], set()).add(name)
    for table, info in sorted(fields.items()):
        names = {table} | views_over.get(table, set()) | ({info['view_of']} if info['view_of'] else set())
        name_pattern = re.compile('|'.join(TOKEN.format(re.escape(n)) for n in sorted(names)))
        for column in info['columns']:
            column_pattern = re.compile(TOKEN.format(re.escape(column)))
            writers, readers, key_readers, report_refs = [], [], [], []
            # Schema triggers write too (for example the files view's INSERT trigger fills paths.uri).
            schema_scanner = scanners.get('src/glossapi/ledger/schema.py')
            for value, function, line in (schema_scanner.strings if schema_scanner else ()):
                for match in SQL_INSERT.finditer(value):
                    if match.group(1) in names and match.group(3) is not None and column_pattern.search(match.group(3)) and 'CREATE TRIGGER' in value.upper():
                        writers.append(f'src/glossapi/ledger/schema.py:trigger:{line}')
            for rel, scanner in src.items():
                in_report = rel in REPORT_MODULES
                for value, function, line in scanner.strings:
                    where = f'{rel}:{function}:{line}'
                    for match in SQL_INSERT.finditer(value):
                        if match.group(1) in names and (match.group(3) is None or column_pattern.search(match.group(3))) and not in_report:
                            writers.append(where)
                    if SQL_READ.search(value) and name_pattern.search(value) and (column_pattern.search(value) or re.search(r'\*', value)):
                        (report_refs if in_report else readers).append(where)
                for key, function, line in scanner.subscripts:
                    if key == column:
                        (report_refs if in_report else key_readers).append(f'{rel}:{function}:{line}')
            test_hits = sorted(set(test_ids(lambda value: bool(column_pattern.search(value)) and bool(name_pattern.search(value)))))
            flags = [flag for flag, bad in (('no_writer', not writers and info['type'] == 'table'), ('no_reader', not readers and not report_refs and not key_readers),
                                            ('no_test', not test_hits)) if bad]
            entries.append(dict(object='field', name=f'{table}.{column}', table_type=info['type'], status='active' if writers or info['type'] == 'view' else 'no_writer',
                                writers=sorted(set(writers)), readers=sorted(set(readers)), key_readers=sorted(set(key_readers))[:40],
                                report_cli=sorted(set(report_refs)), tests=test_hits, flags=flags))

    counts = {}
    for entry in entries:
        group = counts.setdefault(entry['object'], dict(total=0, no_writer=0, no_reader=0, no_test=0, legacy_import_only=0))
        group['total'] += 1
        for flag in entry['flags']: group[flag] += 1
        if entry['status'] == 'legacy_import_only': group['legacy_import_only'] += 1
    return dict(generator='tools/ledger_field_event_map.py', tree=str(tree), tree_head=git_head(tree),
                catalogue_version=constant_int(catalogue, 'CATALOGUE_VERSION'), schema_version=constant_int(ast.parse(schema_path.read_text()), 'SCHEMA_VERSION'),
                ddl=ddl, counts=counts, computed_suffixes=sorted({c for c, _ in computed}),
                limits=['tools/harness/ and tests/harness/ (golden and performance measurement tooling) are not scanned',
                        'generic codes that are also ordinary values (deleted, corrupt, missing, ...) over-count references',
                        'codes assembled from backend data at run time can be missed; computed writers match <stage>+suffix only',
                        "a field's row['col'] reader cannot tell which table the row came from (key_readers)",
                        'readers in report/CLI modules are listed under report_cli, not readers',
                        'every reason code is also surfaced generically by report --why/status (the reason_code column); '
                        'no_reader means no code-specific consumer, not that the code is invisible'],
                entries=entries)


# Writers that no literal scan can see, each named with the code that computes the value (W18 M14).
COMPUTED = {
    ('field', 'paths.path_id'): 'SQLite INTEGER PRIMARY KEY rowid alias, assigned by the files view INSERT trigger (ledger/schema.py)',
    ('field', 'ledger_format.singleton'): 'ledger/schema.py FORMAT_SCHEMA_SQL (migration 4) INSERT of the header row',
    ('field', 'ledger_format.json_dictionary_id'): 'ledger/schema.py FORMAT_SCHEMA_SQL (migration 4) INSERT of the header row',
    ('field', 'ledger_format.json_dictionary_sha256'): 'ledger/schema.py FORMAT_SCHEMA_SQL (migration 4) INSERT of the header row',
}


def repository_map(data):
    """The portable, checked-in form: computed writers applied, ``path:function`` references without line numbers."""
    strip = lambda items: sorted({':'.join(item.split(':')[:2]) if not item.startswith('tests/') else item for item in items})
    entries = []
    for entry in data['entries']:
        entry = dict(entry)
        computed = COMPUTED.get((entry['object'], entry['name']))
        if computed and entry['status'] == 'no_writer':
            entry.update(status='computed', computed_writers=[computed], flags=[flag for flag in entry['flags'] if flag != 'no_writer'])
        for key in ('writers', 'computed_writers', 'readers', 'key_readers', 'report_cli'):
            if key in entry:
                entry[key] = strip(entry[key]) if key != 'computed_writers' else sorted(set(entry[key]))
        entries.append(entry)
    counts = {}
    for entry in entries:
        group = counts.setdefault(entry['object'], dict(total=0, no_writer=0, no_reader=0, no_test=0, legacy_import_only=0, computed=0))
        group['total'] += 1
        for flag in entry['flags']: group[flag] += 1
        if entry['status'] in ('legacy_import_only', 'computed'): group[entry['status']] += 1
    return dict(generator=data['generator'], catalogue_version=data['catalogue_version'], schema_version=data['schema_version'],
                counts=counts, computed_suffixes=data['computed_suffixes'], limits=data['limits'], entries=entries)


def markdown(data):
    lines = [f"# Ledger field/event map ({data['tree_head'] or 'unknown head'})", '',
             f"Generated by `{data['generator']}` from `{data['tree']}`; catalogue v{data['catalogue_version']}, schema v{data['schema_version']}. "
             'Static scan; see the JSON `limits`.', '', '## Counts', '',
             '| object | total | no writer | no reader | no test | legacy import only |', '| --- | --- | --- | --- | --- | --- |']
    for kind, group in data['counts'].items():
        lines.append(f"| {kind} | {group['total']} | {group['no_writer']} | {group['no_reader']} | {group['no_test']} | {group['legacy_import_only']} |")
    def short(items, n=2):
        if not items: return '—'
        head = ', '.join('`'+item.split(':')[0].replace('src/glossapi/', '').replace('tests/', '')+(':'+item.split(':')[1] if ':' in item and not item.startswith('tests/') else '')+'`' for item in items[:n])
        return head + (f' +{len(items)-n}' if len(items) > n else '')
    for kind in ('event_kind', 'reason_code', 'status', 'snapshot_kind', 'field'):
        rows = [e for e in data['entries'] if e['object'] == kind]
        if not rows: continue
        lines += ['', f'## {kind} ({len(rows)})', '', '| name | status | writers | readers | report/CLI | tests | flags |', '| --- | --- | --- | --- | --- | --- | --- |']
        for e in rows:
            writers = e['writers'] or e.get('computed_writers', [])
            lines.append(f"| `{e['name']}` | {e['status']} | {short(writers)} | {short(e['readers'] or e.get('key_readers', []))} | {short(e['report_cli'], 1)} | "
                         f"{len(e['tests'])} | {', '.join(e['flags']) or '—'} |")
    return '\n'.join(lines) + '\n'


def serialize(data):
    """The checked-in JSON: no tree path or head (the test compares it with a fresh scan of the same tree)."""
    portable = {key: value for key, value in data.items() if key not in ('tree', 'tree_head', 'ddl', 'entries')}
    head = json.dumps(portable, indent=1, ensure_ascii=False)[:-2]
    rows = ',\n'.join('  ' + json.dumps(entry, ensure_ascii=False, separators=(',', ':')) for entry in data['entries'])
    return head + ',\n "entries": [\n' + rows + '\n ]\n}\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('tree', nargs='?', default=str(DEFAULT_TREE))
    parser.add_argument('--json', default=str(HERE.parent/'docs/ledger_field_event_map.json'))
    parser.add_argument('--md', default=None)
    parser.add_argument('--detail', action='store_true')
    arguments = parser.parse_args(argv)
    data = build(arguments.tree)
    if not arguments.detail:
        data = {**repository_map(data), 'ddl': {'failures': data['ddl']['failures']}, 'tree': data['tree'], 'tree_head': data['tree_head']}
        Path(arguments.json).write_text(serialize(data))
    else:
        Path(arguments.json).write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    if arguments.md:
        Path(arguments.md).write_text(markdown(data))
    print(json.dumps(dict(tree=data['tree'], head=data['tree_head'], counts=data['counts'], ddl_failures=data['ddl']['failures']), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
