"""K4(b): the download path through ``Corpus.run`` (loopback HTTP, a few tiny PDFs).

- K4BP-G1: a publication interrupted after its first ``downloads/`` copy resumes with the frozen names, publishes every
  download once, writes one results row per source and leaves no task running.
- K4BP-G4/G5: an amended plan keeps each source's name, never gives a new source a name in use, publishes the new
  download, and its preview shows the re-fetch the run performs.
- K4BP-G3: memory_saving=extreme deletes each owned download after its last document reader while the run goes on.
- K4BP-L1: a failed URL is that document's typed failure; the corpus-wide publication succeeds for the others and a
  debug-off run leaves no ``.tmp`` product behind.
- K4C2-G2: only the document extraction names a publication source (an OCR run's page-quality step does not).
"""
import json
from pathlib import Path
import threading

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

pytest.importorskip('aiohttp')
pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def pdf_bytes(tmp_path, text):
    from test_ledger_fixset_lg import pdf
    return pdf(tmp_path/'pdfs'/(text.replace(' ', '_')+'.pdf'), text).read_bytes()


class Server:
    """Loopback server over a mutable ``path -> bytes`` table; a path that is absent answers 404."""
    def __init__(self, files):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.files, self.requests = dict(files), []
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append(self.path)
                body = outer.files.get(self.path)
                if body is None:
                    self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
                self.send_response(200); self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def url(self, path):
        return f'http://127.0.0.1:{self.server.server_port}{path}'

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


SETTINGS = {'download': {'rate_limit': 0}, 'extract': {'repair_fonts': False}}


def results_frame(root):
    import pandas as pd
    return pd.read_parquet(root/'download_results/download_results_sources.parquet')


def export_by_url(root):
    return {row['url']: row['document'] for row in map(json.loads, (root/'export.jsonl').read_text().splitlines())}


def task_rows(root):
    with open_readonly(root/'corpus.sqlite3') as connection:
        table = connection.event_table
        rows = []
        for task_id, run_id, operation in connection.execute('SELECT task_id,run_id,operation FROM tasks ORDER BY rowid'):
            last = connection.execute(f"SELECT status,reason_code,message FROM {table} WHERE task_id=? AND event_kind='lifecycle' "
                                      'ORDER BY event_id DESC LIMIT 1', (task_id,)).fetchone()
            rows.append(dict(task_id=task_id, run_id=run_id, operation=operation, status=last[0], reason_code=last[1], message=last[2]))
        return rows


def test_interrupted_download_publication_resumes_with_frozen_names_and_one_row_per_source(tmp_path, monkeypatch):
    from glossapi.stages import compatibility
    with Server({'/a.pdf': pdf_bytes(tmp_path, 'ALPHA text'), '/b.pdf': pdf_bytes(tmp_path, 'BRAVO text')}) as server:
        corpus = Corpus(tmp_path/'empty', tmp_path/'out')
        original = compatibility.materialize_stage_outputs
        calls = []
        def die_after_first_copy(corpus_, result, workspace, paths):
            original(corpus_, result, workspace, paths)
            if any('/downloads/' in str(path) for path in paths.values()):
                calls.append(paths)
                # Process death after the first downloads/ copy, before its registration finished the wrapper.
                raise KeyboardInterrupt('simulated kill at download publication')
        monkeypatch.setattr(compatibility, 'materialize_stage_outputs', die_after_first_copy)
        with pytest.raises(KeyboardInterrupt):
            corpus.run([server.url('/a.pdf'), server.url('/b.pdf')], stages=('extract', 'clean', 'export'), settings=SETTINGS)
        monkeypatch.setattr(compatibility, 'materialize_stage_outputs', original)
        assert len(calls) == 1 and sorted(p.name for p in (tmp_path/'out/downloads').iterdir()) == ['AAA_000.pdf']
        run_id = next(row['run_id'] for row in task_rows(tmp_path/'out') if row['operation'] == 'download.compatibility_publish')
        result = Corpus(tmp_path/'empty', tmp_path/'out').run(resume=run_id)
    root = tmp_path/'out'
    assert result['state'] == 'completed', result
    assert sorted(p.name for p in (root/'downloads').iterdir()) == ['AAA_000.pdf', 'AAB_001.pdf']
    frame = results_frame(root)
    assert len(frame) == 2 and frame['download_success'].all() and frame['url'].is_unique
    assert set(frame['filename']) == {'AAA_000.pdf', 'AAB_001.pdf'}
    texts = export_by_url(root)
    assert 'ALPHA' in texts[server.url('/a.pdf')] and 'BRAVO' in texts[server.url('/b.pdf')]
    assert not [row for row in task_rows(root) if row['status'] == 'running'], 'a killed wrapper task was left running'


def test_amended_plan_keeps_names_per_source_publishes_new_downloads_and_previews_the_refetch(tmp_path):
    from glossapi.cli import plan_documents
    alpha, bravo, charlie, delta = (pdf_bytes(tmp_path, text) for text in ('ALPHA text', 'BRAVO text', 'CHARLIE text', 'DELTA text'))
    with Server({'/a.pdf': alpha, '/b.pdf': bravo, '/c.pdf': charlie}) as server:
        root = tmp_path/'out'
        manifest = lambda *names: {'manifest_version': 1, 'documents': [{'url': server.url('/'+name)} for name in names]}
        first = Corpus(tmp_path/'empty', root).run([server.url('/a.pdf'), server.url('/b.pdf')], stages=('extract', 'clean', 'export'),
                                                    settings=SETTINGS)
        assert first['state'] == 'completed'
        names_a = dict(zip(results_frame(root)['url'], results_frame(root)['filename']))
        server.files['/a.pdf'] = delta  # the kept URL now serves other bytes
        plan = plan_documents(root, manifest=manifest('a.pdf', 'c.pdf'), stages=['extract', 'clean', 'export'],
                              settings={'download': {'rate_limit': 0}, 'extract': {'repair_fonts': False}})
        targets = {target['source_uri']: target for target in plan['targets']}
        # K4BP-G5: the preview shows the fetch the run performs for a source an earlier run downloaded.
        assert targets[server.url('/a.pdf')]['download']['reacquire'] is True
        assert targets[server.url('/c.pdf')]['download']['reacquire'] is False
        before = len(server.requests)
        second = Corpus(tmp_path/'empty', root).run(plan=plan)
        assert second['state'] == 'completed', second
        assert server.requests[before:].count('/a.pdf') == 1
    frame = results_frame(root)
    names = dict(zip(frame['url'], frame['filename']))
    assert names[server.url('/a.pdf')] == names_a[server.url('/a.pdf')], 'a source keeps its name'
    assert names[server.url('/c.pdf')] not in names_a.values(), 'a new source took a name in use'
    assert (root/'downloads'/names[server.url('/c.pdf')]).read_bytes() == charlie, 'the new download was not published'
    assert (root/'downloads'/names[server.url('/a.pdf')]).read_bytes() == delta, 'the re-acquired bytes were not published'
    texts = export_by_url(root)
    assert 'DELTA' in texts[server.url('/a.pdf')] and 'CHARLIE' in texts[server.url('/c.pdf')]
    # The export holds exactly run B's plan; the removed URL stays only in the ledger history (K4BI-G3).
    assert set(texts) == {server.url('/a.pdf'), server.url('/c.pdf')}
    with open_readonly(root/'corpus.sqlite3') as connection:
        from glossapi.ledger.reader import report_why, select_run
        why = report_why(connection, server.url('/b.pdf'), run=select_run(connection, first['run_id']))
    assert why['occurrences'] and why['occurrences'][0]['disposition']['decision'] == 'include'
    with open_readonly(root/'corpus.sqlite3') as connection:
        from glossapi.ledger.reader import evidence
        replaced = [evidence(connection, {'details_json': row[0]}).get('observation') for row in connection.execute(
            f"SELECT details_json FROM {connection.event_table} WHERE event_kind='artifact' AND reason_code='missing'")]
    assert 'replaced_by_later_acquisition' in replaced, 'the replaced downloads/ copy of the earlier version still reads present'


def test_a_failed_url_is_a_typed_document_failure_and_the_publication_succeeds(tmp_path):
    with Server({'/a.pdf': pdf_bytes(tmp_path, 'ALPHA text')}) as server:
        root = tmp_path/'out'
        result = Corpus(tmp_path/'empty', root).run([server.url('/a.pdf'), server.url('/gone.pdf')],
            stages=('extract', 'clean', 'sections', 'export'), settings={**SETTINGS, 'download': {'rate_limit': 0, 'max_retries': 0}})
    assert result['state'] == 'completed_with_failures'
    rows = task_rows(root)
    publication = next(row for row in rows if row['operation'] == 'download.compatibility_publish')
    assert publication['status'] == 'ok', publication
    failed = [row for row in rows if row['operation'] == 'download.acquire' and row['status'] == 'failed']
    assert len(failed) == 1 and failed[0]['reason_code'] and failed[0]['reason_code'] != 'download_exception'
    assert all(row['status'] in ('ok', 'skipped') for row in rows if row['operation'].endswith('.compatibility_publish'))
    leftovers = [path for path in root.rglob('*') if path.is_file() and '.tmp' in str(path.relative_to(root)) and '.glossapi' not in path.parts]
    assert not leftovers and not (root/'cleaning_report.parquet').exists()
    assert set(export_by_url(root)) == {server.url('/a.pdf')}


def test_extreme_deletes_each_download_after_its_last_reader_while_the_run_goes_on(tmp_path, monkeypatch):
    from glossapi.stages import disk_budget
    from glossapi.stages.retention import resolved_settings, retention_facts
    def guard(root, settings):
        if resolved_settings(settings)['memory_saving'] == 'extreme':
            (Path(root).resolve()/'.glossapi/staging').mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(disk_budget, 'require_disk_guard', guard)  # Linux mounts only; admission is unchanged
    texts = ('ALPHA text', 'BRAVO text', 'CHARLIE text')
    with Server({f'/{i}.pdf': pdf_bytes(tmp_path, text) for i, text in enumerate(texts)}) as server:
        root = tmp_path/'out'
        settings = {**SETTINGS, 'memory_saving': 'extreme', 'disk_budget_bytes': 1 << 30, 'max_source_bytes': 1 << 20,
                    'max_attempt_bytes': 1 << 20}
        result = Corpus(tmp_path/'empty', root).run([server.url(f'/{i}.pdf') for i in range(3)], stages=('extract', 'clean', 'export'),
                                                    settings=settings)
    assert result['state'] == 'completed', result
    assert not (root/'downloads').exists() or not any((root/'downloads').iterdir())
    with open_readonly(root/'corpus.sqlite3') as connection:
        table = connection.event_table
        downloads = connection.execute("SELECT file_id,version_id,presence FROM current_file_state WHERE kind='download_payload'").fetchall()
        assert len(downloads) == 3 and {row[2] for row in downloads} == {'deleted'}
        deleted_at = {row[1]: max(d['event_id'] for d in retention_facts(connection, row[0])['deletions']) for row in downloads}
        starts = {version: event for version, event in connection.execute(
            f"SELECT t.version_id,MIN(e.event_id) FROM tasks t JOIN {table} e USING(task_id) WHERE t.operation='download.acquire' "
            "AND e.reason_code='stage_started' GROUP BY t.version_id")}
        last_extract = {version: event for version, event in connection.execute(
            f"SELECT t.version_id,MAX(e.event_id) FROM tasks t JOIN {table} e USING(task_id) WHERE t.operation='extract.document' "
            "AND e.event_kind='lifecycle' AND e.status='ok' GROUP BY t.version_id")}
    order = sorted(starts, key=starts.get)
    # Streaming (owner design): the first download is gone before the last one starts, and each after its reader.
    assert deleted_at[order[0]] < starts[order[-1]]
    assert all(deleted_at[version] > last_extract[version] for version in deleted_at)


def test_only_document_extraction_names_a_publication_source():
    """K4C2-G2: an OCR run's extract.page_quality step is an extract-stage operation, not a publication source."""
    import inspect
    from glossapi.stages import run
    source = inspect.getsource(run.prepare_streaming)
    assert "operation.specification.operation=='extract.document' and operation.document.version_id in completed_versions" in source
