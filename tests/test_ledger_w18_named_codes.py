"""W18 M14: the reason codes the HTTP dry run emitted but no test named (K4(b) prep), each asserted where it is written."""
import threading

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

pytest.importorskip('aiohttp')
pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def emitted(root):
    with open_readonly(root/'corpus.sqlite3') as connection:
        return {(row[0], row[1]) for row in connection.execute(f'SELECT DISTINCT operation,reason_code FROM {connection.event_table} e '
                                                                'LEFT JOIN tasks t USING(task_id)')}


def serve(body):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_download_export_inventory_and_corpus_seal_codes(tmp_path):
    from test_ledger_fixset_lg import pdf
    server = serve(pdf(tmp_path/'pdfs/a.pdf', 'Named codes body').read_bytes())
    try:
        root = tmp_path/'out'
        result = Corpus(tmp_path/'empty', root).run([f'http://127.0.0.1:{server.server_port}/a.pdf'], stages=('extract', 'clean', 'export'),
                                                    settings={'download': {'rate_limit': 0}, 'extract': {'repair_fonts': False}})
    finally:
        server.shutdown(); server.server_close()
    assert result['state'] == 'completed'
    codes = emitted(root)
    assert ('download.acquire', 'download_completed') in codes
    assert ('export.compatibility_publish', 'export_completed') in codes
    assert (None, 'source_inventory_bound') in codes
    assert ('export.compatibility_publish', 'waiting_for_corpus_seal') in codes


def test_clean_exception_names_the_failed_publication(tmp_path, monkeypatch):
    from glossapi.corpus.phase_clean import CleanPhaseMixin
    from test_ledger_fixset_lg import pdf
    def broken(self, *args, **kwargs):
        raise ValueError('cleaner refused this corpus')
    monkeypatch.setattr(CleanPhaseMixin, 'clean', broken)
    root = tmp_path/'out'
    result = Corpus(tmp_path/'empty', root).run([pdf(tmp_path/'in/a.pdf', 'Clean exception body')], stages=('extract', 'clean', 'export'),
                                                settings={'extract': {'repair_fonts': False}})
    assert result['state'] == 'completed_with_failures'
    assert ('clean.compatibility_publish', 'clean_exception') in emitted(root)


def test_waiting_for_disk_when_no_attempt_fits_the_budget(tmp_path, monkeypatch):
    from pathlib import Path
    from glossapi.stages import disk_budget
    from test_ledger_fixset_lg import pdf
    monkeypatch.setattr(disk_budget, 'require_disk_guard', lambda root, settings: (Path(root).resolve()/'.glossapi/staging').mkdir(parents=True, exist_ok=True))
    root = tmp_path/'out'
    result = Corpus(tmp_path/'empty', root).run([pdf(tmp_path/'in/a.pdf', 'Budget body')], stages=('extract', 'clean'),
        settings={'memory_saving': 'extreme', 'disk_budget_bytes': 1024, 'max_source_bytes': 4096, 'max_attempt_bytes': 4096,
                  'extract': {'repair_fonts': False}})
    assert result['state'] == 'waiting'
    assert ('extract.document', 'waiting_for_disk') in emitted(root)
