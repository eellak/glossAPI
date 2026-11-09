"""W18 M2: HTTP and native attempt diagnostics follow the debug setting through Corpus.run (K4(b))."""
import json
import socket
import threading

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

pytest.importorskip('aiohttp')
pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


class CutOnce:
    """The first GET announces the full length and closes after half the body; later GETs serve it whole."""
    def __init__(self, body):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.body, self.gets = body, 0
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/a.pdf':
                    self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
                outer.gets += 1
                self.send_response(200); self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Length', str(len(outer.body))); self.end_headers()
                if outer.gets == 1:
                    self.wfile.write(outer.body[:len(outer.body)//2]); self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                self.wfile.write(outer.body)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start(); return f'http://127.0.0.1:{self.server.server_port}/a.pdf'

    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


@pytest.mark.parametrize('debug', ['off', 'on'])
def test_http_and_native_diagnostics_follow_the_debug_setting(tmp_path, debug):
    from glossapi.ledger.reader import report_why, select_run
    from test_ledger_fixset_lg import pdf
    body = pdf(tmp_path/'pdfs/a.pdf', 'Diagnostics body text that is long enough to be cut in half').read_bytes()
    root = tmp_path/'out'
    with CutOnce(body) as url:
        result = Corpus(tmp_path/'empty', root).run([url], stages=('extract', 'clean'),
                                                    settings={'debug': debug, 'download': {'rate_limit': 0}, 'extract': {'repair_fonts': False}})
    assert result['state'] == 'completed', result
    with open_readonly(root/'corpus.sqlite3') as connection:
        rows = connection.execute("SELECT file_id,kind,presence,uri,sha256,bytes FROM current_file_state").fetchall()
        bodies = [row for row in rows if '/request_bodies/' in row[3]]
        assert bodies, 'the cut first response left no retained request body'
        why = report_why(connection, url, run=select_run(connection, result['run_id']))
        listed = {item['file_id'] for item in why['files']}
        native = [row for row in rows if row[3].endswith('stage2_detailed_table_report.csv')]
        if debug == 'on':
            assert all(row[1] == 'attempt_diagnostic' and row[2] == 'present' for row in bodies)
            assert {row[0] for row in bodies} <= listed
            assert native and all(row[2] == 'present' for row in native)
            # Census: every regular file under .glossapi/artifacts is a present ledger row, and every present row exists.
            private = root/'.glossapi/artifacts'
            on_disk = {path.as_uri() for path in private.rglob('*') if path.is_file()}
            present = {row[3] for row in rows if row[2] == 'present' and row[3].startswith(private.as_uri())}
            assert on_disk == present, (sorted(on_disk - present)[:5], sorted(present - on_disk)[:5])
        else:
            assert all(row[2] == 'deleted' for row in bodies), bodies
            # The receipt still names the released body's hash and size.
            receipts = [row for row in rows if row[1] == 'worker_receipt' and row[2] == 'present']
            named = any(any(str(row[5]) in path.read_text() and (row[4] in path.read_text() or 'request0.part' in path.read_text())
                            for path in [__import__('pathlib').Path(__import__('glossapi.stages.scheduler', fromlist=['local_path']).local_path(r[3])) for r in receipts])
                        for row in bodies)
            assert named, 'no receipt names the released request body'
