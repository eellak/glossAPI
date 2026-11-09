"""W18 M13: every corpus-wide barrier the run executes is previewed by plan --json and documented (K4(b))."""
from pathlib import Path
import threading

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import open_readonly

pytest.importorskip('aiohttp')
pytest.importorskip('glossapi_rs_cleaner')
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def test_documentation_table_is_generated_from_the_barrier_table():
    from glossapi.stages.barriers import markdown_table, BARRIERS
    text = (REPO/'docs/architecture/corpus_barriers.md').read_text()
    assert markdown_table() in text
    assert all(row['why'] and row['continues'] for row in BARRIERS)
    # The table describes the implementation: annotation partitions over PyArrow (annotation_partitions); Dask is
    # only a pinned parity-test oracle (FINAL-ASTRA F09).
    assert not any('dask' in str(value).lower() for row in BARRIERS for value in row.values())
    annotation = next(row for row in BARRIERS if row['name'] == 'annotation_batch')
    assert 'PyArrow partitions' in str(annotation['why'])


def test_every_executed_barrier_is_previewed_and_documented(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from glossapi.cli import plan_documents
    from glossapi.stages.barriers import BARRIERS
    from test_ledger_fixset_lg import pdf
    remote = pdf(tmp_path/'pdfs/remote.pdf', 'REMOTE text body').read_bytes()
    local = pdf(tmp_path/'in/local.pdf', 'LOCAL text body')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header('Content-Length', str(len(remote))); self.end_headers(); self.wfile.write(remote)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/remote.pdf'
        manifest = {'manifest_version': 1, 'documents': [{'url': url}, {'path': str(local)}]}
        stages = ['extract', 'clean', 'sections', 'annotate', 'export']
        settings = {'download': {'rate_limit': 0}, 'extract': {'repair_fonts': False}}
        plan = plan_documents(tmp_path/'out', manifest=manifest, stages=stages, settings=settings)
        result = Corpus(tmp_path/'empty', tmp_path/'out').run(plan=plan)
    finally:
        server.shutdown(); server.server_close(); thread.join()
    assert result['state'] == 'completed', result
    with open_readonly(tmp_path/'out/corpus.sqlite3') as connection:
        executed = [row[0] for row in connection.execute(
            "SELECT operation FROM tasks WHERE run_id=? AND operation LIKE '%.compatibility_publish' ORDER BY ordinal", (result['run_id'],))]
        sealed = connection.execute(f"SELECT COUNT(*) FROM {connection.event_table} WHERE run_id=? AND reason_code='waiting_for_corpus_seal'",
                                    (result['run_id'],)).fetchone()[0]
    assert sealed == len(executed)
    previewed = [barrier['executed_as'] for barrier in plan['barriers']]
    assert previewed[0] == 'freeze' and previewed[1:] == executed, (previewed, executed)
    documented = {row['executed_as'] for row in BARRIERS}
    assert set(executed) <= documented
    text = (REPO/'docs/architecture/corpus_barriers.md').read_text()
    assert all(f'`{operation}`' in text for operation in executed)
    assert all(barrier['why'] and barrier['continues'] for barrier in plan['barriers'])
