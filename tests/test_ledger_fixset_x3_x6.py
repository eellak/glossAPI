"""Fix set X3-X6: run selection, report --why, math availability, log
ownership, request-body retention, disk peaks, retention ownership and
external-plugin devices. (Its credential-redaction part was withdrawn with the
credential channel, owner 1 Nov 2026.)"""
import asyncio
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from glossapi.ledger import LedgerWriter


def cli(*arguments):
    from glossapi.cli import main
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(list(arguments))
    return code, json.loads(buffer.getvalue())


def run_pdfs(tmp_path, sources, **settings):
    from glossapi import Corpus
    corpus = Corpus(tmp_path/'empty', tmp_path/'output')
    result = corpus.run(sources, stages=('extract', 'clean', 'export'),
                        settings={'extract': {'repair_fonts': False}, **settings})
    return corpus, result


# ---------------------------------------------------------------- X3 -------

def _children(connection):
    return {json.loads(row[1])['wrapper_run_id']: row[0] for row in connection.execute(
        "SELECT run_id,details_json FROM events WHERE event_kind='run' AND message LIKE 'wrapper:%'")}


def test_status_and_report_default_to_the_users_streaming_run(tmp_path):
    from test_ledger_stages import simple_pdf
    from glossapi.ledger.reader import select_run
    from glossapi.ledger.writer import open_readonly
    corpus, result = run_pdfs(tmp_path, [simple_pdf(tmp_path/'source.pdf')])
    assert result['state'] == 'completed'
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
        children = _children(connection)
        assert children and set(children.values()) == {result['run_id']}
        latest = connection.execute('SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1').fetchone()[0]
        assert latest in children  # the old "latest row" rule picked this wrapper
        assert select_run(connection)['run_id'] == result['run_id']
        assert select_run(connection, latest)['run_id'] == latest  # explicit child stays readable
    for command in ('status', 'report'):
        code, envelope = cli(command, '--corpus', str(corpus.output_dir))
        assert code == 0 and envelope['run_id'] == result['run_id'], envelope['errors']
    # A wrapper the user calls directly is a user run, not a hidden child.
    corpus.clean(num_threads=1)
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
        direct = connection.execute('SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1').fetchone()[0]
        assert direct not in _children(connection)
        assert select_run(connection)['run_id'] == direct


def test_read_interfaces_carry_no_obsolete_shadow_labels(tmp_path):
    from test_ledger_stages import simple_pdf
    from glossapi.ledger.reader import plan_view
    corpus, result = run_pdfs(tmp_path, [simple_pdf(tmp_path/'source.pdf')])
    code, plan = cli('plan', '--corpus', str(corpus.output_dir), '--stages', 'export,math')
    assert code == 0
    body = plan['data']
    assert 'execution_available' not in body and 'compatibility_profile' not in body
    assert body['execution']['profile'] == 'streaming_document_dag'
    assert body['execution']['available'] is False
    assert body['execution']['unavailable_stages'] == {'math': 'math_backend_unavailable'}
    steps = {step['stage']: step for step in body['documents'][0]['steps']}
    assert steps['math']['executable_by_run'] is False and steps['export']['executable_by_run'] is True
    assert plan_view(None, selectors=['https://example.org/a.pdf'])['execution']['available'] is True
    from glossapi.ledger.writer import open_readonly
    with open_readonly(corpus.output_dir/'corpus.sqlite3') as connection:
        wrapper = next(iter(_children(connection)))
    for arguments in (('status',), ('report',), ('status', '--run', wrapper), ('report', '--run', wrapper)):
        _, envelope = cli(*arguments, '--corpus', str(corpus.output_dir))
        assert 'shadow' not in json.dumps(envelope['data']), arguments


def test_report_why_explains_a_really_failed_document(tmp_path):
    from test_ledger_stages import simple_pdf
    bad = tmp_path/'bad.pdf'
    bad.write_bytes(b'%PDF-1.4\nnot a real pdf body\n')
    corpus, result = run_pdfs(tmp_path, [simple_pdf(tmp_path/'good.pdf'), bad])
    assert result['state'] == 'completed_with_failures'
    code, envelope = cli('report', '--corpus', str(corpus.output_dir), '--why', str(bad))
    assert code == 0 and envelope['run_id'] == result['run_id']
    stages = {stage['operation']: stage for stage in envelope['data']['stages']}
    extract = stages['extract.document']
    assert extract['status'] == 'failed'
    error = extract['attempts'][-1]['error']
    assert error['type'].endswith('PdfiumError') and 'Failed to load document' in error['message']
    assert error['basis'] == 'recorded_backend_observation'
    assert stages['clean.initial']['status'] == 'skipped'
    assert 'upstream_failed' in [reason['reason_code'] for reason in stages['clean.initial']['reasons']]
    assert not any(item['decision'] == 'include' for item in envelope['data']['inclusion'])
    assert bad.read_bytes() == b'%PDF-1.4\nnot a real pdf body\n'  # supplied original untouched


def test_report_why_explains_an_unavailable_source(tmp_path):
    from glossapi import Corpus
    url = 'http://127.0.0.1:9/missing.pdf'  # loopback discard port: refused, no network
    corpus = Corpus(tmp_path/'empty', tmp_path/'output')
    result = corpus.run([url], stages=('extract', 'clean', 'export'),
        settings={'download': {'max_retries': 1, 'pre_ping_domains': False, 'sleep': 0, 'request_timeout': 5}})
    assert result['state'] == 'completed_with_failures'
    code, envelope = cli('report', '--corpus', str(corpus.output_dir), '--why', url)
    assert code == 0 and envelope['run_id'] == result['run_id']
    stages = {stage['operation']: stage for stage in envelope['data']['stages']}
    download = stages['download.acquire']
    assert download['status'] == 'failed'
    assert 'download_retry_exhausted' in [reason['reason_code'] for reason in download['reasons']]
    assert 'ClientConnector' in download['attempts'][-1]['error']['type']
    assert stages['extract.document']['status'] == 'skipped'
    assert 'upstream_failed' in [reason['reason_code'] for reason in stages['extract.document']['reasons']]
    assert any(item['reason_code'] == 'export_canonical_markdown_missing' and item['decision'] == 'export_canonical_markdown_missing'
               for item in envelope['data']['inclusion'])
    assert envelope['data']['document']['source_records'][0]['source_uri'] == url


def test_report_why_after_intermediates_and_outputs_are_gone(tmp_path):
    from test_ledger_stages import simple_pdf
    source = simple_pdf(tmp_path/'source.pdf')
    corpus, result = run_pdfs(tmp_path, [source], debug='off')
    assert result['state'] == 'completed'
    clean = corpus.cleaned_markdown_dir/'source.md'
    assert clean.is_file()
    clean.unlink()  # an external deletion after the run; the ledger is the only authority
    for selector in (str(source), str(corpus.output_dir/'markdown/source.md')):
        code, envelope = cli('report', '--corpus', str(corpus.output_dir), '--why', selector)
        assert code == 0 and envelope['run_id'] == result['run_id'], envelope['errors']
        data = envelope['data']
        assert {stage['status'] for stage in data['stages'] if stage['scope'] == 'document'} == {'ok'}
        deleted = [item for item in data['files'] if item['presence'] == 'deleted']
        assert deleted and all(item['retention']['deletions'] and item['sha256'] for item in deleted)
        assert any(item['kind'] == 'raw_markdown' for item in deleted)
        assert any(item['decision'] == 'include' and item['record_hash'] for item in data['inclusion'])
    assert source.is_file()


class _MissingLegacyCodeFormula:
    """Docling is installed, but its 2.81 tree has no legacy CodeFormula module."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'docling.models.code_formula_model':
            raise ModuleNotFoundError("No module named 'docling.models.code_formula_model'", name=fullname)
        return None


def _stub_docling_without_legacy_module(monkeypatch):
    for name in ('docling', 'docling.datamodel', 'docling.models'):
        module = ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    accelerator = ModuleType('docling.datamodel.accelerator_options')
    accelerator.AcceleratorOptions = object; accelerator.AcceleratorDevice = SimpleNamespace(AUTO='auto')
    base = ModuleType('docling.datamodel.base_models'); base.ItemAndImageEnrichmentElement = object
    monkeypatch.setitem(sys.modules, 'docling.datamodel.accelerator_options', accelerator)
    monkeypatch.setitem(sys.modules, 'docling.datamodel.base_models', base)
    monkeypatch.delitem(sys.modules, 'docling.models.code_formula_model', raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_MissingLegacyCodeFormula(), *sys.meta_path])


def _math_request(tmp_path):
    from test_ledger_text_stages import stage_request
    source = tmp_path/'source.pdf'; source.write_bytes(b'%PDF-stub')
    document = tmp_path/'source.docling.json'; document.write_text('{}')
    return stage_request(tmp_path, 'math.infer', [('source_payload', source), ('docling_json', document)])


def test_math_reports_explicit_unavailable_backend_not_a_layout_install_hint(tmp_path, monkeypatch):
    pytest.importorskip('docling_core')
    from glossapi.stages.execution import ReceiptCollector
    from glossapi.stages.math import MathStage
    _stub_docling_without_legacy_module(monkeypatch)
    result = MathStage().run_document(_math_request(tmp_path), reporter=ReceiptCollector())
    assert result.status == 'failed' and result.reason_code == 'math_backend_unavailable'
    assert 'math_backend_unavailable' in result.error.message
    assert 'docling.models.code_formula_model' in result.error.message
    assert "glossapi[layout]" not in result.error.message
    assert result.decisions[0]['inference_performed'] is False
    assert result.decisions[0]['backend_status'] == 'unavailable'


def test_math_real_docling_281_is_explicitly_unavailable(tmp_path):
    """Runs in the qualified Docling venv; skipped where Docling is absent."""
    pytest.importorskip('docling_core')
    pytest.importorskip('docling')
    import importlib.util
    if importlib.util.find_spec('docling.models.code_formula_model') is not None:
        pytest.skip('legacy CodeFormula module present; this gate is for Docling 2.81')
    from glossapi.ocr.math.enrich import enrich_from_docling_json
    from glossapi.stages.execution import ReceiptCollector
    from glossapi.stages.math import MathStage
    with pytest.raises(ImportError) as caught:
        enrich_from_docling_json(*[tmp_path/'unused']*4, device='cpu')
    assert 'math_backend_unavailable' in str(caught.value) and 'glossapi[layout]' not in str(caught.value)
    result = MathStage().run_document(_math_request(tmp_path), reporter=ReceiptCollector())
    assert result.reason_code == 'math_backend_unavailable' and result.decisions[0]['inference_performed'] is False


def test_injected_math_kernel_does_not_count_as_real_inference(tmp_path, monkeypatch):
    import glossapi.ocr.math as math_package
    from glossapi.stages.execution import ReceiptCollector
    from glossapi.stages.math import MathStage
    def injected(document, source, markdown, latex_map, **options):
        markdown.write_text('MATH')
        latex_map.write_text(json.dumps({'page_no': 1, 'item_index': 1, 'latex': 'x', 'accept_score': 1})+'\n')
        return {'items': 1, 'accepted': 1, 'time_sec': 0.0}
    monkeypatch.setitem(math_package.__dict__, 'enrich_from_docling_json', injected)
    result = MathStage().run_document(_math_request(tmp_path), reporter=ReceiptCollector())
    assert result.status == 'ok' and result.reason_code == 'math_completed'
    assert result.decisions[0]['inference_performed'] is False
    assert result.decisions[0]['backend']['real_inference'] is False


# ---------------------------------------------------------------- X4 -------

def test_second_downloader_cannot_append_to_a_sealed_request_log(tmp_path):
    from glossapi.gloss_downloader import GlossDownloader
    first = GlossDownloader(output_dir=tmp_path/'first', pre_ping_domains=False)
    first.logger.info('first corpus request')
    first.progress_logger.info('first corpus progress')
    for handler in [*first.logger.handlers, *first.progress_logger.handlers]:
        handler.flush()
    sealed = {name: (tmp_path/'first/logs'/name).read_bytes() for name in ('download_request.logs', 'download_progress.logs')}
    second = GlossDownloader(output_dir=tmp_path/'second', pre_ping_domains=False)
    second.logger.warning('second corpus request')
    second.progress_logger.warning('second corpus progress')
    for handler in [*second.logger.handlers, *second.progress_logger.handlers]:
        handler.flush()
    for name, body in sealed.items():
        assert (tmp_path/'first/logs'/name).read_bytes() == body, name
    assert b'second corpus request' in (tmp_path/'second/logs/download_request.logs').read_bytes()
    shared = [logging.getLogger('glossapi.gloss_downloader'), logging.getLogger('glossapi.gloss_downloader.progress')]
    assert not any(isinstance(handler, logging.FileHandler) for logger in shared for handler in logger.handlers)
    second.close_logs()
    assert not second.logger.handlers and not second.progress_logger.handlers
    first.close_logs()


class _Content:
    def __init__(self, chunks, error):
        self.chunks, self.error = chunks, error
    async def iter_chunked(self, size):
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


class _Response:
    def __init__(self, url, chunks, error=None):
        self.url, self.status, self.history = url, 200, ()
        self.headers = {'Content-Type': 'application/pdf'}
        self.content = _Content(chunks, error)
    def raise_for_status(self):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, url, bodies):
        self.url, self.bodies = url, list(bodies)
    def get(self, url, **options):
        chunks, error = self.bodies.pop(0)
        return _Response(url, chunks, error)


def test_inner_retry_keeps_earlier_request_body_with_receipt_link(tmp_path):
    aiohttp = pytest.importorskip('aiohttp')
    from glossapi.gloss_downloader import GlossDownloader
    from glossapi.stages.receipts import capture_transport
    url = 'https://example.invalid/source.pdf'
    first_partial = b'%PDF-1.4\nfirst request partial body'
    final = b'%PDF-1.4\n' + b'y'*1000 + b'%%EOF'
    downloader = GlossDownloader(output_dir=tmp_path, pre_ping_domains=False, max_retries=2, sleep=0)
    session = _Session(url, [([first_partial], aiohttp.ClientPayloadError('stream cut')), ([final], None)])
    async def acquire():
        return await downloader._download_via_streaming_get(session=session, row_index=3, url=url, headers={},
            timeout=None, retry_count=0, filename_base='ABC_123', referer=None)
    with capture_transport('standard', {}) as transport:
        returned = asyncio.run(acquire())
    assert returned[0] is True and returned[1] == 'ABC_123.pdf'
    assert (tmp_path/'downloads/ABC_123.pdf').read_bytes() == final
    assert sorted(path.name for path in (tmp_path/'downloads').iterdir()) == ['ABC_123.pdf']
    retained = tmp_path/'request_bodies/3.request0.part'
    assert retained.read_bytes() == first_partial
    requests = transport['requests']
    assert [item['retry_index'] for item in requests] == [0, 1]
    assert requests[0]['retained_body'] == {'relative_path': 'request_bodies/3.request0.part',
        'bytes': len(first_partial), 'superseded_by_sequence': 1}
    assert 'retained_body' not in requests[1]
    from glossapi.stages.execution import outputs_in
    kinds = {output.relative_path: output.kind for output in outputs_in(tmp_path)}
    assert kinds['request_bodies/3.request0.part'] == 'attempt_diagnostic'
    assert kinds['downloads/ABC_123.pdf'] == 'download_payload'
    downloader.close_logs()


def test_disk_high_water_survives_the_final_boundary(tmp_path):
    from test_ledger_stages import simple_pdf
    from glossapi.stages.disk_budget import DiskBudget
    corpus, result = run_pdfs(tmp_path, [simple_pdf(tmp_path/'source.pdf')], debug='off')
    disk = result['disk']
    assert disk['high_water_bytes'] > 0
    # Retention released intermediates after the peak: the peak must not be
    # replaced by the smaller post-retention present bytes (old: 0).
    assert disk['deleted_logical_bytes'] > 0 and disk['high_water_bytes'] > disk['present_bytes']
    with LedgerWriter(corpus.output_dir/'corpus.sqlite3') as writer:
        persisted = [json.loads(row[0]).get('disk_high_water_bytes') for row in writer.connection.execute(
            "SELECT details_json FROM events WHERE run_id=? AND event_kind='run'", (result['run_id'],))]
        assert max(value for value in persisted if value is not None) == disk['high_water_bytes']
        # A later fresh observer (e.g. a status/resume boundary) never resets the peak to zero.
        fresh = DiskBudget(writer, corpus.output_dir, result['run_id'], {'debug': 'off'}).report()
        assert fresh['high_water_bytes'] == disk['high_water_bytes'] > fresh['present_bytes']


# ---------------------------------------------------------------- X5 -------

def _settings(debug, memory_saving):
    return dict(debug=debug, memory_saving=memory_saving, disk_budget_bytes=4*1024**3,
                max_source_bytes=1024**2, max_attempt_bytes=4*1024**2)


def _retention_case(writer, root, debug, memory_saving, *, aliases=True):
    """One remote document read by extract; five candidate files of its source hash."""
    body = b'%PDF-original-entity'
    digest = __import__('hashlib').sha256(body).hexdigest()
    outside = root.parent/(root.name+'-outside'); outside.mkdir()
    with writer.transaction():
        foreign_run = writer.run(_settings('off', 'normal'), {}, {}, code_revision='other-run')
        document = writer.document('https://example.org/source.pdf')
        version = writer.version(document, source_sha256=digest, stored_sha256=digest)
        operations = [{'stage': stage, 'document': {'version_id': version},
                       'specification': {'input_roles': [] if stage == 'download' else ['source_payload']}, 'inputs': []}
                      for stage in ('download', 'extract')]
        run = writer.run(_settings(debug, memory_saving), {'operations': operations}, {}, code_revision='test')
        tasks = [writer.task(run, stage, stage+'.test', {}, version_id=version, ordinal=index)
                 for index, stage in enumerate(('download', 'extract'))]
        files = {}
        def register(name, path, *, owner=run, kind='download_payload', ownership='managed', retention='initial'):
            if not path.exists() and not path.is_symlink():
                path.write_bytes(body)
            files[name] = (writer.artifact(owner, 'download', kind, path.as_uri(), version_id=version, sha256=digest,
                bytes=len(body), ownership=ownership, retention_class=retention), path)
        register('owned_download', root/'owned.pdf')
        register('foreign_run_download', root/'foreign.pdf', owner=foreign_run)
        supplied = root/'supplied.pdf'
        register('supplied_original', supplied, kind='source_payload', ownership='initial_external')
        target = outside/'linked-target.pdf'; target.write_bytes(body)
        (outside/'via-dir.pdf').write_bytes(body)
        if aliases:
            (root/'linked.pdf').symlink_to(target)
            register('symlink_alias', root/'linked.pdf')
            (root/'alias-dir').symlink_to(outside, target_is_directory=True)
            register('directory_alias', root/'alias-dir'/'via-dir.pdf')
        for task in tasks:
            writer.event(run, event_kind='lifecycle', reason_code='stage_planned', status='planned', task_id=task, version_id=version)
    with writer.transaction():
        attempt = writer.attempt(tasks[1], worker_id='reader', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        for name, (file_id, _) in files.items():
            writer.input(attempt, file_id, 'source_payload:'+name, digest)
        writer.finish(run, tasks[1], attempt, status='ok', reason_code='extract_completed')
    return run, tasks, files, {'target': target, 'via_dir': outside/'via-dir.pdf'}


@pytest.mark.parametrize('debug', ['off', 'on'])
@pytest.mark.parametrize('memory_saving', ['normal', 'extreme'])
def test_retention_contract_by_ownership_for_all_four_settings(tmp_path, debug, memory_saving):
    from glossapi.stages.retention import RetentionController, retention_facts, retire_downloaded_sources
    root = tmp_path/'corpus'; root.mkdir()
    with LedgerWriter(root/'corpus.sqlite3') as writer:
        run, tasks, files, outside = _retention_case(writer, root, debug, memory_saving)
        retire_downloaded_sources(writer, root, run, _settings(debug, memory_saving))
        controller = RetentionController(writer, root, run, _settings(debug, memory_saving))
        for name, (file_id, _) in files.items():
            if not retention_facts(writer.connection, file_id)['declaration']:
                controller.declare(file_id, [tasks[1]], source=True)
            controller.delete(file_id)
        expected_deleted = {'owned_download'} if memory_saving == 'extreme' else set()
        deleted = {name for name, (_, path) in files.items() if not path.exists() and not path.is_symlink()}
        assert deleted == expected_deleted
        assert outside['target'].read_bytes() == outside['via_dir'].read_bytes() == b'%PDF-original-entity'
        assert (root/'linked.pdf').is_symlink() and (root/'alias-dir').is_symlink()
        pins = {name: controller.eligibility(file_id)[2] for name, (file_id, _) in files.items()}
        assert 'external_original' in pins['supplied_original']
        if memory_saving == 'normal':
            assert all('initial_source_normal' in pins[name] for name in ('owned_download', 'foreign_run_download'))
        else:
            assert pins['owned_download'] == ['already_deleted']
            assert 'foreign_run_source' in pins['foreign_run_download']


@pytest.mark.parametrize('debug', ['off', 'on'])
def test_extreme_never_deletes_another_runs_download_of_the_same_bytes(tmp_path, debug):
    from glossapi.stages.retention import retire_downloaded_sources
    root = tmp_path/'corpus'; root.mkdir()
    with LedgerWriter(root/'corpus.sqlite3') as writer:
        run, tasks, files, _ = _retention_case(writer, root, debug, 'extreme', aliases=False)
        retire_downloaded_sources(writer, root, run, _settings(debug, 'extreme'))
        assert not files['owned_download'][1].exists()
        assert files['foreign_run_download'][1].read_bytes() == b'%PDF-original-entity'
        assert files['supplied_original'][1].read_bytes() == b'%PDF-original-entity'


# ---------------------------------------------------------------- X6 -------

@pytest.mark.parametrize('accelerator', ['mps', 'cuda:0'])
def test_external_plugin_without_device_runs_on_cpu_on_accelerator_hosts(tmp_path, monkeypatch, accelerator):
    import pandas as pd
    from glossapi import Corpus
    from glossapi.ocr import defaults
    from test_ledger_text_stages import plugin_fixture
    dispatch, pages = plugin_fixture(monkeypatch, pages=1)
    monkeypatch.setattr(defaults, 'select_device', lambda device=None: device or accelerator)
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'sequential-adapters')
    corpus = Corpus(tmp_path/'input', tmp_path/'output')
    corpus.input_dir.mkdir(exist_ok=True)
    (corpus.input_dir/'source.pdf').write_bytes(b'%PDF-stub')
    pages._write_outputs(corpus.output_dir, 'source', 'OLD_NATIVE_SENTINEL\n\nOld native body.', 1)
    corpus.clean(drop_bad=False, num_threads=1)
    metadata = corpus._get_cached_metadata_parquet()
    frame = pd.read_parquet(metadata); frame['needs_ocr'] = True; frame['ocr_pages'] = ''; frame.to_parquet(metadata, index=False)
    corpus.ocr(mode='ocr_bad', ocr_options=SimpleNamespace(kind='stub-plugin'))
    assert 'STUB OCR 1' in (corpus.cleaned_markdown_dir/'source.md').read_text()
    # An explicit accelerator for an external engine is still rejected, not replaced.
    with pytest.raises((ValueError, RuntimeError), match='Device is not qualified'):
        corpus.ocr(mode='ocr_bad', ocr_options=SimpleNamespace(kind='stub-plugin'), device=accelerator, reprocess_completed=True)
