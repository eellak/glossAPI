"""K4g fix round 5: corpus-level products independent of the completion order; latin_percentage on the exported text.

1. The node's K4g final gates on aee0ae8 (kill matrix, worker-ENOSPC and full-disk-inside-clean arms) still found four
   corpus-level products that differ after a retried document, with every document's own products equal:
   - ``sections_for_annotation``, ``classified_sections`` and ``fully_annotated_sections``: the same rows per document
     in another document order, so every sequential ``id`` / ``row_id`` differed. The section publication enumerated
     the texts in ``os.listdir`` order, which on tmpfs is the order they were written (newest first): the retried
     documents came first. Fix: one canonical order, the sorted file names (the document export's order).
   - ``download_results``: equal values and Arrow types, but the pandas metadata gave ``latin_word_shape_rate``
     numpy_type float64 instead of object (a first pass creates the column as object; a repeated clean read it back
     from the parquet as float64 and filled it in place). Fix: the column starts as the first pass's object column.
   The retried runs below use a tmpfs-like listing (newest entry first, in every process of both runs), so they fail
   on aee0ae8 on any filesystem; an English fixture gives latin_word_shape_rate a measured value.
2. X2 (aee0ae8 vs the f1eb02cc receipt): ``latin_percentage`` after OCR is now the cleaner's own value (fix 4), which
   is the definition every plan uses (ASCII Latin letters / non-whitespace characters of the cleaned text, the same
   pass and surface as ``percentage_greek``); the old value was the OCR-profile value of the text before OCR artefact
   cleaning, kept by the duplicate-column rule. But the score-only pass counted the OCR supplement's composition
   lines, which FG-G02 excludes from text metrics. Fix: the score-only pass measures a marker-free surface.
3. The supplied-file metadata rule (K4BI-G9): every ``--files`` source gets null ``url`` / ``download_success`` /
   ``extract_error``, also the sixth, which lands in the extraction's second batch (on 5cedabc it alone got
   ``''`` / False / ``''``; fixed by fix 4, pinned here).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from test_pdf_font_repair import greek_fixture  # noqa: F401  (module fixture)
from test_k4g_fix4 import CHILD, PDFS, SCAN, assert_same_products, products

REPO = Path(__file__).resolve().parents[1]
CORPUS_PARQUETS = ('sections/sections_for_annotation.parquet', 'classified_sections.parquet',
                   'fully_annotated_sections.parquet', 'download_results/download_results.parquet')
ENGLISH = ['The archive keeps printed reports and articles for readers and students.',
           'Every document has a title, an author, a date and several numbered sections.',
           'Readers search the catalogue, open a record and download the original file.',
           'Librarians check each scan, correct the metadata and publish the collection.',
           'Some reports describe engineering projects, energy systems and transport networks.',
           'Other articles discuss history, language teaching, music and modern literature.',
           'The service records how often each document is read and which pages are printed.',
           'Volunteers transcribe handwritten letters so that the text can be searched later.']

FAULT = textwrap.dedent('''
    import os
    if os.environ.get('K4G5_LISTDIR') == 'tmpfs':
        # Linux tmpfs lists the newest directory entry first: the order the files were written (or renamed into place).
        _real_listdir = os.listdir
        def _tmpfs_listdir(path='.'):
            names = _real_listdir(path)
            try:
                base = os.fsdecode(os.fspath(path))
                return sorted(names, key=lambda name: (-os.stat(os.path.join(base, os.fsdecode(name)), follow_symlinks=False).st_ctime_ns, name))
            except Exception:
                return names
        os.listdir = _tmpfs_listdir
    if os.environ.get('GLOSSAPI_TEST_STUB_OCR') == '1':
        from test_ocr_same_basename import STUB
        exec(STUB)
    STAGE = os.environ.get('K4G4_FAULT_STAGE')
    if STAGE in ('extract', 'clean'):
        import errno, importlib
        from pathlib import Path
        owner = getattr(importlib.import_module('glossapi.stages.' + STAGE), {'extract': 'ExtractStage', 'clean': 'CleanStage'}[STAGE])
        name = 'run_document' if STAGE == 'clean' else '_run_document'
        original = getattr(owner, name)
        def wrapped(self, request, *args, **kwargs):
            marker = Path(os.environ['K4G4_MARKER'])
            if any(os.environ['K4G4_TARGET'] in Path(str(getattr(item, 'path', item))).name for item in request.inputs) \\
                    and not marker.exists():
                marker.write_text('failed once')
                raise OSError(errno.ENOSPC, 'No space left on device (injected worker write failure)')
            return original(self, request, *args, **kwargs)
        setattr(owner, name, wrapped)
''')


def english_pdf(path: Path, lines=ENGLISH) -> Path:
    """A born-digital page of plain English (Helvetica, ASCII): >= 20 words of >= 3 letters, so the raw-text
    latin_word_shape_rate is measured (not null)."""
    body = ' '.join(f'({line}) Tj T*' for line in lines)
    stream = f'BT /F1 11 Tf 14 TL 50 780 Td {body} ET'.encode('ascii')
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [4 0 R] /Count 1 >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>',
               b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream']
    payload = b'%PDF-1.4\n'; offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(payload)); payload += str(number).encode() + b' 0 obj\n' + obj + b'\nendobj\n'
    xref = len(payload)
    payload += f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode() + b''.join(f'{v:010} 00000 n \n'.encode() for v in offsets)
    payload += f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n'.encode() + str(xref).encode() + b'\n%%EOF\n'
    path.write_bytes(payload)
    return path


def run_case(root: Path, sources, *, stages, seed=False, fault=None, target='', listdir=None, steps=('start',)):
    (root / 'fault').mkdir(parents=True)
    (root / 'fault' / 'sitecustomize.py').write_text(FAULT)
    (root / 'child.py').write_text(CHILD)
    env = {**os.environ, 'K4G4_MARKER': str(root / 'failed-once'), 'K4G4_TARGET': target, 'K4G4_STAGES': ','.join(stages),
           'K4G4_SEED': '1' if seed else '', 'PYTHONDONTWRITEBYTECODE': '1', 'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([str(root / 'fault'), *filter(None, [os.environ.get('PYTHONPATH')]),
                                          str(REPO / 'src'), str(REPO / 'tests')])}
    for key in ('GLOSSAPI_STAGE_PROFILE', 'GLOSSAPI_GREEK_WORDLIST', 'K4G4_FAULT_STAGE', 'GLOSSAPI_TEST_STUB_OCR', 'K4G5_LISTDIR'):
        env.pop(key, None)
    if fault:
        env['K4G4_FAULT_STAGE'] = fault
    if listdir:
        env['K4G5_LISTDIR'] = listdir
    if 'ocr' in stages:
        env['GLOSSAPI_TEST_STUB_OCR'] = '1'
    out = {}
    for step in steps:
        extra = [str(source) for source in sources] if step == 'start' else []
        done = subprocess.run([sys.executable, '-B', str(root / 'child.py'), str(root / 'run'), step, *extra],
                              capture_output=True, text=True, env=env, cwd=root, timeout=900)
        assert done.returncode == 0, done.stderr[-4000:]
        out[step] = json.loads(done.stdout.strip().splitlines()[-1])
    out['products'] = products(root / 'run' / 'output')
    out['output'] = root / 'run' / 'output'
    return out


def _needs_natives():
    if importlib.util.find_spec('glossapi_rs_cleaner') is None or importlib.util.find_spec('glossapi_rs_noise') is None:
        pytest.skip('the native cleaner/noise modules are not installed')


# ------------------------------------------------- item 1: corpus-level products do not depend on retry order ---

PLAN = ('extract', 'clean', 'sections', 'annotate', 'export')
CASES = {
    # name: (seeded metadata, fault stage, target document)
    'extract-alpha-seeded': (True, 'extract', 'alpha'),
    'clean-alpha-local': (False, 'clean', 'alpha'),
    'extract-english-seeded': (True, 'extract', 'english'),
}


@pytest.fixture(scope='module')
def order_runs(tmp_path_factory, greek_fixture):  # noqa: F811
    _needs_natives()
    inputs = tmp_path_factory.mktemp('k4g-fix5-inputs')
    shutil.copyfile(greek_fixture['broken'], inputs / 'greek.pdf')
    for name in ('alpha.pdf', 'beta.pdf'):
        shutil.copyfile(PDFS / name, inputs / name)
    sources = [inputs / 'alpha.pdf', inputs / 'beta.pdf', inputs / 'greek.pdf', english_pdf(inputs / 'english.pdf')]
    references, runs = {}, {}
    for name, (seed, fault, target) in CASES.items():
        if seed not in references:
            references[seed] = run_case(tmp_path_factory.mktemp('k4g-fix5-reference'), sources, stages=PLAN, seed=seed,
                                        listdir='tmpfs')
        runs[name] = dict(reference=references[seed],
                          retried=run_case(tmp_path_factory.mktemp('k4g-fix5-' + name), sources, stages=PLAN, seed=seed,
                                           fault=fault, target=target, listdir='tmpfs', steps=('start', 'resume')))
    return runs


@pytest.mark.parametrize('case', sorted(CASES))
def test_retried_run_corpus_products_equal_the_uninterrupted_run(order_runs, case):
    run = order_runs[case]
    reference, retried = run['reference'], run['retried']
    assert reference['start']['state'] == 'completed' and not reference['start']['failed']
    start, resume = retried['start'], retried['resume']
    assert start['state'] == 'completed_with_failures', start
    assert [row[1] for row in start['failed']] == ['disk_full'] and start['failed'][0][0].startswith(CASES[case][1] + '.'), start
    assert resume['state'] == 'completed' and resume['counts'] == reference['start']['counts'], (resume, reference['start'])
    for name in CORPUS_PARQUETS:  # the four corpus-level products exist in both runs, so the comparison covers them
        assert name in reference['products'] and name in retried['products'], name
    assert_same_products(retried, reference)


def test_english_fixture_measures_latin_word_shape_rate(order_runs):
    """The download_results comparison above covers a measured (non-null) latin_word_shape_rate."""
    frame = pd.read_parquet(order_runs['extract-alpha-seeded']['reference']['output'] / 'download_results/download_results.parquet')
    rate = frame.set_index('filename').loc['english.pdf', 'latin_word_shape_rate']
    assert not pd.isna(rate) and rate > 0.9


def test_latin_word_shape_rate_column_does_not_depend_on_the_table_history(tmp_path):
    """A table that already holds the column (read back from the parquet as float64) gives the same column, values and
    pandas metadata as a first pass (object): a.md is English (a measured rate), b.md is absent (null)."""
    import pyarrow as pa
    from glossapi.corpus.phase_clean import _enrich_extraction_metrics
    (tmp_path / 'markdown').mkdir()
    (tmp_path / 'markdown' / 'a.md').write_text('\n'.join(ENGLISH), encoding='utf-8')
    first = _enrich_extraction_metrics(pd.DataFrame({'filename': ['a.pdf', 'b.pdf']}), tmp_path, tmp_path / 'markdown')
    first.to_parquet(tmp_path / 'table.parquet', index=False)
    persisted = pd.read_parquet(tmp_path / 'table.parquet')
    assert str(persisted['latin_word_shape_rate'].dtype) == 'float64'  # the history: read back as float64
    again = _enrich_extraction_metrics(persisted, tmp_path, tmp_path / 'markdown')
    tables = [pa.Table.from_pandas(frame, preserve_index=False) for frame in (first, again)]
    assert tables[0].column('latin_word_shape_rate').to_pylist()[0] > 0.9
    assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize('profile', ['legacy', 'cutover'])
def test_section_ids_follow_the_sorted_file_names_not_the_listing_order(tmp_path, monkeypatch, profile):
    from glossapi import Corpus
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', profile)
    real_listdir = os.listdir
    outputs = []
    for order in (['zeta.md', 'beta.md', 'alpha.md'], ['alpha.md', 'zeta.md', 'beta.md']):
        root = tmp_path / ''.join(name[0] for name in order)
        corpus = Corpus(root / 'input', root / 'output')
        for name in ('zeta', 'alpha', 'beta'):
            text = f'# {name}\n\nBody of {name}.\n\n## Second\n\nMore text.'
            (corpus.markdown_dir / (name + '.md')).write_text(text)
        corpus.good_files = ['alpha', 'beta', 'zeta']
        monkeypatch.setattr(os, 'listdir', lambda path, order=order, corpus=corpus:
                            list(order) if Path(path) == corpus.markdown_dir else real_listdir(path))
        corpus.section()
        frame = pd.read_parquet(corpus.sections_parquet)
        outputs.append((corpus.sections_parquet.read_bytes(), frame[['id', 'filename']].values.tolist()))
    assert outputs[0][0] == outputs[1][0]
    assert [filename for _, filename in outputs[0][1]] == ['alpha'] * 2 + ['beta'] * 2 + ['zeta'] * 2
    assert [identifier for identifier, _ in outputs[0][1]] == [1, 2, 3, 4, 5, 6]


# ----------------------------------------------- item 2: latin_percentage follows its definition on the export ---

def cleaner_definition(text: str, directory: Path) -> tuple[float, float]:
    """``latin_percentage`` / ``percentage_greek`` as the code defines them: the native cleaner's
    percentage_latin_cleaned / percentage_greek_cleaned (ASCII Latin resp. Greek letters per non-whitespace character
    of the cleaned text), here measured on ``text`` without the OCR supplement's composition lines (FG-G02)."""
    import glossapi_rs_cleaner as cleaner
    from glossapi.ocr.page_artifacts import strip_supplement_markers
    (directory / 'in').mkdir(parents=True)
    (directory / 'out').mkdir()
    (directory / 'in' / 'document.md').write_text(strip_supplement_markers(text), encoding='utf-8')
    with contextlib.redirect_stdout(io.StringIO()):
        cleaner.run_complete_pipeline(str(directory / 'in'), str(directory / 'out'), str(directory / 'report.parquet'),
                                      ['greek', 'latin'], 1, True)
    row = pd.read_parquet(directory / 'report.parquet').iloc[0]
    return float(row['percentage_latin_cleaned']), float(row['percentage_greek_cleaned'])


def _records(export: Path) -> dict:
    return {record['filename']: record for record in
            (json.loads(line) for line in export.read_text().splitlines() if line.strip())}


def test_exported_latin_percentage_is_its_definition_after_a_page_supplement(tmp_path):
    """Corpus.ocr on a born-digital book whose pages 2 and 4 are OCR'd (the FG-G02 child: a stub engine, the default
    profile): the exported text carries the supplement's composition lines, and latin_percentage / percentage_greek
    equal the definition on the exported text without them. On aee0ae8 the markers were counted; on 5cedabc
    latin_percentage was the OCR-profile value of the text before OCR artefact cleaning."""
    _needs_natives()
    from test_ledger_fixset_fg import ocr_corpus
    output = ocr_corpus(tmp_path, 'book', seed='2')
    record = _records(output / 'export.jsonl')['book']
    assert '<!-- ocr-supplement-begin: pages=2,4 ' in record['document']
    assert (record['latin_percentage'], record['percentage_greek']) == cleaner_definition(record['document'], tmp_path / 'definition')


def test_exported_latin_percentage_is_its_definition_in_a_staged_ocr_run(tmp_path):
    """A staged run with OCR in the plan (stub engine; the scan is OCR'd whole) and a born-digital Latin document that
    is not OCR'd: every record's latin_percentage / percentage_greek equal the definition on its exported text, as in
    a plan without OCR (on 5cedabc: the OCR-profile value instead)."""
    _needs_natives()
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    shutil.copyfile(PDFS / 'alpha.pdf', inputs / 'alpha.pdf')
    shutil.copyfile(SCAN, inputs / 'polytonic_scan.pdf')
    run = run_case(tmp_path / 'run', [inputs / 'alpha.pdf', inputs / 'polytonic_scan.pdf'],
                   stages=('extract', 'clean', 'ocr', 'export'))
    assert run['start']['state'] == 'completed', run['start']
    records = _records(run['output'] / 'export.jsonl')
    assert sorted(records) == ['alpha', 'polytonic_scan'] and records['polytonic_scan']['ocr_success'] is True
    for name, record in records.items():
        assert (record['latin_percentage'], record['percentage_greek']) == \
            cleaner_definition(record['document'], tmp_path / ('definition-' + name)), name


@pytest.mark.parametrize('profile', ['legacy', 'cutover'])
def test_score_only_clean_does_not_count_the_supplement_markers(tmp_path, monkeypatch, profile):
    """The score-only clean (the refresh after OCR) measures a marked text without its composition lines, on both
    stage profiles; an unmarked text is measured as before (here: as itself)."""
    _needs_natives()
    from glossapi import Corpus
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', profile)
    corpus = Corpus(tmp_path / 'input', tmp_path / 'output')
    corpus.cleaned_markdown_dir.mkdir(parents=True, exist_ok=True)
    body = 'Κείμενο της σελίδας με ελληνικά γράμματα και λίγα λατινικά words here.\n'
    marked = ('<!-- ocr-pages: 1 -->\n<!-- ocr-supplement-begin: pages=1 run_id=0123456789abcdef -->\n' + body * 20
              + '<!-- ocr-supplement-end: pages=1 -->\n<!-- /ocr-pages -->\n')
    (corpus.cleaned_markdown_dir / 'marked.md').write_text(marked, encoding='utf-8')
    (corpus.cleaned_markdown_dir / 'plain.md').write_text(body * 20, encoding='utf-8')
    corpus.clean(input_dir=corpus.cleaned_markdown_dir, drop_bad=False, write_cleaned_files=False)
    frame = pd.read_parquet(corpus.output_dir / 'download_results' / 'download_results.parquet').set_index('filename')
    for name, text in (('marked', marked), ('plain', body * 20)):
        latin, greek = cleaner_definition(text, tmp_path / ('definition-' + name))
        assert (frame.at[name + '.pdf', 'latin_percentage'], frame.at[name + '.pdf', 'percentage_greek']) == (latin, greek), name
    assert frame.at['marked.pdf', 'latin_percentage'] == frame.at['plain.pdf', 'latin_percentage']
    staging = corpus.output_dir / '.glossapi' / 'staging'
    assert not staging.exists() or not list(staging.glob('clean-metric-surface-*'))  # the scoring copy is removed


# ------------------------------------------------------------- item 3: one metadata rule for every --files source ---

def test_every_supplied_file_gets_the_one_metadata_rule(tmp_path):
    """Six local files: the extraction publishes them in batches of five, so the sixth row is added by the processing
    state, not the artifact scan. All six get null url / download_success / extract_error (K4BI-G9)."""
    from glossapi.cli import main
    sources = tmp_path / 'sources'; sources.mkdir()
    files = [english_pdf(sources / f'{letter}.pdf', [f'Document {letter}: ' + line for line in ENGLISH[:3]]) for letter in 'abcdef']
    corpus = tmp_path / 'corpus'
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(['run', '--corpus', str(corpus), '--json', '--stages', 'extract,clean,export', '--files', *map(str, files)])
    envelope = json.loads(buffer.getvalue())
    assert code == 0 and envelope['data']['state'] == 'completed', envelope
    frame = pd.read_parquet(corpus / 'download_results' / 'download_results.parquet').set_index('filename')
    assert sorted(frame.index) == [f'{letter}.pdf' for letter in 'abcdef']
    cells = {name: [frame.at[name, field] for field in ('url', 'download_success', 'extract_error')] for name in frame.index}
    invented = {name: values for name, values in cells.items() if not all(pd.isna(value) for value in values)}
    assert not invented, invented
    records = [json.loads(line) for line in (corpus / 'export.jsonl').read_text().splitlines() if line.strip()]
    assert len(records) == 6
    assert all(record['url'] is None and record['download_success'] is None and record['extract_error'] is None
               for record in records), [(r['filename'], r['url'], r['download_success'], r['extract_error']) for r in records]
