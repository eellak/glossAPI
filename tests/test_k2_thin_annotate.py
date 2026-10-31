"""K2 dask replacement: own partition path vs the frozen c91590b dask path (byte-level oracle).

Every differential case runs both implementations on the same input and compares the part files
(names and bytes), the classified Parquet bytes, the frame handed to the index adjustment (values,
dtypes, index, order) and the recorded observations. The oracle needs dask; the dask-free tests
(default model discovery, bounded memory, partition plan) always run.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip('sklearn')
import glossapi
import glossapi.gloss_section_classifier as gsc
from glossapi import annotation_partitions as partitions
from glossapi.gloss_section import section_schema
from reference_oracles import oracle as reference_oracle

MODEL = Path(glossapi.__file__).resolve().parent/'models/section_classifier.joblib'
TEXTS = ['Εισαγωγή στο αντικείμενο της μελέτης και στους στόχους της εργασίας.',
         'Το κύριο σώμα του κειμένου με μερικές προτάσεις για τον έλεγχο.',
         'Παπαδόπουλος, Α. (2001). Ένα βιβλίο. Αθήνα: Εκδόσεις. Σμίθ, Β. (1999). Άρθρο.',
         '| Κεφάλαιο πρώτο | 1 |\n| Κεφάλαιο δεύτερο | 5 |\n| Κεφάλαιο τρίτο | 9 |\n| Κεφάλαιο τέταρτο | 13 |\n'
         '| Κεφάλαιο πέμπτο | 17 |\n| Κεφάλαιο έκτο | 21 |\n| Κεφάλαιο έβδομο | 25 |\n| Κεφάλαιο όγδοο | 29 |',
         'Παράρτημα με πίνακες και πρόσθετα στοιχεία.', '']
HEADERS = ['Εισαγωγή', 'Κεφάλαιο 1', 'Βιβλιογραφία', 'Περιεχόμενα', 'Παράρτημα', None]


def rows(documents, *, start=1, duplicate_every=0, nulls=True):
    out, ident = [], start
    for number, count in enumerate(documents):
        for ordinal in range(count):
            pick = (number * 7 + ordinal * 3) % len(TEXTS)
            header = HEADERS[pick] if nulls or HEADERS[pick] is not None else 'Τίτλος'
            section = TEXTS[pick] if nulls or pick != len(TEXTS) - 1 else 'κείμενο'
            out.append(dict(id=ident, row_id=f'row_{ident}', filename=f'DOC_{number:03d}', has_table=pick == 3,
                            has_list=ordinal % 4 == 1, has_other=False, has_text=True, header=header,
                            place='body', section=section if (ordinal % 9 != 5 or not nulls) else None,
                            json_section='{}', section_length=len(section or '') // 10 + ordinal,
                            section_propo=(ordinal * 37) % 400))
            ident = ident if duplicate_every and ordinal % duplicate_every == 0 else ident + 1
    return out


def sectioner_file(path, documents, **options):
    """The production layout: ParquetWriter(section_schema()), one row group per document."""
    schema = section_schema()
    with pq.ParquetWriter(path, schema=schema) as writer:
        data = rows(documents, **options)
        for name in dict.fromkeys(row['filename'] for row in data):
            frame = pd.DataFrame([row for row in data if row['filename'] == name])
            writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
    return path


class FlakyModel:
    """Labels by header; raises for any partition that holds a 'BOOM' header."""
    def predict(self, X):
        if (X['header'] == 'BOOM').any():
            raise RuntimeError('injected prediction failure')
        return np.array(['π' if 'Περιεχ' in h else ('β' if 'Βιβλιο' in h else 'άλλο') for h in X['header']], dtype=object)


def classifier(model):
    instance = gsc.GlossSectionClassifier()
    if model == 'packaged':
        instance.load_model(str(MODEL))
    else:
        instance.loaded_model = instance.clf_pipeline = model
    return instance


def run(kind, source, output, *, model='packaged', n_cpus=4, monkeypatch=None):
    instance = classifier(model)
    captured, events = {}, []
    original = instance._adjust_predictions_with_index_detection
    def capture(df, **kwargs):
        captured['frame'] = df.copy(deep=True)
        return original(df, **kwargs)
    instance._adjust_predictions_with_index_detection = capture
    def record(reason, **details):
        details.pop('model_path', None)
        if 'reported_error' in details:
            details['reported_error'] = dict(details['reported_error'], traceback=details['reported_error']['traceback'].strip().splitlines()[-1])
        events.append((reason, {k: (str(v) if k == 'path' else v) for k, v in details.items()}))
    monkeypatch.setattr(gsc, 'observe', record)
    if kind == 'reference':
        from k2_dask_reference import _process_predictions_with_dask
        _process_predictions_with_dask(instance, str(source), str(output), pretrained_model=False, n_cpus=n_cpus)
    else:
        instance._process_predictions(str(source), str(output), pretrained_model=False, n_cpus=n_cpus)
    return captured.get('frame'), events


def tree(root):
    root = Path(root)
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}


def differential(tmp_path, monkeypatch, source, *, prepare=None, **options):
    reference_oracle('dask')
    results = {}
    for kind in ('reference', 'own'):
        work = tmp_path/kind
        work.mkdir()
        local = work/'sections.parquet'
        shutil.copy2(source, local)
        if prepare:
            prepare(work)
        try:
            frame, events = run(kind, local, work/'classified.parquet', monkeypatch=monkeypatch, **options)
            error = None
        except Exception as caught:  # both paths must fail the same way, after the same files
            frame, events, error = None, [], (type(caught).__name__, str(caught))
        events = [(reason, {k: (v.replace(str(work), '<work>') if isinstance(v, str) else v) for k, v in details.items()})
                  for reason, details in events]
        results[kind] = dict(frame=frame, events=events, files=tree(work), error=error)
    reference, own = results['reference'], results['own']
    assert own['error'] == reference['error']
    assert sorted(own['files']) == sorted(reference['files'])
    for name, body in reference['files'].items():
        assert own['files'][name] == body, name
    if reference['frame'] is None:
        assert own['frame'] is None
    else:
        pd.testing.assert_frame_equal(own['frame'], reference['frame'], check_exact=True)
        assert own['frame'].index.tolist() == reference['frame'].index.tolist()
    # Fallback observations: dask appended them in thread-scheduler order; the own path uses partition order.
    fallback = lambda events: [e for e in events if e[0] == 'annotation_prediction_fallback']
    others = lambda events: [e for e in events if e[0] != 'annotation_prediction_fallback']
    assert others(own['events']) == others(reference['events'])
    key = lambda event: event[1]['row_ids']
    assert sorted(fallback(own['events']), key=key) == sorted(fallback(reference['events']), key=key)
    assert fallback(own['events']) == sorted(fallback(own['events']), key=key)
    return results


def labels(path):
    frame = pd.read_parquet(path)
    return list(zip(frame['filename'], frame['id'], frame['predicted_section']))


# ------------------------------------------------------------------ oracle cases

@pytest.mark.parametrize('documents', [[12, 30, 1, 7, 25, 4, 18], [3], [1, 1, 1, 40], [64] * 3])
def test_sectioner_layout_matches_dask_bytes_order_and_labels(tmp_path, monkeypatch, documents):
    source = sectioner_file(tmp_path/'input.parquet', documents)
    result = differential(tmp_path, monkeypatch, source)
    # Cross-partition filenames: 20 partitions over these documents split most of them.
    parts = [pq.read_table(tmp_path/'own/classified.parquet.tmp'/f'part.{i}.parquet') for i in range(20)]
    owners = [set(part.column('filename').to_pylist()) for part in parts]
    if sum(documents) >= 40:
        assert any(len([o for o in owners if name in o]) > 1 for name in set().union(*owners))
    assert len(labels(tmp_path/'own/classified.parquet')) == sum(documents)
    assert result['own']['frame'].index.has_duplicates == (sum(documents) > 1)


def test_extra_columns_nulls_and_pandas_layout(tmp_path, monkeypatch):
    frame = pd.DataFrame(rows([9, 14, 6]))
    frame['score'] = np.where(np.arange(len(frame)) % 3 == 0, np.nan, np.arange(len(frame)) * .5)
    frame['note'] = [None if i % 4 else f'n{i}' for i in range(len(frame))]
    frame['count'] = np.arange(len(frame), dtype='int32')
    source = tmp_path/'input.parquet'
    frame.to_parquet(source, index=False)
    differential(tmp_path, monkeypatch, source)


def test_null_and_list_columns_where_dask_failed(tmp_path, monkeypatch):
    """Declared difference: dask could not infer a schema for object columns; the own path keeps their input type."""
    frame = pd.DataFrame(rows([9, 14]))
    frame['empty'] = None
    frame['tags'] = [[i, i + 1] for i in range(len(frame))]
    source = tmp_path/'input.parquet'
    frame.to_parquet(source, index=False)
    if reference_oracle('dask'):
        with pytest.raises(pa.ArrowInvalid):
            run('reference', source, tmp_path/'reference.parquet', monkeypatch=monkeypatch)
    run('own', source, tmp_path/'own.parquet', monkeypatch=monkeypatch)
    written = pq.read_table(tmp_path/'own.parquet.tmp')
    assert written.schema.field('empty').type == pa.null() and written.schema.field('tags').type == pa.list_(pa.int64())
    assert sorted(pd.read_parquet(tmp_path/'own.parquet')['id']) == sorted(frame['id'])


def test_stored_index_column_is_dropped_like_dask(tmp_path, monkeypatch):
    frame = pd.DataFrame(rows([8, 5])).set_index('row_id')
    source = tmp_path/'input.parquet'
    frame.to_parquet(source)  # the index is stored as a column with pandas metadata
    differential(tmp_path, monkeypatch, source)


def test_duplicate_ids(tmp_path, monkeypatch):
    source = sectioner_file(tmp_path/'input.parquet', [10, 10, 10], duplicate_every=3)
    assert pd.read_parquet(source)['id'].duplicated().any()
    differential(tmp_path, monkeypatch, source)


@pytest.mark.parametrize('layout', ['no_row_groups', 'one_empty_row_group'])
def test_empty_input_writes_empty_partitions(tmp_path, monkeypatch, layout):
    source = tmp_path/'input.parquet'
    if layout == 'no_row_groups':
        pq.ParquetWriter(source, schema=section_schema()).close()
    else:
        pd.DataFrame(rows([2])).iloc[:0].to_parquet(source, index=False)
    differential(tmp_path, monkeypatch, source)


def test_missing_required_column_fails_the_same_way(tmp_path, monkeypatch):
    reference_oracle('dask')
    source = tmp_path/'input.parquet'
    pd.DataFrame(rows([4])).drop(columns=['has_list']).to_parquet(source, index=False)
    errors = []
    for kind in ('reference', 'own'):
        with pytest.raises(ValueError) as caught:
            run(kind, source, tmp_path/f'{kind}.parquet', monkeypatch=monkeypatch)
        errors.append(str(caught.value))
    assert errors[0] == errors[1] == "Input parquet file missing required columns: ['has_list']"


def test_prediction_exception_falls_back_per_partition(tmp_path, monkeypatch):
    data = rows([12, 18, 10])  # 40 rows: two per partition
    data[4]['header'] = data[17]['header'] = 'BOOM'
    source = tmp_path/'input.parquet'
    pd.DataFrame(data).to_parquet(source, index=False)
    result = differential(tmp_path, monkeypatch, source, model=FlakyModel())
    fallbacks = [details for reason, details in result['own']['events'] if reason == 'annotation_prediction_fallback']
    assert [details['row_ids'] for details in fallbacks] == [[5, 6], [17, 18]]
    assert all(details['fallback_label'] == 'άλλο' and details['reported_error']['message'] == 'injected prediction failure'
               for details in fallbacks)
    raw = pd.read_parquet(tmp_path/'own/classified.parquet.tmp')
    assert set(raw.loc[raw['id'].isin([5, 6, 17, 18]), 'predicted_section']) == {'άλλο'}


@pytest.mark.parametrize(('blocksize', 'n_cpus'), [(2048, 4), (2048, 1), (6000, 1), (1, 2)])
def test_multi_partition_inputs_follow_dask_blocksize_rules(tmp_path, monkeypatch, blocksize, n_cpus):
    reference_oracle('dask')
    from dask.dataframe.io.parquet import arrow as dask_arrow
    monkeypatch.setattr(dask_arrow.ArrowDatasetEngine, 'default_blocksize', classmethod(lambda cls: f'{blocksize} B'))
    monkeypatch.setattr(partitions, 'BLOCKSIZE', blocksize)
    source = sectioner_file(tmp_path/'input.parquet', [3, 1, 8, 2, 5, 9, 1, 4, 6, 2, 7, 3])
    metadata = pq.ParquetFile(source).metadata
    groups = partitions.input_partitions(metadata)
    assert len(groups) > 1  # adaptive mode is exercised
    differential(tmp_path, monkeypatch, source, n_cpus=n_cpus)


def _seed_output(source, keep_rows, *, extra=None, drop=None):
    def prepare(work):
        half = work/'half.parquet'
        pq.write_table(pq.read_table(source).slice(0, keep_rows), half)
        instance = classifier('packaged')
        instance._process_predictions(str(half), str(work/'classified.parquet'), pretrained_model=False, n_cpus=4)
        shutil.rmtree(work/'classified.parquet.tmp'); half.unlink()
        if extra or drop:
            table = pq.read_table(work/'classified.parquet')
            if drop:
                table = table.drop_columns(drop)
            if extra:
                table = table.append_column(extra, pa.array([f'legacy{i}' for i in range(table.num_rows)]))
            pq.write_table(table, work/'classified.parquet')
    return prepare


@pytest.mark.parametrize('variant', ['plain', 'mixed_schema', 'all_done', 'unreadable_checkpoint'])
def test_interrupted_resume_matches_dask(tmp_path, monkeypatch, variant):
    source = sectioner_file(tmp_path/'input.parquet', [12, 30, 7, 25])
    total = pq.ParquetFile(source).metadata.num_rows
    if variant == 'plain':
        prepare = _seed_output(source, 31)
    elif variant == 'mixed_schema':
        prepare = _seed_output(source, 31, extra='legacy_note', drop=['place'])
    elif variant == 'all_done':
        prepare = _seed_output(source, total)
    else:
        prepare = lambda work: (work/'classified.parquet').write_bytes(b'not parquet')
    result = differential(tmp_path, monkeypatch, source, prepare=prepare)
    if variant == 'all_done':
        assert result['own']['frame'] is None and not (tmp_path/'own/classified.parquet.tmp').exists()
    else:
        assert sorted(pd.read_parquet(tmp_path/'own/classified.parquet')['id']) == list(range(1, total + 1))


def test_stale_partition_files_are_read_back_like_dask(tmp_path, monkeypatch):
    source = sectioner_file(tmp_path/'input.parquet', [12, 30, 7])
    def prepare(work):
        classifier('packaged')._process_predictions(str(work/'sections.parquet'), str(work/'old.parquet'), pretrained_model=False, n_cpus=6)
        os.rename(work/'old.parquet.tmp', work/'classified.parquet.tmp'); (work/'old.parquet').unlink()
    differential(tmp_path, monkeypatch, source, prepare=prepare)


# ------------------------------------------------------------------ dask-free properties

def test_partition_plan_matches_dask_repartition_rules():
    assert partitions.repartition_plan(1, 3) == [[(0, (0, 3))], [(0, (1, 3))], [(0, (2, 3))]]
    assert partitions.repartition_plan(3, 3) == [[(0, None)], [(1, None)], [(2, None)]]
    assert partitions.repartition_plan(8, 3) == [[(0, None), (1, None)], [(2, None), (3, None), (4, None)], [(5, None), (6, None), (7, None)]]
    more = partitions.repartition_plan(3, 8)
    assert [piece[0][0] for piece in more] == [0, 0, 1, 1, 2, 2, 2, 2]


def test_prediction_memory_is_bounded_by_one_input_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(partitions, 'BLOCKSIZE', 3000)
    source = sectioner_file(tmp_path/'input.parquet', [20] * 12)
    groups = partitions.input_partitions(pq.ParquetFile(source).metadata)
    loaded, predicted = [], []
    real_read, real_groups = pq.ParquetFile.read, pq.ParquetFile.read_row_groups
    monkeypatch.setattr(pq.ParquetFile, 'read', lambda self, *a, **k: loaded.append(real_read(self, *a, **k).num_rows) or real_read(self, *a, **k))
    monkeypatch.setattr(pq.ParquetFile, 'read_row_groups', lambda self, *a, **k: loaded.append(real_groups(self, *a, **k).num_rows) or real_groups(self, *a, **k))
    instance = classifier(FlakyModel())
    original = instance.clf_pipeline.predict
    instance.clf_pipeline.predict = lambda X: predicted.append(len(X)) or original(X)
    instance._process_predictions(str(source), str(tmp_path/'out.parquet'), pretrained_model=False, n_cpus=4)
    largest = max(sum(20 for _ in group) for group in groups)
    assert len(groups) > 1 and largest < 240
    prediction_phase = loaded[:len(groups)]  # one read per input partition, then the read-back
    assert max(prediction_phase) == largest and sum(prediction_phase) == 240
    assert max(predicted) <= largest and sum(predicted) == 240


ISOLATED = '''
import importlib.abc, sys
class NoDask(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'dask':
            raise ModuleNotFoundError('dask is not installed', name=fullname)
sys.meta_path.insert(0, NoDask())
'''


def test_default_model_annotation_runs_without_dask(tmp_path):
    """Full annotation (classify + full annotation) with the default model discovery, dask unavailable."""
    code = ISOLATED + f'''
from pathlib import Path
import pandas as pd
sys.path.insert(0, {str(Path(__file__).parent)!r})
from test_default_models import sections_fixture
from glossapi import Corpus
corpus = Corpus({str(tmp_path/'input')!r}, {str(tmp_path/'out')!r})
rows = sections_fixture(corpus)
corpus.annotate()
classified = pd.read_parquet(corpus.classified_parquet)
assert sorted(classified.id) == [row['id'] for row in rows], classified
assert classified.predicted_section.notna().all()
assert corpus.fully_annotated_parquet.is_file()
assert not any(name.split('.')[0] == 'dask' for name in sys.modules)
print('labels', list(classified.predicted_section))
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path(glossapi.__file__).resolve().parents[1]), os.environ.get('PYTHONPATH', '')])})
    assert result.returncode == 0, result.stderr[-4000:]
    assert 'labels' in result.stdout
