"""Partitioned section prediction over PyArrow (replaces dask.dataframe in the classifier).

Annotation products, including ``classified_sections.parquet.tmp/part.<i>.parquet``, are compared
byte for byte, and the legacy row order of the classified output depends on the partition-local index
of the read-back predictions. This module therefore reproduces the observable choices of the dask path
it replaces (dask 2026.8 ``read_parquet`` → ``repartition`` → ``map_partitions`` → ``to_parquet``):

- input partitions: the whole file when its row groups total at most 2 × 256 MiB (uncompressed),
  otherwise row groups merged greedily while a partition stays within 256 MiB;
- ``repartition(npartitions=n)``: equal partition count unchanged; more partitions split each input
  partition evenly (``np.linspace``), the last one taking the remainder count; fewer partitions
  concatenate runs of ``int(i * old / new)``;
- strings read as ``pd.StringDtype("pyarrow")`` (dates and decimals as ``pd.ArrowDtype``), as dask's
  ``convert-string`` mapper does;
- an empty prediction result is replaced by the meta frame (``map_partitions`` does that);
- every part file is written with one schema inferred from the meta frame (object columns as
  ``string``), ``preserve_index=False`` and snappy compression;
- a directory is read back file by file in natural name order, each with its own RangeIndex,
  concatenated without empty frames.

What is bounded and what is not (CP-K2a F2):

- Prediction: ``predict`` never sees more than ``PREDICTION_ROWS`` rows. An output partition made of
  several input partitions (fewer outputs than inputs), or larger than the budget, is predicted in
  pieces of at most ``PREDICTION_ROWS`` rows and written to its part file in row groups of at most that
  size, so the largest prediction frame and write buffer stay fixed as the corpus grows. A part file
  holds the same rows in the same order as before; when it fits in one piece its bytes are unchanged
  (one ``write_table``), otherwise only its row-group layout differs and the read-back frame is the same.
  Predictions are row-wise; the fallback label after a prediction error applies to the failing piece.
- Reading: one input partition at a time (by construction at most 2 x 256 MiB uncompressed for a whole
  file, 256 MiB for merged row groups, or one larger row group) is held while its pieces are predicted.
- Not bounded here: the classifier reads every part file back and concatenates the whole corpus for
  post-processing (and, on resume, existing plus new rows), exactly as the dask path did. That
  whole-corpus barrier is recorded in GAPS (K4/W5); this module does not claim to remove it.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BLOCKSIZE = 256 * 2 ** 20  # dask's default read_parquet blocksize ("256 MiB")
PREDICTION_ROWS = 16384  # fixed row budget of one prediction frame (CP-K2a F2); independent of the corpus size


def arrow_to_pandas_type(arrow_type):
    if arrow_type == pa.string() or arrow_type == pa.large_string():
        return pd.StringDtype('pyarrow')
    if arrow_type == pa.date32() or arrow_type == pa.date64() or pa.types.is_decimal(arrow_type):
        return pd.ArrowDtype(arrow_type)
    return None


def to_frame(table):
    return table.to_pandas(categories=None, use_threads=False, ignore_metadata=False, types_mapper=arrow_to_pandas_type)


def read_frame(path, columns=None):
    """One Parquet file, or a directory of part files, as dask's read_parquet(...).compute() frames it."""
    if os.path.isdir(path):
        key = lambda name: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', name)]
        names = sorted((name for name in os.listdir(path) if not name.startswith(('.', '_'))), key=key)
        frames = [read_frame(os.path.join(path, name), columns) for name in names]
        nonempty = [frame for frame in frames if len(frame)]
        return frames[0] if not nonempty else pd.concat(nonempty, join='outer', sort=False)
    with pq.ParquetFile(path, pre_buffer=True) as source:
        return to_frame(source.read(columns=columns, use_threads=False, use_pandas_metadata=True))


def input_partitions(metadata, blocksize=None):
    """Row-group lists, one per input partition; ``None`` means the whole file."""
    blocksize = BLOCKSIZE if blocksize is None else blocksize
    sizes = [metadata.row_group(index).total_byte_size for index in range(metadata.num_row_groups)]
    if not sizes or sum(sizes) <= 2 * blocksize:
        return [None]
    parts, current, total = [], [0], sizes[0]
    for index, size in enumerate(sizes[1:], start=1):
        if total + size <= blocksize:
            current.append(index); total += size
        else:
            parts.append(current); current, total = [index], size
    parts.append(current)
    return parts


def repartition_plan(count, new):
    """For each output partition, the input partition indices and (start, stop) split position."""
    if new == count:
        return [[(index, None)] for index in range(count)]
    if new < count:
        ratio = count / new
        bounds = [int(index * ratio) for index in range(new + 1)]
        if bounds[0] > 0:
            bounds.insert(0, 0)
        if bounds[-1] < count:
            bounds[-1] = count
        return [[(index, None) for index in range(start, stop)] for start, stop in zip(bounds, bounds[1:])]
    div, mod = divmod(new, count)
    splits = [div] * count
    splits[-1] += mod
    return [[(index, (piece, pieces) if pieces > 1 else None)] for index, pieces in enumerate(splits) for piece in range(pieces)]


def write_schema(meta, source_schema):
    """dask: pyarrow schema of the non-empty meta frame.

    Object columns (null, list, struct or binary input types) take their input type. dask failed on
    them (``object()`` placeholders in its non-empty meta cannot be converted).
    """
    fields = []
    for name in meta.columns:
        if meta[name].dtype == object and name in source_schema.names:
            fields.append(pa.field(name, source_schema.field(name).type))
        else:
            fields.append(pa.Schema.from_pandas(meta[[name]], preserve_index=False).field(name))
    return pa.schema(fields)


def predict_partitions(path, temp_output, npartitions, prepare, predict, *, blocksize=None, rows=None):
    """Read, prepare, split, predict and write every part file; returns the part count.

    ``predict`` receives at most ``rows`` (default ``PREDICTION_ROWS``) rows at a time."""
    budget = max(1, int(PREDICTION_ROWS if rows is None else rows))
    with pq.ParquetFile(path, pre_buffer=True) as source:
        groups = input_partitions(source.metadata, blocksize)
        meta = prepare(to_frame(source.schema_arrow.empty_table()))
        meta['predicted_section'] = ''
        schema = write_schema(meta.reset_index(drop=True), source.schema_arrow)
        os.makedirs(temp_output, exist_ok=True)

        def load(index):
            group = groups[index]
            table = (source.read(use_threads=False, use_pandas_metadata=True) if group is None else
                     source.read_row_groups(group, use_threads=False, use_pandas_metadata=True))
            return prepare(to_frame(table))

        def to_table(frame):
            return pa.Table.from_pandas(frame.reset_index(drop=True), nthreads=1, preserve_index=False, schema=schema)

        written, cached = 0, (None, None)
        for pieces in repartition_plan(len(groups), npartitions):
            target = os.path.join(temp_output, f'part.{written}.parquet')
            writer, pending, buffered, buffered_rows = None, None, [], 0

            def flush():
                # One row group per flush; a part that needs a single flush is written exactly as before.
                nonlocal writer, pending, buffered, buffered_rows
                if not buffered:
                    return
                frame = buffered[0] if len(buffered) == 1 else pd.concat(buffered, join='outer', sort=False)
                buffered, buffered_rows = [], 0
                table = to_table(frame)
                if pending is None and writer is None:
                    pending = table
                    return
                if writer is None:
                    writer = pq.ParquetWriter(target, schema, compression='snappy')
                    writer.write_table(pending)
                    pending = None
                writer.write_table(table)

            for index, split in pieces:
                if cached[0] != index:
                    cached = (None, None)
                    cached = (index, load(index))  # each input partition is read once, in order
                frame = cached[1]
                if split is not None:
                    cuts = np.linspace(0, len(frame), split[1] + 1).astype(int)
                    frame = frame.iloc[cuts[split[0]]:cuts[split[0] + 1]]
                for start in range(0, len(frame), budget):
                    predicted = predict(frame.iloc[start:start + budget])
                    if buffered_rows + len(predicted) > budget:
                        flush()
                    buffered.append(predicted)
                    buffered_rows += len(predicted)
            flush()
            if writer is not None:
                writer.close()
            else:
                # dask's map_partitions returns the meta frame for an empty result
                table = pending if pending is not None and pending.num_rows else to_table(meta)
                with open(target, 'wb') as handle:
                    pq.write_table(table, handle, compression='snappy')
            written += 1
        return written
