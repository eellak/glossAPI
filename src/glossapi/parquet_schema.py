"""
Standardized Parquet Schema definitions for GlossAPI pipeline.

This module defines standard schemas for parquet files used throughout the GlossAPI
pipeline, ensuring consistency between different pipeline stages.
"""

from __future__ import annotations

from .ledger.journal import observe, observe_store, observe_source_rows, record_uncommitted_write

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None  # type: ignore[assignment]
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union, cast

import numpy as np
import pandas as pd

try:  # Optional dependency – lazily confirmed via _ensure_pyarrow()
    import pyarrow as pa  # type: ignore[import]
    import pyarrow.parquet as pq  # type: ignore[import]
except ImportError:  # pragma: no cover - exercised when pyarrow missing
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

from filelock import FileLock

from ._naming import canonical_stem

_BOOLEAN_METADATA_COLUMNS: Tuple[str, ...] = (
    "download_success",
    "is_duplicate",
    "needs_ocr",
    "ocr_success",
    "math_enriched",
    "enriched_math",
    "is_empty",
)


def _ensure_pyarrow() -> Tuple[Any, Any]:
    """Import pyarrow on demand, surfacing a friendly error if missing."""

    global pa, pq  # noqa: PLW0603 - shared cache for optional dependency
    if pa is not None and pq is not None:
        return pa, pq
    try:
        import pyarrow as pa_mod  # type: ignore[import]
        import pyarrow.parquet as pq_mod  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "pyarrow is required to read or write metadata parquets. "
            "Install it with `pip install pyarrow` inside the active environment."
        ) from exc
    pa = pa_mod  # type: ignore[assignment]
    pq = pq_mod  # type: ignore[assignment]
    return pa, pq


def _coerce_bool_value(value: Any) -> object:
    """Normalize common truthy/falsey representations to pandas boolean domain."""

    if value is None:
        return pd.NA
    if value is pd.NA:
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, Number) and not isinstance(value, bool):
        try:
            if pd.isna(value):
                return pd.NA
        except Exception:
            pass
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "t"}:
            return True
        if lowered in {"false", "0", "no", "n", "f"}:
            return False
        if lowered in {"", "none", "null"}:
            return pd.NA
        return pd.NA
    try:
        return bool(value)
    except Exception:
        return pd.NA


def _boolean_series(series: pd.Series) -> pd.Series:
    values = [_coerce_bool_value(v) for v in series.tolist()]
    return pd.Series(pd.array(values, dtype="boolean"), index=series.index)


def _nullable_or(lhs: object, rhs: object) -> object:
    lhs_norm = _coerce_bool_value(lhs)
    rhs_norm = _coerce_bool_value(rhs)
    if lhs_norm is True or rhs_norm is True:
        return True
    if lhs_norm is pd.NA or rhs_norm is pd.NA:
        return pd.NA
    return False


def _named_filenames(filenames: Any) -> pd.Series:
    """True where a row has a filename to merge by (a non-blank string); a skipped duplicate URL or a failed download
    has ``""`` (or no value) and is an occurrence row that no filename-keyed merge may match (LS6-G04)."""
    return pd.Series(filenames).map(lambda value: isinstance(value, str) and bool(value.strip())).astype(bool)


_FLOAT_EXACT_LIMIT = 2 ** 53


def exact_integer_columns(*frames: pd.DataFrame) -> list:
    """The same frames, where a numpy integer column that holds a value a float64 cannot represent exactly
    (``|value| > 2**53``) in *any* of them is made nullable (``Int64``/``UInt64``) in *all* of them.

    Legacy pandas row unions (``combine_first``/``update``/reindex) turn an integer column into float64 as soon as one
    row is missing; that is exact for every value up to 2**53 and rounds the rest (K2E-F3). Converting only the
    columns that would round keeps every other column's dtype and value exactly as the legacy merge produced them."""
    unsafe = set()
    for frame in frames:
        for column in frame.columns:
            values: Any = frame[column]
            dtype = values.dtype
            if (pd.api.types.is_integer_dtype(dtype) and not isinstance(dtype, pd.api.extensions.ExtensionDtype)
                    and len(frame) and bool(values.max() > _FLOAT_EXACT_LIMIT
                                            or (dtype.kind == 'i' and values.min() < -_FLOAT_EXACT_LIMIT))):
                unsafe.add(column)
    result = []
    for frame in frames:
        frame = frame.copy() if unsafe & set(frame.columns) else frame
        for column in unsafe & set(frame.columns):
            column_values: Any = frame[column]
            frame[column] = _nullable(column_values)
        result.append(frame)
    return result


def _nullable_integers(df: pd.DataFrame) -> pd.DataFrame:
    """A copy whose numpy integer columns are pandas nullable integers (exact under row unions with missing rows)."""
    result = df.copy()
    for column in result.columns:
        dtype = result[column].dtype
        if pd.api.types.is_integer_dtype(dtype) and not isinstance(dtype, pd.api.extensions.ExtensionDtype):
            result[column] = result[column].astype("UInt64" if pd.api.types.is_unsigned_integer_dtype(dtype) else "Int64")
    return result


def _prepare_metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized boolean columns and legacy aliases aligned."""

    if df.empty:
        return df.copy()
    result = df.copy()

    math_series: Optional[pd.Series] = None
    if "math_enriched" in result.columns:
        math_series = _boolean_series(result["math_enriched"])
    if "enriched_math" in result.columns:
        enriched_series = _boolean_series(result["enriched_math"])
        if math_series is None:
            math_series = enriched_series
        else:
            math_series = math_series.combine(enriched_series, _nullable_or)
    if math_series is not None:
        math_series = math_series.astype("boolean")
        result["math_enriched"] = math_series
        result["enriched_math"] = math_series

    for column in _BOOLEAN_METADATA_COLUMNS:
        if column not in result.columns:
            continue
        result[column] = _boolean_series(result[column])

    return result


class MetadataPolicyError(ValueError):
    """Base of the metadata target policy's typed refusals; legacy broad ``except`` blocks re-raise it (CP-K2e F5)."""


class MetadataAliasError(MetadataPolicyError):
    """A metadata parquet target that cannot be written safely: a hard-linked file (atomic replacement would detach
    the other names), or two hard-linked names of one file used as input and output (CP-K2d F2, K2F round 3)."""


class MetadataSchemaError(MetadataPolicyError):
    """A merge whose result cannot hold every preserved value exactly (CP-K2d F4); raised before anything is
    replaced."""


class MetadataStaleGenerationError(MetadataPolicyError):
    """A captured generation of a metadata file would be republished over a file that changed after the capture
    (CP-K2e F1); raised under the lock, before anything is replaced."""


class MetadataDestinationError(MetadataPolicyError):
    """A non-metadata writer (e.g. the CLI exporter) was pointed at a metadata parquet (CP-K2e F1)."""


METADATA_SIGNATURES = ({'filename'}, {'download_success', 'download_error', 'filename_base'},
                       {'filter', 'mojibake_badness_score', 'greek_badness_score'})


def metadata_parquet_content(path: Union[str, Path]) -> bool:
    """Whether the bytes at ``path`` are a metadata parquet, whatever the file is called (CP-K3c2 C2-F1): a regular
    file that starts with the parquet magic ``PAR1`` and whose schema has no ``section`` column and has
    ``filename``, the downloader signature or the cleaner-metrics signature (the ``find_metadata_parquet`` schema
    test). Reads only the magic and the parquet footer; an unreadable file is not metadata."""
    candidate = Path(path)
    try:
        if not candidate.is_file():
            return False
        with candidate.open('rb') as stream:
            if stream.read(4) != b'PAR1':
                return False
        names = set(pq.read_schema(candidate).names)
    except Exception:
        return False
    return 'section' not in names and any(signature <= names for signature in METADATA_SIGNATURES)


def selectable_metadata_parquet(path: Union[str, Path]) -> bool:
    """Whether ``path`` is a parquet ``find_metadata_parquet`` could select: a ``.parquet`` file (not ``.partial.``;
    both compared case-insensitively, CP-K3c2 C2-F1) whose content passes ``metadata_parquet_content``."""
    name = Path(path).name.casefold()
    if not name.endswith('.parquet') or '.partial.' in name:
        return False
    return metadata_parquet_content(path)


def _on_disk_spelling(path: Path) -> Path:
    """The spelling the filesystem stores for ``path`` (a ``realpath`` result). On a case-insensitive volume
    ``realpath`` keeps the caller's letter case, so ``DOWNLOAD_RESULTS.PARQUET`` and ``download_results.parquet``
    would be two identities of one file (CP-K3c2 C2-F1). Where the platform can say (``fcntl.F_GETPATH``, macOS),
    the nearest existing ancestor is replaced by its stored spelling and the missing tail is kept; the answer is used
    only when it differs from the input in letter case alone (a hard link may report another name, which is not a
    respelling). Elsewhere, and on any error, ``path`` is returned unchanged."""
    getpath = getattr(fcntl, 'F_GETPATH', None) if fcntl is not None else None
    if getpath is None:
        return path
    existing, tail = path, []
    while not os.path.lexists(existing):
        if existing.parent == existing:
            return path
        tail.append(existing.name)
        existing = existing.parent
    try:
        descriptor = os.open(existing, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
    except OSError:
        return path
    try:
        stored = os.fsdecode(fcntl.fcntl(descriptor, getpath, b'\0' * 1024).split(b'\0', 1)[0])
    except OSError:
        return path
    finally:
        os.close(descriptor)
    if not stored or stored == os.fspath(existing) or stored.casefold() != os.fspath(existing).casefold():
        return path
    return Path(stored, *reversed(tail))


def resolve_metadata_target(path: Union[str, Path]) -> Path:
    """THE metadata target policy (K2F round 3): the physical file a metadata path names.

    ``os.path.realpath(os.path.expanduser(path))``: components are resolved in traversal order (a directory symlink
    followed by ``..`` climbs from the link's target, as the OS does; no lexical ``abspath`` folding first, CP-K2d F1),
    and a file symlink resolves to its target. Lock, read and atomic replacement all use this path; the caller's
    spelling is kept only for messages, journal observations and return values. Writing through a file symlink
    therefore updates the real file and the link survives (CP-K2d F3). On a case-insensitive volume the existing
    part of the result takes the filesystem's stored letter case (``_on_disk_spelling``, CP-K3c2 C2-F1), so every
    case alias of one file has one identity: one lock file, and a replacement keeps the file's name."""
    return _on_disk_spelling(Path(os.path.realpath(os.path.expanduser(os.fspath(path)))))


# Round-2 name, kept as an alias of the one policy.
metadata_path_identity = resolve_metadata_target


def refuse_metadata_destination(path: Union[str, 'os.PathLike[str]'], *, writer: str) -> None:
    """A non-metadata writer (the CLI exporter, ``input_manifest.write_manifest``) never writes a metadata target
    (K2F round 4, CP-K2e F1), under the full metadata-target rule (CP-K3 F2). The destination is resolved to its
    physical file first (``resolve_metadata_target``: file and directory symlinks, ``..`` in traversal order); then,
    before anything is written:

    - an existing physical file with more than one name (``st_nlink > 1``) raises ``MetadataAliasError``, whatever
      its suffix: another name may be a metadata parquet, and a hard link cannot be traced back to it without
      searching the filesystem, so every hard-linked destination is refused;
    - an existing physical file whose bytes are a metadata parquet (``metadata_parquet_content``: parquet magic and
      the ``find_metadata_parquet`` schema test, whatever the name or suffix), or a ``download_results/*.parquet``
      name compared case-insensitively, raises ``MetadataDestinationError`` (CP-K3c2 C2-F1: on a case-insensitive
      volume ``DOWNLOAD_RESULTS.PARQUET`` is the metadata file; the resolution also takes the stored case)."""
    physical = resolve_metadata_target(path)
    try:
        links = os.stat(physical).st_nlink
    except FileNotFoundError:
        links = 0
    if links > 1:
        raise MetadataAliasError(f"refusing to {writer} through {os.fspath(path)} ({physical}): it is hard-linked "
                                 f"({links} names), and another name may be corpus metadata; choose a new path")
    if metadata_parquet_content(physical) or (physical.suffix.casefold() == '.parquet'
                                              and physical.parent.name.casefold() == 'download_results'):
        raise MetadataDestinationError(f"refusing to {writer} over metadata parquet {os.fspath(path)} ({physical}); "
                                       "choose a path outside the corpus metadata")


def refuse_hard_linked_target(target: Union[str, Path], given: Union[str, Path, None] = None) -> None:
    """Hard links are refused (CP-K2d F2): an existing target with ``st_nlink > 1`` raises ``MetadataAliasError``
    before anything is read or written; an atomic replacement cannot keep the other names pointing at the update."""
    try:
        links = os.stat(target).st_nlink
    except FileNotFoundError:
        return
    if links > 1:
        raise MetadataAliasError(f'Metadata parquet {given if given is not None else target} is hard-linked '
                                 f'({links} names for {target}); atomic replacement would detach the other names. '
                                 'Use one name (or a symbolic link) for a metadata file.')


def same_metadata_file(first: Union[str, Path], second: Union[str, Path]) -> bool:
    """Whether two paths name the same physical metadata parquet: equal resolved targets, or, when both exist,
    ``os.path.samefile`` of the resolved targets (case-insensitive spellings; hard links, which the writers refuse)."""
    resolved_first, resolved_second = resolve_metadata_target(first), resolve_metadata_target(second)
    if resolved_first == resolved_second:
        return True
    return (os.path.exists(resolved_first) and os.path.exists(resolved_second)
            and os.path.samefile(resolved_first, resolved_second))


@contextmanager
def metadata_target(path: Union[str, Path]) -> Iterator[Path]:
    """THE choke point for every metadata parquet writer (K2F round 3): yields the resolved physical target while
    holding its lock, after refusing a hard-linked target (checked before the lock and again under it). The caller
    reads and atomically replaces only the yielded path (``_write_metadata_parquet_locked``,
    ``_write_serialized_locked``, ``_copy_file_locked``, or a product publication given it as physical target)."""
    target = resolve_metadata_target(path)
    refuse_hard_linked_target(target, path)
    with _parquet_lock(target):
        refuse_hard_linked_target(target, path)
        yield target


@contextmanager
def _parquet_lock(path: Path) -> Iterator[None]:
    """Serialize writers via file-based lock, falling back to atomic rename.

    The lock file sits next to ``resolve_metadata_target(path)``, whatever spelling the caller used. Writers enter it
    through ``metadata_target`` (which also refuses hard links); this is its locking primitive."""

    path = resolve_metadata_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    if FileLock is not None:
        lock = FileLock(str(lock_path))
        with lock:
            try:
                yield
            finally:
                # Remove the lock file while still holding it. A waiter that opened the removed inode
                # sees st_nlink == 0 after its flock and retries on the path (filelock >= 4.2.4).
                # Removing it after the release let two processes hold the lock on two inodes at once.
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
        return

    acquired = False
    handle: Optional[int] = None
    try:
        while not acquired:
            try:
                handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                acquired = True
            except FileExistsError:
                time.sleep(0.05)
        yield
    finally:
        if handle is not None:
            try:
                os.close(handle)
            except Exception:
                pass
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _replace_in_place(tmp_path: Path, target: Path) -> None:
    """The one in-place replacement of a metadata target (fix 7, R1): inside a cutover wrapper the bytes are first
    bound in the ledger as that wrapper's uncommitted write (``record_uncommitted_write``; durable before the
    replace, and if the ledger cannot take the record the target is left untouched), then replaced atomically."""
    record_uncommitted_write(tmp_path, target)
    os.replace(tmp_path, target)


def _write_metadata_parquet_locked(df: pd.DataFrame, parquet_path: Path, observed: Optional[Path] = None) -> None:
    """Write + atomic replace of the resolved target yielded by ``metadata_target`` (the caller holds it).
    ``observed``: the caller's spelling, recorded by the journal."""
    prepared = _prepare_metadata_frame(df)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pa_mod, pq_mod = _ensure_pyarrow()
    table = pa_mod.Table.from_pandas(prepared, preserve_index=False)
    tmp_path = parquet_path.with_suffix(parquet_path.suffix + f".{uuid.uuid4().hex}.tmp")
    try:
        pq_mod.write_table(table, tmp_path)
        _replace_in_place(tmp_path, parquet_path)
        observe_store(observed or parquet_path, kind="metadata_parquet")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _write_metadata_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
    with metadata_target(parquet_path) as target:
        _write_metadata_parquet_locked(df, target, observed=parquet_path)


def _copy_file_locked(source: Path, target: Path, *, expected_sha256: Optional[str] = None) -> None:
    """Copy a file's bytes over the resolved target yielded by ``metadata_target`` (temporary + atomic replace).

    ``expected_sha256`` (a recovery restoring registered bytes; fix 8): the temporary copy must hold exactly these
    bytes, else nothing is replaced. The check is on the bytes about to be installed, not on the source before the
    copy, so a source that changes after its own verification is never published."""
    import shutil
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name('.' + target.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        shutil.copyfile(source, tmp_path)
        if expected_sha256 is not None:
            digest = hashlib.sha256()
            with open(tmp_path, 'rb') as handle:
                for block in iter(lambda: handle.read(1 << 20), b''):
                    digest.update(block)
            if digest.hexdigest() != expected_sha256:
                raise MetadataStaleGenerationError(f'{source} no longer holds the registered bytes; {target} was not replaced')
        _replace_in_place(tmp_path, target)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _write_serialized_locked(df: pd.DataFrame, parquet_path: Path, serialize) -> None:
    """``serialize(frame, temporary_path)`` + atomic replace of the resolved target yielded by ``metadata_target``."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parquet_path.with_suffix(parquet_path.suffix + f".{uuid.uuid4().hex}.tmp")
    try:
        serialize(df, tmp_path)
        _replace_in_place(tmp_path, parquet_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def update_metadata_parquet(parquet_path: Union[str, Path], update, *, serialize=None) -> pd.DataFrame:
    """Locked read-modify-write of a metadata parquet: read, ``update(frame) -> frame``, write, one critical section.

    ``update`` receives the current rows (an empty frame when the file does not exist yet) and returns the rows to
    write; returning ``None`` writes nothing. Concurrent writers cannot lose each other's rows: each reads only
    after the previous writer replaced the file.

    ``serialize(frame, temporary_path)``, when given, writes the bytes instead of the normalising metadata writer, so
    a caller that always wrote its file with its own serializer (``DataFrame.to_parquet``, a plain pyarrow table)
    keeps its bytes; the file is still replaced atomically inside the lock (CP-K2a F1).

    Lock, read and replacement all use the resolved physical target (``metadata_target``, K2F round 3): any spelling
    of the file (``..`` after a directory symlink, a file symlink) updates the same file; hard links are refused."""
    given = Path(parquet_path)
    with metadata_target(given) as target:
        current = pd.read_parquet(target) if target.exists() else pd.DataFrame()
        updated = update(current)
        if updated is not None:
            if serialize is None:
                _write_metadata_parquet_locked(updated, target, observed=given)
            else:
                _write_serialized_locked(updated, target, serialize)
        return current if updated is None else updated


def replace_metadata_parquet(parquet_path: Union[str, Path], frame: pd.DataFrame, *, serialize=None) -> None:
    """Replace a metadata parquet whole with ``frame`` (no read), atomically, through ``metadata_target``: for a
    writer whose output is an explicit new artifact (e.g. a distinct target of ``rewrite_metadata_parquet``)."""
    given = Path(parquet_path)
    with metadata_target(given) as target:
        if serialize is None:
            _write_metadata_parquet_locked(frame, target, observed=given)
        else:
            _write_serialized_locked(frame, target, serialize)


def rewrite_metadata_parquet(source: Union[str, Path], target: Union[str, Path], transform, *, serialize=None) -> pd.DataFrame:
    """Read ``source``, ``transform(frame) -> frame``, write ``target``; returns the written frame.

    The same-file decision is made here, on resolved targets only (K2F round 3). Same physical file → one locked
    read/transform/atomic replace of it through ``metadata_target`` (a concurrent locked commit is transformed, never
    overwritten). Two hard-linked names of one file → ``MetadataAliasError`` before anything is read or written.
    Otherwise ``target`` is an explicit new artifact: ``source`` is read as it is and ``target`` is replaced whole
    through ``metadata_target``. Either way a successful rewrite has updated the requested target. A missing
    ``source`` raises the ``pd.read_parquet`` error before any output exists."""
    source_target, output_target = resolve_metadata_target(source), resolve_metadata_target(target)
    if source_target != output_target and os.path.exists(source_target) and os.path.exists(output_target) \
            and os.path.samefile(source_target, output_target):
        if os.stat(source_target).st_nlink > 1:
            raise MetadataAliasError(f'{source} and {target} are hard-linked names of one file ({source_target}); '
                                     'atomic replacement would detach them. Use one name.')
        output_target = source_target  # one file under two case spellings: in place
    if source_target == output_target:
        def _in_place(current):
            if not source_target.exists():
                pd.read_parquet(source)  # the missing-file error the direct read raised
            return transform(current)
        return update_metadata_parquet(source, _in_place, serialize=serialize)
    frame = transform(pd.read_parquet(source))
    replace_metadata_parquet(target, frame, serialize=serialize)
    return frame


def pandas_parquet(frame: pd.DataFrame, path: Path) -> None:
    """The ``DataFrame.to_parquet(path, index=False)`` bytes the download and downloader writers always produced."""
    frame.to_parquet(path, index=False)


def _nullable(series: pd.Series) -> pd.Series:
    """The same values in a dtype that can also hold a missing value without changing them: numpy integers →
    ``Int64``/``UInt64``, numpy booleans → ``boolean``; every other dtype (float, object, string, datetime,
    extension types) already holds missing values and is kept as it is (CP-K2d F4)."""
    dtype = series.dtype
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return series
    if pd.api.types.is_bool_dtype(dtype):
        return series.astype('boolean')
    if pd.api.types.is_integer_dtype(dtype):
        return series.astype('UInt64' if dtype.kind == 'u' else 'Int64')
    return series


def _missing_like(series: pd.Series, index) -> pd.Series:
    """An all-missing column of ``series``'s (nullable) dtype for ``index``."""
    filler = _nullable(series).iloc[:0].reindex(range(len(index)))
    filler.index = index
    return filler


def _same_value(left, right) -> bool:
    left_missing = pd.api.types.is_scalar(left) and pd.isna(left)
    right_missing = pd.api.types.is_scalar(right) and pd.isna(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    try:
        equal = left == right
        if isinstance(equal, (bool, np.bool_)):
            return bool(equal)
    except Exception:
        pass
    return np.array_equal(np.asarray(left, dtype=object), np.asarray(right, dtype=object))


def _dtype_family(dtype) -> Optional[str]:
    """The typed family a persisted column must keep through a union, or ``None`` when exact values suffice (numeric,
    boolean, string and object columns: ``_nullable`` moves those between numpy and nullable dtypes on purpose).
    Supported persisted extension boundary (K2E-F4): these families round-trip through pyarrow parquet; decimal and
    list values are objects and keep only what the parquet serializer keeps."""
    if isinstance(dtype, pd.CategoricalDtype):
        return 'category'
    if isinstance(dtype, pd.DatetimeTZDtype):
        return 'datetime with time zone'
    if isinstance(dtype, pd.PeriodDtype):
        return 'period'
    if isinstance(dtype, pd.IntervalDtype):
        return 'interval'
    return None


def _unify_categories(frames: list, column: str) -> None:
    """Give every frame's categorical ``column`` the union of all their categories (first-seen order), so the union
    stays categorical instead of falling back to object (K2E-F4). Ordered categoricals are left to the family check."""
    dtypes = [frame[column].dtype for frame in frames if column in frame.columns]
    if len(dtypes) < 2 or not all(isinstance(d, pd.CategoricalDtype) and not d.ordered for d in dtypes):
        return
    categories = list(dict.fromkeys(value for d in dtypes for value in d.categories))
    for frame in frames:
        if column in frame.columns and list(frame[column].cat.categories) != categories:
            frame[column] = frame[column].cat.set_categories(categories)


def _require_exact(original: pd.DataFrame, result: pd.DataFrame, start: int, what: str) -> None:
    """Every value of ``original`` reappears exactly at ``result`` rows ``start:start+len`` (compared as Python
    objects, so 9007199254740993 never equals 9007199254740992.0); else ``MetadataSchemaError``."""
    for column in original.columns:
        before = original[column]
        after = result[column].iloc[start:start + len(original)]
        if after.dtype == before.dtype:
            continue  # same dtype: no conversion happened
        family = _dtype_family(before.dtype)
        if family is not None and _dtype_family(after.dtype) != family:
            # Equal labels are not enough (K2E-F4): a categorical, tz-aware, period or interval column that becomes
            # object/str has lost its type (categories and order, time zone, frequency, closed side).
            raise MetadataSchemaError(
                f'Merging {what} would change column {column!r} from {before.dtype} ({family}) to {after.dtype}; '
                'nothing was replaced.')
        for position, (left, right) in enumerate(zip(before.astype(object), after.astype(object))):
            if not _same_value(left, right):
                raise MetadataSchemaError(
                    f'Merging {what} would change column {column!r} at row {position}: {left!r} ({before.dtype}) -> '
                    f'{right!r} ({after.dtype}); nothing was replaced.')


def lossless_concat(frames, *, what: str = 'metadata rows') -> pd.DataFrame:
    """``pd.concat(frames, ignore_index=True)`` without silent value changes (CP-K2d F4).

    Columns that some frame lacks are first made nullable in the frames that have them (``_nullable``) and added to
    the others as missing values of that dtype, so integers, booleans and extension types keep exact values and
    their dtype family; then every original value is checked against the result, and an inexact union raises
    ``MetadataSchemaError`` before the caller replaces anything. Frames with identical columns concatenate exactly
    as ``pd.concat`` did."""
    frames = list(frames)
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    partial = {column for column in columns if any(column not in frame.columns for frame in frames)}
    aligned = []
    for frame in frames:
        frame = frame.copy()
        for column in columns:
            if column in frame.columns:
                if column in partial:
                    frame[column] = _nullable(frame[column])
            else:
                like = next(other[column] for other in frames if column in other.columns)
                frame[column] = _missing_like(like, frame.index)
        aligned.append(frame[columns])
    for column in columns:
        _unify_categories(aligned, column)
    result = pd.concat(aligned, ignore_index=True)
    start = 0
    for frame in frames:
        _require_exact(frame, result, start, what)
        start += len(frame)
    return result


def merge_owned_rows(current: pd.DataFrame, rows: pd.DataFrame, key: str, owned=()) -> pd.DataFrame:
    """Publish this operation's ``rows`` over the ``current`` file without losing another writer's rows, columns or
    values.

    The result is ``rows`` in their order; columns only ``current`` has are joined onto them by ``key`` (another
    writer's fields of the same document), and ``current`` rows whose key is neither in ``rows`` nor in ``owned``
    (keys this operation wrote before and now replaces) are appended unchanged. With nothing concurrent (``current``
    holds only this operation's earlier rows and no other column) the result is ``rows`` itself.

    Rule (CP-K2c F2, CP-K2d F4/F5): a merge never drops a current row, column or exact value it cannot match. When
    ``current`` (or ``rows``) has no ``key`` column — e.g. the keyless downloader shape
    ``download_success``/``download_error``/``filename_base`` that ``find_metadata_parquet`` selects — no row can be
    matched, so every current row is kept, after ``rows``. A ``current`` with zero rows still contributes its
    columns. Unions are lossless (``lossless_concat``; joined columns made nullable first) or raise
    ``MetadataSchemaError`` before anything is replaced. Preserving is chosen over refusing unmatched rows: refusing
    would leave this operation's results unpublished, while appending loses nothing."""
    if current.columns.empty:
        return rows
    if current.empty:
        result = rows.copy()
        for column in current.columns:
            if column not in result.columns:
                result[column] = _missing_like(current[column], result.index)
        return result
    if key not in current.columns or key not in rows.columns:
        return lossless_concat([rows, current], what='rows that cannot be matched by ' + key)
    published = set(rows[key].dropna().tolist())
    extra = [column for column in current.columns if column not in rows.columns]
    result = rows
    if extra:
        lookup = current[current[key].notna()].drop_duplicates(key, keep='last').set_index(key)[extra]
        for column in extra:
            lookup[column] = _nullable(lookup[column])
        joined = rows.join(lookup, on=key)
        matched = rows[key].isin(lookup.index).to_numpy()
        if matched.any():
            expected = lookup.loc[rows.loc[matched, key]].reset_index(drop=True)
            _require_exact(expected, joined.loc[matched, extra].reset_index(drop=True), 0, 'joined columns')
        result = joined
    leftover = current[~current[key].isin(published | set(owned))]
    if len(leftover):
        result = lossless_concat([result, leftover], what='unmatched current rows')
    return result


class ParquetSchema:
    """
    Defines standardized schema for parquet files in the GlossAPI pipeline.
    
    This class provides methods to validate, read, and write parquet files
    with consistent schemas for different pipeline stages.
    
    The pipeline uses two distinct types of parquet files:
    
    1. Metadata Parquet:
       - Each row represents a file (one-to-one relationship with files)
       - Essential columns: filename, URL column (configurable), extraction quality
       - Used by: downloader, extractor, and filter stages
       - Example: download_results.parquet
       - Typical location: {output_dir}/download_results/
       - Schema: METADATA_SCHEMA, DOWNLOAD_SCHEMA
    
    2. Sections Parquet:
       - Each row represents a section from a file (many-to-one relationship with files)
       - Essential columns: filename, title, content, section, predicted_section
       - Used by: section and annotation stages
       - Examples: sections_for_annotation.parquet, classified_sections.parquet
       - Typical location: {output_dir}/sections/
       - Schema: SECTION_SCHEMA, CLASSIFIED_SCHEMA
    
    When the pipeline runs, it first creates and populates a metadata parquet,
    then uses it to filter files, and finally creates section parquets from the
    filtered files.
    """
    
    def __init__(self, pipeline_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ParquetSchema with optional pipeline configuration.
        
        Args:
            pipeline_config: Configuration dictionary with settings such as
                url_column, which will be used throughout the pipeline
        """
        # TODO: Add more robust configuration options for each parquet type from input metadata and downloder, to section, and two phases of annotaiton.
        # TODO: Add support for consolidated sections parquet handling
        # TODO: Add methods to find the latest sections parquet in a pipeline
        self.config = pipeline_config or {}
        self.url_column = self.config.get('url_column', 'url')
    
    # Basic schema with common fields used across all parquet files
    if pa is not None:
        COMMON_SCHEMA = pa.schema(
            [
                ("id", pa.string()),
                ("row_id", pa.int64()),
                ("filename", pa.string()),
            ]
        )
    else:  # pragma: no cover - pyarrow not installed
        COMMON_SCHEMA = None
    
    # Metadata schema for files used by downloader and quality assessment
    if pa is not None:
        METADATA_SCHEMA = pa.schema(
            [
                ("filename", pa.string()),
                ("url", pa.string()),  # Can be customized with url_column parameter
                ("download_success", pa.bool_()),
                ("download_error", pa.string()),
                ("trigrams", pa.string()),  # Values: "natural", "unnatural", "unknown"
                ("processing_stage", pa.string()),  # Tracks progress through pipeline
                ("badness_score", pa.float64()),
                ("percentage_greek", pa.float64()),
                ("percentage_latin", pa.float64()),
                ("ocr_pages", pa.string()),
                ("ocr_pages_count", pa.int64()),
                ("ocr_pages_unflagged_count", pa.int64()),
                ("extract_status", pa.string()),
                ("page_count", pa.int64()),
                ("pua_char_count", pa.int64()),
                ("control_char_count", pa.int64()),
                ("font_repair_status", pa.string()),
                ("font_repair_backend", pa.string()),
                ("font_repair_fonts", pa.int64()),
                ("font_repair_valid_rate_before", pa.float64()),
                ("font_repair_valid_rate_after", pa.float64()),
            ]
        )
    else:  # pragma: no cover - pyarrow not installed
        METADATA_SCHEMA = None
    
    # Additional schemas for specific pipeline stages
    if pa is not None:
        DOWNLOAD_SCHEMA = pa.schema(
            [
                ("url", pa.string()),  # Will be replaced with the actual url_column
                ("download_success", pa.bool_()),
                ("download_error", pa.string()),
                ("download_retry_count", pa.int32()),
                ("filename", pa.string()),
                ("file_ext", pa.string()),
                ("is_duplicate", pa.bool_()),
                ("duplicate_of", pa.string()),
                ("source_row", pa.int32()),
                ("url_index", pa.int32()),
                ("filename_base", pa.string()),
            ]
        )
    else:  # pragma: no cover - pyarrow not installed
        DOWNLOAD_SCHEMA = None
    
    # The section and annotation outputs as emitted (T2-G12): ``gloss_section.section_schema()`` and the classifier's
    # output, which adds ``predicted_section``. tests/test_k4g_b2_legacy.py binds these constants to the emitter.
    if pa is not None:
        SECTION_SCHEMA = pa.schema(
            [
                ("id", pa.int64()),
                ("row_id", pa.string()),
                ("filename", pa.string()),
                ("has_table", pa.bool_()),
                ("has_list", pa.bool_()),
                ("has_other", pa.bool_()),
                ("has_text", pa.bool_()),
                ("header", pa.string()),
                ("place", pa.string()),
                ("section", pa.string()),
                ("json_section", pa.string()),
                ("section_length", pa.int64()),
                ("section_propo", pa.int64()),
            ]
        )
    else:  # pragma: no cover - pyarrow not installed
        SECTION_SCHEMA = None

    if pa is not None:
        CLASSIFIED_SCHEMA = pa.schema([*SECTION_SCHEMA, pa.field("predicted_section", pa.string())])  # type: ignore[union-attr]
    else:  # pragma: no cover - pyarrow not installed
        CLASSIFIED_SCHEMA = None
    
    def get_required_metadata(self) -> Dict[str, str]:
        """
        Get required metadata fields for GlossAPI parquet files.
        
        Returns:
            Dict[str, str]: Dictionary of required metadata fields and their descriptions
        """
        return {
            'pipeline_version': 'GlossAPI pipeline version',
            'created_at': 'ISO format timestamp when the file was created',
            'source_file': 'Original source file that generated this parquet',
            'processing_stage': 'Pipeline processing stage (download, extract, section, etc)'
        }
    
    def validate_schema(self, df: pd.DataFrame, schema_type: str = 'common') -> Tuple[bool, List[str]]:
        """
        Validate that a DataFrame conforms to the specified schema.
        
        Args:
            df: DataFrame to validate
            schema_type: Type of schema to validate against ('common', 'download', 'section', 'classified', 'metadata')
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, missing_columns)
        """
        if schema_type.lower() == 'download':
            required_columns = [field.name for field in self.DOWNLOAD_SCHEMA]
            # Make sure to use the configured url_column
            if self.url_column != 'url' and 'url' in required_columns:
                required_columns.remove('url')
                required_columns.append(self.url_column)
        elif schema_type.lower() == 'section':
            required_columns = [field.name for field in self.SECTION_SCHEMA]
        elif schema_type.lower() == 'classified':
            required_columns = [field.name for field in self.CLASSIFIED_SCHEMA]
        elif schema_type.lower() == 'metadata':
            required_columns = ['filename']
            # Make sure to use the configured url_column
            required_columns.append(self.url_column)  
        else:  # Default to common schema
            required_columns = [field.name for field in self.COMMON_SCHEMA]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        return len(missing_columns) == 0, missing_columns
    
    def add_metadata(self, table: pa.Table, metadata: Dict[str, str]) -> pa.Table:
        """
        Add metadata to a PyArrow Table.
        
        Args:
            table: PyArrow Table to add metadata to
            metadata: Dictionary of metadata to add
            
        Returns:
            pa.Table: Table with added metadata
        """
        # Add pipeline configuration to metadata
        if self.config:
            for key, value in self.config.items():
                if key not in metadata:
                    metadata[f'config_{key}'] = str(value)
        # Convert all metadata values to strings
        metadata_bytes = {k.encode(): str(v).encode() for k, v in metadata.items()}
        
        # Add required metadata if missing
        required_metadata = self.get_required_metadata()
        for key in required_metadata:
            if key not in metadata:
                metadata_bytes[key.encode()] = f"MISSING: {required_metadata[key]}".encode()
        
        return table.replace_schema_metadata(metadata_bytes)
    
    def read_parquet(self, file_path: Union[str, Path], validate: bool = True, schema_type: str = 'common') -> pd.DataFrame:
        """
        Read a parquet file with validation.
        
        Args:
            file_path: Path to parquet file
            validate: Whether to validate the schema
            schema_type: Type of schema to validate against
            
        Returns:
            pd.DataFrame: DataFrame from parquet file
        """
        df = pd.read_parquet(file_path)
        
        if validate:
            is_valid, missing_columns = self.validate_schema(df, schema_type)
            if not is_valid:
                print(f"Warning: Parquet file {file_path} is missing required columns: {missing_columns}")
                
                # Add missing columns with default values
                for col in missing_columns:
                    if col == "font_repair_backend":
                        df[col] = "unknown"
                    elif col == "font_repair_status":
                        df[col] = "skipped"
                    elif col == "font_repair_fonts":
                        df[col] = 0
                    elif col in ("font_repair_valid_rate_before", "font_repair_valid_rate_after"):
                        df[col] = float("nan")
                    elif col in ['id', 'filename', 'title', 'section', 'predicted_section', 'download_error', 'file_ext', 'ocr_pages', 'extract_status']:
                        df[col] = ''
                    elif col in ['row_id', 'download_retry_count', 'source_row', 'url_index', 'ocr_pages_count', 'ocr_pages_unflagged_count']:
                        df[col] = 0
                    elif col in ['page_count', 'pua_char_count', 'control_char_count']:
                        df[col] = pd.NA
                    elif col == 'download_success':
                        df[col] = False
                    elif col == 'is_duplicate':
                        df[col] = False
                    elif col == 'probability':
                        df[col] = 0.0
                    elif col == 'filename_base':
                        df[col] = ''
        
        return df
    
    def find_metadata_parquet(self, directory: Union[str, Path], require_url_column: bool = False) -> Optional[Path]:
        """
        Find the first valid metadata parquet file in a directory.
        
        Looks for parquet files that don't have section-specific columns
        like 'title' and 'header', and prioritizes files with the url_column.
        
        Args:
            directory: Directory to search for parquet files
            require_url_column: If True, require the URL column to be present; if False, only require filename column
            
        Returns:
            Optional[Path]: Path to the first valid metadata parquet, or None if not found
        """
        import logging
        logger = logging.getLogger(__name__)
        
        directory = Path(directory)
        if not directory.exists():
            logger.debug(f"Directory {directory} does not exist")
            return None
            
        # Get all parquet files in the directory
        parquet_files = [
            f for f in directory.glob('**/*.parquet')
            if '.partial.' not in f.name and not f.name.endswith('.partial.parquet') and '.glossapi' not in f.relative_to(directory).parts
        ]
        if not parquet_files:
            logger.debug(f"No parquet files found in {directory}")
            return None
            
        # Check for download_results files first
        download_files = [f for f in parquet_files if 'download_results' in str(f)]
        if download_files:
            logger.debug(f"Found {len(download_files)} download_results files")
        
        # If any download_results parquets exist, check them first
        search_order = download_files + [f for f in parquet_files if f not in download_files]
        metrics_signature = {"filter", "mojibake_badness_score", "greek_badness_score"}
        downloader_signature = {"download_success", "download_error", "filename_base"}

        # First pass: prefer metrics parquet if present anywhere in the tree
        for file_path in search_order:
            try:
                df = pd.read_parquet(file_path)
                columns = df.columns.tolist()
                if 'section' in columns:
                    continue
                if metrics_signature.issubset(set(columns)):
                    logger.info(f"Found metrics parquet with cleaner columns: {file_path}")
                    return file_path
            except Exception as e:
                logger.debug(f"Error reading parquet {file_path}: {e}")
                continue

        # Second pass: look for downloader metadata parquet
        for file_path in search_order:
            try:
                df = pd.read_parquet(file_path)
                columns = df.columns.tolist()

                # Skip section-level parquets – identified by a 'section' column
                if 'section' in columns:
                    logger.debug(f"Skipping section-level parquet: {file_path}")
                    continue

                col_set = set(columns)

                if downloader_signature.issubset(col_set):
                    if require_url_column and self.url_column not in col_set:
                        logger.warning(
                            "Downloader parquet missing required %s column: %s",
                            self.url_column,
                            file_path,
                        )
                        logger.debug(f"Available columns: {columns}")
                        continue
                    logger.info(f"Found downloader metadata parquet: {file_path}")
                    return file_path

                # For other metadata parquets - they don't have title/header but have filename
                if 'filename' in columns:
                    if require_url_column:
                        # Check if required URL column exists
                        if self.url_column in columns:
                            logger.info(f"Found metadata parquet with filename and {self.url_column}: {file_path}")
                            return file_path
                        else:
                            # Missing URL column
                            logger.warning(f"Found parquet with filename column but no {self.url_column} column: {file_path}")
                            logger.debug(f"Available columns: {columns}")
                    else:
                        # URL not required, filename is enough
                        logger.info(f"Found metadata parquet with filename (URL not required): {file_path}")
                        return file_path
                else:
                    logger.debug(f"Found parquet without filename column: {file_path}")
            except Exception as e:
                logger.debug(f"Error reading parquet {file_path}: {e}")
                continue
                
        logger.warning(f"No suitable metadata parquet found in {directory}")
        return None
    
    def ensure_metadata_parquet(self, base_dir: Union[str, Path]) -> Optional[Path]:
        """
        Ensure that a consolidated metadata parquet exists for a pipeline run.

        When the canonical ``download_results/download_results.parquet`` file is
        missing, this method gathers the available artifacts (downloads, markdown,
        metrics, math sidecars, etc.) and synthesises a best-effort metadata table
        so downstream stages always have a consistent entry point.

        Returns:
            Optional[Path]: Path to the ensured parquet, or ``None`` when no rows
            could be inferred.
        """
        import logging
        import math

        base_dir = Path(base_dir)
        download_results_dir = base_dir / "download_results"
        parquet_path = download_results_dir / "download_results.parquet"
        if parquet_path.exists():
            return parquet_path

        download_results_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)

        def _candidates() -> List[Path]:
            return sorted(
                f
                for f in download_results_dir.glob("*.parquet")
                if ".partial." not in f.name and f.name != parquet_path.name
            )

        def _read_existing(candidate: Path) -> pd.DataFrame:
            try:
                return pd.read_parquet(candidate)
            except Exception as exc:
                logger.debug("Failed to read existing metadata parquet %s: %s", candidate, exc)
                return pd.DataFrame()

        # One rule for every record (K4BI-G9): a field holds the value its producer computed, else null. The
        # artifact scan fills what it observes below; nothing here invents a value for a field no producer filled.
        defaults: Dict[str, Any] = {
            self.url_column: pd.NA,
            "filename": "",
            "file_ext": "",
            "document_type": pd.NA,
            "download_success": pd.NA,
            "download_error": pd.NA,
            "download_retry_count": pd.NA,
            "is_duplicate": pd.NA,
            "duplicate_of": pd.NA,
            "source_row": pd.NA,
            "url_index": pd.NA,
            "filename_base": "",
            "filter": pd.NA,
            "extract_status": pd.NA,
            "extract_error": pd.NA,
            "font_repair_status": pd.NA,
            "font_repair_backend": pd.NA,
            "font_repair_fonts": pd.NA,
            "font_repair_valid_rate_before": pd.NA,
            "font_repair_valid_rate_after": pd.NA,
            "needs_ocr": pd.NA,
            "ocr_pages": pd.NA,
            "ocr_pages_count": pd.NA,
            "ocr_pages_unflagged_count": pd.NA,
            "pua_char_count": pd.NA,
            "control_char_count": pd.NA,
            "ocr_success": pd.NA,
            "processing_stage": "",
            "mojibake_badness_score": pd.NA,
            "mojibake_latin_percentage": pd.NA,
            "percentage_greek": pd.NA,
            "char_count_no_comments": pd.NA,
            "is_empty": pd.NA,
            "page_count": pd.NA,
            "pages_total": pd.NA,
            "pages_with_formula": pd.NA,
            "formula_total": pd.NA,
            "formula_avg_pp": pd.NA,
            "formula_p90_pp": pd.NA,
            "phase_recommended": pd.NA,
            "math_enriched": pd.NA,
            "math_items": pd.NA,
            "math_accepted": pd.NA,
            "math_accept_rate": pd.NA,
            "math_time_sec": pd.NA,
        }

        records: Dict[str, Dict[str, Any]] = {}

        def _append_stage(row: Dict[str, Any], stage: str) -> None:
            if not stage:
                return
            current = row.get("processing_stage") or ""
            stages = [part for part in current.split(",") if part]
            if stage not in stages:
                stages.append(stage)
            row["processing_stage"] = ",".join(stages)

        def _row_for(stem: str) -> Dict[str, Any]:
            canonical = canonical_stem(stem)
            row = records.get(canonical)
            if row is None:
                row = {key: value for key, value in defaults.items()}
                row["filename"] = f"{canonical}.pdf"
                row["file_ext"] = "pdf"
                row["filename_base"] = canonical
                records[canonical] = row
            return row

        # 1) Original downloads (PDFs)
        downloads_dir = base_dir / "downloads"
        pdf_paths: List[Path] = []
        if downloads_dir.exists():
            pdf_paths.extend(sorted(p for p in downloads_dir.rglob("*.pdf") if p.is_file()))
        pdf_paths.extend(sorted(p for p in base_dir.glob("*.pdf") if p.is_file()))

        for pdf_path in pdf_paths:
            stem = canonical_stem(pdf_path)
            row = _row_for(stem)
            # No URL is invented for a PDF found on disk; a producer (the downloader) records it.
            row["filename"] = pdf_path.name
            row["file_ext"] = pdf_path.suffix.lstrip(".").lower()
            row["filename_base"] = stem
            row["download_success"] = True
            row["download_error"] = ""
            _append_stage(row, "download")

        # 2) Markdown outputs (extraction + cleaning)
        markdown_sources = [
            (base_dir / "markdown", "extract"),
            (base_dir / "clean_markdown", "clean"),
        ]
        for md_dir, stage in markdown_sources:
            if not md_dir.exists():
                continue
            for md_path in md_dir.glob("*.md"):
                if not md_path.is_file():
                    continue
                stem = canonical_stem(md_path)
                row = _row_for(stem)
                if not row["filename"]:
                    row["filename"] = f"{stem}.pdf"
                row["filename_base"] = stem
                _append_stage(row, "extract")
                if stage == "clean":
                    _append_stage(row, "clean")

        # 3) Latex maps indicate successful math enrichment
        json_dir = base_dir / "json"
        if json_dir.exists():
            for latex_map in json_dir.glob("*.latex_map.jsonl"):
                if not latex_map.is_file():
                    continue
                stem = canonical_stem(latex_map)
                row = _row_for(stem)
                row["math_enriched"] = True
                _append_stage(row, "math")

        # 4) Metrics JSON (page counts, formula counts)
        def _ingest_metrics_payload(stem: str, payload: Dict[str, Any]) -> None:
            row = _row_for(stem)
            page_count = payload.get("page_count")
            if page_count is not None:
                try:
                    row["page_count"] = int(page_count)
                except Exception:
                    row["page_count"] = page_count
            pages = payload.get("pages") or []
            if isinstance(pages, list) and pages:
                row["pages_total"] = len(pages)
                try:
                    formula_counts = [int(p.get("formula_count", 0) or 0) for p in pages]
                except Exception:
                    formula_counts = []
                if formula_counts:
                    total_formula = sum(formula_counts)
                    if pd.isna(row["formula_total"]):
                        row["formula_total"] = total_formula
                    row["formula_avg_pp"] = (
                        float(total_formula) / max(1, len(formula_counts))
                    )
                    formula_counts_sorted = sorted(formula_counts)
                    idx = max(0, math.ceil(0.9 * len(formula_counts_sorted)) - 1)
                    row["formula_p90_pp"] = float(formula_counts_sorted[idx])
                    row["pages_with_formula"] = sum(1 for c in formula_counts if c > 0)

        metrics_dir = json_dir / "metrics" if json_dir.exists() else None
        if metrics_dir and metrics_dir.exists():
            for metrics_file in metrics_dir.glob("*.metrics.json"):
                if not metrics_file.is_file():
                    continue
                stem = canonical_stem(metrics_file)
                try:
                    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(payload, dict):
                    _ingest_metrics_payload(stem, payload)
            for per_page in metrics_dir.glob("*.per_page.metrics.json"):
                if not per_page.is_file():
                    continue
                name = per_page.name
                if not name.endswith(".per_page.metrics.json"):
                    continue
                stem = canonical_stem(per_page)
                try:
                    payload = json.loads(per_page.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(payload, dict):
                    _ingest_metrics_payload(stem, payload)

        # 5) Triage sidecars (phase recommendation, formula stats)
        triage_dir = base_dir / "sidecars" / "triage"
        if triage_dir.exists():
            for triage_file in triage_dir.glob("*.json"):
                if not triage_file.is_file():
                    continue
                try:
                    payload = json.loads(triage_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                stem = canonical_stem(triage_file)
                row = _row_for(stem)
                for key in ("formula_total", "formula_avg_pp", "formula_p90_pp", "pages_total", "pages_with_formula"):
                    if key in payload and pd.isna(row.get(key)):
                        row[key] = payload.get(key)
                if "phase_recommended" in payload and pd.isna(row.get("phase_recommended")):
                    row["phase_recommended"] = payload.get("phase_recommended")

        # 6) Math sidecars (items, acceptance rate, duration)
        math_dir = base_dir / "sidecars" / "math"
        if math_dir.exists():
            for math_file in math_dir.glob("*.json"):
                if not math_file.is_file():
                    continue
                try:
                    payload = json.loads(math_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                stem = canonical_stem(math_file)
                row = _row_for(stem)
                items = payload.get("items")
                accepted = payload.get("accepted")
                time_sec = payload.get("time_sec")
                if items is not None:
                    row["math_items"] = items
                if accepted is not None:
                    row["math_accepted"] = accepted
                if items:
                    try:
                        row["math_accept_rate"] = float(accepted) / float(items) if accepted is not None else row["math_accept_rate"]
                    except Exception:
                        pass
                if time_sec is not None:
                    row["math_time_sec"] = time_sec
                row["math_enriched"] = True
                _append_stage(row, "math")

        # Extraction worker sidecars also exist for failed/no-Markdown documents.
        extraction_columns = ("extract_status", "page_count")
        extract_dir = base_dir / "sidecars" / "extract"
        for sidecar in sorted(extract_dir.glob("*.json")):
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
            except (OSError, ValueError):
                continue
            row = _row_for(sidecar.stem)
            status = payload.get("extract_status") or payload.get("failure_mode")
            if status:
                row["extract_status"] = str(status)
            if payload.get("page_count") is not None:
                row["page_count"] = payload["page_count"]

        df_records = pd.DataFrame(list(records.values())) if records else pd.DataFrame()
        # Ensure column order is stable for readability
        ordered_columns = list(defaults.keys())
        if "filename" not in ordered_columns:
            ordered_columns.insert(0, "filename")
        if self.url_column not in ordered_columns:
            ordered_columns.insert(0, self.url_column)
        df_records = df_records.reindex(columns=ordered_columns, fill_value=pd.NA)
        if not df_records.empty:
            df_records.sort_values(by="filename", inplace=True)

        # The artifact scan above is this writer's evidence. The existing parquet is read, merged and replaced
        # inside one locked section (K2 batch 3), so a concurrent writer's rows are never overwritten by a merge
        # based on an older read. The canonical file and the candidate are re-checked under the lock.
        while True:
            existing_candidates = _candidates()
            target_path: Optional[Path] = existing_candidates[0] if existing_candidates else None
            # Through the metadata choke point (K2F round 3): the resolved physical target is locked, read and
            # replaced (a symlinked candidate keeps its link; a hard-linked one is refused); the caller's spelling is
            # only returned and observed.
            with metadata_target(target_path or parquet_path) as physical:
                if parquet_path.exists():
                    return parquet_path
                current = _candidates()
                if (current[0] if current else None) != target_path:
                    continue  # another writer created a candidate meanwhile: lock that one instead
                existing_df = _read_existing(physical) if target_path is not None else pd.DataFrame()
                return self._merge_metadata_locked(
                    df_records, existing_df, target_path, parquet_path, defaults, ordered_columns,
                    extraction_columns, base_dir, logger, physical=physical)

    @staticmethod
    def _merge_metadata_locked(df_records, existing_df, target_path, parquet_path, defaults, ordered_columns,
                               extraction_columns, base_dir, logger, *, physical):
        """Merge this writer's artifact rows into the existing rows and write ``physical`` (the resolved target the
        caller holds through ``metadata_target``); ``target_path``/``parquet_path`` are returned and observed."""
        if target_path is not None:
            combined = df_records
            if combined.empty and not existing_df.empty:
                combined = existing_df.copy()
            elif not combined.empty and not existing_df.empty and "filename" in existing_df.columns:
                def _merge_unique(records, existing):
                    base_idx = records.set_index("filename", drop=False)
                    # Exact union (S1): integer columns become nullable before the row union, so a value above
                    # 2**53 survives rows the existing table does not have (plain int64 would become float64).
                    existing_idx = _nullable_integers(existing).set_index("filename", drop=False)
                    base_idx = base_idx.combine_first(existing_idx)
                    base_idx.update(existing_idx)
                    # Fresh extraction evidence wins over older parquet state.
                    evidence = records.set_index("filename")
                    for column in extraction_columns:
                        known = evidence[column].notna() & (evidence[column] != "")
                        base_idx.loc[evidence.index[known], column] = evidence.loc[known, column]
                    return base_idx.reset_index(drop=True)

                if not bool(existing_df["filename"].duplicated().any()):
                    combined = _merge_unique(combined, existing_df)
                else:
                    # Occurrence rows (LS6-G04): a skipped duplicate URL and a failed URL both have filename "".
                    # Only a named row is merged by filename; a row without a name matches no artifact and is
                    # carried through as it is. A name on several rows gets the artifact evidence on each
                    # occurrence (occurrence k is merged with the evidence of its own name), so no row is lost,
                    # duplicated or merged into another row.
                    named = _named_filenames(existing_df["filename"]).to_numpy()
                    keyed: Any = existing_df[named]
                    occurrence = keyed.groupby("filename", sort=False).cumcount().to_numpy()
                    parts: List[Any] = [] if len(keyed) else [combined]
                    for k in sorted(set(occurrence.tolist())):
                        layer: Any = keyed[occurrence == k]
                        records: Any = combined if k == 0 else combined[combined["filename"].isin(list(layer["filename"]))]
                        parts.append(_merge_unique(records, layer) if len(records) else layer)
                    if (~named).any():
                        parts.append(existing_df[~named])
                    combined = cast(Any, lossless_concat(parts, what="metadata occurrence rows"))
            elif combined.empty:
                combined = existing_df.copy()

            if combined.empty:
                logger.warning(
                    "Unable to update metadata parquet under %s – no artifacts discovered",
                    base_dir,
                )
                return target_path

            if not existing_df.empty and "download_success" in existing_df.columns and "download_success" in combined.columns:
                # An acquired download keeps its "download" stage after memory_saving=extreme deleted the file
                # (the artifact scan above sees only files still present).
                stages = combined["processing_stage"].fillna("").astype(str) if "processing_stage" in combined.columns else pd.Series("", index=combined.index)
                acquired = (combined["download_success"] == True) & ~stages.map(lambda value: "download" in value.split(","))  # noqa: E712
                if acquired.any():
                    combined.loc[acquired, "processing_stage"] = stages[acquired].map(lambda value: ",".join(["download", *[p for p in value.split(",") if p]]))
            for column, default in defaults.items():
                if column not in combined.columns:
                    combined[column] = default
            # Exact union (S1): every column of the existing table (the caller's own, the HTTP downloader's
            # selection_id, resource_identity, base_domain, ...) is kept after the fixed columns, with its type.
            extra = [column for column in [*existing_df.columns, *combined.columns] if column not in ordered_columns]
            combined = combined.reindex(columns=[*ordered_columns, *dict.fromkeys(extra)], fill_value=pd.NA)
            combined.sort_values(by="filename", inplace=True)
            _write_metadata_parquet_locked(combined, physical, observed=target_path)
            logger.info(
                "Updated existing metadata parquet with %d row(s): %s",
                len(combined),
                target_path,
            )
            return target_path

        if df_records.empty:
            logger.warning(
                "Unable to synthesise metadata parquet under %s – no artifacts discovered",
                base_dir,
            )
            return None

        df_records.sort_values(by="filename", inplace=True)
        _write_metadata_parquet_locked(df_records, physical, observed=parquet_path)
        logger.info(
            "Synthesised metadata parquet with %d row(s): %s",
            len(df_records),
            parquet_path,
        )
        return parquet_path

    @staticmethod
    def normalize_metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Expose metadata normalisation for callers who need in-memory updates."""

        return _prepare_metadata_frame(df)

    def write_metadata_parquet(self, df: pd.DataFrame, parquet_path: Union[str, Path]) -> None:
        """Persist metadata updates with schema/dtype normalisation and locking."""

        target = Path(parquet_path)
        _write_metadata_parquet(df, target)

    def is_valid_metadata_parquet(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a parquet file conforms to the metadata schema used by downloader.
        
        Args:
            filepath: Path to the parquet file to check
            
        Returns:
            bool: True if the file has the required metadata fields
        """
        try:
            schema = pq.read_schema(filepath)
            # Check for url_column (which might be custom) and filename
            required_fields = [self.url_column, 'filename']
            return all(field in schema.names for field in required_fields)
        except Exception:
            return False
            
    def create_basic_metadata_parquet(self, markdown_dir: Union[str, Path], output_dir: Union[str, Path]) -> Union[Path, None]:
        """
        Create a simple metadata parquet file from a directory of markdown files.
        This is used when there is no existing parquet file to update.
        
        Args:
            markdown_dir: Directory containing markdown files
            output_dir: Directory where to create the parquet file
            
        Returns:
            Path: Path to the created parquet file, or None if creation failed
        """
        pipeline_root = Path(output_dir)
        ensured = self.ensure_metadata_parquet(pipeline_root)
        if ensured is not None:
            return ensured

        try:
            markdown_dir = Path(markdown_dir)
            
            # Create output directory if it doesn't exist
            download_results_dir = pipeline_root / "download_results"
            os.makedirs(download_results_dir, exist_ok=True)
            
            # Get all markdown files in the input directory
            markdown_files = list(markdown_dir.glob("*.md"))
            if not markdown_files:
                print(f"No markdown files found in {markdown_dir}")
                return None
                
            # Create a DataFrame with just filenames
            data = []
            for md_file in markdown_files:
                entry = {
                    'filename': md_file.name,
                    self.url_column: ""  # Minimal URL placeholder
                }
                data.append(entry)
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Set output path for the parquet file
            output_path = download_results_dir / "download_results.parquet"
            
            # Write to parquet without adding complex metadata. Create-if-missing inside the metadata lock
            # (CP-K2a F1): a file another writer created meanwhile is kept, never replaced by this seed.
            created = []
            def _seed(current):
                if output_path.exists():
                    return None
                created.append(True)
                return df
            update_metadata_parquet(output_path, _seed,
                                    serialize=lambda frame, path: pq.write_table(pa.Table.from_pandas(frame), path))
            if not created:
                return output_path

            print(f"Created new metadata parquet file at {output_path}")
            return output_path
            
        except MetadataPolicyError:
            raise  # policy refusals are never swallowed (CP-K2e F5)
        except Exception as e:
            print(f"Error creating metadata parquet file: {e}")
            return None
            
    def is_download_result_parquet(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a parquet file contains download results with success/error information.
        
        Args:
            filepath: Path to the parquet file to check
            
        Returns:
            bool: True if the file has download result fields
        """
        try:
            schema = pq.read_schema(filepath)
            # Check for download result fields
            required_fields = ['download_success', 'filename']
            return all(field in schema.names for field in required_fields)
        except Exception:
            return False
            
    def is_sections_parquet(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a parquet file contains section data from extracted files.
        This identifies the second type of parquet in the pipeline - the sections parquet.
        
        Args:
            filepath: Path to the parquet file to check
            
        Returns:
            bool: True if the file has section data fields
        """
        try:
            schema = pq.read_schema(filepath)
            # Check for required section fields
            required_fields = ['filename', 'title', 'content', 'section']
            return all(field in schema.names for field in required_fields)
        except Exception:
            return False
        
    def add_processing_stage(self, df: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Add or update processing stage column in a DataFrame.
        
        Args:
            df: Input DataFrame to update
            stage: Processing stage value to set (e.g., 'downloaded', 'extracted', 'classified')
            
        Returns:
            pd.DataFrame: Updated DataFrame with processing_stage column
        """
        df['processing_stage'] = stage
        return df
        
    def verify_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if a DataFrame contains all required columns and return missing ones.
        
        Args:
            df: DataFrame to check
            required_columns: List of column names that should be present
            
        Returns:
            Tuple containing:
            - bool: True if all required columns are present
            - List[str]: List of missing columns (empty if all present)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        return (len(missing_columns) == 0, missing_columns)
    
    def write_parquet(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        schema_type: str = 'common',
        validate: bool = True
    ) -> None:
        """
        Write a DataFrame to parquet with standard schema and metadata.
        
        Args:
            df: DataFrame to write
            file_path: Path to write parquet file
            metadata: Dictionary of metadata to include
            schema_type: Type of schema to use
            validate: Whether to validate the schema before writing
        """
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Validate and fix schema if needed
        if validate:
            is_valid, missing_columns = self.validate_schema(df_copy, schema_type)
            if not is_valid:
                print(f"Adding missing columns to DataFrame: {missing_columns}")
                
                # Add missing columns with default values
                for col in missing_columns:
                    if col == "font_repair_backend":
                        df_copy[col] = "unknown"
                    elif col == "font_repair_status":
                        df_copy[col] = "skipped"
                    elif col == "font_repair_fonts":
                        df_copy[col] = 0
                    elif col in ("font_repair_valid_rate_before", "font_repair_valid_rate_after"):
                        df_copy[col] = float("nan")
                    elif col in ['id', 'filename', 'title', 'section', 'predicted_section', 'download_error', 'file_ext', 'ocr_pages', 'extract_status']:
                        df_copy[col] = ''
                    elif col in ['row_id', 'download_retry_count', 'source_row', 'url_index', 'ocr_pages_count', 'ocr_pages_unflagged_count']:
                        df_copy[col] = 0
                    elif col in ['page_count', 'pua_char_count', 'control_char_count']:
                        df_copy[col] = pd.NA
                    elif col == 'download_success':
                        df_copy[col] = False
                    elif col == 'is_duplicate':
                        df_copy[col] = False
                    elif col == 'probability':
                        df_copy[col] = 0.0
                    elif col == 'filename_base':
                        df_copy[col] = ''
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df_copy)
        
        # Add metadata if provided
        if metadata:
            table = self.add_metadata(table, metadata)
        
        # Write to parquet, through the metadata choke point (K2F round 3: resolved target, lock, atomic replace)
        replace_metadata_parquet(file_path, df_copy, serialize=lambda _frame, path: pq.write_table(table, path))
        print(f"Parquet file written to {file_path} with schema type '{schema_type}'")


def fold_download_rows(output_dir: Union[str, Path], published: Union[str, Path]) -> list:
    """Carry a download publication's rows into the canonical metadata table when the corpus already has one.

    The download serializer writes ``download_results_<manifest>.parquet``; the metadata readers and the export
    select ``download_results.parquet`` when it exists (a corpus that began with local inputs). Without this the URL,
    success and identity of a later download never reached the row the export joins: CP-K4a F1 (streaming) and
    CP-K4ac2 R1 (sequential ``download()`` → ``extract`` → ``clean`` → ``jsonl``). Under the metadata lock the
    download-owned columns of each published row replace that filename's values (other columns are kept) and new
    filenames are added. Identity-safe: a canonical row that already records a *different* URL or resource identity
    for the same filename is never overwritten; those filenames are returned (and the published file keeps them).
    """
    canonical = Path(output_dir)/'download_results'/'download_results.parquet'
    published = Path(published)
    if not canonical.is_file() or not published.is_file() or canonical.resolve() == published.resolve():
        return []
    rows = pd.read_parquet(published)
    if 'filename' not in rows.columns or rows.empty:
        return []
    conflicts: list = []

    def text(value):
        return value if isinstance(value, str) and value else None

    # Occurrence rows (LS6-G04): a skipped duplicate URL or a failed download has no filename, so it is never keyed
    # by filename (two of them are two decisions, not one "" row, and never a filename conflict). Each is matched by
    # its URL instead: it replaces the canonical row without a filename that records the same URL (a re-fold or a
    # retried failure), else it is added.
    named = _named_filenames(rows['filename']).to_numpy()
    urls = rows['url'].map(text) if 'url' in rows.columns else pd.Series(None, index=rows.index, dtype=object)
    occurrence = ~named & urls.notna().to_numpy()
    folded: Any = _nullable_integers(rows[(named & ~rows['filename'].where(named).duplicated(keep='last').to_numpy())
                                     | (occurrence & ~urls.where(occurrence).duplicated(keep='last').to_numpy())])
    blank: Any = folded[~_named_filenames(folded['filename']).to_numpy()]
    rows: Any = folded[_named_filenames(folded['filename']).to_numpy()]
    if folded.empty:
        return []

    def nonempty_concat(parts):
        filled = [part for part in parts if len(part)]
        if len(filled) > 1:
            return lossless_concat(filled, what='folded download results')
        return filled[0] if filled else parts[0]

    def fold(current):
        if current.empty or 'filename' not in current.columns:
            return folded
        current = _nullable_integers(current)
        current_named = _named_filenames(current['filename']).to_numpy()
        recorded = current.loc[current_named].drop_duplicates('filename', keep='last').set_index('filename')
        keep = []
        for name, url, identity in zip(rows['filename'], rows['url'] if 'url' in rows.columns else [None]*len(rows),
                                       rows['resource_identity'] if 'resource_identity' in rows.columns else [None]*len(rows)):
            if name in recorded.index:
                old_url = text(recorded.at[name, 'url']) if 'url' in recorded.columns else None
                old_identity = text(recorded.at[name, 'resource_identity']) if 'resource_identity' in recorded.columns else None
                if (old_url and text(url) and old_url != text(url)) or (old_identity and text(identity) and old_identity != text(identity)):
                    conflicts.append(name)
                    continue
            keep.append(name)
        update = rows[rows['filename'].isin(keep)]
        known = current['filename'].isin(list(update['filename']))
        merged = current[known].set_index('filename', drop=False)
        indexed = update.set_index('filename', drop=False)
        merged = indexed.loc[indexed.index.intersection(merged.index)].combine_first(merged).reindex(
            columns=[*current.columns, *[c for c in update.columns if c not in current.columns]])
        added = update[~update['filename'].isin(list(current['filename']))]
        if len(blank):
            current_urls = pd.Series(current['url'].map(text) if 'url' in current.columns else None, index=current.index, dtype=object)
            again = ~current_named & current_urls.isin(list(blank['url'])).to_numpy()
            if again.any():
                replaced = current[again].reset_index(drop=True)
                published_rows = blank.set_index(blank['url']).loc[list(current_urls[again])].reset_index(drop=True)
                merged = nonempty_concat([merged.reset_index(drop=True), published_rows.combine_first(replaced).reindex(
                    columns=list(merged.columns))])
                known = known | pd.Series(again, index=current.index)
            added = nonempty_concat([added, blank[~blank['url'].isin(list(current_urls[again]))]])
        result = lossless_concat([current[~known], merged.reset_index(drop=True), added], what='folded download results')
        return result.sort_values('filename', kind='stable').reset_index(drop=True)[
            [*current.columns, *[c for c in result.columns if c not in current.columns]]]

    update_metadata_parquet(canonical, fold)
    if conflicts:
        import logging
        logging.getLogger(__name__).warning('Download rows not folded into %s: their filenames already record another URL: %s', canonical, conflicts[:10])
    return conflicts
