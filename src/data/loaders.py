"""Read/inspect/write tabular data without modifying raw sources.

Design rules (task book §5.1, §5.3):

* readers only read; the writer is the single place that writes derived data and
  it refuses any destination below ``data/raw/`` unless the caller passes the
  explicit ``allow_raw=True`` escape hatch (used only by tests that need to prove
  the refusal, never by the pipeline);
* inspection uses parquet **file metadata** for row counts and schema so a
  multi-gigabyte file is never loaded just to be counted;
* chunked iteration is the future big-data entry point: it never materialises a
  whole table;
* csv/tsv/parquet are supported; anything else raises with a clear message.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.data.contracts import ContractError
from src.utils import PROJECT_ROOT, ensure_dir, resolve_path

#: Suffixes that the reader/writer understand.
READABLE_SUFFIXES: frozenset[str] = frozenset({".csv", ".tsv", ".parquet"})
WRITABLE_SUFFIXES: frozenset[str] = frozenset({".csv", ".tsv", ".parquet"})

SEPARATORS: dict[str, str] = {".csv": ",", ".tsv": "\t"}

_SAMPLE_ROWS = 5


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def _raw_path_parts(path: Path) -> bool:
    """True when the resolved path has a ``data/raw`` component pair."""
    parts = [part.lower() for part in path.parts]
    for left, right in zip(parts, parts[1:]):
        if left == "data" and right == "raw":
            return True
    return False


def assert_not_raw_path(path: str | Path) -> Path:
    """Return the resolved path or raise if it lies below a ``data/raw`` directory.

    Both the project ``data/raw/`` tree and any other resolved ``.../data/raw/...``
    destination are refused, so a caller can never accidentally overwrite the
    protected raw tree by using an absolute temporary path that mimics it.
    """
    resolved = resolve_path(path)
    project_raw = (PROJECT_ROOT / "data" / "raw").resolve()
    if _raw_path_parts(resolved) or resolved == project_raw or project_raw in resolved.parents:
        raise ContractError(
            f"Refusing to write into a raw-data directory: {resolved}. "
            "Raw data is read-only; derived files must go to data/processed/ or a new versioned "
            "path. Pass allow_raw=True only for an explicit, audited exception."
        )
    return resolved


def _require_supported(path: Path, *, purpose: str) -> str:
    suffix = path.suffix.lower()
    allowed = READABLE_SUFFIXES if purpose == "read" else WRITABLE_SUFFIXES
    if suffix not in allowed:
        raise ValueError(
            f"Unsupported table format {suffix!r} for {purpose}ing {path}; "
            f"expected one of {sorted(allowed)}."
        )
    return suffix


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def read_table(path: str | Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a whole csv/tsv/parquet file, optionally selecting ``columns``."""
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Table not found: {resolved}")
    suffix = _require_supported(resolved, purpose="read")
    if columns is not None and not isinstance(columns, (list, tuple)):
        raise TypeError("columns must be a list of names or None.")
    if suffix == ".parquet":
        try:
            return pd.read_parquet(resolved, columns=list(columns) if columns else None)
        except KeyError as exc:
            raise ContractError(
                f"Requested columns are missing from {resolved}: {exc}"
            ) from exc
    try:
        return pd.read_csv(
            resolved,
            sep=SEPARATORS[suffix],
            usecols=list(columns) if columns else None,
        )
    except ValueError as exc:
        # pandas raises ValueError for missing usecols; make the contract explicit.
        raise ContractError(f"Cannot read columns {columns!r} from {resolved}: {exc}") from exc


def iter_table_batches(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    batch_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Yield the table in chunks of at most ``batch_size`` rows.

    Parquet files are streamed row-group by row-group through
    ``pyarrow.parquet.ParquetFile.iter_batches``; csv/tsv files are streamed with
    ``pandas.read_csv(chunksize=...)``.  No chunk exceeds ``batch_size`` rows and
    the full table is never materialised.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Table not found: {resolved}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
    suffix = _require_supported(resolved, purpose="read")
    if suffix == ".parquet":
        parquet = pq.ParquetFile(resolved)
        try:
            for batch in parquet.iter_batches(
                batch_size=batch_size, columns=list(columns) if columns else None
            ):
                if batch.num_rows:
                    yield batch.to_pandas()
        except KeyError as exc:
            raise ContractError(
                f"Requested columns are missing from {resolved}: {exc}"
            ) from exc
        return
    try:
        reader = pd.read_csv(
            resolved,
            sep=SEPARATORS[suffix],
            usecols=list(columns) if columns else None,
            chunksize=batch_size,
        )
        for chunk in reader:
            yield chunk
    except ValueError as exc:
        raise ContractError(f"Cannot stream {resolved} with columns {columns!r}: {exc}") from exc


def inspect_table(path: str | Path) -> dict:
    """Cheap table summary: ``path``, ``bytes``, ``rows``, ``schema``, ``sample``.

    Parquet row count and schema come from file metadata (no data load); csv/tsv
    rows are counted by streaming chunks.  ``sample`` is the first rows rendered
    as a string for human inspection.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Table not found: {resolved}")
    suffix = _require_supported(resolved, purpose="read")
    size = int(resolved.stat().st_size)
    if suffix == ".parquet":
        parquet = pq.ParquetFile(resolved)
        rows = int(parquet.metadata.num_rows)
        schema = {
            field.name: f"{field.type}" for field in parquet.schema_arrow
        }
        first = next(parquet.iter_batches(batch_size=_SAMPLE_ROWS), None)
        sample_frame = first.to_pandas() if first is not None else pd.DataFrame(columns=list(schema))
    else:
        rows = 0
        sample_frame = pd.DataFrame()
        for chunk in pd.read_csv(  # streaming; the full file is never held in memory
            resolved, sep=SEPARATORS[suffix], chunksize=100_000
        ):
            rows += int(len(chunk))
            if sample_frame.empty:
                sample_frame = chunk.head(_SAMPLE_ROWS)
        schema = {column: str(dtype) for column, dtype in sample_frame.dtypes.items()}
    return {
        "path": str(resolved),
        "bytes": size,
        "rows": rows,
        "schema": schema,
        "sample": sample_frame.to_string(),
    }


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def write_table(frame: pd.DataFrame, path: str | Path, *, allow_raw: bool = False) -> Path:
    """Write ``frame`` to csv/tsv/parquet and return the resolved destination.

    The destination may not lie below ``data/raw`` (unless ``allow_raw=True``),
    and an existing file is never overwritten silently by the pipeline: callers
    are expected to provide a fresh versioned name.  A ``DatetimeIndex`` is kept
    as the index column so timestamps survive the round-trip.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"write_table expects a pandas DataFrame, got {type(frame).__name__}.")
    resolved = resolve_path(path)
    if not allow_raw:
        assert_not_raw_path(resolved)
    suffix = _require_supported(resolved, purpose="write")
    ensure_dir(resolved.parent)
    keep_index = isinstance(frame.index, pd.DatetimeIndex)
    if suffix == ".parquet":
        frame.to_parquet(resolved, index=keep_index)
    else:
        frame.to_csv(resolved, sep=SEPARATORS[suffix], index=keep_index)
    return resolved


__all__ = [
    "READABLE_SUFFIXES",
    "WRITABLE_SUFFIXES",
    "assert_not_raw_path",
    "inspect_table",
    "iter_table_batches",
    "read_table",
    "write_table",
]
