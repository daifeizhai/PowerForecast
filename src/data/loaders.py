"""Read raw tables without modifying source files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.utils import resolve_path


def locate_source(source: dict) -> Path:
    candidates = [source["path"], *source.get("fallback_paths", [])]
    for candidate in candidates:
        path = resolve_path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError("No dataset file found. Tried: " + ", ".join(candidates))


def read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = resolve_path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", usecols=columns)
    raise ValueError(f"Unsupported table format: {suffix}. Use parquet, csv, or tsv.")


def load_source(source: dict) -> pd.DataFrame:
    columns = list(source["columns"].values())
    if source.get("timestamp"):
        columns.append(source["timestamp"])
    return read_table(locate_source(source), list(dict.fromkeys(columns)))


def inspect_table(path: str | Path) -> dict:
    path = resolve_path(path)
    if path.suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(path)
        first_batch = next(parquet.iter_batches(batch_size=5), None)
        sample = first_batch.to_pandas() if first_batch is not None else pd.DataFrame()
        rows = parquet.metadata.num_rows
        schema = {field.name: str(field.type) for field in parquet.schema_arrow}
    else:
        sample = read_table(path)
        rows = len(sample)
        schema = {column: str(dtype) for column, dtype in sample.dtypes.items()}
        sample = sample.head(5)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "schema": schema,
        "sample": sample.to_string(),
    }
