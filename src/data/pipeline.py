"""Config-driven preparation with immutable raw inputs and quality metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.adapters import adapt_frame
from src.data.cleaning import clean_frame, quality_report
from src.data.loaders import load_source, locate_source
from src.data.resampling import resample_frame
from src.utils import config_digest, resolve_path, write_json


def prepare_dataset(config: dict, output: str | Path | None = None, start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, dict]:
    if config.get("frequency", "15min") != "15min":
        raise ValueError("This framework's 96/672-step tasks require frequency=15min.")
    output = resolve_path(output or config["processed_path"])
    raw_root = resolve_path("data/raw")
    sources = {name: locate_source(spec) for name, spec in config["sources"].items()}
    protected = [raw_root, *[path.parent for path in sources.values()]]
    if any(output.is_relative_to(directory) for directory in protected):
        raise ValueError("Prepared output must not be written into a raw source directory.")
    frames = []
    source_info = {}
    for name, spec in config["sources"].items():
        raw = adapt_frame(load_source(spec), spec, config["timezone"])
        # Mask physically invalid source values BEFORE interval aggregation.
        # With fill_limit=0 this changes no temporal information.
        source_cleaning = {**config.get("cleaning", {}), "ffill_limit": 0}
        raw = clean_frame(raw, raw.columns[0], source_cleaning)
        if start:
            raw = raw.loc[start:]
        if end:
            raw = raw.loc[:end]
        if raw.empty:
            raise ValueError(f"Source {name} has no rows in the requested interval.")
        frames.append(resample_frame(raw, config.get("frequency", "15min")))
        stat = sources[name].stat()
        source_info[name] = {"path": str(sources[name]), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    # Use the overlap of source time ranges, retaining missing interior bins.
    overlap_start, overlap_end = max(frame.index.min() for frame in frames), min(frame.index.max() for frame in frames)
    if overlap_start > overlap_end:
        raise ValueError("Configured sources have no overlapping time range.")
    frame = pd.concat(frames, axis=1).loc[overlap_start:overlap_end]
    if frame.columns.has_duplicates:
        raise ValueError("Canonical column names must be unique across sources.")
    for column, formula in config.get("derived_columns", {}).items():
        components = formula["sum"]
        # A missing component makes the derived label missing as well.
        frame[column] = frame[components].sum(axis=1, min_count=len(components))
    frame = clean_frame(frame, config["target"], config.get("cleaning", {}))
    report = {
        "dataset": config["name"],
        "target": config["target"],
        "target_unit": config.get("target_unit", "source_units"),
        "derived_columns": config.get("derived_columns", {}),
        "timezone_verified": config.get("timezone_verified", False),
        "timestamp_assumption": config["timezone"],
        "interval_label": "start; [timestamp, timestamp + 15 minutes)",
        "config_digest": config_digest(config),
        "requested_start": start,
        "requested_end": end,
        "sources": source_info,
        "quality": quality_report(frame),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    write_json(output.with_suffix(".metadata.json"), report)
    return frame, report
