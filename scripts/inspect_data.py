"""Read-only inspection of a raw table or of the configured synthetic dataset.

Usage (from the project root, CPU only)::

    # one raw csv/tsv/parquet file
    & .\\.venv\\Scripts\\python.exe -m scripts.inspect_data --path data/processed/example.parquet --rows 3

    # the synthetic dataset built from config (quality report, no writes except --output)
    & .\\.venv\\Scripts\\python.exe -m scripts.inspect_data --config configs/datasets/synthetic.yaml

This tool never modifies the inspected data: it only reads it, and the optional
``--output`` JSON report refuses to be written below ``data/raw/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/inspect_data.py` as well as -m
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.contracts import TARGET_COLUMN
from src.data.loaders import assert_not_raw_path, inspect_table, iter_table_batches
from src.data.preprocessing import quality_report, validate_frame
from src.data.synthetic import SyntheticDataset, build_synthetic_dataset_from_config
from src.utils import load_config, resolve_path, write_json

DEFAULT_CONFIG = "configs/datasets/synthetic.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one raw table (schema, row count, sample) or build the configured synthetic "
            "dataset and print its data-quality report. Read-only for the inspected data."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path", help="inspect one csv/tsv/parquet file")
    group.add_argument(
        "--config",
        help=f"build and quality-report the configured synthetic dataset (e.g. {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="number of sample rows to show or export (default: 5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional JSON report path (never data/raw/)",
    )
    return parser


def _parse_rows(parser: argparse.ArgumentParser, rows: int) -> int:
    if rows < 0:
        parser.error("--rows must be non-negative")
    return int(rows)


def _sample_text(path: Path, rows: int) -> str:
    if rows == 0:
        return ""
    try:
        first = next(iter_table_batches(path, batch_size=max(rows, 1)))
    except StopIteration:
        return ""
    return first.head(rows).to_string()


def _inspect_path(path: str, rows: int) -> dict[str, Any]:
    report = inspect_table(path)
    if rows != 5:
        report["sample"] = _sample_text(resolve_path(path), rows)
    report["mode"] = "file"
    report["sample_rows"] = rows
    return report


def _inspect_config(config_path: str, rows: int) -> dict[str, Any]:
    config = load_config(config_path)
    dataset: SyntheticDataset = build_synthetic_dataset_from_config(config)
    frequency = dataset.spec.native_frequency or "15min"
    validation = validate_frame(dataset.frame, dataset.spec)
    quality = quality_report(dataset.frame, dataset.spec, frequency=frequency)
    report: dict[str, Any] = {
        "mode": "synthetic_config",
        "config_path": str(resolve_path(config_path)),
        "dataset": {
            "dataset_id": dataset.spec.dataset_id,
            "site_id": dataset.spec.site_id,
            "domain_id": dataset.spec.domain_id,
            "role": dataset.spec.role.value,
            "synthetic": bool(dataset.spec.synthetic),
            "rows": int(len(dataset.frame)),
            "columns": [str(column) for column in dataset.frame.columns],
            "target_column": TARGET_COLUMN,
            "target_present": TARGET_COLUMN in dataset.frame.columns,
            "feature_names": list(dataset.spec.feature_names),
            "feature_count": len(dataset.spec.feature_names),
            "anchor": dataset.spec.trace_anchor_timestamp,
            "frequency": frequency,
            "timezone": dataset.spec.timezone,
            "timezone_verified": bool(dataset.spec.timezone_verified),
            "target_spec": dataset.spec.target.to_dict() if dataset.spec.target is not None else None,
        },
        "quality": quality,
        "validation": validation,
        "sample": dataset.frame.head(rows).to_string(),
        "sample_rows": rows,
        "unlabeled_sources": [],
    }
    for source in dataset.unlabeled_sources:
        report["unlabeled_sources"].append(
            {
                "dataset_id": source.dataset_id,
                "role": source.spec.role.value,
                "target": None,
                "rows": int(len(source.frame)),
                "feature_names": list(source.feature_names),
                "feature_count": len(source.feature_names),
                "contains_power_column": TARGET_COLUMN in source.frame.columns,
                "quality": quality_report(source.frame, source.spec, frequency=frequency),
                "validation": validate_frame(source.frame, source.spec),
                "sample": source.frame.head(rows).to_string(),
            }
        )
    return report


def _print_file_report(report: dict[str, Any]) -> None:
    print(f"File:  {report['path']}")
    print(f"Bytes: {report['bytes']}")
    print(f"Rows:  {report['rows']}")
    print("Schema:")
    for name, dtype in report["schema"].items():
        print(f"  {name}: {dtype}")
    print("Sample:")
    print(report["sample"])


def _print_quality(quality: dict[str, Any], *, header: str) -> None:
    print(header)
    print(
        f"  rows={quality['rows']} start={quality['start']} end={quality['end']} "
        f"frequency={quality['frequency']}"
    )
    print(
        f"  timezone={quality['timezone']} verified={quality['timezone_verified']} "
        f"anchor_matches_first={quality['anchor_matches_first_timestamp']}"
    )
    print(
        f"  duplicate_timestamps={quality['duplicate_timestamps']} "
        f"missing_intervals={quality['missing_intervals']} "
        f"off_grid_rows={quality['off_grid_rows']}"
    )
    print(
        f"  trace_days={quality['trace_day_count']} "
        f"complete_trace_days={quality['complete_trace_days']} "
        f"target_present={quality['target_present']} "
        f"target_coverage={quality['target_coverage_count']}"
    )
    print("  NaN fraction per column:")
    for column, fraction in quality["nan_fraction"].items():
        print(f"    {column}: {fraction:.4%} ({quality['nan_counts'][column]} cells)")
    if quality["out_of_range_counts"]:
        print("  out-of-physical-range counts:")
        for column, count in quality["out_of_range_counts"].items():
            print(f"    {column}: {count}")
    else:
        print("  out-of-physical-range counts: none")


def _print_config_report(report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    print(f"Dataset: {dataset['dataset_id']} (synthetic={dataset['synthetic']})")
    print(f"  role={dataset['role']} frequency={dataset['frequency']} anchor={dataset['anchor']}")
    print(f"  columns={dataset['columns']}")
    _print_quality(report["quality"], header="Supervised target quality:")
    print("  sample:")
    print(report["sample"])
    for source in report["unlabeled_sources"]:
        print(
            f"Unlabelled source: {source['dataset_id']} "
            f"(F={source['feature_count']}, features={source['feature_names']}, "
            f"contains_power_column={source['contains_power_column']})"
        )
        _print_quality(source["quality"], header="  Source quality:")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    rows = _parse_rows(parser, args.rows)
    try:
        if args.path:
            report = _inspect_path(args.path, rows)
            _print_file_report(report)
        else:
            report = _inspect_config(args.config, rows)
            _print_config_report(report)
        if args.output:
            output = assert_not_raw_path(args.output)
            write_json(output, report)
            print(f"JSON report written: {output}")
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
