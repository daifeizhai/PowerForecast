"""Stage 1 target-domain data preparation CLI (task book §3, Milestone 2).

Usage (from the project root, CPU only)::

    & .\\.venv\\Scripts\\python.exe -m scripts.prepare_target_power \
        --config configs/datasets/nlr_target.yaml

The command reads the read-only raw NLR ESIF parquet files, builds the complete
anchored 15-minute facility-power label
(``facility_power_kw = it + cooling + hvac + plug_and_light + pump``, label type
``facility_derived_kw``), resamples the target-domain weather covariates onto the
same grid, and writes the Stage 1 artefacts under ``data/processed/``:

* ``processed_target_power.parquet``     core target columns only
* ``target_components.parquet``          component interval means (label composition only)
* ``processed_target_features.parquet``  long-format weather features (role target_only)
* ``target_metadata.json``               label/timezone/anchor/provenance metadata
* ``target_quality.json``                coverage/sampling/selection quality report

Nothing is ever written below ``data/raw/``; passing ``--out data/raw/...`` is
refused before any file is created. Existing Stage 1 artefact names are
regenerated atomically (raw data and unrelated files are untouched).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/prepare_target_power.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import target_power
from src.data.loaders import assert_not_raw_path, inspect_table
from src.utils import (
    file_metadata,
    load_config,
    resolve_path,
    utc_timestamp,
)

DEFAULT_CONFIG = "configs/datasets/nlr_target.yaml"
DEFAULT_OUT_DIR = "data/processed"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Stage 1 target-domain facility-power label and target-domain weather "
            "features from the read-only NLR ESIF raw parquet files, then write the Stage 1 "
            "artefacts under data/processed/."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"declared dataset YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "optional inclusive raw-sample lower bound (ISO timestamp, UTC). Default: read the "
            "whole archive so the excluded pre-2016 era is counted exactly."
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        help="optional exclusive raw-sample upper bound (ISO timestamp, UTC).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"processed output directory (default: the config value, {DEFAULT_OUT_DIR}); "
        "any path below data/raw/ is refused.",
    )
    return parser


def _source_metadata(config: dict[str, Any], raw_counts: dict[str, int]) -> list[dict[str, Any]]:
    """Raw-file metadata with streamed SHA256 hashes (raw files are never loaded whole)."""
    entries: list[dict[str, Any]] = []
    for role, key in (
        ("power_label_source", "power"),
        ("target_domain_weather_source", "weather"),
    ):
        path = config["sources"][key]["path"]
        metadata = file_metadata(path)
        summary = inspect_table(path)
        entries.append(
            {
                "role": role,
                "path": metadata["path"],
                "bytes": metadata["bytes"],
                "modified_at": metadata["modified_at"],
                "sha256": metadata["sha256"],
                "rows": summary["rows"],
                "schema": summary["schema"],
                "raw_data_modified": False,
                "raw_rows_loaded_for_this_run": raw_counts.get(key),
            }
        )
    return entries


def _first_31_day_summary(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    points_per_day = int(config["points_per_day"])
    anchor = frame.index[0]
    window = frame.iloc[: points_per_day * 31]
    return {
        "points_per_day": points_per_day,
        "expected_slots": points_per_day * 31,
        "actual_slots": int(len(window)),
        "valid_slots": int(window["power_valid"].sum()),
        "valid_fraction": float(window["power_valid"].mean()) if len(window) else 0.0,
        "start": window.index[0].isoformat() if len(window) else None,
        "end": window.index[-1].isoformat() if len(window) else None,
        "facility_power_kw_mean": float(window["power_kw"].mean()) if len(window) else None,
        "facility_power_kw_min": float(window["power_kw"].min()) if len(window) else None,
        "facility_power_kw_max": float(window["power_kw"].max()) if len(window) else None,
        "anchor": anchor.isoformat() if len(frame) else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = copy.deepcopy(load_config(args.config))
        if args.out is not None:
            config["outputs"]["out_dir"] = str(args.out)
        if args.start is not None:
            # A bounded CLI run declares its own trace anchor so the debug frame is not
            # padded with empty grid slots between the canonical anchor and the slice start.
            bounded_anchor = pd.Timestamp(args.start).floor(config["frequency"])
            if bounded_anchor.tzinfo is not None:
                bounded_anchor = bounded_anchor.tz_convert("UTC").tz_localize(None)
            config["analysis"]["window_start"] = bounded_anchor.isoformat()
        output_dir = resolve_path(config["outputs"]["out_dir"])
        # Refuse a data/raw destination before doing any expensive work.
        assert_not_raw_path(output_dir)

        power_path = config["sources"]["power"]["path"]
        timestamp_field = config["sources"]["power"]["timestamp_field"]
        components = list(config["label"]["derived_columns"])
        weather_path = config["sources"]["weather"]["path"]
        weather_timestamp = config["sources"]["weather"]["timestamp_field"]
        weather_fields = [
            str(field["source_field"])
            for field in config["sources"]["weather"]["fields"].values()
        ]
        weather_columns = [weather_timestamp, *weather_fields]
        unit_column = config["sources"]["weather"].get("raw_unit_column")
        if unit_column:
            weather_columns.append(str(unit_column))

        print(f"Reading power components from {power_path}")
        raw_power = target_power.load_raw_components(
            power_path,
            start=args.start,
            end=args.end,
            columns=[timestamp_field, *components],
        )
        print(f"  raw power rows loaded: {len(raw_power):,}")

        frame, component_frame, target_audit = (
            target_power.build_target_power_frame_with_components(raw_power, config)
        )
        del raw_power
        print(
            f"Target frame built: {len(frame):,} rows on a complete "
            f"{config['frequency']} grid ({frame.index[0].isoformat()} -> "
            f"{frame.index[-1].isoformat()})"
        )

        print(f"Reading weather covariates from {weather_path}")
        raw_weather = target_power.load_raw_components(
            weather_path,
            # The feature builder excludes the 1970 sentinel and any row outside the
            # target grid itself, so the sentinel audit stays visible in the report.
            start=args.start,
            end=args.end,
            columns=weather_columns,
        )
        print(f"  raw weather rows loaded: {len(raw_weather):,}")
        feature_frame, feature_audit = target_power.build_target_feature_frame(
            raw_weather, frame.index, config
        )
        del raw_weather
        print(f"Target feature frame built: {len(feature_frame):,} long-format rows")

        sources = _source_metadata(
            config,
            {
                "power": int(target_audit["raw"]["rows_loaded"]),
                "weather": int(feature_audit["rows_loaded"]),
            },
        )

        metadata = target_power.build_target_metadata(
            frame, target_audit, config, source_metadata=sources
        )
        bounded_run = args.start is not None or args.end is not None
        metadata["run_mode"] = "bounded_cli" if bounded_run else "canonical_full_archive"
        metadata["data_selection"]["cli_start_override"] = args.start
        metadata["data_selection"]["cli_end_override"] = args.end
        quality = {
            "stage": "stage1_target_power_quality",
            "generated_at_utc": utc_timestamp(),
            "dataset_id": config["dataset_id"],
            "site_id": config["site_id"],
            "domain_id": config["domain_id"],
            "label_type": metadata["label_type"],
            "timezone_verified": False,
            "run_mode": metadata["run_mode"],
            "cli_overrides": {"start": args.start, "end": args.end},
            "target": target_audit,
            "features": feature_audit,
            "first_31_day_window": _first_31_day_summary(frame, config),
            "notes": [
                "This quality report describes real NLR ESIF raw data; nothing is synthetic.",
                "Missing intervals are kept on the grid with power_kw=NaN and power_valid=false.",
                "No label or feature value is imputed, back-filled or otherwise fabricated.",
            ],
        }
        written = target_power.write_target_artifacts(
            frame,
            component_frame,
            feature_frame,
            metadata,
            quality,
            config,
            out_dir=output_dir,
        )

        first31 = quality["first_31_day_window"]
        print("\nStage 1 target data quality summary")
        print(f"  dataset_id:      {config['dataset_id']}  (label_type={metadata['label_type']})")
        print(f"  timezone:        {metadata['timezone']}  timezone_verified=false (provisional)")
        print(f"  anchor:          {metadata['anchor_timestamp']}")
        print(
            f"  grid:            {target_audit['grid']['slots']:,} slots "
            f"({target_audit['grid']['start']} -> {target_audit['grid']['end']})"
        )
        print(
            f"  valid labels:    {target_audit['validity']['valid_slots']:,} "
            f"({100.0 * target_audit['validity']['valid_fraction']:.2f}%), "
            f"invalid={target_audit['validity']['invalid_slots']:,}, "
            f"missing intervals={target_audit['grid']['missing_interval_count']:,}"
        )
        print(
            f"  native sampling: expected={target_audit['expected_native_samples_per_interval']}"
            f"/interval, mean={target_audit['native_sampling']['mean_samples_per_interval']:.2f}, "
            f"median spacing={target_audit['native_sampling']['spacing_seconds_median']} s"
        )
        print(
            f"  first 31 trace days: {first31['valid_slots']}/{first31['expected_slots']} valid "
            f"(mean={first31['facility_power_kw_mean']:.2f} kW)"
        )
        print(
            f"  excluded pre-anchor raw rows: "
            f"{target_audit['raw']['rows_excluded_before_anchor']:,} "
            f"(all-components-NaN: "
            f"{target_audit['raw']['rows_excluded_before_anchor_with_all_components_nan']:,})"
        )
        print("\nWritten artefacts")
        for role, path in written["paths"].items():
            print(f"  {role:18s} {path}")
        print(f"\nMetadata label honesty: {metadata['label']['honesty_note']}")
        print("Composition warning: components must NOT be used as future-known features.")
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
