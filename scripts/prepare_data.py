"""Prepare the synthetic dataset and write versioned processed parquet files.

Usage (from the project root, CPU only)::

    & .\\.venv\\Scripts\\python.exe -m scripts.prepare_data --config configs/datasets/synthetic.yaml

The command never needs pre-existing raw data: the data is generated in memory,
written to ``data/processed/`` under fresh versioned names, and each parquet file
is accompanied by a JSON metadata sidecar.  Nothing is written below
``data/raw/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/prepare_data.py` as well as -m
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import write_processed
from src.data.synthetic import (
    SYNTHETIC_DATASET_ID,
    SyntheticDataset,
    build_synthetic_dataset_from_config,
)
from src.utils import load_config

DEFAULT_CONFIG = "configs/datasets/synthetic.yaml"
DEFAULT_OUT_DIR = "data/processed"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the deterministic synthetic facility dataset (supervised target plus two "
            "heterogeneous unlabelled pretraining sources) and write versioned processed files "
            "with JSON metadata sidecars."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"dataset YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"processed output directory (default: {DEFAULT_OUT_DIR}); never data/raw/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the config seed",
    )
    parser.add_argument(
        "--no-covariates",
        action="store_true",
        help="build the target frame without covariates (F=0 pure-power input)",
    )
    return parser


def _summary_lines(payload: dict[str, Any], *, kind: str) -> list[str]:
    quality = payload["quality"]
    sampling = payload["sampling"]
    lines = [
        f"{kind}: {payload['dataset_id']}",
        f"  file:     {payload['file']}",
        f"  sidecar:  {payload['sidecar']}",
        f"  rows={quality['rows']} interval={sampling['frequency']} "
        f"start={quality['start']} end={quality['end']}",
        f"  timezone={payload['timezone']['declared']} verified={payload['timezone']['verified']} "
        f"synthetic={payload['synthetic']}",
        f"  quality: complete_trace_days={quality['complete_trace_days']} "
        f"missing_intervals={quality['missing_intervals']} "
        f"duplicate_timestamps={quality['duplicate_timestamps']} "
        f"target_missing={quality['target_nan_count']}",
        f"  sha256={payload['sha256']}",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        dataset_id = config.get("dataset_id", SYNTHETIC_DATASET_ID)
        if dataset_id != SYNTHETIC_DATASET_ID:
            raise ValueError(
                f"config dataset_id={dataset_id!r} is not buildable by the synthetic generator "
                f"({SYNTHETIC_DATASET_ID!r}); real sources need a registered adapter."
            )
        if args.seed is not None:
            config = {**config, "seed": int(args.seed)}
        if args.no_covariates:
            config = {**config, "covariates": False}
        dataset: SyntheticDataset = build_synthetic_dataset_from_config(config)

        extra: dict[str, Any] = {
            "generator": "scripts.prepare_data",
            "config_path": str(args.config),
            "config": {key: value for key, value in config.items()},
            "unlabeled_source_ids": [source.dataset_id for source in dataset.unlabeled_sources],
            "covariates": bool(dataset.spec.features),
            "seed": int(config.get("seed", 7)),
        }
        target_payload = write_processed(
            dataset.frame,
            dataset.spec,
            out_dir=args.out,
            tag="synthetic",
            source_files=None,
            extra=extra,
        )
        print("\n".join(_summary_lines(target_payload, kind="Supervised target written")))
        print(
            "  covariates: "
            + (", ".join(dataset.spec.feature_names) if dataset.spec.feature_names else "(none, F=0)")
        )

        for source in dataset.unlabeled_sources:
            payload = write_processed(
                source.frame,
                source.spec,
                out_dir=args.out,
                tag="synthetic_unlabeled",
                source_files=None,
                extra={
                    **extra,
                    "generator": "scripts.prepare_data",
                    "unlabeled_source_id": source.dataset_id,
                    "feature_names": list(source.feature_names),
                    "F": len(source.feature_names),
                    "contains_power_column": False,
                },
            )
            print("\n".join(_summary_lines(payload, kind="Unlabelled source written")))
            print(f"  features (F={len(source.feature_names)}): {', '.join(source.feature_names)}")
            print("  contains_power_column=False")
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
