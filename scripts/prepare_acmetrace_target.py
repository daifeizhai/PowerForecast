"""Assemble the AcmeTrace Seren target-domain dataset (power-transfer-v2).

Usage (from the project root, CPU only)::

    & .\\.venv\\Scripts\\python.exe -m scripts.prepare_acmetrace_target

    # bounded smoke run (1 CSV chunk per file; DEBUG ONLY, never cited as full coverage)
    & .\\.venv\\Scripts\\python.exe -m scripts.prepare_acmetrace_target --max-chunks 1 --no-write

The command reads the READ-ONLY complete AcmeTrace snapshot at ``E:/Data/AcmeTrace`` (or
``$ACMETRACE_RAW_ROOT``) and writes the 15-minute target dataset:

* ``data/processed/acmetrace_target_power.parquet`` -- Seren ``cluster_it`` power label,
  mask-first, never facility kW;
* ``data/processed/acmetrace_target_features.parquet`` -- runtime / job-trace features on
  the same 15-minute grid, mask-first;
* ``data/processed/acmetrace_target_metadata.json`` -- label semantics, feature registry,
  provenance, readiness;
* ``data/processed/acmetrace_target_quality.json`` -- coverage, missingness and descriptive
  power<->feature overlap statistics;
* ``results/analysis/acmetrace/acmetrace_target_run_summary.json`` -- run summary.

The power label requires ``data/processed/acmetrace_power_15min.parquet`` produced by
``python -m scripts.audit_acmetrace_power``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/prepare_acmetrace_target.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import acmetrace_target as target
from src.data.loaders import assert_not_raw_path
from src.utils import resolve_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the AcmeTrace Seren 15-minute target-domain dataset "
            "(cluster-IT power label + runtime / job-trace features; no facility-kW claim)."
        )
    )
    parser.add_argument(
        "--config",
        default=target.CONFIG_PATH,
        help=f"target config YAML (default: {target.CONFIG_PATH})",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help=(
            "override the AcmeTrace raw root (default: config source.raw_root, i.e. "
            "E:/Data/AcmeTrace); applied through ACMETRACE_RAW_ROOT"
        ),
    )
    parser.add_argument(
        "--power-parquet",
        default=None,
        help="override the 15-minute power parquet (default: config source.power_parquet)",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="override the processed output directory (must not be under data/raw/)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="override the analysis output directory",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="DEBUG ONLY: read at most N CSV chunks per source file (bounded pilot run)",
    )
    parser.add_argument("--skip-gpu", action="store_true", help="skip the GPU_UTIL feature")
    parser.add_argument("--skip-jobs", action="store_true", help="skip job-trace features")
    parser.add_argument("--no-write", action="store_true", help="compute but write no artefacts")
    parser.add_argument("--quiet", action="store_true", help="suppress per-chunk progress lines")
    return parser


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source = dict(config["source"])
    outputs = dict(config["outputs"])
    if args.power_parquet:
        source["power_parquet"] = str(assert_not_raw_path(args.power_parquet))
    if args.processed_dir:
        processed = assert_not_raw_path(args.processed_dir)
        outputs["processed_dir"] = str(processed)
        for key in ("power_parquet", "features_parquet", "metadata_json", "quality_json"):
            outputs[key] = str(processed / Path(str(outputs[key])).name)
    if args.results_dir:
        outputs["results_dir"] = str(assert_not_raw_path(args.results_dir))
    config["source"] = source
    config["outputs"] = outputs
    return target.validate_target_config(config)


def _print_summary(result: dict[str, Any]) -> None:
    metadata = result["metadata"]
    quality = result["quality"]
    label = metadata["label"]
    print("AcmeTrace Seren target dataset assembled (REAL data, read-only raw tree)")
    print(
        f"  label: {label['target_name']} ({label['target_scope']}, {label['unit']}) "
        f"slots={metadata['time']['slots']} valid={quality['power']['valid_slots']} "
        f"({quality['power']['valid_fraction']:.4f}) "
        f"mean={quality['power']['power_kw_mean']:.1f} kW"
        if quality["power"]["power_kw_mean"] is not None
        else f"  label: {label['target_name']} (no valid slots)"
    )
    print(
        f"  time: {metadata['time']['first_label_timestamp_utc']} -> "
        f"{metadata['time']['last_label_timestamp_utc']} (UTC-naive interval starts)"
    )
    print("  features (observed fraction on the label grid):")
    for entry in metadata["features"]["audit"]:
        print(
            f"    {entry['feature_name']}: {entry['observed_bins']}/{entry['label_bins']} "
            f"({entry['observed_fraction']:.3f}) "
            f"coverage_mean={entry['coverage_mean_observed']}"
        )
    job = metadata.get("job_trace")
    if job is not None:
        print(
            f"  job trace: rows={job['rows']} jobs_with_lifetime={job['audit']['jobs_with_lifetime']} "
            f"max_active_jobs={job['audit']['max_active_jobs']:.0f} "
            f"max_active_gpu={job['audit']['max_active_gpu']:.0f}"
        )
    print("  correlation (descriptive only):")
    for entry in quality["feature_power_correlation"]:
        print(
            f"    {entry['feature_name']}: bins={entry['overlap_valid_bins']} "
            f"pearson_r={entry['pearson_r']}"
        )
    print(f"  elapsed: {result['elapsed_seconds']:.1f} s")
    print("  artefacts:")
    for key, path in result["outputs"].items():
        print(f"    {key}: {path}")
    print(
        "  readiness: ready_for_15min_analysis=True ready_for_training=False "
        "(cluster-IT label, inferred W units, unverified site timezone; never facility kW)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error("--max-chunks must be >= 1")
    import os

    if args.raw_root:
        os.environ["ACMETRACE_RAW_ROOT"] = str(Path(args.raw_root).resolve())
    if args.results_dir:
        # resolve through the raw-path guard as well
        assert_not_raw_path(args.results_dir)
    try:
        config = target.load_target_config(args.config)
        config = _apply_overrides(config, args)
        progress = None if args.quiet else (lambda message: print(f"  {message}", flush=True))
        result = target.run_pipeline(
            config,
            max_chunks=args.max_chunks,
            include_gpu=not args.skip_gpu,
            include_jobs=not args.skip_jobs,
            write=not args.no_write,
            progress=progress,
        )
        _print_summary(result)
        if args.max_chunks is not None:
            print(
                "\n  [bounded-debug] only the first "
                f"{args.max_chunks} chunks per source file were read; these numbers are NOT "
                "full-coverage results."
            )
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
