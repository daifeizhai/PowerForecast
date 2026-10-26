"""Stage-1 source-domain CLI: build the M100 panel and write Stage-1 artefacts.

Usage (from the project root, CPU only)::

    & .\\.venv\\Scripts\\python.exe -m scripts.prepare_source_features ^
        --config configs/datasets/sources.yaml

    # bounded run for a quick check
    & .\\.venv\\Scripts\\python.exe -m scripts.prepare_source_features ^
        --nodes 10 11 --max-rows-per-node 2000

Writes, by default:

* ``data/processed/processed_source_features.parquet`` -- long-format REAL M100 source
  panel with exactly the task-book logical fields (no label/target column);
* ``data/processed/source_metadata.json`` -- field semantics, native-frequency audit,
  missing semantics, file SHA256, UCC/Alibaba audit status, training-approval status;
* ``data/processed/source_quality.json`` -- coverage/missingness/range-violation audit;
* ``configs/datasets/feature_registry.yaml`` -- the tracked field registry (auto-generated
  from ``sources.yaml`` + ``src/data/registry.py`` alignment).

The command refuses to write anywhere below ``data/raw/``.  It never resamples UCC to a
fake 15-minute series, never fabricates data, and never approves a source for training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/prepare_source_features.py` too
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import source_features as sf
from src.data.loaders import assert_not_raw_path

DEFAULT_CONFIG = sf.SOURCES_CONFIG
DEFAULT_OUT_DIR = sf.DEFAULT_PROCESSED_DIR


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Stage-1 REAL M100 source-feature panel (long format), audit the native "
            "15-minute sampling and UCC's native 10-minute sampling, and write the source "
            "metadata/quality artefacts plus the tracked feature-registry YAML."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"source config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"processed output directory (default: {DEFAULT_OUT_DIR}); never data/raw/",
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        default=None,
        metavar="STEM",
        help="optional M100 file stems to include (e.g. --nodes 10 11); default = all files",
    )
    parser.add_argument(
        "--max-rows-per-node",
        type=int,
        default=None,
        help="optional bound on native rows per node (bounded runs/tests only)",
    )
    parser.add_argument(
        "--registry",
        default=sf.FEATURE_REGISTRY_PATH,
        help=f"tracked feature-registry YAML destination (default: {sf.FEATURE_REGISTRY_PATH})",
    )
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="do not rewrite the tracked feature-registry YAML",
    )
    parser.add_argument(
        "--skip-ucc-audit",
        action="store_true",
        help="skip the bounded UCC audit (quick M100-only smoke checks)",
    )
    return parser


def _print_summary(
    panel_rows: int,
    summary: dict[str, Any],
    metadata: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    registry_path: Path | None,
    registry: dict[str, Any] | None,
) -> None:
    native = metadata["native_frequency"]["m100_exadata"]
    ucc = metadata["native_frequency"]["ucc2023"]
    print("Stage-1 source panel written (REAL M100 node telemetry, not facility power)")
    print(f"  panel: {panel_rows} rows x {len(summary['columns'])} logical columns")
    print(
        f"  domains={summary['domain_count']} features={summary['feature_count']} "
        f"frequency={summary['frequency']} points_per_day={summary['points_per_day']}"
    )
    print(f"  memory during build: {summary['memory_bytes'] / 1e6:.1f} MB")
    print(
        f"  domain ids: {', '.join(summary['domains'][:6])}"
        + (" ..." if len(summary["domains"]) > 6 else "")
    )
    print(f"  features: {', '.join(summary['features'])}")
    print(
        f"  M100 native-frequency verdict: {native['verdict']} "
        f"(declared={native['declared_frequency']}, target={native['target_frequency']})"
    )
    if ucc.get("aggregation"):
        aggregation = ucc["aggregation"]
        print(
            f"  UCC aggregation status: {aggregation['status']} "
            f"(native={aggregation['native_frequency']}, target={aggregation['target_frequency']})"
        )
        print(f"  UCC verdict: {aggregation['verdict']}")
    if artifacts:
        print(f"  parquet:  {artifacts['panel_parquet']} ({artifacts['panel_bytes'] / 1e6:.1f} MB, "
              f"{artifacts['panel_rows']} rows)")
        print(f"  metadata: {artifacts['metadata_json']} ({artifacts['metadata_bytes'] / 1e3:.1f} kB)")
        print(f"  quality:  {artifacts['quality_json']} ({artifacts['quality_bytes'] / 1e3:.1f} kB)")
    observed = [
        item["observed_fraction_overall"] for item in metadata["features"].values()
    ]
    if observed:
        print(
            f"  observed fraction per feature: min={min(observed):.3f} max={max(observed):.3f} "
            f"mean={sum(observed) / len(observed):.3f}"
        )
    if registry is not None and registry_path is not None:
        counts = registry.get("alignment_status_counts", {})
        print(f"  field registry: {registry_path}")
        for key, item in counts.items():
            print(
                f"    {key}: status={item['status']} shared={item['shared']} "
                f"source_only={item['source_only']} target_only={item['target_only']} "
                f"incomparable={item['incomparable']}"
            )
    approval = metadata["training_approval"]
    print(
        "  training approval: "
        f"approved_for_pretraining={approval['approved']} "
        f"approved_for_facility_kw_training={approval['approved_for_facility_kw_training']}"
    )
    print(f"  approval reason: {approval['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        # Refuse a raw destination before doing any (potentially expensive) work.
        assert_not_raw_path(args.out)
        if not args.skip_registry:
            assert_not_raw_path(args.registry)
        if args.max_rows_per_node is not None and args.max_rows_per_node < 1:
            raise ValueError("--max-rows-per-node must be >= 1.")
        config = sf.load_source_config(args.config)

        panel, summary = sf.build_source_panel(
            config,
            nodes=args.nodes,
            max_rows_per_node=args.max_rows_per_node,
        )
        ucc_audit = None if args.skip_ucc_audit else sf.build_ucc_audit(config)
        metadata = sf.build_source_metadata(
            panel,
            config,
            panel_summary=summary,
            ucc_audit=ucc_audit,
        )
        quality = sf.build_source_quality(panel, config, panel_summary=summary)
        artifacts = sf.write_source_artifacts(panel, metadata, quality, out_dir=args.out)

        registry: dict[str, Any] | None = None
        registry_path: Path | None = None
        if not args.skip_registry:
            registry = sf.build_feature_registry_payload(config)
            registry_path = sf.write_feature_registry(registry, path=args.registry)

        _print_summary(
            len(panel),
            summary,
            metadata,
            artifacts,
            registry_path=registry_path,
            registry=registry,
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
