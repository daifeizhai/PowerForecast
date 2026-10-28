"""AcmeTrace admission audit CLI: inventory, power audit, hierarchy, 15-minute adaptation.

Usage (from the project root, CPU only)::

    & .\\.venv\\Scripts\\python.exe -m scripts.audit_acmetrace_power

    # bounded smoke run (a few chunks per file, useful while developing)
    & .\\.venv\\Scripts\\python.exe -m scripts.audit_acmetrace_power --max-chunks 2

    # skip the streamed SHA256 of the ~80 GB local tree
    & .\\.venv\\Scripts\\python.exe -m scripts.audit_acmetrace_power --skip-hash

Reads every local file under ``E:/Data/AcmeTrace`` (streamed, never loaded whole) and
writes:

* ``data/processed/acmetrace_inventory.json`` -- local inventory/integrity vs the HF tree;
* ``data/processed/acmetrace_power_audit.json`` -- GPU/server/cluster power audit;
* ``data/processed/acmetrace_power_metadata.json`` -- units/grid/hierarchy metadata;
* ``data/processed/acmetrace_power_15min.parquet`` -- native -> 15-minute interval means
  with explicit ``valid``/``coverage_ratio``/``n_samples`` (``cluster_it`` for Seren,
  ``cluster_gpu_sum`` for Kalos, never facility power).

Raw data is read-only; the command refuses any destination below ``data/raw/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import acmetrace_power as acme
from src.data.loaders import assert_not_raw_path
from src.utils import ensure_dir, resolve_path, utc_timestamp, write_json

DEFAULT_CONFIG = acme.CONFIG_PATH


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AcmeTrace admission audit: streamed inventory + GPU/IPMI power audit + "
            "hierarchy analysis + native->15-minute adaptation (no facility-power claim)."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"config YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument(
        "--raw-root",
        default=None,
        help=(
            "override the raw AcmeTrace root (default: config raw_root, i.e. E:/Data/AcmeTrace); "
            "the override is applied via ACMETRACE_RAW_ROOT before the config is loaded"
        ),
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="override the processed output directory (must not be under data/raw/)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="override the analysis output directory (run summary JSON)",
    )
    parser.add_argument("--skip-hash", action="store_true", help="skip the streamed SHA256 of local files")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="bounded smoke run: process at most N CSV chunks per file (never for reported results)",
    )
    parser.add_argument(
        "--reuse-inventory",
        action="store_true",
        help="reuse an existing data/processed/acmetrace_inventory.json (raw files unchanged)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="run the audit but do not write artefacts (smoke checks only)",
    )
    return parser


def _resolve_output_config(config: dict, args: argparse.Namespace) -> dict:
    outputs = dict(config["outputs"])
    if args.processed_dir:
        processed = assert_not_raw_path(args.processed_dir)
        outputs["processed_dir"] = str(processed)
        outputs["inventory_json"] = str(processed / acme.INVENTORY_FILENAME)
        outputs["power_audit_json"] = str(processed / acme.POWER_AUDIT_FILENAME)
        outputs["metadata_json"] = str(processed / acme.METADATA_FILENAME)
        outputs["power_15min_parquet"] = str(processed / acme.POWER_15MIN_FILENAME)
    if args.results_dir:
        outputs["results_dir"] = str(assert_not_raw_path(args.results_dir))
    for key in ("inventory_json", "power_audit_json", "metadata_json", "power_15min_parquet"):
        assert_not_raw_path(outputs[key])
    config["outputs"] = outputs
    return config


def _print_summary(result: Mapping[str, Any]) -> None:
    inventory = result["inventory"]
    summary = inventory["summary"]
    audit = result["power_audit"]
    table = result["table_stats"]
    print("AcmeTrace admission audit complete (streamed; raw tree read-only)")
    print(
        f"  inventory: local_files={summary['local_files']} "
        f"dataset_csv={summary['dataset_csv_files']} bytes={summary['local_bytes']:,} "
        f"lines_streamed={summary['local_lines_streamed']:,}"
    )
    print(
        f"  HF tree: files={summary['tree_files']} present={summary['files_present_locally']} "
        f"missing={summary['files_missing_locally']} "
        f"sha256_mismatches={len(summary['sha256_mismatches'])}"
    )
    for cluster_key, payload in audit["clusters"].items():
        print(
            f"  {cluster_key} GPU power: devices={payload['device_count']} nodes={payload['node_count']} "
            f"native_rows={payload['native_rows']:,} interval={payload['native_interval_mode_seconds']}s "
            f"valid_cells={payload['valid_series_cells']:,}/{payload['device_series_cells']:,} "
            f"zero_cells={payload['zero_series_cells']:,}"
        )
    nodes = audit["node_sets"]
    print(
        f"  IPMI nodes: total={nodes['ipmi_nodes']} seren_inventory={nodes['seren_node_inventory_nodes']} "
        f"intersection={nodes['ipmi_intersect_seren_inventory']} ipmi_only={nodes['ipmi_only_nodes']} "
        f"gpu_nodes_without_ipmi={len(nodes['seren_gpu_nodes_without_ipmi'])}"
    )
    for file_key, payload in audit["ipmi"]["files"].items():
        print(
            f"  IPMI {file_key}: rows={payload['native_rows']:,} nodes={payload['nodes']} "
            f"interval={payload['native_interval_mode_seconds']}s"
        )
    for key, stats in audit["aggregation_15min"]["level_summary"].items():
        if "rows_total" in stats:
            print(
                f"  15min {key}: rows={stats['rows_total']:,} valid={stats['rows_valid']:,} "
                f"device_valid_fraction={stats.get('device_valid_fraction'):.4f}"
            )
        else:
            print(
                f"  15min {key}: slots={stats['slots_total']:,} valid={stats['slots_valid']:,} "
                f"invalid={stats['slots_invalid']:,} coverage_mean={stats['coverage_mean']:.4f}"
            )
    if table:
        print(f"  parquet: {table['path']} rows={table['rows']:,} bytes={table['bytes']:,}")
    print("  important rows for V3.2 windows:")
    print("    Seren cluster IT : cluster_id='seren',  level='cluster_it',      device_id='seren_cluster_it'")
    print("    Kalos GPU sum    : cluster_id='kalos',  level='cluster_gpu_sum', device_id='kalos_cluster_gpu_sum'")
    print("  readiness: ready_for_training=False; ready_for_15min_analysis=" +
          str(audit["readiness"]["ready_for_15min_analysis"]))
    for reason in audit["readiness"]["reasons"]:
        print(f"    - {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.raw_root:
        os.environ["ACMETRACE_RAW_ROOT"] = str(Path(args.raw_root).resolve())
    try:
        config = _resolve_output_config(acme.load_acmetrace_config(args.config), args)
        inventory_payload = None
        if args.reuse_inventory:
            inventory_file = resolve_path(str(config["outputs"]["inventory_json"]))
            if inventory_file.is_file():
                inventory_payload = json.loads(inventory_file.read_text(encoding="utf-8"))
        result = acme.run_pipeline(
            config,
            hash_contents=not args.skip_hash,
            max_chunks_per_file=args.max_chunks,
            inventory_payload=inventory_payload,
            write=not args.no_write,
        )
        results_dir = resolve_path(str(config["outputs"]["results_dir"]))
        if not args.no_write:
            ensure_dir(results_dir)
            write_json(
                results_dir / "acmetrace_power_run_summary.json",
                {
                    "generated_at_utc": utc_timestamp(),
                    "dataset_id": "acmetrace",
                    "inventory_summary": result["inventory"]["summary"],
                    "node_sets": result["node_sets"],
                    "level_summary": result["power_audit"]["aggregation_15min"]["level_summary"],
                    "table_stats": result["table_stats"],
                    "readiness": result["power_audit"]["readiness"],
                },
            )
        _print_summary(result)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
