"""AcmeTrace runtime-metric / pairing / job-trace admission audit CLI.

Usage (from the project root, CPU only)::

    & .\\.venv\\Scripts\\python.exe -m scripts.audit_acmetrace_ops

    # bounded debug run (2 chunks per file; NOT a full audit)
    & .\\.venv\\Scripts\\python.exe -m scripts.audit_acmetrace_ops ^
        --clusters kalos --max-chunks 2

The CLI streams every ``E:/Data/AcmeTrace`` utilization/IPMI CSV in bounded chunks
(REAL data, read-only), maps GPU columns to ``(node IP, gpu index)``, verifies native
sampling, audits per-device missingness/ranges/XID, pairs power with operational metrics at
GPU and node level where it is defensible, audits the job traces (pre-origin vs post-hoc
fields) and writes:

* ``data/processed/acmetrace_ops_audit.json``
* ``data/processed/acmetrace_pairing_audit.json``
* ``data/processed/acmetrace_job_audit.json``
* ``results/analysis/acmetrace/*.csv`` summary tables

No file below ``data/raw/`` is ever written and no continuous series is fabricated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/audit_acmetrace_ops.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import acmetrace_ops as ops


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the REAL AcmeTrace runtime metrics, power<->ops pairing and job traces "
            "(streamed; no raw writes; no fabricated series)."
        )
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=list(ops.CLUSTERS),
        choices=list(ops.CLUSTERS),
        help="clusters to audit (default: both)",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help=(
            "Raw AcmeTrace root (default: $ACMETRACE_RAW_ROOT or E:/Data/AcmeTrace). "
            "Overrides the module-level raw/util/ipmi/job paths for this run."
        ),
    )
    parser.add_argument("--out", default="data/processed", help="artifact JSON directory")
    parser.add_argument(
        "--analysis-out",
        default="results/analysis/acmetrace",
        help="summary CSV directory",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help="optional override of the streaming chunk size (default: per-file-type)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="DEBUG ONLY: stop each file after N chunks (bounded/degraded audit)",
    )
    parser.add_argument("--skip-pairing", action="store_true", help="skip pairing matrices")
    parser.add_argument("--skip-ipmi", action="store_true", help="skip Seren IPMI power files")
    parser.add_argument("--skip-jobs", action="store_true", help="skip job-trace audit")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "optional upper-case utilization-metric whitelist (e.g. NODE_CPU_UTILIZATION). "
            "Default: audit every metric. A whitelist run is explicitly PARTIAL and the payload "
            "records which metrics were read."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.raw_root:
            root = str(Path(args.raw_root).resolve())
            ops.RAW_ROOT = root
            ops.UTIL_ROOT = f"{root}/data/utilization"
            ops.IPMI_ROOT = f"{ops.UTIL_ROOT}/ipmi"
            ops.JOB_ROOT = f"{root}/data/job_trace"
        if args.chunk_rows is not None and args.chunk_rows < 1:
            raise ValueError("--chunk-rows must be >= 1")
        if args.max_chunks is not None and args.max_chunks < 1:
            raise ValueError("--max-chunks must be >= 1")

        def progress(message: str) -> None:
            print(f"  {message}", flush=True)

        print(
            "Auditing AcmeTrace runtime metrics (REAL data, streamed chunks; "
            "no raw writes)",
            flush=True,
        )
        ops_audit, pairing_audit, job_audit = ops.run_audit(
            clusters=tuple(args.clusters),
            chunk_rows=args.chunk_rows,
            include_ipmi=not args.skip_ipmi,
            include_jobs=not args.skip_jobs,
            skip_pairing=args.skip_pairing,
            max_chunks=args.max_chunks,
            metrics=args.metrics,
            progress=progress,
        )
        if args.max_chunks is not None:
            note = {
                "bounded_pilot": True,
                "max_chunks_per_file": int(args.max_chunks),
                "coverage": (
                    "PARTIAL/PILOT COVERAGE: each file was read for only the first "
                    f"{args.max_chunks} streaming chunks; row counts, spans, missingness, ranges, "
                    "pairing spans and job linkage are therefore bounded pilot figures, not the "
                    "full-file audit. Do not cite as full coverage."
                ),
            }
            for payload in ops_audit["clusters"].values():
                for record in payload["files"]:
                    record["bounded_debug"] = True
            ops_audit["bounded_debug"] = dict(note)
            pairing_audit["bounded_debug"] = dict(note)
            job_audit["bounded_debug"] = dict(note)
        paths = ops.write_audit_artifacts(
            ops_audit,
            pairing_audit,
            job_audit,
            out_dir=args.out,
            analysis_dir=args.analysis_out,
        )
        print("\nArtifacts written:", flush=True)
        for name, path in paths.items():
            print(f"  {name}: {path}", flush=True)

        print("\nCluster summaries:", flush=True)
        for cluster, payload in ops_audit["clusters"].items():
            summary = payload["summary"]
            print(
                f"  {cluster}: files={summary['files']} rows={summary['rows']} "
                f"placeholder={summary['placeholder_files']} "
                f"missing_overall={(summary['missing_fraction_overall'] or 0):.4f} "
                f"intervals={summary['native_interval_seconds_groups']} "
                f"xid_nonzero={summary['xid_nonzero_values']}",
                flush=True,
            )
            for pair in pairing_audit["clusters"].get(cluster, {}).get("gpu_pairs", []):
                print(
                    f"    gpu pair {pair['metric_a']}~{pair['metric_b']}: "
                    f"devices={pair['matched_devices']} slots={pair['matched_device_slots']} "
                    f"match_rate={pair['match_rate']} "
                    f"native_instants={(pair.get('native_window') or {}).get('matched_device_instants')}",
                    flush=True,
                )
            for pair in pairing_audit["clusters"].get(cluster, {}).get("node_pairs", []):
                print(
                    f"    node pair {pair['metric_a']}~{pair['metric_b']}: "
                    f"devices={pair['matched_devices']} slots={pair['matched_device_slots']} "
                    f"match_rate={pair['match_rate']}",
                    flush=True,
                )
        for cluster, payload in job_audit["clusters"].items():
            plausibility = payload["plausibility"]
            print(
                f"  jobs {cluster}: rows={payload['rows']} gpu_jobs={plausibility['gpu_jobs']} "
                f"bad_duration={plausibility['duration_negative_or_zero']} "
                f"post_hoc_fields={payload['post_hoc_fields']} "
                f"linkage_corr={(payload.get('cluster_linkage') or {}).get('pearson_active_gpu_vs_mean_util')}",
                flush=True,
            )
            impact = payload.get("timezone_impact", {})
            print(f"    timezone: {impact.get('impact_if_ignored')}", flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
