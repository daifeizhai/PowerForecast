"""Read-only full-file M100/UCC2023 power-data audit; no fitting or imputation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MD5 = {
    "M100/0.tar": "ae8a04b847562dfa807863100dc6777d",
    "M100/1.tar": "6313d691ce2e280bb363681983b2decf",
    "UCC2023/dataset1_combined.csv": "0b931f2b2cc4f236ef9b66e2716b69b2",
    "UCC2023/dataset2_combined.csv": "8660d51cfb62dfc15f65c79406cb715f",
}
SOURCES = {
    "m100_aggregated": "https://zenodo.org/records/7541722",
    "m100_ipmi": "https://gitlab.com/ecs-lab/exadata/-/blob/main/documentation/plugins/ipmi.md",
    "m100_processing": "https://gitlab.com/ecs-lab/exadata/-/tree/main/parquet_dataset/node_aggregated_data",
    "ucc2023": "https://zenodo.org/records/10069402",
    "ucc2023_paper": "https://hpc.ec.tuwien.ac.at/files/UCC_23_data_center_analysis.pdf",
}


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    assert (before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
    return dict(bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                md5=md5.hexdigest(), sha256=sha.hexdigest())


def number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def longest_run(times, step_ns):
    times = np.unique(np.asarray(times, dtype="int64"))
    if not len(times):
        return dict(points=0, days=0., full_24h_blocks=0, start=None, end=None)
    cuts = np.r_[0, np.flatnonzero(np.diff(times) != step_ns) + 1, len(times)]
    lengths = np.diff(cuts)
    best = int(lengths.argmax())
    n = int(lengths[best])
    return dict(points=n, days=n * step_ns / 86400e9,
                full_24h_blocks=int(n * step_ns // int(86400e9)),
                start=pd.Timestamp(times[cuts[best]]).isoformat(),
                end=pd.Timestamp(times[cuts[best + 1] - 1]).isoformat())


def audit_node(frame, dataset, node, time_col, power_col, minutes, core, member):
    ts = pd.to_datetime(frame[time_col], errors="coerce")
    t = pd.DatetimeIndex(ts).as_unit("ns").asi8
    step = int(minutes * 60e9)
    timestamp_ok = ~ts.isna().to_numpy()
    on_grid = timestamp_ok & (t % step == 0)
    duplicate_all = ts.duplicated(keep=False).to_numpy() & timestamp_ok
    p = pd.to_numeric(frame[power_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(p)
    valid = finite & (p >= 0) & on_grid & ~duplicate_all
    core_values = frame[core].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    joint = valid & np.isfinite(core_values).all(axis=1)
    unique_times = np.unique(t[on_grid])
    diffs = np.diff(np.unique(t[timestamp_ok]))
    expected = int((unique_times[-1] - unique_times[0]) // step + 1) if len(unique_times) else 0
    duplicate_keys = pd.Index(ts[duplicate_all]).unique()
    conflicts = 0
    if len(duplicate_keys):
        values = frame.loc[duplicate_all].drop(columns=[time_col])
        row_hash = pd.util.hash_pandas_object(values, index=False).to_numpy()
        dup = pd.DataFrame({"t": t[duplicate_all], "h": row_hash})
        conflicts = int((dup.groupby("t").h.nunique() > 1).sum())
    day_ns = int(86400e9)
    day_ids, day_counts = np.unique(t[valid] // day_ns, return_counts=True)
    complete_days = day_ids[day_counts == 1440 // minutes]
    complete = longest_run(complete_days * day_ns, day_ns)
    run = longest_run(t[valid], step)
    joint_run = longest_run(t[joint], step)
    good = p[finite]
    q = np.quantile(good, [.01, .25, .5, .75, .99]) if len(good) else np.full(5, np.nan)
    fence = q[3] + 3 * (q[3] - q[1])
    result = dict(dataset=dataset, node=str(node), member=member, rows=len(frame),
        timestamp_timezone=str(getattr(ts.dt, "tz", None)), timestamp_invalid=int((~timestamp_ok).sum()),
        first_timestamp=str(ts.min()), last_timestamp=str(ts.max()),
        span_days=float((t[timestamp_ok].max() - t[timestamp_ok].min() + step) / day_ns),
        expected_grid_points=expected, unique_grid_points=len(unique_times),
        missing_grid_points=expected-len(unique_times), off_grid_rows=int((timestamp_ok & ~on_grid).sum()),
        duplicate_rows_excess=int(ts[timestamp_ok].duplicated().sum()),
        duplicate_rows_involved=int(duplicate_all.sum()), duplicate_keys=len(duplicate_keys),
        conflicting_duplicate_keys=conflicts, nonincreasing_input_steps=int((np.diff(t[timestamp_ok]) <= 0).sum()),
        native_step_pairs=int((diffs == step).sum()), gap_pairs=int((diffs > step).sum()),
        short_or_off_step_pairs=int(((diffs != step) & (diffs < step)).sum()),
        max_gap_minutes=float(diffs.max() / 60e9) if len(diffs) else 0.,
        power_null_rows=int(np.isnan(p).sum()), power_infinite_rows=int(np.isinf(p).sum()),
        power_negative_rows=int((finite & (p < 0)).sum()), power_zero_rows=int((finite & (p == 0)).sum()),
        power_valid_unique_grid_points=int(valid.sum()), power_valid_coverage_pct=100*int(valid.sum())/expected if expected else 0.,
        power_min_w=number(np.min(good)) if len(good) else None,
        power_max_w=number(np.max(good)) if len(good) else None,
        power_mean_w=number(np.mean(good)) if len(good) else None,
        power_q01_w=number(q[0]), power_median_w=number(q[2]), power_q99_w=number(q[4]),
        power_above_q3_plus_3iqr=int((good > fence).sum()),
        longest_power_run_points=run["points"], longest_power_run_days=run["days"],
        longest_power_run_full_days=run["full_24h_blocks"], longest_power_run_start=run["start"], longest_power_run_end=run["end"],
        complete_calendar_days=len(complete_days), longest_complete_calendar_days=complete["points"],
        core_auxiliary_columns=",".join(core), joint_valid_points=int(joint.sum()),
        longest_joint_run_days=joint_run["days"], longest_joint_run_start=joint_run["start"], longest_joint_run_end=joint_run["end"],
        points_for_14d_history_plus_24h=int(15 * 1440 // minutes),
        points_for_14d_history_plus_3d=int(17 * 1440 // minutes),
        has_contiguous_14d_plus_24h=run["days"] >= 15, has_contiguous_14d_plus_3d=run["days"] >= 17)
    result["all_columns_nonmissing_rows"] = int(frame.notna().all(axis=1).sum())
    if "total_power_min" in frame:
        low = frame.total_power_min.to_numpy(dtype=float, na_value=np.nan)
        high = frame.total_power_max.to_numpy(dtype=float, na_value=np.nan)
        std = frame.total_power_std.to_numpy(dtype=float, na_value=np.nan)
        result["power_aggregate_order_violations"] = int(((low > p+1e-6) | (p > high+1e-6) | (std < 0)).sum())
    if "value" in frame:
        result["nagios_missing_rows"] = int(frame.value.isna().sum())
        result["nagios_nonzero_rows"] = int((frame.value.fillna(0) != 0).sum())
    if "CPU_Load" in frame:
        result["cpu_outside_0_100_rows"] = int(((frame.CPU_Load < 0) | (frame.CPU_Load > 100)).sum())
        result["ram_used_exceeds_ram_rows"] = int((frame.Ram_Used > frame.Ram + 1e-6).sum())
        result["used_cores_exceed_cores_rows"] = int((frame.CPU_cores_used > frame.CPU_cores + 1e-6).sum())
    column_rows = []
    for col in frame.columns:
        if col == time_col or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        a = frame[col].to_numpy(dtype=float, na_value=np.nan)
        ok = np.isfinite(a)
        column_rows.append(dict(dataset=dataset, node=str(node), column=col, rows=len(a),
            null_rows=int(np.isnan(a).sum()), infinite_rows=int(np.isinf(a).sum()),
            negative_rows=int((ok & (a < 0)).sum()), zero_rows=int((ok & (a == 0)).sum()),
            finite_rows=int(ok.sum()), min=number(a[ok].min()) if ok.any() else None,
            max=number(a[ok].max()) if ok.any() else None,
            mean=number(a[ok].mean()) if ok.any() else None,
            all_missing=bool((~ok).all()), constant_finite=bool(ok.any() and a[ok].min() == a[ok].max())))
    return result, column_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT/"data/raw")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrite a prior audit.")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; use a new directory.")
    args.output.mkdir(parents=True)
    files, nodes, columns, schemas, extracted, duplicates, populations = [], [], [], {}, [], [], []
    for relative, expected in OFFICIAL_MD5.items():
        path = args.raw_root/relative
        h = hashes(path)
        files.append(dict(path=relative, **h, official_md5=expected, official_md5_matches=h["md5"] == expected))
        print(f"Identity {relative}: official MD5 match={h['md5'] == expected}", flush=True)
    for archive_name in ["0.tar", "1.tar"]:
        with tarfile.open(args.raw_root/"M100"/archive_name, "r:") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith(".parquet"):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                payload = archive.extractfile(member).read()
                node = Path(member.name).stem
                table = pq.read_table(BytesIO(payload))
                frame = table.to_pandas()
                schemas[f"M100/{member.name}"] = {f.name: str(f.type) for f in table.schema}
                local = args.raw_root/"M100/extracted"/member.name
                if local.is_file():
                    digest = hashlib.sha256(payload).hexdigest()
                    local_hash = hashes(local)
                    extracted.append(dict(path=str(local.relative_to(args.raw_root)), sha256=local_hash["sha256"],
                                          matches_archive_member=local_hash["sha256"] == digest))
                core = ["ambient_avg", "p0_power_avg", "p1_power_avg", "gpu0_core_temp_avg",
                        "gpu1_core_temp_avg", "gpu3_core_temp_avg", "gpu4_core_temp_avg"]
                result, feature = audit_node(frame, "M100", node, "timestamp", "total_power_avg", 15, core,
                                             f"{archive_name}:{member.name}")
                nodes.append(result)
                columns.extend(feature)
                print(f"M100 node {node}: {len(frame)} rows, longest valid power={result['longest_power_run_days']:.3f} days", flush=True)
                del frame, table, payload
    for filename, dataset in [("dataset1_combined.csv", "UCC_D1"), ("dataset2_combined.csv", "UCC_D2")]:
        frame = pd.read_csv(args.raw_root/"UCC2023"/filename)
        schemas[dataset] = {col: str(dtype) for col, dtype in frame.dtypes.items()}
        # Whitespace belongs to the CSV representation, not to a distinct host identity.
        original = frame.Series.astype(str)
        stripped = original.str.strip()
        whitespace_rows = int((original != stripped).sum())
        frame["Series"] = stripped
        complete = frame.notna().all(axis=1)
        published_rows = 2653869 if dataset == "UCC_D1" else 961158
        populations.append(dict(dataset=dataset, published_cleaned_rows=published_rows,
            actual_file_rows=len(frame), actual_unique_host_ids=int(frame.Series.nunique()),
            all_columns_nonmissing_rows=int(complete.sum()),
            hosts_with_all_columns_nonmissing_rows=int(frame.loc[complete,"Series"].nunique()),
            complete_rows_match_paper=int(complete.sum()) == published_rows))
        repeated = frame.duplicated(["Series", "Time"], keep=False)
        for stamp, group in frame.loc[repeated].groupby("Time", sort=True):
            duplicates.append(dict(dataset=dataset, clock_time=stamp,
                                   rows=len(group), hosts=int(group.Series.nunique())))
        for node, group in frame.groupby("Series", sort=True):
            result, feature = audit_node(group.drop(columns="Series"), dataset, node, "Time", "Power", 10,
                ["CPU_Load", "Ram_Used", "CPU1_Temperature", "CPU2_Temperature"], filename)
            nodes.append(result)
            columns.extend(feature)
        print(f"{dataset}: {len(frame)} rows, {frame.Series.nunique()} hosts, trimmed identity rows={whitespace_rows}", flush=True)
        schemas[dataset+"_identity_whitespace_rows"] = whitespace_rows
        del frame
    node_frame = pd.DataFrame(nodes).sort_values(["dataset", "node"])
    feature_frame = pd.DataFrame(columns).sort_values(["dataset", "node", "column"])
    node_frame.to_csv(args.output/"nodes.csv", index=False)
    feature_frame.to_csv(args.output/"features.csv", index=False)
    pd.DataFrame(duplicates).to_csv(args.output/"duplicate-clock-times.csv", index=False)
    summaries = []
    for name, group in node_frame.groupby("dataset", sort=True):
        counters = ["rows", "expected_grid_points", "missing_grid_points", "power_null_rows", "power_infinite_rows",
                    "power_negative_rows", "power_zero_rows", "duplicate_rows_excess", "duplicate_keys",
                    "conflicting_duplicate_keys", "off_grid_rows", "timestamp_invalid", "power_valid_unique_grid_points",
                    "has_contiguous_14d_plus_24h", "has_contiguous_14d_plus_3d"]
        record = dict(dataset=name, nodes=len(group), **{k: int(group[k].sum()) for k in counters})
        for col in ["span_days", "longest_power_run_days", "longest_joint_run_days", "complete_calendar_days", "longest_complete_calendar_days"]:
            record[col] = dict(min=float(group[col].min()), median=float(group[col].median()), max=float(group[col].max()))
        record["first_timestamp"] = group.first_timestamp.min()
        record["last_timestamp"] = group.last_timestamp.max()
        record["power_min_w"] = float(group.power_min_w.min())
        record["power_max_w"] = float(group.power_max_w.max())
        record["power_null_pct"] = 100 * record["power_null_rows"] / record["rows"]
        record["power_valid_grid_pct"] = 100 * record["power_valid_unique_grid_points"] / record["expected_grid_points"]
        summaries.append(record)
    for row in files:
        after = (args.raw_root/row["path"]).stat()
        row["unchanged_size_mtime_after_audit"] = (after.st_size, after.st_mtime_ns) == (row["bytes"], row["mtime_ns"])
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version, pandas=pd.__version__, numpy=np.__version__, sources=SOURCES,
        input_files=files, existing_extracted_files=extracted, summaries=summaries, schemas=schemas,
        ucc_population_comparison=populations,
        rules=dict(read_only=True, no_interpolation=True, no_training=True, eagle_not_opened=True,
            power_valid="Finite nonnegative power, valid native-grid timestamp, no duplicate key; zero power counted separately.",
            duplicate_policy="All repeated node/timestamp keys excluded from strict continuity, including identical repeats.",
            duration="N consecutive samples * native interval; last timestamp is inclusive. A day is 24 hours.",
            complete_calendar_day="96 valid unique points (M100) or 144 (UCC) in a date; UTC for M100, unresolved local clock for UCC.",
            joint_valid="Power-valid plus finite declared core auxiliary fields; no imputation or extra physical filters.",
            high_power_flag="Above per-node Q3 + 3*IQR; descriptive diagnostic only, never auto-deleted.",
            time_semantics="M100 timestamps UTC; aggregation boundary/sample counts not established by this audit. UCC CSV timestamps are timezone-naive.",
            scope="Both local M100 tarballs only, not all racks; both full UCC CSVs; distinct datasets/hosts are never summed as facility power."))
    (args.output/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    success = all(r["official_md5_matches"] and r["unchanged_size_mtime_after_audit"] for r in files)
    success &= all(r["matches_archive_member"] for r in extracted)
    print(json.dumps(dict(identities_ok=success, summaries=summaries), ensure_ascii=False), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
