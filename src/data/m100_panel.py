"""Published M100 15min wide tables: node total W -> kW, no re-aggregation."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.domains import DomainMetadata, validate_compatible
from src.data.google_panel import DomainPanel
from src.utils import resolve_path


def adapt_m100_aggregated(frame, *, node_id, rack_id, metadata):
    """The input is already aggregated. Missing/duplicate labels remain unavailable."""
    if not {"timestamp", "total_power_avg"} <= set(frame):
        raise ValueError("M100 wide tables require timestamp and measured total_power_avg (W).")
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, errors="raise")).as_unit("ns")
    if times.tz is None or str(times.tz) != "UTC" or times.hasnans:
        raise ValueError("M100 published timestamps must be explicitly UTC, without NaT.")
    if np.any(times.asi8 % int(pd.Timedelta(minutes=15).value)):
        raise ValueError("M100 published timestamps must align to the native 15min grid.")
    if not len(times):
        raise ValueError("M100 node table is empty.")
    values = pd.to_numeric(frame.total_power_avg, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values) & (values >= 0)
    duplicated = times.duplicated(keep=False)
    series = pd.Series(np.where(finite, values*.001, np.nan), index=times, name="node_power_kw")
    series = series.loc[~duplicated].sort_index()
    grid = pd.date_range(times.min(), times.max(), freq="15min", name="timestamp")
    adapted = series.reindex(grid).to_frame()
    domain = f"m100_rack{rack_id}_node{node_id}"
    contract = DomainMetadata(**{**metadata, "domain_id": domain, "node_id": str(node_id),
                                 "rack_id": str(rack_id), "target_field": "node_power_kw"})
    if (contract.target_kind, contract.target_scope, contract.target_unit, contract.raw_frequency,
            contract.label_origin, contract.sampling_semantics) != ("active_power", "node", "kW", "15min", "measured", "interval_mean"):
        raise ValueError("M100 wide-table physical contract must preserve measured 15min node power in kW.")
    audit = dict(raw_rows=len(frame), duplicate_rows_involved=int(duplicated.sum()),
        invalid_published_values=int((~finite).sum()), missing_grid_slots=len(grid)-times.nunique(),
        missing_target_intervals=int(adapted.node_power_kw.isna().sum()),
        target_conversion="total_power_avg * 0.001 W->kW; no sum, interpolation or repeated aggregation",
        raw_samples_per_interval=None, aggregation_boundary_verified=False,
        history_availability_upper_bound_minutes=15, extra_reporting_delay_assumed_minutes=0,
        feature_policy="Target history and known relative calendar only; no contemporaneous power min/max/std or status as future inputs.")
    return adapted, contract, audit


def load_m100_domains(config):
    root = resolve_path(config["raw_data"])
    if (config.get("published_target"), config.get("published_unit"), config.get("target"),
            config.get("target_unit")) != ("total_power_avg", "W", "node_power_kw", "kW"):
        raise ValueError("Use the audited M100 measured-node target and explicit W->kW conversion.")
    anchor = pd.Timestamp(config["trace_start"])
    if anchor.tz is None:
        raise ValueError("M100 experiment start must include UTC.")
    anchor = anchor.tz_convert("UTC").as_unit("ns")
    grid = pd.date_range(anchor, periods=int(config["expected_days"])*96, freq="15min", name="timestamp")
    series, metadata, coverage, sources, audit = {}, {}, {}, {}, {}
    for spec in config["archives"]:
        path = root/spec["filename"]
        before = path.stat()
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b""):
                digest.update(block)
        if digest.hexdigest() != spec["sha256"]:
            raise ValueError(f"{path.name}: M100 archive SHA256 differs from the audited source.")
        sources[path.name] = dict(path=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
            sha256=digest.hexdigest(), source_documentation=config["source_documentation"], rack_id=spec["rack_id"])
        with tarfile.open(path, "r:") as archive:
            members = archive.getmembers()
            expected = {f"{node}.parquet" for node in spec["node_ids"]}
            if len(members) != len(expected) or {m.name for m in members} != expected or any(not m.isfile() for m in members):
                raise ValueError(f"{path.name}: unexpected or duplicate node members; never extract or infer identities.")
            for member in sorted(members, key=lambda m: int(Path(m.name).stem)):
                payload = archive.extractfile(member).read()
                frame = pq.read_table(BytesIO(payload), columns=["timestamp", "total_power_avg"]).to_pandas()
                adapted, contract, record = adapt_m100_aggregated(frame, node_id=Path(member.name).stem,
                    rack_id=spec["rack_id"], metadata=config["domain_metadata"])
                domain = contract.domain_id
                if domain in series:
                    raise ValueError("Duplicate M100 physical domain.")
                selected = adapted.reindex(grid)
                if not np.isfinite(selected.node_power_kw).all():
                    raise ValueError(f"{domain}: the predeclared common period has missing data; no replacement or interpolation.")
                series[domain], metadata[domain] = selected, contract
                coverage[domain] = pd.Series(1, index=grid, name="published_aggregate_present")
                audit[domain] = dict(**record, archive=path.name, member=member.name,
                    member_sha256=sha256(payload).hexdigest(), experiment_intervals=len(grid),
                    selected_period_missing=0, quality_screen="Fixed common period selected from availability masks only; not load amplitude or forecast error.")
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError("M100 source changed during adaptation.")
    if len(series) != config["expected_domains"]:
        raise ValueError("M100 node count differs from the predeclared dataset.")
    validate_compatible(list(metadata.values()))
    return DomainPanel(series, metadata, coverage, sources, anchor, audit)
