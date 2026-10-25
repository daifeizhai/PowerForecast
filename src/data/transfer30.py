"""Native published-slot aggregation, with explicit original-clock provenance."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from src.data.domains import DomainMetadata, validate_compatible
from src.data.google_panel import DomainPanel
from src.utils import load_config, resolve_path


def file_hash(path):
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_published_power(frame, *, timestamp, power, start, days, native_minutes):
    """No interpolation: each 30min label needs every unique native published slot."""
    if native_minutes not in {10, 15}:
        raise ValueError("The common protocol requires native 10min or 15min published slots.")
    times = pd.DatetimeIndex(pd.to_datetime(frame[timestamp], errors="raise")).as_unit("ns")
    if times.hasnans or np.any(times.asi8 % (native_minutes*60*10**9)):
        raise ValueError("Native power timestamps must be nonmissing and aligned.")
    start = pd.Timestamp(start)
    if (times.tz is None) != (start.tz is None):
        raise ValueError("Never silently assign UTC to a naive source clock.")
    grid = pd.date_range(start, periods=days*1440//native_minutes, freq=f"{native_minutes}min")
    values = pd.to_numeric(frame[power], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    duplicate = times.duplicated(keep=False)
    valid = np.isfinite(values) & (values >= 0)
    observed = pd.Series(np.where(valid, values, np.nan), index=times).loc[~duplicate].reindex(grid)
    # The grid is explicit, so missing rows and NaNs both invalidate the bin.
    per_bin = 30//native_minutes
    blocks = observed.to_numpy().reshape(-1, per_bin)
    counts = np.isfinite(blocks).sum(1)
    means = np.where(counts == per_bin, blocks.mean(1)/1000., np.nan)
    source_clock = grid[::per_bin]
    return pd.DataFrame({"node_power_kw": means, "source_clock_timestamp": [t.isoformat() for t in source_clock]},
                        index=source_clock), counts


def load_transfer_domains(dataset, roles):
    groups = ("m100", "ucc")
    assigned = {g: set(sum((roles[g][r] for r in ("source", "validation", "test")), [])) for g in groups}
    retired = set(roles["retired_m100_previous_test"])
    if assigned["m100"] & retired:
        raise ValueError("Previously evaluated M100 test identities are retired from the new studies.")
    protocol_anchor = pd.Timestamp(dataset["protocol_anchor"]).as_unit("ns")
    protocol_grid = pd.date_range(protocol_anchor, periods=dataset["days"]*48, freq="30min", name="timestamp")
    series, metadata, coverage, sources, audit = {}, {}, {}, {}, {}

    def add(frame, counts, domain, group, native, source_id, **extra):
        if domain not in assigned[group] or frame.node_power_kw.isna().any():
            raise ValueError(f"{domain}: fixed assigned domain or complete 30min labels unavailable; no replacement.")
        original_start = frame.index[0].isoformat()
        frame.index = protocol_grid
        series[domain], coverage[domain] = frame, pd.Series(counts, index=protocol_grid)
        test = domain in roles[group]["test"]
        metadata[domain] = DomainMetadata(dataset_id="m100_exadata_30min" if group == "m100" else "ucc2023_30min",
            site_id="marconi100" if group == "m100" else "ucc_research_cloud", domain_id=domain,
            node_id=str(source_id), rack_id=extra.get("rack_id"), target_field="node_power_kw", target_kind="active_power",
            target_scope="node", target_unit="kW", label_origin="measured", raw_frequency=f"{native}min",
            evaluation_frequency="30min", timezone="UTC" if group == "m100" else "unverified_source_local_clock",
            timezone_verified=group == "m100", sampling_semantics="interval_mean",
            interval_definition="Arithmetic mean of complete native published slots in each 30min source-clock bucket; protocol coordinate is not physical UTC.",
            coverage=f"Exactly {30//native} unique finite native power labels; no interpolation; no raw underlying sample-count claim.",
            source_version="Official local published power files; SHA256 pinned; original clocks retained.",
            assumptions=("Published bucket available by its label plus 30min, with no additional reporting delay.",
                         "Offline cross-dataset benchmark across nonoverlapping collection years; not prospective deployment chronology."),
            usage_role="final_holdout" if test else "development",
            feature_availability={"target": "history_only", "trace_calendar": "future_known"})
        audit[domain] = dict(group=group, source_clock_start=original_start,
            source_clock_timezone_verified=group == "m100", history_availability_upper_bound_minutes=30,
            native_slots_per_bin=30//native, complete_bins=len(frame), **extra)

    m100 = load_config(dataset["m100_config"])
    for spec in m100["archives"]:
        path = resolve_path(m100["raw_data"])/spec["filename"]
        digest = file_hash(path)
        if digest != spec["sha256"]:
            raise ValueError("M100 archive hash changed.")
        sources["M100/"+path.name] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                domain = f"m100_rack{spec['rack_id']}_node{Path(member.name).stem}"
                if domain not in assigned["m100"]:
                    continue  # Retired/quality-excluded node records are not opened.
                raw = pd.read_parquet(BytesIO(archive.extractfile(member).read()), columns=["timestamp", "total_power_avg"])
                frame, counts = aggregate_published_power(raw, timestamp="timestamp", power="total_power_avg",
                    start=dataset["m100_clock_start"], days=dataset["days"], native_minutes=15)
                add(frame, counts, domain, "m100", 15, Path(member.name).stem, rack_id=str(spec["rack_id"]))
    for spec in dataset["ucc_files"]:
        path = resolve_path(dataset["ucc_root"])/spec["filename"]
        digest = file_hash(path)
        if digest != spec["sha256"]:
            raise ValueError("UCC CSV hash changed.")
        sources["UCC/"+path.name] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
        raw = pd.read_csv(path, usecols=["Series", "Time", "Power"])
        raw.Series = raw.Series.str.strip()
        for source_id, group in raw.groupby("Series", sort=True):
            domain = "ucc_"+spec["cohort"]+"_"+source_id.replace("-", "_")
            if domain not in assigned["ucc"]:
                continue
            frame, counts = aggregate_published_power(group, timestamp="Time", power="Power",
                start=dataset["ucc_clock_start"], days=dataset["days"], native_minutes=10)
            add(frame, counts, domain, "ucc", 10, source_id, hardware_cohort=spec["cohort"])
    if set(series) != set.union(*assigned.values()):
        raise ValueError("Assigned transfer domain files are missing.")
    validate_compatible(list(metadata.values()))
    return DomainPanel(series, metadata, coverage, sources, protocol_anchor, audit)
