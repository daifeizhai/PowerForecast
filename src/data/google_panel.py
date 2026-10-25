"""Strict Google power-domain ingestion and shared per-series windows for SAPF."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from src.data.adapters import adapt_frame
from src.data.domains import DomainMetadata, validate_compatible
from src.data.features import trace_calendar_features
from src.data.resampling import resample_frame
from src.data.splitting import WindowBatch, build_windows, concatenate_domain_batches
from src.utils import resolve_path


@dataclass
class DomainPanel:
    """One complete timestamp grid per physical domain; missing targets stay missing."""
    series: dict[str, pd.DataFrame]
    metadata: dict[str, DomainMetadata]
    coverage: dict[str, pd.Series]
    sources: dict
    anchor: pd.Timestamp
    audit: dict

    @property
    def target_field(self):
        fields = {meta.target_field or "measured_power_util" for meta in self.metadata.values()}
        if len(fields) != 1:
            raise ValueError("A jointly evaluated panel needs one explicitly compatible target field.")
        return next(iter(fields))

    @property
    def frequency(self):
        frequencies = {m.evaluation_frequency for m in self.metadata.values()}
        if len(frequencies) != 1:
            raise ValueError("A panel must have one declared evaluation cadence.")
        return next(iter(frequencies))

    @property
    def points_per_day(self):
        return int(pd.Timedelta(days=1)/pd.Timedelta(self.frequency))


def load_google_domains(config: dict) -> DomainPanel:
    """Strict new-protocol adapter; the historical D14/D22 path remains unchanged."""
    root = resolve_path(config["raw_data"])
    files = sorted(root.glob("cell*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"Google domain files missing in {root}; no data are downloaded automatically.")
    anchor = pd.Timestamp(config["trace_start"])
    if anchor.tz is None:
        raise ValueError("Google trace_start must include the documented timezone offset.")
    anchor = anchor.tz_convert("UTC").as_unit("ns")
    expected_start = int(config["trace_start_microseconds"])
    days = int(config["expected_days"])
    grid = pd.date_range(anchor, periods=days*96, freq="15min", name="timestamp")
    mapping_path = root / "machine_to_pdu_mapping.csv.gz"
    mapping = pd.read_csv(mapping_path)
    if not {"machine_id", "cell", "pdu"}.issubset(mapping) or mapping.isna().any().any():
        raise ValueError("Google machine mapping requires nonmissing machine_id/cell/pdu.")
    memberships = mapping.groupby(["cell", "machine_id"]).pdu.agg(lambda values: sorted(set(values)))
    overlaps = {}
    ambiguous_counts = {}
    for (cell, _), domains in memberships[memberships.map(len) > 1].items():
        for domain in domains:
            key = f"cell{cell}_{domain}"
            overlaps.setdefault(key, set()).update(f"cell{cell}_{d}" for d in domains if d != domain)
            ambiguous_counts[key] = ambiguous_counts.get(key, 0)+1
    counts = mapping.groupby(["cell", "pdu"]).machine_id.nunique()
    series, metadata, coverage, sources, audit = {}, {}, {}, {}, {}
    for path in [mapping_path, *files]:
        sources[path.name] = dict(path=str(path), bytes=path.stat().st_size,
                                  sha256=sha256(path.read_bytes()).hexdigest())
    for path in files:
        raw = pd.read_csv(path)
        required = {"time", "cell", "pdu", "measured_power_util", "bad_measurement_data"}
        if not required.issubset(raw) or raw.empty:
            raise ValueError(f"{path.name}: missing Google fields or empty sequence.")
        if raw[["cell", "pdu"]].isna().any().any() or raw.cell.nunique() != 1 or raw.pdu.nunique() != 1:
            raise ValueError(f"{path.name}: each file must describe exactly one domain.")
        cell, power_domain = str(raw.cell.iloc[0]), str(raw.pdu.iloc[0])
        domain = f"cell{cell}_{power_domain}"
        if path.name != f"{domain}.csv.gz" or domain in series:
            raise ValueError(f"Duplicate or inconsistent Google domain identity: {path.name}")
        raw = raw.sort_values("time").reset_index(drop=True)
        micros = pd.to_numeric(raw.time, errors="raise").to_numpy(dtype=np.int64)
        if (np.diff(micros) <= 0).any() or ((micros-expected_start) % 300_000_000 != 0).any():
            raise ValueError(f"{domain}: duplicate or off-grid native 5min timestamps.")
        if micros[0] < expected_start or micros[-1] >= expected_start + days*86400*1_000_000:
            raise ValueError(f"{domain}: timestamps outside the declared trace; refusing to shift day numbering.")
        flags = raw.bad_measurement_data.astype(str).str.lower().str.strip()
        if not flags.isin(["true", "false", "1", "0"]).all():
            raise ValueError(f"{domain}: invalid measurement quality flags.")
        target = pd.to_numeric(raw.measured_power_util, errors="raise").to_numpy(float)
        valid = ~flags.isin(["true", "1"]).to_numpy() & np.isfinite(target) & (target >= 0)
        raw["measured_power_util"] = np.where(valid, target, np.nan)
        raw["timestamp"] = anchor + pd.to_timedelta(micros-expected_start, unit="us")
        adapted = adapt_frame(raw, {"timestamp": "timestamp", "columns": {"measured_power_util": "measured_power_util"}}, "UTC")
        sample_count = adapted.measured_power_util.resample("15min", closed="left", label="left").count().reindex(grid, fill_value=0)
        frame = resample_frame(adapted).reindex(grid)
        # Unlike the historical valid-slot mask, the new task requires a complete
        # three-sample target interval, including in history; never impute labels.
        frame.loc[sample_count < 3, "measured_power_util"] = np.nan
        spec = {**config["domain_metadata"], "site_id": f"cell{cell}", "domain_id": domain,
                "power_domain_id": power_domain}
        metadata[domain] = DomainMetadata(**spec)
        series[domain], coverage[domain] = frame, sample_count
        audit[domain] = dict(raw_rows=len(raw), raw_first_microseconds=int(micros[0]),
                            raw_last_microseconds=int(micros[-1]), bad_or_missing_raw=int((~valid).sum()),
                            missing_target_intervals=int(frame.measured_power_util.isna().sum()),
                            complete_15min_intervals=int((sample_count == 3).sum()),
                            mapped_machine_count=int(counts.get((cell, power_domain), 0)),
                            ambiguous_mapping_machine_count=ambiguous_counts.get(domain, 0),
                            mapping_overlap_domains=sorted(overlaps.get(domain, set())),
                            mapping_policy="Descriptive only; ambiguous memberships cannot justify aggregation or equipment independence.")
    if len(series) != int(config["expected_domains"]):
        raise ValueError(f"Expected {config['expected_domains']} Google domains, found {len(series)}.")
    validate_compatible(list(metadata.values()))
    return DomainPanel(series, metadata, coverage, sources, anchor, audit)


def domain_windows(panel: DomainPanel, domains: list[str], origins: list[int], context: int, horizon: int):
    """Build each series independently through the existing public window builder.

    Dk means the first k trace days are observed; the first label is D(k+1).
    Invalid windows are reported rather than padded. Domain IDs are side metadata.
    """
    if not domains or len(set(domains)) != len(domains) or not origins or len(set(origins)) != len(origins):
        raise ValueError("Choose distinct nonempty domains and origins.")
    batches = {}
    skipped = 0
    target_field = panel.target_field
    period = panel.points_per_day
    for domain in domains:
        frame = panel.series[domain]
        expected = pd.date_range(frame.index[0], periods=len(frame), freq=panel.frequency, name=frame.index.name)
        if not frame.index.equals(expected) or frame.index[0] != panel.anchor:
            raise ValueError(f"{domain}: preserve the complete declared grid and trace anchor.")
        known = trace_calendar_features(frame.index, panel.anchor)
        history = pd.concat([frame[[target_field]].astype(np.float32), known], axis=1)
        parts = {}
        for day in origins:
            position = int(day)*period
            if position < context or position+horizon > len(frame):
                raise ValueError(f"{domain}: insufficient full history/horizon at D{day}.")
            latency = panel.audit.get(domain, {}).get("history_availability_upper_bound_minutes", 0)
            if frame.index[position-1]+pd.Timedelta(minutes=latency) > frame.index[position]:
                raise ValueError("A published historical aggregate would be unavailable at the issue time.")
            # Check missingness without catching unrelated builder errors.
            if not np.isfinite(frame[target_field].iloc[position-context:position+horizon]).all():
                skipped += 1
                continue
            parts[str(day)] = build_windows(history, known, frame[target_field],
                                            (position, position+horizon), context, horizon, period, period)
        if parts:
            batches[domain] = concatenate_domain_batches(parts)
    if not batches:
        raise ValueError(f"No valid domain windows; skipped={skipped}, origins={origins}.")
    combined = concatenate_domain_batches(batches)
    return WindowBatch(combined.inputs, combined.targets, skipped)
