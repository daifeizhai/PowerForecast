"""Physical label contracts kept separate from numerical forecasting features."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DomainMetadata:
    dataset_id: str
    site_id: str
    domain_id: str
    target_kind: str
    target_scope: str
    target_unit: str
    label_origin: str
    timezone: str
    timezone_verified: bool
    raw_frequency: str
    sampling_semantics: str
    interval_definition: str
    coverage: str
    source_version: str
    power_domain_id: str | None = None
    rack_id: str | None = None
    node_id: str | None = None
    mapping_route: str | None = None
    assumptions: tuple[str, ...] = ()
    feature_availability: dict = field(default_factory=dict)
    target_field: str | None = None
    usage_role: str = "development"
    evaluation_frequency: str = "15min"

    def __post_init__(self):
        required = (self.dataset_id, self.site_id, self.domain_id, self.target_scope,
                    self.timezone, self.interval_definition, self.coverage, self.source_version)
        if not all(isinstance(v, str) and v.strip() for v in required):
            raise ValueError("Domain identity, time, coverage and source version must be explicit.")
        if not isinstance(self.timezone_verified, bool):
            raise ValueError("timezone_verified must be a boolean, not an inferred truth value.")
        if self.usage_role not in {"development", "final_holdout"}:
            raise ValueError("usage_role must be development or final_holdout.")
        if self.label_origin not in {"measured", "energy_converted", "component_sum", "calibrated_estimate",
                                     "assumed_scenario", "not_estimable"}:
            raise ValueError(f"Unknown label_origin: {self.label_origin}")
        units = {"active_power": {"W", "kW"}, "energy": {"Wh", "kWh"},
                 "power_utilization": {"utilization_fraction", "%"}}
        if self.target_kind not in units or self.target_unit not in units[self.target_kind]:
            raise ValueError(f"Incompatible target_kind/unit: {self.target_kind}/{self.target_unit}")
        raw_seconds = pd.Timedelta(self.raw_frequency).total_seconds()
        output_seconds = pd.Timedelta(self.evaluation_frequency).total_seconds()
        if output_seconds not in {900, 1800} or raw_seconds <= 0 or raw_seconds > output_seconds or output_seconds % raw_seconds:
            raise ValueError("Native sampling cannot form the declared 15min/30min labels; no interpolation is permitted.")
        if self.sampling_semantics not in {"sample_mean", "interval_mean", "interval_energy"}:
            raise ValueError("Declare sample_mean, interval_mean or interval_energy sampling semantics.")
        if self.target_kind == "energy" and self.sampling_semantics != "interval_energy":
            raise ValueError("Cumulative energy needs a separately audited conversion.")
        if self.label_origin == "component_sum" and not self.mapping_route:
            raise ValueError("component_sum requires its mapping route and coverage.")
        forbidden = [k for k, v in self.feature_availability.items()
                     if v == "future_known" and k not in {"calendar", "trace_calendar"}]
        if forbidden:
            raise ValueError(f"Only calendar is currently future_known: {forbidden}")

    def to_dict(self) -> dict:
        return asdict(self)


def validate_compatible(metadata: list[DomainMetadata]) -> None:
    """Require one measurement boundary and label origin before joint training."""
    if not metadata or len({m.domain_id for m in metadata}) != len(metadata):
        raise ValueError("Choose distinct, explicitly identified domains.")
    keys = {(m.target_kind, m.target_scope, m.target_unit, m.label_origin, m.sampling_semantics,
             m.interval_definition, m.evaluation_frequency) for m in metadata}
    if len(keys) != 1 or metadata[0].label_origin == "not_estimable":
        raise ValueError("Incompatible physical targets, scopes, units, label origins or intervals.")


def assert_data_usage(metadata: DomainMetadata, phase: str) -> None:
    """Reserve Eagle as a whole dataset; observed history is allowed only at inference."""
    identity = metadata.dataset_id.lower().replace("-", "_")
    reserved = metadata.usage_role == "final_holdout" or identity in {"nlr_eagle", "nrel_eagle", "eagle"}
    if reserved and phase not in {"test", "inference_history"}:
        raise ValueError(f"{metadata.dataset_id} is a final holdout; no training, calibration, validation or strategy design.")


def convert_target(values, source: DomainMetadata, *, target_kind: str, target_scope: str,
                   target_unit: str, interval_hours: float | None = None):
    """Explicit scale/energy conversion; never change the measured boundary."""
    if source.target_scope != target_scope:
        raise ValueError("Conversion cannot change target_scope or equipment coverage.")
    result = np.array(values, dtype=float, copy=True)
    if source.target_kind == target_kind and source.target_unit == target_unit:
        return result
    if source.target_kind == target_kind == "active_power":
        factors = {("W", "kW"): .001, ("kW", "W"): 1000.}
        if (source.target_unit, target_unit) in factors:
            return result * factors[source.target_unit, target_unit]
    if (source.target_kind == "energy" and target_kind == "active_power"
            and source.sampling_semantics == "interval_energy" and interval_hours is not None
            and np.isfinite(interval_hours) and interval_hours > 0):
        if source.target_unit in {"Wh", "kWh"} and target_unit in {"W", "kW"}:
            # This contract describes individual native intervals, not cumulative meters.
            if not np.isclose(interval_hours, pd.Timedelta(source.raw_frequency).total_seconds()/3600):
                raise ValueError("Energy interval duration disagrees with the source contract.")
            energy_kwh = result * (.001 if source.target_unit == "Wh" else 1.)
            return energy_kwh / interval_hours * (1000. if target_unit == "W" else 1.)
    raise ValueError("Unsupported physical conversion; utilization/chip/GPU/IT/facility labels are not interchangeable.")


def aggregate_components(components: dict[str, pd.Series], *, expected_components: list[str],
                         complete_coverage: bool, nonoverlapping: bool) -> pd.Series:
    """New aggregations require evidence of complete, disjoint, synchronous coverage."""
    if not complete_coverage or not nonoverlapping or not expected_components:
        raise ValueError("Aggregation requires complete coverage and nonoverlapping components.")
    if len(set(expected_components)) != len(expected_components) or set(components) != set(expected_components):
        raise ValueError("Missing or duplicate aggregation components.")
    indices = [s.index for s in components.values()]
    if any(not indices[0].equals(index) for index in indices[1:]):
        raise ValueError("Aggregation components must have identical intervals.")
    return pd.concat(components, axis=1).sum(axis=1, min_count=len(components))


def metadata_from_config(config: dict) -> DomainMetadata | None:
    spec = config.get("domain_metadata")
    if not spec:
        return None
    metadata = DomainMetadata(**spec)
    if metadata.target_field is not None and config.get("target", metadata.target_field) != metadata.target_field:
        raise ValueError("Target field changed: update its physical domain_metadata contract before running.")
    for field in ("target_unit", "timezone", "timezone_verified"):
        if field in config and config[field] != getattr(metadata, field):
            raise ValueError(f"Dataset and domain_metadata disagree on {field}.")
    return metadata
