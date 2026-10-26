"""Data layer: contracts, sources, protocols, windows.

The public entry points are re-exported here so callers do not need to know the
module layout.  Unlabelled and labelled flows stay strictly separate.
"""

from src.data.contracts import (
    TARGET_COLUMN,
    AuditStatus,
    ContractError,
    DatasetRole,
    DatasetSpec,
    FeatureSpec,
    ForecastBatch,
    ForecastInputs,
    InsufficientDataError,
    IntervalStatistic,
    LabelOrigin,
    LeakageError,
    PhysicalUnitError,
    PretrainBatch,
    TargetEligibilityError,
    TargetKind,
    TargetScope,
    TargetSpec,
    from_kw,
    require_facility_active_power_kw,
    require_pretrain_batch,
    to_kw,
)

__all__ = [
    "TARGET_COLUMN",
    "AuditStatus",
    "ContractError",
    "DatasetRole",
    "DatasetSpec",
    "FeatureSpec",
    "ForecastBatch",
    "ForecastInputs",
    "InsufficientDataError",
    "IntervalStatistic",
    "LabelOrigin",
    "LeakageError",
    "PhysicalUnitError",
    "PretrainBatch",
    "TargetEligibilityError",
    "TargetKind",
    "TargetScope",
    "TargetSpec",
    "from_kw",
    "require_facility_active_power_kw",
    "require_pretrain_batch",
    "to_kw",
]
