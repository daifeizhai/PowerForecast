"""Field-semantic registry: cross-domain alignment and target-only field routing.

The registry answers three questions that string matching cannot:

1. Which input fields are genuinely **comparable** across a pretraining source
   and the target domain? Comparability requires the same canonical semantics,
   the same unit, and a compatible interval statistic. Two columns that merely
   share a name are *not* merged when their unit or statistic differs.
2. Which fields are **source-only** (unlabelled pretraining representation) and
   which are **target-only** (kept for a local supervised branch)?
3. What are the fusion weights? The global fusion weight is explicitly
   **unresolved** (``None``). The only value defined in this round is the local
   routing weight ``target_only_local_weight = 1.0``, which means "do not
   additionally suppress a target-only field inside its own local branch". It is
   *not* a physical regression coefficient and *not* a share of the final
   prediction.

The registry never fabricates alignment: when no comparable shared feature
exists it returns an explicit ``no_shared_features`` status with reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from src.data.contracts import (
    REAL_MEASUREMENT_ORIGINS,
    AuditStatus,
    ContractError,
    DatasetRole,
    DatasetSpec,
    FeatureSpec,
    IntervalStatistic,
    LabelOrigin,
    TargetKind,
    TargetScope,
    TargetSpec,
    normalize_unit,
    require_facility_active_power_kw,
)
from src.utils import resolve_path

#: Local routing weight for a target-only field inside its own local branch.
TARGET_ONLY_LOCAL_WEIGHT: float = 1.0

#: The global fusion weight is a research decision that is NOT taken this round.
UNRESOLVED_GLOBAL_FUSION_WEIGHT: None = None

COMPARABLE_STATISTICS: frozenset[tuple[IntervalStatistic, IntervalStatistic]] = frozenset(
    {
        (IntervalStatistic.INTERVAL_MEAN, IntervalStatistic.INTERVAL_MEAN),
        (IntervalStatistic.INSTANTANEOUS, IntervalStatistic.INSTANTANEOUS),
        (IntervalStatistic.INTERVAL_INTEGRAL, IntervalStatistic.INTERVAL_INTEGRAL),
        (IntervalStatistic.EVENT_STATE, IntervalStatistic.EVENT_STATE),
    }
)


class FeatureRelation(str, Enum):
    SHARED = "shared"
    SOURCE_ONLY = "source_only"
    TARGET_ONLY = "target_only"
    INCOMPARABLE = "incomparable"


class AlignmentStatus(str, Enum):
    READY = "ready"
    NO_SHARED_FEATURES = "no_shared_features"
    PENDING_ADAPTATION = "pending_adaptation"


@dataclass(frozen=True)
class FieldRoute:
    canonical_name: str
    relation: FeatureRelation
    unit: str
    statistic: IntervalStatistic
    domains: tuple[str, ...]
    raw_names: Mapping[str, str]
    target_only_local_weight: float | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "relation": self.relation.value,
            "unit": self.unit,
            "statistic": self.statistic.value,
            "domains": list(self.domains),
            "raw_names": dict(self.raw_names),
            "target_only_local_weight": self.target_only_local_weight,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AlignmentResult:
    """Outcome of aligning one pretraining source with the target domain."""

    source_id: str
    target_id: str
    shared_features: tuple[str, ...]
    source_only_features: tuple[str, ...]
    target_only_features: tuple[str, ...]
    incomparable_features: tuple[str, ...]
    status: AlignmentStatus
    target_only_local_weight: float
    global_fusion_weight: None
    blocked_reasons: tuple[str, ...] = ()
    routes: tuple[FieldRoute, ...] = ()

    @property
    def is_transferable(self) -> bool:
        return self.status is AlignmentStatus.READY and bool(self.shared_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "shared_features": list(self.shared_features),
            "source_only_features": list(self.source_only_features),
            "target_only_features": list(self.target_only_features),
            "incomparable_features": list(self.incomparable_features),
            "status": self.status.value,
            "is_transferable": self.is_transferable,
            "target_only_local_weight": self.target_only_local_weight,
            "global_fusion_weight": self.global_fusion_weight,
            "blocked_reasons": list(self.blocked_reasons),
            "routes": [route.to_dict() for route in self.routes],
        }


def _comparable(left: FeatureSpec, right: FeatureSpec) -> tuple[bool, tuple[str, ...]]:
    notes: list[str] = []
    if normalize_unit(left.unit).lower() != normalize_unit(right.unit).lower():
        notes.append(
            f"unit differs ({left.unit!r} vs {right.unit!r}); values are not comparable "
            "without an explicit, audited conversion"
        )
    if (left.statistic, right.statistic) not in COMPARABLE_STATISTICS:
        notes.append(
            f"statistic differs ({left.statistic.value!r} vs {right.statistic.value!r}); "
            "an interval mean and an instantaneous sample are different quantities"
        )
    if left.future_known != right.future_known:
        notes.append("one side is known-future and the other is observed-only")
    return (not notes, tuple(notes))


def _target_only_config(alignment: AlignmentResult) -> dict[str, Any]:
    """Configuration handed to a future local-branch model."""
    return {
        "target_only_features": list(alignment.target_only_features),
        "target_only_local_weight": TARGET_ONLY_LOCAL_WEIGHT,
        "global_fusion_weight": UNRESOLVED_GLOBAL_FUSION_WEIGHT,
        "status": "local_routing_only",
        "notes": (
            "target_only_local_weight=1.0 only means the local branch does not additionally "
            "suppress these fields. It is not a physical coefficient and not a global fusion "
            "share; the global fusion weight is unresolved by research decision."
        ),
    }


class FieldRegistry:
    """Registers dataset feature semantics and aligns source/target fields."""

    def __init__(self) -> None:
        self._specs: dict[str, DatasetSpec] = {}

    # -- registration -----------------------------------------------------
    def register(self, spec: DatasetSpec) -> None:
        if not isinstance(spec, DatasetSpec):
            raise ContractError("FieldRegistry.register expects a DatasetSpec.")
        key = spec.dataset_id
        if key in self._specs and self._specs[key] is not spec:
            raise ContractError(f"Dataset {key!r} is already registered with different content.")
        self._specs[key] = spec

    def register_many(self, specs: Iterable[DatasetSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, dataset_id: str) -> DatasetSpec:
        try:
            return self._specs[dataset_id]
        except KeyError as exc:
            raise ContractError(
                f"Dataset {dataset_id!r} is not registered. Known: {sorted(self._specs)}."
            ) from exc

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    # -- alignment --------------------------------------------------------
    def align(self, *, source_id: str, target_id: str) -> AlignmentResult:
        source = self.get(source_id)
        target = self.get(target_id)
        if source.role is DatasetRole.UNLABELED_PRETRAIN and source.target is not None:
            raise ContractError(f"Unlabelled source {source_id!r} must not declare a target.")
        if target.role is DatasetRole.UNLABELED_PRETRAIN:
            raise ContractError(
                f"Target domain {target_id!r} is declared unlabelled; alignment needs a supervised "
                "target domain."
            )
        # Only features that may enter a model at all (observed inputs) take part.
        source_features = {
            name: spec for name, spec in source.features.items() if not spec.future_known
        }
        target_features = {
            name: spec for name, spec in target.features.items() if not spec.future_known
        }
        shared: list[str] = []
        incomparable: list[str] = []
        source_only = sorted(set(source_features) - set(target_features))
        target_only = sorted(set(target_features) - set(source_features))
        routes: dict[str, FieldRoute] = {}
        blocked: list[str] = []

        for name in sorted(set(source_features) & set(target_features)):
            left, right = source_features[name], target_features[name]
            comparable, notes = _comparable(left, right)
            if comparable:
                shared.append(name)
                routes[name] = FieldRoute(
                    canonical_name=name,
                    relation=FeatureRelation.SHARED,
                    unit=left.unit,
                    statistic=left.statistic,
                    domains=(source_id, target_id),
                    raw_names={source_id: left.raw_name, target_id: right.raw_name},
                )
            else:
                incomparable.append(name)
                routes[name] = FieldRoute(
                    canonical_name=name,
                    relation=FeatureRelation.INCOMPARABLE,
                    unit=f"{left.unit} != {right.unit}",
                    statistic=left.statistic,
                    domains=(source_id, target_id),
                    raw_names={source_id: left.raw_name, target_id: right.raw_name},
                    notes=notes,
                )
                blocked.append(f"{name}: " + "; ".join(notes))

        for name in source_only:
            spec = source_features[name]
            routes[name] = FieldRoute(
                canonical_name=name,
                relation=FeatureRelation.SOURCE_ONLY,
                unit=spec.unit,
                statistic=spec.statistic,
                domains=(source_id,),
                raw_names={source_id: spec.raw_name},
            )
        for name in target_only:
            spec = target_features[name]
            routes[name] = FieldRoute(
                canonical_name=name,
                relation=FeatureRelation.TARGET_ONLY,
                unit=spec.unit,
                statistic=spec.statistic,
                domains=(target_id,),
                raw_names={target_id: spec.raw_name},
                target_only_local_weight=TARGET_ONLY_LOCAL_WEIGHT,
                notes=(
                    "kept with its raw value, mask and source description; routed to a local "
                    "branch with target_only_local_weight=1.0 (no global weight assigned)",
                ),
            )

        if not shared:
            status = AlignmentStatus.NO_SHARED_FEATURES
            blocked.append(
                "no comparable shared feature between "
                f"{source_id!r} and {target_id!r}; transfer is not available yet and no "
                "fake shared representation is fabricated"
            )
        elif incomparable:
            status = AlignmentStatus.PENDING_ADAPTATION
        else:
            status = AlignmentStatus.READY

        return AlignmentResult(
            source_id=source_id,
            target_id=target_id,
            shared_features=tuple(shared),
            source_only_features=tuple(source_only),
            target_only_features=tuple(target_only),
            incomparable_features=tuple(incomparable),
            status=status,
            target_only_local_weight=TARGET_ONLY_LOCAL_WEIGHT,
            global_fusion_weight=UNRESOLVED_GLOBAL_FUSION_WEIGHT,
            blocked_reasons=tuple(blocked),
            routes=tuple(routes[name] for name in sorted(routes)),
        )

    def target_only_config(self, *, source_id: str, target_id: str) -> dict[str, Any]:
        return _target_only_config(self.align(source_id=source_id, target_id=target_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasets": [spec.to_dict() for spec in self._specs.values()],
            "target_only_local_weight": TARGET_ONLY_LOCAL_WEIGHT,
            "global_fusion_weight": UNRESOLVED_GLOBAL_FUSION_WEIGHT,
        }


# ---------------------------------------------------------------------------
# dataset-source registry (roles / audit status / label honesty)
# ---------------------------------------------------------------------------

REGISTRY_PATH = "configs/datasets/registry.yaml"

REGISTRY_ROLES = frozenset(
    {"unlabeled_pretrain", "target_supervised", "final_holdout", "candidate", "demo"}
)
LABEL_KEYS = frozenset(
    {
        "target_kind",
        "target_scope",
        "target_unit",
        "label_origin",
        "measurement_boundary",
        "sampling_semantics",
        "synthetic",
    }
)


def load_source_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load and validate ``configs/datasets/registry.yaml``."""
    with resolve_path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or "sources" not in payload:
        raise ContractError(f"{path} must contain a top-level 'sources' mapping.")
    return validate_source_registry(payload)


def validate_source_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check role/audit/label consistency and compute formal eligibility."""
    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ContractError("Registry 'sources' must be a nonempty mapping.")
    described: dict[str, dict[str, Any]] = {}
    for source_id, entry in sources.items():
        if not isinstance(entry, Mapping):
            raise ContractError(f"Registry entry {source_id!r} must be a mapping.")
        role = entry.get("candidate_role")
        if role not in REGISTRY_ROLES:
            raise ContractError(
                f"Registry entry {source_id!r} has candidate_role={role!r}; expected one of "
                f"{sorted(REGISTRY_ROLES)}."
            )
        audit = entry.get("audit_status")
        if audit not in {status.value for status in AuditStatus}:
            raise ContractError(f"Registry entry {source_id!r} has invalid audit_status={audit!r}.")
        label_payload = entry.get("label")
        target: TargetSpec | None = None
        if label_payload is not None:
            if not isinstance(label_payload, Mapping):
                raise ContractError(f"Registry entry {source_id!r} label must be a mapping or null.")
            missing = LABEL_KEYS - set(label_payload)
            if missing:
                raise ContractError(
                    f"Registry entry {source_id!r} label is missing keys: {sorted(missing)}."
                )
            target = TargetSpec(
                kind=TargetKind(label_payload["target_kind"]),
                scope=TargetScope(label_payload["target_scope"]),
                unit=str(label_payload["target_unit"]),
                label_origin=LabelOrigin(label_payload["label_origin"]),
                measurement_boundary=str(label_payload["measurement_boundary"]),
                sampling_semantics=str(label_payload["sampling_semantics"]),
                synthetic=bool(label_payload["synthetic"]),
            )
        if role == "unlabeled_pretrain" and target is not None:
            raise ContractError(
                f"Registry entry {source_id!r} is unlabeled_pretrain but declares a label; "
                "unlabelled sources must not claim kW."
            )
        claimed = bool(entry.get("can_produce_facility_kw", False))
        eligible, reason = _facility_eligibility(target, audit)
        if claimed and not eligible:
            raise ContractError(
                f"Registry entry {source_id!r} claims can_produce_facility_kw but is not eligible: "
                f"{reason}"
            )
        described[source_id] = {
            "display_name": entry.get("display_name", source_id),
            "site_id": entry.get("site_id"),
            "domain_id": entry.get("domain_id", source_id),
            "candidate_role": role,
            "native_frequency": entry.get("native_frequency"),
            "timezone": entry.get("timezone"),
            "timezone_verified": bool(entry.get("timezone_verified", False)),
            "audit_status": audit,
            "label": target.to_dict() if target is not None else None,
            "can_produce_facility_kw": bool(claimed and eligible),
            "facility_kw_eligible": eligible,
            "blocked_reason": None if eligible else reason,
            "notes": entry.get("notes", ""),
        }
    return {
        "sources": described,
        "target_only_local_weight": TARGET_ONLY_LOCAL_WEIGHT,
        "global_fusion_weight": UNRESOLVED_GLOBAL_FUSION_WEIGHT,
    }


def _facility_eligibility(target: TargetSpec | None, audit_status: str) -> tuple[bool, str | None]:
    if target is None:
        return False, "no power label declared"
    try:
        require_facility_active_power_kw(target, allow_synthetic=False)
    except Exception as exc:  # TargetEligibilityError / PhysicalUnitError
        return False, str(exc)
    if target.label_origin not in REAL_MEASUREMENT_ORIGINS:
        return False, f"label_origin={target.label_origin.value!r} is not a real measurement"
    if audit_status != AuditStatus.VERIFIED.value:
        return False, f"audit_status={audit_status!r} is not verified"
    return True, None


def describe_roles(registry: Mapping[str, Any]) -> dict[str, list[str]]:
    """Group registered sources by role for documentation and manifests."""
    grouped: dict[str, list[str]] = {}
    for source_id, entry in registry["sources"].items():
        grouped.setdefault(entry["candidate_role"], []).append(source_id)
    return {role: sorted(ids) for role, ids in sorted(grouped.items())}
