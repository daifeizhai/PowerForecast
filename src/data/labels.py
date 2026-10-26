"""Label-type taxonomy (task book §3.1 C).

The six label types must never be mixed inside one evaluation table.  This
module is the single place that maps a :class:`~src.data.contracts.TargetSpec`
(or an unlabelled dataset) onto that taxonomy, and it guards the main power
model so that node-level / utilisation / unlabelled data cannot silently
produce a facility-level score.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from src.data.contracts import (
    ContractError,
    DatasetRole,
    DatasetSpec,
    LabelOrigin,
    TargetEligibilityError,
    TargetKind,
    TargetScope,
    TargetSpec,
)


class LabelType(str, Enum):
    FACILITY_MEASURED_KW = "facility_measured_kw"
    FACILITY_DERIVED_KW = "facility_derived_kw"
    IT_MEASURED_KW = "it_measured_kw"
    NODE_MEASURED_KW = "node_measured_kw"
    RELATIVE_POWER = "relative_power"
    UNLABELED = "unlabeled"


LABEL_DESCRIPTIONS: dict[LabelType, str] = {
    LabelType.FACILITY_MEASURED_KW: "设施总有功功率实测（独立设施总表/总计量点）",
    LabelType.FACILITY_DERIVED_KW: "设施功率派生值（经校验的分项功率求和，不是独立总表实测）",
    LabelType.IT_MEASURED_KW: "IT 设备或系统级实测功率（不是设施整体）",
    LabelType.NODE_MEASURED_KW: "服务器/节点级实测功率（不是设施整体）",
    LabelType.RELATIVE_POWER: "功率利用率或归一化功率（无量纲，不能直接换算为 kW）",
    LabelType.UNLABELED: "无功率标签（仅运行指标，可用于无标签预训练）",
}

#: Label types that may carry a facility-level kW main task.
FACILITY_KW_LABEL_TYPES: frozenset[LabelType] = frozenset(
    {LabelType.FACILITY_MEASURED_KW, LabelType.FACILITY_DERIVED_KW}
)


def classify_target(target: TargetSpec | None) -> LabelType:
    """Map a declared target to exactly one label type."""
    if target is None:
        return LabelType.UNLABELED
    if target.kind is TargetKind.POWER_UTILIZATION:
        return LabelType.RELATIVE_POWER
    if target.label_origin is LabelOrigin.POWER_UTILIZATION:
        return LabelType.RELATIVE_POWER
    if target.kind is not TargetKind.ACTIVE_POWER:
        raise TargetEligibilityError(
            f"Label type of target kind {target.kind.value!r} is not defined by the taxonomy."
        )
    if target.label_origin is LabelOrigin.ESTIMATED:
        raise TargetEligibilityError(
            "estimated power is not allowed as a main-task label: it is a model output, not an "
            "observation. Register it as an auxiliary proxy instead."
        )
    if target.scope is TargetScope.IT:
        return LabelType.IT_MEASURED_KW
    if target.scope in {TargetScope.NODE, TargetScope.SERVER, TargetScope.RACK}:
        return LabelType.NODE_MEASURED_KW
    if target.scope in {TargetScope.FACILITY, TargetScope.POWER_DOMAIN}:
        if target.label_origin is LabelOrigin.COMPONENT_SUM:
            return LabelType.FACILITY_DERIVED_KW
        return LabelType.FACILITY_MEASURED_KW
    raise TargetEligibilityError(
        f"Label type of target scope {target.scope.value!r} is not defined by the taxonomy."
    )


def classify_dataset(spec: DatasetSpec) -> LabelType:
    """Label type of a whole dataset description (role + target)."""
    if spec.role is DatasetRole.UNLABELED_PRETRAIN:
        if spec.target is not None:
            raise ContractError("An unlabelled dataset cannot declare a target label.")
        return LabelType.UNLABELED
    return classify_target(spec.target)


def require_main_model_label(label_type: LabelType, *, dataset_id: str = "") -> None:
    """The main power model A only accepts facility-level kW labels."""
    if label_type not in FACILITY_KW_LABEL_TYPES:
        where = f" for dataset {dataset_id!r}" if dataset_id else ""
        raise TargetEligibilityError(
            f"Main power model requires facility-level kW labels{where}; got {label_type.value!r} "
            f"({LABEL_DESCRIPTIONS[label_type]}). Node/IT/utilisation/unlabelled labels belong to "
            "explicitly named auxiliary tasks, never to the facility main task."
        )


def require_no_label_mixing(label_types: Iterable[LabelType]) -> LabelType:
    """One evaluation table may only contain one label type (task book §3.1 C)."""
    unique = set(label_types)
    if len(unique) > 1:
        raise TargetEligibilityError(
            "An evaluation table mixes label types "
            f"{sorted(label_type.value for label_type in unique)}; facility kW, derived kW, "
            "IT kW, node kW and relative power must be reported separately."
        )
    if not unique:
        raise ContractError("No label types were supplied for an evaluation table.")
    return unique.pop()


def describe_label_types() -> dict[str, str]:
    return {label.value: LABEL_DESCRIPTIONS[label] for label in LabelType}
