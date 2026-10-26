"""Field-registry acceptance tests (task book T06, T07).

Heterogeneous unlabelled sources must be readable independently, comparability
must be judged on semantics/unit/statistic rather than on raw column names, and
target-only fields must be preserved with a local weight of 1.0 while the global
fusion weight stays unresolved.
"""

from __future__ import annotations

import pytest

from src.data.contracts import (
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
)
from src.data.registry import (
    AlignmentStatus,
    FeatureRelation,
    FieldRegistry,
    TARGET_ONLY_LOCAL_WEIGHT,
    UNRESOLVED_GLOBAL_FUSION_WEIGHT,
    describe_roles,
    load_source_registry,
    validate_source_registry,
)

SYNTHETIC_TARGET = TargetSpec(
    kind=TargetKind.ACTIVE_POWER,
    scope=TargetScope.FACILITY,
    unit="kW",
    label_origin=LabelOrigin.SYNTHETIC_SIMULATED,
    measurement_boundary="simulated facility bus",
    sampling_semantics="15-minute interval mean",
    synthetic=True,
)


def unlabelled_spec(dataset_id: str, features: dict[str, FeatureSpec]) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=dataset_id,
        site_id="source_site",
        domain_id=dataset_id,
        role=DatasetRole.UNLABELED_PRETRAIN,
        native_frequency="1min",
        timezone="UTC",
        timezone_verified=False,
        features=features,
        target=None,
        audit_status=AuditStatus.PARTIAL,
    )


def target_spec(features: dict[str, FeatureSpec]) -> DatasetSpec:
    return DatasetSpec(
        dataset_id="target_facility",
        site_id="target_site",
        domain_id="target_facility",
        role=DatasetRole.TARGET_SUPERVISED,
        native_frequency="15min",
        timezone="UTC",
        timezone_verified=True,
        trace_anchor_timestamp="2024-01-01T00:00:00Z",
        interval_semantics="15-minute interval mean",
        availability_semantics="reported at interval end",
        features=features,
        target=SYNTHETIC_TARGET,
        provenance={"synthetic": True},
        audit_status=AuditStatus.VERIFIED,
    )


def feature(name: str, unit: str = "ratio", **kwargs) -> FeatureSpec:
    return FeatureSpec(raw_name=name, canonical_name=name, unit=unit, **kwargs)


def registry() -> FieldRegistry:
    source_a = unlabelled_spec(
        "source_a", {"cpu_util": feature("cpu_util"), "mem_util": feature("mem_util")}
    )
    source_b = unlabelled_spec(
        "source_b",
        {"gpu_util": feature("gpu_util"), "task_count": feature("task_count", "count")},
    )
    target = target_spec(
        {
            "cpu_util": feature("cpu_util"),
            "gpu_util": feature("gpu_util"),
            "cooling_kw": feature("cooling_kw", "kW"),
            "extra_metric": feature("extra_metric"),
        }
    )
    registry = FieldRegistry()
    registry.register_many([source_a, source_b, target])
    return registry


# ---------------------------------------------------------------------------
# T06 - heterogeneous unlabelled sources, semantic (not positional) alignment
# ---------------------------------------------------------------------------


def test_unlabelled_sources_keep_their_own_feature_sets() -> None:
    fields = registry()
    assert fields.get("source_a").feature_names == ("cpu_util", "mem_util")
    assert fields.get("source_b").feature_names == ("gpu_util", "task_count")
    assert fields.get("source_a").target is None
    assert fields.get("source_b").target is None


def test_alignment_is_semantic_not_positional() -> None:
    result = registry().align(source_id="source_a", target_id="target_facility")
    assert result.shared_features == ("cpu_util",)
    assert result.source_only_features == ("mem_util",)
    assert result.target_only_features == ("cooling_kw", "extra_metric", "gpu_util")
    assert result.status is AlignmentStatus.READY
    assert result.is_transferable is True
    # The second source shares a different field with the target.
    second = registry().align(source_id="source_b", target_id="target_facility")
    assert second.shared_features == ("gpu_util",)
    assert "cpu_util" in second.target_only_features


def test_different_units_are_not_merged_even_with_the_same_name() -> None:
    fields = FieldRegistry()
    source = unlabelled_spec("source_unit", {"cpu_util": feature("cpu_util", "ratio")})
    target = target_spec(
        {
            "cpu_util": feature("cpu_util", "percent"),
            "cooling_kw": feature("cooling_kw", "kW"),
        }
    )
    fields.register_many([source, target])
    result = fields.align(source_id="source_unit", target_id="target_facility")
    assert result.shared_features == ()
    assert "cpu_util" in result.incomparable_features
    assert result.status is AlignmentStatus.NO_SHARED_FEATURES
    assert result.is_transferable is False
    assert result.blocked_reasons


def test_different_statistics_are_not_merged() -> None:
    fields = FieldRegistry()
    source = unlabelled_spec(
        "source_stat",
        {"task_count": feature("task_count", "count")},
    )
    target = target_spec(
        {
            "task_count": feature(
                "task_count",
                "count",
                statistic=IntervalStatistic.LIFECYCLE_SUMMARY,
            ),
            "cooling_kw": feature("cooling_kw", "kW"),
        }
    )
    fields.register_many([source, target])
    result = fields.align(source_id="source_stat", target_id="target_facility")
    assert "task_count" in result.incomparable_features
    assert result.shared_features == ()


def test_no_shared_feature_returns_explicit_pending_state() -> None:
    fields = FieldRegistry()
    source = unlabelled_spec("source_fan", {"fan_speed": feature("fan_speed", "rpm")})
    target = target_spec({"cooling_kw": feature("cooling_kw", "kW")})
    fields.register_many([source, target])
    result = fields.align(source_id="source_fan", target_id="target_facility")
    assert result.status is AlignmentStatus.NO_SHARED_FEATURES
    assert result.is_transferable is False
    assert result.shared_features == ()
    # No fake shared representation is invented to fill the gap.
    assert any("no comparable shared feature" in reason for reason in result.blocked_reasons)


def test_duplicate_registration_with_different_content_fails() -> None:
    fields = registry()
    with pytest.raises(ContractError, match="already registered"):
        fields.register(unlabelled_spec("source_a", {"other": feature("other")}))


# ---------------------------------------------------------------------------
# T07 - target-only fields and unresolved global fusion weight
# ---------------------------------------------------------------------------


def test_target_only_fields_are_preserved_with_local_weight() -> None:
    result = registry().align(source_id="source_a", target_id="target_facility")
    target_only_routes = {
        route.canonical_name: route
        for route in result.routes
        if route.relation is FeatureRelation.TARGET_ONLY
    }
    assert set(target_only_routes) == set(result.target_only_features)
    for route in target_only_routes.values():
        assert route.target_only_local_weight == TARGET_ONLY_LOCAL_WEIGHT == 1.0
        assert "target_facility" in route.domains
        assert route.raw_names  # source description preserved

    config = registry().target_only_config(
        source_id="source_a", target_id="target_facility"
    )
    assert config["target_only_local_weight"] == 1.0
    assert config["global_fusion_weight"] is None
    assert config["status"] == "local_routing_only"
    assert set(config["target_only_features"]) == set(result.target_only_features)


def test_global_fusion_weight_is_never_assigned() -> None:
    result = registry().align(source_id="source_a", target_id="target_facility")
    assert result.global_fusion_weight is UNRESOLVED_GLOBAL_FUSION_WEIGHT
    assert result.global_fusion_weight is None
    payload = result.to_dict()
    assert payload["global_fusion_weight"] is None
    assert payload["target_only_local_weight"] == 1.0
    # Nothing in the registry exposes a resolved global weight such as 1.0.
    assert "global_weight" not in payload


def test_alignment_rejects_unlabelled_target_domain() -> None:
    fields = registry()
    with pytest.raises(ContractError, match="unlabelled"):
        fields.align(source_id="source_a", target_id="source_a")


# ---------------------------------------------------------------------------
# source registry (configs/datasets/registry.yaml) honesty checks
# ---------------------------------------------------------------------------


def test_source_registry_blocks_formal_facility_kw_for_candidates() -> None:
    registry = load_source_registry("configs/datasets/registry.yaml")
    sources = registry["sources"]
    assert registry["global_fusion_weight"] is None
    assert registry["target_only_local_weight"] == 1.0

    google = sources["google_power2019"]
    assert google["label"]["kind"] == "power_utilization"
    assert google["can_produce_facility_kw"] is False
    assert google["facility_kw_eligible"] is False

    alibaba = sources["alibaba_gpu_v2020"]
    assert alibaba["label"] is None
    assert alibaba["can_produce_facility_kw"] is False

    m100 = sources["m100_exadata"]
    assert m100["label"]["scope"] == "node"
    assert m100["can_produce_facility_kw"] is False

    esif = sources["nlr_esif_components"]
    assert esif["label"]["label_origin"] == "component_sum"
    assert esif["can_produce_facility_kw"] is False
    assert "audit_status" in esif["blocked_reason"]

    synthetic = sources["synthetic_facility"]
    assert synthetic["label"]["label_origin"] == "synthetic_simulated"
    assert synthetic["can_produce_facility_kw"] is False
    assert "synthetic" in synthetic["blocked_reason"]


def test_source_registry_roles_are_grouped() -> None:
    registry = load_source_registry("configs/datasets/registry.yaml")
    roles = describe_roles(registry)
    assert "synthetic_facility" in roles["demo"]
    assert "alibaba_gpu_v2020" in roles["unlabeled_pretrain"]
    assert "google_power2019" in roles["candidate"]


def test_registry_rejects_a_false_facility_kw_claim() -> None:
    payload = {
        "sources": {
            "google_power2019": {
                "candidate_role": "candidate",
                "audit_status": "unverified",
                "label": {
                    "target_kind": "power_utilization",
                    "target_scope": "facility",
                    "target_unit": "ratio",
                    "label_origin": "power_utilization",
                    "measurement_boundary": "util panel",
                    "sampling_semantics": "unverified",
                    "synthetic": False,
                },
                "can_produce_facility_kw": True,
            }
        }
    }
    with pytest.raises(ContractError, match="can_produce_facility_kw"):
        validate_source_registry(payload)


def test_registry_rejects_unlabelled_source_with_a_label() -> None:
    payload = {
        "sources": {
            "alibaba": {
                "candidate_role": "unlabeled_pretrain",
                "audit_status": "partial",
                "label": {
                    "target_kind": "active_power",
                    "target_scope": "facility",
                    "target_unit": "kW",
                    "label_origin": "measured",
                    "measurement_boundary": "x",
                    "sampling_semantics": "y",
                    "synthetic": False,
                },
                "can_produce_facility_kw": False,
            }
        }
    }
    with pytest.raises(ContractError, match="must not claim kW"):
        validate_source_registry(payload)
