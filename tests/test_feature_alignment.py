"""Stage-1 field-alignment acceptance tests (M100 vs NLR target, M100 vs UCC).

The alignment must be semantic (unit + statistic + measurement point), not
name-based.  The honest Stage-1 outcome is that the NLR target carries no field
comparable to M100 node telemetry, so no shared representation may be fabricated.
Target-only fields keep their Stage-4 reservation (``local_weight: 1.0``) while the
global fusion weight stays unresolved; same-name/different-unit fields are classified
``incomparable`` and never merged.  The tracked registry YAML is validated against the
live config + ``FieldRegistry.align`` result.
"""

from __future__ import annotations

import pytest

from src.data import source_features as sf
from src.data.contracts import (
    DatasetRole,
    LabelOrigin,
    TargetKind,
    TargetScope,
)
from src.data.labels import LabelType, classify_target
from src.data.registry import (
    AlignmentStatus,
    FeatureRelation,
    TARGET_ONLY_LOCAL_WEIGHT,
    UNRESOLVED_GLOBAL_FUSION_WEIGHT,
)

ALLOWED_FIELD_ROLES = {"source_shared", "source_only", "target_only", "incomparable"}


@pytest.fixture(scope="module")
def config() -> dict:
    return sf.load_source_config()


@pytest.fixture(scope="module")
def alignments(config: dict):
    return sf.build_alignment_results(config)


@pytest.fixture(scope="module")
def registry_payload(config: dict) -> dict:
    return sf.build_feature_registry_payload(config)


# ---------------------------------------------------------------------------
# M100 vs NLR target domain
# ---------------------------------------------------------------------------


def test_m100_vs_target_has_no_fabricated_shared_fields(alignments) -> None:
    result = alignments["m100_vs_nlr"]
    assert result.shared_features == ()
    assert result.incomparable_features == ()
    assert result.status is AlignmentStatus.NO_SHARED_FEATURES
    assert result.is_transferable is False
    assert len(result.source_only_features) == 12
    assert set(result.target_only_features) == {
        "site_outdoor_temperature_c",
        "site_outdoor_humidity_pct",
    }
    assert any("no comparable shared feature" in reason for reason in result.blocked_reasons)
    # M100 node power/temperature/fan fields are source-only; the target's facility
    # label components and site weather are target-only.  Nothing is invented to fill F.
    route_by_name = {route.canonical_name: route for route in result.routes}
    for name in result.source_only_features:
        assert route_by_name[name].relation is FeatureRelation.SOURCE_ONLY
    for name in result.target_only_features:
        route = route_by_name[name]
        assert route.relation is FeatureRelation.TARGET_ONLY
        assert route.target_only_local_weight == TARGET_ONLY_LOCAL_WEIGHT == 1.0
        assert "nlr_esif" in route.domains
        assert route.raw_names and "nlr_esif" in route.raw_names


def test_target_only_reservation_and_unresolved_global_weight(alignments, registry_payload) -> None:
    result = alignments["m100_vs_nlr"]
    assert result.target_only_local_weight == 1.0
    assert result.global_fusion_weight is UNRESOLVED_GLOBAL_FUSION_WEIGHT
    assert result.global_fusion_weight is None

    fusion = registry_payload["fusion"]
    assert fusion["target_only_local_weight"] == 1.0
    assert fusion["global_fusion_weight"] is None
    assert fusion["execution_status"] == "not_executed_in_stage_1_or_2"

    # Walk the whole document: no `global_fusion_weight` may ever be resolved to 1.
    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "global_fusion_weight":
                    assert value is None, (
                        f"global_fusion_weight must stay None, found {value!r}"
                    )
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(registry_payload)

    # Only target-only fields carry a future_mapping reservation; source-only and
    # incomparable fields carry none at all.
    for source in registry_payload["sources"].values():
        for field in source["fields"]:
            if field["feature_role"] == "target_only":
                assert field["future_mapping"]["local_weight"] == 1.0
                assert field["future_mapping"]["global_fusion_weight"] is None
                assert field["future_mapping"]["status"] == "semantic_reservation_only"
            else:
                assert field["future_mapping"] is None
            if field["feature_role"] == "source_only":
                assert field["role_reason"].strip()


# ---------------------------------------------------------------------------
# M100 vs UCC
# ---------------------------------------------------------------------------


def test_incomparable_same_name_fields_are_not_merged(alignments, registry_payload) -> None:
    # FieldRegistry call direction: source=UCC, target=M100 comparison-only node label.
    result = alignments["m100_vs_ucc"]
    assert result.shared_features == ()
    assert result.status is AlignmentStatus.NO_SHARED_FEATURES
    assert "node_total_power_w" in result.incomparable_features
    assert "node_total_power_w" not in result.source_only_features
    assert "node_total_power_w" not in result.target_only_features
    blocking = [reason for reason in result.blocked_reasons if "node_total_power_w" in reason]
    assert blocking, "the incomparable field must carry an explicit blocking reason"
    joined = " ".join(blocking)
    assert "unit" in joined
    assert "statistic" in joined
    # The UCC-only and M100-only field counts must add up with the incomparable field.
    assert len(result.source_only_features) == 15
    assert len(result.target_only_features) == 11
    assert len(result.incomparable_features) == 1

    ucc_fields = {
        field["canonical_name"]: field
        for field in registry_payload["sources"]["ucc2023"]["fields"]
    }
    field = ucc_fields["node_total_power_w"]
    assert field["feature_role"] == "incomparable"
    assert field["feature_role"] != "source_shared"
    assert field["role_reason"].strip()
    assert "unit" in field["role_reason"] or "Unit" in field["role_reason"]
    # UCC is kept at its native 10-minute frequency and is not approved for training.
    assert registry_payload["sources"]["ucc2023"]["native_frequency"] == "10min"
    assert registry_payload["sources"]["ucc2023"]["aggregation_status"] == "native_preserved_not_15min"
    assert registry_payload["sources"]["ucc2023"]["approved_for_training"] is False


def test_alignment_status_and_reasons_are_explicit(alignments, registry_payload) -> None:
    registry_keys = {
        "m100_vs_nlr": "m100_exadata__vs__nlr_esif",
        "m100_vs_ucc": "m100_exadata__vs__ucc2023",
    }
    for key, result in alignments.items():
        assert result.status is AlignmentStatus.NO_SHARED_FEATURES
        assert result.blocked_reasons
        section = registry_payload["alignment"][registry_keys[key]]
        assert section["status"] == "no_shared_features"
        assert section["shared_features"] == []
        assert section["global_fusion_weight"] is None
        # Every route, including only-fields and incomparable fields, has a reason.
        for name in (
            list(section["source_only_features"])
            + list(section["target_only_features"])
            + list(section["incomparable_features"])
        ):
            assert section["field_reasons"][name], f"{key}:{name} has no explicit reason"


# ---------------------------------------------------------------------------
# target-domain reference facts
# ---------------------------------------------------------------------------


def test_target_reference_is_honest_and_label_components_are_not_features(config: dict) -> None:
    spec = sf.build_target_dataset_spec(config)
    assert spec.role is DatasetRole.TARGET_SUPERVISED
    assert spec.feature_names == ("site_outdoor_temperature_c", "site_outdoor_humidity_pct")
    assert spec.target is not None
    target = spec.target
    assert target.kind is TargetKind.ACTIVE_POWER
    assert target.scope is TargetScope.FACILITY
    assert target.label_origin is LabelOrigin.COMPONENT_SUM
    assert target.unit == "kW"
    assert target.synthetic is False
    assert classify_target(target) is LabelType.FACILITY_DERIVED_KW
    # The five power components compose the label and are NOT candidate model inputs.
    for component in ("it_power_kw", "cooling_kw", "hvac_kw", "plug_and_light_kw", "pump_kw"):
        assert component not in spec.features
    assert spec.provenance["label_composition_input_eligible"] is False
    assert spec.provenance["owner"] == "target-power"


def test_source_spec_is_unlabelled_and_node_level(config: dict) -> None:
    spec = sf.build_source_dataset_spec(config)
    assert spec.role is DatasetRole.UNLABELED_PRETRAIN
    assert spec.target is None
    assert spec.feature_names == tuple(
        [
            "node_total_power_w",
            "cpu0_power_w",
            "cpu1_power_w",
            "cpu0_memory_power_w",
            "cpu0_io_power_w",
            "psu0_input_power_w",
            "psu1_input_power_w",
            "disk_power_w",
            "node_ambient_temp_c",
            "gpu0_core_temp_c",
            "gpu1_core_temp_c",
            "fan0_0_speed_rpm",
        ]
    )
    assert spec.provenance["node_level_not_facility"] is True
    assert spec.provenance["approved_for_training"] is False


# ---------------------------------------------------------------------------
# tracked registry YAML
# ---------------------------------------------------------------------------


def test_registry_yaml_loads_and_matches_live_alignment(config: dict, registry_payload: dict) -> None:
    loaded = sf.load_feature_registry(config["paths"]["feature_registry"])
    assert loaded == registry_payload, (
        "configs/datasets/feature_registry.yaml is stale; regenerate with "
        "`python -m scripts.prepare_source_features`"
    )


def test_registry_yaml_is_internally_consistent(registry_payload: dict) -> None:
    assert registry_payload["fusion"]["global_fusion_weight"] is None
    assert registry_payload["fusion"]["target_only_local_weight"] == 1.0
    assert registry_payload["sources"]["m100_exadata"]["approved_for_training"] is False
    assert registry_payload["sources"]["ucc2023"]["approved_for_training"] is False
    alibaba = registry_payload["sources"]["alibaba_gpu_v2020"]
    assert alibaba["status"] == "not_ingested"
    assert alibaba["fields"] == []
    assert alibaba["reasons"]

    for source_id, source in registry_payload["sources"].items():
        for field in source["fields"]:
            assert field["feature_role"] in ALLOWED_FIELD_ROLES, (source_id, field["canonical_name"])
            assert field["role_reason"].strip(), (source_id, field["canonical_name"])
            assert field["unit"].strip()
            assert field["semantic_type"] in sf.SEMANTIC_TYPES
            assert field["aggregation_rule"] in sf.AGGREGATION_RULES
            assert field["availability"] in sf.AVAILABILITIES
            assert field["missing_policy"] == "mask"
            if field["feature_role"] in {"source_only", "target_only", "incomparable"}:
                assert field["role_context"] in {
                    "m100_exadata__vs__nlr_esif",
                    "m100_exadata__vs__ucc2023",
                }

    label = registry_payload["sources"]["nlr_esif"]["label"]
    assert label["label_type"] == "facility_derived_kw"
    assert label["label_origin"] == "component_sum"
    assert label["input_eligible"] is False
    for component in label["composition_fields"]:
        assert component not in {
            field["canonical_name"] for field in registry_payload["sources"]["nlr_esif"]["fields"]
        }

    counts = registry_payload["alignment_status_counts"]
    assert counts["m100_vs_nlr"]["shared"] == 0
    assert counts["m100_vs_nlr"]["target_only"] == 2
    assert counts["m100_vs_ucc"]["incomparable"] == 1
    assert all(item["status"] == "no_shared_features" for item in counts.values())
