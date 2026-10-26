"""Protocol configuration semantics (T02-config, T09, T10, T11, T15).

These tests prove that the time engine is *configuration driven*: horizons, history length,
origins and training cutoffs come from YAML, nothing dataset-specific is hard-coded, and the
optional historical presets only apply when a run explicitly selects them.
"""

from __future__ import annotations

import pytest

from src.data.splitting import PROTOCOL_PHASES, Protocol, load_protocol, training_window_audit
from src.utils import load_config

DEFAULT = "configs/protocols/default.yaml"
FIXED_24H = "configs/protocols/fixed_24h.yaml"
FIXED_7D = "configs/protocols/fixed_7d.yaml"


def test_default_protocol_is_generic_and_parses() -> None:
    cfg = load_config(DEFAULT)
    protocol = Protocol.from_config(cfg)

    assert protocol.protocol_id == "configurable_v1"
    assert protocol.frequency == "15min"
    assert protocol.points_per_day == 96
    assert protocol.context_points == 1344
    assert dict(protocol.horizons) == {"4h": 16, "8h": 32, "24h": 96, "3d": 288, "7d": 672}
    assert protocol.trace_anchor_policy == "dataset_explicit"
    assert protocol.origin_mode == "explicit"
    assert protocol.availability_policy == "strict"
    assert protocol.protocol_type == "configurable"
    assert protocol.horizon_points(None) == 96  # configured default_horizon, not a code constant
    assert protocol.overlap_warning is False

    # Every declared horizon maps to its exact point count, by name and by count.
    for name, points in (("4h", 16), ("8h", 32), ("24h", 96), ("3d", 288), ("7d", 672)):
        assert protocol.horizon_points(name) == points
        assert protocol.horizon_name(str(points)) == name
        assert protocol.horizon_points(str(points)) == points

    # Nothing dataset-specific is baked into the generic protocol.
    assert protocol.development_origins == ()
    assert protocol.test_origins == ()
    assert protocol.smoke_origins == ()
    assert dict(protocol.train_cutoffs) == {}
    assert protocol.raw["development_origins"] == []
    assert protocol.raw["test_origins"] == []
    assert protocol.raw["train_cutoffs"] == {}


def test_protocol_phases_are_the_four_lifecycle_phases() -> None:
    assert PROTOCOL_PHASES == ("development", "finalize", "final-test", "smoke")
    protocol = Protocol.from_config(load_config(DEFAULT), origins=[3, 5])
    assert protocol.origins_for("development") == (3, 5)
    assert protocol.origins_for("finalize") == ()
    assert protocol.origins_for("final-test") == (3, 5)
    assert protocol.origins_for("smoke") == (3, 5)


def test_context_points_override_changes_history_without_code_change() -> None:
    cfg = load_config(DEFAULT)
    base = Protocol.from_config(cfg)
    assert base.context_points == 1344
    for points in (96, 672, 1344):
        protocol = Protocol.from_config(cfg, context_points=points)
        assert protocol.context_points == points
        assert protocol.to_dict()["context_points"] == points
    # The override is per-protocol-instance and never mutates the shared configuration.
    assert base.context_points == 1344
    assert Protocol.from_config(cfg).context_points == 1344


def test_horizon_override_selects_declared_horizon_and_default() -> None:
    cfg = load_config(DEFAULT)
    for name, points in (("4h", 16), ("8h", 32), ("24h", 96), ("3d", 288), ("7d", 672)):
        protocol = Protocol.from_config(cfg, horizon=name)
        assert protocol.default_horizon == name
        assert protocol.horizon_points(None) == points
        # The full mapping stays available for manifests and reporting.
        assert dict(protocol.horizons) == {
            "4h": 16,
            "8h": 32,
            "24h": 96,
            "3d": 288,
            "7d": 672,
        }


def test_fixed_24h_preset_is_an_optional_instance() -> None:
    protocol = load_protocol(FIXED_24H)
    assert protocol.protocol_type == "fixed_24h"
    assert protocol.context_points == 1344
    assert protocol.horizon_points("24h") == 96
    assert protocol.horizon_points(None) == 96
    assert protocol.origins_for("development") == (14, 16, 18, 20)
    assert protocol.origins_for("final-test") == tuple(range(22, 31))
    assert protocol.origins_for("finalize") == ()
    assert protocol.overlap_warning is False
    # LightGBM-style cutoffs are recorded configuration data, not engine behaviour.
    assert dict(protocol.train_cutoffs) == {"development": 14, "final": 22}
    assert protocol.to_dict()["train_cutoffs"] == {"development": 14, "final": 22}


def test_fixed_7d_preset_is_an_optional_instance_and_flags_overlap() -> None:
    protocol = load_protocol(FIXED_7D)
    assert protocol.protocol_type == "fixed_7d"
    assert protocol.context_points == 1344
    assert protocol.horizon_points("7d") == 672
    assert protocol.horizon_points(None) == 672
    assert protocol.origins_for("development") == (14, 15)
    assert protocol.origins_for("final-test") == (22, 23, 24)
    assert protocol.overlap_warning is True
    assert dict(protocol.train_cutoffs) == {"development": 14, "final": 22}


def test_training_cutoffs_are_configuration_data_not_rewritten_by_code() -> None:
    cfg = load_config(FIXED_24H)
    cfg["train_cutoffs"] = {"development": 10, "final": 20}
    protocol = Protocol.from_config(cfg)

    assert dict(protocol.train_cutoffs) == {"development": 10, "final": 20}
    assert protocol.to_dict()["train_cutoffs"] == {"development": 10, "final": 20}

    # Unrelated protocol operations never rewrite the configured cutoffs.
    protocol.horizon_points("24h")
    protocol.origins_for("final-test")
    protocol.to_dict()
    assert dict(protocol.train_cutoffs) == {"development": 10, "final": 20}

    # The original configuration is deep-copied: later caller mutations cannot change it.
    cfg["train_cutoffs"]["development"] = 1
    assert protocol.train_cutoffs["development"] == 10

    # The generic protocol declares no cutoffs at all.
    generic = Protocol.from_config(load_config(DEFAULT))
    assert dict(generic.train_cutoffs) == {}


def test_cutoffs_are_honoured_as_parameters_not_assumed_values() -> None:
    # A cutoff is an input to feasibility checks; moving it changes the answer, so the engine
    # cannot be relying on a hidden fixed cutoff.
    early = training_window_audit(train_cutoff_day=14, context_points=1344, horizon_points=672)
    later = training_window_audit(train_cutoff_day=21, context_points=1344, horizon_points=672)
    assert early["feasible_windows"] == 0
    assert later["feasible_windows"] == 1
    assert early["train_cutoff_day"] == 14
    assert later["train_cutoff_day"] == 21


def test_unknown_horizon_and_phase_fail_loudly() -> None:
    protocol = load_protocol(FIXED_24H)
    with pytest.raises(ValueError):
        protocol.horizon_name("5h")
    with pytest.raises(ValueError):
        protocol.horizon_points("5h")
    with pytest.raises(ValueError):
        protocol.horizon_name("999")
    with pytest.raises(ValueError):
        protocol.origins_for("train")
    # A horizon that the selected preset does not declare is rejected instead of invented.
    with pytest.raises(ValueError):
        Protocol.from_config(load_config(FIXED_24H), horizon="7d")
    # Slots that do not tile a 24h trace day are rejected rather than silently rescaled.
    with pytest.raises(ValueError):
        Protocol.from_config(load_config(DEFAULT), context_points=0)
