from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.domains import DomainMetadata
from src.data.google_panel import DomainPanel, domain_windows
from src.data.transfer30 import aggregate_published_power
from src.evaluation.backtest import score_predictions
from src.evaluation.benefit_metrics import risk_summary
from src.models.ml.benefit import BenefitTransferPolicy, JointBenefitPredictor
from src.training.benefit_learning import reduce_priors
from src.utils import load_config


def test_30min_native_means_preserve_missing_and_clock_identity():
    for native, expected in [(15, [.15, .35]), (10, [.2, .5])]:
        count = 30//native
        index = pd.date_range("2019-04-09", periods=48*count, freq=f"{native}min")
        raw = pd.DataFrame({"Time": index, "Power": np.arange(1, len(index)+1)*100.})
        frame, counts = aggregate_published_power(raw, timestamp="Time", power="Power", start=index[0], days=1, native_minutes=native)
        np.testing.assert_allclose(frame.node_power_kw.iloc[:2], expected)
        assert frame.index.tz is None and not frame.source_clock_timestamp.iloc[0].endswith("+00:00")
        assert (counts == count).all()
        raw.loc[1, "Power"] = np.nan
        frame, counts = aggregate_published_power(raw, timestamp="Time", power="Power", start=index[0], days=1, native_minutes=native)
        assert pd.isna(frame.node_power_kw.iloc[0]) and counts[0] == count-1
        with pytest.raises(ValueError, match="UTC"):
            aggregate_published_power(raw, timestamp="Time", power="Power", start="2019-04-09T00:00Z", days=1, native_minutes=native)


def test_zero_transfer_is_exact_and_no_veto_is_a_single_intervention():
    gain = np.array([[-1., -2.], [2., 1.], [1., 3.]])
    risk = np.array([[.1, .1], [.1, .2], [.9, .9]])
    candidate = dict(minimum_gain_pp=0., maximum_negative_risk=.5, strength=.5, top_k=1)
    weights, gate, _ = BenefitTransferPolicy.route(gain, risk, candidate)
    assert np.array_equal(gate, [0., .5, 0.])
    assert np.array_equal(weights[[0, 2]], np.zeros((2, 2)))
    forced, forced_gate, _ = BenefitTransferPolicy.route(gain, risk, candidate, veto=False)
    assert np.array_equal(weights[1], forced[1]) and np.all(forced_gate == .5)
    assert forced[0, 0] == 1 and forced[2, 1] == 1


def test_risk_diagnostics_distinguish_detection_from_rejecting_everything():
    frame = pd.DataFrame(dict(model=["sapf_gain"]*4, split=["test"]*4, horizon=["24h"]*4,
        actual_gain_pp=[-1., 1., -1., 1.], predicted_gain_pp=[-1., 2., -.5, -.1], negative_risk=[.9, .8, .2, .1],
        weight=[0., 1., 0., 0.], transfer_gate=[.5, .5, 0., 0.], target_domain=["target"]*4,
        origin=["o0", "o0", "o1", "o1"]))
    result = risk_summary(frame).iloc[0]
    assert result.risk_AUROC == .75 and result.risk_Brier == pytest.approx(.325)
    assert result.risk_precision == .5 and result.risk_recall == .5
    assert result.zero_transfer_fraction == .5 and result.positive_gain_precision == 1.
    frame.negative_risk = 1.
    assert risk_summary(frame).iloc[0].risk_AUROC == .5


def test_historical_priors_leave_pseudo_target_out():
    rows = pd.DataFrame([dict(expert_domain=i, target_domain=j, expert_MAPE=100. if j == "b" else 1., local_MAPE=2.)
                         for i in ["a", "b", "c"] for j in ["a", "b", "c"] if i != j])
    prior = reduce_priors(rows, ["a", "b", "c"], excluded_target="b")
    assert prior["a"][0] == 1 and prior["c"][0] == 1
    rows.loc[rows.target_domain == "b", "expert_MAPE"] = 999999.
    assert prior == reduce_priors(rows, ["a", "b", "c"], excluded_target="b")


@pytest.mark.parametrize("bad", ["external_target", "self_expert", "future_prior", "future_label"])
def test_benefit_fit_rejects_identity_or_time_leakage(bad):
    row = dict(expert_domain="a", pseudo_target="b", origin_day=24, prior_label_end_day=24, label_end_day=25)
    if bad == "external_target":
        row["pseudo_target"] = "held_out_test"
    elif bad == "self_expert":
        row["pseudo_target"] = "a"
    elif bad == "future_prior":
        row["prior_label_end_day"] = 25
    else:
        row["label_end_day"] = 51
    with pytest.raises(ValueError):
        JointBenefitPredictor(dict(n_estimators=1)).fit(pd.DataFrame([row]), ["a", "b", "c"])


def synthetic_panel():
    index = pd.date_range("2000-01-03", periods=72*48, freq="30min", tz="UTC", name="timestamp")
    t = np.arange(len(index))
    series, metadata, audit, roles = {}, {}, {}, {}
    for group, clock in [("m100", pd.Timestamp("2022-01-01T00:00Z")), ("ucc", pd.Timestamp("2019-04-09"))]:
        roles[group] = dict(source=[], validation=[], test=[])
        for n in range(7):
            domain = f"{group}_fixture_{n}"
            role = "source" if n < 3 else "validation" if n < 5 else "test"
            roles[group][role].append(domain)
            values = .3+.03*n+.02*np.sin(2*np.pi*t/48)+.005*np.cos(2*np.pi*t/336+n/10)
            source_time = pd.date_range(clock, periods=len(index), freq="30min")
            series[domain] = pd.DataFrame(dict(node_power_kw=values, source_clock_timestamp=[s.isoformat() for s in source_time]), index=index)
            metadata[domain] = DomainMetadata(dataset_id=group+"_fixture", site_id=group, domain_id=domain, target_field="node_power_kw",
                target_kind="active_power", target_scope="node", target_unit="kW", label_origin="measured",
                timezone="UTC" if group == "m100" else "unverified_source_local_clock", timezone_verified=group == "m100",
                raw_frequency="15min" if group == "m100" else "10min", evaluation_frequency="30min", sampling_semantics="interval_mean",
                interval_definition="Published-slot mean, synthetic fixture only.", coverage="Synthetic complete grid.", source_version="fixture",
                usage_role="final_holdout" if role == "test" else "development")
            audit[domain] = dict(source_clock_start=clock.isoformat(), source_clock_timezone_verified=group == "m100",
                                 history_availability_upper_bound_minutes=30)
    roles.update(retired_m100_previous_test=["retired_node"], quality_excluded=[])
    return DomainPanel(series, metadata, {}, {}, index[0], audit), roles


def test_30min_history_daily_and_nonoverlap_are_time_consistent():
    panel, roles = synthetic_panel()
    batch = domain_windows(panel, roles["m100"]["test"], list(range(60, 70)), 672, 144)
    assert batch.inputs.points_per_day == 48 and batch.inputs.history.shape[1] == 672
    assert batch.inputs.take(slice(0, 2)).points_per_day == 48
    result = score_predictions(batch, batch.targets+.01)
    assert result.daily_metrics["n"] == len(batch)*3
    assert len(result.per_day) == 3
    assert result.nonoverlap_metrics["n"] == 2*4*144


def test_global_suite_freezes_before_every_direction_and_keeps_target_clocks(tmp_path, monkeypatch):
    import src.training.transfer_suite as suite
    from src.evaluation.transfer_audit import audit_transfer_suite
    panel, roles = synthetic_panel()
    config = load_config("configs/sapf_transfer_suite.yaml")
    config["domain_roles"] = roles
    config["results_dir"] = str(tmp_path/"runs")
    model_config = load_config(config["model_config"])
    model_config["benefit"].update(n_estimators=8, max_depth=3, min_samples_leaf=2)
    model_config["sapf"].update(top_k=[1], temperatures=[1.], feature_groups=["shape"], transfer_strengths=[.5], reliability_powers=[1.])
    model_config["policy_grid"].update(minimum_gain_pp=[0.], maximum_negative_risk=[.5], strengths=[.5], top_k=[1])
    path = tmp_path/"models.yaml"
    path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    config["model_config"] = str(path)
    monkeypatch.setattr(suite, "load_transfer_domains", lambda *args: panel)
    original = suite._evaluate
    def evaluate(folder, panel, roles, config, task, split, *args, **kwargs):
        if split == "test":
            root = folder.parents[1]
            frozen = json.loads((root/"frozen-before-test.json").read_text(encoding="utf-8"))
            assert len(frozen["studies"]) == 3
        return original(folder, panel, roles, config, task, split, *args, **kwargs)
    monkeypatch.setattr(suite, "_evaluate", evaluate)
    folder = suite.run_transfer_suite(SimpleNamespace(demo=False), config)
    audit = audit_transfer_suite(folder)
    assert audit["passed"] and audit["all_models_frozen_before_any_test"]
    assert audit["source_only_leave_domain_meta_labels"] and audit["m100_source_models_shared_without_retuning"]
    table = pd.read_parquet(next((folder/"studies/m100_to_ucc").glob("ucc_fixture*/3d/sapf_gain/test_predictions.parquet")))
    assert table.source_clock_timestamp.str.startswith("2019-").all()
    assert not table.source_clock_timestamp.str.endswith("+00:00").any()
    assert not table.source_clock_timezone_verified.any()
    for group in ["m100", "ucc"]:
        train = pd.read_csv(folder/"groups"/group/"benefit/gain-training.csv")
        assert not (train.expert_domain == train.pseudo_target).any()
        assert train.feature_horizon_days.nunique() == 2
