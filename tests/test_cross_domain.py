from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.cross_domain import fit_batch, training_origins, validate_protocol
from src.data.adapters import adapt_frame
from src.data.domains import DomainMetadata, aggregate_components, convert_target, validate_compatible, metadata_from_config
from src.data.google_panel import DomainPanel, domain_windows, load_google_domains
from src.data.pipeline import prepare_dataset
from src.evaluation.cross_domain import audit_cross_domain_run, summarize_transfer
from src.training.trainer import train_model
from src.utils import config_digest, load_config, preparation_config_digest


def metadata(domain="cella_pdu1"):
    spec = deepcopy(load_config("configs/datasets/google_cross_domain.yaml")["domain_metadata"])
    spec.update(domain_id=domain, site_id="cella", power_domain_id=domain.split("_")[-1])
    return DomainMetadata(**spec)


@pytest.fixture
def panel():
    index = pd.date_range("2019-05-01 07:00", periods=31*96, freq="15min", tz="UTC", name="timestamp")
    points = np.arange(len(index))
    series = {f"cella_pdu{i}": pd.DataFrame({"measured_power_util": .4+i*.05+.02*np.sin(2*np.pi*points/96)
                                           + .005*np.cos(2*np.pi*points/672+i)}, index=index)
              for i in range(1, 5)}
    return DomainPanel(series, {d: metadata(d) for d in series},
                       {d: pd.Series(3, index=index) for d in series}, {}, index[0], {})


def pilot_config():
    config = deepcopy(load_config("configs/cross_domain_google.yaml"))
    config.update(target_domains=["cella_pdu1"], single_source_domains=["cella_pdu2"], save_plots=False)
    return config


def test_physical_units_scopes_origins_and_native_sampling():
    source = replace(metadata(), target_kind="active_power", target_scope="node", target_unit="W")
    values = np.array([1000., np.nan])
    converted = convert_target(values, source, target_kind="active_power", target_scope="node", target_unit="kW")
    assert converted[0] == 1 and values[0] == 1000 and np.isnan(converted[1])
    energy = replace(source, target_kind="energy", target_unit="kWh", raw_frequency="15min", sampling_semantics="interval_energy")
    assert convert_target([2.], energy, target_kind="active_power", target_scope="node", target_unit="kW", interval_hours=.25)[0] == 8
    with pytest.raises(ValueError, match="target_scope"):
        convert_target([1], source, target_kind="active_power", target_scope="facility", target_unit="kW")
    with pytest.raises(ValueError, match="Unsupported physical conversion"):
        convert_target([.5], metadata(), target_kind="active_power", target_scope="power_domain", target_unit="kW")
    for frequency in ("10min", "1h"):
        with pytest.raises(ValueError, match="Native sampling"):
            replace(metadata(), raw_frequency=frequency)
    with pytest.raises(ValueError, match="Incompatible physical"):
        validate_compatible([metadata(), replace(metadata("b"), label_origin="assumed_scenario")])
    with pytest.raises(ValueError, match="future_known"):
        replace(metadata(), feature_availability={"weather_observation": "future_known"})


def test_aggregation_requires_disjoint_complete_same_intervals():
    index = pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
    values = {"it": pd.Series([1., np.nan], index=index), "cooling": pd.Series([2., 3.], index=index)}
    result = aggregate_components(values, expected_components=list(values), complete_coverage=True, nonoverlapping=True)
    assert result.iloc[0] == 3 and np.isnan(result.iloc[1])
    with pytest.raises(ValueError, match="nonoverlapping"):
        aggregate_components(values, expected_components=list(values), complete_coverage=True, nonoverlapping=False)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        aggregate_components(values, expected_components=["it", "cooling", "pump"], complete_coverage=True, nonoverlapping=True)


def test_generic_adapter_refuses_cross_domain_duplicate_averaging():
    raw = pd.DataFrame({"time": ["2024-01-01", "2024-01-01"], "domain_id": ["rack_a", "rack_b"], "load": [1, 9]})
    with pytest.raises(ValueError, match="partition by physical domain"):
        adapt_frame(raw, {"timestamp": "time", "columns": {"load": "load"}}, "UTC")


def test_domain_grid_missingness_full_horizons_and_readonly(panel):
    batch = domain_windows(panel, ["cella_pdu1", "cella_pdu2"], [22], 1344, 672)
    assert batch.inputs.history.shape == (2, 1344, 5)
    assert batch.inputs.future_known.shape == (2, 672, 4)
    assert batch.targets.shape == (2, 672)
    assert batch.inputs.origins[0] == batch.inputs.origins[1]
    assert batch.inputs.domain_ids == ("cella_pdu1", "cella_pdu2")
    assert not np.array_equal(batch.targets[0], batch.targets[1])
    with pytest.raises(ValueError, match="read-only"):
        batch.inputs.history[0, 0, 0] = 0
    assert batch.inputs.take(np.array([1])).domain_ids == ("cella_pdu2",)
    before = domain_windows(panel, ["cella_pdu1"], [22], 1344, 96)
    panel.series["cella_pdu1"].iloc[22*96:, 0] = 9999
    after = domain_windows(panel, ["cella_pdu1"], [22], 1344, 96)
    np.testing.assert_array_equal(before.inputs.history, after.inputs.history)
    np.testing.assert_array_equal(before.inputs.future_known, after.inputs.future_known)
    panel.series["cella_pdu2"].iloc[22*96, 0] = np.nan
    skipped = domain_windows(panel, ["cella_pdu2", "cella_pdu3"], [22, 23], 1344, 96)
    assert len(skipped) == 2 and skipped.skipped_windows == 2
    panel.series["cella_pdu3"] = panel.series["cella_pdu3"].drop(panel.series["cella_pdu3"].index[100])
    with pytest.raises(ValueError, match="complete 15min grid"):
        domain_windows(panel, ["cella_pdu3"], [22], 1344, 96)


def test_lodo_training_scaler_excludes_target_and_respects_cutoffs(panel):
    config = pilot_config()
    plans = validate_protocol(config, panel, ["24h", "7d"])
    assert plans["7d"]["final_origins"] == [14, 15]
    assert plans["7d"]["development_origins"] == []
    sources = plans["7d"]["folds"]["cella_pdu1"]["source_domains"]
    batch = fit_batch(panel, sources, [14, 15], 1344, 672, forbidden_domain="cella_pdu1")
    assert set(batch.inputs.domain_ids) == set(sources)
    assert max(pd.Timestamp(t) for t in batch.inputs.target_times[:, -1]) < panel.anchor+pd.Timedelta(days=22)
    first = train_model("ridge", batch, {"alpha": 100}).model
    panel.series["cella_pdu1"].iloc[:, 0] = 1e9
    other = train_model("ridge", fit_batch(panel, sources, [14, 15], 1344, 672, forbidden_domain="cella_pdu1"), {"alpha": 100}).model
    np.testing.assert_array_equal(first.estimator[0].mean_, other.estimator[0].mean_)
    np.testing.assert_array_equal(first.estimator[1].coef_, other.estimator[1].coef_)
    with pytest.raises(ValueError, match="Held-out target"):
        fit_batch(panel, ["cella_pdu1"], [14], 1344, 672, forbidden_domain="cella_pdu1")
    with pytest.raises(ValueError, match="No training windows"):
        training_origins(1344, 672, 14)
    panel.audit["cella_pdu1"] = {"mapping_overlap_domains": ["cella_pdu2"]}
    with pytest.raises(ValueError, match="ambiguous machine mapping"):
        validate_protocol(config, panel, ["7d"])
    panel.audit.clear()
    config["temporal_protocol"]["7d"].update(development_cutoff_day=21, development_origins=[21])
    with pytest.raises(ValueError, match="end by final cutoff"):
        validate_protocol(config, panel, ["7d"])


def test_google_raw_time_anchor_coverage_and_mapping_are_preserved(tmp_path):
    config = deepcopy(load_config("configs/datasets/google_cross_domain.yaml"))
    config.update(raw_data=str(tmp_path), expected_domains=2, expected_days=1)
    mapping = pd.DataFrame({"machine_id": [1, 2], "cell": ["a", "b"], "pdu": ["pdu1", "pdu2"]})
    mapping.to_csv(tmp_path/"machine_to_pdu_mapping.csv.gz", index=False)
    for cell, pdu in [("a", "pdu1"), ("b", "pdu2")]:
        raw = pd.DataFrame({"time": 600_000_000+np.arange(288)*300_000_000, "cell": cell, "pdu": pdu,
                            "measured_power_util": .6, "bad_measurement_data": False})
        raw.loc[1, "bad_measurement_data"] = True
        raw.drop(index=range(12, 15)).to_csv(tmp_path/f"cell{cell}_{pdu}.csv.gz", index=False)
    loaded = load_google_domains(config)
    frame = loaded.series["cella_pdu1"]
    assert frame.index[0] == pd.Timestamp("2019-05-01T07:00Z") and len(frame) == 96
    assert frame.iloc[[0, 4], 0].isna().all()
    assert loaded.coverage["cella_pdu1"].iloc[0] == 2
    assert loaded.coverage["cella_pdu1"].iloc[4] == 0
    assert loaded.metadata["cella_pdu1"].timezone_verified is True
    assert loaded.audit["cella_pdu1"]["mapped_machine_count"] == 1
    mapping.loc[1] = [1, "a", "pdu2"]
    mapping.to_csv(tmp_path/"machine_to_pdu_mapping.csv.gz", index=False)
    ambiguous = load_google_domains(config)
    assert ambiguous.audit["cella_pdu1"]["mapping_overlap_domains"] == ["cella_pdu2"]
    assert ambiguous.audit["cella_pdu1"]["ambiguous_mapping_machine_count"] == 1


def test_nlr_metadata_is_annotation_only_and_new_preparation_saves_counts(tmp_path):
    config = load_config("configs/datasets/nlr.yaml")
    original = deepcopy({k: v for k, v in config.items() if k != "domain_metadata"})
    assert preparation_config_digest(config) == config_digest(original)
    assert config["domain_metadata"]["label_origin"] == "component_sum"
    assert config["domain_metadata"]["timezone_verified"] is False
    changed_target = {**config, "target": "it_power_kw"}
    with pytest.raises(ValueError, match="Target field changed"):
        metadata_from_config(changed_target)
    config["cleaning"]["ffill_limit"] += 1
    assert preparation_config_digest(config) != config_digest(original)
    raw_dir = tmp_path/"raw"
    raw_dir.mkdir()
    path = raw_dir/"load.csv"
    pd.DataFrame({"ts": ["2024-01-01T00:00Z", "2024-01-01T00:01Z"], "value": [1., 3.]}).to_csv(path, index=False)
    config = dict(name="small", timezone="UTC", target="load", target_unit="kW", frequency="15min",
                  processed_path=str(tmp_path/"processed"/"load.parquet"),
                  sources={"power": {"path": str(path), "timestamp": "ts", "columns": {"load": "value"}}},
                  domain_metadata=replace(metadata(), target_kind="active_power", target_scope="facility", target_unit="kW",
                                          timezone="UTC", timezone_verified=False, raw_frequency="1min").to_dict())
    frame, report = prepare_dataset(config)
    assert frame.load.iloc[0] == 2
    assert report["domain_metadata"]["timezone_verified"] is False
    coverage = pd.read_parquet(report["sample_coverage"]["path"])
    assert coverage.iloc[0, 0] == 2


def test_negative_transfer_uses_only_same_target_zero_shot_comparisons():
    rows = [dict(target_domain="target", horizon="24h", algorithm="ridge", split="test", evaluation="curve",
                 n=96, mape_n=96, model=name, source_strategy=strategy, MAPE=value)
            for name, strategy, value in [("single", "single_source", 2.), ("multi", "multi_source", 3.),
                                           ("reference", "target_supervised_reference", .1)]]
    comparisons, summary = summarize_transfer(pd.DataFrame(rows))
    assert len(comparisons) == 1 and comparisons.iloc[0].delta_MAPE_pp == 1
    assert comparisons.iloc[0].negative_transfer and len(summary) == 3


def test_full_cross_domain_lifecycle_uses_shared_scores_freeze_and_restore(panel, tmp_path, monkeypatch):
    import scripts.cross_domain_experiment as runner
    config = pilot_config()
    config["results_dir"] = str(tmp_path/"runs")
    monkeypatch.setattr(runner, "load_google_domains", lambda _: panel)
    monkeypatch.setattr(runner, "print_cross_domain_results", lambda *args, **kwargs: None)
    original_backtest = runner.backtest
    def check_freeze(model, batch, *args, **kwargs):
        if set(batch.inputs.domain_ids) == {"cella_pdu1"} and min(batch.inputs.origins) >= panel.anchor+pd.Timedelta(days=22):
            assert list((tmp_path/"runs").glob("*/frozen-before-test.json"))
        return original_backtest(model, batch, *args, **kwargs)
    monkeypatch.setattr(runner, "backtest", check_freeze)
    folder = runner.run_cross_domain(SimpleNamespace(), config)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    audit = audit_cross_domain_run(folder)
    assert audit["passed"] and audit["restore_checks"] == 10
    table = pd.read_csv(folder/"metrics.csv")
    assert set(table[table.split == "test"].horizon) == {"24h", "7d"}
    assert len(table[(table.split == "test") & (table.evaluation == "curve")]) == 10
    assert not len(table[(table.horizon == "7d") & (table.split != "test")])
    path = next(folder.glob("cella_pdu1/24h/*/test_predictions.parquet"))
    saved = pd.read_parquet(path)
    saved.loc[0, "actual"] += .1
    saved.to_parquet(path)
    with pytest.raises(ValueError, match="different test targets"):
        audit_cross_domain_run(folder)
