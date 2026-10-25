from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
import yaml

from src.data.cross_domain import fit_batch, validate_protocol
from src.data.google_panel import DomainPanel, domain_windows
from src.data.m100_panel import adapt_m100_aggregated, load_m100_domains
from src.evaluation.cross_domain import audit_cross_domain_run
from src.models import create_model
from src.models.ml.ablation import ABLATION_MODES
from src.utils import load_config


def metadata():
    return deepcopy(load_config("configs/datasets/m100_nodes.yaml")["domain_metadata"])


def test_aggregated_adapter_converts_watts_once_and_keeps_missing_slots():
    index = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC").as_unit("us")
    raw = pd.DataFrame({"timestamp": index.delete(3), "total_power_avg": [500., 600., np.nan, -10., np.inf, 700., 800.],
                        "total_power_max": 999999.})
    before = raw.copy(deep=True)
    frame, contract, audit = adapt_m100_aggregated(raw, node_id=2, rack_id=0, metadata=metadata())
    assert frame.index.dtype == "datetime64[ns, UTC]"
    assert frame.columns.tolist() == ["node_power_kw"]
    assert frame.node_power_kw.iloc[:2].tolist() == [.5, .6]
    assert frame.node_power_kw.iloc[2:6].isna().all()
    assert frame.node_power_kw.iloc[-1] == .8
    assert audit["raw_samples_per_interval"] is None and audit["missing_grid_slots"] == 1
    assert (contract.target_scope, contract.target_unit, contract.raw_frequency) == ("node", "kW", "15min")
    pd.testing.assert_frame_equal(raw, before)
    duplicate = pd.concat([raw.iloc[:2], raw.iloc[[1]]], ignore_index=True)
    result, _, audit = adapt_m100_aggregated(duplicate, node_id=2, rack_id=0, metadata=metadata())
    assert result.node_power_kw.iloc[1:].isna().all() and audit["duplicate_rows_involved"] == 2


def test_aggregated_adapter_refuses_ambiguous_time_and_wrong_physics():
    raw = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2, freq="15min"), "total_power_avg": 500.})
    with pytest.raises(ValueError, match="UTC"):
        adapt_m100_aggregated(raw, node_id=2, rack_id=0, metadata=metadata())
    raw.timestamp = raw.timestamp.dt.tz_localize("UTC")+pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="grid"):
        adapt_m100_aggregated(raw, node_id=2, rack_id=0, metadata=metadata())
    raw.timestamp -= pd.Timedelta(minutes=1)
    wrong = {**metadata(), "target_scope": "facility"}
    with pytest.raises(ValueError, match="physical contract"):
        adapt_m100_aggregated(raw, node_id=2, rack_id=0, metadata=wrong)
    with pytest.raises(ValueError, match="total_power_avg"):
        adapt_m100_aggregated(raw.rename(columns={"total_power_avg": "p0_power_avg"}), node_id=2, rack_id=0, metadata=metadata())


def test_tar_adapter_checks_file_identity_and_never_requires_45_aggregate_rows(tmp_path):
    path = tmp_path/"0.tar"
    index = pd.date_range("2024-01-01", periods=3*96, freq="15min", tz="UTC")
    with tarfile.open(path, "w") as archive:
        for node in [2, 3]:
            payload = pd.DataFrame({"timestamp": index, "total_power_avg": 500.+node}).to_parquet(index=False)
            member = tarfile.TarInfo(f"{node}.parquet")
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))
    config = load_config("configs/datasets/m100_nodes.yaml")
    config.update(raw_data=str(tmp_path), trace_start=str(index[0]), expected_days=3, expected_domains=2,
                  archives=[dict(filename="0.tar", rack_id="0", node_ids=[2, 3], sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    panel = load_m100_domains(config)
    assert len(panel.series) == 2
    assert panel.series["m100_rack0_node2"].node_power_kw.eq(.502).all()
    assert all(count.eq(1).all() for count in panel.coverage.values())
    config["archives"][0]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="SHA256"):
        load_m100_domains(config)


@pytest.fixture
def m100_panel():
    index = pd.date_range("2024-01-01", periods=60*96, freq="15min", tz="UTC", name="timestamp")
    points = np.arange(len(index))
    series, metas, audit, coverage = {}, {}, {}, {}
    for node in range(6):
        power = 500+80*node+20*np.sin(2*np.pi*points/96+.1*node)+3*np.cos(2*np.pi*points/672)
        frame, contract, record = adapt_m100_aggregated(pd.DataFrame({"timestamp": index, "total_power_avg": power}),
            node_id=node, rack_id=node//3, metadata=metadata())
        series[contract.domain_id], metas[contract.domain_id] = frame, contract
        audit[contract.domain_id], coverage[contract.domain_id] = record, pd.Series(1, index=index)
    return DomainPanel(series, metas, coverage, {}, index[0], audit)


def small_config(tmp_path, panel):
    config = load_config("configs/sapf_m100.yaml")
    domains = list(panel.series)
    config["domain_roles"] = dict(source=domains[:2], validation=domains[2:4], test=domains[4:])
    config["results_dir"] = str(tmp_path/"runs")
    config["save_plots"] = False
    config["pairwise_source_diagnostics"] = False
    models = load_config(config["model_config"])
    for name in ["sapf", "sapf_similarity"]:
        models["models"][name].update(top_k=[1, 2], temperatures=[1.], feature_groups=["shape"],
                                      transfer_strengths=[.5], reliability_powers=[1.])
    path = tmp_path/"models.yaml"
    path.write_text(yaml.safe_dump(models), encoding="utf-8")
    config["model_config"] = str(path)
    return config


def test_m100_windows_are_node_isolated_with_exact_14day_history(m100_panel, tmp_path):
    config = small_config(tmp_path, m100_panel)
    plan = validate_protocol(config, m100_panel, ["24h", "3d"])
    for task, horizon in [("24h", 96), ("3d", 288)]:
        batch = domain_windows(m100_panel, plan["roles"]["test"], [40], 1344, horizon)
        assert batch.inputs.history.shape == (2, 1344, 5) and batch.targets.shape == (2, horizon)
        assert batch.inputs.future_known.shape == (2, horizon, 4)
        for i, domain in enumerate(batch.inputs.domain_ids):
            np.testing.assert_array_equal(batch.targets[i], m100_panel.series[domain].node_power_kw.iloc[40*96:40*96+horizon].to_numpy(np.float32))
    with pytest.raises(ValueError, match="Held-out"):
        fit_batch(m100_panel, plan["roles"]["test"], [14], 1344, 96, forbidden_domains=plan["roles"]["test"])
    first = plan["roles"]["test"][0]
    m100_panel.audit[first]["history_availability_upper_bound_minutes"] = 16
    with pytest.raises(ValueError, match="unavailable"):
        domain_windows(m100_panel, [first], [40], 1344, 96)


def test_controlled_ablations_keep_other_choices_fixed(m100_panel, tmp_path):
    domains = list(m100_panel.series)
    source = domain_windows(m100_panel, domains[:2], [14, 15, 16], 1344, 96)
    validation = domain_windows(m100_panel, domains[2:4], [30, 31], 1344, 96)
    test = domain_windows(m100_panel, domains[4:], [40], 1344, 96)
    params = dict(base_params={"alpha": 100., "feature_mode": "observed_history"}, top_k=[1], temperatures=[1.],
                  feature_groups=["shape"], transfer_strengths=[.5], reliability_powers=[1.])
    with threadpool_limits(1):
        full = create_model("sapf", params).fit(source).calibrate(validation)
        reference_prediction, original = full.predict_with_diagnostics(test.inputs)
        path = full.save(tmp_path/"full")
        initial_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for name, disabled in ABLATION_MODES.items():
            model = create_model(name, {"reference_model": full}).fit(source)
            prediction, diagnostic = model.predict_with_diagnostics(test.inputs)
            assert model.calibration["selected"] == full.calibration["selected"]
            assert model.calibration["reliability"] == full.calibration["reliability"]
            assert model.calibration["intervention"]["independently_retuned"] is False
            if disabled == "selection":
                np.testing.assert_allclose(diagnostic["weights"], .5)
                np.testing.assert_array_equal(diagnostic["transfer_gate"], original["transfer_gate"])
            elif disabled == "fallback":
                np.testing.assert_array_equal(diagnostic["weights"], original["weights"])
                np.testing.assert_array_equal(diagnostic["transfer_gate"], 1.)
            elif disabled == "normalization":
                np.testing.assert_array_equal(diagnostic["weights"], original["weights"])
                np.testing.assert_array_equal(diagnostic["transfer_gate"], original["transfer_gate"])
                assert not np.allclose(prediction, reference_prediction)
            np.testing.assert_allclose(joblib.load(model.save(tmp_path/name)).predict(test.inputs), prediction)
            with pytest.raises(ValueError, match="source pool"):
                create_model(name, {"reference_model": full}).fit(validation)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == initial_hash
        np.testing.assert_array_equal(full.predict(test.inputs), reference_prediction)


def test_m100_both_tasks_freeze_units_and_all_interventions(m100_panel, tmp_path, monkeypatch):
    import src.data.m100_panel as adapter
    from src.training.experiment import run_sapf
    config = small_config(tmp_path, m100_panel)
    monkeypatch.setattr(adapter, "load_m100_domains", lambda _: m100_panel)
    folder = run_sapf(SimpleNamespace(demo=False), config)
    audit = audit_cross_domain_run(folder)
    assert audit["passed"] and audit["controlled_ablations_matched"] and audit["physical_contract_preserved"]
    manifest_path = folder/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["target"] == "node_power_kw" and manifest["target_unit"] == "kW"
    assert len(manifest["adapted_data_sha256"]) == 6
    assert pd.read_csv(folder/"metrics.csv").target_unit.eq("kW").all()
    manifest["target_unit"] = "W"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not audit_cross_domain_run(folder)["physical_contract_preserved"]
