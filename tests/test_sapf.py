from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.data.cross_domain import fit_batch, validate_protocol
from src.data.domains import assert_data_usage
from src.data.google_panel import domain_windows
from src.data.shape_features import load_signature
from src.data.synthetic import make_demo_domains
from src.evaluation.backtest import score_predictions
from src.evaluation.cross_domain import audit_cross_domain_run, summarize_transfer
from src.models import MODEL_NAMES, create_model
from src.models.ml.normalized import normalized_inputs
from src.utils import load_config


@pytest.fixture
def panel():
    return make_demo_domains(31, 42)


def small_config(tmp_path):
    config = deepcopy(load_config("configs/sapf_google.yaml"))
    config["results_dir"] = str(tmp_path/"runs")
    config["save_plots"] = False
    config["pairwise_source_diagnostics"] = False
    config["domain_roles"] = {
        "source": ["demo_cell0_pdu0", "demo_cell0_pdu1"],
        "validation": ["demo_cell1_pdu0", "demo_cell1_pdu1"],
        "test": ["demo_cell2_pdu0", "demo_cell2_pdu1"],
    }
    return config


def test_active_default_registry_and_3day_contract():
    config = load_config("configs/models.yaml")
    assert set(config["default_models"]) <= set(MODEL_NAMES)
    assert {"normalized_ridge", "sapf_similarity", "sapf"} <= set(config["default_models"])
    assert "hdsf" not in MODEL_NAMES and "proposed" not in MODEL_NAMES
    assert load_config("configs/sapf_google.yaml")["horizons"] == {"24h": 96, "3d": 288}
    for name in config["default_models"]:
        create_model(name, config["models"][name])


def test_history_normalization_ignores_future_and_restores_scale(panel):
    domains = list(panel.series)
    train = domain_windows(panel, domains[:2], [14, 15], 1344, 288)
    target = domain_windows(panel, [domains[4]], [22], 1344, 288)
    transformed, location, scale = normalized_inputs(target.inputs)
    assert transformed.history.shape == target.inputs.history.shape
    assert abs(transformed.history[:, :, 0].mean()) < 1e-6
    with threadpool_limits(1):
        model = create_model("normalized_ridge", {"base_params": {"alpha": 100}}).fit(train)
        before = model.predict(target.inputs)
        # Affine changes of observed target history should restore the same affine forecast.
        changed_history = np.array(target.inputs.history, copy=True)
        changed_history[:, :, 0] = changed_history[:, :, 0]*3+2
        changed = replace(target.inputs, history=changed_history)
        np.testing.assert_allclose(model.predict(changed), before*3+2, rtol=1e-5, atol=1e-5)
    original = target.inputs.history.copy()
    normalized_inputs(target.inputs)
    np.testing.assert_array_equal(target.inputs.history, original)
    panel.series[domains[4]].iloc[22*96:, 0] = 99999
    later = domain_windows(panel, [domains[4]], [22], 1344, 288)
    np.testing.assert_array_equal(model.predict(target.inputs), model.predict(later.inputs))
    constant = replace(target.inputs, history=np.full_like(target.inputs.history, .5))
    assert np.isfinite(normalized_inputs(constant)[0].history).all()


def test_role_and_eagle_firewalls(panel, tmp_path):
    config = small_config(tmp_path)
    plan = validate_protocol(config, panel, ["24h", "3d"])
    assert plan["tasks"]["3d"]["development_origins"] == [14, 15]
    assert plan["tasks"]["3d"]["final_origins"] == list(range(14, 20))
    with pytest.raises(ValueError, match="Held-out"):
        fit_batch(panel, config["domain_roles"]["test"], [14], 1344, 96,
                  forbidden_domains=config["domain_roles"]["test"])
    reserved = replace(next(iter(panel.metadata.values())), dataset_id="nlr_eagle", usage_role="final_holdout")
    for phase in ["training", "validation", "calibration", "source_selection"]:
        with pytest.raises(ValueError, match="final holdout"):
            assert_data_usage(reserved, phase)
    assert_data_usage(reserved, "test")
    assert_data_usage(reserved, "inference_history")
    overlapping = deepcopy(config)
    overlapping["domain_roles"]["validation"][0] = overlapping["domain_roles"]["source"][0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_protocol(overlapping, panel, ["24h"])
    config["temporal_protocol"]["3d"]["validation_origins"] = [21]
    with pytest.raises(ValueError, match="Validation labels"):
        validate_protocol(config, panel, ["3d"])


def test_calibration_is_independent_and_prediction_state_is_frozen(panel, tmp_path):
    domains = list(panel.series)
    source = domain_windows(panel, domains[:3], [14, 15], 1344, 288)
    validation = domain_windows(panel, domains[4:6], [18, 19], 1344, 288)
    target = domain_windows(panel, domains[8:10], [22], 1344, 288)
    params = dict(base_params={"alpha": 100}, top_k=[1, 3], temperatures=[1.0],
                  feature_groups=["shape"], transfer_strengths=[0, .5, 1], reliability_powers=[1])
    with threadpool_limits(1):
        model = create_model("sapf", params).fit(source)
        with pytest.raises(ValueError, match="independent"):
            model.calibrate(source)
        model.calibrate(validation)
        calibration = deepcopy(model.calibration)
        model.refit(domain_windows(panel, domains[:3], [14, 15, 16, 17, 18, 19], 1344, 288))
        assert model.calibration == calibration
        saved = model.save(tmp_path/"model")
        digest = sha256(saved.read_bytes()).hexdigest()
        prediction, diagnostic = model.predict_with_diagnostics(target.inputs)
        assert prediction.shape == (2, 288)
        np.testing.assert_allclose(diagnostic["weights"].sum(1), 1)
        assert np.all((diagnostic["transfer_gate"] >= 0) & (diagnostic["transfer_gate"] <= 1))
        other = np.array(target.inputs.history, copy=True)
        other[1] = 1000
        poisoned = replace(target.inputs, history=other)
        np.testing.assert_array_equal(model.predict(poisoned)[0], prediction[0])
        np.testing.assert_allclose(joblib.load(saved).predict(target.inputs), prediction)
        assert digest == sha256(saved.read_bytes()).hexdigest()
        assert model.calibration == calibration
        with pytest.raises(ValueError, match="source pool"):
            model.refit(domain_windows(panel, domains[:2], [14, 15], 1344, 288))


def test_3day_daily_aggregation_and_nonoverlap(panel):
    batch = domain_windows(panel, [list(panel.series)[0]], list(range(22, 29)), 1344, 288)
    result = score_predictions(batch, batch.targets+0.01, anchor=batch.inputs.origins[0])
    assert result.daily_metrics["n"] == 7*3
    assert result.metrics["n"] == 7*288
    assert len(result.per_day) == 3
    assert len(result.nonoverlap_metrics["origins"]) == 3  # D22, D25, D28
    assert result.nonoverlap_metrics["n"] == 3*288


def test_load_signatures_separate_level_from_shape():
    shape = 1+.1*np.sin(np.arange(1344)*2*np.pi/96)
    first, scaled = load_signature(shape), load_signature(shape*3)
    assert scaled[0]-first[0] == pytest.approx(np.log(3))
    np.testing.assert_allclose(first[1:], scaled[1:], atol=1e-12)


def test_paired_comparison_keeps_the_strong_raw_pool_reference():
    rows = [dict(target_domain="t", horizon="24h", split="test", evaluation="curve", n=96, mape_n=96,
                 source_strategy="multi_source", model=name, MAPE=value)
            for name, value in [("ridge", 2.), ("normalized_ridge", 3.), ("sapf", 2.5), ("three_day_mean", 4.)]]
    pairs, _ = summarize_transfer(pd.DataFrame(rows))
    raw = pairs[pairs.comparison == "selective_vs_raw_pool"].iloc[0]
    normalized = pairs[pairs.comparison == "selective_vs_normalized_pool"].iloc[0]
    assert raw.negative_transfer and raw.delta_MAPE_pp == .5
    assert not normalized.negative_transfer and normalized.delta_MAPE_pp == -.5


def test_calibration_and_evaluation_thresholds_cannot_drift(tmp_path):
    from src.training.experiment import run_sapf
    config = small_config(tmp_path)
    config["mape_threshold"] = .1
    with pytest.raises(ValueError, match="same MAPE threshold"):
        run_sapf(SimpleNamespace(models=["sapf"]), config)


def test_legacy_aggregate_scores_are_not_labeled_as_worst_domain(capsys):
    from src.evaluation.reporting import print_results
    metrics = pd.DataFrame([dict(horizon="24h", model="ridge", split="test", evaluation="curve", MAPE=3.)])
    print_results(metrics, manifest=dict(data="legacy ESIF", target_unit="kW", requested_horizons=["24h"]))
    text = capsys.readouterr().out
    assert "ridge | 3.0000% | N/A" in text
    assert "多源 Ridge" not in text


def test_complete_default_cli_lifecycle_freezes_all_tasks_before_test(panel, tmp_path, monkeypatch):
    import src.training.experiment as experiment
    config = small_config(tmp_path)
    monkeypatch.setattr(experiment, "load_google_domains", lambda _: panel)
    original = experiment._evaluate
    def evaluate(folder, task, split, *args, **kwargs):
        if split == "test":
            frozen = json.loads((folder/"frozen-before-test.json").read_text(encoding="utf-8"))
            assert set(frozen["tasks"]) == {"24h", "3d"}
            assert len(frozen["training_records"]) == 12
        return original(folder, task, split, *args, **kwargs)
    monkeypatch.setattr(experiment, "_evaluate", evaluate)
    folder = experiment.run_sapf(SimpleNamespace(demo=False), config)
    assert audit_cross_domain_run(folder)["passed"]
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["domain_roles"]["source"] == config["domain_roles"]["source"]
    paths = list(folder.glob("demo_cell2*/3d/*/test_predictions.parquet"))
    assert len(paths) == 12
    path = paths[0]
    frame = pd.read_parquet(path)
    frame.loc[0, "actual"] += .1
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="same target"):
        audit_cross_domain_run(folder)


def test_archive_is_complete_and_replay_source_is_separate():
    from src.legacy import materialize_archive, verify_archive
    manifest = verify_archive()
    assert manifest["baseline_commit"].startswith("bc6a439")
    assert "src/models/proposed/hdsf.py" in manifest["source_files"]
    assert "configs/google_hdsf.yaml" in manifest["source_files"]
    assert len(manifest["historical_runs"]) == 44
    root = materialize_archive()
    assert (root/"scripts/run_experiment.py").is_file()
    assert (root/"src/models/proposed/hdsf.py").is_file()
    assert not Path("src/models/proposed/hdsf.py").exists()


@pytest.mark.parametrize("runtime_kind", ["current", "historical"])
def test_legacy_data_routing_survives_original_runtime_relaunch(runtime_kind):
    import subprocess
    import sys
    from src.legacy import legacy_environment, materialize_archive
    from src.utils import PROJECT_ROOT
    frozen = materialize_archive()
    config = load_config(frozen/"configs/google_hdsf.yaml")
    runtime = sys.executable if runtime_kind == "current" else config["python_executable"]
    if not Path(runtime).exists():
        pytest.skip("Original historical numerical environment is unavailable on this host.")
    code = "import json,src.utils as u; print(json.dumps([str(u.PROJECT_ROOT), str(u.resolve_path('data/raw/Google')), u.load_config('configs/google_hdsf.yaml')['comparison_context_days']]))"
    result = subprocess.run([runtime, "-B", "-c", code], cwd=frozen, env=legacy_environment(frozen),
                            check=True, capture_output=True, text=True, encoding="utf-8")
    root, data, context = json.loads(result.stdout)
    assert Path(root) == frozen
    assert Path(data) == PROJECT_ROOT/"data/raw/Google"
    assert context == 7
