from types import SimpleNamespace

import pytest

from scripts.run_experiment import run_experiment
from scripts.reproduce_google_hdsf import run_reproduction


def test_google_protocol_dispatch_does_not_enter_nlr_pipeline(monkeypatch):
    import scripts.reproduce_google_hdsf as historical
    captured = {}

    def reproduction(args, config):
        captured.update(config)
        return "historical-panel-run"

    monkeypatch.setattr(historical, "run_reproduction", reproduction)
    assert run_experiment(SimpleNamespace(config="configs/google_hdsf.yaml")) == "historical-panel-run"
    assert captured["protocol"] == "google_hdsf_d14_d22"


@pytest.mark.parametrize("override", [{"demo": True}, {"start": "2019-05-02"},
                                     {"models": ["hdsf"]}, {"ablation": "target_only"}])
def test_historical_protocol_rejects_overrides_that_change_the_prescribed_experiment(override):
    values = dict(demo=False, start=None, end=None, dataset_config=None, model_config=None,
                  prepared_data=None, ablation="full", models=None)
    values.update(override)
    with pytest.raises(ValueError, match="prescribed member pool"):
        run_reproduction(SimpleNamespace(**values), {})
