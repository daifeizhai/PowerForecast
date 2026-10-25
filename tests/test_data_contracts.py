from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data.adapters import adapt_frame
from src.data.domains import DomainMetadata, aggregate_components, convert_target, validate_compatible
from src.data.google_panel import load_google_domains
from src.data.heterogeneous import ALIBABA_COLUMNS, adapt_m100_total_power, read_alibaba_table
from src.utils import load_config


def metadata():
    return DomainMetadata(**load_config("configs/datasets/google_cross_domain.yaml")["domain_metadata"])


def test_physical_unit_scope_and_sampling_boundaries():
    source = replace(metadata(), target_kind="active_power", target_scope="node", target_unit="W")
    values = np.array([1000., np.nan])
    converted = convert_target(values, source, target_kind="active_power", target_scope="node", target_unit="kW")
    assert converted[0] == 1 and values[0] == 1000 and np.isnan(converted[1])
    with pytest.raises(ValueError, match="target_scope"):
        convert_target(values, source, target_kind="active_power", target_scope="facility", target_unit="kW")
    with pytest.raises(ValueError, match="Unsupported physical"):
        convert_target([.5], metadata(), target_kind="active_power", target_scope="power_domain", target_unit="kW")
    with pytest.raises(ValueError, match="Native sampling"):
        replace(metadata(), raw_frequency="10min")
    with pytest.raises(ValueError, match="Incompatible"):
        validate_compatible([metadata(), replace(metadata(), domain_id="other", label_origin="assumed_scenario")])
    with pytest.raises(ValueError, match="future_known"):
        replace(metadata(), feature_availability={"gpu_observation": "future_known"})


def test_aggregation_and_generic_adapter_never_merge_domains():
    index = pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")
    parts = {"it": pd.Series([1, np.nan], index=index), "cooling": pd.Series([2, 3], index=index)}
    total = aggregate_components(parts, expected_components=list(parts), complete_coverage=True, nonoverlapping=True)
    assert total.iloc[0] == 3 and np.isnan(total.iloc[1])
    with pytest.raises(ValueError, match="nonoverlapping"):
        aggregate_components(parts, expected_components=list(parts), complete_coverage=True, nonoverlapping=False)
    raw = pd.DataFrame({"time": ["2024-01-01", "2024-01-01"], "domain_id": ["a", "b"], "load": [1, 10]})
    with pytest.raises(ValueError, match="partition by physical domain"):
        adapt_frame(raw, {"timestamp": "time", "columns": {"load": "load"}}, "UTC")


def test_google_anchor_quality_and_mapping_uncertainty(tmp_path):
    config = load_config("configs/datasets/google_cross_domain.yaml")
    config.update(raw_data=str(tmp_path), expected_domains=2, expected_days=1)
    pd.DataFrame({"machine_id": [1, 1], "cell": ["a", "a"], "pdu": ["pdu1", "pdu2"]}).to_csv(
        tmp_path/"machine_to_pdu_mapping.csv.gz", index=False)
    for pdu in ["pdu1", "pdu2"]:
        raw = pd.DataFrame({"time": 600000000+np.arange(288)*300000000, "cell": "a", "pdu": pdu,
                            "measured_power_util": .6, "bad_measurement_data": False})
        raw.loc[1, "bad_measurement_data"] = True
        raw.drop(index=range(12, 15)).to_csv(tmp_path/f"cella_{pdu}.csv.gz", index=False)
    panel = load_google_domains(config)
    assert panel.anchor == pd.Timestamp("2019-05-01T07:00Z")
    assert len(panel.series["cella_pdu1"]) == 96
    assert panel.series["cella_pdu1"].iloc[[0, 4], 0].isna().all()
    assert panel.coverage["cella_pdu1"].iloc[0] == 2
    assert panel.audit["cella_pdu1"]["mapping_overlap_domains"] == ["cella_pdu2"]


def test_m100_node_total_watts_to_kw_without_inventing_missing_samples():
    times = pd.date_range("2024-01-01", periods=90, freq="20s", tz="UTC")
    raw = pd.DataFrame({"timestamp": times, "node": "1", "metric": "total_power", "value": 500.0})
    raw = raw.drop(index=89)
    frame, contract, counts = adapt_m100_total_power(raw, "1", timezone="UTC", timezone_verified=False)
    assert frame.node_power_kw.iloc[0] == .5
    assert np.isnan(frame.node_power_kw.iloc[1])
    assert counts.tolist() == [45, 44]
    assert contract.target_scope == "node" and contract.label_origin == "measured"
    assert contract.timezone_verified is False
    raw["metric"] = "gpu0_power"
    with pytest.raises(ValueError, match="total_power"):
        adapt_m100_total_power(raw, "1", timezone="UTC")


def test_alibaba_headerless_schema_is_resource_only(tmp_path):
    columns = ALIBABA_COLUMNS["pai_sensor_table"].split(",")
    path = tmp_path/"pai_sensor_table.csv"
    pd.DataFrame([["id"]*6+[1.0]*(len(columns)-6)]).to_csv(path, header=False, index=False)
    frame = read_alibaba_table(path)
    assert list(frame) == columns and len(frame) == 1
    assert frame.attrs["trustworthy_power_target"] is False
    assert "lifetime" in frame.attrs["availability"]
    assert not any("power" in c for c in frame.columns)
    pd.DataFrame([[1, 2]]).to_csv(path, header=False, index=False)
    with pytest.raises(ValueError, match="documented columns"):
        read_alibaba_table(path)
