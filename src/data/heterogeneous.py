"""Audited heterogeneous source schemas; resource statistics never become fake power labels."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.adapters import adapt_frame
from src.data.domains import DomainMetadata
from src.data.resampling import resample_frame

ALIBABA_COLUMNS = {
    "pai_job_table": "job_name,inst_id,user,status,start_time,end_time",
    "pai_task_table": "job_name,task_name,inst_num,status,start_time,end_time,plan_cpu,plan_mem,plan_gpu,gpu_type",
    "pai_instance_table": "job_name,task_name,inst_name,worker_name,inst_id,status,start_time,end_time,machine",
    "pai_sensor_table": "job_name,task_name,worker_name,inst_id,machine,gpu_name,cpu_usage,gpu_wrk_util,avg_mem,max_mem,avg_gpu_wrk_mem,max_gpu_wrk_mem,read,write,read_count,write_count",
    "pai_group_tag_table": "inst_id,user,gpu_type_spec,group,workload",
    "pai_machine_spec": "machine,gpu_type,cap_cpu,cap_mem,cap_gpu",
    "pai_machine_metric": "worker_name,machine,start_time,end_time,machine_cpu_iowait,machine_cpu_kernel,machine_cpu_usr,machine_gpu,machine_load_1,machine_net_receive,machine_num_worker,machine_cpu",
}

ALIBABA_CSV_SHA256 = {
    "pai_group_tag_table": "d23fcb56d88b2976fda4a62708d77b2a8c98635c4b443f100ed5553d4cd07fbe",
    "pai_instance_table": "b179f7d7e0927a6663d719b728c92640447c5a0fc3c6e4edff31e7207ab9dd17",
    "pai_job_table": "379ecb3becaba347f44a53bf7eb53e54b185221b2a0338a3f828828d269ba96c",
    "pai_machine_metric": "8b6592daab0373ef7fec730198b4df5891b1722c2ab4c7016249166e131402d3",
    "pai_machine_spec": "251e9e33649155a6a935fdc1dc0e90cd4698206936b60bd8fce60ff2ccd51439",
    "pai_sensor_table": "12dd9929b70f3efe18d1279d9873e3a519c64f431613b23b059ed8c46a376dd7",
    "pai_task_table": "6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de",
}


def read_alibaba_table(path, *, nrows=None):
    path = Path(path)
    key = path.stem
    if key not in ALIBABA_COLUMNS:
        raise ValueError("Choose a documented Alibaba GPU v2020 table.")
    frame = pd.read_csv(path, header=None, nrows=nrows)
    columns = ALIBABA_COLUMNS[key].split(",")
    if frame.shape[1] != len(columns):
        raise ValueError(f"{path.name}: expected {len(columns)} documented columns, got {frame.shape[1]}.")
    frame.columns = columns
    frame.attrs.update(dataset_id="alibaba_gpu_v2020", trustworthy_power_target=False,
        target_kind="resource_activity", label_origin="not_estimable",
        availability="instance_lifetime_or_interval_end; not automatically known before completion",
        source="https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020")
    return frame


def audit_alibaba(root, hashes=False):
    rows = []
    for name, expected in ALIBABA_CSV_SHA256.items():
        path = Path(root)/(name+".csv")
        if not path.exists():
            rows.append(dict(table=name, present=False))
            continue
        frame = read_alibaba_table(path, nrows=1000)
        record = dict(table=name, present=True, bytes=path.stat().st_size, columns=list(frame),
                      sample_rows=len(frame), trustworthy_power_target=False)
        if hashes:
            digest, count, last = sha256(), 0, b""
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8*1024*1024), b""):
                    digest.update(block)
                    count += block.count(b"\n")
                    last = block[-1:]
            record.update(sha256=digest.hexdigest(), expected_sha256=expected,
                          official_hash_matches=digest.hexdigest() == expected,
                          rows=count+int(bool(last) and last != b"\n"))
        rows.append(record)
    return dict(dataset_id="alibaba_gpu_v2020", tables=rows,
                use="Unlabeled resource-pattern research only; lifecycle means and maxima require after-completion availability.",
                forbidden="No fabricated kW supervision, no replication of lifetime means into 15min observations.")


def adapt_m100_total_power(frame, node, *, timezone: str, timezone_verified=False, timestamp_unit=None):
    """Audited metric=total_power node table. Returns kW sample means and actual sample counts."""
    required = {"timestamp", "node", "value"}
    if not required.issubset(frame):
        raise ValueError("M100 total_power requires timestamp, node and value from the official metric partition.")
    selected = frame.loc[frame.node.astype(str) == str(node)].copy()
    if selected.empty:
        raise ValueError(f"No M100 samples for node={node}.")
    if "metric" in selected and set(selected.metric) != {"total_power"}:
        raise ValueError("Only measured total_power is accepted; chip/GPU metrics cannot stand in for node total.")
    if pd.api.types.is_numeric_dtype(selected.timestamp):
        if timestamp_unit not in {"s", "ms", "us", "ns"}:
            raise ValueError("Numeric M100 timestamps require an explicitly audited timestamp_unit.")
        selected["timestamp"] = pd.to_datetime(selected.timestamp, unit=timestamp_unit, utc=True)
    power = pd.to_numeric(selected.value, errors="raise")
    selected["value"] = power.where(np.isfinite(power) & (power >= 0))
    adapted = adapt_frame(selected, {"timestamp": "timestamp", "columns": {"node_power_kw": "value"},
                                    "scales": {"node_power_kw": .001}}, timezone)
    counts = adapted.node_power_kw.resample("15min", label="left", closed="left").count()
    result = resample_frame(adapted)
    result.loc[counts < 45, "node_power_kw"] = np.nan
    contract = DomainMetadata(dataset_id="m100_exadata", site_id="marconi100", domain_id=f"m100_node_{node}",
        node_id=str(node), target_field="node_power_kw", target_kind="active_power", target_scope="node",
        target_unit="kW", label_origin="measured", timezone=timezone, timezone_verified=timezone_verified,
        raw_frequency="20s", sampling_semantics="sample_mean",
        interval_definition="start; [timestamp, timestamp + 15 minutes)",
        coverage="At least 45 finite unique-timestamp samples per 15min bin; actual counts retained; no integration claim.",
        source_version="M100 ExaData IPMI metric=total_power; schema adapter, not a downloaded/validated local dataset",
        feature_availability={"target": "history_only", "calendar": "future_known"})
    return result, contract, counts
