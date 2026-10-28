# AcmeTrace Seren 目标域接入报告（power-transfer-v2）

> 分支：`research/power-transfer-v2`（自 `a98d3df` 基线新建；数据层从 `power-transfer-v1` 移植，本轮未 commit/push）。
> 快照：`E:/Data/AcmeTrace`（HF `Qinghao/AcmeTrace`，revision `a484f776…`，只读）。
> 产物生成：2026-10-28T07:05:55Z（15:05:55 +08:00）；全量装配耗时 **555.428 s**（`canonical_light_run`，`bounded_debug=false`）。
> 数字来源：`data/processed/acmetrace_target_metadata.json`、`acmetrace_target_quality.json`、`results/analysis/acmetrace/acmetrace_target_run_summary.json`、`v2_target_run.log`、`v2_power_audit_run.log` 及两个目标域 parquet；本轮未重跑长扫描。

## 1. 背景与范围

v2 研究线在 Stage 1+2 基线上把 AcmeTrace 完整快照接成“带运行指标的目标域”：主路径（A，`power_backbone`）只看历史 kW；辅助路径为 B 无标签运行指标表征、C 目标域有监督功率映射、D 主辅融合。本轮只做数据层，**不训练模型、不声称任何精度收益**。

本报告覆盖 `src/data/acmetrace_target.py` + `configs/datasets/acmetrace_target.yaml` + `scripts/prepare_acmetrace_target.py`（另有 `tests/test_acmetrace_target.py`）所装配的 Seren 15 分钟目标域数据集；功率审计与运行指标审计分别由 v1 移植的 `src/data/acmetrace_power.py`、`src/data/acmetrace_ops.py` 负责，细节见配套 [docs/ACMETRACE_OPS_AUDIT_V2.md](ACMETRACE_OPS_AUDIT_V2.md)。原始快照只读（写路径经 `assert_not_raw_path` 拒绝）。

## 2. 数据可用性与规模

| 来源/层级 | 原生采样 | 覆盖与规模 | 在本轮中的角色 |
|---|---|---|---|
| Seren GPU（DCGM，`GPU_UTIL.csv`） | 15 s | **1,992 卡 / 249 GPU 节点**；538,469 行；2023-05-16T09:43:15 → 2023-08-18T00:00 | 特征 `gpu_utilization` |
| Seren 节点指标（CPU / 内存 / IB 收 / 发） | 15 s | 248–249 个设备；每文件约 281k–282k 行；2023-07-01 → 2023-08-19 | 特征 `node_*` 4 列 |
| Seren IPMI 服务器功率（`Sys_Total_Power`） | **300 s** | **291 个节点**（CPU_D 8 / GPU_AB 113 / GPU_C 170 节点；7,051,781 行）；与 Seren 节点清单交集 248，IPMI-only 43 | 标签 Seren `cluster_it` |
| Seren job trace（`trace_seren.csv`） | 事件表 | 818,327 行，其中 806,551 条有完整生命周期；2023-02-28T16:15 → 2023-08-22T07:00（16,764 个 15 min 槽） | 特征 `job_*` 3 列 |
| Kalos GPU（DCGM） | 15 s | 2,344 卡 / 293 节点；**仅 GPU 级**，无 IPMI 服务器功率；15 min 槽 1,507 / 有效 1,156；约 16 天（2023-08） | 非目标域，注册为跨集群辅助候选 |

HF 快照树：**75 个文件 / 80,798,873,409 B（≈75.2 GiB）**；本地快照 153 个文件 / 80,798,902,594 B（其中 42 个 dataset CSV）。

## 3. 标签定义与诚实边界

标签 `seren_cluster_it_kw`（`label_type=acmetrace_seren_cluster_it_derived_kw`，`label_origin=component_sum`，`target_kind=active_power`，`target_scope=it`）：

- **定义**：每个 300 s 原生槽内，对**有有效读数的 Seren 服务器节点**取 `Sys_Total_Power`，先做 15 min 区间均值，再按上报节点求和；标签为区间起点、UTC-naive 轴。零/负功率按无效/离线观测处理，绝不当成真实 0 kW。
- **规模与时间**：8,352 槽（87 天），时间 **2023-05-14T16:00 → 2023-08-09T15:45**（UTC-naive 区间起点）；有效 **8,187**（**98.02%**），缺失 165；有效值均值 **731.75 kW**（min 409.63 / max 972.11）；槽 coverage 均值 0.9729。
- **不是设施功率**：标签是 IT 侧集群聚合（服务器 IPMI 按上报节点求和），**不含制冷、PDU/变压器损耗、照明等设施负荷**，也不是独立设施总表计量；`never_facility_power=true`，禁止报告为 facility kW。
- **也不是完整集群总量**：求和只覆盖上报节点（预期 248 个节点，43 个 IPMI-only 节点被排除），coverage<1 时只是“上报节点之和”。
- **单位未核验**：源 README 未写功率单位；量级证据（CPU 服务器约 0.47–0.96 kW、8 卡服务器最高约 6.5 kW、单卡 DCGM 约 90–420 W）表明原始值更像 W，kW 由推断 W 换算。`unit_inferred_from="W"`、`unit_verified=false`，每个 kW 数字都保留这层不确定性。
- **时区未核验**：monitor 时间戳带显式 `+08:00` 偏移，解析后转 UTC 再去掉时区（UTC-naive）；站点时区无官方文档，`timezone_verified=false`。不得把时间当作本地墙钟时间使用。
- **缺测不填补**：缺测保留 NaN + 显式掩码；无效槽保持 NaN，绝不 0 kW；不插值、不零填；离线节点剔除求和并降低 `coverage_ratio`。
- **契约强制**：`validate_target_config` 拒绝 `target_scope=facility`、非 `cluster_it` 级别与 `unit_verified=true` 等不诚实声明；`tests/test_acmetrace_target.py` 用合成 fixture 固定这些规则（不读原始数据）。

## 4. 运行特征与覆盖

8 个集群级特征（`role=target_domain_runtime_auxiliary`），观测覆盖率相对于标签网格：

| 特征 | 单位 | 观测槽 / 标签槽 | 比例 | 观测段平均 coverage | 观测窗口起止 |
|---|---|---|---:|---:|---|
| `node_cpu_utilization` | percent（未核验） | 3,807 / 8,352 | 45.58% | 0.9987 | 2023-07-01T00:00 → 2023-08-09T15:45 |
| `node_memory_utilization` | percent（未核验） | 3,797 / 8,352 | 45.46% | 0.9988 | 同上 |
| `node_ib_receive_rate` | unverified_rate | 3,798 / 8,352 | 45.47% | 0.9987 | 同上 |
| `node_ib_send_rate` | unverified_rate | 3,798 / 8,352 | 45.47% | 0.9987 | 同上 |
| `gpu_utilization` | percent（未核验） | 8,175 / 8,352 | 97.88% | 0.9563 | 2023-05-16T09:45 → 2023-08-09T15:45 |
| `job_active_count` | count | 8,352 / 8,352 | 100% | 1.0 | 2023-05-14T16:00 → 2023-08-09T15:45 |
| `job_active_gpu_num` | gpu_count | 8,352 / 8,352 | 100% | 1.0 | 同上 |
| `job_submitted_count` | count | 8,352 / 8,352 | 100% | 1.0 | 同上 |

运行特征是**上报设备的均值**（GPU 为每卡 DCGM 均值、节点指标为上报节点均值），**不是完整集群总量**；job 特征的 100% 只在记录的 trace 区间内成立（coverage 定义即 1.0 in span），trace 自身完整性不可独立验证，且无 node ID，只能做集群级活动特征。

**描述性 Pearson（仅证据，不是预测收益声明）**：

| 特征 | 重叠有效槽 | Pearson r |
|---|---:|---:|
| `gpu_utilization` | 8,067 | 0.8821 |
| `node_ib_send_rate` | 3,702 | 0.7492 |
| `node_ib_receive_rate` | 3,702 | 0.7492 |
| `node_cpu_utilization` | 3,711 | 0.5276 |
| `node_memory_utilization` | 3,701 | 0.5078 |
| `job_active_gpu_num` | 8,187 | 0.3659 |
| `job_active_count` | 8,187 | 0.0319 |
| `job_submitted_count` | 8,187 | 0.0251 |

这些系数在各自可用的重叠窗口内计算（CPU/内存/IB 仅约 3.7k 槽重叠），是**描述性重叠统计**；`quality_warnings` 明示其不是 predictive value 的证据，不构成因果或收益声明。

## 5. 15 分钟装配规则

- **网格**：900 s / 96 点每日，标签为区间**起点**，区间 `[t, t+15min)`；UTC-naive。
- **标签聚合**：300 s IPMI 有效样本 → 每节点 15 min 区间均值 → 上报节点求和；`power_valid = 上报节点 coverage ≥ 0.5`；无效槽 NaN。
- **特征聚合**：15 s 指标 → 每 15 min 对上报设备取均值；job 活动按半开生命周期 `[start_time, end_time)` 在 bin 内计数/求和（缺 start/end 的作业剔除并计数，`gpu_num` 缺失按 0 计入 GPU 活动数）。
- **coverage 定义**：每个 15 min 槽对预期设备取 `min(原生样本数, 预期样本数) / 预期样本数` 的均值（15 s 指标预期 60、IPMI 预期 3）；无样本设备贡献 0；封顶 1.0。
- **重复/乱序**：重复时间戳计入审计但不从均值中删除；乱序行计数；两者都不改变“不伪造样本”的口径。
- **对齐**：所有特征落到标签网格；源观测窗之外的槽保持 NaN + `observed_mask=False`，不补零。
- **掩码优先**：`observed = found & finite & coverage ≥ 0.5`；无效槽的 `feature_value` 一律 NaN。

## 6. 产物与注册表

| 产物 | 形状 / 大小 | 说明 |
|---|---|---|
| `data/processed/acmetrace_target_power.parquet` | 8,352 × 5（140,073 B） | `timestamp, site_id, power_kw, power_valid, coverage_ratio`；sha256 `a9bb4dde…` |
| `data/processed/acmetrace_target_features.parquet` | 66,816 × 7（505,174 B） | 8 特征 × 8,352 槽长表；sha256 `622bd3c0…` |
| `data/processed/acmetrace_target_metadata.json` | 18,843 B | 标签语义、特征注册、时区、来源、readiness |
| `data/processed/acmetrace_target_quality.json` | 12,660 B | 覆盖率/缺失/描述性相关性与质量告警 |
| `results/analysis/acmetrace/acmetrace_target_run_summary.json` / `v2_target_run.log` | 8,077 B / 12,386 B | 运行摘要与控制台日志 |
| 上游 `acmetrace_power_15min.parquet`（v2 轻量功率审计重建） | 23,883,790 行 / 68,820,108 B | sha256 `89088355…`；`acmetrace_inventory.json` 102,047 B |

注册表（`configs/datasets/registry.yaml`）：`acmetrace_seren` 为 `target_supervised`（`native_frequency="15min_adapted"`，`audit_status=partial`，`can_produce_facility_kw=false`）；`acmetrace_kalos` 为 `candidate`（label 为 null、GPU 级、短历史，`can_produce_facility_kw=false`）。`fusion.target_only_local_weight=1.0`，全局融合权重仍为 `null`（未决）。

就绪状态：`ready_for_15min_analysis=true`、`ready_for_training=false`；原因即单位推断 W 未核验、站点时区未核验、cluster-IT 非设施 kW 且非完整集群总量、job trace 无 node ID。

## 7. 轻量审计策略

- **功率层**：`scripts.audit_acmetrace_power --skip-hash` 只读约 7.6 GB 功率 CSV，跳过对 80.8 GB（75.2 GiB）全树的内容 SHA256；但**尺寸与 HF 树一致性仍完整**：库存 153 个本地文件，HF tree 75 个文件 **75/75 存在、0 尺寸不匹配、0 缺失**（78 个本地文件不在 tree 清单内）。跳过的是内容哈希：52 个带 LFS sha256 的 tree 条目被标为 `sha256_verification="skipped"`（unverified，而非 mismatch），23 个 tree 条目只有 blob_id；`sha256_mismatches=[]`。
- **运行指标层**：运行指标审计用 `--metrics` 白名单**限定到目标域实际使用的 Seren `NODE_CPU_UTILIZATION` / `NODE_MEMORY_UTILIZATION` / `NODE_IB_RECEIVE` / `NODE_IB_SEND`**（另按默认路径审计 Seren IPMI 与 job trace）；白名单运行被显式标为 PARTIAL，不跑全量 17 指标 / 80 GB 审计。目标域装配本身完整流式扫描其使用的 5 个 monitor CSV + Seren job trace（≈6.96 GB），行数/设备数/原生间隔/取值域/缺失策略记入 quality 的 `metric_audit`（其中 `GPU_UTIL` 5,139,607,278 B 由装配扫描，不在本轮 ops 白名单内）。
- **只读与边界**：原始树整体只读，`assert_not_raw_path` 在写产物前拒绝 `data/raw/` 下的路径；跳过的哈希不冒充已验证。

## 8. 与 v2 研究方案的关系

本轮产物是 v2 的**目标域数据基础**，不是模型收益：

- **主路径 A**：提供 15 分钟、mask-first 的 Seren cluster-IT 功率序列（8,352 槽），可用于历史功率预测的主路径/回测研究——但标签是 cluster-IT 而非设施 kW，结论必须限定在该计量边界内。
- **辅助路径 B/C**：8 个集群级运行特征与 job 活动与标签同网格，可支撑无标签运行指标表征、目标域功率映射等研究；特征只覆盖上报设备，不是完整集群量，且无设备级配对。
- **融合 D**：注册表保留 `target_only_local_weight=1.0`，全局融合权重未决；本轮没有训练、没有验证集协议、没有收益数字。
- **诚实定位**：AcmeTrace 不含设施总表功率，任何“设施 kW”结论都不能由本数据集支持；正式设施 kW 数据接入仍是 v2 的阻断点。

## 9. 未做与限制

1. 无设施功率表：标签只是 Seren cluster-IT 聚合，冷却/PDU/变压器/照明等负荷全部缺失。
2. 标签非完整集群总量：预期 248 节点中只对上报节点求和，43 个 IPMI-only 节点被排除；槽 coverage 均值 0.9729。
3. 功率单位由量级推断为 W 且 `unit_verified=false`；站点时区 `timezone_verified=false`。
4. 节点 CPU/内存/IB 特征只在 2023-07-01 之后有数据（约占网格 45.6%）；GPU 利用率从 2023-05-16 起。
5. job trace 无 node ID，且 `max_active_gpu=15,400` 远超观测到的 1,992 个 GPU 列（818,327 条作业中 367,737 条 `gpu_num=0`），作业口径与设备清单的对应关系尚未查清，job 特征只能当集群级活动代理。
6. Kalos 只有 GPU 级、约 16 天短历史，不是本轮目标域；无 Kalos cluster-IT 序列。
7. 全树内容 SHA256 未重算（`--skip-hash`），完整性状态是“尺寸/树一致 + 内容哈希 unverified”。
8. 相关性只在可变重叠窗口计算，是描述性证据，不能外推为预测收益或因果结论。
9. 未做训练/验证切分、未做 source→target 特征映射实验、未做基线对比与任何模型评估。
10. 单位、时区、作业口径等未核验项在 metadata/quality/registry 中显式保留，放松任一口径都需要新一轮审计。

## 10. 复现命令

```powershell
# 1) 功率层（轻量：跳过 80.8 GB 全树 SHA256，尺寸与 HF tree 一致性仍完整）
& .\.venv\Scripts\python.exe -m scripts.audit_acmetrace_power --skip-hash

# 2) 目标域装配（Seren cluster-IT 标签 + 运行特征 + job 活动，15 min）
& .\.venv\Scripts\python.exe -m scripts.prepare_acmetrace_target

# 3) 边界烟雾（每文件只读 1 个 chunk，结果不得引用为全量覆盖）
& .\.venv\Scripts\python.exe -m scripts.prepare_acmetrace_target --max-chunks 1 --no-write

# 4) 轻量运行指标审计（白名单，仅目标域使用的 4 个 NODE_* 指标）
& .\.venv\Scripts\python.exe -m scripts.audit_acmetrace_ops --clusters seren `
    --metrics NODE_CPU_UTILIZATION NODE_MEMORY_UTILIZATION NODE_IB_RECEIVE NODE_IB_SEND

# 5) 契约测试（合成 fixture，不读多 GB 原始数据）
& .\.venv\Scripts\python.exe -m pytest tests/test_acmetrace_target.py -q
```

原始树位置可用 `ACMETRACE_RAW_ROOT` 覆盖（默认 `E:/Data/AcmeTrace`）；运行环境（`metadata.provenance`）：Python 3.13.5，pandas 2.3.3，numpy 2.5.4，pyarrow 24.0.0，config_digest `2e2c25d2…`。功率标签依赖第 1 步产出的 `data/processed/acmetrace_power_15min.parquet`。
