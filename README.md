# PowerForecast

**面向算力中心（数据中心 / 超算中心）设施级有功功率（kW）预测的研究工程**

> 当前分支 `research/power-transfer-v2`：在 Stage 1+2（任务书 V2.0）稳定基线 `a98d3df` 上，
> 接入 **AcmeTrace Seren 目标域数据层**（cluster-IT 功率 + 运行指标 + 作业活动，统一 15 min），
> 详见 [docs/ACMETRACE_TARGET_V2.md](docs/ACMETRACE_TARGET_V2.md)。
> 第三至第五阶段（无标签运行指标预训练、功率映射、主辅融合）**仍只保留接口与数据基础，未实现、未声称提升**。

> 分支历史：`research/power-transfer-v1` 已完成《算力中心中短期负荷预测模型开发任务书 V2.0》的
> **第一阶段（多源数据预处理与特征分析）** 与 **第二阶段（历史功率主模型 A：Chronos-2 + TiRex-2 零样本等权融合）**；
> `research/power-transfer-v2` 从该主线分离，用于“功率主导 + 运行特征辅助”的新一轮研究（数据层先行）。

## 0. V2.0 交付总览（当前主线）

| 项目 | 状态 | 产物/证据 |
|---|---|---|
| 目标域真实功率数据处理（NLR ESIF，`facility_derived_kw`） | ✅ 完成 | `data/processed/processed_target_power.parquet`、`target_metadata.json`、`docs/STAGE1_TARGET_REPORT.md` |
| 目标功率特性分析（曲线/典型日/ACF/爬坡/峰谷/覆盖率） | ✅ 完成 | `results/analysis/stage1/power_characteristics/`（7 图 + `power_stats.json`） |
| 真实源域运行指标处理（M100，16 节点 15min 原生） | ✅ 完成 | `data/processed/processed_source_features.parquet`（1720 万行长表）、`source_metadata.json`、`docs/STAGE1_SOURCE_REPORT.md` |
| 异构字段注册与对齐（共享/源独有/目标独有/不可比） | ✅ 完成 | `configs/datasets/feature_registry.yaml`、`docs/FEATURE_ALIGNMENT_REPORT.md` |
| 主模型 A：Chronos-2 + TiRex-2，P50 零样本，固定 0.5/0.5 | ✅ 真实运行 | `src/models/foundation/`、`src/models/power_main/` |
| 固定 D14/D22 协议：4h / 8h / 24h / 3d（7d 可选） | ✅ 真实运行 | `configs/experiments/power_main_v1.yaml`、`results/runs/<run_id>/` |
| 基线 B0/B1 同协议对照 | ✅ 完成 | 同一 run 内同起报/同标签/同指标 |
| 指标/分层报告/7 类图/可复现记录 | ✅ 完成 | `metrics.csv`、`origin_metrics.csv`、`daily_metrics.csv`、`improvements.csv`、`predictions.parquet`、`figures/` |
| 第三至第五阶段（预训练/映射/融合） | 🚧 未实现 | `docs/TODO_RESEARCH.md`；全局融合权重 `None` |

### 第二阶段真实测试集结果（NLR ESIF，`facility_derived_kw`，D22–D30 起报，CPU 零样本）

| 预测跨度 | B0 前一天同刻 WAPE | B1 前三天均值 WAPE | A1 Chronos-2 WAPE | A2 TiRex-2 WAPE | **A3 等权融合 WAPE** |
|---|---|---|---|---|---|
| 4h | 9.05% | 9.74% | 1.27% | 1.11% | **1.17%** |
| 8h | 9.14% | 9.80% | 1.49% | 1.38% | **1.41%** |
| 24h | 7.59% | 8.61% | 5.82% | 5.95% | **5.87%** |
| 3d | 7.58% | 8.40% | 6.07% | 6.57% | **6.23%** |
| 7d（可选） | 10.61% | 8.69% | 6.85% | 7.68% | **7.12%** |

**如实说明**：等权融合并不总是优于两个单模型 —— 4h/8h 上 TiRex-2 略优、24h/3d/7d 上 Chronos-2 略优，
A3 只在 4h/8h 上优于两者之一；不使用测试集选择权重（见 `docs/DELIVERY_REPORT_V2.md`）。

### 快速命令

```powershell
# 第一阶段：真实目标域 / 源域数据处理与分析（只读原始数据，写 data/processed 与 results/analysis）
python -m scripts.prepare_target_power --config configs/datasets/nlr_target.yaml
python -m scripts.analyze_power        --config configs/datasets/nlr_target.yaml
python -m scripts.prepare_source_features --config configs/datasets/sources.yaml
python -m scripts.analyze_sources      --config configs/datasets/sources.yaml

# 第二阶段：真实零样本主模型（离线权重，4h/8h/24h/3d × dev+test × 5 模型）
python -m scripts.run_power_main --config configs/experiments/power_main_v1.yaml

# 可选 7d 预设（验证长跨度递推）与结果摘要
python -m scripts.run_power_main --horizons 7d --splits development test
python -m scripts.summarize_power_run --run results/runs/<run_id>

# 权重/依赖自检（独立准备步骤；实验默认离线，不下载）
python -m scripts.probe_foundation_models
```

真实实验需要 `torch(CPU) + chronos-forecasting==2.3.2 + tirex-2==0.3.0`（见 `requirements-optional.txt`）
以及本地冻结权重缓存；权重缺失时脚本明确报错，绝不以随机初始化或别的模型输出冒充。

## 1. 阶段 0：通用工程骨架（本分支前序交付）

> 以下内容来自本轮之前的骨架重建，仍是当前工程的地基（数据契约、时间协议、冻结生命周期、合成 smoke）。
> 合成数据结果一律标记 `synthetic=true`、`research_result=false`，不产出真实设施 kW 科研结论。

## 1.1 阶段 0 交付是什么

| 能力 | 状态 |
|---|---|
| 四层数据契约（Dataset / Feature / Target / Batch）与物理单位、计量边界、标签来源校验 | ✅ 已实现 |
| 无标签 `PretrainBatch` 与有标签 `ForecastInputs/ForecastBatch` 严格分离 | ✅ 已实现 |
| 跨源字段语义注册表、目标独有字段保留、全局融合权重显式未决（`None`） | ✅ 已实现 |
| 通用可配置时间协议（trace 锚点、Dk 日号、任意 L/H/origins/cutoffs） | ✅ 已实现 |
| 可选历史预设 `fixed_24h` / `fixed_7d`（仅配置，代码不硬编码 D14/D22） | ✅ 已实现 |
| 两个免拟合 CPU 基线：`seasonal_naive`、`three_day_mean` | ✅ 已实现 |
| 四阶段运行（development / finalize / final-test / smoke）+ 冻结/重载/哈希校验 | ✅ 已实现 |
| MAPE/WAPE/MAE/RMSE + per-origin / per-lead / per-forecast-day / daily-mean | ✅ 已实现 |
| 31+ trace 日合成数据 smoke（24h 与 7d） | ✅ 已实现 |
| 无标签表征预训练、跨域迁移、目标域映射、主辅融合 | 🚧 仅接口（`planned`，调用即明确失败） |
| LightGBM / LSTM / TCN / Transformer / PatchTST 等 | 🚧 `planned`，不可被选中 |
| 真实设施级 kW 数据接入与正式论文实验 | ⛔ 本轮阻断（见 `docs/TODO_RESEARCH.md`） |

详细模块状态矩阵见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §3。

## 2. 安装（CPU，无需 GPU）

```powershell
# Python >= 3.11
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -e .
python -m pytest -q
```

- 核心依赖：`numpy / pandas / pyarrow / PyYAML / pytest`（见 `requirements.txt`）；
- 可选依赖：`requirements-optional.txt`（绘图、未来拟合类基线），**不要求 GPU**；
- 不安装任何深度学习框架或基座模型权重。

## 3. 最小闭环命令（必须实际跑通）

```powershell
# 24h 合成 smoke：一次命令内部完成 准备 → 冻结 → 重载 → 测试
python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 24h

# 7d 合成 smoke（3 条以上预测窗口会重叠，产物中显式标注）
python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 7d

# 可选历史标准预设（仅验证配置能力，不要求所有模型套用）
python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 24h --protocol configs/protocols/fixed_24h.yaml
python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 7d  --protocol configs/protocols/fixed_7d.yaml

# 数据工具（不依赖任何原始数据）
python -m scripts.prepare_data --config configs/datasets/synthetic.yaml
python -m scripts.inspect_data --config configs/datasets/synthetic.yaml
```

输出在 `results/runs/<UTC时间>_<随机ID>/`，控制台与 `manifest.json` 都会明确标注：

```text
[synthetic] Engineering validation only (synthetic=true, research_result=false).
```

### 阶段与冻结

```powershell
# 1) 冻结（不读测试标签、不评分）
python -m scripts.run_experiment --config configs/smoke.yaml --demo --phase finalize --horizon 24h
# 2) 从冻结运行恢复并测试（缺少冻结凭据会被拒绝）
python -m scripts.run_experiment --config configs/smoke.yaml --demo --phase final-test --horizon 24h --frozen-run results/runs/<冻结run目录>
```

## 4. 运行产物

```text
results/runs/<UTC时间>_<随机短ID>/
├── manifest.json            # 状态、Git SHA、环境、配置哈希、seed、synthetic/research_result
├── data_audit.json          # 计量边界、单位、时区、标签来源、数据质量
├── split_audit.json         # 锚点、每个 Dk 起报时间、窗口可用性、重叠说明
├── batch_summary.json       # 批次形状/字段/掩码（finalize 阶段显式跳过）
├── model_state/state.json   # 可恢复的模型/配置状态（无拟合基线也必须冻结）
├── frozen_before_test.json  # 冻结摘要与哈希，写入后不可原地更新
├── predictions.parquet      # 逐点预测（16 列固定 schema，可复算指标）
├── metrics.json / metrics.csv
├── test_report.json         # 冻结校验、重载、指标复算一致性
└── figures/                 # 可选图（当前骨架默认不画图）
```

## 5. 防泄漏与诚实性（硬约束）

1. **未来信息**：协变量/已知未来特征的可用时刻晚于起报时刻 → 契约层 `LeakageError`；上下文时间戳不得等于或晚于起报时刻。
2. **标签截止**：监督样本要求上下文与标签都在训练截止内；`training_window_audit` 对 D14 + 1344 上下文 + 672/96 标签直接返回 0 个可行窗口。
3. **锚点统一**：96 个连续 15min 槽 = 一个 trace 日；某域开头缺测不会各自重编 D1，内部缺测不改变后续日号。
4. **不掩盖缺失**：缺失值保留掩码（`past_power_valid` / `observed_mask`），绝不把缺失当作 0 kW；基线在历史不完整时明确拒绝窗口。
5. **不伪造标签**：`synthetic_simulated` 永远不会被写成 `measured`；利用率/节点功率不会被当成设施 kW。
6. **不覆盖证据**：run 目录不可覆盖；冻结凭据二次写入报错；失败写 `status=failed`。

## 6. 目录结构

```text
configs/{base,smoke}.yaml, protocols/, datasets/, models/
data/{raw,interim,processed}/       # 内容不入 Git
docs/{REBUILD_AUDIT,ARCHITECTURE,DATA_CONTRACT,EXPERIMENT_PROTOCOL,TODO_RESEARCH}.md
results/                            # 运行产物，不入 Git
scripts/{run_experiment,prepare_data,inspect_data}.py
src/data/{contracts,registry,loaders,adapters,preprocessing,features,splitting,windows,synthetic}.py
src/models/{interfaces,baselines,power_backbone,resource_encoder,target_mapping,fusion}.py
src/training/{lifecycle,runner}.py
src/evaluation/{metrics,backtest,artifacts}.py
tests/test_{contracts,unlabeled,registry,windows,protocols,leakage,baselines,lifecycle,smoke}.py
```

## 7. 科研定位与边界

- 主任务：**设施级总有功功率（kW）** 的未来绝对功率序列预测，正式输出 15min；
- 主路径（A，`power_backbone`）必须能仅凭历史 kW 独立完成预测 —— 本轮用两个真实基线证明工程完整性；
- 辅助路径：B 无标签运行指标预训练、C 目标域有监督功率映射、D 主辅融合 —— 本轮**只有接口与配置语义**，不得声称已实现或已有精度收益；
- 目标独有字段：保留、登记、`target_only_local_weight=1.0`（仅为局部路由语义），全局融合权重未决为 `None`；
- 数据来源角色与审计状态见 `configs/datasets/registry.yaml`；本轮**没有任何真实来源**可用于正式设施 kW 结果。

## 8. 文档

| 文档 | 内容 |
|---|---|
| [docs/REBUILD_AUDIT.md](docs/REBUILD_AUDIT.md) | 从 `main` 重建的分支/SHA、逐路径清理清单、复用与重写理由、数据保护证据 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层架构、模块状态矩阵、依赖方向、设计决策、扩展点 |
| [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) | 四层数据契约、单位换算、批次语义、注册表对齐规则、必须失败的输入 |
| [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | 四阶段、trace 日号、起报/标签、可选预设、防泄漏检查、指标口径、产物 |
| [docs/TODO_RESEARCH.md](docs/TODO_RESEARCH.md) | 未实现研究模块、真实数据阻断点、方法学待决事项、遗留风险 |
| [docs/ACMETRACE_TARGET_V2.md](docs/ACMETRACE_TARGET_V2.md) | v2 数据层：AcmeTrace Seren 目标域（cluster-IT 标签 + 运行特征 + job 活动）接入报告 |
| [docs/ACMETRACE_OPS_AUDIT_V2.md](docs/ACMETRACE_OPS_AUDIT_V2.md) | v2 轻量运行指标审计：4 个 NODE_* 指标 + IPMI 配对 + job trace 证据与诚实边界 |

## 9. 下一阶段

1. 真实设施总表 kW 数据接入与计量边界审计（当前无可用来源）；
2. 无标签运行指标表征预训练（`resource_encoder` 真实训练）；
3. 目标域有监督功率映射（`target_mapping`）；
4. 主辅融合策略与冻结选型（`fusion`，当前全局权重未决）；
5. 4h / 8h / 3d 正式起报协议冻结；
6. 多步监督模型（含 LightGBM 合法特征设计）与训练样本量/历史长度分离研究。

## 10. v2 数据层：AcmeTrace Seren 目标域（本分支新增，2026-10-28）

`research/power-transfer-v2` 在 Stage 1+2 基线上接入完整 AcmeTrace 快照
（`E:/Data/AcmeTrace`，HF revision `a484f776…`，153 文件 / 约 75 GiB），形成“带运行指标的
目标域”供新一轮研究使用。**注意：AcmeTrace 没有设施级总表功率**，Seren 的标签是
**cluster-IT 功率**（IPMI 服务器功率按上报节点求和），不是设施 kW；单位由量级推断为 W、
站点时区未官方核验——这些限制在元数据与注册表中显式保留。

```powershell
# 1) 功率层（读约 7.6 GB 功率 CSV；轻量模式跳过 80.8 GB（75.2 GiB）全树 SHA256）
& .\.venv\Scripts\python.exe -m scripts.audit_acmetrace_power --skip-hash

# 2) 目标域装配（Seren cluster-IT 标签 + NODE_*/GPU 运行指标 + job trace，15 min）
& .\.venv\Scripts\python.exe -m scripts.prepare_acmetrace_target

# 3) 轻量运行指标审计（白名单，仅目标域使用的 4 个 NODE_* 指标 + IPMI + job trace）
& .\.venv\Scripts\python.exe -m scripts.audit_acmetrace_ops --clusters seren --metrics NODE_CPU_UTILIZATION NODE_MEMORY_UTILIZATION NODE_IB_RECEIVE NODE_IB_SEND

# 4) 补充只读证据（逐节点功率-指标 Pearson + 集群级 job/GPU 利用率 linkage）
& .\.venv\Scripts\python.exe -m scripts.analyze_acmetrace_linkage

# 边界烟雾（每文件只读 1 个 chunk，结果不得引用为全量覆盖）
& .\.venv\Scripts\python.exe -m scripts.prepare_acmetrace_target --max-chunks 1 --no-write
```

产物（`data/processed/`，均不入 Git）：

| 文件 | 内容 |
|---|---|
| `acmetrace_power_15min.parquet` | 全部功率层级（Seren GPU/服务器/cluster-IT、Kalos GPU）的 15 min 区间均值 + 掩码 |
| `acmetrace_target_power.parquet` | 目标域标签：Seren cluster-IT kW（`power_valid` / `coverage_ratio`） |
| `acmetrace_target_features.parquet` | 目标域运行特征长表（节点 CPU/内存/IB、GPU 利用率、作业活动） |
| `acmetrace_target_metadata.json` | 标签语义、特征注册、时间/时区、来源文件、就绪状态 |
| `acmetrace_target_quality.json` | 覆盖率/缺失统计与功率–特征描述性相关性（只作证据，不作收益声明） |

新增模块：`src/data/acmetrace_power.py`（功率审计，自 v1 分支移植）、`src/data/acmetrace_ops.py`
（运行指标/配对/作业审计，自 v1 分支移植 + 轻量 `--metrics` 白名单）、`src/data/acmetrace_target.py`
（v2 目标域装配）、`scripts/analyze_acmetrace_linkage.py`（补充只读证据：逐节点功率-指标 Pearson
与集群级 job/GPU 利用率 linkage）。数据角色见 `configs/datasets/registry.yaml` 的 `acmetrace_seren`
（`target_supervised`）与 `acmetrace_kalos`（`candidate`）；接入细节与限制见
[docs/ACMETRACE_TARGET_V2.md](docs/ACMETRACE_TARGET_V2.md) 和
[docs/ACMETRACE_OPS_AUDIT_V2.md](docs/ACMETRACE_OPS_AUDIT_V2.md)。
