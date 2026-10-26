# PowerForecast

**面向算力中心（数据中心 / 超算中心）设施级有功功率（kW）预测的研究工程骨架**

> 本分支 `research/power-transfer-v1` 是从 `origin/main` 重建的**新工程骨架**，只包含数据契约、时间协议、两个真实可运行的 CPU 基线、评价与冻结生命周期。
> **本轮不包含**任何迁移学习 / 无标签预训练 / 主辅融合的算法实现，也不产出真实设施 kW 的科研结论。合成数据结果一律标记 `synthetic=true`、`research_result=false`。

## 1. 本轮交付是什么

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

## 9. 下一阶段

1. 真实设施总表 kW 数据接入与计量边界审计（当前无可用来源）；
2. 无标签运行指标表征预训练（`resource_encoder` 真实训练）；
3. 目标域有监督功率映射（`target_mapping`）；
4. 主辅融合策略与冻结选型（`fusion`，当前全局权重未决）；
5. 4h / 8h / 3d 正式起报协议冻结；
6. 多步监督模型（含 LightGBM 合法特征设计）与训练样本量/历史长度分离研究。
