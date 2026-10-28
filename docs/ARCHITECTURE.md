# 架构说明（ARCHITECTURE）

> 本文件描述**代码实际结构**。凡标注"仅接口"的模块，本轮没有真实算法实现，调用会明确失败。

> **V2.0 阶段一/二补充（当前主线）**：在下方骨架之上新增了两条真实数据/模型链路，
> 详细说明见 `docs/STAGE1_TARGET_REPORT.md`、`docs/STAGE1_SOURCE_REPORT.md`、
> `docs/FEATURE_ALIGNMENT_REPORT.md`、`docs/STAGE2_MAIN_MODEL.md`：
>
> ```text
> 真实目标域: data/raw/NLR → src/data/target_power.py → data/processed/processed_target_power.parquet
>                                └→ scripts/analyze_power.py → results/analysis/stage1/power_characteristics/
> 真实源域:   data/raw/M100  → src/data/source_features.py → data/processed/processed_source_features.parquet
>                                └→ FieldRegistry.align → configs/datasets/feature_registry.yaml
> 主模型 A:   历史 kW → src/models/foundation/{chronos2,tirex2}.py → src/models/power_main/ensemble.py(0.5/0.5)
>                        → scripts/run_power_main.py → results/runs/<run_id>/{predictions,metrics,...}
> ```
>
> 第三~五阶段（无标签预训练、目标域映射、主辅融合）仍未实现，仅保留接口与数据基础。

## 1. 一句话架构

```text
合成/合法数据源
   │  loaders / adapters / preprocessing（只读原始、版本化写出）
   ▼
DatasetSpec ── FeatureSpec / TargetSpec（语义、单位、可用时刻、标签诚实性）
   │  registry：跨源字段对齐、目标独有字段路由（全局融合权重 = None）
   ▼
splitting：trace 锚点 / 日号 / 协议 / 训练可用性
   │  windows：ForecastInputs / ForecastBatch / PretrainBatch（构造即防泄漏）
   ▼
models：power_backbone(仅接口) + 两个真实 CPU 基线 + resource/target_mapping/fusion(仅接口)
   │  training.runner：分批推理 + 形状/有限性/单位校验
   ▼
evaluation：MAPE/WAPE/MAE/RMSE + 分层报告 + predictions/metrics 落盘与复算
   │  training.lifecycle：freeze → frozen_before_test.json → 哈希校验 → 重载 → 才允许评分
   ▼
results/runs/<UTC>_<id>/（不可覆盖，synthetic/research_result 显式标记）
```

## 2. 目录与职责

```text
PowerForecast/
├── configs/
│   ├── base.yaml                 通用实验配置（可继承）
│   ├── smoke.yaml                合成 smoke 配置（override base）
│   ├── protocols/
│   │   ├── default.yaml          通用协议（无任何数据集日号）
│   │   ├── fixed_24h.yaml        可选历史预设 A
│   │   └── fixed_7d.yaml         可选历史预设 B
│   ├── datasets/
│   │   ├── synthetic.yaml        合成数据集参数
│   │   └── registry.yaml         来源角色/审计状态/标签诚实性登记
│   └── models/baselines.yaml     模型注册表（implemented vs planned）
├── data/{raw,interim,processed}/ 数据目录（内容不入 Git）
├── docs/                         本文件 + 数据契约 + 实验协议 + 清理审计 + TODO
├── results/                      运行产物（不入 Git）
├── scripts/
│   ├── run_experiment.py         唯一正式实验 CLI（四阶段）
│   ├── prepare_data.py           数据工具：合成数据落盘 + sidecar
│   └── inspect_data.py           数据工具：schema/样例/质量检查
├── src/
│   ├── utils.py                  路径/配置/哈希/JSON/seed/git/环境
│   ├── data/
│   │   ├── contracts.py          DatasetSpec/FeatureSpec/TargetSpec + Batch 契约
│   │   ├── registry.py           字段语义对齐 + 目标独有字段 + 来源登记校验
│   │   ├── loaders.py            csv/parquet 只读、分块、inspect、拒绝写 data/raw
│   │   ├── adapters.py           原字段→规范字段、单位换算、时区/去重
│   │   ├── preprocessing.py      质量报告、因果填充、版本化落盘 + sidecar
│   │   ├── features.py           仅用过去/日历的特征构造
│   │   ├── windows.py            OriginPlan、Forecast(Pretrain) 批次构造
│   │   ├── splitting.py          协议解析、trace 网格、训练可用性审计
│   │   └── synthetic.py          31+ trace 日合成数据 + 两个异构无标签源
│   ├── models/
│   │   ├── interfaces.py         ModelStatus / 四个 Protocol / 注册表
│   │   ├── baselines.py          seasonal_naive、three_day_mean（真实实现）
│   │   ├── power_backbone.py     主路径抽象（仅接口）
│   │   ├── resource_encoder.py   无标签预训练抽象（仅接口）
│   │   ├── target_mapping.py     目标域功率映射抽象（仅接口）
│   │   └── fusion.py             主辅融合抽象（仅接口）
│   ├── training/
│   │   ├── lifecycle.py          四阶段约束、冻结、哈希校验、重载
│   │   └── runner.py             分批推理与输出校验（截断显式记录）
│   └── evaluation/
│       ├── metrics.py            MAPE/WAPE/MAE/RMSE + 近零阈值策略
│       ├── backtest.py           per-origin/per-lead/per-day/daily-mean + 重叠标记
│       └── artifacts.py          RunDirectory、predictions.parquet、manifest
└── tests/                        T01–T22 对应测试（113+ 用例）
```

## 3. 模块状态矩阵（真实实现 vs 仅接口）

| 模块 | 状态 | 证据 |
|---|---|---|
| `data/contracts.py` | ✅ 真实实现 | 构造即校验；`tests/test_contracts.py` |
| `data/registry.py` | ✅ 真实实现 | 语义对齐、独有字段、全局权重 `None`；`tests/test_registry.py` |
| `data/synthetic.py` | ✅ 真实实现 | 31+ 日、F=0/多变量、两个异构无标签源、`synthetic_simulated` 标签 |
| `data/splitting.py` | ✅ 真实实现 | 锚点日号、协议、训练可用性；`tests/test_protocols.py` |
| `data/windows.py` | ✅ 真实实现 | 防泄漏批次；`tests/test_windows.py`、`tests/test_leakage.py` |
| `data/acmetrace_power.py` | ✅ 真实实现（v2 分支） | AcmeTrace 完整性/功率层级/15 min 适配；`tests/test_acmetrace_power.py`；`--skip-hash` 模式把未核验哈希标为 unverified 而非 mismatch |
| `data/acmetrace_ops.py` | ✅ 真实实现（v2 分支） | 运行指标/功率-指标配对/作业审计；`--metrics` 白名单支持轻量局部审计；`tests/test_acmetrace_ops.py` |
| `data/acmetrace_target.py` | ✅ 真实实现（v2 分支） | Seren cluster-IT 目标域装配（标签 + 运行特征 + 作业活动，15 min，mask-first）；`tests/test_acmetrace_target.py` |
| `models/baselines.py` | ✅ 真实实现 | 两个免拟合 CPU 基线；`tests/test_baselines.py` |
| `evaluation/*` | ✅ 真实实现 | 指标 + 分层 + 落盘；`tests/test_baselines.py`、`tests/test_smoke.py` |
| `training/lifecycle.py` | ✅ 真实实现 | 冻结/重载/哈希；`tests/test_lifecycle.py` |
| `models/power_backbone.py` | 🚧 **仅接口** | `status=PLANNED`，调用抛 `NotImplementedError` |
| `models/resource_encoder.py` | 🚧 **仅接口** | 同上 |
| `models/target_mapping.py` | 🚧 **仅接口** | 同上 |
| `models/fusion.py` | 🚧 **仅接口** | 同上；无 `0*aux`、无固定权重 |
| LightGBM/深度模型 | 🚧 **planned** | 注册表有名称但不可被选中 |

## 4. 依赖方向（单向，避免循环）

```text
utils  ←  contracts  ←  registry / features / loaders / adapters / preprocessing
                          ←  splitting  ←  windows
                                              ←  models.interfaces ← baselines
                                              ←  evaluation.*
                                              ←  training.*
                                              ←  scripts.run_experiment
```

- `contracts.py` 只依赖 numpy/pandas，不依赖任何业务模块；
- `windows.py` 依赖 `contracts` + `features` + `splitting`，不依赖模型；
- 模型只接收 `ForecastInputs`，无法接触标签；
- CLI 是唯一把数据、模型、评估、生命周期串起来的地方。

## 5. 关键设计决策

1. **契约即边界**：无标签/有标签、单位、计量范围、可用时刻不是文档约定，而是构造函数校验；非法输入根本构造不出批次。
2. **标签诚实性内建**：`synthetic=True ⇔ label_origin=synthetic_simulated`；合成结果 manifest 必带 `synthetic=true, research_result=false`。
3. **物理语义分离**：`active_power` 与 `power_utilization` 是不同目标类型；节点/设施是不同 scope；`require_facility_active_power_kw()` 是正式主任务守门人。
4. **目标独有字段不丢**：注册表保留字段与来源，`target_only_local_weight=1.0` 只是局部路由语义，`global_fusion_weight=None` 表示研究未决。
5. **时间协议与数据解耦**：通用引擎 + 可选预设 YAML；底层代码不含 D14/D22 字面量。
6. **无标签源不伪造时间序列**：只有事件级表格的源登记为候选，不强制生成假 `PretrainBatch`。
7. **冻结是硬门槛**：`final-test` 无冻结凭据直接拒绝；冻结凭据包含配置/协议/数据/模型状态哈希，跨 run 重载会校验。
8. **不掩盖错误**：输出形状/有限性/单位校验失败即报错；非负截断必须显式配置并记录次数。

## 6. 数据流（有标签主路径）

```text
frame(UTC 15min, canonical cols)
   └─ DatasetSpec（features + target + anchor + timezone）
        └─ TraceGrid(anchor)  → Dk 日号 / origin_of_day(k)=anchor+k*96
             └─ Protocol(context_points, horizons, origins, cutoffs)
                  └─ OriginPlan（可行 origin + 被丢弃原因 + 重叠标记）
                       └─ ForecastBatch
                            inputs.past_power_kw [B,L] + 掩码
                            inputs.past_covariates [B,L,F]（可用时刻<=origin）
                            inputs.future_known [B,H,K]（仅日历，available_at=origin）
                            targets_kw [B,H]（仅 active_power/kW）
```

## 7. 生命周期与产物

```text
development : 构建批次 → 允许的拟合/诊断 → 评分（不作为正式结果）
finalize    : 构建几何计划（不构建测试批次、不读测试标签）→ 冻结 → 结束
final-test  : require_frozen_record → 哈希校验 → 重载 → 预测 → 评分 → 复算一致性检查
smoke       : 单命令内部依次 finalize → final-test（仅合成数据）
```

产物目录见 `docs/EXPERIMENT_PROTOCOL.md` §8；关键门槛：

- `manifest.json` 状态只有 `completed` / `failed`，失败写 `error`；
- `predictions.parquet` 的 16 列严格固定，可据 `actual/predicted/valid_label` 复算指标；
- CLI 完成前会用落盘预测复算指标，与 `metrics.json` 不一致则拒绝标记 completed；
- `frozen_before_test.json` 写入后不可原地更新，第二次冻结直接报错。

## 8. 扩展点

| 想做的事 | 应该落在哪 |
|---|---|
| 新增数据源 | `loaders/adapters` + `registry.yaml` 登记角色与标签语义；正式使用前完成审计 |
| 新增模型 | 实现 `BasePowerForecaster`，在 `models/interfaces.py` 注册并写测试；未实现不得注册为 implemented |
| 新增时间协议 | `configs/protocols/*.yaml`；不修改 `splitting.py/windows.py` 的通用逻辑 |
| 新增指标 | `evaluation/metrics.py`（保持可复算 + 单位/阈值口径） |
| 新增产物 | `evaluation/artifacts.py` 的 `RunDirectory`（不可覆盖语义） |
| 无标签预训练 | `resource_encoder.py`（`PretrainBatch` 入口，禁止标签） |
| 映射/融合研究 | `target_mapping.py` / `fusion.py`；完成训练/选择/冻结后才进入正式注册表 |
