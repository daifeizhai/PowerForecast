# 实验时间协议（EXPERIMENT_PROTOCOL）

> 实现位置：`src/data/splitting.py`（协议解析 + trace 网格 + 训练可用性）、`src/data/windows.py`（起报计划 + 批次构造）、`configs/protocols/`。
> 原则：**通用引擎可配置，固定 D14/D22 与 24h/7d 起报只作为可选历史预设，不在底层代码里硬编码。**

## 1. 四个运行阶段

| 阶段 | 作用 | 可拟合参数 | 可读取正式测试真值用于选择 |
|---|---|---|---|
| `development` | 预检、开发诊断、参数选择 | 仅在当期合法训练资料上 | 不可以 |
| `finalize` | 按冻结配置做最终合法训练/配置，保存状态并生成冻结凭据 | 可以，截止由配置指定 | 不可以 |
| `final-test` | 从冻结凭据**恢复**模型，在测试起报处预测并评分 | 不可以 | 只能预测完成后评分 |
| `smoke` | 单命令合成数据全流程（准备→冻结→重载→测试） | 仅 demo 规则 | 不产生正式科研结果 |

强制逻辑（实现于 `scripts/run_experiment.py` + `src/training/lifecycle.py`）：

1. `finalize` 不读取测试标签，不据测试误差改模型/字段/窗口/权重；
2. `final-test` 必须指向已存在的冻结运行目录，校验模型/配置/协议/数据哈希后加载才允许推理；没有 `frozen_before_test.json` 直接拒绝；
3. `smoke` 可在一条命令内完成"冻结→重载→评分"，但 manifest 必含 `synthetic=true`、`research_result=false`；
4. `requires_fit=False` 的基线同样冻结：冻结内容至少包含基线类型、算法版本、配置、字段与目标契约、代码摘要；
5. 冻结后若需改代码/模型/数据契约/选择标准，必须开启**新的 run**，不得覆盖旧 run 的证据；
6. 测试阶段允许把"起报前已发布的真实历史观测"喂给 `predict()`；这与"用测试真值调参"是两件事。

## 2. 通用可配置协议（`configs/protocols/default.yaml`）

```yaml
protocol_id: configurable_v1
frequency: 15min
points_per_day: 96
trace_anchor_policy: dataset_explicit
context_points: 1344          # 可改 96 / 672 / 1344 / ...
horizons: {4h: 16, 8h: 32, 24h: 96, 3d: 288, 7d: 672}
origin_mode: explicit
development_origins: []       # 通用协议不预置任何日号
test_origins: []
train_cutoffs: {}             # 由数据集/模型/阶段分别规定
availability_policy: strict
protocol_type: configurable
default_horizon: 24h
```

可独立设置：数据集锚点、预测粒度（`points_per_day`/`frequency`）、历史长度 `context_points`、预测时长 `H`、训练截止、开发起报、测试起报、特征可用性策略、冻结/复用策略。

## 3. trace 日号与锚点（防错位的关键）

- 每个目标数据集由**共同原始时间参考**确定唯一 `trace_anchor_timestamp`，所有域共用；
- 固定 **96 个连续 15min 槽 = 一个 trace 日**，不要求等于当地自然日，也不因 DST 变成 23/25 小时；
- `D1` = 从锚点起的第 1 个 96 槽区间；`Dk` 结束的起报时刻 = **第 k+1 个区间起点**（`origin_of_day(k) = anchor + k×96 槽`）；
- 内部缺测不改变锚点与日号：不能通过删除缺测行重新凑齐 96 点，也不能把原始首时刻平移至当地午夜；
- 某个域开头缺测时，**不能**各自把第一条有效记录重编为 D1；
- 时间索引同时保存真实时间戳与 trace 槽号/日号；时区未核验的源保留"不确定"标志，不自动宣称 UTC。

## 4. 起报、上下文与标签

| 概念 | 定义 | 实现 |
|---|---|---|
| context | 起报前紧邻的 `context_points` 个槽 | `past_timestamps[-1] = origin - 1 步` |
| origin | 第一个待预测区间起点，也是信息截断时刻 | `origin_timestamps == target_timestamps[:,0]` |
| labels | 从 origin 起的 `H` 个槽 | `H` 由协议 `horizons` 决定 |
| 标签日 | 起报 Dk → 标签 D(k+1)… | fixed_24h：D22…D30 → D23…D31 |

## 5. 可选历史预设（只有显式选择才生效）

### 5.1 `configs/protocols/fixed_24h.yaml`

| 项目 | 设置 |
|---|---|
| 预测时长 | 96 点（24h） |
| 默认历史 | 1344 点（14 个 trace 日） |
| 开发起报 | D14, D16, D18, D20 |
| 测试起报 | D22, D23, …, D30（9 条） |
| 测试标签 | D23 … D31 |
| LightGBM 截止（仅配置数据） | 开发 14 / 最终 22 |
| 测试阶段 | 模型/预处理/参数固定；只滚动使用各起报前许可历史 |

### 5.2 `configs/protocols/fixed_7d.yaml`

| 项目 | 设置 |
|---|---|
| 预测时长 | 672 点（7d） |
| 默认历史 | 1344 点 |
| 开发起报 | D14, D15 |
| 测试起报 | D22, D23, D24 |
| 标签覆盖 | 最晚到 D31 结束 |
| LightGBM 截止（仅配置数据） | 开发 14 / 最终 22 |
| 重叠 | 三条预测曲线大量复用同一真值 → **不得**当作 2016 个独立观测 |

**重要统计事实**：7d 预设的 3 条测试曲线相互重叠。逐起报结果分别保存；可以另报按唯一时间戳去重的描述性统计，但不得用去重口径偷换已声明主指标，也不得据重叠误差声称"高度显著"。

### 5.3 4h / 8h / 3d

- 引擎对全部五个 H 都提供 `[B,H]` 通用接口；
- 正式开发/测试起报日**本轮不擅自决定**；
- 只用于合成数据的 smoke 标记 `protocol_type=demo`，不冒充正式协议；
- 3d 可预留非重叠窗口评分能力，具体方式须后续冻结。

### 5.4 预设的性质

D14/D22 与固定 24h/7d 起报是**可选历史标准协议实例**，写入独立 YAML 并通过专门单元测试；通用引擎与 `Protocol` 类不写死任何日号。LightGBM 的 D14/D22 只是该预设的配置数据，代码不会把它强加给其他模型。

## 6. 训练可用性与防泄漏检查

1. 一个监督样本合法，当且仅当**全部上下文观测**与**全部未来标签**都符合该训练阶段截止与可用时刻规定；必须同时检查标签区间结束、标签 `available_at`、训练截止三者关系，而不是只比时间戳；
2. **历史长度 ≠ 训练数据量**：完整 14 天上下文 + 完整 7 天标签在 D14 截止前可训练窗口数 = 0；完整 24h 标签同样为 0（`training_window_audit` 返回 `insufficient_training_windows`）；
3. 窗口不足时返回结构化 `insufficient_training_windows` 或使用事先声明的免拟合基线，不静默缩短标签、不拿 D15 之后的标签训练、不用测试值补窗；
4. 开发起报 D14/D15 的 7d 真值与 D22 可用的训练期标签可能重叠：若同一标签已用于最终拟合，该结果只能标记 `development_diagnostic`；
5. 开发阶段选参与其自身标签可用截止一致，不能用 D20 之后才产生的真值宣称"在 D14 已完成选参"；
6. 测试阶段在第一次评分前冻结全部参数/选择策略/预处理/融合配置；D22 之后滚动起报可接受起报前新观测，但**不得重训/重选参**；
7. 外部无标签预训练源的时间范围不必沿用目标域 D14/D22，但须遵守事前选定的源/目标隔离与可用性边界，且不能把最终留出目标域当作预训练语料；
8. 审计字段：`origin_timestamp`、`last_context_timestamp`、`max_input_available_at`、`train_label_end`、`max_train_label_available_at`、`train_cutoff`、`model_freeze_time`、`test_start_marker`。越界必须报错并给出域、字段、时间、原因。

## 7. 评价指标与报告层级

**主指标 MAPE（百分数）**：

- 分母近零剔除策略必须预先声明并记录：`mape_threshold`、`mape_n`、`mape_excluded`；
- 不得把近零真值静默改成 1，也不得随意更改阈值后不记录。

**辅助指标**：

- `WAPE = 100 × Σ|y−ŷ| / Σ|y|`（百分数；分母近零时标记不可定义）；
- `MAE = mean(|y−ŷ|)`（kW）；
- `RMSE = sqrt(mean((y−ŷ)²))`（kW）。

报告层级：

| 层级 | 说明 |
|---|---|
| `per_origin` | 每次起报 H 点的 MAPE/WAPE/MAE/RMSE |
| `per_lead` | h = 1…H 的误差 |
| `aggregate` | 同一协议下预声明的总体汇总 |
| `per_forecast_day` | H 为 96 整数倍时，按 96 点一组分别评分 |
| `daily_mean_power` | 日均功率评价；4h/8h 的预测窗均值只能叫 `window_mean_power`，**不得**称为日均 |

本轮基线只有预定义算法，**不通过比较测试误差动态选赢家**；因此不存在"用测试集挑模型"的行为。不要把 `100%−MAPE` 未加定义地称为设施预测"准确率"。

## 8. 运行目录与产物

每次运行创建全新、不可覆盖目录：

```text
results/runs/<UTC时间>_<随机短ID>/
├── manifest.json            # 状态、代码 SHA、环境、配置、seed、目标来源、synthetic/research_result
├── data_audit.json          # 计量范围、单位、时间、标签性质
├── split_audit.json         # 锚点、每个 Dk 实际起报时间、窗口可用性、重叠说明
├── batch_summary.json       # 批次形状/字段/掩码概况
├── model_state/             # 有参数则存参数；无拟合基线存可恢复配置状态
├── frozen_before_test.json  # 冻结摘要与 SHA256（写入后不原地更新）
├── predictions.parquet      # 可复算逐点数据
├── metrics.json / metrics.csv
├── test_report.json         # 冻结校验、重载、指标复算一致性
└── figures/                 # 可选图，标注 synthetic
```

`predictions.parquet` 列：`run_id, dataset_id, domain_id, scope, label_origin, synthetic, model, protocol_id, horizon, origin_timestamp, timestamp, lead, actual, predicted, unit, valid_label`。
每个预测可依据 `actual/predicted/valid_label` 复算指标，并与 `metrics.json` 一致（CLI 会在完成前校验，不一致即拒绝标记 completed）。

失败时 manifest 记录 `status=failed` 与错误信息，不写"完成"。

## 9. 可复现性

相同 seed、数据与配置应得到相同的逐点预测与指标（允许运行目录名、耗时、创建时间不同）。合成数据生成器 & 基线均为确定性实现；CLI 在构建数据前调用 `seed_everything(seed)`。

## 10. 命令

```powershell
# 安装与测试
python -m pip install -e .
python -m pytest -q

# 最小合成闭环（synthetic=true, research_result=false）
python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 24h
python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 7d

# 可选历史预设（仅验证配置能力）
python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 24h --protocol configs/protocols/fixed_24h.yaml
python -m scripts.run_experiment --config configs/smoke.yaml --demo --horizon 7d  --protocol configs/protocols/fixed_7d.yaml

# 数据工具
python -m scripts.prepare_data --config configs/datasets/synthetic.yaml
python -m scripts.inspect_data --config configs/datasets/synthetic.yaml
```

注意：最小的 smoke 命令**不依赖**人工预先执行预处理；`--demo` 会在 run 入口安全生成合成数据，且只写入 `results/` 下的新版本目录。
