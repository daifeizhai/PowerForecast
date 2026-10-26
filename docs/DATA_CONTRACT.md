# 数据契约（DATA_CONTRACT）

> 实现位置：`src/data/contracts.py`（语义与构造校验）、`src/data/registry.py`（跨源对齐与独有字段）。
> 本文件描述**代码实际执行的规则**；契约由构造函数强制执行，而不是靠约定或文档自觉。

## 1. 四层语义：Dataset / Domain / Feature / Target

| 对象 | 代码 | 回答的问题 |
|---|---|---|
| Dataset | `DatasetSpec` | 这份数据是什么角色、来自哪个版本、是否可用于正式设施 kW 任务 |
| Domain | `DatasetSpec.site_id` / `domain_id` | 同一数据集中哪个站点/域，锚点与日号必须全数据集中共享 |
| Feature | `FeatureSpec` | 每个输入字段的规范语义、单位、统计方式、可用时刻、可信状态 |
| Target | `TargetSpec` | 标签测的是什么、在哪一层计量、什么单位、从哪来、是否合成 |

### 1.1 `FeatureSpec`

| 字段 | 含义 | 规则 |
|---|---|---|
| `raw_name` | 原始字段名 | 非空；同一数据集内不得重复 |
| `canonical_name` | 规范语义名（唯一 join key） | 非空；`DatasetSpec.features` 的 key 必须等于它 |
| `unit` | 物理单位 | 非空；功率 `W/kW/MW/GW`，无量纲 `ratio/percent/%/...` |
| `statistic` | `instantaneous` / `interval_mean` / `interval_integral` / `event_state` / `lifecycle_summary` | 事例后统计量（`lifecycle_summary`）**不能**声明为 `future_known` |
| `physical_range` | 合法物理范围 `(low, high)` | 有限且 `low < high`；越界值由预处理登记，不静默裁剪 |
| `native_frequency` | 原生采样周期 | 允许与正式 15min 不同（无标签源尤其如此） |
| `availability_lag` | 观测区间结束到可用于决策的延迟 | ≥ 0；未知延迟必须写进 `notes` 作为假设，不得默认声称实时 |
| `future_known` | 是否为起报时已知的未来特征 | 默认只有审计过的日历信息为 `True` |
| `missing_policy` | `mask` / `reject` | 缺失必须保留掩码，不能编码为"真实测得 0" |
| `source` / `audit_status` / `notes` | 来源、审计状态、假设披露 | `verified` / `partial` / `unverified` |

**同名字段不同单位或统计方式不得仅凭名字合并**：见 §4 的对齐规则。

### 1.2 `TargetSpec`

| 字段 | 说明 |
|---|---|
| `kind` | `active_power`（真实功率）或 `power_utilization`（无量纲代理） |
| `scope` | `facility` / `power_domain` / `rack` / `node` / `server` / `it` |
| `unit` | `active_power` 必须是功率单位；`power_utilization` 必须是无量纲单位 |
| `label_origin` | `measured` / `component_sum` / `energy_converted` / `estimated` / `power_utilization` / `synthetic_simulated` |
| `measurement_boundary` | 计量边界（必填，例如"设施总表"vs"节点 PSU 输入"） |
| `sampling_semantics` | 采样语义（必填，例如"15 分钟区间平均"vs"5 分钟瞬时采样"） |
| `available_at_lag` | 标签可用的额外延迟（≥ 0） |
| `synthetic` | 合成演示标记 |

构造时强制：

1. `synthetic=True` ⇔ `label_origin=synthetic_simulated`（二者互为必要条件）；
2. `active_power` 不允许 `label_origin=power_utilization`；
3. `power_utilization` 不允许声明功率单位；
4. `active_power` 只能使用功率单位。

### 1.3 `DatasetSpec` 与角色合法性

| 角色 | `target` | 是否可进入正式 facility-kW 监督训练 |
|---|---|---|
| `unlabeled_pretrain` | **必须为 `None`** | 不可以（无标签源声明标签会直接报错） |
| `target_supervised` | **必须声明 `TargetSpec`** | 仅当 `kind=active_power`、`scope=facility`、单位为功率且标签来源合法时 |
| `final_holdout` | 可声明也可为 `None` | 不可以；仅在冻结后进行许可的最终评估 |
| 合成演示 | `synthetic=True` | 只用于演示；`research_result` 恒为 `false` |

`target_supervised` 还必须声明 `timezone`、`trace_anchor_timestamp`、`interval_semantics`、`availability_semantics`。
`timezone_verified=False` 是合法状态（保留不确定性），但任何依赖本地时间的操作都会通过 `spec.require_verified_timezone()` 明确阻断，不会自动假设 UTC。

`require_facility_active_power_kw(target, allow_synthetic=False)` 是正式主任务的守门函数：

- 拒绝 `kind != active_power`（利用率不是 kW）；
- 拒绝 `scope != facility`（节点/服务器功率属于另立的辅助任务）；
- 拒绝非功率单位；
- 默认拒绝 `synthetic_simulated`（合成数据不能产出正式研究结论）。

### 1.4 单位与标签换算

- `to_kw(values, unit)`：`W→1e-3`、`kW→1`、`MW→1e3`、`GW→1e6`；
- 无量纲（`ratio`/`percent`/…）调用 `to_kw` **必须抛 `PhysicalUnitError`**：利用率转 kW 需要外部容量标定，不能由本工程猜测；
- `ForecastBatch.targets_kw` 要求声明的目标单位就是 kW；W/MW 必须先经 `to_kw` 换算再建批次，避免"名为 kW 实为 W"；
- `component_sum` 与 `measured` 是两个不同的 `label_origin`，序列化后保持区分，不会被"升级"为实测。

## 2. 无标签批次 `PretrainBatch`

```text
features          float[B,L,F]      # 运行指标，不含 kW
observed_mask     bool[B,L,F]       # True = 真实可用观测
timestamps        datetime64[B,L]   # UTC
available_at      datetime64[B,L,F] # 该观测可用于决策的最早时刻
feature_names     tuple[str, ...]   # 本批次自己的字段语义
domain_ids        tuple[str, ...]   # [B]
sampling_metadata dict              # native_frequency / time_delta 等
```

强制规则：

1. **没有** `targets_kw` / `targets` / `future_kw` / `power_labels` 属性；
2. `F` 可以因源而异，允许分批进入编码器，不要求硬对齐成固定维度；
3. `observed_mask=True` 处必须有限；`False` 处保留 NaN，缺失不编码为 0；
4. `available_at ≥ timestamps`（观测不能先于其发生而被使用）；
5. `require_pretrain_batch(batch)` 拒绝任何监督批次——无标签入口不可能误收标签。

## 3. 有标签批次 `ForecastInputs` / `ForecastBatch`

### 3.1 `ForecastInputs`（推理时可见的一切，不含标签）

```text
past_power_kw      float[B,L]       # 功率主路径历史
past_power_valid   bool[B,L]        # 历史可用性掩码
past_timestamps    datetime64[B,L]
origin_timestamps  datetime64[B]    # 第一个待预测区间起点 = 信息截断时刻
target_timestamps  datetime64[B,H]
target_spec        TargetSpec
domain_ids         tuple[str, ...]
past_covariates    float[B,L,F] | None     # F=0 时 None
past_covariate_mask / past_covariate_available_at
covariate_names    tuple[str, ...]
future_known       float[B,H,K] | None     # K=0 时 None
future_known_mask / future_known_available_at
future_known_names tuple[str, ...]
metadata           dict
```

强制规则（构造即校验）：

1. `origin_timestamps == target_timestamps[:, 0]`：起报时刻就是第一个待预测 15 分钟区间的起点；
2. 所有 `past_timestamps < origin`：尚未结束的区间不能当作已观测历史；
3. `past_power_valid=False` 处允许 NaN，**不得悄悄当作 0 kW**；
4. 对 `mask=True` 的协变量/已知未来特征，要求 `available_at ≤ origin`，否则抛 `LeakageError`；
   被 mask 掉的条目不是输入，不参与可用性判定；
5. `F=0`、`K=0` 完全合法：纯功率主路径必须能独立完成预测；
6. `target_spec` 必须是 `active_power` 且单位为功率（`past_power_kw` 的物理前提）。

### 3.2 `ForecastBatch`

```text
inputs             ForecastInputs
targets_kw         float[B,H]       # 仅 active_power + kW
target_valid_mask  bool[B,H]        # 无效标签保留掩码，不补造
target_available_at datetime64[B,H] | None
```

强制规则：`targets_kw` 形状必须 `[B,H]` 且与 `inputs` 一致；`mask=True` 处必须有限；单位必须是 kW；利用率目标根本无法装入。

### 3.3 接口边界

- `fit(ForecastBatch)` / `score_batch(ForecastBatch, ...)` 可以读标签；
- `predict(ForecastInputs)` 在类型层面**没有**未来真实标签；
- 无标签数据只走 `PretrainBatch` / `resource_pretrain`，绝不传给监督 `fit`。

## 4. 字段注册表与跨源对齐（`src/data/registry.py`）

### 4.1 关系判定

对源域与目标域的同名字段，只有**语义、单位、统计方式都可比**才判为 `SHARED`：

| 判定 | 条件 |
|---|---|
| `shared` | 规范语义同名 + 单位一致 + 统计方式相容 + 同为观测量 |
| `incomparable` | 同名但单位或统计方式不同 → 不合并，登记原因 |
| `source_only` | 仅预训练源有 |
| `target_only` | 仅目标域有（保留原值、掩码、来源说明） |

`AlignmentResult.status`：

- `ready`：存在共享字段且无不可比字段；
- `pending_adaptation`：存在共享字段但另有不可比字段，需要显式适配层；
- `no_shared_features`：**没有任何可比共享字段**，明确返回"暂不可迁移"，不硬凑同维特征、不伪造变量。

### 4.2 目标域独有字段

1. 不被丢弃：保留 `raw_names`、单位、统计、来源域；
2. 生成配置 `target_only_local_weight = 1.0`：仅表示"该字段在其本地分支中默认不被额外抑制"；
3. **不是**物理回归系数，**不是**最终预测占比；
4. `global_fusion_weight = None`（未决），本轮不训练、不模拟最终融合；
5. 测试（`tests/test_registry.py`）证明：独有字段仍在路由表中、局部权重可见、全局权重恒为 `None`。

### 4.3 数据来源登记表（`configs/datasets/registry.yaml`）

登记每个来源的角色、标签七要素、审计状态与 `can_produce_facility_kw`。校验规则：

- `unlabeled_pretrain` 声明标签 → 报错；
- 声称可产出正式设施 kW、但标签不是 `verified` 的真实设施级实测 → 报错；
- 目前**没有任何真实来源**被判定可用于正式设施 kW 结果（Google 利用率 / Alibaba 无标签 / M100 节点级 / NLR ESIF 分项和均显式阻断，理由写入 `blocked_reason`）。

## 5. 必须失败的输入（安全边界）

以下情形由契约层直接抛错，测试覆盖：

| 情形 | 异常 |
|---|---|
| 无标签源声明功率标签 | `ContractError` |
| 利用率/比例目标进入 kW 批次 | `PhysicalUnitError` / `TargetEligibilityError` |
| 节点级功率伪装设施总功率 | `TargetEligibilityError` |
| 合成标签声明为 `measured` | `TargetEligibilityError` |
| 未来真实数据以 `available_at > origin` 进入输入 | `LeakageError` |
| 上下文时间戳 ≥ 起报时刻 | `LeakageError` |
| `targets_kw` 形状/单位/范围冲突 | `ContractError` / `PhysicalUnitError` |
| 历史不足 | `InsufficientDataError`（由窗口/基线层抛出并记录原因） |
