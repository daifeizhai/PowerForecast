# 交付报告（任务书 V2.0：第一阶段 + 第二阶段）

> 分支：`research/power-transfer-v1`（自 `origin/main` `3f97a745` 延续；前序骨架提交 `f2b2b40`）
> 执行环境：Windows 11 + Python 3.13.5 (.venv) + CPU（无 NVIDIA），真实本地原始数据，真实冻结权重。
> 本报告只记录**实际发生**的事实；未运行/未完成项逐条标注。

## 1. 工程修改总结

| 提交 | 内容 |
|---|---|
| `f0b28d7` | feat: Stage 1 target/source 数据管线、字段注册表与分析工具 |
| `01f6880` | feat: Stage 2 Chronos-2/TiRex-2 适配、固定 0.5/0.5 融合、实验层与图表 |
| 本文件（docs 提交） | README / 依赖 / ARCHITECTURE / 本交付报告 |

**新增关键文件**

- 阶段一：
  - `src/data/target_power.py`、`src/data/source_features.py`、`src/data/labels.py`
  - `scripts/prepare_target_power.py`、`scripts/analyze_power.py`、`scripts/prepare_source_features.py`、`scripts/analyze_sources.py`
  - `configs/datasets/nlr_target.yaml`、`configs/datasets/sources.yaml`、`configs/datasets/feature_registry.yaml`
  - `docs/STAGE1_TARGET_REPORT.md`、`docs/STAGE1_SOURCE_REPORT.md`、`docs/FEATURE_ALIGNMENT_REPORT.md`
- 阶段二：
  - `src/models/foundation/{offline,chronos2,tirex2}.py`、`src/models/power_main/{__init__,ensemble}.py`
  - `configs/models/foundation.yaml`、`configs/protocols/power_main_fixed.yaml`、`configs/experiments/power_main_v1.yaml`
  - `src/evaluation/power_run.py`、`src/evaluation/visualization.py`
  - `scripts/run_power_main.py`、`scripts/probe_foundation_models.py`、`scripts/summarize_power_run.py`
  - `docs/STAGE2_MAIN_MODEL.md`

**旧代码复用**：保留并复用前序骨架的数据契约/时间窗口/指标/运行目录/冻结生命周期（126 个既有测试仍全绿）；
`src/models/baselines.py` 仅做**授权的最小增量重构**（新增 `predict_from_context` 供 Stage-2 统一调用），
B0/B1 数值行为不变（回归测试证明逐值相等）。旧 `main` 基础模型适配思路仅作参考，
Stage-2 适配器按当前依赖版本（chronos-forecasting 2.3.2 / tirex-2 0.3.0）重新实现并实测。
未修改 `main`、旧研究分支、`data/raw/`、未跟踪结果；**未执行 push**。

## 2. 第一阶段结果

### 2.1 目标域（NLR ESIF，真实）

| 项目 | 数值/结论 |
|---|---|
| 标签 | `facility_power_kw = it+cooling+hvac+plug_and_light+pump` 分项区间均值求和 → **`facility_derived_kw`**（不是独立设施总表实测，绝不写成 `facility_measured_kw`） |
| 原始数据 | `esif.influx.buildingData.PUE.combined.parquet`（4,569,543 行，原生 ~60 s）SHA256 `19cd1240…`；只读，大小/哈希未变 |
| 时区 | `UTC`，`timezone_verified: false`（显式未核验；全部日号基于锚点，不做本地时区/DST 假设） |
| 锚点/选择 | 锚点 `2016-06-12T06:00:00Z` = 首个五分项全有限的 15min 槽；2015-11-10～2016-06-12 的 136,484 行原始记录五分项全 NaN，作为**数据选择边界**排除并计数（非压缩缺测） |
| 15min 网格 | 323,035 槽 / 3,365 trace 日；有效 310,367（96.08%）；12,663 缺测槽保留为 `power_kw=NaN, power_valid=false`，不插值、不回填 |
| 首个 31 trace 日 | **2,976/2,976 有效（100%）**，mean 820.18 kW，min 606.55，max 950.63（真实原始数据集成测试断言） |
| 覆盖率阈值 | `1/15`（“至少一个原生样本”存在性门槛，已文档化；不是 0.8 的完整度门槛）；逐槽 `coverage_ratio` 落盘供质量感知评价 |
| 目标独有字段 | 室外温度（°F→°C）、相对湿度；登记为 `target_only`，`future_mapping.local_weight: 1.0` 仅作 Stage-4 语义保留，`global_fusion_weight: null`；**本阶段未使用** |
| 特征分析 | 曲线/典型日（工作日-周末）/小时-日-周/ACF（水平与一阶差分）/爬坡波动/峰谷/覆盖率 7 图 + `power_stats.json`；如实报告：水平 ACF 被多年趋势主导，一阶差分 ACF≈0，**无显著 24h/7d 周期峰**（日轮廓幅值约占水平标准差 7.4%） |

产物：`data/processed/{processed_target_power,target_components,processed_target_features}.parquet`、
`data/processed/{target_metadata,target_quality}.json`、
`results/analysis/stage1/power_characteristics/`（7 图 + `power_stats.json`）。

### 2.2 源域（M100 ExaData，真实）+ 字段对齐

| 项目 | 数值/结论 |
|---|---|
| 数据 | `data/raw/M100/extracted/*.parquet`（16 节点，原生 15min，UTC；原始 1,311,025 行） |
| 原生频率审计 | 所有相邻间隔均为 900 s 整数倍、0 重复时间戳 → **15min_native_kept，未重采样** |
| 面板 | `processed_source_features.parquet`：**17,204,772 行**长表（timestamp, domain_id, feature_name, feature_value, unit, observed_mask, coverage_ratio），41.5 MB；**不含任何标签列**；缺失保留 mask+NaN，未零填充 |
| 字段 | 12 个 `_avg` 运行指标（节点总功率/GPU 功率/内存与 IO 功率/PSU 输入功率/温度/风扇/PCIe 等），每字段单位、语义、统计规则、可用性、范围均有元数据；`value`/`__index_level_0__` 等簿记列排除 |
| UCC 2023 | 原生 10min 已审计（bounded 300k 行）；10→15min **不可无假设地转换** → 保留原生频率并标记 `native_preserved_not_15min`，未伪造 15min 产品；dataset2 存在重复 (series,time) 与负时间差，已记录 |
| Alibaba GPU v2020 | 仅登记为 `candidate`（本地 `pai_machine_metric.csv` 无表头，事件表未聚合）；未生成连续序列 |
| 对齐（`FieldRegistry.align`） | M100 vs NLR：**shared 0 / source_only 12 / target_only 2 / incomparable 0**；M100 vs UCC：shared 0，1 个“同名假设”因单位/统计不可比而**不合并**；`global_fusion_weight` 全程为 `null` |
| 训练可用性 | **没有任何源域被批准用于训练**（`approved_for_training: false`，附原因：字段通道映射与单位部分依赖厂商遥测推断） |

产物：`data/processed/{processed_source_features.parquet,source_metadata.json,source_quality.json}`、
`configs/datasets/feature_registry.yaml`、
`results/analysis/stage1/source_characteristics/`（6 图 + `source_stats.json` + `alignment.json`）。

## 3. 第二阶段结果（真实零样本，CPU）

最终主实验运行：`results/runs/20261026T173302Z_108532`（16 个产物文件，含 7 图）；
可选 7d 运行：`results/runs/20261026T174015Z_db77c4`。
数据模型：仅历史 14 天 kW（1344 点）→ 未来绝对 kW；**未使用任何 CPU/GPU/任务特征**；无微调、无权重搜索。

### 3.1 测试集（D22–D30 起报，标签 D23–D31；WAPE % / MAE kW / RMSE kW / MAPE %）

| 跨度 | B0 前一天同刻 | B1 前三天均值 | A1 Chronos-2 | A2 TiRex-2 | **A3 等权融合** |
|---|---|---|---|---|---|
| 4h | 9.0522 / 71.1693 / 85.2141 / 9.2265 | 9.7427 / 76.5979 / 87.0254 / 10.1096 | 1.2660 / 9.9536 / 14.6635 / 1.2710 | **1.1054 / 8.6906 / 13.5235 / 1.1060** | 1.1743 / 9.2321 / 13.8605 / 1.1775 |
| 8h | 9.1438 / 71.4821 / 85.5530 / 9.2596 | 9.7954 / 76.5762 / 87.2812 / 10.1356 | 1.4895 / 11.6446 / 15.5393 / 1.5312 | **1.3840 / 10.8194 / 14.5301 / 1.4128** | 1.4067 / 10.9973 / 14.7171 / 1.4416 |
| 24h | 7.5873 / 61.3517 / 75.5128 / 7.7044 | 8.6134 / 69.6484 / 79.7074 / 8.9116 | **5.8224 / 47.0806 / 66.8937 / 5.9026** | 5.9533 / 48.1386 / 69.9189 / 5.9517 | 5.8741 / 47.4983 / 67.9579 / 5.9130 |
| 3d | 7.5841 / 61.6032 / 76.8598 / 7.7428 | 8.4024 / 68.2500 / 79.9029 / 8.6173 | **6.0744 / 49.3406 / 62.5530 / 6.1977** | 6.5720 / 53.3822 / 67.3711 / 6.6375 | 6.2323 / 50.6232 / 64.1845 / 6.3275 |
| 7d（可选） | 10.6140 / 86.2149 / 102.0855 / 10.4781 | 8.6868 / 70.5606 / 84.5245 / 8.7303 | **6.8528 / 55.6631 / 68.1917 / 6.8887** | 7.6751 / 62.3424 / 76.5264 / 7.6262 | 7.1187 / 57.8228 / 71.1657 / 7.1123 |

相对基线改进（WAPE）：A3 在 test 上 4h −87.0%、8h −84.6%、24h −22.6%、3d −17.8%、7d −32.9%
（以 B0 为参照；以 B1 为参照分别为 −87.9%/−85.6%/−31.8%/−25.8%/−18.1%）。

**必须如实报告的事实**：等权融合 A3 **并不总是最优**。
4h/8h 上 TiRex-2 单独最好（A3 居中）；24h/3d/7d test 上 Chronos-2 单独最好（A3 次之或与 A2 接近）；
development 7d 上 B1（前三天均值）WAPE 10.71% 甚至优于 A3 的 13.15%。
本阶段按任务书要求**不以测试集调整融合权重**，保持固定 0.5/0.5 并如实呈现。

### 3.2 分层报告

- 逐起报点指标：`origin_metrics.csv`；跨起报均值/标准差/极值：`metrics.csv`；
- 提前量误差：`predictions.parquet` 的 `lead_step` + `fig7`；
- 24h 完整日平均功率误差（test，A3）：MAPE 5.3764% / WAPE 5.2053% / MAE 42.09 kW / RMSE 53.44 kW；
- 3d 分天（test，A3）：288 点整体 WAPE 6.2323%；
  第 1/2/3 天 96 点 WAPE = 5.2600 / 6.5442 / 6.8607%；
  第 1/2/3 天日均功率 WAPE = 4.5667 / 4.9200 / 5.1684%；
- 3d/7d 起报窗口重叠已在 `manifest.json`、`metrics.json`、`split_audit` 与图中显式标注“非独立样本”；
- 7 类图（真实逐点数据生成）：`figures/fig1_history_and_forecast.png`、`fig2_chronos2_vs_actual.png`、
  `fig3_tirex2_vs_actual.png`、`fig4_ensemble_vs_members.png`、`fig5_metrics_by_horizon.png`、
  `fig6_multiday_per_day_error.png`、`fig7_error_by_lead_and_origin.png`。

### 3.3 长跨度与批量正确性

- TiRex-2 原生 `future_len=320`；7d（672）触发递推：`blocks=3, hops=2, block_size=320,
  context_update=predicted_median_only`，**只使用自身已预测中位数**，API 层面无法传入真实未来值；
- 批量独立性：同一序列单独预测 vs 放入批量（其他样本不同）逐位一致（`batch_size=1`，`cross_learning=False`）；
- 输出校验：形状/有限性/单位失败即报错；负值只记录不静默截断（本次运行负值计数见 manifest timings）。

## 4. 自动化测试报告

**命令**：`& .\.venv\Scripts\python.exe -m pytest -q`
**结果**：**243 passed, 0 failed, 6 warnings（约 200 s）**；无跳过。

| 测试文件 | 数量 | 覆盖 |
|---|---:|---|
| 既有骨架（9 文件） | 126 | 契约/注册表/窗口/协议/泄漏/基线/生命周期/合成 smoke |
| `test_target_power.py` + `test_preprocessing.py` | 43 | 15min 网格压缩防护、重复/非有限、标签分类、覆盖率、分项和一致性、原始保护、真实首 31 日集成 |
| `test_source_features.py` + `test_feature_alignment.py` | 19 | 长表 schema、掩码、多域独立、原生频率、注册表一致性、无伪造共享字段、`global_fusion_weight=null` |
| `test_model_adapters.py` + `test_ensemble.py` | 39 | 真实离线加载、P50、16/32/96/288/672、确定性、批量不变性、缺权重报错、固定权重与拒绝其他权重、B0/B1 适配等价 |
| `test_power_run.py` + `test_no_leakage.py` | 16 | 真实产物布局、指标可复算、改进表、逐日起报、失败路径、覆盖不压缩与未来值扰动不变 |

真实权重测试较慢（首次加载 + CPU 推理；空闲时约 40–90 s，机器繁忙时可超过 90 s），但全部通过。
**未运行/未覆盖**：跨 GMS/网络事件场景（无数据）、第三~五阶段模块（未实现，接口测试通过）。

## 5. 环境与可复现说明

| 项目 | 值 |
|---|---|
| 设备 | Intel Core Ultra 5 125H，31.5 GB RAM，**CPU**（无 NVIDIA，`torch 2.8.0+cpu`） |
| 依赖 | numpy 2.5.4 / pandas 2.3.3 / pyarrow 24.0.0 / PyYAML 6.0.3 / torch 2.8.0+cpu / chronos-forecasting 2.3.2 / tirex-2 0.3.0 / transformers 4.57.6 / accelerate 1.15.0 / matplotlib 3.11.2 |
| 权重缓存 | `D:\AAA_Project\PowerData_Prediction\.model-cache`（HF_HUB_CACHE 布局；**离线**读取，未下载、未随机初始化） |
| Chronos-2 | `amazon/chronos-2`，revision `29ec3766d36d6f73f0696f85560a422f50e8498c`，`model.safetensors` SHA256 `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`（477,930,472 B） |
| TiRex-2 | `NX-AI/TiRex-2`，revision `05e5b26db52bfb256f1ae1bdf785589850482de3`，`model.ckpt` SHA256 `184b160ffbe4c01a26beeba14015ff3507c7497e1f3577114187bbc1d19fcac1`（380,613,375 B） |
| 配置 | `configs/experiments/power_main_v1.yaml`（协议/起报/模型/融合/运行时），配置 SHA256 与逐次运行写入 manifest |
| 复现命令 | 见 README「快速命令」；数据哈希、权重哈希、Git SHA、依赖版本、耗时、跳过窗口全部在 `manifest.json` |
| 指标复算 | CLI 在标记 completed 前用落盘 `predictions.parquet` 重算并比对 `metrics.csv`（本次 40 行全部一致，`metric_recomputation.matches=true`） |

**可复现性实测**：同 seed/配置/权重连续两次完整运行（`20261026T170722Z_d968c9` 与 `20261026T173302Z_108532`）：
基线 B0/B1 逐点**完全一致**（diff=0）；两个基础模型逐点最大差 **1.8×10⁻⁴ kW**（相对 2×10⁻⁷，float32 舍入级别），
指标最大相对差 **1.2×10⁻⁶**。未声称逐位一致，此为 CPU float32 浮点归约差异，不影响结论。

## 6. 遗留问题（如实）

1. **标签性质**：目标设施功率是五分项汇总派生值（`facility_derived_kw`），不是独立设施总表实测；
   若需 `facility_measured_kw` 结论，需要新的独立总表计量数据。
2. **时区未核验**：原始时间戳按 UTC 处理（`timezone_verified=false`）；研究日锚点不依赖本地时区，但对外报告当地时间需另行核验。
3. **等权融合不是普适最优**：4h/8h TiRex-2 更优、24h/3d/7d Chronos-2 更优、development 7d B1 更优；
   固定 0.5/0.5 是**事前设计**而非数据调优结果，不得据此宣称融合提升。
4. **训练样本量/历史长度**：零样本仅需 14 天推理输入；若未来加入 LightGBM/TCN/映射模型，需单独分析监督样本与截止（本轮未做）。
5. **源域可迁移性**：M100 与 NLR 目标域**没有可辩护的共享字段**（节点遥测 vs 设施功率/室外天气），
   因此第三阶段“共享表示迁移”目前缺少共享字段基础；需要含 CPU/GPU/任务指标的目标域数据，或完成 UCC 全量审计/接入。
6. **源域训练审批**：M100 `audit_status=partial`、UCC `unverified`、Alibaba 仅 candidate，**均未批准用于训练**。
7. **7d 递推质量**：递推机制已验证（3 块/2 hop，仅用预测中位数），但 7d 长跨度误差累积未做专项研究；7d 为可选预设。
8. **冷启动耗时**：真实权重首次加载/首次推理受 OS 页缓存与其他进程影响（0.2–45 s 波动）；稳态约 0.26 s/0.52 s（B=2）。
9. **数据规模**：本轮只使用 NLR 首个 31 trace 日固定协议；更长滚动实验（任务书 §5.8）未开展，未混用其他片段结果。
10. **UCC/Alibaba**：UCC 10min 未转 15min；Alibaba 无表头/事件表未聚合，仅登记。

## 7. 阶段三衔接说明

第一阶段已为第三/四阶段准备好：

- **目标域**：`processed_target_power.parquet`（标签+valid+coverage）、`processed_target_features.parquet`
  （室外温湿度，`target_only`，`local_weight=1.0` 仅语义保留）、`target_metadata.json`（标签类型/边界/时区/锚点）；
- **源域**：`processed_source_features.parquet`（1720 万行长表，含 `timestamp/domain_id/feature_name/feature_value/unit/observed_mask/coverage_ratio`），
  可直接按域 pivot 为 `PretrainBatch`（`src.data.contracts.PretrainBatch` 已就绪，无标签、掩码优先、保留原生频率）；
- **字段注册**：`configs/datasets/feature_registry.yaml` 给出每个字段的规范名/单位/语义/聚合规则/可用性/角色
  （shared/source_only/target_only/incomparable + 原因）；`FieldRegistry.align` 可复算，`global_fusion_weight=None` 保持未决；
- **对接方式**（未来实现，不在本轮）：
  `PretrainBatch → BaseResourceEncoder.pretrain`（源域动态规律）→ 目标域未来运行指标预测 →
  `BaseTargetMapper`（运行指标→kW 辅助曲线）→ `BaseFusion`（与 A3 主线融合，权重另行冻结）；
- **前置缺口**：目标域缺少 CPU/GPU/任务/内存等运行指标，源域与目标域无可比共享字段；
  阶段三开始前需要补充目标域运行指标采集，或改用带运行指标的目标数据集，并完成源域字段审计与训练审批。
