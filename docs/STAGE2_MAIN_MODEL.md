# 第二阶段：历史功率主模型 A（方法、协议与复现）

> 对应任务书 V2.0 §4–§7、§10–§13。本文件描述**已实现并实际运行**的内容；数值结果见运行目录
> `results/runs/<run_id>/`，本文件只列命令与结构，避免把不同运行的结果混写进方法文档。

## 1. 模型定义

主模型 A 是**零样本、权重冻结**的两模型等权融合，只使用目标数据中心历史功率：

```text
历史 14 天 kW (1344 × 15min)
        ├── Chronos-2  (amazon/chronos-2, P50 中位数) ──┐
        └── TiRex-2    (NX-AI/TiRex-2,  P50 中位数) ──┤
                                                        ↓
                                           固定等权融合 w = (0.5, 0.5)
                                                        ↓
                                              未来 H 点绝对功率 (kW)
```

- **不使用** CPU/GPU/任务等辅助路径特征（主路径接口只接收 `context_kw`）；
- **不微调**、不做目标域训练、不搜索融合权重；
- 这里的 0.5/0.5 是**主路径内部**两个基础模型的固定系数，**不是**未来主辅路径 A/B 之间的融合权重；
- 两个成员都取 **P50 中位数**；不允许返回 NaN/Inf，不允许截断/复制末值冒充原模型输出。

### 1.1 基础模型适配（`src/models/foundation/`）

| 成员 | 权重来源 | 输出 | 长跨度策略 |
|---|---|---|---|
| `Chronos2Forecaster` | `amazon/chronos-2` 本地快照，`Chronos2Pipeline.predict_quantiles(..., quantile_levels=[0.5], cross_learning=False)` | `[B,H]` kW | 原生支持 H=16/32/96/288/672 |
| `TiRex2Forecaster` | `NX-AI/TiRex-2` 本地 `model.ckpt`，`tirex2.load_model(..., use_flex_attention=False, compile=False)`，P50 = 分位数组 index 4 | `[B,H]` kW | 原生 `future_len=320`；H>320 时仅用**自身已预测中位数**递推，记录 block/hops，不追加真实未来值 |

离线权重解析（`src/models/foundation/offline.py`）：`try_to_load_from_cache` 定位完整快照 →
`local_files_only=True` / `HF_HUB_OFFLINE=1` → 记录快照路径、文件 SHA256、字节数与 revision；
缓存缺失时**报错而不是下载**（下载只允许在独立的权重准备步骤中显式开启 `--allow-download`）。

### 1.2 等权融合（`src/models/power_main/ensemble.py`）

- `FixedWeightedEnsemble(weights=(0.5,0.5))`：构造时校验权重非负、和为 1、且**只允许 0.5/0.5**；
- 没有任何自动权重搜索/开发集调参接口；
- 成员形状/有限性错误会明确指出是哪个成员失败；
- `describe()` 把固定权重、成员 checkpoint 摘要写入实验记录。

### 1.3 对照基线

| 编号 | 模型 | 实现 |
|---|---|---|
| B0 | 前一天同刻 | `src/models/baselines.py::SeasonalNaiveForecaster`（末 96 点日轮廓平铺/截断） |
| B1 | 前三天同刻均值 | `src/models/baselines.py::ThreeDayMeanForecaster`（近 3 个 trace 日同时隙均值） |
| A1 | Chronos-2 | `Chronos2Forecaster` |
| A2 | TiRex-2 | `TiRex2Forecaster` |
| A3 | 等权融合 | `FixedWeightedEnsemble` |

所有模型在**相同起报点、相同真实标签、相同有效掩码、相同指标代码**下比较；
"不要求等权融合一定优于两个单模型，若没有提升如实报告"。

## 2. 数据与标签

- 目标域：NLR ESIF `facility_derived_kw`（见 `docs/STAGE1_TARGET_REPORT.md`）；
- 主路径输入仅为历史 kW；**设施分项功率不作为未来已知特征**（它们本身是标签组成项）；
- 时间协议：锚点 = 首个有效 15 分钟槽（2016-06-12 06:00 UTC），按连续 96 槽 = 一个研究日；
  `Dk` 结束的起报时刻为第 k+1 个区间起点；缺测不重编日号。

## 3. 固定 D14/D22 协议（配置驱动，代码不写死）

`configs/protocols/power_main_fixed.yaml` + `configs/experiments/power_main_v1.yaml`：

| 任务 | 点数 | 开发起报 | 测试起报 | 说明 |
|---|---|---|---|---|
| 4h | 16 | D14, D16, D18, D20 | D22…D30 | 24h 测试标签覆盖 D23…D31 |
| 8h | 32 | 同上 | 同上 | |
| 24h | 96 | 同上 | 同上 | 9 次完整 24h 预测，标签 D23–D31 |
| 3d | 288 | D14, D16, D18 | D22…D28 | 7 次 3d 预测，最晚到 D31 结束；开发/测试窗口重叠 |
| 7d | 672 | D14, D15 | D22, D23, D24 | 可选预设（`optional_horizons`），本阶段不强制运行 |

重叠声明：3d（以及 7d）窗口复用真值，**不是独立统计样本**；`manifest.json` 与 `split_audit` 会显式标注，
`daily_metrics.csv` 按预测日拆分，不把重叠点当作独立样本做显著性结论。

## 4. 运行方式

```powershell
# 全量：4h/8h/24h/3d × development+test × 5 个模型（真实权重、离线）
python -m scripts.run_power_main --config configs/experiments/power_main_v1.yaml

# 只跑测试集 24h + 3d
python -m scripts.run_power_main --horizons 24h 3d --splits test

# 指定模型子集（调试/复算）
python -m scripts.run_power_main --models seasonal_naive power_ensemble

# 权重准备（独立步骤，允许联网下载；实验推理默认离线）
python -m scripts.probe_foundation_models --allow-download
```

输出目录（每次运行新建、不可覆盖）：

```text
results/runs/<run_id>/
├── manifest.json         # Git/环境/配置哈希/数据哈希/锚点/起报日/模型检查点哈希/权重/耗时/跳过原因
├── metrics.csv           # 每 (split,horizon,model) 的 MAE/RMSE/WAPE/MAPE + 跨起报分布统计
├── origin_metrics.csv    # 逐起报点指标
├── daily_metrics.csv     # 24h 日均功率误差；3d/7d 第 1/2/3… 天曲线误差与日均误差
├── improvements.csv      # 相对 B0/B1 的改进百分比
├── predictions.parquet   # 逐点预测（dataset/site/target_kind/model/origin/lead/actual/prediction…）
├── target_quality.json   # 阶段一目标域质量审计
├── source_quality.json   # 阶段一源域质量审计
├── logs/run.log
└── figures/              # 任务书 §7.3 的 7 类图（由已保存的逐点预测生成）
```

## 5. 指标与分层报告

- 主报告：`WAPE`（%）、`MAE`（kW）、`RMSE`（kW），辅助 `MAPE`（%，含近零阈值与有效样本数）；
- 逐起报点误差 + 全体有效点汇总 + 跨起报均值/标准差；
- 相对 B0/B1 的改进百分比；
- 随预测提前量（lead）变化的误差；
- 24h：完整日平均功率误差；
- 3d：整体 288 点误差 + 第 1/2/3 天曲线误差 + 三个日均功率误差；
- 所有指标均可由 `predictions.parquet` 独立复算：`scripts/run_power_main.py` 在标记 completed 前
  会用落盘预测重算并与 `metrics.csv` 比对，不一致即判失败。

## 6. 防泄漏与正确性保证

1. **构造期防泄漏**：`ForecastInputs` 要求上下文时间戳严格早于起报时刻；已知未来特征 `available_at ≤ origin`；
   主路径不构造任何协变量/未来特征（`F=0, K=0`）；
2. **预测函数只接收上下文与 H**：`predict(context_kw, horizon)` 签名层面没有标签参数；
   `tests/test_no_leakage.py` 用篡改未来真值的方式证明预测不变；
3. **缺测不压缩时间轴**：`tests/test_no_leakage.py` 删除中间日，D14/D22 起报位置与日号不变；
4. **批量一致性**：同一序列单独预测与放入批量（不同其他样本）输出一致（`tests/test_model_adapters.py`）；
5. **输出校验**：形状/有限性/单位（kW）失败即报错；负值默认**只记录不截断**，
   如开启 `postprocess.clip_negative` 则保存裁剪前后两列并在 manifest 记录规则；
6. **不做静默短路**：缺权重、H 不合法、上下文不足都明确报错。

## 7. 可复现性

- 权重：本地快照 + 文件 SHA256 + revision 写入 manifest；推理设备 CPU、dtype float32；
- 环境：Python 与主要依赖版本、Git 分支/SHA、配置 SHA256 写入 manifest；
- 数据：目标功率 parquet 与 metadata 的文件 SHA256 写入 manifest；
- 随机性：零样本确定性推理；`seed` 仍写入记录；
- 不使用历史缓存冒充新实验：默认每次真实重算（无预测缓存）。

## 8. 本阶段未做（明确边界）

- 未实现第三阶段（无标签运行指标动态预训练）、第四阶段（目标域功率映射）、第五阶段（主辅融合）；
- 未对 Chronos-2/TiRex-2 做任何微调；
- 7d 协议仅在配置中预留，是否运行见交付报告；
- 未接入新的真实设施总表实测数据（目标标签是分项汇总派生值）。
