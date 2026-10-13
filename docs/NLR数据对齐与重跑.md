# NLR 文件对齐与重跑（2026-10-13）

按用户约定，本轮只更换 NLR 原始数据文件，沿用既有实验协议。新文件接入、预处理、普通默认实验、HDSF 对照实验和预测指标复算均已完成。

## 数据核对

当前文件位于 `data/raw/NLR_Dataset/`：

| 来源 | 当前文件 | 行数 | SHA-256 |
| --- | --- | ---: | --- |
| 功率 | `esif.influx.buildingData.PUE.combined.parquet` | 4,569,543 | `19cd12405dde9144b1a360e8c8418666c399a3d0d15a7f846880d71ab22f9dd4` |
| 天气 | `esif.influx.buildingData.outside.combined.parquet` | 4,034,917 | `97b424993fa77a15117fb2c4659a2c327fc83280f943fab47d9036260289a6a0` |

两文件的 SHA-256 分别与旧 `power.parquet / weather.parquet` 的审计记录一致。因此本地新旧数据内容相同，变化是文件名与路径。上述哈希可用于和同学核对原始文件；本轮未获取同学的处理配置或结果，不能据此断言双方处理口径和数值已完全一致。

重新预处理得到 `data/processed/nlr_esif_combined.parquet`，共 343,687 行，全部字段、时间戳及缺失位置与旧 `data/processed/nlr.parquet` 逐值一致。旧处理数据和旧实验目录继续保留。本轮来源核验记录为 `results/analysis/nlr-esif-combined/source-replacement-audit.json`，并复制到两份新实验目录。

## 沿用的协议

- 目标仍为 IT、制冷、HVAC、照明插座、泵五项功率的区间均值之和，单位 kW；分项不齐时目标保持缺失。
- 15 分钟网格、14 天历史输入、24h / 7d 完整曲线、每日滚动起报。
- 起止不完整日剔除，内部缺测保留；按完整日顺序约 60/20/20 划分。
- 初始拟合仅使用训练集，验证按 MAPE 选择，最终重拟合训练＋验证；所有任务测试前冻结，测试不重新拟合或调整融合权重。
- 未来输入仅含已知日历；阈值、模型参数、种子、默认清单、预测下界和评分定义沿用原配置。
- UTC 仍为未核实的时间假设。本轮不更改时区口径。

| 分区 | 起点（UTC） | 终点（UTC） | 天数 |
| --- | --- | --- | ---: |
| 训练 | 2016-06-13 00:00 | 2021-12-21 23:45 | 2018 |
| 验证 | 2021-12-22 00:00 | 2023-10-24 23:45 | 672 |
| 测试 | 2023-10-25 00:00 | 2025-08-28 23:45 | 674 |

研究边界内共 3364 个日历日，3026 天具有完整聚合目标；保留 12726 个缺失目标点。

## 两份已完成实验

| 预设 | 实验目录 | 原有输入约定 | 训练窗口 24h / 7d | 验证窗口 24h / 7d | 测试窗口 24h / 7d |
| --- | --- | --- | --- | --- | --- |
| 普通默认 | `results/runs/20261013T120651Z_416998` | 目标、配置的历史协变量、日历 | 686 / 510 | 150 / 100 | 328 / 251 |
| HDSF 对照 | `results/runs/20261013T120519Z_abc86d` | 目标、日历 | 969 / 726 | 312 / 227 | 328 / 251 |

普通默认清单为 `naive / seasonal_naive / three_day_mean / ridge / hdsf`；HDSF 对照清单为 `seasonal_naive / three_day_mean / seasonal_ar / dlinear / hdsf`。两份实验均展开基座成员、本地成员及融合消融，并在统一终端表中展示全部已评估方法。

两种预设沿用各自原有特征约定，缺测历史协变量使普通预设的训练和验证窗口较少。因此两种 HDSF 校准结果不能作为同条件的算法优劣对比。

| 预设 | 任务 | 曲线 MAPE (%) | 日均 MAPE (%) | 曲线 RMSE (kW) | 日均 RMSE (kW) | 曲线 WAPE (%) | 日均 WAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通默认 | 24h | 5.3013 | 4.2512 | 209.0586 | 174.6173 | 3.9456 | 3.2216 |
| 普通默认 | 7d | 11.1120 | 10.0367 | 364.9748 | 346.7459 | 7.5803 | 7.0113 |
| HDSF 对照 | 24h | 5.2950 | 4.2641 | 208.6780 | 173.9386 | 3.9321 | 3.2196 |
| HDSF 对照 | 7d | 11.0896 | 10.0184 | 359.0128 | 340.2688 | 7.5732 | 7.0040 |

上表均为 HDSF 独立测试指标；全部方法见各目录的 `metrics.csv`。完整 HDSF 对照报告见 [本轮重跑结果](HDSF-NLR数据对齐重跑结果.md)，普通默认报告位于其实验目录的 `report.md`。

本地模型与融合校准重新执行；基座预测按输入、模型快照及适配实现绑定的缓存读取。本次计算成本属于含缓存运行。原始内容相同，因此本轮是同内容重复实验，不构成新数据集上的泛化证据；测试结果仅用于报告，未用来重新调整方法。

## 复核与复现

29 项现有自动化测试通过。两份实验的保存预测均重新计算曲线与日均 MAPE/RMSE/WAPE，冻结模型哈希和处理数据哈希核对通过；每份实验的 128 行核心指标均与各自对应旧预设一致。两份实验各检查了 10 个模型的保存恢复预测，均在数值容差内一致；普通预设按生产批大小核对时最大差异为 0。各实验的 `metric-recomputation-audit.json`、`rerun-verification.json` 与 `source-replacement-audit.json` 保留具体证据。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/analyze_nlr.py --output results/analysis/nlr-esif-combined
.\.venv\Scripts\python.exe scripts/run_experiment.py
.\.venv\Scripts\python.exe scripts/run_hdsf.py

# 查看本轮已存结果，无需重新训练
.\.venv\Scripts\python.exe scripts/run_experiment.py --results results/runs/20261013T120651Z_416998
.\.venv\Scripts\python.exe scripts/run_hdsf.py --results results/runs/20261013T120519Z_abc86d
```

重跑日志和进程身份记录位于 `results/logs/nlr-esif-default-20261013.*` 与 `results/logs/nlr-esif-hdsf-20261013.*`。
