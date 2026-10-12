# PowerForecast

**面向算力中心的中短期电力负荷预测研究框架**

> 项目状态：基础框架已实现。数据检查、15 分钟预处理、时间划分、基线训练、24 小时 / 7 天滚动评估与结果导出可运行。可选模型需要额外依赖，自主创新模型仍待设计；本项目尚未完成论文复现或精度目标验证。

## 1. 项目简介

本项目研究算力中心（数据中心、超算/HPC 中心）的电力负荷变化规律，构建兼顾**预测精度、可解释性和场景适应性**的多变量时间序列预测方法。

重点任务：

- **24 小时预测**：每次预测未来 **96 个 15 分钟点**。
- **7 天预测**：每次预测未来 **672 个 15 分钟点**。
- 融合历史 IT 负荷、制冷/辅助设施负荷、室外温度、湿度及时间特征。
- 统一比较统计模型、机器学习、深度学习和拟提出的创新模型。
- 输出可复现的预测结果、误差指标、可视化图表与消融实验。

课题希望实现高精度负荷预测（目标精度 95%），但该数值是**研究目标而非本项目已实现的结果**。具体“准确率”口径须在实验报告中明确。

## 2. 数据集

| 数据集 | 用途 | 注意事项 |
| --- | --- | --- |
| **NLR HPC Facility PUE**（优先） | 设施级功率、IT/辅助系统功率、环境变量联合预测 | 先核对字段口径、时区、连续性与缺失率 |
| **Google PowerData 2019** | 功率利用率预测与基准对照 | 利用率不直接等于 kW；时间跨度较短 |
| **Google ClusterData 2019** | CPU/任务等算力活动辅助分析 | 不是直接的实测功率标签 |
| **Alibaba Cluster Trace** | 任务调度、资源使用及算电关系研究 | 多数版本缺少连续的真实功率标签 |

NLR 官方地址：<https://data.nlr.gov/submissions/300>

建议将下载后的文件放到以下目录：

```text
data/raw/nlr/power.parquet
data/raw/nlr/weather.parquet
```

原始数据原则上**只读、不可覆盖**；数据处理后的文件写到 `data/interim/` 或 `data/processed/`。原始数据、模型权重和大体积结果通常不提交 Git。

## 3. 项目目录结构

```text
PowerForecast/
│
├── configs/                     # 配置文件
│   ├── datasets/                # 各数据集字段映射
│   │   ├── nlr.yaml
│   │   ├── other.yaml			 # 其他数据集配置
│   ├── models.yaml              # 模型参数
│   └── experiment.yaml          # 预测任务与划分规则
│
├── data/
│   ├── raw/                     # 原始数据
│   └── processed/               # 统一15分钟数据
│
├── src/
│   ├── data/
│   │   ├── loaders.py           # 读取不同数据集
│   │   ├── adapters.py          # 字段、单位统一
│   │   ├── cleaning.py          # 缺失值和异常值
│   │   ├── resampling.py        # 15分钟重采样
│   │   ├── features.py          # 特征工程
│   │   └── splitting.py         # 时间划分与样本生成
│   │
│   ├── models/
│   │   ├── base.py              # 统一模型接口
│   │   ├── statistical/         # SARIMA
│   │   ├── ml/                  # LightGBM、XGBoost
│   │   ├── deep/                # LSTM、TCN、DLinear
│   │   └── proposed/            # 自主创新模型
│   │
│   ├── training/
│   │   ├── trainer.py           # 模型训练及兼容性检查
│   │   └── predictor.py         # 多步预测
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # MAPE、WAPE、MAE、RMSE
│   │   ├── backtest.py          # 滚动预测评价
│   │   └── visualization.py     # 论文图表
│   │
│   └── utils.py                 # 日志、随机种子
│
├── scripts/
│   ├── inspect_data.py          # 数据检查
│   ├── prepare_data.py          # 数据预处理
│   └── run_experiment.py        # 一键实验
│
├── results/                     # 结果、图表、模型权重
├── tests/                       # 关键功能测试
├── requirements.txt
└── README.md
```

## 4. 安装与快速运行

Python 3.11 及以上。Windows PowerShell 示例（项目中已有 `.venv` 时可直接使用）：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 不依赖原始数据，检查两个预测任务的完整流程
.\.venv\Scripts\python.exe scripts/run_experiment.py --demo

# 检查本地 NLR 原始字段与样例；预处理不会改动原文件
.\.venv\Scripts\python.exe scripts/inspect_data.py
.\.venv\Scripts\python.exe scripts/prepare_data.py

# 默认执行 naive、seasonal_naive、ridge，以及 24h、7d 两项任务
.\.venv\Scripts\python.exe scripts/run_experiment.py

# 缩小日期范围或指定模型 / 预测任务
.\.venv\Scripts\python.exe scripts/run_experiment.py --start 2019-01-01 --end 2019-12-31
.\.venv\Scripts\python.exe scripts/run_experiment.py --models seasonal_naive extra_trees --horizons 24h

# 消融：full / target_only / no_weather / no_auxiliary
.\.venv\Scripts\python.exe scripts/run_experiment.py --demo --ablation no_weather

.\.venv\Scripts\python.exe -m pytest -q
```

入口也支持 `python -m scripts.run_experiment` 等模块形式。配置中的相对路径以项目根目录为基准。所有入口提供 `--help`。

`configs/datasets/nlr.yaml` 优先读取 `data/raw/nlr/`，也兼容当前目录下已有的 `NLR_Dataset/power.parquet`、`weather.parquet`，无需复制原始数据。`other.yaml` 是其他 CSV / TSV / Parquet 数据集的字段映射模板，需要根据实际字段修改。

## 5. 数据口径与实验协议

- NLR 本地文件的时间字段为 `ts`，天气字段为 `outdoor_air_temp`、`outdoor_air_humidity`。温度按[官方数据说明](https://data.nlr.gov/submissions/300)从华氏度转为摄氏度，湿度为百分比。
- 默认目标 `facility_power_kw` 是 IT、制冷、HVAC、照明插座、泵五项功率的区间均值之和，属于**派生的分项汇总标签**，不等同于直接测量的设施总功率。任一分项缺失，该区间标签保持缺失。可修改 `target` 为 `it_power_kw` 做 IT 功率预测。
- 原始时间戳没有时区信息；官方目录页也未明确时区。配置暂按 UTC 解释，`timezone_verified: false` 会写入质量报告。正式研究前应核实数据时区，调整配置后重新预处理。
- 删除配置范围外的异常时间记录（包括天气中的 1970 年记录）；按固定物理范围屏蔽异常值，重复时间戳取均值，再对齐 15 分钟网格。区间采用左闭右开 `[t, t+15min)`，时间戳标注区间起点；预测起点 `t` 的历史最多包含 `t-15min` 区间。
- 只对历史协变量做有限的前向填充，默认最多 4 个区间。目标标签不填充，不使用后向填充或双向插值；缺失历史或标签的样本窗口会跳过，并记录数量。长缺测或过短日期范围可能没有可用的 7 天窗口，此时明确报错。
- 按时间顺序划分训练 / 验证 / 测试，默认比例为 70% / 15% / 15%。每个窗口的全部预测标签必须落在所属划分内；验证和测试窗口可使用起点之前已观测到的历史。
- 模型拟合与标准化仅使用训练数据；依据验证集 WAPE 选择最佳模型，所有模型在测试推理前冻结。滚动评估更新历史观测，不重新拟合模型。默认每 96 个点产生一次起点；7 天预测区间可重叠，指标中的 `n` 统计全部预测实例。
- 未来输入仅包含已知的日历特征。真实未来 IT 负荷、制冷负荷和天气观测不会进入输入；未来天气预报接入需另行实现。
- WAPE、MAPE 使用百分数；MAPE 排除绝对值不超过 `1e-8` 的零标签，报告 `mape_n`；全零标签的 WAPE / MAPE 无定义时写为 JSON `null`。不将 `100-WAPE` 自动宣称为准确率。

## 6. 模型与扩展

| 模型 | 当前实现 | 依赖 |
| --- | --- | --- |
| `naive`、`seasonal_naive` | 最新值 / 上一日周期基线，默认执行 | 基础依赖 |
| `ridge`、`extra_trees` | 直接多步回归；历史及未来日历展开为特征 | scikit-learn |
| `sarima` | 在末段连续训练数据拟合参数；每个起点对历史滤波后预测 | statsmodels |
| `lightgbm`、`xgboost` | 每个预测步一个回归器，长预测任务计算量较大 | 对应可选库 |
| `lstm`、`tcn`、`dlinear` | CPU 可用的基础网络与训练封装；LSTM / TCN 使用历史变量，DLinear 使用目标历史，均不使用未来日历 | torch |
| `proposed` | 显式扩展占位，运行会抛出 `NotImplementedError` | 待设计 |

可选模型是初版实现，未作为论文方法复现，也未在本次环境中验证训练结果。按需安装单个库，或安装全部可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
.\.venv\Scripts\python.exe scripts/run_experiment.py --models lstm --horizons 24h
```

缺少依赖时会提示具体安装方式，不会悄悄替换成基线。`configs/models.yaml` 管理参数，`configs/experiment.yaml` 管理随机种子、窗口、划分、起点步长和输出。扩展模型继承 `src.models.base.BaseForecaster`，实现 `fit(WindowBatch)` 和 `predict(ForecastInputs)`，再接入 `src/models/__init__.py` 的注册工厂；预测接口不接收标签。

## 7. 结果与验证

每次实验建立独立的 `results/runs/<UTC时间>_<随机标识>/` 目录，保留：

- `manifest.json`：完整配置、库版本、数据来源、划分边界、特征清单、实验状态和选中模型。
- `metrics.csv`：各模型验证 / 测试的 WAPE、MAPE、MAE、RMSE。
- 各任务的 `windows.json`、`selection.json`：有效及跳过窗口数，以及测试前冻结的验证集选择结果。
- 各模型的验证 / 测试预测 Parquet、逐步长指标 CSV、指标 JSON、首个测试起点曲线 PNG 和模型文件。基础模型保存为 `.joblib`，深度模型保存为 `.pt` 状态字典。

预处理同时生成 `data/processed/nlr.metadata.json`，记录单位、派生公式、时间假设、来源文件信息与缺失率。配置发生变化后应重新预处理；实验入口会检查配置是否与预处理记录匹配。

测试覆盖字段及单位转换、微秒时间戳对齐、缺失标签保护、因果填充、时间划分边界、未来观测隔离、指标零值口径、基线训练预测和模型保存。合成数据结果只用于流程验证，不能作为真实负荷预测的研究结论。
