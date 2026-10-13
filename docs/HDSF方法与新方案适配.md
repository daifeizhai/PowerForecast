# HDSF 方法与新方案适配

本项目以《算力中心中短期用电负荷预测——研究与实验方案》为新数据实验协议，复现旧库 HDSF 的核心结构。旧库 GitHub HEAD 与本地 HEAD 已核对为 `83f044c3259faa432eecf62abb1de70425269cce`；读取旧库，不修改其未提交变更，也不使用旧预测包作为新数据结果。

## 方法理解

HDSF（Half-Day Simplex Fusion）将 15 分钟预测曲线按 48 点分段：24h 有两个段，7d 有十四个段。每段分别学习非负、和为 1 的成员权重，优化验证点的绝对误差总和。求解器采用成对坐标下降，将权重从成员 i 移至 j，一维最优步长由加权中位数给出，方向是 `P_i-P_j`，可行区间是 `[-w_j,w_i]`。

分量 A 包含基座时序模型，分量 B 在 A 的基础上加入本地因果成员。最终曲线为 `(1-mix)*A+mix*B`。mix 在 41 个固定候选点上通过验证交叉检验选择。半日权重用于刻画不同提前量、上午/下午的成员优势；本地模型提供近期水平与日内形状。这种设计是否适合 NLR，必须通过相同窗口上的单成员和融合消融验证，不能直接引用旧 Google 数据的改善幅度。

旧库依据：[HDSF 主流程](https://github.com/daifeizhai/PowerData_Prediction/blob/83f044c3259faa432eecf62abb1de70425269cce/power_forecast/hdsf_experiment.py)、[融合求解](https://github.com/daifeizhai/PowerData_Prediction/blob/83f044c3259faa432eecf62abb1de70425269cce/power_forecast/attainment_fusion.py)、[本地模型](https://github.com/daifeizhai/PowerData_Prediction/blob/83f044c3259faa432eecf62abb1de70425269cce/power_forecast/attainment_local.py)。原始推理适配器已复制到 `src/models/foundation/`；模型权重从可配置的只读缓存加载，新预测按模型身份和历史输入哈希存入本项目 `results/cache/`。

## 新方案优先的调整

| 项目 | 旧实验 | NLR 主实验 |
| --- | --- | --- |
| 数据 | Google 31 天、57 序列利用率 | NLR 3364 个日历日（保留内部缺测）、单条派生设施功率 |
| 划分 | 固定 D14/D22 与固定开发/测试起报 | 完整日约 60/20/20；至少 8 天验证、9 天测试 |
| 输入 | 基座成员存在 14、21、28 天版本 | 全部成员统一 14 天；长版本标识保留，但上下文统一 |
| 分段拟合 | 每段 pooled L1 单纯形 | 保留相同求解器与 48 点段长 |
| 混合选择 | 留一起报 WAPE、旧开发门槛 | MAPE；五折验证并剔除与留出折重叠的标签窗口 |
| 本地 24h | 57 序列元数据及扩展历史 LightGBM/线性混合 | NLR 14 天因果滞后、日内/水平统计、真实日历；四种种子残差树与线性滞后等比混合 |
| 本地 7d | 最多 24 天水平/形状平滑 | 相同半衰期 3/5/10 天公式，主实验限制在 14 天输入内 |
| 最终训练 | 固定 D22 | 验证后重拟合本地/对照模型到测试之前；融合策略保持固定 |
| 评价 | WAPE 曲线/7d 日均 | MAPE 为主，同时 RMSE/WAPE；24h、7d 日均分别计分 |

这些调整属于“新方案下的 HDSF 迁移复现”，不应称为旧 D14/D22 协议或旧精度数值的逐项复现。`cv_mode: loo` 可启用逐起报留一，但短验证期若所有 7 天目标窗口互相重叠，会明确报错，避免制造无独立标签的检验。

## NLR 数据解释

[官方目录](https://data.nlr.gov/submissions/300)将 IT、制冷、HVAC、照明插座、泵字段标为 kW，室外温度为华氏度。目标是五项已测分项的区间均值之和，属于派生汇总标签；不等同于独立总表读数。未来真实天气、IT 和辅助负荷不进入输入。本轮 HDSF 与对照仅使用历史目标与已知日历。

原文件位于 `data/raw/NLR_Dataset/`。处理保留完整时间轴，物理异常屏蔽在聚合之前，重复时间戳取均值，15 分钟区间均值后求分项总和；目标不填补。区间均值使用当时可用的有限分钟记录，五项区间均值都存在时标签有效。“目标完整日”表示 96 个有效聚合点，并非断言全天 1440 个原始分钟都完整；原始区间覆盖数另行审查。起止不完整日统一剔除，其余缺测导致相应窗口跳过。源时间戳无时区，官方目录和下载的 README 也未说明；UTC 是待核实的假设，这会影响真实当地日界线和星期特征，正式发布前需要核实。

## 运行与证据

预处理以 `primary_source: power` 保留 2015-11 起的完整功率时间轴，不让天气起点裁剪数据。实际五项设施负荷标签在更早区间几乎全部不可构造；主实验按目标首尾完整覆盖日取 2016-06-13 至 2025-08-28，共 3364 天。较早 IT 观测仍保留在预处理文件中。首尾因时间网格及目标不可用而剔除的行数分别记录，内部缺测日期不删除、不压缩时间轴。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/analyze_nlr.py
.\.venv\Scripts\python.exe scripts/run_hdsf.py
.\.venv\Scripts\python.exe scripts/report_hdsf.py results/runs/<运行目录> --output docs/HDSF-NLR复现结果.md
.\.venv\Scripts\python.exe -m pytest -q
```

`run_hdsf.py` 等价于统一入口 `run_experiment.py --config configs/hdsf.yaml`，支持同样的参数。基础框架默认入口也已调整为 14 天、完整日自适应划分、MAPE 选择并导出日均表。

本机 `.venv` 复用了已有 Python 3.13 的基座模型依赖（`include-system-site-packages=true`），基础库与 Parquet 来自项目环境。重建独立环境可安装 `requirements-hdsf.txt` 并修改 `configs/hdsf_models.yaml` 中的权重缓存路径。精确实际版本、模型快照身份、配置、源文件哈希、种子与设备信息写入每次运行的 manifest/fusion 文件。

所有任务的训练、验证、重拟合先完成，再保存 `frozen-before-test.json`，之后才构造测试窗口并推理。HDSF 的总体模型选择采用交叉检验分数，导出的融合拟合 validation 分数标为校准样本内结果。结果包括所有基座成员、基线、季节 AR、DLinear、本地模型以及四种融合消融；每种方法都有完整曲线与日均的 MAPE/RMSE/WAPE CSV 和 Python 表格。

Toto 与 TimesFM 3.0 的滚动起报按原生独立 batch 轴处理，起报点不会成为同一样本的变量。`verify_foundation_batch.py` 对不同合成历史核对原始串行与批量 672 点预测，并污染其他 batch 样本检查隔离。批处理只产生浮点舍入量级差异，验证记录在 `results/analysis/nlr/batch-equivalence.json`，不计为真实数据研究精度。

逐起报点、逐预测天、非重叠 7 天检验、冻结模型文件和预测 Parquet 一并保留。报告命令重新读取预测行计分并检查冻结模型哈希。完整 7/14/28 天敏感性、多数据集泛化与业务特征研究属于后续研究，不与本次完成的 HDSF 迁移混淆。
