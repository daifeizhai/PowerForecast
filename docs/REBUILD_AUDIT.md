# REBUILD_AUDIT —— 从 `main` 重建工程骨架的文件级审计

> 本文件在**任何删除动作之前**写定（任务书 §2.2 / §11-P0 要求）。
> 依据：`PowerForecast_main分支重建工程骨架_AI执行任务书_v2.0_定稿.md`

## 1. 基准与分支

| 项目 | 值 |
|---|---|
| 执行日期（本地） | 2026-10-26 |
| 远端 | `https://github.com/daifeizhai/PowerForecast.git` |
| `origin/main` SHA（fetch 后） | `3f97a7453aa76f87c6b98ec10158a6532a0b58aa` |
| 新实验分支 | `research/power-transfer-v1`（自 `origin/main` 以 `--no-track` 创建） |
| 新分支基点 | `3f97a7453aa76f87c6b98ec10158a6532a0b58aa`（`git merge-base --is-ancestor origin/main HEAD` 通过） |
| 审计前工作树 | `git status --porcelain=v1 --untracked-files=all` 输出为空（干净、无未跟踪文件） |

**旧分支 SHA 快照（本轮不改动、不推送、不重置）**

| 分支 | SHA |
|---|---|
| `main`（本地） | `3f97a7453aa76f87c6b98ec10158a6532a0b58aa` |
| `research/cross-domain-v2` | `bc6a439e8e5dcf955cc206957699db3bd476dc99` |
| `research/sapf-robust-check` | `bbebd6d64cae8f2c60cbfb978535a5e75eec2ea1` |
| `research/sapf-v1`（本地） | `f89dc7dd1f6a22b1c936d60a7a343388b13f5658` |
| `research/sapf-gain-v2`（本地） | `e0af261290c928f86e61fff4dfcc84bc1a79c0c5` |

执行前确认：`research/power-transfer-v1` 在本地与远端**均不存在**，无同名冲突。

## 2. 原始数据与未跟踪成果保护

以下路径全部被 `.gitignore` 忽略，**不属于 Git 跟踪范围**，本轮对它们**不执行任何清理命令**（无 `git clean`、无 `reset --hard`、无 `stash`、不删目录）。

| 路径 | 文件数 | 字节数 | 说明 |
|---|---:|---:|---|
| `data/raw/` | 95 | 7,309,969,813 | 原始数据：`Alibaba-GPU-v2020/`、`Google/`、`M100/`、`NLR_Dataset/`、`UCC2023/` |
| `data/interim/` | 3 | 15,811,152 | 旧中间产物（`nlr_weather_overlap_*`） |
| `data/processed/` | 7 | 32,366,510 | 旧预处理产物（`google_power2019*`、`nlr*`） |
| `results/` | 66,625 | 10,990,596,010 | 旧运行目录、缓存、日志、分析结果 |
| `.venv/`、`.pytest_cache/` | — | — | 本地环境与测试缓存 |

保护措施：

1. 删除只用 `git rm -- <显式相对路径>`，路径逐条列于 §3，**不使用通配符**；
2. 删除对象仅限 §3 的 **Git 跟踪文件**；`data/`、`results/`、`.venv/` 下被忽略内容不在 `git ls-files` 中，因此不可能被 `git rm` 命中；
3. 删除后复核 `git status` / `git diff --name-status`，并再次统计 §2 表格的文件数与字节数，确认无变化；
4. 旧实验历史保留在 `main` 及旧研究分支的提交中，可从原分支恢复，不需要在新分支复制。

## 3. 计划删除的 Git 跟踪文件（72 个，逐条）

### 3.1 旧配置（8）

```text
configs/all_models_nlr.yaml
configs/datasets/nlr.yaml
configs/datasets/other.yaml
configs/experiment.yaml
configs/google_hdsf.yaml
configs/hdsf.yaml
configs/hdsf_models.yaml
configs/models.yaml
```

理由：`main` 的配置以 Google-HDSF / NLR 18 类模型对照与 D14/D22 为隐含默认；新工程需要 `base.yaml`、`smoke.yaml`、`protocols/`、`datasets/`、`models/` 的显式配置语义（通用协议 + 可选预设）。旧配置内容与旧代码强耦合，不逐条移植。

### 3.2 旧依赖清单（2）

```text
requirements-hdsf.txt
requirements-lock.txt
```

理由：`requirements-hdsf.txt` 固化 chronos/timesfm/granite/tirex/toto/triton 等基座与 GPU 相关依赖，属于被移除的研究主线；锁文件绑定旧环境（含 xgboost 等已不使用的包）。新工程使用 `requirements.txt`（CPU 核心）+ `requirements-optional.txt`（可选包，不要求 GPU）。

### 3.3 旧报告与旧图（7 篇文档 + 19 张图）

```text
docs/Google-HDSF-D14-D22复现.md
docs/HDSF-NLR复现结果.md
docs/HDSF-NLR数据对齐重跑结果.md
docs/HDSF方法与新方案适配.md
docs/NLR数据对齐与重跑.md
docs/双数据集全模型核对.md
docs/算力中心中短期用电负荷预测——研究与实验方案.md
docs/figures/google-hdsf-20261014T162657Z_google_8ef216/24h_forecast.png
docs/figures/google-hdsf-20261014T162657Z_google_8ef216/7d_forecast.png
docs/figures/hdsf-nlr-20261012T143252Z_18d8be/24h_error_analysis.png
docs/figures/hdsf-nlr-20261012T143252Z_18d8be/24h_forecast.png
docs/figures/hdsf-nlr-20261012T143252Z_18d8be/24h_frozen_weights.png
docs/figures/hdsf-nlr-20261012T143252Z_18d8be/7d_error_analysis.png
docs/figures/hdsf-nlr-20261012T143252Z_18d8be/7d_forecast.png
docs/figures/hdsf-nlr-20261012T143252Z_18d8be/7d_frozen_weights.png
docs/figures/hdsf-nlr-20261013T120519Z_abc86d/24h_error_analysis.png
docs/figures/hdsf-nlr-20261013T120519Z_abc86d/24h_forecast.png
docs/figures/hdsf-nlr-20261013T120519Z_abc86d/24h_frozen_weights.png
docs/figures/hdsf-nlr-20261013T120519Z_abc86d/7d_error_analysis.png
docs/figures/hdsf-nlr-20261013T120519Z_abc86d/7d_forecast.png
docs/figures/hdsf-nlr-20261013T120519Z_abc86d/7d_frozen_weights.png
```

理由：均为旧 HDSF/NLR 研究线的成果记录与图，属于任务书 §2.3 明确要求在新分支移走的**已跟踪旧成果**。旧内容在 `main` 与旧研究分支中完整保留。新分支的 `docs/` 重建为 `REBUILD_AUDIT`、`ARCHITECTURE`、`DATA_CONTRACT`、`EXPERIMENT_PROTOCOL`、`TODO_RESEARCH`。旧研究方案文档中的合理约定（15min、kW、Dk 日号）已在 `EXPERIMENT_PROTOCOL.md` 中重新表述，不再引用旧文。

### 3.4 旧脚本（8）

```text
scripts/analyze_nlr.py
scripts/audit_hdsf_core.py
scripts/audit_model_coverage.py
scripts/google_comparison.py
scripts/report_hdsf.py
scripts/reproduce_google_hdsf.py
scripts/run_hdsf.py
scripts/verify_foundation_batch.py
```

理由：全部服务于旧 HDSF / 基座模型 / Google-NLR 对照主线。新分支只保留一个正式实验入口 `scripts/run_experiment.py`，外加两个数据工具 `prepare_data.py`、`inspect_data.py`。`main` 中 `prepare_data.py` / `inspect_data.py` / `run_experiment.py` 属"重写"类别（§4），不在本删除清单内。

### 3.5 `src/data/` 旧模块（4）

```text
src/data/cleaning.py
src/data/google_panel.py
src/data/pipeline.py
src/data/resampling.py
```

理由：`cleaning.py` 的 `clean_frame` 以"目标列在历史特征中"为前提，且默认 `ffill` 协变量；`google_panel.py` 绑定 Google 利用率面板业务假设（利用率不得当作 kW）；`pipeline.py` 是旧"读入→特征→窗口→模型"单管道，混同无标签与有标签数据流；`resampling.py` 只做简单均值重采样，缺少瞬时/均值/电量语义声明与完整性阈值。其可验证思想（区间左标签、质量统计）在新 `preprocessing.py` 中重写并加强。

### 3.6 `src/evaluation/` 旧模块（2）

```text
src/evaluation/reporting.py
src/evaluation/visualization.py
```

理由：报告层与旧 run 目录格式、旧 18 模型对照表深度耦合；绘图层依赖旧结果协议。新分支由 `artifacts.py`（新 run 目录、manifest、冻结凭据、predictions.parquet）承担，绘图改为可选且必须标注 `synthetic`。

### 3.7 `src/models/` 旧模型与特殊实验（20）

```text
src/models/base.py
src/models/deep/__init__.py
src/models/deep/forecasters.py
src/models/foundation/__init__.py
src/models/foundation/config.py
src/models/foundation/forecaster.py
src/models/foundation/foundation.py
src/models/foundation/hdsf_patchtst.py
src/models/foundation/new_foundation.py
src/models/foundation/prediction_cache.py
src/models/foundation/usc_timesfm.py
src/models/ml/__init__.py
src/models/ml/tabular.py
src/models/proposed/__init__.py
src/models/proposed/hdsf.py
src/models/proposed/local.py
src/models/proposed/simplex.py
src/models/statistical/__init__.py
src/models/statistical/baselines.py
src/models/statistical/sarima.py
```

理由：任务书 §2.3 明确"`src/models/foundation/`、`src/models/proposed/`、不需要的旧深度/统计模型、HDSF 脚本与相关旧配置不作为新主线保留"。`base.py` 的 `BaseForecaster.fit(WindowBatch)` 以单一批次同时承载历史/未来标签，未区分无标签预训练与监督拟合，且把目标藏在 `history[:, :, 0]`。新分支改为 `interfaces.py`（`BasePowerForecaster` / `BaseResourceEncoder` / `BaseTargetMapper` / `BaseFusion`）+ 两个真实可运行基线 + 明确 `planned` 状态的抽象接口，不保留旧模型实现。

### 3.8 `src/training/` 旧模块（2）

```text
src/training/predictor.py
src/training/trainer.py
```

理由：`trainer.py` 直接使用旧 `deep`/`ml` 模型；`predictor.py` 是旧预测-截断辅助。新分支改为 `lifecycle.py`（四阶段：development / finalize / final-test / smoke，冻结与恢复校验）与 `runner.py`（共享运行逻辑）。

### 3.9 旧测试（5）

```text
tests/test_all_models.py
tests/test_data.py
tests/test_forecasting.py
tests/test_google_protocol.py
tests/test_hdsf.py
```

理由：测试对象是被删除的旧模型/旧管道/旧 Google 协议。新分支按任务书 §10 重建 T01–T22 对应测试文件。

## 4. 保留 / 复用 / 重写清单（26 个跟踪文件）

| 类别 | 路径 | 处理 | 复用理由或重写原因 |
|---|---|---|---|
| 基础环境 | `.gitignore` | 修订 | 保护 `data/`、`results/`、`.venv/` 的规则正确，保留；补充新产物路径与本地缓存 |
| 基础环境 | `pyproject.toml` | 修订 | setuptools + `src*`/`scripts*` 包发现 + pytest `pythonpath=["."]` 可用；保留 `from src...` 导入习惯，补充可选依赖与元数据 |
| 基础环境 | `requirements.txt` | 修订 | CPU 核心依赖（numpy/pandas/pyarrow/PyYAML/matplotlib/pytest）合理；移除运行期不需要的项，保持无 GPU |
| 基础环境 | `requirements-optional.txt` | 修订 | 保留"可选、不强制安装"的语义；去掉与本轮无关的基座模型包 |
| 占位 | `data/raw/.gitkeep`、`data/interim/.gitkeep`、`data/processed/.gitkeep`、`results/.gitkeep` | 保留 | 目录占位，保证数据/结果目录存在且实际内容不入 Git |
| 包骨架 | `src/__init__.py`、`scripts/__init__.py` | 保留 | 支持 `python -m scripts.run_experiment` |
| 数据读取 | `src/data/loaders.py` | 重写 | `read_table`/`inspect_table` 的分块/元数据思想可复用；去掉 `locate_source` 的旧 `fallback_paths` 协议与 `load_source` 旧 source 配置耦合 |
| 数据适配 | `src/data/adapters.py` | 重写 | 时区定位、`ambiguous="raise"`、UTC 归一、单位 scale/offset、重复时间戳合并等逻辑可验证并复用；改由 `DatasetSpec`/`FeatureSpec` 驱动，不再依赖旧 `source` 字典 |
| 特征 | `src/data/features.py` | 重写 | 旧实现服务于"目标在历史特征第 0 列"协议；新分支拆分为 `features.py`（仅用起报前历史/已知日历）与 `windows.py`（批次构造） |
| 时间切分 | `src/data/splitting.py` | 重写 | 旧 `chronological_splits`/`complete_days`/`adaptive_day_splits` 以自然日/比例切分为主，缺少统一 trace 锚点、Dk 日号、origin 与训练截止语义；保留"按时间顺序切分、不用未来统计"的原则，重写为通用协议解析 + trace 网格 + 可用性校验 |
| 合成数据 | `src/data/synthetic.py` | 重写 | 旧 `make_demo_data` 只有单一数据源且列含 `power_utilization` 类语义风险；新分支需要 31+ trace 日、F=0 与多变量两种输入、两个异构无标签源、局部缺失掩码、`synthetic_simulated` 标签 |
| 指标 | `src/evaluation/metrics.py` | 重写 | 旧 `forecast_metrics` 的 MAPE/WAPE/MAE/RMSE 计算公式可复用（含阈值与排除计数），但需补充 `wape_defined`、阈值单位语义、`mape_excluded` 口径说明，并按新评价层级使用 |
| 回测 | `src/evaluation/backtest.py` | 重写 | 旧 `per_origin`/`per_lead`/`per_day` 分层思想可复用；需修正：4h/8h 的均值不得称为日均、7d 重叠窗口标注、`predictions` 表结构需含新契约字段 |
| 训练生命周期 | `src/training/__init__.py` | 重写 | 旧为空包，需导出生命周期 API |
| 模型包 | `src/models/__init__.py` | 重写 | 旧注册表包含已删除模型；新分支注册表只含两个真实基线与 `planned` 状态接口，未实现模型不得显示为"已支持" |
| 数据包 | `src/data/__init__.py` | 重写 | 需导出新契约 |
| 评估包 | `src/evaluation/__init__.py` | 重写 | 需导出新指标与存档接口 |
| 工具 | `src/utils.py` | 重写 | `PROJECT_ROOT`/`resolve_path`/`load_config`/`config_digest`/`write_json`/`seed_everything`/`configure_logging` 可复用；需补充 `git_sha`、流式文件哈希、run 目录命名、JSON 不可序列化处理 |
| 数据工具 | `scripts/inspect_data.py` | 重写 | CLI 骨架（argparse、错误退出、可选 JSON 输出）可复用；改为按 `DatasetSpec` 审计字段而非旧 `source` 配置 |
| 数据工具 | `scripts/prepare_data.py` | 重写 | 需支持"合成数据落盘到 `data/processed/` 版本化路径 + sidecar 元数据"，旧脚本绑定 NLR/Google 导入 |
| 正式入口 | `scripts/run_experiment.py` | 重写 | 旧入口 18 模型 + HDSF + 旧协议；新入口必须是唯一正式实验 CLI（`--config/--protocol/--horizon/--phase/--demo/--run-dir/--frozen-run/--seed/--output`） |
| 文档 | `README.md` | 重写 | 旧 README（52KB）描述已移除的 HDSF/18 模型主线；新 README 只描述新骨架、安装、四条 smoke 命令、阶段与产物 |

## 5. 删除执行与复核

1. 执行 `git rm -- <§3 显式路径...>`（72 条，无通配符）；
2. `git status --porcelain=v1 --untracked-files=all` 复核：只有 §3 的 `D` 记录，无其它变更；
3. 重新统计 §2 表格四类目录的文件数与字节数，与删除前完全一致；
4. 确认 `data/raw/`、`data/interim/`、`data/processed/`、`results/` 无删除、无移动；
5. 删除结果与偏差（如有）追加记录在本文件末尾。

## 6. 执行结果复核记录（删除后实测）

### 6.1 清理动作

- 执行命令：`git rm -- <72 条显式路径>`，**无通配符**；
- 结果：`git status` 中恰好 72 条 `D`，另有本文件为新增；无计划外文件变更；
- 复核方式：重新创建的新分支上执行，未触碰 `main`、旧研究分支、`.git/`。

### 6.2 数据保护实测（删除前 → 删除后）

| 路径 | 删除前 | 删除后 | 结论 |
|---|---:|---:|---|
| `data/raw/` | 95 文件 / 7,309,969,813 B | 95 文件 / 7,309,969,813 B | **完全一致，原始数据零改动** |
| `data/interim/` | 3 文件 / 15,811,152 B | 3 文件 / 15,811,152 B | **完全一致** |
| `data/processed/` | 7 文件 / 32,366,510 B | 19 文件 / 33,293,832 B | 文件数增加：新骨架的 `prepare_data` 按版本化路径新增 12 个合成数据产物（不入 Git）；**无删除、无覆盖** |
| `results/` | 66,625 文件 / 10,990,596,010 B | 66,739 文件 / 10,993,050,345 B | 文件数增加：本轮 4 条 smoke + finalize/final-test 运行目录（不入 Git）；**无删除、无覆盖** |

本地原始数据、旧结果、未跟踪文件均未被清理命令命中。新增的 `pytest-of-*/` 临时目录已加入 `.gitignore`。

### 6.3 分支与 SHA（未提交前）

- 当前分支：`research/power-transfer-v1`；基点 `3f97a7453aa76f87c6b98ec10158a6532a0b58aa`；
- `git merge-base --is-ancestor origin/main HEAD` 通过；
- `main`、`research/cross-domain-v2`、`research/sapf-robust-check`、`research/sapf-v1`、`research/sapf-gain-v2` 的 SHA 与 §1 快照**完全一致**（未被修改/重置/强推）。

### 6.4 安全边界实测

- `python -m scripts.run_experiment --config configs/smoke.yaml --demo --phase final-test --horizon 24h`（无冻结凭据）→ 退出码 1，提示需要 `--frozen-run`，且未创建 run 目录；
- `--phase finalize` 后再用 `--frozen-run` 执行 `final-test` → 成功，`test_report.json` 记录哈希校验与从磁盘重载；
- 合成结果的 `manifest.json` 均为 `synthetic=true`、`research_result=false`；
- `data/raw/` 无法被 `write_table` / `write_processed` 写入（对应测试通过）。

