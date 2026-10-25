# 收益预测与双向迁移：冻结证据

对应真实运行 `results/runs/20261025T143438Z_sapf_transfer_771b85`，正式入口 `scripts/run_experiment.py --config configs/sapf_transfer_suite.yaml`。完整解释见[结果报告](../../BENEFIT_TRANSFER_RESULTS.md)和[协议](../../BENEFIT_TRANSFER_PROTOCOL.md)。本目录仅整理一次运行的结果，不重新拟合或挑选更好测试结果。

- `preregistration.json`：访问新测试误差之前的完整配置、质量选择、协议和计算源码SHA-256。
- `frozen-before-test.json`：两个源组、全部三项研究共118个模型状态的统一冻结；`audit.json`为公共套件审计。
- `frozen-computational-source.tar.gz`：与预注册匹配的计算源码、测试及配置，供严格复现；registry保留当时首阶段状态。
- `studies/`：逐研究原始指标、角色、时间方案、测试开始、冻结、保存恢复与风险诊断。
- `groups/`：纯源训练收益样本/历史性能先验、验证选择候选和最终训练记录；无目标测试标签进入拟合。
- `all-metrics.csv`：5040条节点/方法/时长/曲线或日均指标，包含独立验证选择期和最终测试，需按split区分。
- `all-metric-summary.csv`：节点等权MAPE/WAPE/RMSE/MAE；RMSE/MAE单位kW，WAPE不是池化总功率口径。
- `all-domain-summary.csv`：最终测试平均/最差MAPE、相对本地/原始池/归一化池的更差节点比例。
- `all-risk-summary.csv`：专家配对风险识别、混淆计数、收益误差、选择精度及回退率；附源训练基率常数Brier对照。
- `gain-calibration-diagnostics.csv`、`cohort-summary.csv`、`feature-importances.csv`：测试后描述性诊断；树特征重要性不代表因果，不用于本轮重新选参。
- `independent-verification.json`：独立指标、秩和AUROC、Brier、模型组件与880项混合复核。
- `native-verification.json`：132个节点从1292544个原生点独立聚合456192个标签，未调用正式适配函数。
- `verify_independently.py`、`verify_native_power.py`：只读复核脚本；原始运行、模型和本地数据须存在，`--output`须为新路径；用法见结果报告。
- `pytest-final.log`、`real-run.log`、`results-latest.log`：完整测试、正式运行与普通CLI展示记录。PowerShell原日志保留在results，证据副本转成UTF-8便于阅读。
- `mean-mape.png`、`worst-mape.png`、`risk-and-fallback.png`：标准Matplotlib独立图，已逐张检查。
- `export_evidence.py`：生成本目录表格和图的只读分析脚本，创建新证据目录时拒绝覆盖。
- `evidence.json`：本目录逐文件哈希与检查摘要；文件自身不递归哈希。原运行大型模型/预测/适配缓存保留原位置。

内部合成2000年时间轴不是实际UTC；实际源钟表保留在原预测中。UCC源时区未确认，M100/UCC采集年份不同，双向结果属于离线迁移基准。节点或服务器实测功率不等于设施总表。Eagle整体未打开。
