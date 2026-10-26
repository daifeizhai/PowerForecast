# 最终方向验证的证据入口

测试前协议标签：`sapf-robust-protocol-20261026`，提交`a722278`。源侧运行：`results/runs/20261026T080606Z_sapf_robust_dev_8b8652`。本目录保留源开发manifest、准入决定、全部域级权重比较、质量表、旧评分网格交集及预注册哈希；原始观测和模型仍保留在本机数据与results目录。

`computational-source.zip`保存预注册中全部计算源码的原始字节，避免Windows Git换行转换影响SHA-256复核。解压到隔离工作目录后，逐项对照preregistration.json的computational_source_sha256；使用项目已有虚拟环境，并在该目录提供原始数据路径。不要直接在已完成实验上覆盖模型、冻结文件或预测。

源开发和最终评价均通过`scripts/run_experiment.py --config configs/sapf_robust.yaml`，分别指定`--phase development`或`--phase final --development-run <完整源开发目录>`。最终评价还需要`results/analysis/robust_direction/preregistration.json`及其对应源开发manifest。复现是已知数据的重放，不重新成为独立泛化证据。`--results latest`只读显示完成的真实运行；`scripts/audit_cross_domain.py --run <最终目录> --output <新审计文件>`只读复算；`scripts/plot_robust.py <最终目录>`仅从保存的结果绘图。

本轮只有新的评分日期；设备和中心身份已知，跨数据集为离线研究。Eagle不在本轮数据范围内。动态未准入的格不得补开实验后合并为本轮预注册成绩。

## 完成的结果

正式运行：`results/runs/20261026T085237Z_sapf_robust_c9cded`。研究解释见[最终报告](../../SAPF_ROBUST_RESULTS.md)。

- `final/manifest.json`、`frozen-before-test.json`、`preregistration.json`：最终配置、193份模型状态冻结、源侧准入及代码哈希。
- `final/metrics.csv`：全部域级曲线、日均和3d非重叠指标；`domain-summary.csv`：全部方法的域等权均值、最差域、两种本地基线的更差域比例和点数。WAPE是逐域计算后等权平均，不是点池化WAPE。
- `final/statistics.csv`：配对区间、H1的8项及H2预留16项Holm校正；未执行动态有明确标记。`decision.json`保存按预定条件自动汇总的通过数，本轮没有自动成功。
- `final/per-origin.parquet`、`routing.parquet`、`routing-summary.csv`：冻结预测的起报指标、权重和机会诊断。正式知识利用率只解释共享同一E/L的固定、受限固定和动态三组；基线与其他结构不能据这些列作知识利用因果推断。
- `final/audit.json`、`independent-numpy-check.json`：586项冻结哈希、148项恢复、73,290项公共指标与2,352项独立NumPy核验。`public-cli-fixture-audit.json`验证公共CLI的合成流程分派，**不是额外真实测试**。
- `descriptive/`：单源域级误差矩阵、原始输出分布、机架/发布组与历史CV分组、最差域；`posthoc_diagnostics.py`是只读结果分析源码，不训练、选参或预测。
- `figures/`：权重敏感性、配对置信区间及三项研究的单源误差图。3d正式alpha为预选0.25，不能按图中测试最优值改选。
- `test-verification.json`、两个测试日志：80项不同测试的通过记录，含首次旧协议源码漂移失败及固定代码下重跑说明。`environment.json`保留实际Python和包版本；`ordinary-console.log`保留普通`--results latest`输出。
- `final/original-run-file-hashes.json`：本机完整604个运行文件的SHA-256。逐点预测、原始单源NPZ、193份模型及适配数据仍在完整运行目录，未把大型模型和原始数据重复提交Git。

轻量证据可核查汇总和重算起报统计；逐点审计与模型恢复必须保留本机完整运行目录及原始数据。对结果重放应使用副本，不覆盖原冻结资产。图表已实际渲染检查，未按目标结果修改模型或阈值。
