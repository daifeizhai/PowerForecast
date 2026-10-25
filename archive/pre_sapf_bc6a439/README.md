# SAPF 之前的冻结归档

基准：bc6a439（research/cross-domain-v2）。SAPF 整理分支：research/sapf-v1。归档目的为保留 HDSF、18 模型、旧 Google D14/D22、NLR 完整日实验及第一轮 LODO 的代码、配置、报告和复现证据。

- [source.tar.gz](source.tar.gz)：Git 基准的 131 个跟踪文件，包括原 README、源码、配置、测试、报告及图。
- [manifest.json](manifest.json)：压缩包及各成员 SHA-256、规范 LF 校验、44 次历史运行的原位置与 manifest/metrics 摘要。
- [archived-active-paths.json](archived-active-paths.json)：从主线移出的 75 个路径及移出前/归档字节哈希。Windows Git 导出换行可能不同，删除前逐文件按显式 LF/CRLF 规范比较内容。
- [protected-artifacts.json](protected-artifacts.json)：20,229 个既有数据/运行/缓存/权重的只读清单；部分大原始文件、检查点和缓存按大小/mtime，而非完整内容哈希检查。

原始数据、历史模型、预测和缓存不移动、不复制为多份大文件。它们仍在 manifest 指向的位置，只读归档；新的 SAPF latest 不选择这些历史运行。源码文件的旧导入路径在冻结包中完整保留，主线模型注册表不再包含 HDSF/预训练成员/旧 proposed 占位。

统一入口提供显式归档访问：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --legacy-config configs/google_hdsf.yaml --help
.\.venv\Scripts\python.exe scripts/run_experiment.py --legacy-config configs/hdsf.yaml --results results/runs/20261013T120519Z_abc86d
.\.venv\Scripts\python.exe scripts/run_experiment.py --results results/runs/20261014T191923Z_google_9a3ad6
```

归档运行时先验证包和每个成员，解包到 results/cache/legacy_source/<SHA>/。源码、模型导入和配置均来自冻结包，data/results/已有环境路径指向原工作区。可信路由钩子使这套路径规则在旧 Google 切换到原数值环境后继续生效，两种 Python 环境均已核对。不会自动下载缺失权重或数据。重跑旧实验需要相应本地环境与外部只读检查点；不会在默认 SAPF 流程启动旧训练。

本轮已核对冻结源恢复、旧 CLI 帮助、NLR ESIF 的 Ridge 24h/7d 保存预测，以及 HDSF 两个任务模型的反序列化/配置读取。没有重跑全部基座。详细结果在 SAPF_RESULTS.md 及本地 results/analysis/sapf/legacy-replay-verification.json。

旧文档与原始复现步骤可以从压缩包或已验证解包目录读取；不要把历史成绩或旧物理口径改写成 SAPF 的新结果。
