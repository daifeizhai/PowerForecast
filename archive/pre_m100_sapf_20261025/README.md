# M100 接入前的 SAPF 工作区快照

source.tar.gz 按字节保存 M100 接入前的 95 个源文件、配置、测试、报告和证据文件；manifest.json 记录逐文件 SHA-256 及压缩包摘要。该快照保留当时尚未提交的 SAPF 工作区，Git 基准为 bc6a439，不冒称新的 Git 提交。

保存目的为追踪先前 Google 结果与后续 M100 扩展的边界。原始 Google 运行目录、预测、模型和证据副本仍保留原位；本次不重训或改写它们。当前实验仍只经 scripts/run_experiment.py 进入，快照不作为第二条活动实验流程。
