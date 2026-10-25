# 收益学习与30分钟跨数据集扩展前快照

此处保留本轮实现前的133个工作区文件，包括15分钟SAPF/M100源码、配置、文档及证据。不是新的Git提交；基准仍为bc6a439。

压缩包为 [source.tar.gz](source.tar.gz)，逐文件摘要见 [manifest.json](manifest.json)。压缩包SHA-256：`2eda5456d03a2c0ce3453f14d4dffefdca83b599462239f5ff61c7b2fd76e8c0`。既有运行的模型、预测和原始数据保留原位置，不覆盖。

本轮新增30分钟通用采样支持，旧协议的原始默认值仍为96点/日。严格复现旧运行的源码哈希时应使用此快照；不能将当前扩展后的源码冒称旧运行的原始源码。新运行的精确计算源码另封存在 [frozen-computational-source.tar.gz](../../docs/figures/benefit-transfer-20261025T143438Z_771b85/frozen-computational-source.tar.gz)。
