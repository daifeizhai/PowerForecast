# SAPF 多源异构数据核验

核验日期：2026-10-25。登记配置见 [registry.yaml](../configs/datasets/registry.yaml)。第一轮 SAPF 的数据状态快照保存在 results/analysis/sapf/datasets.json；随后到位的 M100/UCC 本地文件已重新全量审计，详见 [M100/UCC2023 实际文件审计报告](M100_UCC2023_DATA_AUDIT.md)及其证据。新审计不覆盖第一轮快照，也没有生成估算 kW 标签。

## 1. 已用于监督实验的数据

Google 本地有 57 个电力域、31 个协议日，目标是功率利用率，单位 utilization_fraction；依据[官方定义](https://github.com/google/cluster-data/blob/master/PowerData2019.md)，不能当作设施总表 kW。现有严格适配继续保留 24 个不完整 15 分钟区间，要求历史及标签完整，未来只含相对相位。机器映射的歧义与 MVPP 结构另行登记；不依据机器计数推导功率。

新 SAPF 的 14/11/10 个源/验证/测试域在同一种物理目标下比较；没有把 ESIF 或外部服务器瓦数与 Google 混成一张精度表。

## 2. Alibaba GPU v2020：完成 schema 与文件身份核验

已按[官方无表头 schema 与 SHA-256 清单](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020)读取本地七张 CSV，并逐文件流式校验；**7/7 匹配**。适配函数 read_alibaba_table 保留资源活动身份，显式标记无可信功率标签。

| 表 | 本地行数 | 主要语义 |
| --- | ---: | --- |
| pai_group_tag_table | 1,055,032 | 实例语义分组 |
| pai_instance_table | 7,522,002 | 实例开始/结束及机器标识 |
| pai_job_table | 1,055,501 | 作业生命周期 |
| pai_machine_metric | 2,009,423 | 与实例时段相关的机器资源指标 |
| pai_machine_spec | 1,897 | 机器 CPU/GPU/内存配置 |
| pai_sensor_table | 3,033,232 | 实例全生命周期资源均值/最大值 |
| pai_task_table | 1,261,050 | 任务状态与资源请求 |

官方明确指出 sensor 表中的资源量按实例全生命周期汇总，部分内存字段是最大值。不能把这些事后统计重复铺到实例之前的 15 分钟区间，也不能把计划 CPU/GPU 当实际使用或功率。当前只完成 schema 与资源活动准入；无标签预训练、资源序列构造和跨数据集共享编码尚未实现。

## 3. M100 ExaData：本地 15 分钟聚合包已审计

[IPMI 文档](https://gitlab.com/ecs-lab/exadata/-/blob/main/documentation/plugins/ipmi.md)定义 total_power 为节点 W，底层标称 20 秒采样。当前本地实际对应[官方 15 分钟聚合记录](https://zenodo.org/records/7541722)，不是此前引用的 [20 秒原始月度分区](https://zenodo.org/records/7588815)。0.tar/1.tar 的官方 MD5 均匹配，已直接读取全部 33 个节点，而非只检查 16 个已解包文件。

目标为 total_power_avg，另有 std/min/max；实际时间戳为 UTC、15 分钟对齐，未见重复。2,782,426 行中功率缺失 230,383 行（8.2799%），另有 174,642 个缺失时间槽；每节点最长连续有效功率段为 64.11—152.89 天。当前宽表没有 CPU/GPU 利用率，不能把恒为零且单位未明的 gv100cardX 当利用率。

原 adapt_m100_total_power 只面向原始 timestamp/value/node 长表，按 45 个有效 20 秒采样聚合，不适用于已聚合宽表。后续已新增专用 adapt_m100_aggregated，直接 total_power_avg W→kW，并完成[冻结的 M100 跨节点实验](M100_RESULTS.md)；当时的数据审计快照仍保留。聚合包没有原始计数和已核实的区间左右边界，实验仅使用过去公开槽，按单区间延迟上界假定历史可用性，不把该假定写成实测证据。

## 4. UCC2023：完整 CSV 已审计，10 分钟与时区仍需独立协议

[官方 Zenodo](https://zenodo.org/records/10069402)两份 CSV 的官方 MD5 均匹配。Power 为服务器 W，实际记录均对齐原生 **10 分钟**；D1 有 81 个主机标识、2,748,525 行，D2 有 64 个、1,281,024 行。全列非缺失行数与[作者论文](https://hpc.ec.tuwien.ac.at/files/UCC_23_data_center_analysis.pdf)清洗后行数一致；原文件本身仍有缺失，D1/D2 功率缺失率为 0.0030%/13.8361%。D1 实际主机标识数与论文 80 台口径不同，尚未获得解释。

时间字段无时区；两份 CSV 在 2019-04-07 02:00—02:50 均有重复钟表时间，与墨尔本夏令时回拨吻合，但仍须核实来源时区和 fold。本次保守排除全部重复键后，D1 最长连续功率段为 53.39—74.30 天，D2 为 0—74.30 天；D2 一台主机完全没有功率，另有严重断续主机。辅助字段还存在全缺、单位倍率待核实和核数语义问题，详见完整报告。

10 分钟不能整分为 15 分钟原生区间；不能插值后冒称 15 分钟实测真值。可另立原生 10 分钟协议（历史/24h/3d=2016/144/432 点）或共同 30 分钟协议（672/48/144 点），但本次没有生成新时序或修改 SAPF 默认配置。

## 5. Eagle 与 ESIF

NLR Eagle 整体登记为 final_holdout：训练、验证、校准、策略设计均禁止，最终推理才可用起报前历史；本轮只核对目录是否存在，没有打开观测或标签。模型训练边界对保留身份及 final_holdout 元数据有明确拒绝检查。

NLR ESIF 仍是五分项派生设施标签 component_sum，UTC 尚未核实。它没有变成 Eagle 节点数据，也不用于证明跨设施总表泛化。其旧预测和模型只在归档兼容性核对中按原协议读取，未用于本轮 SAPF 参数设计。

后续应分别建立节点、机架、设施目标。共享编码可以研究跨单位的时序形状，但监督标签、尺度恢复、计量边界和最终评价必须独立记录；归一化不能让不同物理真值变成相同标签。
