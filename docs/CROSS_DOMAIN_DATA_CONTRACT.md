# 跨域数据与域契约

实现位于 src/data/domains.py，扩展现有适配、预处理与窗口。契约声明与数值处理结果分别记录，不因增加声明而制造新观测或独立标签。

## 1. 元数据与数值边界

| 内容 | 已实现字段/行为 |
| --- | --- |
| 来源身份 | dataset_id / site_id / domain_id，可选 power_domain_id / rack_id / node_id / target_field |
| 物理目标 | target_kind / target_scope / target_unit；合批要求物理边界、单位、来源、采样和区间语义相容 |
| 标签来源 | measured / energy_converted / component_sum / calibrated_estimate / assumed_scenario / not_estimable；分项汇总需 mapping_route 和覆盖声明 |
| 时间 | 时区、布尔 timezone_verified、原生频率、采样语义、区间定义；保留真实网格及内部缺测 |
| 覆盖/来源 | coverage / source_version / assumptions，运行归档附源文件哈希、逐域采样计数和质量表 |
| 输入可用性 | feature_availability；目前只有日历/相对相位可以 future_known |
| 窗口侧身份 | 可省略 ForecastInputs.domain_ids，长度等于窗口数，不进入数值矩阵 |

history=[N,L,C] / future_known=[N,H,K] / targets=[N,H]、fit(batch) 与 predict(inputs) 不变。新合批创建独立只读数组，take() 同步选择域身份；旧模型无需修改签名。

## 2. 允许与拒绝

- 允许同一计量范围的 W/kW 换算；区间 Wh/kWh 可按声明的实际区间小时数转功率。需另存转换身份，转换函数不会将估算标签升格为实测。
- 拒绝利用率转为未经标定的 kW，拒绝用单位换算改变 CPU/GPU、node、rack、IT 或 facility 边界。合批不混合 measured、component_sum 和情景标签。
- 新聚合函数要求完整、不重叠、同一区间和唯一组件，任一组件缺失仍为缺失。NLR 原五项求和保留，只声明“所供五分项齐全”，不声称覆盖独立设施总表。
- 不接受 10 分钟/小时数据靠插值冒充原生 15 分钟观测；仅可整分形成 15 分钟的原生频率进入本轮可比契约。
- 通用适配器遇到多个显式域身份会报错，要求先分域；Google 每文件须唯一 (cell,pdu)，不同域同刻是不同窗口。
- Google 新标签需 3/3 有效样本，否则整区间缺失；历史或标签不完整窗口跳过。NLR 有限样本均值及历史协变量有限前向填充保持原样。
- future_known 只有日历/相对相位；未来真实目标、天气、IT/GPU 负荷、质量标记、任务事后统计均不作为输入。Google 生产利用率与机器映射也不进入数值特征。

错误明确指出对象，不猜测式转换。自动 A—H 选路、累计电表处理与通用功耗/PUE 拟合尚未实现。

## 3. 当前数据审计

| 数据 | 本地证据 | 契约与评价范围 |
| --- | --- | --- |
| NLR ESIF | 两份 combined parquet 存在；功率 4,569,543 行、天气 4,034,917 行；处理文件/metadata 可读 | active_power / facility / kW / component_sum / route C；一个设施分项序列 |
| Google PowerData 2019 | 57 个时序 + 映射，每域 8,928 原始点、31 个协议日 | power_utilization / power_domain / utilization_fraction / measured |
| NLR Eagle | 当前数据目录未定位 | 待节点/机架计量、单位、覆盖证据，不以 ESIF 替代 |
| Alibaba GPU v2020 | 7 个 CSV 及对应 tar.gz 已在本地，样例看来没有表头 | 权威字段、采样、功率真值未审核，暂不接入 |
| M100 / 墨尔本服务器 | 当前数据目录未定位 | 仅登记，不下载、不编造样本 |

候选清单：[配置](../configs/datasets/cross_domain_candidates.yaml)。

NLR 时间过滤后的功率范围为 2015-11-10—2025-08-29，天气从 2016-06-12 起，较短天气不裁剪功率源。异常年份过滤和单位换算保留，timezone_verified=false。功率的 343,687 个区间中，五项最小有效计数恰为 15 的有 279,816 个；部分为 0，部分大于 15，反映缺测、采样抖动及频率变化，不能视为连续积分真值。完整记录：[NLR 数据审计](../results/analysis/cross-domain-v2/nlr-data-audit.json)。

Google 时间依据[官方 notebook](https://github.com/google/cluster-data/blob/master/power_trace_analysis_colab.ipynb)：原数值为相对 2019-05-01 00:00 PT 前 600 秒的微秒数。新适配明确将 600000000 对应 2019-05-01T00:00:00-07:00，UTC 首点为 07:00；不按各文件最小时间重新平移 D1。数值日历仍只使用相对日/周相位。

57 域共 169,632 个区间，**24 个缺少 3/3 有效样本**，新协议保持缺失。机器映射为 96,616 行，按 (cell,machine_id) 有 2 个跨 PDU 歧义身份，涉及 cellc 的 pdu41/42/43。机器计数不能用于聚合或 kW 标定；目标和源池若涉及这些映射重叠，规划会拒绝，要求先审核或显式修改源池。本次三个目标不涉及歧义。MVPP 的 0 个映射项仅表示文件没有相应记录，不表示物理设备数为零。

## 4. 旧数据兼容与新产物

nlr.yaml 只新增 domain_metadata。数值配置摘要排除该纯声明字段，仍匹配旧 sidecar；改动清洗、来源、目标或频率仍会导致失配。修改目标列还须同步物理契约，避免 IT 被误标为 facility。

新预处理在 metadata 中记录契约并输出 .coverage.parquet，每列为填充前有限、去重时间戳的采样数。本轮没有重新预处理或覆盖既有文件。新 NLR 运行的 manifest 叠加当前声明并注明来源，不追改历史 sidecar 或 manifest。

Google 新协议从原始文件经过扩展 google_panel.py，不调用历史 HDSF 的填补/评分逻辑。运行中的 data-audit.json / sample-coverage.parquet / manifest.json 记录哈希、身份及覆盖。轻量副本见[证据摘要](figures/cross-domain-20261024T165116Z_d531e9/evidence.json)；大型原始数据、预测与模型仍保留于本地运行目录。
