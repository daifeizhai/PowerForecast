# 跨域研究第一阶段仓库审计

审计日期：2026-10-25（Asia/Shanghai）。已完成 P0—P4 的首轮可运行闭环：Google 电力域利用率 LODO pilot、数据契约、公共流程复用与回归验证。HAST、全 57 折、跨设施总表验证及新增短预测任务不在本轮完成范围。

## 1. Git 基准与用户工作

- 基准提交：3f97a7453aa76f87c6b98ec10158a6532a0b58aa，原分支 main。
- 已创建并切换到 research/cross-domain-v2；工程验收时尚未提交。后续按用户 2026-10-25 的明确要求提交当前已暂存的完整版本，不推送，也不改写 main 的提交指针。
- 开始时已有：修改中的 README；已暂存的《模型输入输出统一约束》《算电映射与统一功率标签规范》《项目阶段性工作总结》；未跟踪的《异构算力中心跨域自适应负荷预测_研究路线报告》。
- 未执行 reset、clean、stash、删除或强制覆盖。README 在现有工作之上修正已实现状态并新增第 13 节；其余原有文档保持字节一致。工程验收时 Git index 与记录一致，改造未混入用户暂存区；提交阶段检查确认相关实现及规范文档均已暂存，按当前暂存快照一起提交。
- 已核对根目录及父目录约定，没有找到适用的 AGENTS.md。

本地只读产物清单：[baseline.json](../results/analysis/cross-domain-v2/baseline.json)。记录 18,585 个既有文件，涵盖原始/处理/中间数据、历史运行、预测缓存及外部离线权重，不复制大文件。已保存实验、处理数据及较小原始文件记录 SHA-256；其他文件记录大小/修改时间。具体覆盖见[证据摘要](figures/cross-domain-20261024T165116Z_d531e9/evidence.json)。

这次初版清单扫描耗时较长，其中源码摘要和 Git 状态在扫描末尾取得，不能把这两个字段当作改造前完整源码快照；改造前状态以上述初始检查为准，原始代码以基准 Git 提交为准。此限制不影响逐文件保护检查，审计工具现已提前捕获 Git 状态。[18,585/18,585 文件保护检查通过](../results/analysis/cross-domain-v2/protection-verification.json)，[11 份既有文档及 index 检查通过](../results/analysis/cross-domain-v2/user-work-protection.json)。仅检查大小/mtime 的文件不额外宣称哈希证明。

## 2. README 表述与实现核对

| README 表述 | 代码/数据实际情况 | 处理决定 |
| --- | --- | --- |
| 统一入口及 HDSF 转发 | run_experiment.py 控制公共阶段，run_hdsf.py 转发；Google 旧预设走历史桥接 | 保留，只增加新协议分发 |
| ForecastInputs / WindowBatch / BaseForecaster | 预测输入无标签，返回完整 [N,H]；原接口只有时间侧信息 | 保留签名；追加可省略 domain_ids，新多域合批数组只读 |
| 18 个对照及 HDSF | 模型工厂、默认清单与历史运行均存在 | 不删除、不改公式；跨域先复用 Ridge |
| NLR 设施五分项求和 | 原始功率文件 6 列、4,569,543 行，没有独立设施总表列 | 声明 component_sum；不改数值处理或历史标签 |
| NLR UTC 未核实 | naive ts 暂按 UTC，原 metadata 为 false | 保持 false；采样计数审计另存 |
| 分钟聚合不保证 15 条 | 功率区间最小分项计数分布含 0—30，仅 279,816/343,687 个区间恰为 15 | 保存实际计数，不改成严格积分或全采满标准 |
| Google 57 域及 D14/D22 | 57 个时序 gzip + 1 个映射文件；每域 8,928 条五分钟记录，31 天；历史监督输入 7 天 | 原配置、相对坐标和历史运行不改，新预设独立命名 |
| Google 旧评分掩码 | 历史路径保留部分有效点均值及 target_valid，某些全无效块有因果历史填补 | 新协议独立要求 3/3，历史/标签不补；不声称逐值复现 |
| 14 天输入 | NLR 已实现；旧 Google 监督输入为 7 天，其 7d D14 训练每域仅一个窗口 | 新 LODO 用 14 天，D22 每完整源域仅两个 7d 窗口；独立开发不可用则明确 N/A |
| 多域尚未通用支持 | google_panel.py 有历史面板窗口，但未排除目标域 | 增加逐域适配和留域编排，保留原函数行为 |
| 4h / 3d 待实现 | HDSF 及展示仍有 96/672 点约束 | 继续暂缓，不添加不可用命令 |
| 标签元数据尚未贯通 | 原 pipeline 仅存目标/单位/派生公式等 | 新增契约、采样计数及 manifest 声明，旧 sidecar 不回写 |

README 第 12 节原描述的是 2026-10-23 文档整理状态，现已标明历史时间并链接第 13 节，避免继续把本轮成果写成未实现。

## 3. 保留、改造、废弃候选与暂缓

| 决定 | 文件/能力 | 依据与边界 |
| --- | --- | --- |
| 保留 | 模型注册、HDSF、权重、旧 Google 配置/桥接 | 仍被默认入口、复现和测试调用，没有已验证替代 |
| 保留 | 历史数据、metadata、预测、manifest、缓存和报告 | 保持原身份，只读复算与保护验证 |
| 改造 | adapters.py / pipeline.py、数据配置、domains.py | 拒绝混域、校验物理含义、元数据与数值分离、保存采样计数 |
| 改造 | splitting.py / google_panel.py / cross_domain.py | 原窗口逐序列复用，增加域身份和 LODO/截止规划 |
| 改造 | 公共训练/评价归档 | 提取 artifacts.py / lifecycle.py 供两个协议共用；指标公式保留于原模块 |
| 改造 | 统一 CLI、reporting.py | 新协议配置分发，逐域仍用公共中文表，支持 --results latest |
| 废弃候选 | 本轮未选定可安全删除模块 | 未证明无调用且有替代；proposed 占位仍有明确失败行为测试 |
| 暂缓 | HAST、自动 A—H、4h/3d、跨设施总表 | 本轮数据和工程边界不支持完成声明 |
| 暂缓 | Eagle/M100/墨尔本接入、Alibaba 正式适配 | 仅 Alibaba 文件已找到，字段与可信标签未完成审核 |

没有引入新模型公式，新策略是原 Ridge 的不同训练条件。跨域配置默认展示全部已验证策略，旧默认清单继续保留 HDSF 和 18 个对照。

## 4. 验证及剩余范围

- 完整测试：**51 passed**，[日志](../results/analysis/cross-domain-v2/pytest-final.log)。新增 9 项契约、窗口、隔离、流程和篡改检测；第三方弃用/可选 CUDA 等警告未导致断言失败，未升级环境。
- 原双数据集 **144/144** 覆盖及独立计分通过：[记录](../results/analysis/cross-domain-v2/historical-coverage.json)。
- 历史 NLR HDSF：10 个冻结模型哈希及 4 项曲线/日均指标重算通过：[记录](../results/analysis/cross-domain-v2/hdsf-historical-recomputation.json)。未重新训练或恢复整套历史 HDSF/基座。
- 新运行 **114 项指标复算、42 项恢复、域隔离、截止及先冻结后测试**通过：[记录](figures/cross-domain-20261024T165116Z_d531e9/audit.json)。
- 合成双任务 smoke、真实 NLR 双基线 smoke、历史 Google 表及 --results latest 均检查通过。

详细命令和结论见[首轮结果](CROSS_DOMAIN_PHASE1_RESULTS.md)。历史产物保护通过不证明跨设施泛化，目前只有 Google 三个电力域的条件化 pilot。
