# PowerForecast · SAPF

**选择性自适应功率预测（Selective Adaptive Power Forecasting）研究框架。**

当前阶段为[最终方向验证协议](docs/SAPF_ROBUST_PROTOCOL.md)：先审计z-score源专家，在源侧独立开发数据选固定权重，只有24h/3d源侧存在稳定收益才开展受限动态融合。新日期测试前统一冻结所有模型，随后一次性评价。使用CPU与传统机器学习，Eagle继续封存。开发分支为`research/sapf-robust-check`。此前[Gain-v2四周协议](docs/SAPF_GAIN_V2_PROTOCOL.md)与结果作为已查看的开发证据保留。

[本轮结果与方向决策](docs/SAPF_ROBUST_RESULTS.md)已完成：固定融合在双向24h/3d平均MAPE降低约2.90%—10.01%，但H1完整统计/稳健性门槛未通过；正向3d最差节点恶化8.85%，本轮动态未超过同池固定。建议保留基础模型和简单迁移，停止继续优化当前门控。193份状态冻结、148项恢复和73,290项指标复算通过。新日期属于已知中心的离线验证，不是全新中心证明，Eagle未使用。

[四周执行结果](docs/SAPF_GAIN_V2_RESULTS.md)已完成：M100内部及正向4h有所改善，正向24h/3d恶化，反向仍全回退；简单z-score固定混合仍是强对照。73项完整测试、237状态冻结、204项恢复及14,616条指标复算通过。唯一审计器浮点比较修订有独立记录，模型和375个已有结果文件未改。工程交付完成，不代表科学门槛全部达到；[当前版本复现说明](docs/figures/gain-v2-20261025T175359Z_42a876/README.md)区分原预注册与仅用于重放的副本。

当前主线研究哪些负荷规律可共享、哪些源域值得迁移，以及如何减少未知域上的负迁移。已实现基于 Ridge 的历史窗口归一化、源域相似性、验证域可靠性，以及源内留域监督的收益/风险预测与零迁移拒绝。尚未实现完整深度共享表征架构或跨设施总表精度验收。

预测任务为 **24h / 3d**、过去 **14天** 输入。原 Google/M100 协议为15分钟：1344点历史、96/288点预测；新增 M100↔UCC 协议为30分钟：672点历史、48/144点预测。7d、HDSF 和旧预测方案转入[归档](archive/pre_sapf_bc6a439/README.md)，不参与默认运行。原始数据、历史预测、模型与缓存均保留。

Google 10 个测试电力域的 SAPF 平均曲线 MAPE 为 **24h 3.0232%、3d 3.2241%**。24h 平均优于原始多源 Ridge，但两任务最差域均更差，3d 平均也未超过 Ridge；当前不能认定已全面改善鲁棒性，完整证据见[Google结果报告](docs/SAPF_RESULTS.md)。

后续已完成 [M100 两机架跨节点实验](docs/M100_RESULTS.md)：33 节点按 12/11/10 划分源/验证/测试，专用 15min 宽表适配将 total_power_avg 从 W 转为 kW；加入归一化、源选择、历史回退及可靠性四项独立关闭实验。SAPF 优于原始 Ridge，但仍未超过前三天同刻均值；这不是跨数据中心泛化验证。

最新完成[收益预测与双向跨数据集实验](docs/BENEFIT_TRANSFER_RESULTS.md)：M100新划分8/4/10，UCC为12/8/90，旧M100测试节点退出全部新角色，三个研究在任何新测试前统一冻结。M100→UCC的24h收益策略平均MAPE为3.5849%，本地历史为3.6746%；该方向3d及反向两个时长均全部回退本地。收益门控尚未稳定超过固定混合；NLR Eagle继续封存。不同协议、日期和物理目标的分数分开解释。

## 1. 运行

Python 3.11+；优先使用已有 .venv，无需升级本机环境。首次安装只需 requirements.txt，其他基础模型的依赖见 requirements-optional.txt。

```powershell
# 最终方向验证：先源侧开发；final需预注册JSON和完整development运行路径
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/sapf_robust.yaml --phase development
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/sapf_robust.yaml --phase final --development-run results/runs/20261026T080606Z_sapf_robust_dev_8b8652
# Gain-v2：完整联合冻结；需先具备该版本的预注册JSON
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/sapf_gain_v2.yaml
# 最新：收益预测、M100新留节点及M100↔UCC；完整联合30min协议
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/sapf_transfer_suite.yaml
# M100：当前新增的真实节点功率实验，包含四项受控消融；协议与 Google 分开
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/sapf_m100.yaml
# 默认真实 Google：独立源域、验证域、测试域，运行 24h / 3d
.\.venv\Scripts\python.exe scripts/run_experiment.py
# 同一入口：合成流程检查、查看已存结果、缩小运行范围
.\.venv\Scripts\python.exe scripts/run_experiment.py --demo
.\.venv\Scripts\python.exe scripts/run_experiment.py --results latest
.\.venv\Scripts\python.exe scripts/run_experiment.py --models seasonal_naive three_day_mean ridge normalized_ridge --horizons 3d
# 完整指标只在需要时打印；默认终端展示配置、进度、平均/最差 MAPE、保存位置
.\.venv\Scripts\python.exe scripts/run_experiment.py --results latest --detailed
.\.venv\Scripts\python.exe -m pytest -q
```

默认配置为 [sapf_google.yaml](configs/sapf_google.yaml)，模型参数与默认清单集中在 [models.yaml](configs/models.yaml)。默认比较：前一天同刻、前三天同刻均值、多源 Ridge、历史窗口归一化 Ridge、相似性选择、SAPF 可靠性选择。另对每个源域保存原始/归一化单源诊断；终端只列出预先固定的首个单源，全部矩阵和指标保存在文件中。

M100 使用独立的 [sapf_m100.yaml](configs/sapf_m100.yaml) 与 [m100_models.yaml](configs/m100_models.yaml)，默认另含四项冻结设置下的单机制关闭实验。两套数据不混合拟合、不混用单位，`--results latest` 按各运行保存的协议与物理目标显示。

最新联合研究使用 [sapf_transfer_suite.yaml](configs/sapf_transfer_suite.yaml)、[transfer30.yaml](configs/datasets/transfer30.yaml) 和 [benefit_models.yaml](configs/benefit_models.yaml)，默认完整比较10种方法，包括等权纯迁移、固定50%混合、收益拒绝及继承同一设置的强制迁移消融。该协议须同时运行两个时长和全部方法；不接受缩小模型/时长或合成演示选项。`--results latest` 默认分三项研究显示同一完整运行，模型复用不因目标数据集而重新调参。

latest 只选择完成的真实 SAPF 运行。合成结果只验证程序；旧结果用明确目录查看，不并入新排行榜。

## 2. 实验协议

当前 Google 方案按每个 cell 内自然 PDU 顺序，以位置模 3 分配源域/验证域/测试域。先排除不完整采样、歧义映射、MVPP 结构和此前三个 pilot 目标。最终为 **14 源域 / 11 独立验证域 / 10 测试域**，角色互不重叠；不是同时留出独立设施。

| 项目 | 24h | 3d |
| --- | --- | --- |
| 历史/预测长度 | 1344 / 96 | 1344 / 288 |
| 源开发拟合标签截止 | D18 | D18 |
| 验证域起报 | D18/D19 | D18/D19 |
| 源最终拟合标签截止 | D22 | D22 |
| 测试域起报 | D22—D30 | D22—D28 |
| 最终源训练窗口 | 每域 8 个，共 112 个 | 每域 6 个，共 84 个 |
| 测试起报/曲线点 | 90 / 8640 | 70 / 20160 |

Dk 表示已观测前 k 个 trace 日，预测从 D(k+1) 开始。Google 按官方起点恢复实际时间，日历特征只使用相对日/周相位；目标历史与未来标签不填补，每个 15 分钟区间需 3/3 有效原始点。完整性筛选只用质量掩码，不使用误差；结论受完整观测条件限制。

所有策略、候选网格、源池和阈值在测试前声明。SAPF 只用源域拟合模型，用独立验证域选择候选及估计可靠性；全部任务最终拟合后统一保存冻结文件，再创建测试窗口。滚动推理可使用起报前目标历史，不能重新拟合或用未来标签选源。已看过的历史 Google 发布数据不等于新采集的未触碰数据。

## 3. SAPF 机制与公平对照

1. **归一化：** 每个训练或推理窗口只用过去 14 天目标计算均值和标准差，将目标历史及训练标签转换到局部尺度；预测后恢复原尺度。标准差下界为 max(0.01 × |历史均值|, 1e-6)。目标历史统计不参与全局权重拟合。
2. **相似性：** 从历史计算水平、变异系数、日/周相关、相对趋势与爬坡、日内轮廓波动。特征尺度只由源域拟合，以距离、top-k 和温度加权归一化源专家，参数只由验证域选择。
3. **可靠性选择：** 使用源专家在独立验证域的误差估计可靠性，并按目标历史与源域的距离调整迁移强度；允许回退到固定的前三天同刻均值。验证目标为域平均 MAPE + 0.1 × 最差域 MAPE，候选网格见模型配置。

MAPE、WAPE、RMSE、MAE 仍由唯一公共指标函数计算；MAPE 排除 |标签|≤1e-8 并记录计数。曲线和日均分开计分，跨域平均按域等权。负迁移在相同目标、起报和标签条件下比较固定方法；分别记录原始/归一化增加源域，以及 SAPF 相对归一化多源的变化。3d 窗口可重叠，另存每 3 天起报的非重叠检验。

验证集上用于选参的 SAPF 分数是选择期分数，不能当作无偏泛化成绩。最终结论须同时看测试平均、最差域及负迁移比例，不能只引用较好域或将相关性写成因果。

30分钟扩展的[冻结协议](docs/BENEFIT_TRANSFER_PROTOCOL.md)另行规定72天区间、全局节点角色、源内留域收益标签及外部验证。收益模型仅用训练节点拟合，42维特征仅来自历史、源内先验和模型预测；无合格正收益源时权重严格为0。UCC原始时区未核实，保留原钟表字符串；内部合成坐标不是实际UTC。两数据集采集年份不同，结果属于离线迁移基准，不能声称M100→UCC是2019年可部署的因果回测。

## 4. 数据边界

| 数据 | 用途与当前状态 |
| --- | --- |
| Google PowerData 2019 | 当前机制研究；功率利用率/电力域，不能改称设施总表 kW |
| Alibaba GPU v2020 | 本地 7 张表的无表头 schema 已适配、官方哈希核对；资源统计无可信功率标签，全生命周期统计不能提前使用 |
| M100 ExaData | 15min宽表与共同30min适配均已完成，total_power_avg由W→kW；两个机架内跨节点及与UCC双向离线实验，保留聚合边界和上报时延假定 |
| UCC2023 | 原生10min Power服务器W，严格3槽聚合30min，110个完整节点按12/8/90划分；选择避开重复小时的固定区间，源时区仍未核实 |
| NLR Eagle | 整体保留为最终独立测试域，禁止训练、验证、选参和策略设计；本轮未打开观测或标签 |
| NLR ESIF | 设施五分项 component_sum、kW、UTC 未核实；只作物理契约与旧实验参考，区别于 Eagle |

来源与当前已实现适配契约见 [registry.yaml](configs/datasets/registry.yaml)，Gain-v2版本在本轮测试前更新并预注册；旧状态仍保存在基线提交f89dc7d及各轮快照。详见[多源核验](docs/SAPF_DATA_AUDIT.md)、[原始文件审计](docs/M100_UCC2023_DATA_AUDIT.md)和[Gain-v2协议](docs/SAPF_GAIN_V2_PROTOCOL.md)。原始审计与旧预测保持各自证据版本，不能以当前源码冒称旧运行原始哈希。

```powershell
# 只核对本地数据，不下载外部大数据
.\.venv\Scripts\python.exe scripts/audit_datasets.py --hashes --output results/analysis/sapf/datasets-recheck.json
# 全量审计本地 M100/UCC 功率、原生采样、缺口、重复和连续长度；目录必须为新路径
.\.venv\Scripts\python.exe scripts/audit_power_sources.py --output results/analysis/power_sources/recheck_01
# 通用预处理必须显式选择配置与新的版本路径；已有产物拒绝覆盖
.\.venv\Scripts\python.exe scripts/prepare_data.py --config configs/datasets/nlr.yaml --output data/processed/nlr_esif_audited_v2.parquet
```

## 5. 架构、产物与复现

| 位置 | 职责 |
| --- | --- |
| scripts/run_experiment.py | 唯一实验与结果入口；显式访问归档 |
| src/data/ | 物理契约、逐域适配、14 天窗口、角色/时间隔离、历史特征 |
| src/data/m100_panel.py | M100 15min 宽表、节点身份、W→kW、缺失掩码、发布槽可用性约束 |
| src/data/transfer30.py | M100/UCC完整原生槽的30min聚合、原始钟表溯源、全局留域角色 |
| src/data/contracts.py / gain_v2_panel.py | DataAdapter统一物理/时钟/质量/available_at契约；角色对应原始日期，Google桥接及M100/UCC15/30min检查 |
| src/models/ml/normalized.py / sapf.py / ablation.py | 可保存恢复的归一化、选择性机制与冻结设置下的独立消融，预测不接收标签 |
| src/models/ml/benefit.py / src/training/benefit_learning.py | 源内留域收益/风险监督、历史表现先验、精确零迁移与受控强制迁移对照 |
| src/training/transfer_suite.py / src/evaluation/transfer_audit.py | 双向源组共同冻结、相同M100模型复用、跨数据集目标标签隔离和统一审计 |
| src/models/ml/level_shape.py / gain_v2.py | 四种Ridge表示、源内监督的四种收益目标、独立校准、两步互补融合与受控干预 |
| src/training/gain_v2_experiment.py / src/evaluation/gain_v2_audit.py | Gain-v2四研究/三时长统一冻结；共享紧凑产物、恢复及逐域评分复算 |
| src/evaluation/statistics.py / gain_v2_diagnostics.py | 非重叠3d、域级/块bootstrap、HAC-DM、Holm、正迁移识别与校准曲线 |
| src/training/experiment.py / lifecycle.py | 源拟合、验证选择、最终拟合、全任务冻结、测试 |
| src/evaluation/ | 公共评分/归档、紧凑输出、配对诊断与独立复算 |
| archive/pre_sapf_bc6a439/ | 旧源码压缩包、逐文件哈希、44 次历史运行索引及产物保护清单 |

每次运行建立独立 results/runs/<UTC>_sapf_<标识>/，保存完整配置、角色、原始来源哈希、源码摘要、开发/最终模型、验证选择、frozen-before-test.json、test-start.json、完整预测、曲线/日均/逐起报指标、源权重、相似性与误差分析、恢复和复算记录。错误运行保留 failed 状态，不标成完成。旧模型、数据和预测不写回。

新增模型须实现 BaseForecaster.fit(WindowBatch) / predict(ForecastInputs)，默认清单、公共评分、冻结/恢复与真实 CLI 一起验证。history=[N,L,C]、future_known=[N,H,K]、targets=[N,H]；目标在历史第 0 列。域身份是侧边元数据，数组按只读使用。未来输入只包含已知日历，不加入未来天气、IT/GPU 或任务事后统计。

```powershell
# 对 SAPF 已存预测复算（output 必须为新路径）
.\.venv\Scripts\python.exe scripts/audit_cross_domain.py --run results/runs/<SAPF运行目录> --output results/analysis/sapf/recheck.json
# 同一入口显式打开冻结的旧 Google/HDSF 流程；这里只显示帮助
.\.venv\Scripts\python.exe scripts/run_experiment.py --legacy-config configs/google_hdsf.yaml --help
# 历史分数按原记录显示，不重训
.\.venv\Scripts\python.exe scripts/run_experiment.py --results results/runs/20261014T191923Z_google_9a3ad6
```

原Google设计/结果见 [SAPF_PROTOCOL.md](docs/SAPF_PROTOCOL.md) 和 [SAPF_RESULTS.md](docs/SAPF_RESULTS.md)；最新联合研究见 [BENEFIT_TRANSFER_PROTOCOL.md](docs/BENEFIT_TRANSFER_PROTOCOL.md) 与 [BENEFIT_TRANSFER_RESULTS.md](docs/BENEFIT_TRANSFER_RESULTS.md)。完整深度共享表征、更广泛中心验证及 Eagle 最终验证仍属后续工作。
