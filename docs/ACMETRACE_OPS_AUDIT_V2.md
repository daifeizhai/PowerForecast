# AcmeTrace 运行指标轻量审计报告（power-transfer-v2）

> 分支：`research/power-transfer-v2`；数据快照：`E:/Data/AcmeTrace`（只读；Windows + pwsh）。本轮未 commit/push，不修改 `src/`、`configs/`、`tests/`、`README` 与其它文档，不向 `data/raw/` 或 E: 原始树写任何文件。
> 主命令：`python -m scripts.audit_acmetrace_ops --clusters seren --metrics NODE_CPU_UTILIZATION NODE_MEMORY_UTILIZATION NODE_IB_RECEIVE NODE_IB_SEND`。
> 主审计产物生成：**2026-10-28T07:06:59Z（15:06:59 +08:00）**；主命令实际读取 **≈2.73 GB**（≈2.54 GiB）；§5/§6 的补充计算另读 ≈7.87 GB（含 `GPU_UTIL` 5.14 GB），不属于受限白名单。 
> 数字来源：`data/processed/acmetrace_ops_audit.json`（139,399 B）等 3 个 JSON + `results/analysis/acmetrace/` 下 CSV/日志；标“补充”的数字来自 §5/§6 的只读补充计算 `scripts/analyze_acmetrace_linkage.py`（生成时间见 `v2_ops_supplement.json` 的 `generated_at_utc`；最近一次重跑与首次数字完全一致），不是官方审计载荷字段。

## 1. 目的与范围

本轮要回答三个准入问题：Seren 的 4 个 `NODE_*` 运行指标是否足够干净可作目标域辅助特征；IPMI 服务器功率（`Sys_Total_Power`）能否与它们按 15 分钟配对；Seren job trace 哪些字段能作预测特征、哪些是事后字段。

全量审计（17 个利用率指标 + IPMI + job trace，Seren 单集群）需读 ≈68.0 GB（Seren 利用率 66.98 GB + IPMI 0.91 GB + job trace 0.10 GB）；全树（含 Kalos）为 80.8 GB。本轮按任务要求**只读 Seren 的 4 个 `NODE_*` 指标**（1.72 GB）+ 3 个 IPMI 文件（0.91 GB）+ `trace_seren.csv`（0.10 GB），合计 ≈2.73 GB，约为全量 Seren 运行指标审计的 **4.0%**、全树的 **3.4%**。因此：

- **这是受限白名单审计，不是全量 17 指标审计**。`acmetrace_ops_audit.json` 的 `notes` 原文记录了
  “PARTIAL util-metric scan: only NODE_CPU_UTILIZATION, NODE_IB_RECEIVE, NODE_IB_SEND, NODE_MEMORY_UTILIZATION were audited … all other utilization metrics were NOT read”；
- 其余 13 个利用率指标（含 `GPU_UTIL`、`POWER_USAGE`、`FB_USED`、`XID_ERRORS` 等）**未读取、未审计**；Kalos 集群本轮完全未扫描；
- 所有原始 CSV 均以流式分块只读方式打开，NaN 缺失保持缺失，不插值、不零填、不伪造连续序列；
- 三个 JSON 均**不含 `bounded_debug` 字段**（该字段只在 `--max-chunks` 试点出现），即本轮是对 4 个文件的全文件扫描。

## 2. 方法
- **流式统计**：逐文件分块读；每设备统计缺失比例、取值域、越界计数；原生采样间隔 = 相邻时间戳差值的众数（15 s 验证通过）。
- **15 分钟配对**：按 900 s bin 对每设备取 bin 内均值（sum / 观测数）；`Sys_Total_Power == 0` 视为无效/离线观测，上游排除且绝不当作真实 0 W；IPMI 300 s 与节点指标 15 s **无原生同刻样本**，只能做 15 分钟 bin 配对；`matched` = 两侧在同一 (设备, bin) 都有观测，`match_rate = matched slots / union slots`。
- **作业审计**：13 字段；严格区分 pre-origin 与 post-hoc 字段；时间戳按显式 offset 解析后转 UTC 再比较；重算 duration/queue/gpu_time 做一致性核验；job trace 无 node ID，只允许集群级 linkage。
- **单位**：功率单位 “W” 是量级证据推断，所有指标 `unit_verified=false`（源 README 无单位）。

## 3. 运行指标证据表（4 个 NODE_*，Seren）

| 指标 | rows | 设备列 | native interval | 缺失比例（聚合 / 单设备最大） | 取值域 min–max（median / p99） | 时间戳偏移 |
|---|---:|---:|---:|---:|---|---|
| `NODE_CPU_UTILIZATION` | 282,119 | 248 | 15 s | 0.117% / 3.42% | 0.032 – 100.0（7.35 / 55.52） | +08:00 |
| `NODE_MEMORY_UTILIZATION` | 281,542 | 249 | 15 s | 0.119% / 3.44% | 0.929 – 99.983（19.65 / 49.22） | +08:00 |
| `NODE_IB_RECEIVE` | 281,581 | 249 | 15 s | 0.131% / 3.45% | 0 – 23,689.02（0 / 6,993.66） | +08:00 |
| `NODE_IB_SEND` | 281,581 | 249 | 15 s | 0.131% / 3.45% | 0 – 23,695.48（0 / 6,992.56） | +08:00 |

合计 **1,126,823 行 / 4 文件**；整体缺失 **0.125%**（349,377 / 280,296,808 个取值），`fully_missing_devices=0`；重复时间戳 0，乱序行 0；gap 10–12 个（最大 gap：CPU 375 s，内存 8,790 s，IB 8,100 s）；时间跨度 **2023-07-01T00:00:00Z → 2023-08-19T00:00:00Z**（原始 offset 均为 `+08:00`，已转 UTC）。设备集合：CPU 为 248 个设备，内存/IB 为 249 个（多出 `10.140.0.245`，见 `device_sets.csv`）；4 个文件设备集无缺失差异。越界：CPU/内存超出 [0,100] 的值为 0，负值 0；IB 为无上界量，负值 0；XID 不适用。取值域 min–max 来自全文件扫描，median/p99 来自 20 万值抽样（`value_evidence`），单位未核验。

## 4. IPMI 服务器功率与 15 分钟配对

| IPMI 文件 | rows | 节点数 | native interval | 300 s 网格覆盖 | `Sys_Total_Power` 范围（推断 W） | 0 值行数 |
|---|---:|---:|---:|---:|---|---:|
| `CPU_D_Power` | 168,804 | 8 | 300 s | 0.848 | 468 – 960 | 0 |
| `GPU_AB_Power` | 2,708,675 | 113 | 300 s | 0.957 | 0 – 6,554 | 2 |
| `GPU_C_Power` | 4,174,302 | 170 | 300 s | 0.987 | 0 – 6,363 | 1 |

合计 **7,051,781 行 / 291 个节点**（8+113+170）；与节点指标的共同设备：CPU 248、内存/IB 249；IPMI-only 节点 42–43 个。3 个 0 W 行按无效/离线排除，不参与功率统计与配对。`unit_guess=W_unverified`、`unit_verified=false`；`CPU_D_Power` 的 `CST` 时间戳按 `+08:00` 假设解析。官方 `acmetrace_pairing_audit.json` / `pairing_summary.csv` 的 15 分钟配对（模块只对 CPU、内存、IB-RECEIVE 三个指标做节点级配对）：

| 配对（IPMI ~ 指标） | 共同设备 | 匹配设备 | 匹配设备-槽 | union 槽 | match rate | 共同 bins | 联合跨度（UTC） |
|---|---:|---:|---:|---:|---:|---:|---|
| `Sys_Total_Power` ~ `NODE_CPU_UTILIZATION` | 248 | 246 | 917,463 | 942,983 | 0.9729 | 3,804 | 2023-07-01T00:00Z → 2023-08-09T15:45Z |
| `Sys_Total_Power` ~ `NODE_MEMORY_UTILIZATION` | 249 | 247 | 918,952 | 944,546 | 0.9729 | 3,795 | 同上 |
| `Sys_Total_Power` ~ `NODE_IB_RECEIVE` | 249 | 247 | 918,963 | 944,793 | 0.9727 | 3,796 | 同上 |

**`NODE_IB_SEND` 不在模块的 `NODE_PAIRING_METRICS` 中（仅 CPU / 内存 / IB_RECEIVE），因此官方 pairing 载荷与 CSV 没有 IB_SEND 行**；这是模块定义的配对白名单，不是本轮运行失败。同理，`GPU_UTIL`/`POWER_USAGE` 未扫描，`pairing_audit.json` 的 `gpu_pairs` 为空数组。IB_SEND 的配对覆盖与 Pearson 见 §5 补充计算。`unsafe_pairings` 明确列出：Kalos 无 IPMI；GPU 功率与服务器功率是不同测量层级、不融合；job trace 无 node ID；IPMI 300 s 与节点 15 s 无法原生同刻配对。

## 5. 补充：`Sys_Total_Power` 与各 NODE_* 的 15 分钟逐节点 Pearson

官方审计载荷只报告覆盖匹配槽，不报告相关系数。为使“功率—运行指标”证据完整（并覆盖 `NODE_IB_SEND`），本轮用模块自身的扫描/分箱函数做了**只读补充计算**（bins、0 W 排除、15 分钟均值口径与 §4 完全一致；每节点要求 ≥96 个匹配槽，即 ≥24 h）：脚本 `scripts/analyze_acmetrace_linkage.py`，产物 `results/analysis/acmetrace/v2_node_power_pearson.csv`、`v2_node_power_pearson_by_node.csv`、`v2_ops_supplement.json`。

| 指标 | 匹配设备-槽 | match rate | 参与节点数 | 逐节点 Pearson 中位 | p10 – p90 | r > 0 节点 | r ≥ 0.5 节点 |
|---|---:|---:|---:|---:|---|---:|---:|
| `NODE_CPU_UTILIZATION` | 917,463 | 0.9729 | 236 | **0.3393** | 0.091 – 0.631 | 230 | 61 |
| `NODE_MEMORY_UTILIZATION` | 918,952 | 0.9729 | 237 | **0.4139** | 0.139 – 0.664 | 236 | 88 |
| `NODE_IB_RECEIVE` | 918,963 | 0.9727 | 237 | **0.5460** | 0.126 – 0.814 | 224 | 126 |
| `NODE_IB_SEND` | 918,963 | 0.9727 | 237 | **0.5461** | 0.126 – 0.814 | 224 | 126 |

（逐节点 Pearson 范围：CPU −0.096…0.849、内存 −0.006…0.808、IB-RECEIVE −0.204…0.886、IB-SEND −0.205…0.886；节点数 236/237 是 ≥96 匹配槽的节点数，少于匹配设备数 246/247。）注意统计量口径：上表是**设备级时间共动**（每个节点各自的时间序列相关，再汇总中位数）；目标域报告 §4 的 Pearson 是**集群均值池化**相关。两者不可互相替换引用；二者都是描述性证据，不是预测收益或因果结论。

## 6. 作业 trace 审计（Seren）
`E:/Data/AcmeTrace/data/job_trace/trace_seren.csv`：**99,262,320 B / 818,327 行 / 13 字段**。字段完整性：12 个字段非空 100%，`gpu_time` 非空 450,590（44.94% 为空，CPU-only 作业 367,737 条）。时间范围（UTC）：`submit_time` 2023-02-28T16:18:22Z → 2023-08-16T15:59:42Z；`end_time` 最晚 2023-08-22T07:01:59Z。

- **pre-origin 字段 7 个**：`job_id`、`user`、`node_num`、`gpu_num`、`cpu_num`、`type`、`submit_time`；
  **post-hoc 字段 6 个**：`state`、`start_time`、`end_time`、`duration`、`queue`、`gpu_time`。post-hoc 字段不得作为未来特征（`job_field_classification.csv`）。
- **时区**：job trace 的 submit/start/end 原始 offset 均为 `+08:00`，本轮 4 个利用率指标 offset 也为 `+08:00`；规则是“按显式 offset 解析 → 转 UTC → 再比较”，因此若忽略 offset 的错位为 0 h。但站点时区本身无官方文档（`timezone_verified=false`），`CST` 字面量按 `+08:00` 假设解析。
- **一致性**：duration 时长重算 mismatch 0、queue 重算 mismatch 0、gpu_time 重算 mismatch 0；`end<start` 0、负 queue 0；`duration=0` 共 **11,776** 行（保留不删）；资源申请范围 `node_num` 1–156、`gpu_num` 0–1024、`cpu_num` 最大 9,216。
- **state 分布**：COMPLETED 523,350 / FAILED 190,398 / CANCELLED 60,467 / PREEMPTED 43,838 / NODE_FAIL 199 / TIMEOUT 75；**type 分布**：CPU 367,737 / Eval 292,212 / SFT 66,971 / Other 65,624 / MLLM 13,288 / Debug 8,469 / Pretrain 4,026。
- **linkage**：受限白名单未扫描 `GPU_UTIL`，模块只在存在 GPU 利用率累加器时计算集群级时序关联，因此官方 `acmetrace_job_audit.json` 的 `cluster_linkage` 为 **null**、`cluster_linkage.csv` 为空（2 B）。补充计算（模块同一函数，完整 Seren `GPU_UTIL` 5,139,607,278 B / 538,469 行 + 完整 job trace 818,327 行）结果：**8,979 个 15 min bins 全部重叠，Pearson（活跃 GPU 作业数 vs 平均 GPU 利用率）= 0.8671**；平均 GPU 利用率 58.87%，活跃 GPU 作业数均值 1,268.95 / 最大 1,926；util span 2023-05-16T09:30Z → 2023-08-18T00:00Z，job span 2023-02-28T16:18:22Z → 2023-08-22T07:01:59Z。仍只是**集群级**关联：job trace 无 node ID，无法做设备级作业↔指标配对。

## 7. 诚实性限制（引用前必读）

1. **受限审计 ≠ 全量审计**：只审计 4/17 个利用率指标；`GPU_UTIL`、`POWER_USAGE` 等 13 个指标与 Kalos 集群未读取，本报告不能声称“AcmeTrace 运行指标全量审计通过”。
2. **单位未核验**：“W”由量级推断（CPU 服务器 ≈0.47–0.96 kW、8 卡服务器最高 ≈6.5 kW 等），源 README 无单位；`NODE_IB_*` 的 bytes 或 bytes/s 语义未定。
3. **时区未核验**：显式 `+08:00` 已正确解析并转 UTC，但站点时区无官方文档；`CPU_D_Power` 的 `CST` 按 `+08:00` 假设。
4. **无设施功率**：AcmeTrace 只有服务器 IPMI 与 DCGM GPU 功率，没有设施总表；服务器功率 ≠ 设施功率，二者与 GPU 功率是不同测量层级、不融合。
5. **非完整集群量**：配对只覆盖上报设备；291 个 IPMI 节点中 42–43 个 IPMI-only 不在节点指标清单；节点指标缺失与覆盖见 §3。
6. **job trace**：无 node ID → 只能集群级关联；post-hoc 字段不得用作未来特征；`gpu_time` 44.94% 为空、`duration=0` 11,776 行；trace 自身完整性不可独立验证。
7. **补充计算边界**：§5 的 Pearson 与 §6 的 linkage 是本轮补充只读计算（复用 `src/data/acmetrace_ops.py` 的函数，未改代码），**不是**受限审计官方载荷字段；描述性相关不等于预测收益或因果。
8. **历史产物**：2026-10-27 的 bounded pilot（每文件仅前 3 个 chunk）与本轮同名 JSON/CSV 已被本轮全文件扫描覆盖；`pairing_span_corrections.json` 是试点时代的 sidecar（解释旧版 1970 年时间戳显示 bug），不作为本轮证据引用。

## 8. 复现命令

```powershell
# 1) 受限运行指标审计（主命令；本轮产物即由此产生）
& .\.venv\Scripts\python.exe -m scripts.audit_acmetrace_ops --clusters seren `
    --metrics NODE_CPU_UTILIZATION NODE_MEMORY_UTILIZATION NODE_IB_RECEIVE NODE_IB_SEND `
    2>&1 | Tee-Object -FilePath results\analysis\acmetrace\v2_ops_audit_run.log

# 2) 补充只读计算（IB_SEND 配对覆盖 + 逐节点 Pearson + 集群级 linkage）
& .\.venv\Scripts\python.exe -m scripts.analyze_acmetrace_linkage `
    2>&1 | Tee-Object -FilePath results\analysis\acmetrace\v2_ops_supplement_run.log
```

原始树位置可用 `--raw-root` 或 `ACMETRACE_RAW_ROOT` 覆盖（默认 `E:/Data/AcmeTrace`）。`--max-chunks` 是调试选项，其结果不得引用为全量覆盖。

## 9. 产物清单

| 产物 | 大小 | 内容 |
|---|---:|---|
| `data/processed/acmetrace_ops_audit.json` | 139,399 B | 4 指标逐文件记录、summary、PARTIAL 说明 |
| `data/processed/acmetrace_pairing_audit.json` | 41,753 B | 3 个节点级配对、IPMI 文件审计、timezone_table、unsafe_pairings |
| `data/processed/acmetrace_job_audit.json` | 8,055 B | job 行数/字段/时区/plausibility（`cluster_linkage=null`） |
| `results/analysis/acmetrace/runtime_file_summary.csv` / `pairing_summary.csv` | 1,820 B / 694 B | 4 个 NODE_* + 3 个 IPMI 统计；3 行官方节点级配对结果 |
| `results/analysis/acmetrace/job_field_classification.csv` / `device_sets.csv` | 1,296 B / 344 B | 13 字段 pre-origin / post-hoc 分类；节点指标设备集差异 |
| `results/analysis/acmetrace/cluster_linkage.csv` | 2 B | 空：受限审计未计算 linkage |
| `results/analysis/acmetrace/v2_ops_audit_run.log` | 2,164 B | 主命令控制台日志 |
| `scripts/analyze_acmetrace_linkage.py` | 15,177 B | 补充只读计算脚本（§5/§6 可复现入口） |
| `results/analysis/acmetrace/v2_ops_supplement.json` / `v2_node_power_pearson.csv` / `v2_node_power_pearson_by_node.csv` / `v2_ops_supplement_run.log` | 7,508 B / 2,034 B / 60,234 B / 1,139 B | 补充计算的元数据、逐指标 Pearson、逐节点明细与日志（最近一次重跑） |
| `results/analysis/acmetrace/pairing_span_corrections.json` | 1,950 B | 历史 pilot sidecar，不用于本轮 |

原始树：`E:/Data/AcmeTrace`（Seren 利用率 17 文件共 66.98 GB；本轮只读其中 4 个文件 1.72 GB）。
