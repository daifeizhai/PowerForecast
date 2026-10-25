# Gain-v2四周阶段证据

对应 [SAPF_GAIN_V2_RESULTS.md](../../SAPF_GAIN_V2_RESULTS.md) 与[冻结协议](../../SAPF_GAIN_V2_PROTOCOL.md)。原运行位于`results/runs/20261025T175359Z_sapf_gain_v2_42a876`，所有模型和完整预测仍在原目录。此目录不含大型模型、原始数据或新的模型拟合。

| 文件 | 口径 |
| --- | --- |
| manifest / predeclared-plan / preregistration / frozen-before-test | 原始方案、全局角色和237个状态的冻结；预注册时85份计算文件 |
| audit.json / audit-amendment.json / failed-manifest-before-audit-repair.json | 全部审计及唯一审计修订；375个原结果文件哈希不变，原失败不隐藏 |
| metrics.csv / all-domain-summary.csv | 14,616条测试记录、全部18方法、曲线/日均/非重叠3d；4h日均为N/A、n=0 |
| studies/ | 逐研究指标、风险、曲线、源→目标配对、oracle选择、恢复及开始记录 |
| matrices/ / source-target-matrices-long.csv | 48张单源→目标MAPE矩阵；按研究、时长、四种表示分开 |
| groups/ | 纯源收益训练表、历史性能先验、独立校准数据/曲线、选择候选与训练记录 |
| value-training-summary.csv / calibration-parameters.csv | 样本数、排序比较数和独立校准系数，不是独立中心数 |
| all-risk-summary.csv / all-calibration-curves.csv | 第0步共同边际收益定义的识别、校准与迁移成功指标 |
| statistics.csv / risk-statistics.csv | 节点/块bootstrap、非重叠、辅助DM、Holm及风险cluster区间 |
| posthoc-representation-mixtures*.csv | 测试后将四个冻结专家池与固定α网格交叉，仅描述性、不选模型、不进入主检验族 |
| independent-verification.json / verify_independently.py | 独立原生槽、质量位、NumPy指标、手工专家、路由和oracle检查；不fit |
| pytest-final.log / pytest-audit-repair.log | 预测前73项全量测试及审计修复后单项完整端到端回归 |
| real-run.log / results-latest.log | 原运行及正常CLI；原日志的审计失败保留，UTF-8副本便于阅读 |
| paired-effects.png | 预声明边际主方法的条件节点/块区间；3d用非重叠主检验 |
| representation-mixtures.png | 事后表示交叉诊断；不冒称预注册结论 |
| risk-calibration.png | 新测试固定分箱的原始/校准风险曲线，配对数和cluster区间另存 |
| export_evidence.py | 只读导出和作图脚本，拒绝覆盖既有证据目录 |
| frozen-current-source.tar.gz / replication-preregistration.json / environment.json | 含审计补丁的精确源码、仅供重放的预注册及环境版本 |
| evidence.json / delivery-verification.json | 本目录哈希、原结果保护和交付核对；不自递归哈希 |

运行独立复核（原运行和原始数据须在本机，output必须为新路径）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -W ignore::FutureWarning docs/figures/gain-v2-20261025T175359Z_42a876/verify_independently.py --output results/analysis/gain_v2/independent-new.json
```

严格复现的原模型计算源码见[预测前快照](../gain-v2-prereg-20261026/frozen-computational-source.tar.gz)，当前快照唯一科学流程差异是修正审计中的精确回退判据。审计修改不改变拟合、推理、评分或统计。Git在Windows可能转换源码行尾，严格字节复现应使用压缩包，并使用environment.json对应依赖。

在**新checkout/新工作目录**中重放当前含审计修复的流程：恢复本地原始数据到配置路径，按需解压当前源码快照，再将`replication-preregistration.json`复制到`results/analysis/gain_v2/preregistration.json`，运行统一入口：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/sapf_gain_v2.yaml
```

不要覆盖既有预注册或结果。这个副本明确标记为已知测试上的复现，不能再次宣称新独立泛化测试；原预注册和协议标签保留原样。

UCC源时区未知、开发日期晚于新的目标日期，结果是离线基准。M100仅两个机架；两个中心内的节点不等于这么多个独立中心。目标为计算节点/物理服务器功率，不能作为设施总有功功率验收。Eagle保持封存。
