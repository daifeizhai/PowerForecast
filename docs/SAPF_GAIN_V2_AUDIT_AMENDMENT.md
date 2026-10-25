# Gain-v2末端审计修订

本轮一次拟合与全部测试预测完成后，原运行在审计器的零迁移检查处进入failed。检查要求`final_MAPE == local_MAPE`逐位相等；实际相同的预测数组因C/F内存布局导致浮点求和顺序不同，最大差2.842170943040401e−14个MAPE百分点。

直接读取已存预测验证后，所有零迁移曲线与本地基线逐点完全一致。因此仅修复 [gain_v2_audit.py](../src/evaluation/gain_v2_audit.py)，改为核对相同域、起报和lead的完整预测值精确相等。未改指标公式、预测、训练、校准、阈值、表示、统计或任何源/测试数据。

原[失败manifest](figures/gain-v2-20261025T175359Z_42a876/failed-manifest-before-audit-repair.json)和[修订记录](figures/gain-v2-20261025T175359Z_42a876/audit-amendment.json)保留。修订记录包含唯一许可文件的前后SHA-256、全局冻结哈希、375个已有结果文件的保护哈希。当前审计严格拒绝其他计算文件变动；报告明确`computational_source_unchanged=false`和唯一审计例外，而不改写原预注册源码摘要。

修复后的公共审计、原四研究端到端fixture回归与独立原生数据/指标/门控重建均通过。独立核对包含11,047个零迁移方法×起报，最大预测差0 kW。运行状态仅在只读复核通过后恢复complete，原错误作为`original_execution_error`保留；没有第二次模型拟合或重新挑选结果。
