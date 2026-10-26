"""Compact SAPF console output; complete metrics remain in shared artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pandas as pd

from src.utils import resolve_path

METHOD_NAMES = {
    "naive": "最后值", "seasonal_naive": "前一天同刻", "three_day_mean": "前三天同刻均值",
    "ridge": "多源 Ridge", "extra_trees": "ExtraTrees", "normalized_ridge": "历史窗口归一化 Ridge",
    "normalized_extra_trees": "历史窗口归一化 ExtraTrees", "sapf_similarity": "相似性选择",
    "sapf": "SAPF 可靠性选择", "dlinear": "DLinear", "lstm": "LSTM", "tcn": "TCN",
    "seasonal_ar": "季节 AR", "sarima": "SARIMA", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "sapf_no_normalization": "SAPF 关闭归一化", "sapf_no_selection": "SAPF 关闭源域选择",
    "sapf_no_fallback": "SAPF 关闭历史回退", "sapf_no_reliability": "SAPF 关闭可靠性加权",
    "uniform_transfer": "等权纯迁移", "fixed_blend": "固定50%历史混合",
    "sapf_gain": "收益预测与零迁移拒绝", "sapf_gain_no_veto": "收益策略强制非零迁移",
    "ridge_raw": "Ridge 原始尺度", "ridge_zscore": "Ridge 局部z-score",
    "ridge_level_shape": "Ridge Level+Shape双通道", "ridge_scale_metadata": "Ridge z-score+均值/尺度",
    "fixed_blend_25": "固定25%迁移", "fixed_blend_50": "固定50%迁移", "fixed_blend_75": "固定75%迁移",
    "gain_v2_single": "Gain-v2 单专家标签", "gain_v2_fixed": "Gain-v2 固定融合标签",
    "gain_v2_marginal": "Gain-v2 边际融合标签", "gain_v2_rank": "Gain-v2 配对排序",
    "gain_v2_forced": "Gain-v2 强制迁移消融", "gain_v2_uncalibrated": "Gain-v2 关闭校准消融",
    "oracle_posthoc": "事后oracle（不可部署）",
}


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _number(value):
    return f"{value:.4f}%" if value is not None and math.isfinite(float(value)) else "N/A"


def print_results(metrics, *, manifest, windows=None, fusions=None, detailed=False):
    configure_console()
    print(f"\n配置：{manifest.get('research_method', manifest.get('protocol', '归档实验'))} | "
          f"数据={manifest['data']} | 单位={manifest['target_unit']}")
    if manifest.get("is_demo"):
        print("合成数据：只验证程序，不作为研究结果。")
    roles = manifest.get("domain_roles", {})
    if roles:
        cadence = manifest.get("experiment", {}).get("frequency", "15min")
        print(f"源域 {len(roles['source'])} / 独立验证域 {len(roles['validation'])} / 测试域 {len(roles['test'])}；"
              f"14 天历史，{cadence}采样，测试前冻结。")
        if 'calibration' in roles:
            print(f"另有独立校准域 {len(roles['calibration'])}，与策略验证域分开；新时段测试，节点身份曾见。")
    rows = metrics[(metrics.split == "test") & (metrics.evaluation == "curve")]
    methods = manifest.get("console_methods") or rows.model.drop_duplicates().tolist()
    for task in manifest["requested_horizons"]:
        print(f"\n{task} | 方法 | 平均 MAPE | 最差域 MAPE | 较本地历史更差域")
        part = rows[rows.horizon == task]
        for method in methods:
            values = part.loc[part.model == method, "MAPE"]
            if values.empty:
                continue
            label = manifest.get("method_names", {}).get(method, METHOD_NAMES.get(method, method))
            if not manifest.get("protocol", "").startswith("sapf_"):
                label = manifest.get("method_names", {}).get(method, method)
            worst = values.max() if "target_domain" in part else None
            worse = "N/A"
            if "target_domain" in part and method not in {"naive", "seasonal_naive", "three_day_mean"}:
                local = part[part.model == "three_day_mean"][["target_domain", "MAPE"]]
                matched = part[part.model == method][["target_domain", "MAPE"]].merge(local, on="target_domain", suffixes=("", "_local"))
                if len(matched):
                    count = int((matched.MAPE > matched.MAPE_local+1e-10).sum())
                    worse = f"{count}/{len(matched)} ({100*count/len(matched):.1f}%)"
            print(f"{task} | {label} | {_number(values.mean())} | {_number(worst)} | {worse}")
    if manifest.get('protocol')=='sapf_gain_v2':
        nonoverlap=metrics[(metrics.split=='test')&(metrics.evaluation=='curve_nonoverlap')]
        print('\n3d 非重叠主检验 | 方法 | 平均 MAPE | 最差域 MAPE')
        for method in methods:
            values=nonoverlap.loc[nonoverlap.model==method,'MAPE']
            if not values.empty:print(f"3d | {METHOD_NAMES.get(method,method)} | {_number(values.mean())} | {_number(values.max())}")
        print('4h不含完整日均值，daily_mean显示N/A；oracle仅为事后有限候选上界。')
    if detailed:
        columns = [c for c in ["target_domain", "horizon", "model", "MAPE", "WAPE", "RMSE", "MAE", "n"] if c in rows]
        print("\n完整曲线指标（WAPE/MAPE 为 %，RMSE/MAE 为目标单位）：")
        print(rows[columns].to_string(index=False))
    print("\n完整曲线/日均 MAPE、WAPE、RMSE、MAE 和样本数保存在 metrics.csv。")
    if "target_domain" in rows:
        print("更差域比例按前三天同刻均值比较，是负迁移的性能代理；不是源域作用的因果证明。")
    if "target_domain" not in rows:
        print("归档只有聚合指标时，最差域显示 N/A；不从聚合分数反推逐域结果。")


def print_saved_results(run_dir, *, results_dir="results/runs", detailed=False):
    if str(run_dir) == "latest":
        candidates = []
        for path in resolve_path(results_dir).glob("*/manifest.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (manifest.get("status") == "complete" and manifest.get("protocol", "").startswith("sapf_")
                    and not manifest.get("is_demo", False)):
                candidates.append(path)
        if not candidates:
            raise FileNotFoundError("No completed real SAPF result; legacy runs require an explicit result path.")
        run_dir = max(candidates, key=lambda p: p.parent.name).parent
    folder = resolve_path(run_dir)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Display requires a completed experiment.")
    if manifest.get('protocol')=='sapf_robust_direction_v1':
        from src.evaluation.robust_report import print_robust_results
        print_robust_results(folder)
        return folder
    if manifest.get("protocol") in {"sapf_transfer_suite_v1", "sapf_gain_v2_suite"}:
        for spec in manifest["studies"]:
            child = folder/"studies"/spec["name"]
            child_manifest = json.loads((child/"manifest.json").read_text(encoding="utf-8"))
            print_results(pd.read_csv(child/"metrics.csv"), manifest=child_manifest, detailed=detailed)
        print(f"结果保存：{folder}", flush=True)
        return folder
    if not manifest.get("protocol", "").startswith("sapf_"):
        print("归档结果：按保存的原始协议和指标显示，不重训、不重新贴标签。")
    print_results(pd.read_csv(folder/"metrics.csv"), manifest=manifest, detailed=detailed)
    print(f"结果保存：{folder}", flush=True)
    return folder
