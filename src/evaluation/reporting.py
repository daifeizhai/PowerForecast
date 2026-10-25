"""Terminal tables following PowerData_Prediction.pipeline.print_results."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from unicodedata import east_asian_width

import pandas as pd

from src.utils import resolve_path


METHOD_NAMES = {
    "naive": "最后值保持", "seasonal_naive": "前一天同刻",
    "three_day_mean": "前三天同刻均值", "seasonal_ar": "季节自回归",
    "ridge": "Ridge", "extra_trees": "ExtraTrees", "sarima": "SARIMA",
    "lightgbm": "LightGBM", "xgboost": "XGBoost", "lstm": "LSTM", "tcn": "TCN",
    "dlinear": "DLinear", "chronos2": "Chronos-2", "chronos2_long": "Chronos-2", "tirex2": "TiRex-2",
    "timesfm25": "TimesFM 2.5", "toto2_joint": "Toto 2.0（联合）",
    "toto2_single": "Toto 2.0（单变量）", "timesfm30": "TimesFM 3.0",
    "timesfm30_long": "TimesFM 3.0（长版标识）", "timesfm30_uni": "TimesFM 3.0（单变量）",
    "patchtstfm": "PatchTST-FM", "toto2": "Toto 2.0", "patchtstfm_long": "PatchTST-FM", "local24": "本地24h（树/线性）",
    "local7": "本地7d（水平/形状）", "hdsf": "HDSF（最终融合）",
    "hdsf_foundation_only": "HDSF（仅基座）", "hdsf_with_local": "HDSF（含本地）",
    "hdsf_equal_weights": "HDSF（等权平均）", "hdsf_global_weights": "HDSF（全视野权重）",
}
CALIBRATED_METHODS = {"hdsf", "hdsf_foundation_only", "hdsf_with_local", "hdsf_global_weights"}


def configure_console():
    # Windows redirected Python output must use the same encoding as the saved preview.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _width(value):
    return sum(2 if east_asian_width(char) in ("W", "F") else 1 for char in value)


def _print_table(headers, rows):
    widths = [max(_width(row[i]) for row in [headers, *rows]) for i in range(len(headers))]

    def render(row):
        cells = []
        for i, value in enumerate(row):
            padding = " "*(widths[i]-_width(value))
            cells.append(value+padding if i == 0 else padding+value)
        return " | ".join(cells)

    print(render(headers))
    print("-+-".join("-"*size for size in widths))
    for row in rows:
        print(render(row))


def _number(value, *, percent=False, decimals=4, calibrated=False):
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{value:.{decimals}f}"+("%" if percent else "")+("*" if calibrated else "")


def _reduction(baseline, value):
    if baseline is None or value is None or not math.isfinite(baseline) or baseline <= 0:
        return None
    return 100*(1-value/baseline)


def print_results(metrics, *, manifest, windows=None, fusions=None):
    """Show all evaluated methods; preserve saved metrics and validation/test meaning."""
    configure_console()
    windows, fusions = windows or {}, fusions or {}
    keys = ["horizon", "model", "split", "evaluation"]
    if metrics.duplicated(keys).any():
        raise ValueError("Result rows must be unique per task, model, split and evaluation.")
    lookup = metrics.set_index(keys).to_dict("index")
    test = metrics[metrics.split == "test"]
    print(f"\n数据：{manifest['data']}；目标：{manifest['target']}；单位：{manifest['target_unit']}。")
    if manifest.get("is_demo"):
        print("合成数据：仅用于流程验证，不作为研究结果。")
    for task in manifest["requested_horizons"]:
        methods = test.loc[test.horizon == task, "model"].drop_duplicates().tolist()
        if not methods:
            continue
        baseline = manifest.get("baseline_models", {}).get(task, "seasonal_naive" if task == "24h" else "three_day_mean")
        methods = ([baseline] if baseline in methods else [])+[name for name in methods if name != baseline]
        names = {**METHOD_NAMES, **manifest.get("method_names", {}), **manifest.get("task_method_names", {}).get(task, {})}
        for canonical, member in (("chronos2", "chronos2_long"), ("patchtstfm", "patchtstfm_long")):
            if canonical in methods and member in methods:
                names[member] = names[canonical]+"（HDSF成员）"
        for method in methods:
            if method.startswith("hdsf_member__"):
                base = method.removeprefix("hdsf_member__")
                names[method] = names.get(base, base)+"（HDSF成员）"
        if task == "7d":
            names["seasonal_naive"] = "前一天轮廓重复"
        name = lambda method: names.get(method, method)
        score = lambda method, evaluation="curve", split="test": lookup.get((task, method, split, evaluation), {})
        pct = lambda value, **options: _number(value, percent=True, **options)
        decrease = lambda method, metric, evaluation="curve": pct(
            _reduction(score(baseline, evaluation).get(metric), score(method, evaluation).get(metric)), decimals=2)
        rows = []
        for method in methods:
            curve, daily = score(method), score(method, "daily_mean")
            if task == "24h":
                validation = score(method, split="validation")
                rows.append([name(method), pct(validation.get("WAPE"), calibrated=method in CALIBRATED_METHODS),
                    pct(validation.get("MAPE"), calibrated=method in CALIBRATED_METHODS),
                    pct(curve.get("WAPE")), pct(curve.get("MAPE")), decrease(method, "WAPE")])
            else:
                rows.append([name(method), pct(curve.get("WAPE")), pct(curve.get("MAPE")),
                    decrease(method, "WAPE"), pct(daily.get("WAPE")), pct(daily.get("MAPE"))])
        if task == "24h":
            print("\n未来24小时完整负荷曲线", flush=True)
            _print_table(["方法", "验证 WAPE", "验证 MAPE", "测试 WAPE", "测试 MAPE", "相对基线误差减少"], rows)
            print(f"\n未来24小时日均负荷与 RMSE（{manifest['target_unit']}）")
            extra = [[name(method), pct(score(method, "daily_mean").get("WAPE")),
                pct(score(method, "daily_mean").get("MAPE")), _number(score(method).get("RMSE")),
                _number(score(method, "daily_mean").get("RMSE")), decrease(method, "MAPE"),
                decrease(method, "MAPE", "daily_mean")] for method in methods]
            _print_table(["方法", "日均 WAPE", "日均 MAPE", "曲线 RMSE", "日均 RMSE", "曲线 MAPE 减少", "日均 MAPE 减少"], extra)
        else:
            print("\n未来7天曲线与每日平均负荷", flush=True)
            _print_table(["方法", "曲线 WAPE", "曲线 MAPE", "曲线误差减少", "每日均值 WAPE", "每日均值 MAPE"], rows)
            print(f"\n未来7天 RMSE（{manifest['target_unit']}）与误差减少幅度")
            extra = [[name(method), _number(score(method).get("RMSE")),
                _number(score(method, "daily_mean").get("RMSE")), decrease(method, "MAPE"),
                decrease(method, "MAPE", "daily_mean"), decrease(method, "WAPE", "daily_mean")] for method in methods]
            _print_table(["方法", "曲线 RMSE", "日均 RMSE", "曲线 MAPE 减少", "日均 MAPE 减少", "日均 WAPE 减少"], extra)
        reference = score(baseline) or score(methods[0])
        daily_reference = score(baseline, "daily_mean") or score(methods[0], "daily_mean")
        coverage = windows.get(task, {}).get("test", {})
        print(f"对照基线：{name(baseline)}。表中主表的误差减少按 WAPE 计算；正值改善，负值变差。")
        if baseline not in methods:
            print("本次未运行该基线，相对误差减少显示 N/A。")
        print(f"测试覆盖 {coverage.get('windows', 'N/A')} 个起报点；曲线 {reference.get('n', 'N/A')} 个点，"
              f"日均 {daily_reference.get('n', 'N/A')} 个样本；跳过 {coverage.get('skipped', 'N/A')} 个缺测窗口。")
        threshold = manifest["experiment"].get("mape_threshold", 1e-8)
        print(f"MAPE 仅计 |真实值| > {threshold:g}；曲线剔除 {reference.get('mape_excluded', 'N/A')} 点，"
              f"日均剔除 {daily_reference.get('mape_excluded', 'N/A')} 个样本。")
        if task == "24h" and any(method in CALIBRATED_METHODS for method in methods):
            print("* 验证列为融合校准的样本内指标；HDSF 选择使用验证交叉检验分数。")
        if task in fusions:
            print(f"HDSF 验证交叉检验 MAPE：{pct(fusions[task]['cv_origin_mean']['MAPE'])}；"
                  f"冻结 mix={fusions[task]['mix']:.3f}。")
        selected = manifest.get("selected_models", {}).get(task)
        if selected:
            print(f"验证阶段选择：{name(selected)}（测试前冻结）。")
    print("\n"+manifest.get("metrics_aggregation_description", "指标按保存的预测实例汇总；日均先逐起报、逐目标日聚合再计分。"))


def print_saved_results(run_dir, *, results_dir="results/runs"):
    """Re-display a completed run; 'latest' selects the newest completed real-data run."""
    if str(run_dir) == "latest":
        for path in sorted(resolve_path(results_dir).glob("*/manifest.json"), reverse=True):
            info = json.loads(path.read_text(encoding="utf-8"))
            if info.get("status") == "complete" and not info.get("is_demo", False):
                run_dir = path.parent
                break
        else:
            raise FileNotFoundError("No completed real-data result found.")
    folder = resolve_path(run_dir)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Display requires a completed experiment.")
    if manifest.get("protocol") == "cross_domain_google_lodo_v1":
        print_cross_domain_results(pd.read_csv(folder/"metrics.csv"), manifest=manifest, folder=folder)
        print(f"结果目录：{folder}", flush=True)
        return folder
    windows, fusions = {}, {}
    for task in manifest["requested_horizons"]:
        for path, destination in ((folder/task/"windows.json", windows), (folder/task/"hdsf/fusion.json", fusions)):
            if path.exists():
                destination[task] = json.loads(path.read_text(encoding="utf-8"))
    print_results(pd.read_csv(folder/"metrics.csv"), manifest=manifest, windows=windows, fusions=fusions)
    print(f"结果目录：{folder}", flush=True)
    return folder


def print_cross_domain_results(metrics, *, manifest, folder):
    """Keep the ordinary Chinese tables, scoped to one held-out physical domain."""
    configure_console()
    print("\nGoogle 电力域 LODO：利用率任务；目标有监督模型仅作不同训练条件的参考。")
    print("零样本方法的验证列为 N/A；源域开发诊断单列保存，不使用目标域标签选参。")
    for target in manifest["target_domains"]:
        print(f"\n留出目标域：{target}")
        windows = {}
        for task in manifest["requested_horizons"]:
            path = folder/target/task/"windows.json"
            if path.exists():
                windows[task] = json.loads(path.read_text(encoding="utf-8"))
        selected = metrics[metrics.target_domain == target]
        task_names = {}
        for task in manifest["requested_horizons"]:
            task_names[task] = {row.model: f"{METHOD_NAMES.get(row.algorithm, row.algorithm)}（多源 {len(json.loads(row.source_domains))}域）"
                                for row in selected[(selected.horizon == task) & (selected.source_strategy == "multi_source")].itertuples()}
        print_results(selected, manifest={**manifest, "task_method_names": task_names}, windows=windows)
    summary_path = folder/"domain-summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        print("\n跨目标域等权汇总（MAPE；参考模型与零样本训练条件不同）")
        names = manifest.get("method_names", {}).copy()
        for row in metrics[metrics.source_strategy == "multi_source"].itertuples():
            names[row.model] = f"{METHOD_NAMES.get(row.algorithm, row.algorithm)}（多源联合）"
        _print_table(["任务 / 方法", "目标域数", "平均 MAPE", "最差 MAPE"], [
            [f"{row.horizon} / {names.get(row.model, row.model)}", str(row.targets),
             _number(row.mean_MAPE, percent=True), _number(row.worst_MAPE, percent=True)]
            for row in summary.itertuples()])
    negative_path = folder/"negative-transfer.csv"
    if negative_path.exists() and negative_path.stat().st_size > 2:
        negative = pd.read_csv(negative_path)
        for task, group in negative.groupby("horizon", sort=False):
            valid = group.negative_transfer.dropna()
            count = int(valid.sum())
            affected = group.loc[group.negative_transfer == True, "target_domain"].nunique()
            total_targets = group.target_domain.nunique()
            print(f"{task} 增加源域后 MAPE 增大的比较：{count}/{len(valid)}；"
                  f"涉及目标域 {affected}/{total_targets}；"
                  "以预先指定单源为基线，同一目标域和相同起报窗口；不与目标有监督参考相减。")
