"""Recompute exported prediction metrics and produce a readable NLR report."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.evaluation.metrics import forecast_metrics
from src.evaluation.reporting import print_saved_results
from src.data.splitting import complete_days
from src.utils import resolve_path, write_json


def markdown_table(frame):
    def fmt(value):
        if pd.isna(value):
            return "—"
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def report(folder, output=None):
    folder = resolve_path(folder)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Report requires a completed independent test run.")
    threshold = manifest["experiment"]["mape_threshold"]
    metrics = pd.read_csv(folder/"metrics.csv")
    frozen = json.loads((folder/"frozen-before-test.json").read_text(encoding="utf-8"))
    prepared_path = Path(manifest["data_provenance"]["path"])
    assert sha256(prepared_path.read_bytes()).hexdigest() == manifest["data_provenance"]["sha256"], "Prepared data changed after the run."
    checks = []
    for name, expected in frozen["model_sha256"].items():
        assert sha256((folder/name).read_bytes()).hexdigest() == expected, f"Frozen model changed: {name}"
    for task in manifest["requested_horizons"]:
        for name in metrics.model.unique():
            path = folder/task/name/"test_predictions.parquet"
            if not path.exists():
                continue
            table = pd.read_parquet(path)
            computed = forecast_metrics(table.actual.to_numpy(), table.predicted.to_numpy(), threshold)
            row = metrics[(metrics.horizon == task) & (metrics.model == name) & (metrics.split == "test") & (metrics.evaluation == "curve")].iloc[0]
            for metric in ("MAPE", "WAPE", "RMSE"):
                assert np.isclose(computed[metric], row[metric], rtol=1e-8, atol=1e-8), f"Metric mismatch: {task}/{name}/{metric}"
            daily = table.assign(target_day=(table.lead-1)//96).groupby(["origin", "target_day"])[["actual", "predicted"]].mean()
            computed_daily = forecast_metrics(daily.actual.to_numpy(), daily.predicted.to_numpy(), threshold)
            daily_row = metrics[(metrics.horizon == task) & (metrics.model == name) & (metrics.split == "test") & (metrics.evaluation == "daily_mean")].iloc[0]
            for metric in ("MAPE", "WAPE", "RMSE"):
                assert np.isclose(computed_daily[metric], daily_row[metric], rtol=1e-5, atol=1e-5), f"Daily mismatch: {task}/{name}/{metric}"
            checks.append({"task": task, "model": name, "curve": computed, "daily_mean": computed_daily})
    write_json(folder/"metric-recomputation-audit.json", {
        "frozen_models_unchanged": True, "prepared_data_unchanged": True, "prediction_rows_checked": checks,
        "report_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    figure_dir = folder/"report_figures"
    figure_dir.mkdir(exist_ok=True)
    # State thresholds use training history; each origin's state uses its preceding 14 days only.
    data = pd.read_parquet(prepared_path)
    data, _ = complete_days(data, manifest["dataset"]["timezone"], manifest["target"])
    target = data[manifest["target"]]
    train_stop = pd.Timestamp(manifest["splits"]["train"]["end"])
    train_start = pd.Timestamp(manifest["splits"]["train"]["start"])
    context = int(manifest["experiment"]["context_length"])
    state_features = []
    for pos in range(context, len(target)-96+1, 96):
        if target.index[pos] > train_stop or target.index[pos] < train_start:
            continue
        past = target.iloc[pos-context:pos].to_numpy()
        if np.isfinite(past).all():
            daily = past.reshape(-1, 96).mean(1)
            state_features.append((np.std(past)/max(abs(np.mean(past)), 1e-8), abs(daily[-1]-daily[:-1].mean())/max(abs(daily.mean()), 1e-8)))
    state_features = np.asarray(state_features)
    cv_cut, shift_cut = np.quantile(state_features, .75, axis=0)
    state_rows, nonoverlap_rows, variability_rows = [], [], []
    for task in manifest["requested_horizons"]:
        fusion = json.loads((folder/task/"hdsf"/"fusion.json").read_text(encoding="utf-8"))
        wa, wb = np.asarray(fusion["weights_a"]), np.asarray(fusion["weights_b"])
        effective_weights = (1-fusion["mix"])*np.pad(wa, ((0, 0), (0, 1)))+fusion["mix"]*wb
        pd.DataFrame(effective_weights, columns=fusion["members"]).rename_axis("segment_zero_based").to_csv(
            folder/task/"hdsf"/"effective_frozen_weights.csv")
        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        heat = ax.imshow(effective_weights.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set(yticks=range(len(fusion["members"])), yticklabels=fusion["members"],
               xticks=range(fusion["segments"]), xticklabels=range(1, fusion["segments"]+1),
               title=f"HDSF {task}: frozen effective weights", xlabel="Half-day segment (48 points)")
        fig.colorbar(heat, ax=ax, label="Member weight")
        fig.savefig(figure_dir/f"{task}_frozen_weights.png", dpi=160)
        plt.close(fig)
        for name in metrics[(metrics.horizon == task) & (metrics.split == "test")].model.unique():
            table = pd.read_parquet(folder/task/name/"test_predictions.parquet")
            labels = {}
            for origin in table.origin.unique():
                position = target.index.get_loc(pd.Timestamp(origin))
                past = target.iloc[position-context:position].to_numpy()
                daily = past.reshape(-1, 96).mean(1)
                cv = np.std(past)/max(abs(np.mean(past)), 1e-8)
                shift = abs(daily[-1]-daily[:-1].mean())/max(abs(daily.mean()), 1e-8)
                labels[origin] = "level_change" if shift > shift_cut else "variable" if cv > cv_cut else "stable"
            table["history_state"] = table.origin.map(labels)
            for state, group in table.groupby("history_state"):
                state_rows.append(dict(task=task, model=name, history_state=state, origins=group.origin.nunique(),
                    **forecast_metrics(group.actual.to_numpy(), group.predicted.to_numpy(), threshold)))
            info = json.loads((folder/task/name/"test_metrics.json").read_text(encoding="utf-8"))
            if info["nonoverlap"]:
                row = info["nonoverlap"]
                nonoverlap_rows.append(dict(task=task, model=name, origins=len(row["origins"]),
                    **{k: row[k] for k in ("MAPE", "RMSE", "WAPE", "n")}))
            per_origin = pd.read_csv(folder/task/name/"test_per_origin.csv")
            variability_rows.append(dict(task=task, model=name, origins=len(per_origin),
                **{f"{metric}_{stat}": getattr(per_origin[metric], stat)(ddof=0)
                   if stat == "std" else getattr(per_origin[metric], stat)()
                   for metric in ("MAPE", "RMSE", "WAPE")
                   for stat in ("mean", "std", "min", "max")}))
        origins = pd.read_csv(folder/task/"hdsf"/"test_per_origin.csv")
        daily_lead = pd.read_csv(folder/task/"hdsf"/"test_per_day.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        origin_errors = pd.Series(origins.MAPE.to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(origins.origin))).asfreq("D")
        axes[0].plot(origin_errors.index, origin_errors.to_numpy(), ".-", lw=.8, ms=2, color="#007c91")
        axes[0].set(title=f"HDSF {task}: error across rolling origins", ylabel="MAPE (%)")
        axes[0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[0].tick_params(axis="x", rotation=30)
        axes[1].plot(daily_lead.forecast_day, daily_lead.MAPE, "o-", color="#da6a00")
        axes[1].set(title="Error by forecast day", xlabel="Forecast day", ylabel="MAPE (%)",
                    xticks=daily_lead.forecast_day, xlim=(.5, daily_lead.forecast_day.max()+.5))
        for ax in axes:
            ax.grid(alpha=.2)
        fig.savefig(figure_dir/f"{task}_error_analysis.png", dpi=160)
        plt.close(fig)
    pd.DataFrame(state_rows).to_csv(folder/"state_metrics.csv", index=False)
    pd.DataFrame(nonoverlap_rows).to_csv(folder/"nonoverlap_metrics.csv", index=False)
    pd.DataFrame(variability_rows).to_csv(folder/"origin_variability.csv", index=False)
    report_variability = []
    for row in variability_rows:
        baseline = "seasonal_naive" if row["task"] == "24h" else "three_day_mean"
        if row["model"] in {"hdsf", baseline}:
            for metric in ("MAPE", "RMSE", "WAPE"):
                report_variability.append({"task": row["task"], "model": row["model"], "metric": metric,
                    "origins": row["origins"], **{stat: row[f"{metric}_{stat}"] for stat in ("mean", "std", "min", "max")}})
    write_json(folder/"state_thresholds.json", dict(fit_end=str(train_stop), training_histories=len(state_features),
        cv_75pct=float(cv_cut), level_change_75pct=float(shift_cut), state_is_history_proxy_not_job_metadata=True))
    test = metrics[metrics.split == "test"]
    main = test[test.model.isin(["seasonal_naive", "three_day_mean", "seasonal_ar", "dlinear", "local24", "local7", "hdsf"])]
    columns = ["horizon", "model", "evaluation", "MAPE", "RMSE", "WAPE", "n", "mape_excluded", "MAPE_reduction_pct", "WAPE_reduction_pct"]
    content = ["# NLR 数据集 HDSF 复现结果", "",
        "本报告由实际预测 Parquet 重新计分，并检查测试前保存的模型文件哈希。合成数据不计入研究结果。", "",
        f"实验目录：`{folder}`。随机种子：{manifest['experiment']['seed']}。目标：五项分项功率的区间均值之和，单位 kW。时区暂按 UTC，尚未核实。", "",
        f"研究跨度 {len(target)//96} 个日历日，其中 {(target.notna().resample('D').sum() == 96).sum()} 天具有 96 个有效聚合目标；"
        f"缺失目标 {target.isna().sum()} 点（{100*target.isna().mean():.4f}%）。内部缺测不删除、不压缩时间轴。"
        f"观测功率最小值 / 中位数 / 最大值为 {target.min():.2f} / {target.median():.2f} / {target.max():.2f} kW。", "",
        "## 协议与有效窗口", "",
        markdown_table(pd.DataFrame([dict(split=name, **value) for name, value in manifest["splits"].items()])), ""]
    hardware_path = resolve_path("results/analysis/nlr/hardware.json")
    if hardware_path.exists():
        hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
        content.extend([f"硬件：{hardware['cpu']}，{hardware['logical_processors']} 个逻辑处理器，"
                        f"{hardware['ram_gib']:.2f} GiB 内存；基座推理为 CPU、4 线程。", ""])
    for task in manifest["requested_horizons"]:
        window_info = json.loads((folder/task/"windows.json").read_text(encoding="utf-8"))
        cfg = json.loads((folder/task/"hdsf"/"fusion.json").read_text(encoding="utf-8"))
        content.extend([f"- {task}："+"；".join(f"{s} 有效 {v['windows']}、跳过 {v['skipped']}" for s, v in window_info.items())+"。",
            f"- {task} HDSF：{cfg['segments']} 个半日段，混合比例 {cfg['mix']:.3f}，验证交叉检验 MAPE {cfg['cv_origin_mean']['MAPE']:.4f}%；"+
            f"融合校准总耗时 {cfg['calibration_seconds']:.1f} 秒（包括基座验证预测）。",
            f"- 验证阶段选中模型：{manifest['selected_models'][task]}；HDSF 的总体模型选择使用交叉检验分数，不使用融合拟合分数。", ""])
    conclusions = []
    for task in manifest["requested_horizons"]:
        curve = test[(test.horizon == task) & (test.evaluation == "curve")].set_index("model")
        hdsf = curve.loc["hdsf"]
        baseline = "seasonal_naive" if task == "24h" else "three_day_mean"
        reduction = hdsf["MAPE_reduction_pct"]
        direction = "降低" if reduction >= 0 else "增加"
        best = curve.MAPE.idxmin()
        conclusions.append(f"- {task}：HDSF 曲线 MAPE {hdsf['MAPE']:.4f}%，"
            f"相对 {baseline} 的 MAPE {direction} {abs(reduction):.2f}%；"
            f"本次测试表中 MAPE 最低的方法为 {best}（{curve.loc[best, 'MAPE']:.4f}%）。"
            "测试排名只用于结果解释，不改变测试前冻结的选择。")
    content.extend(["## 独立测试结果", "", *conclusions, "", markdown_table(main[columns]), "",
        "MAPE/WAPE 为百分数，RMSE 为 kW。24h 的日均任务与 7d 的七个日均任务分别计分。改进幅度为相对误差减少比例；负值表示劣于基线。", "",
        "## 基座成员与消融", "", markdown_table(test[(test.evaluation == "curve") & ~test.model.isin(["seasonal_naive", "three_day_mean", "seasonal_ar", "dlinear"])][columns]), "",
        "消融均在相同验证/测试窗口上计算：仅基座池、加入本地池、等权平均、统一全视野权重。统一全视野权重独立在验证集拟合；测试期不重选。", "",
        "## 非重叠 7 天检验", "", markdown_table(pd.DataFrame(nonoverlap_rows)), "",
        "## 起报点波动", "", markdown_table(pd.DataFrame(report_variability)), "",
        "这里显示 HDSF 与各任务简单基线；全部方法的起报汇总保留在 origin_variability.csv。", "",
        "上表逐起报分别计分后汇总均值、总体标准差、最小值和最大值；主结果表为所有预测实例汇总计分。有效标签点数相同且无 MAPE 剔除时，两种 MAPE 汇总相同；RMSE/WAPE 的汇总口径通常不同。", "",
        "## 计算成本", "", markdown_table(test[(test.evaluation == "curve") & test.model.isin(manifest["requested_models"])][["horizon", "model", "train_seconds", "inference_seconds"]]), "",
        "训练耗时包含初始训练与最终重拟合；HDSF 验证成员生成与融合求解另记为 calibration_seconds。预测缓存已开启，24h 验证阶段复用了先前独立生成的历史输入预测，因此本次校准耗时不代表全部冷启动基座推理成本。测试推理耗时含缓存读写和成员生成。成员行的 0 秒表示成本统一记入 HDSF，不能解释为没有计算成本。", "",
        "## 解释与限制", "",
        "- 核心半日分段单纯形 L1 与 A/B 混合结构来自旧 HDSF；这是已有方法迁移，不能宣称为全新算法。",
        "- 本实验按新方案统一 14 天历史、60/20/20 完整日划分、MAPE 选择；旧 D14/D22、WAPE 验收门槛、21/28 天长上下文不沿用。",
        "- 使用五折验证并剔除预测标签重叠；它是原始逐起报留一策略的长数据适配。7d 的完整交叉检验不是前瞻性在线验证；所有标签都位于验证集内。",
        "- local24 为适配 NLR 的有限历史滞后特征 LightGBM 残差 + 线性滞后回归；不包含旧 57 序列元数据、生产率字段或无限扩展历史。",
        "- NLR 只有一个目标通道，部分联合/单变量及同一上下文长短版本会重复；保留原成员标识并报告，不能把它们视为独立信息源。",
        "- 本次 HDSF 优于任务的简单基线，但未超过测试中最强的单成员；不能声称融合优于所有基座，或将旧数据集的精度结论直接迁移。",
        "- 逐起报误差最大值显著高于均值，部分负荷变化时段存在误差尖峰；没有业务状态记录，不能据此直接推断停机原因。",
        "- 独立半日权重没有段间连续性约束，预测曲线可出现段边界阶跃；本轮保留原结构，未依据测试结果增加平滑。",
        "- 融合拟合后的 validation 行是样本内校准分数；交叉检验分数在 fusion.json 中，独立测试才是泛化结果。",
        "- 缺测未填补，可评估窗口偏向连续完整时段；不能把当前误差推断到缺测、停机或未覆盖工况。",
        "- 原始分钟记录不足 15 条的区间仍按可用有限观测聚合；有效目标意味着五项区间均值齐全。原始覆盖计数另见 results/analysis/nlr/audit.json，不能将聚合完整误解为所有原始分钟完整。",
        "- state_metrics.csv 的状态来自起报前负荷波动与水平变化，阈值只用训练历史；不代表已验证的业务状态标签。",
        "- 7d 滚动窗口相关，逐起报误差和非重叠结果用于描述稳定性，不把重叠点数当独立样本数。",
        "- 本轮完成 HDSF 迁移、对照和融合消融；多数据集泛化、业务特征创新与 7/14/28 天完整敏感性研究仍需后续实验。", "",
        "全部成员及任务的完整指标见 metrics.csv，逐起报、逐预测天、原始预测、冻结配置和模型均保留在实验目录。", ""])
    path = resolve_path(output) if output else folder/"report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content.extend(["## 图形证据", "", "权重图来自测试前冻结的验证融合参数；有效权重合并了 A/B 两个池的混合比例。缺测导致无法起报的日期在误差图中留空，不插值。", ""])
    for task in manifest["requested_horizons"]:
        for label, image_path in ((f"{task} 首个有效测试起报", folder/task/"hdsf"/"forecast.png"),
                                  (f"{task} 起报与提前量误差", figure_dir/f"{task}_error_analysis.png"),
                                  (f"{task} 冻结成员权重", figure_dir/f"{task}_frozen_weights.png")):
            if not path.is_relative_to(folder):
                export_path = path.parent/"figures"/f"hdsf-nlr-{folder.name}"/image_path.name
                # Forecast filenames are shared across tasks; preserve task identity in the export.
                if image_path.name == "forecast.png":
                    export_path = export_path.with_name(f"{task}_forecast.png")
                export_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, export_path)
                image_path = export_path
            content.extend([f"![{label}]({Path(os.path.relpath(image_path, path.parent)).as_posix()})", ""])
    path.write_text("\n".join(content), encoding="utf-8")
    print_saved_results(folder)
    print(f"Verified report: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    report(args.run_dir, args.output)
