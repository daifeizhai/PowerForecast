"""Chronological source-internal leave-domain gain labels and past-only priors."""
import numpy as np
import pandas as pd

from src.data.domains import assert_data_usage
from src.data.google_panel import domain_windows
from src.evaluation.metrics import forecast_metrics
from src.models.ml.benefit import FEATURE_NAMES, JointBenefitPredictor, pair_features
from src.models.ml.sapf import SelectiveTransferForecaster
from src.models.statistical.baselines import ThreeDayMeanForecaster


def source_bank(panel, domains, cutoff, horizon, params, seed):
    for d in domains:
        assert_data_usage(panel.metadata[d], "training")
    period = panel.points_per_day
    origins = list(range(14, cutoff-horizon//period+1))
    batch = domain_windows(panel, domains, origins, 14*period, horizon)
    bank = SelectiveTransferForecaster(**params, seed=seed).fit(batch)
    bank.training_windows_per_expert = len(origins)
    return bank, batch


def reduce_priors(table, source_domains, excluded_target=None):
    result = {}
    for domain in source_domains:
        part = table[(table.expert_domain == domain) & (table.target_domain != excluded_target)]
        if part.empty:
            raise ValueError("At least three source domains are required for leave-target-out historical priors.")
        result[domain] = [float(part.expert_MAPE.mean()), float(part.local_MAPE.mean()),
            float((part.local_MAPE-part.expert_MAPE).mean()), float((part.expert_MAPE > part.local_MAPE+1e-10).mean()),
            float(part.expert_MAPE.max())]
    return result


def train_joint_benefit(panel, sources, config, models, output):
    rows, prior_rows, priors_by_task = [], [], {}
    protocol = config["benefit_training"]
    period = panel.points_per_day
    for task, horizon in config["horizons"].items():
        checkpoint, _ = source_bank(panel, sources, protocol["prior_cutoff_day"], horizon, models["sapf"], config["seed"])
        (output/task/"prior").mkdir(parents=True, exist_ok=True)
        checkpoint.save(output/task/"prior/model")
        past = domain_windows(panel, sources, protocol["prior_origins"], period*14, horizon)
        experts = np.maximum(checkpoint.expert_predictions(past.inputs), 0.)
        local = ThreeDayMeanForecaster().predict(past.inputs)
        prior = []
        for i, (target, origin) in enumerate(zip(past.inputs.domain_ids, past.inputs.origins)):
            local_score = forecast_metrics(past.targets[i], local[i])["MAPE"]
            for s, expert in enumerate(checkpoint.source_domains):
                if expert == target:
                    continue
                prior.append(dict(horizon=task, expert_domain=expert, target_domain=target, origin=str(origin),
                    expert_fit_cutoff_day=protocol["prior_cutoff_day"],
                    label_end_day=int((origin-panel.anchor)/pd.Timedelta(days=1))+horizon//period,
                    expert_MAPE=forecast_metrics(past.targets[i], experts[i, s])["MAPE"], local_MAPE=local_score))
        prior_table = pd.DataFrame(prior)
        prior_rows.extend(prior)
        priors_by_task[task] = reduce_priors(prior_table, checkpoint.source_domains)
        for cutoff in protocol["meta_cutoffs"]:
            bank, _ = source_bank(panel, sources, cutoff, horizon, models["sapf"], config["seed"])
            (output/task/f"cutoff{cutoff}").mkdir(parents=True, exist_ok=True)
            bank.save(output/task/f"cutoff{cutoff}/model")
            end = cutoff+protocol["meta_block_days"]
            origins = list(range(cutoff, end-horizon//period+1))
            for target in sources:
                target_batch = domain_windows(panel, [target], origins, period*14, horizon)
                experts = np.maximum(bank.expert_predictions(target_batch.inputs), 0.)
                local = ThreeDayMeanForecaster().predict(target_batch.inputs)
                features = pair_features(target_batch.inputs, bank, reduce_priors(prior_table, bank.source_domains, target), experts, local)
                for i, origin in enumerate(target_batch.inputs.origins):
                    local_score = forecast_metrics(target_batch.targets[i], local[i])["MAPE"]
                    day = int((origin-panel.anchor)/pd.Timedelta(days=1))
                    for s, expert in enumerate(bank.source_domains):
                        if expert == target:
                            continue
                        score = forecast_metrics(target_batch.targets[i], experts[i, s])["MAPE"]
                        rows.append(dict(horizon=task, expert_domain=expert, pseudo_target=target,
                            expert_fit_cutoff_day=cutoff, origin_day=day, label_end_day=day+horizon//period,
                            prior_label_end_day=int(prior_table.label_end_day.max()),
                            actual_gain_pp=local_score-score, local_MAPE=local_score, expert_MAPE=score,
                            **{"feature_"+name: float(value) for name, value in zip(FEATURE_NAMES, features[i, s])}))
        print(f"[{output.name}] {task} 源域内部留域收益样本完成", flush=True)
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output/"gain-training.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(output/"past-performance.csv", index=False)
    learner = JointBenefitPredictor(models["benefit"], config["seed"]).fit(table, sources)
    return learner, priors_by_task
