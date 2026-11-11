"""Post-fit summaries for the frozen V5.2 evaluation masks."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PRIMARY_SCOPE = {"E0": "full", "E1": "post_preparation", "E2": "post_preparation"}
SCOPE_MASKS = {
    "full": lambda x: x.quality_eligible,
    "post_preparation": lambda x: x.post_preparation_eligible,
    "excluded_preparation": lambda x: x.quality_eligible & x.preparation_excluded,
    "exception_normal10": lambda x: x.quality_eligible & x.is_pre_anomaly_normal10,
}
GROUPS = {
    "logical_collection": "logical_collection_id",
    "leakage_group": "leakage_group_id",
    "acquisition_session": "acquisition_session_id",
}


def describe(series):
    return {f"q{q:02d}": float(series.quantile(q / 100)) for q in (10, 25, 50, 75, 90, 95, 99)}


def alarm_metrics(frame):
    n = len(frame)
    flags = int(frame.flagged.sum())
    return dict(n=n, flags=flags, rate=flags / n if n else np.nan,
                score_min=frame.score.min(), score_max=frame.score.max(),
                **{f"score_{key}": value for key, value in describe(frame.score).items()},
                **{f"ratio_{key}": value for key, value in describe(frame.score_over_threshold).items()})


def grouped(frame):
    yield "overall", "all", frame
    for group_type, column in GROUPS.items():
        for group_id, group in frame.groupby(column, sort=False):
            yield group_type, group_id, group
    restart = frame[frame.logical_collection_id.eq("nov10_restart40")]
    for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 40), (21, 40)):
        yield "restart_segment", f"{lo}-{hi}", restart[restart.original_cycle_order.between(lo, hi)]


def anomaly_metrics(anomaly, normal):
    pos, neg = len(anomaly), len(normal)
    tp, fp = int(anomaly.flagged.sum()), int(normal.flagged.sum())
    labels = np.r_[np.ones(pos, dtype=int), np.zeros(neg, dtype=int)]
    values = np.r_[anomaly.score, normal.score]
    return dict(n=pos, anomaly_n=pos, tp=tp, fn=pos - tp, tpr=tp / pos if pos else np.nan,
                normal_n=neg, paired_normal_n=neg, fp=fp, tn=neg - fp, fpr=fp / neg if neg else np.nan,
                auroc=roc_auc_score(labels, values) if pos and neg else np.nan,
                ap=average_precision_score(labels, values) if pos and neg else np.nan,
                ap_prevalence=pos / (pos + neg) if pos and neg else np.nan,
                anomaly_score_median=anomaly.score.median(), normal_score_median=normal.score.median(),
                anomaly_ratio_median=anomaly.score_over_threshold.median(), normal_ratio_median=normal.score_over_threshold.median(),
                anomaly_score_q10=anomaly.score.quantile(.1), anomaly_score_q90=anomaly.score.quantile(.9))


def matched_comparison(cycle_scores):
    retained = cycle_scores[cycle_scores.eligible_for_primary_metric & cycle_scores.evaluation.isin(["E1", "E2"])]
    paired_rows, cohorts = [], []
    for family, group in retained.groupby("family", sort=False):
        e1 = group[group.evaluation.eq("E1")].set_index("cycle_id")
        e2 = group[group.evaluation.eq("E2")].set_index("cycle_id")
        assert e1.index.is_unique and e2.index.is_unique and set(e1.index) == set(e2.index)
        e2 = e2.loc[e1.index]
        normal = e1.condition.eq("normal")
        old, new = e1.flagged.astype(bool), e2.flagged.astype(bool)
        changed = e1[["family", "condition", "pair_condition", "logical_collection_id", "leakage_group_id", "original_cycle_order"]].copy()
        changed = changed.assign(e1_score=e1.score, e2_score=e2.score, e1_threshold=e1.threshold,
                                 e2_threshold=e2.threshold, e1_score_over_threshold=e1.score_over_threshold,
                                 e2_score_over_threshold=e2.score_over_threshold, e1_flagged=old, e2_flagged=new)
        changed["transition"] = np.select(
            [~old & new & normal, old & ~new & normal, old & new & normal, ~old & ~new & normal,
             ~old & new, old & ~new, old & new],
            ["new_fp", "resolved_fp", "common_fp", "both_normal", "added_tp", "lost_tp", "common_tp"],
            default="both_missed")
        paired_rows.append(changed.reset_index())
        selectors = [("normal", typ, key, part.index) for typ, key, part in grouped(e1[normal]) if len(part)]
        selectors += [("anomaly", "condition", key, e1[e1.condition.eq(key)].index) for key in "ABC"]
        selectors += [("normal", "paired_control", key, e1[normal & e1.pair_condition.eq(key)].index) for key in "ABC"]
        for label, typ, key, idx in selectors:
            old_part, new_part = old.loc[idx], new.loc[idx]
            added, lost = int((~old_part & new_part).sum()), int((old_part & ~new_part).sum())
            both, neither = int((old_part & new_part).sum()), int((~old_part & ~new_part).sum())
            n = len(idx)
            assert n == added + lost + both + neither and int(new_part.sum()) - int(old_part.sum()) == added - lost
            item = dict(family=family, label=label, group_type=typ, group_id=key, n=n,
                        e1_flags=int(old_part.sum()), e2_flags=int(new_part.sum()), e1_rate=float(old_part.mean()),
                        e2_rate=float(new_part.mean()), delta_rate=float(new_part.mean() - old_part.mean()),
                        added_flags=added, lost_flags=lost, common_flags=both, both_unflagged=neither,
                        e1_score_median=e1.loc[idx].score.median(), e2_score_median=e2.loc[idx].score.median(),
                        e1_ratio_median=e1.loc[idx].score_over_threshold.median(), e2_ratio_median=e2.loc[idx].score_over_threshold.median())
            if label == "anomaly":
                a = anomaly_metrics(e1.loc[idx], e1[normal & e1.pair_condition.eq(key)])
                b = anomaly_metrics(e2.loc[idx], e2[normal & e2.pair_condition.eq(key)])
                item.update(added_tp=added, lost_tp=lost, common_tp=both, both_missed=neither,
                            e1_auroc=a["auroc"], e2_auroc=b["auroc"], e1_ap=a["ap"], e2_ap=b["ap"],
                            e1_paired_fp=a["fp"], e2_paired_fp=b["fp"], paired_normal_n=a["paired_normal_n"])
            else:
                item.update(new_fp=added, resolved_fp=lost, common_fp=both, both_normal=neither)
            cohorts.append(item)
    paired = pd.concat(paired_rows, ignore_index=True)
    assert not paired.duplicated(["family", "cycle_id"]).any()
    return pd.DataFrame(cohorts), paired


def build_summaries(cycle_scores):
    normal_rows, anomaly_rows, exceptions, excluded, distributions = [], [], [], [], []
    for (family, evaluation), frame in cycle_scores.groupby(["family", "evaluation"], sort=False):
        family_key = dict(family=family, evaluation=evaluation, model_id=frame.model_id.iloc[0], threshold=float(frame.threshold.iloc[0]))
        eval_frame = frame[frame.original_role.isin(["test", "evaluation_only"])]
        normal_all = eval_frame[eval_frame.condition.eq("normal")]
        full_counts = {(typ, key): len(part) for typ, key, part in grouped(normal_all[normal_all.quality_eligible])}
        for scope, select in SCOPE_MASKS.items():
            normal = normal_all[select(normal_all)]
            scope_key = dict(**family_key, scope=scope, primary_scope=PRIMARY_SCOPE[evaluation] == scope)
            logical = [(key, part) for typ, key, part in grouped(normal) if typ == "logical_collection" and len(part)]
            worst_id, worst = max(logical, key=lambda item: item[1].flagged.mean()) if logical else ("", normal.iloc[0:0])
            for typ, key, part in grouped(normal):
                stat = alarm_metrics(part)
                full_n = full_counts.get((typ, key), 0)
                row = dict(**scope_key, group_type=typ, group_id=key, **stat,
                           fp=stat["flags"], fpr=stat["rate"], full_n=full_n,
                           scope_fraction=len(part) / full_n if full_n else np.nan,
                           worst_logical_collection=worst_id if typ == "overall" else "",
                           worst_fp=int(worst.flagged.sum()) if typ == "overall" else np.nan,
                           worst_n=len(worst) if typ == "overall" else np.nan,
                           worst_fpr=float(worst.flagged.mean()) if typ == "overall" and len(worst) else np.nan)
                normal_rows.append(row)
                if scope == "exception_normal10" and typ == "logical_collection":
                    assert len(part) == 10
                    exceptions.append(dict(**row, condition=part.pair_condition.iloc[0]))
                if scope == "excluded_preparation" and typ in {"overall", "logical_collection"}:
                    excluded.append(dict(**row, condition="normal", model_flagged=stat["flags"],
                                         model_rate=stat["rate"], preparation_rule_excluded_n=len(part),
                                         policy_not_evaluated_n=len(part) if evaluation != "E0" else 0))
            if scope == "exception_normal10":
                continue
            for condition in "ABC":
                all_anomalies = eval_frame[eval_frame.condition.eq(condition) & eval_frame.quality_eligible]
                anomaly = all_anomalies[select(all_anomalies)]
                control = normal[normal.pair_condition.eq(condition)]
                stat = anomaly_metrics(anomaly, control)
                not_evaluated = int(all_anomalies.preparation_excluded.sum())
                anomaly_rows.append(dict(**scope_key, condition=condition, **stat,
                                         quality_full_n=len(all_anomalies), scope_fraction=len(anomaly) / len(all_anomalies),
                                         policy_not_evaluated_n=not_evaluated if evaluation != "E0" else 0,
                                         policy_not_evaluated_fraction=not_evaluated / len(all_anomalies) if evaluation != "E0" else 0.0))
                if scope == "excluded_preparation":
                    excluded.append(dict(**scope_key, condition=condition, group_type="logical_collection",
                                         group_id=anomaly.logical_collection_id.iloc[0], **alarm_metrics(anomaly),
                                         model_flagged=stat["tp"], model_rate=stat["tpr"], preparation_rule_excluded_n=len(anomaly),
                                         policy_not_evaluated_n=len(anomaly) if evaluation != "E0" else 0, full_n=len(all_anomalies),
                                         scope_fraction=len(anomaly) / len(all_anomalies)))
        groups = {"train_full": frame.original_role.eq("train") & frame.quality_eligible,
                  "fit_train_used": frame.fit_train if evaluation == "E2" else frame.original_role.eq("train") & frame.quality_eligible,
                  "calibration": frame.fit_calibration,
                  "normal_full": normal_all.quality_eligible,
                  "normal_post_preparation": normal_all.post_preparation_eligible,
                  "normal_excluded_preparation": normal_all.quality_eligible & normal_all.preparation_excluded}
        for label, selected in groups.items():
            part = (normal_all if label.startswith("normal_") else frame)[selected]
            distributions.append(dict(**family_key, group_id=label, **alarm_metrics(part),
                                      score_q95_higher=part.score.quantile(.95, interpolation="higher")))
        for condition in "ABC":
            subset = eval_frame[eval_frame.condition.eq(condition)]
            for scope in ("full", "post_preparation", "excluded_preparation"):
                distributions.append(dict(**family_key, group_id=f"{condition}_{scope}", **alarm_metrics(subset[SCOPE_MASKS[scope](subset)])))

    matched, paired = matched_comparison(cycle_scores)
    result = {
        "NORMAL_FULL_VS_POST_PREP": pd.DataFrame(normal_rows),
        "ANOMALY_FULL_VS_POST_PREP": pd.DataFrame(anomaly_rows),
        "PRE_ANOMALY_NORMAL10_RESULTS": pd.DataFrame(exceptions),
        "MATCHED_POST_PREP_COMPARISON": matched,
        "PAIRED_ALARM_TRANSITIONS": paired,
        "EXCLUDED_SEGMENT_DIAGNOSTICS": pd.DataFrame(excluded),
        "THRESHOLD_AND_SCORE_DISTRIBUTIONS": pd.DataFrame(distributions),
    }
    return result
