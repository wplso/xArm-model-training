"""Explain the archived bolt model's A121 predictions without fitting."""

import argparse
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp, softmax


CAMPAIGN = "No3_bolt_ABC_interleaved_20261111"
STUDY = "hardware/xarm_model_versions/bolt_known_load_prospective_20261111"
MODEL = "LOGISTIC_BOLT"
CONTRASTS = {"A_vs_normal": "normal", "A_vs_B": "B82", "A_vs_C": "C182"}


def analyze(project, output):
    results = project / "hardware" / CAMPAIGN / "analysis/frozen_evaluation"
    evaluator = runpy.run_path(str(project / STUDY / "evaluate_frozen.py"))
    _, inputs, frozen, bundles = evaluator["load_frozen"](project)
    helper = runpy.run_path(str(project / "hardware/xarm_model_versions"
                                / "binary_logistic_load_alert_20261111/analyze_multiclass.py"))
    features = frozen["features"]
    groups = np.array([helper["sensor_group"](feature) for feature in features])
    stage = inputs.csv(results / "CYCLE_STAGE_MAP.csv").set_index("cycle_id")
    cycles = inputs.csv(results / "CYCLE_FEATURES.csv").set_index("cycle_id")
    x = cycles[features]
    scores = inputs.csv(results / "CYCLE_SCORES.csv", keep_default_na=False)
    scores = scores.set_index(["model_id", "cycle_id"]).loc[MODEL].loc[x.index].join(stage)
    bundle = bundles[MODEL]
    pipe = bundle["model"]
    estimator = pipe[-1]
    classes = list(estimator.classes_)
    a, normal = (classes.index(label) for label in ("A121", "normal"))
    loads = [classes.index(label) for label in ("A121", "B82", "C182")]
    z = pd.DataFrame(pipe[:-1].transform(x), index=x.index, columns=features)
    logits = pipe.decision_function(x)
    np.testing.assert_allclose(logits, z @ estimator.coef_.T + estimator.intercept_,
                               atol=2e-13, rtol=0)
    probabilities = softmax(logits, axis=1)
    np.testing.assert_allclose(probabilities, scores[[f"p_{label}" for label in classes]].astype(float),
                               atol=2e-13, rtol=0)
    threshold = bundle["threshold"]
    archive_alarm = scores["alarm"].astype(str).str.lower().eq("true")
    np.testing.assert_allclose(scores["threshold"], threshold, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(1 - probabilities[:, normal] > threshold, archive_alarm)
    labels = np.array(classes)
    predicted_load = labels[np.array(loads)[probabilities[:, loads].argmax(axis=1)]]
    np.testing.assert_array_equal(np.where(archive_alarm, predicted_load, "normal"),
                                  scores["predicted_label"])

    selectors = {
        "normal_as_A": scores.label.eq("normal") & scores.predicted_label.eq("A121"),
        "B_as_A": scores.label.eq("B82") & scores.predicted_label.eq("A121"),
        "true_A_hit": scores.label.eq("A121") & scores.predicted_label.eq("A121"),
        "true_A_miss": scores.label.eq("A121") & ~archive_alarm,
    }
    if len(scores) != 180 or any(selectors[key].sum() != count for key, count in
                                 zip(selectors, (12, 12, 12, 18), strict=True)):
        raise ValueError("Unexpected evaluation counts or A121 classifications")
    predecessors = {"A1": "N0", "A2": "N5", "B1": "N1", "B2": "N3"}
    contrast_weights = {name: estimator.coef_[a] - estimator.coef_[classes.index(other)]
                        for name, other in CONTRASTS.items()}
    contrast_bias = {name: estimator.intercept_[a] - estimator.intercept_[classes.index(other)]
                     for name, other in CONTRASTS.items()}
    relative_weights = estimator.coef_[loads] - estimator.coef_[normal]
    relative_bias = estimator.intercept_[loads] - estimator.intercept_[normal]
    normal_controls = scores.label.eq("normal") & ~archive_alarm
    correct_b = scores.label.eq("B82") & scores.predicted_label.eq("B82")
    cohort_rows, detail_rows, case_rows, replacement_rows = [], [], [], []

    def describe(name, selected):
        for feature, group in zip(features, groups, strict=True):
            values = x.loc[selected, feature]
            cohort_rows.append(dict(cohort=name, n=len(values), feature=feature, sensor_group=group,
                                    mean=values.mean(), p10=values.quantile(.1),
                                    median=values.median(), p90=values.quantile(.9)))

    def explain(comparison, cohort, selected, reference):
        ref_raw = x.loc[reference].mean()
        ref = z.loc[reference].mean().to_numpy()
        actual = z.loc[selected].to_numpy()
        delta = actual - ref
        if not len(actual) or not reference.any():
            raise ValueError(f"Empty cases or reference for {comparison}/{cohort}")
        values = {name: delta * weight for name, weight in contrast_weights.items()}
        reference_scores = {name: ref @ weight + contrast_bias[name]
                            for name, weight in contrast_weights.items()}
        actual_scores = {name: actual @ weight + contrast_bias[name]
                         for name, weight in contrast_weights.items()}
        values["alarm"], reference_scores["alarm"], actual_scores["alarm"] = helper[
            "alarm_attribution"](actual, ref, relative_weights, relative_bias)
        np.testing.assert_allclose(actual_scores["alarm"],
                                   logsumexp(actual @ relative_weights.T + relative_bias, axis=1),
                                   atol=2e-12, rtol=0)
        for name in values:
            np.testing.assert_allclose(values[name].sum(axis=1),
                                       actual_scores[name] - reference_scores[name],
                                       atol=2e-12, rtol=0)
        for i, cycle in enumerate(x.index[selected]):
            row = scores.loc[cycle]
            base = dict(comparison=comparison, cohort=cohort, cycle_id=cycle, stage=row.stage_id,
                        label=row.label, predicted_label=row.predicted_label,
                        reference_stage=",".join(scores.loc[reference, "stage_id"].unique()),
                        reference_n=int(reference.sum()))
            stats = {f"{name}_{key}": value for name in values for key, value in (
                ("log_odds", actual_scores[name][i]), ("reference_log_odds", reference_scores[name]),
                ("delta", values[name][i].sum()))}
            stats.update({f"{name}_{group}": values[name][i, groups == group].sum()
                          for name in values for group in sorted(set(groups))})
            case_rows.append(dict(base, score=row.score, margin=row.margin, p_A121=row.p_A121,
                                  p_B82=row.p_B82, p_C182=row.p_C182, p_normal=row.p_normal, **stats))
            for j, feature in enumerate(features):
                detail_rows.append(dict(base, feature=feature, sensor_group=groups[j],
                                        raw=x.at[cycle, feature], reference_raw_mean=ref_raw[feature],
                                        delta_standardized=delta[i, j],
                                        **{name: contribution[i, j] for name, contribution in values.items()}))
            replacement_masks = {group: groups == group for group in sorted(set(groups))}
            replacement_masks["ID1_3_5s_IQR"] = np.array(features) == "ID1_3_5s_IQR"
            replacement_masks["ID1_IQR_and_ID2"] = np.isin(
                features, ("ID1_3_5s_IQR", "position_ID2_raw_range", "position_ID2_raw_std"))
            for group, mask in replacement_masks.items():
                revised = np.where(mask, ref, actual[i])
                revised_logits = revised @ estimator.coef_.T + estimator.intercept_
                revised_score = float(expit(logsumexp(revised_logits[loads] - revised_logits[normal])))
                revised_load = classes[np.array(loads)[revised_logits[loads].argmax()]]
                replacement_rows.append(dict(base, replaced_group=group, score_after=revised_score,
                                             alarm_after=revised_score > threshold,
                                             predicted_after=revised_load if revised_score > threshold else "normal"))

    for name, selection in selectors.items():
        describe(name, selection)
        for block in scores.loc[selection, "stage_id"].unique():
            selected = selection & scores.stage_id.eq(block)
            normal_block = predecessors.get(block, block)
            explain("vs_local_unalarmed_normal", name, selected,
                    normal_controls & scores.stage_id.eq(normal_block))
            if name == "B_as_A":
                explain("vs_same_block_correct_B", name, selected, correct_b & scores.stage_id.eq(block))
        if name in ("normal_as_A", "B_as_A"):
            explain("vs_actual_A_hits", name, selection, selectors["true_A_hit"])
    for name, selection in (("normal_unalarmed", normal_controls), ("true_B_correct", correct_b),
                            ("actual_A_all", scores.label.eq("A121"))):
        describe(name, selection)

    historical = inputs.csv(
        project / "hardware/xarm_model_versions/v5_1_multisession_normal_study/metrics/CYCLE_FEATURES.csv"
    ).set_index("cycle_id")
    roles = inputs.csv(project / STUDY / "FIT_CAL_MANIFEST.csv").set_index("cycle_id")
    calibration = inputs.csv(project / STUDY / "CALIBRATION_SCORES.csv").set_index("cycle_id")
    if not all(frame.index.is_unique for frame in (x, historical, roles, calibration)):
        raise ValueError("Duplicate cycle IDs in an input")
    cal_ids = roles.index[roles.role.eq("calibration")]
    np.testing.assert_array_equal(cal_ids.sort_values(), calibration.index.sort_values())
    cal_x = historical.loc[calibration.index, features]
    cal_z = pipe[:-1].transform(cal_x)
    transfer_rows, coefficient_rows = [], []
    new_normal = scores.label.eq("normal")
    for model_id in (MODEL, "LOGISTIC_MONSTER_GEL"):
        fit_bundle = bundles[model_id]
        model = fit_bundle["model"]
        normal_idx = list(model.classes_).index("normal")
        last = model[-1]
        a_idx = list(last.classes_).index("A121")
        for contrast, other in CONTRASTS.items():
            weight = last.coef_[a_idx] - last.coef_[list(last.classes_).index(other)]
            for feature, scaled, scale in zip(features, weight, model[-2].scale_, strict=True):
                coefficient_rows.append(dict(model_id=model_id, contrast=contrast, feature=feature,
                                             coefficient_standardized=scaled, coefficient_raw=scaled / scale))
        cal_scores = 1 - model.predict_proba(cal_x)[:, normal_idx]
        np.testing.assert_allclose(cal_scores, calibration[model_id], atol=2e-13, rtol=0)
        new_scores = 1 - model.predict_proba(x.loc[new_normal])[:, normal_idx]
        for period, sample in (("historical_calibration", cal_scores), ("new_normal", new_scores)):
            boundary = fit_bundle["threshold"]
            transfer_rows.append(dict(model_id=model_id, cohort=period, n=len(sample),
                                      fp=int((sample > boundary).sum()), threshold=boundary,
                                      median=np.quantile(sample, .5), p90=np.quantile(sample, .9)))
    expected_features = roles[roles.role.eq("train")].groupby("label").size().to_dict()
    if expected_features != {"A121": 30, "B82": 30, "C182": 30, "normal": 50}:
        raise ValueError("Unexpected frozen bolt training counts")
    for label, ids in roles.loc[roles.role.eq("train")].groupby("label").groups.items():
        for feature, group in zip(features, groups, strict=True):
            sample = historical.loc[ids, feature]
            cohort_rows.append(dict(cohort=f"fit_{label}", n=len(sample), feature=feature, sensor_group=group,
                                    mean=sample.mean(), p10=sample.quantile(.1),
                                    median=sample.median(), p90=sample.quantile(.9)))
    cal_weights = contrast_weights["A_vs_normal"]
    mean_delta_z = z.loc[new_normal].mean().to_numpy() - cal_z.mean(axis=0)
    shift = pd.DataFrame(dict(feature=features, sensor_group=groups, calibration_raw_mean=cal_x.mean().to_numpy(),
                              new_normal_raw_mean=x.loc[new_normal].mean().to_numpy(),
                              delta_z=mean_delta_z, weight=cal_weights, A_vs_normal=mean_delta_z * cal_weights))
    np.testing.assert_allclose(shift["A_vs_normal"].sum(),
                               (logits[new_normal, a] - logits[new_normal, normal]).mean()
                               - (pipe.decision_function(cal_x)[:, a]
                                  - pipe.decision_function(cal_x)[:, normal]).mean(), atol=2e-12, rtol=0)

    # Re-read the four raw position responses used by the frozen gripper feature.
    raw_rows = []
    for folder, rows in cycles.groupby("source"):
        position = inputs.csv(Path(folder) / "robot/single_queries.csv")
        position = position.loc[position.servo_id.eq(1) & position.phase.eq("No3_A_to_B")
                                & position.status.eq("complete") & position.value_valid.eq(1)]
        for cycle, row in rows.iterrows():
            begin = int(row.No3_begin_ns)
            sample = position.loc[position.cycle_id.eq(cycle)
                                  & position.response_ns.ge(begin + 3_000_000_000)
                                  & position.response_ns.lt(begin + 5_000_000_000)].sort_values("response_ns")
            readings = sample.position_raw.to_numpy()
            assert len(readings) == 4
            iqr = np.diff(np.quantile(readings, [.25, .75], method="linear"))[0]
            np.testing.assert_allclose(iqr, x.at[cycle, "ID1_3_5s_IQR"], atol=1e-14, rtol=0)
            for time, value in zip(sample.response_ns, readings, strict=True):
                raw_rows.append(dict(cycle_id=cycle, stage=scores.at[cycle, "stage_id"], label=row.label,
                                     predicted_label=scores.at[cycle, "predicted_label"],
                                     relative_s=(int(time) - begin) / 1e9, ID1_position_raw=value))
    details = pd.DataFrame(detail_rows)
    summaries = details.groupby(["comparison", "cohort", "sensor_group", "feature"], sort=False).agg(
        n=("cycle_id", "size"), raw_mean=("raw", "mean"), reference_raw_mean=("reference_raw_mean", "mean"),
        mean_delta_z=("delta_standardized", "mean"),
        **{name: (name, "mean") for name in ("alarm", *CONTRASTS)},
    ).reset_index()
    if not inputs.unchanged():
        raise ValueError("A tracked model, dependency or result changed during analysis")
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (
        ("CASE_ATTRIBUTIONS", pd.DataFrame(case_rows)), ("FEATURE_ATTRIBUTIONS", details),
        ("FEATURE_SUMMARY", summaries), ("COHORT_FEATURES", pd.DataFrame(cohort_rows)),
        ("GROUP_REPLACEMENTS", pd.DataFrame(replacement_rows)),
        ("CALIBRATION_TRANSFER", pd.DataFrame(transfer_rows)), ("A_NORMAL_LOG_ODDS_SHIFT", shift),
        ("LOGIT_COEFFICIENTS", pd.DataFrame(coefficient_rows)),
        ("ID1_RAW_3_TO_5S", pd.DataFrame(raw_rows)),
    ):
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    metadata = dict(model=MODEL, threshold=threshold, threshold_log_odds=float(logit(threshold)),
                    evaluated_cycles=len(x), cases={name: int(mask.sum()) for name, mask in selectors.items()},
                    tracked_input_sha256=inputs.hashes, scores_and_predictions_reproduced=True,
                    contribution_sums_verified=True, calibration_scores_reproduced=True,
                    raw_ID1_IQR_reproduced=True, refit=False, threshold_change=False)
    (output / "VERIFICATION.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.project.resolve(), args.output.resolve())
