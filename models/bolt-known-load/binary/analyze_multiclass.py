"""Explain archived OLD_TO_BOLT multiclass logistic scores without fitting a model."""

import argparse
from pathlib import Path
import runpy

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits


def sensor_group(feature):
    if feature.startswith("current_"):
        return "current"
    if feature.startswith("voltage_"):
        return "voltage"
    if feature.startswith(("position_", "ID1_")):
        return "position"
    return "vibration"


def alarm_attribution(z, reference, coefficients, intercepts):
    """Integrate the derivative of logsumexp(load logit minus normal logit)."""
    delta = z - reference
    start = coefficients @ reference + intercepts
    changes = delta @ coefficients.T
    points, weights = np.polynomial.legendre.leggauss(48)
    path_logits = start + ((points[None, :, None] + 1) / 2) * changes[:, None, :]
    average_probability = np.einsum("q,nqk->nk", weights / 2, softmax(path_logits, axis=-1))
    contributions = delta * (average_probability @ coefficients)
    reference_logit = logsumexp(start)
    case_logit = logsumexp(start + changes, axis=1)
    return contributions, reference_logit, case_logit


def analyze(project, output):
    previous = project / "hardware/xarm_model_versions/supervised_load_session_holdout_20261111"
    helper = runpy.run_path(str(previous / "run_supervised.py"))
    old_raw, bolt_raw, _, frozen_v4, _, bolt_roles, _, _ = helper["load_inputs"](project)
    features = frozen_v4["features"]
    old, bolt = helper["input_frames"](old_raw, bolt_raw, features)
    train, cal, evaluation = helper["evaluation_splits"](old, bolt, bolt_roles)["OLD_TO_BOLT"]
    read_csv = helper["read_csv"]
    digest = helper["digest"]
    result_dir = previous / "results"
    archive = read_csv(result_dir / "CYCLE_SCORES.csv")
    archive = archive.loc[archive.split_id.eq("OLD_TO_BOLT") & archive.role.eq("evaluation")]
    multi = archive.loc[archive.model_id.eq("LOGISTIC")].set_index("cycle_id").loc[evaluation.index]
    baseline = archive.loc[archive.model_id.eq("V4_PCA_Q")].set_index("cycle_id").loc[evaluation.index]
    spec = read_csv(result_dir / "MODEL_INDEX.csv").query(
        "split_id == 'OLD_TO_BOLT' and model_id == 'LOGISTIC'"
    ).iloc[0]
    artifact = result_dir / spec.path
    assert digest(artifact) == spec.sha256
    bundle = joblib.load(artifact)
    assert bundle["train_ids"] == train.index.tolist()
    assert bundle["calibration_ids"] == cal.index.tolist()
    assert bundle["features"] == features
    assert bundle["threshold"] == spec.threshold
    assert evaluation.label.equals(multi.label)
    pipe = bundle["model"]
    with threadpool_limits(limits=1):
        x = evaluation[features]
        z = pipe[:-1].transform(x)
        logits = pipe.decision_function(x)
        probabilities = helper["p_scores"](pipe, x, features)
    np.testing.assert_allclose(probabilities, multi[[f"p_{c}" for c in pipe.classes_]], atol=1e-14, rtol=0)
    np.testing.assert_allclose(1 - probabilities.normal, multi.score, atol=1e-14, rtol=0)
    np.testing.assert_array_equal((1 - probabilities.normal).gt(bundle["threshold"]), multi.alarm)

    estimator = pipe[-1]
    classes = list(estimator.classes_)
    a, b, normal = [classes.index(label) for label in ("A121", "B82", "normal")]
    loading = [classes.index(label) for label in ("A121", "B82", "C182")]
    contrast = estimator.coef_[loading] - estimator.coef_[normal]
    relative_intercepts = estimator.intercept_[loading] - estimator.intercept_[normal]
    feature_groups = pd.Series([sensor_group(feature) for feature in features], index=features)

    cases_output, details_output = [], []

    def add_cases(name, cases, references, mode, ref_label):
        assert cases.any() and references.any()
        assert not (cases & references).any()
        z_ref = z[references].mean(axis=0)
        x_ref = x.loc[references].mean(axis=0)
        selected = evaluation.index[cases]
        if mode == "B82_minus_A121":
            weights = estimator.coef_[b] - estimator.coef_[a]
            intercept = estimator.intercept_[b] - estimator.intercept_[a]
            values = (z[cases] - z_ref) * weights
            ref_score = z_ref @ weights + intercept
            case_scores = logits[cases, b] - logits[cases, a]
        else:
            values, ref_score, case_scores = alarm_attribution(
                z[cases], z_ref, contrast, relative_intercepts
            )
            np.testing.assert_allclose(case_scores, logsumexp(
                logits[cases][:, loading] - logits[cases, normal, None], axis=1
            ), atol=1e-12, rtol=0)
        observed_delta = case_scores - ref_score
        np.testing.assert_allclose(values.sum(axis=1), observed_delta, atol=1e-12, rtol=0)
        for i, cycle_id in enumerate(selected):
            measured = multi.loc[cycle_id]
            case_row = dict(
                comparison=name, cycle_id=cycle_id, label=measured.label, pair=measured.pair,
                cohort=measured.cohort, session_id=measured.session_id, type_prediction=measured.prediction,
                v4_alarm=baseline.loc[cycle_id, "alarm"], logistic_alarm=measured.alarm,
                logistic_score=measured.score, logistic_threshold=measured.threshold,
                p_normal=measured.p_normal, p_A121=measured.p_A121, p_B82=measured.p_B82,
                p_C182=measured.p_C182,
                score_explained=mode, reference=ref_label, reference_n=int(references.sum()),
                reference_logit=ref_score, case_logit=case_scores[i], delta_logit=observed_delta[i],
            )
            case_row.update({f"contribution_{group}": values[i, feature_groups.eq(group)].sum()
                             for group in feature_groups.unique()})
            cases_output.append(case_row)
            for j, feature in enumerate(features):
                value = x.at[cycle_id, feature]
                details_output.append(dict(
                    comparison=name, cycle_id=cycle_id, feature=feature, sensor_group=feature_groups.at[feature],
                    raw_value=value, reference_raw_mean=x_ref.at[feature], delta_raw=value-x_ref.at[feature],
                    delta_standardized=z[cases][i, j]-z_ref[j], contribution_logit=values[i, j],
                ))

    a_rows = multi.label.eq("A121")
    misclassified_a = a_rows & multi.alarm & multi.prediction.eq("B82")
    correct_a = a_rows & multi.alarm & multi.prediction.eq("A121")
    assert (misclassified_a.sum(), correct_a.sum()) == (6, 9)
    add_cases("A121_called_B82", misclassified_a, correct_a, "B82_minus_A121", "alarmed_correct_A121")
    for target, count in [("A121", 4), ("B82", 3), ("C182", 1)]:
        paired_normal = multi.label.eq("normal") & multi.pair.eq(target)
        fp, tn = paired_normal & multi.alarm, paired_normal & ~multi.alarm
        assert fp.sum() == count
        assert multi.loc[paired_normal, "session_id"].nunique() == 1
        add_cases(f"{target}_paired_normal_FP", fp, tn, "load_vs_normal", "same_pair_normal_not_alarmed")
    b_hits = multi.label.eq("B82") & multi.alarm
    b_tn = multi.label.eq("normal") & multi.pair.eq("B82") & ~multi.alarm
    assert (b_hits.sum(), b_tn.sum()) == (26, 7)
    add_cases("B82_detected", b_hits, b_tn, "load_vs_normal", "B82_paired_normal_not_alarmed")

    cases = pd.DataFrame(cases_output)
    detail = pd.DataFrame(details_output)
    features_summary = detail.groupby(["comparison", "sensor_group", "feature"], sort=False).agg(
        n=("cycle_id", "size"), raw_case_mean=("raw_value", "mean"),
        raw_reference_mean=("reference_raw_mean", "mean"),
        mean_raw_delta=("delta_raw", "mean"), mean_standardized_delta=("delta_standardized", "mean"),
        mean_logit_contribution=("contribution_logit", "mean"),
        positive_contributions=("contribution_logit", lambda values: values.gt(0).sum()),
    ).reset_index()
    assert len(cases) == 40 and len(detail) == 40 * len(features)
    assert digest(artifact) == spec.sha256
    output.mkdir(parents=True, exist_ok=False)
    cases.to_csv(output / "CASE_ATTRIBUTIONS.csv", index=False)
    detail.to_csv(output / "FEATURE_ATTRIBUTIONS.csv", index=False)
    features_summary.to_csv(output / "ATTRIBUTION_SUMMARY.csv", index=False)
    print(cases.groupby("comparison", sort=False)[["delta_logit", *[
        f"contribution_{group}" for group in feature_groups.unique()
    ]]].mean().round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("attribution"))
    args = parser.parse_args()
    analyze(args.project, args.output)
