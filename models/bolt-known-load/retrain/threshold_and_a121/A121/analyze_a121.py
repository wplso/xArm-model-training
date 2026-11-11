"""Compare the two saved bolt models on identical A121 and paired normal cycles."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp, softmax
from sklearn.metrics import average_precision_score, roc_auc_score


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(frame):
    result = frame.set_index("cycle_id")
    if not result.index.is_unique:
        raise ValueError("Duplicate cycle_id")
    return result


def explain(bundle, x, a_rows, normal_rows, archived, alarm_attribution):
    """Return exact logit differences for the Cartesian A121–normal pairs."""
    pipeline = bundle["model"]
    estimator = pipeline[-1]
    classes = list(estimator.classes_)
    a, normal = (classes.index(label) for label in ("A121", "normal"))
    loads = [classes.index(label) for label in ("A121", "B82", "C182")]
    z = pipeline[:-1].transform(x)
    logits = z @ estimator.coef_.T + estimator.intercept_
    np.testing.assert_allclose(logits, pipeline.decision_function(x), atol=2e-13, rtol=0)
    probability = softmax(logits, axis=1)
    np.testing.assert_allclose(probability, archived[[f"p_{label}" for label in classes]], atol=2e-13, rtol=0)
    alarm_log_odds = logsumexp(logits[:, loads] - logits[:, normal, None], axis=1)
    score = expit(alarm_log_odds)
    np.testing.assert_allclose(score, archived.score, atol=2e-13, rtol=0)
    np.testing.assert_allclose(archived.threshold, bundle["threshold"], atol=1e-15, rtol=0)
    alarm = score > bundle["threshold"]
    np.testing.assert_array_equal(alarm, archived.alarm)
    load_types = np.array(classes)[np.array(loads)[probability[:, loads].argmax(axis=1)]]
    np.testing.assert_array_equal(np.where(alarm, load_types, "normal"), archived.predicted_label)

    a_weight = estimator.coef_[a] - estimator.coef_[normal]
    relative_weights = estimator.coef_[loads] - estimator.coef_[normal]
    relative_bias = estimator.intercept_[loads] - estimator.intercept_[normal]
    cases = z[a_rows]
    references = z[normal_rows]
    a_normal = (cases[None, :, :] - references[:, None, :]) * a_weight
    attribution = np.stack([
        alarm_attribution(cases, reference, relative_weights, relative_bias)[0]
        for reference in references
    ])
    expected_alarm_gap = alarm_log_odds[a_rows][None, :] - alarm_log_odds[normal_rows][:, None]
    a_normal_log_odds = logits[:, a] - logits[:, normal]
    expected_a_gap = a_normal_log_odds[a_rows][None, :] - a_normal_log_odds[normal_rows][:, None]
    np.testing.assert_allclose(attribution.sum(axis=2), expected_alarm_gap, atol=1e-10, rtol=0)
    np.testing.assert_allclose(a_normal.sum(axis=2), expected_a_gap, atol=2e-12, rtol=0)
    effective_weight = a_weight / pipeline[-2].scale_
    x_raw = pipeline[0].transform(x)
    np.testing.assert_allclose(
        effective_weight * (x_raw[a_rows].mean(axis=0) - x_raw[normal_rows].mean(axis=0)),
        a_normal.mean(axis=(0, 1)), atol=2e-12, rtol=0,
    )
    return {
        "score": score, "alarm": alarm, "predicted_load_before_alarm_gate": load_types,
        "alarm_log_odds": alarm_log_odds, "a_normal_log_odds": a_normal_log_odds,
        "alarm_contribution": attribution, "a_normal_contribution": a_normal,
        "a_normal_slope": effective_weight,
        "wins": np.sign(score[a_rows][None, :] - score[normal_rows][:, None]),
        "max_alarm_decomposition_error": float(np.abs(attribution.sum(axis=2) - expected_alarm_gap).max()),
        "max_a_normal_decomposition_error": float(np.abs(a_normal.sum(axis=2) - expected_a_gap).max()),
    }


def main(project, output):
    hardware = project / "hardware"
    model_root = hardware / "xarm_model_versions"
    retrain = hardware / "No3_bolt_normal_retrain_20261111/analysis/retrained_bolt_normal"
    earlier = hardware / "No3_bolt_ABC_interleaved_20261111/analysis/frozen_evaluation"
    frozen = model_root / "bolt_known_load_prospective_20261111"
    historical_features = model_root / "v5_1_multisession_normal_study/metrics/CYCLE_FEATURES.csv"
    helper = model_root / "binary_logistic_load_alert_20261111/analyze_multiclass.py"
    model_manifest = frozen / "FROZEN_MODELS.json"
    new_verification = retrain / "VERIFICATION.json"
    new_manifest = retrain / "FIT_CAL_HOLDOUT_MANIFEST.csv"
    features_path = earlier / "CYCLE_FEATURES.csv"
    scores_path = retrain / "CYCLE_SCORES.csv"
    new_features = retrain / "NEW_NORMAL_FEATURES.csv"
    config = read_json(model_manifest)
    new_config = read_json(new_verification)
    features = config["features"]
    assert features == new_config["features"] and len(features) == len(set(features)) == 24
    source_paths = [
        model_manifest, new_verification, new_manifest, helper, features_path,
        scores_path, new_features, historical_features, earlier / "CYCLE_STAGE_MAP.csv",
        earlier / "CYCLE_SCORES.csv",
    ]
    expected = {
        "old": (frozen, next(row for row in config["models"] if row["model_id"] == "LOGISTIC_BOLT")),
        "expanded": (retrain, new_config["candidate"]),
    }
    bundles = {}
    for key, (folder, spec) in expected.items():
        path = folder / spec["artifact"]
        assert digest(path) == spec["sha256"], f"Unexpected model hash: {path}"
        bundle = joblib.load(path)
        assert list(bundle["features"]) == features and bundle["threshold"] == spec["threshold"]
        source_paths.append(path)
        bundles[key] = bundle
    hashes = {path: digest(path) for path in source_paths}
    raw_features = indexed(pd.read_csv(features_path, keep_default_na=False))
    cohort = raw_features.loc[
        raw_features.label.eq("A121") |
        (raw_features.label.eq("normal") & raw_features.pair_condition.eq("A121"))
    ]
    stage_map = indexed(pd.read_csv(earlier / "CYCLE_STAGE_MAP.csv"))
    cohort = cohort.join(stage_map)
    assert cohort.groupby(["label", "stage_id"]).size().to_dict() == {
        ("normal", "N0"): 30, ("normal", "N5"): 10, ("A121", "A1"): 15, ("A121", "A2"): 15,
    }
    x = cohort[features].astype(float)
    assert np.isfinite(x.to_numpy()).all(), "Unexpected missing or infinite A121 evaluation input"
    a_rows = cohort.label.eq("A121").to_numpy()
    normal_rows = ~a_rows
    previous_scores = pd.read_csv(earlier / "CYCLE_SCORES.csv")
    archived = pd.read_csv(scores_path).query("cohort == 'previously_seen_development_regression'")
    old_archived = archived[archived.model_id.eq("LOGISTIC_BOLT")].set_index("cycle_id").loc[x.index]
    previous = previous_scores[previous_scores.model_id.eq("LOGISTIC_BOLT")].set_index("cycle_id").loc[x.index]
    np.testing.assert_array_equal(old_archived.score, previous.score)
    np.testing.assert_array_equal(old_archived.alarm, previous.alarm)

    helpers = runpy.run_path(str(helper))
    model = {}
    metrics = []
    cycle_details = cohort[["label", "stage_id"]].copy()
    for key, (_, spec) in expected.items():
        model_archive = indexed(archived[archived.model_id.eq(spec["model_id"])]).loc[x.index]
        assert model_archive.label.equals(cohort.label)
        values = explain(bundles[key], x, a_rows, normal_rows, model_archive, helpers["alarm_attribution"])
        model[key] = values
        wins = values["wins"]
        auc = roc_auc_score(a_rows, values["score"])
        np.testing.assert_allclose(auc, (np.count_nonzero(wins > 0) + 0.5 * np.count_nonzero(wins == 0)) / wins.size)
        metrics.append({
            "model": spec["model_id"], "threshold": spec["threshold"], "auc": auc,
            "average_precision": average_precision_score(a_rows, values["score"]),
            "A121_n": int(a_rows.sum()), "paired_normal_n": int(normal_rows.sum()),
            "A121_alarm_n": int(values["alarm"][a_rows].sum()),
            "paired_normal_alarm_n": int(values["alarm"][normal_rows].sum()),
            "A121_score_median": float(np.median(values["score"][a_rows])),
            "normal_score_median": float(np.median(values["score"][normal_rows])),
            "A121_normal_log_odds_mean_gap": float(values["a_normal_contribution"].mean(axis=(0, 1)).sum()),
            "alarm_log_odds_mean_gap": float(values["alarm_contribution"].mean(axis=(0, 1)).sum()),
            "pair_wins": int(np.count_nonzero(wins > 0)), "pair_ties": int(np.count_nonzero(wins == 0)),
        })
        for field in ("score", "alarm", "predicted_load_before_alarm_gate", "alarm_log_odds", "a_normal_log_odds"):
            cycle_details[f"{key}_{field}"] = values[field]

    historical = indexed(pd.read_csv(historical_features, usecols=["cycle_id", *features]))
    added = indexed(pd.read_csv(new_features, usecols=["cycle_id", *features]))
    fit = indexed(pd.read_csv(new_manifest, keep_default_na=False))
    labels = fit.loc[fit.role.eq("train"), "label"]
    old_ids = list(bundles["old"]["train_ids"])
    new_ids = list(bundles["expanded"]["train_ids"])
    assert set(old_ids).issubset(new_ids) and set(labels.index) == set(new_ids) and not set(x.index).intersection(new_ids)
    added_ids = sorted(set(new_ids) - set(old_ids))
    assert len(added_ids) == 120 and labels.loc[added_ids].eq("normal").all()
    old_fit_a = historical.loc[labels.loc[old_ids][lambda rows: rows.eq("A121")].index]
    old_fit_n = historical.loc[labels.loc[old_ids][lambda rows: rows.eq("normal")].index]
    added_normals = added.loc[added_ids]
    assert len(old_fit_a) == 30 and len(old_fit_n) == 50

    raw_gap = x.loc[a_rows].mean() - x.loc[normal_rows].mean()
    feature_effects = pd.DataFrame({
        "sensor_group": [helpers["sensor_group"](feature) for feature in features],
        "evaluation_A121_raw_mean": x.loc[a_rows].mean(),
        "evaluation_normal_raw_mean": x.loc[normal_rows].mean(),
        "evaluation_raw_gap": raw_gap,
        "old_training_A121_raw_mean": old_fit_a.mean(),
        "old_training_normal_raw_mean": old_fit_n.mean(),
        "added_training_normal_raw_mean": added_normals.mean(),
        "expanded_training_normal_raw_mean": pd.concat([old_fit_n, added_normals]).mean(),
    }, index=pd.Index(features, name="feature"))
    old_win = model["old"]["wins"]
    new_win = model["expanded"]["wins"]
    subsets = {
        "all": np.ones_like(old_win, dtype=bool),
        "lost_order": (old_win > 0) & (new_win < 0),
        "gained_order": (old_win < 0) & (new_win > 0),
    }
    for key, values in model.items():
        feature_effects[f"{key}_A121_normal_slope_per_raw"] = values["a_normal_slope"]
        feature_effects[f"{key}_A121_normal_gap_contribution"] = values["a_normal_contribution"].mean(axis=(0, 1))
        for name, mask in subsets.items():
            if mask.any():
                feature_effects[f"{key}_alarm_gap_contribution_{name}"] = values["alarm_contribution"][mask].mean(axis=0)
    for name in ("A121_normal_gap_contribution", *(f"alarm_gap_contribution_{name}" for name, mask in subsets.items() if mask.any())):
        feature_effects[f"expanded_minus_old_{name}"] = feature_effects[f"expanded_{name}"] - feature_effects[f"old_{name}"]
    contributions = feature_effects.filter(regex="_contribution")
    grouped = contributions.groupby(feature_effects.sensor_group).sum()
    block_metrics = []
    block_features = []
    for a_stage, n_stage in (("A1", "N0"), ("A1", "N5"), ("A2", "N0"), ("A2", "N5")):
        a_mask = cohort.stage_id.eq(a_stage).to_numpy()
        n_mask = cohort.stage_id.eq(n_stage).to_numpy()
        pair_mask = n_mask[normal_rows][:, None] & a_mask[a_rows][None, :]
        block_rows = a_mask | n_mask
        block_feature = pd.DataFrame({
            "A121_stage": a_stage, "normal_stage": n_stage,
            "feature": features, "raw_mean_gap": (x.loc[a_mask].mean() - x.loc[n_mask].mean()).to_numpy(),
        })
        for key, values in model.items():
            for contrast in ("a_normal", "alarm"):
                block_feature[f"{key}_{contrast}_gap_contribution"] = values[f"{contrast}_contribution"][pair_mask].mean(axis=0)
            block_metrics.append({
                "A121_stage": a_stage, "normal_stage": n_stage, "model": key,
                "A121_n": int(a_mask.sum()), "normal_n": int(n_mask.sum()),
                "auc": roc_auc_score(a_mask[block_rows], values["score"][block_rows]),
                "A121_score_median": float(np.median(values["score"][a_mask])),
                "normal_score_median": float(np.median(values["score"][n_mask])),
                "alarm_log_odds_mean_gap": values["alarm_contribution"][pair_mask].mean(axis=0).sum(),
            })
        for contrast in ("a_normal", "alarm"):
            block_feature[f"expanded_minus_old_{contrast}_gap_contribution"] = (
                block_feature[f"expanded_{contrast}_gap_contribution"] - block_feature[f"old_{contrast}_gap_contribution"]
            )
        block_features.append(block_feature)
    assert all(digest(path) == previous_hash for path, previous_hash in hashes.items()), "An input changed"
    evidence = {
        "input_sha256": {str(path): value for path, value in hashes.items()},
        "score_probability_threshold_and_predictions_reproduced": True,
        "models_and_thresholds_modified": False,
        "evaluation_cohort": "previously_seen_development_regression",
        "old_training_A121_n": len(old_fit_a), "old_training_normal_n": len(old_fit_n),
        "added_training_normal_n": len(added_normals),
        "max_alarm_decomposition_error": max(m["max_alarm_decomposition_error"] for m in model.values()),
        "max_A121_normal_decomposition_error": max(m["max_a_normal_decomposition_error"] for m in model.values()),
        "n_pairs_by_group": {name: int(mask.sum()) for name, mask in subsets.items()},
        "pair_order_counts": {
            f"{int(a)}->{int(b)}": int(((old_win == a) & (new_win == b)).sum())
            for a in (-1, 0, 1) for b in (-1, 0, 1)
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    feature_effects.to_csv(output / "FEATURE_GAP_ATTRIBUTION.csv")
    grouped.to_csv(output / "SENSOR_GROUP_GAP_ATTRIBUTION.csv")
    cycle_details.to_csv(output / "A121_PAIRED_NORMAL_SCORES.csv")
    pd.DataFrame(metrics).to_csv(output / "METRICS.csv", index=False)
    pd.DataFrame(block_metrics).to_csv(output / "BLOCK_METRICS.csv", index=False)
    pd.concat(block_features, ignore_index=True).to_csv(output / "BLOCK_FEATURE_GAP_ATTRIBUTION.csv", index=False)
    (output / "VERIFICATION.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(evidence["n_pairs_by_group"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.project.resolve(), args.output.resolve())
