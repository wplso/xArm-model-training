"""Refit three one-feature ablations using the archived bolt training split."""

import argparse
import json
from pathlib import Path
import runpy
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits


CAMPAIGN = Path("hardware/No3_bolt_normal_retrain_20261111")
BASELINE = "LOGISTIC_BOLT_NORMAL_EXPANDED"
OLD = "LOGISTIC_BOLT"
LOADS = ("A121", "B82", "C182")
SEEN = "previously_seen_development_regression"
ABLATIONS = {
    "DROP_Z_G_STD": "z_g_std",
    "DROP_ID6_STD": "position_ID6_raw_std",
    "DROP_ID1_IQR": "ID1_3_5s_IQR",
}


def score(pipe, features, threshold, model, cohort, frame):
    probabilities = pd.DataFrame(pipe.predict_proba(frame[features]), columns=pipe.classes_, index=frame.index)
    values = 1 - probabilities.normal
    alarm = values.gt(threshold)
    labels = probabilities[list(LOADS)].idxmax(axis=1).where(alarm, "normal")
    return frame[["cycle_id", "session_id", "label", "pair_condition"]].assign(
        model_id=model, cohort=cohort, score=values, threshold=threshold, alarm=alarm,
        predicted_label=labels, **probabilities.add_prefix("p_"),
    )


def check_baseline(replay, archived):
    replay = replay.set_index(["model_id", "cohort", "cycle_id"]).sort_index()
    archived = archived.loc[archived.model_id.isin((OLD, BASELINE))].set_index(
        ["model_id", "cohort", "cycle_id"]).sort_index()
    if not replay.index.equals(archived.index):
        raise ValueError("Baseline score keys differ from the archived evaluation")
    fields = ["score", "threshold", "p_normal", *[f"p_{label}" for label in LOADS]]
    np.testing.assert_allclose(replay[fields].to_numpy(dtype=float),
                               archived[fields].to_numpy(dtype=float), atol=2e-13, rtol=0)
    for field in ("label", "pair_condition", "alarm", "predicted_label"):
        pd.testing.assert_series_equal(replay[field], archived[field])


def comparison(scores, stages):
    metric_rows, block_rows = [], []
    seen = scores.loc[scores.cohort.eq(SEEN)].merge(stages, on="cycle_id", validate="many_to_one")
    for model, rows in seen.groupby("model_id", sort=False):
        for label, count in zip(LOADS, (40, 20, 20), strict=True):
            control = rows.loc[rows.label.eq("normal") & rows.pair_condition.eq(label)]
            load = rows.loc[rows.label.eq(label)]
            if (len(control), len(load)) != (count, 30):
                raise ValueError("Paired evaluation counts changed")
            pair = pd.concat([control, load])
            positive = pair.label.eq(label)
            metric_rows.append(dict(
                model_id=model, label=label, paired_normal_n=count, paired_normal_fp=int(control.alarm.sum()),
                load_n=len(load), load_tp=int(load.alarm.sum()), correct_type=int(load.predicted_label.eq(label).sum()),
                paired_auroc=roc_auc_score(positive, pair.score),
                paired_average_precision=average_precision_score(positive, pair.score),
                normal_score_median=control.score.median(), load_score_median=load.score.median(),
            ))
        for load_stage, normal_stage in (("A1", "N0"), ("A2", "N5"), ("A2", "N0"), ("A1", "N5")):
            pair = rows.loc[rows.stage_id.isin([load_stage, normal_stage])]
            positive = pair.stage_id.eq(load_stage)
            block_rows.append(dict(model_id=model, load_stage=load_stage, normal_stage=normal_stage,
                                   load_n=int(positive.sum()), normal_n=int((~positive).sum()),
                                   auroc=roc_auc_score(positive, pair.score),
                                   load_tp=int(pair.loc[positive, "alarm"].sum())))
    return pd.DataFrame(metric_rows), pd.DataFrame(block_rows)


def run(project, output):
    if output.exists():
        raise FileExistsError(output)
    campaign = project / CAMPAIGN
    result = campaign / "analysis/retrained_bolt_normal"
    helper = runpy.run_path(str(campaign / "analysis_code/retrain_bolt_normal.py"))
    evaluator, common, inputs, frozen, frozen_bundles, original, earlier = helper["read_reference"](project)
    del evaluator
    metadata = inputs.json(result / "VERIFICATION.json")
    features = metadata["features"]
    spec = metadata["candidate"]
    artifact = inputs.track(result / spec["artifact"])
    if spec["model_id"] != BASELINE or spec["sha256"] != common.sha(artifact) or features != frozen["features"]:
        raise ValueError("Expanded bolt reference differs from the archive")
    baseline = joblib.load(artifact)
    old = frozen_bundles[OLD]
    if baseline["features"] != features or baseline["threshold"] != spec["threshold"]:
        raise ValueError("Expanded reference threshold or features differ")

    manifest = inputs.csv(result / "FIT_CAL_HOLDOUT_MANIFEST.csv", keep_default_na=False)
    reviewed = inputs.csv(campaign / "NORMAL_MANIFEST.csv", keep_default_na=False)
    receipt = inputs.json(campaign / "OPERATOR_REVIEW.json")
    if (receipt["normal_manifest_sha256"] != common.sha(campaign / "NORMAL_MANIFEST.csv")
            or receipt["accepted_cycle_n"] != 240 or receipt["all_cycles_confirmed"] is not True):
        raise ValueError("New normal review does not match the manifest")
    new = inputs.csv(result / "NEW_NORMAL_FEATURES.csv", keep_default_na=False)
    keys = ["cycle_id", "session_id", "cycle_index", "source", "No3_begin_ns", "No3_end_ns", "label", "role", "stage"]
    pd.testing.assert_frame_equal(new[keys], reviewed[keys], check_dtype=False)
    new[features] = new[features].apply(pd.to_numeric, errors="raise")
    fit = pd.concat([original.reset_index(), new.loc[new.role.eq("train")]], ignore_index=True)
    cal = new.loc[new.role.eq("calibration")]
    holdout = new.loc[new.role.eq("evaluation")]
    partition = pd.concat([fit[manifest.columns], cal[manifest.columns], holdout[manifest.columns]], ignore_index=True)
    pd.testing.assert_frame_equal(partition.fillna("").astype(str), manifest.astype(str))
    if (fit.cycle_id.tolist() != baseline["train_ids"] or cal.cycle_id.tolist() != baseline["calibration_ids"]
            or (len(fit), len(cal), len(holdout), len(earlier)) != (260, 60, 60, 180)
            or fit.label.value_counts().to_dict() != {"normal": 170, "A121": 30, "B82": 30, "C182": 30}
            or any(not np.isfinite(frame[features]).all().all() for frame in (fit, cal, holdout, earlier))):
        raise ValueError("Expanded reference split or features changed")
    stages = inputs.csv(project / helper["PREVIOUS"] / "analysis/frozen_evaluation/CYCLE_STAGE_MAP.csv")
    archived = inputs.csv(result / "CYCLE_SCORES.csv", keep_default_na=False)
    scopes = {"new_calibration": cal, "new_normal_holdout": holdout, SEEN: earlier}
    models = {OLD: old, BASELINE: baseline}
    entries = []
    with threadpool_limits(limits=1):
        for name, bundle in models.items():
            entries.extend(score(bundle["model"], features, bundle["threshold"], name, scope, rows)
                           for scope, rows in scopes.items())
        check_baseline(pd.concat(entries, ignore_index=True), archived)
        candidates = {}
        for name, removed in ABLATIONS.items():
            selected = [feature for feature in features if feature != removed]
            if len(selected) != 23:
                raise ValueError(f"Expected one removed feature for {name}")
            pipe = clone(old["model"])
            with warnings.catch_warnings(record=True) as issued:
                warnings.simplefilter("always")
                pipe.fit(fit[selected], fit.label)
            if issued:
                raise RuntimeError(f"Unexpected fitting warning for {name}: {issued[0].message}")
            cal_values = 1 - pipe.predict_proba(cal[selected])[:, list(pipe.classes_).index("normal")]
            threshold = float(np.quantile(cal_values, .95, method="higher"))
            candidates[name] = dict(model=pipe, features=selected, removed_feature=removed,
                                    threshold=threshold, model_id=name, classes=list(pipe.classes_),
                                    train_ids=fit.cycle_id.tolist(), calibration_ids=cal.cycle_id.tolist())
            entries.extend(score(pipe, selected, threshold, name, scope, rows) for scope, rows in scopes.items())

    scores = pd.concat(entries, ignore_index=True)
    metrics, blocks = comparison(scores, stages)
    if len(scores) != 1500 or not inputs.unchanged():
        raise ValueError("Unexpected score count or changed model or data")
    output.mkdir(parents=True)
    (output / "models").mkdir()
    candidate_specs = []
    for name, bundle in candidates.items():
        path = output / "models" / f"{name}.joblib"
        joblib.dump(bundle, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(
            restored["model"].predict_proba(cal[bundle["features"]]),
            bundle["model"].predict_proba(cal[bundle["features"]]), atol=2e-13, rtol=0)
        if restored["threshold"] != bundle["threshold"] or restored["features"] != bundle["features"]:
            raise ValueError("Saved candidate differs from fitted model")
        candidate_specs.append(dict(model_id=name, artifact=path.relative_to(output).as_posix(),
                                    sha256=common.sha(path), threshold=bundle["threshold"],
                                    removed_feature=bundle["removed_feature"], features=bundle["features"]))
    rollup = helper["score_rollup"](scores)
    for filename, table in (
        ("COMPARISON.csv", rollup), ("PAIRED_LOAD_METRICS.csv", metrics),
        ("A121_BLOCK_AUROC.csv", blocks), ("CYCLE_SCORES.csv", scores),
    ):
        common.savecsv(output / filename, table)
    common.savej(output / "VERIFICATION.json", dict(
        candidates=candidate_specs, archived_references_reproduced=True,
        fit_counts=fit.label.value_counts().to_dict(), calibration_n=60, normal_holdout_n=60,
        previous_development_evaluation_n=180, exact_existing_train_cal_evaluation_split=True,
        calibration_rule="q95(method='higher') on N03, strict score > threshold",
        original_models_and_thresholds_unchanged=inputs.unchanged(), input_sha256=inputs.hashes,
    ))
    if not inputs.unchanged():
        raise ValueError("A protected input changed while saving")
    print(rollup.loc[~rollup.cohort.eq("new_calibration"),
                    ["cohort", "model_id", "label", "alarms", "n", "threshold"]].to_string(index=False))
    print(metrics[["model_id", "label", "paired_auroc", "paired_normal_fp"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.output.resolve())
