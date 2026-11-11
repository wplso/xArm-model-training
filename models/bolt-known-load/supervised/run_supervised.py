"""Compare frozen V4 Q and two supervised classifiers on held-out acquisition campaigns."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


TARGETS = ("A121", "B82", "C182")
V4 = "V4_PCA_Q"
OLD_HISTORY = "old_nov04_to_nov05_dawn_continuous"
OLD05_PM = "old_nov05_evening_pairs"
OLD_CAL = "old_nov05_late_normal30"
OLD_COHORTS = {
    ("historical", "previous20"): (OLD_HISTORY, "normal", "B82", 20, "No3_41g_v21_independent_20261104_pm_retry2"),
    ("historical", "B"): (OLD_HISTORY, "B82", "B82", 30, "No3_fixed_base_load90_20261104"),
    ("historical", "normal_a"): (OLD_HISTORY, "normal", "A121", 10, "No3_AC_dose_20261105"),
    ("historical", "normal_c"): (OLD_HISTORY, "normal", "C182", 10, "No3_AC_dose_20261105"),
    ("historical", "A121"): (OLD_HISTORY, "A121", "A121", 20, "No3_AC_dose_20261105"),
    ("historical", "C182"): (OLD_HISTORY, "C182", "C182", 20, "No3_AC_dose_20261105"),
    ("latest30", "latest30"): (OLD_CAL, "normal", "", 30, "No3_normal30_FPR_20261105"),
    **{
        ("recent", condition): (OLD05_PM, label, target, n, "No3_six_anomaly_pairs_20261105")
        for letter, target in zip("ABC", TARGETS, strict=True)
        for condition, label, n in [(letter, target, 20), (f"N_{letter}", "normal", 10)]
    },
}
EVAL_GROUPS = ("G1_nov06_related", "G2_nov10_first40", "G3_nov10_restart40")
CLASSIFIER_TYPES = {"LOGISTIC": LogisticRegression, "RANDOM_FOREST": RandomForestClassifier}


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def indexed(frame):
    frame = frame.set_index("cycle_id")
    if not frame.index.is_unique:
        raise ValueError("Duplicate cycle IDs")
    return frame


def load_inputs(project):
    root = project / "hardware/xarm_model_versions"
    paths = {
        "bolt_features": root / "v5_1_multisession_normal_study/metrics/CYCLE_FEATURES.csv",
        "bolt_audit": root / "v5_1_multisession_normal_study/verification/DATA_VERIFICATION.json",
        "old_features": root / "v4_2_feature_ablation_prebolt/metrics/CYCLE_FEATURES.csv",
        "old_audit": root / "v4_bolt_fixed/cross_mount_previous_20261106/VERIFICATION.json",
        "frozen": root / "v4_bolt_fixed/FROZEN_V4_CANDIDATES.json",
        "v4_model": root / "v4_bolt_fixed/models/V4_PCA.joblib",
        "v4_code": root / "v4_bolt_fixed/analysis_code/normal_models.py",
        "v4_split": root / "v4_bolt_fixed/SPLIT_MANIFEST.csv",
        "old_scores": root / "v4_bolt_fixed/cross_mount_previous_20261106/metrics/CYCLE_SCORES.csv",
        "bolt_scores": root / "normal_only_algorithm_benchmark/metrics/CYCLE_SCORES.csv",
        "bolt_score_audit": root / "normal_only_algorithm_benchmark/metrics/VERIFICATION.json",
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    freeze = read_json(paths["frozen"])
    v4 = next(spec for spec in freeze["models"] if spec["model_id"] == "V4_PCA")
    assert len(v4["features"]) == len(set(v4["features"])) == 24
    assert hashes["v4_model"] == v4["sha256"] and hashes["v4_split"] == freeze["split_manifest_sha256"]
    assert hashes["bolt_features"] == read_json(paths["bolt_audit"])["features_sha256"]
    old_hashes = read_json(paths["old_audit"])["input_sha256"]
    assert hashes["old_features"] == next(value for path, value in old_hashes.items() if path.endswith("v4_2_feature_ablation_prebolt\\metrics\\CYCLE_FEATURES.csv"))
    audit = read_json(paths["bolt_score_audit"])
    assert audit["verified"] and hashes["bolt_scores"] == audit["cycle_scores_sha256"]
    old = indexed(read_csv(paths["old_features"]))
    bolt = indexed(read_csv(paths["bolt_features"]))
    assert len(old) == 1038 and len(bolt) == 649
    assert set(old.index).isdisjoint(bolt.index) and set(old.session_id).isdisjoint(bolt.session_id)
    old_scores = indexed(read_csv(paths["old_scores"]).query("model_id == 'V4_PCA'"))
    bolt_scores = indexed(read_csv(paths["bolt_scores"]).query("split_id == 'PRIMARY' and model_id == 'V4_PCA_Q'"))
    scores = pd.concat([old_scores, bolt_scores], verify_integrity=True)
    assert scores.threshold.eq(v4["threshold"]).all()
    np.testing.assert_array_equal(scores.flagged, scores.score.gt(v4["threshold"]))
    sys.path.insert(0, str(paths["v4_code"].parent))
    try:
        reference = joblib.load(paths["v4_model"])
    finally:
        sys.path.pop(0)
    assert list(reference["features"]) == v4["features"] and reference["threshold"] == v4["threshold"]
    return old, bolt, scores, v4, reference, indexed(read_csv(paths["v4_split"])), paths, hashes


def input_frames(old, bolt, features):
    readings = ["session_id", "source", "No3_begin_ns", "No3_end_ns", "start_pose", *features]
    metadata = ["cohort", "campaign", "label", "pair", "mount"]
    old_parts = []
    for (dataset, cohort), (group, label, pair, n, folder) in OLD_COHORTS.items():
        rows = old.loc[old.dataset.eq(dataset) & old.condition.eq(cohort)]
        assert len(rows) == n and rows.source.str.contains(folder, regex=False).all()
        assert rows.automatic_return.eq(True).all()
        assert rows.mount_condition.eq("monster_gel_prebolt_user_report").all()
        assert rows.normal.eq(label == "normal").all()
        mass_field = "payload_mass_g" if label == "A121" else "frame_added_mass_g"
        if label in TARGETS:
            mass = rows[mass_field].dropna()
            assert mass.eq(int(label[1:])).all()
            if dataset == "recent":
                assert rows.source.str.contains(label, regex=False).all()
        old_parts.append(rows[readings].copy().assign(
            campaign=group, label=label, pair=pair, mount="monster_gel", cohort=cohort,
        ))
    old_use = pd.concat(old_parts, verify_integrity=True)
    codes = dict(zip("ABC", TARGETS, strict=True))
    assert bolt.quality_valid.all() and bolt.automatic_return.eq(True).all()
    assert bolt.sensor_mount.eq("bolt_and_nut_user_reported").all()
    bolt_use = bolt[[*readings, "cohort"]].copy().assign(
        campaign=bolt.leakage_group_id, label=bolt.condition.replace(codes),
        pair=bolt.pair_condition.replace(codes).fillna(""), mount="bolt",
    )
    columns = [*readings, *metadata]
    return old_use[columns], bolt_use[columns]


def evaluation_splits(old, bolt, bolt_roles):
    cal = bolt.loc[bolt_roles.index[bolt_roles.role.eq("validation")]]
    evaluation = bolt[bolt.campaign.isin(EVAL_GROUPS)]
    assert set(evaluation.campaign).isdisjoint(bolt.loc[bolt_roles.index].campaign)
    assert len(cal) == 100
    return {
        "OLD_TO_BOLT": (old[old.campaign.isin([OLD_HISTORY, OLD05_PM])], cal, evaluation),
        "OLD_CAMPAIGN_HOLDOUT": (
            old[old.campaign.eq(OLD_HISTORY)],
            old[old.campaign.eq(OLD_CAL)], old[old.campaign.eq(OLD05_PM)],
        ),
    }


def verify_roles(split, roles, old, bolt, features):
    train, cal, evaluation = roles
    counts = {
        "OLD_TO_BOLT": ({"normal": 70, "A121": 40, "B82": 50, "C182": 40}, 100,
                        {"normal": 160, "A121": 30, "B82": 30, "C182": 30}),
        "OLD_CAMPAIGN_HOLDOUT": ({"normal": 40, "A121": 20, "B82": 30, "C182": 20}, 30,
                                  {"normal": 30, "A121": 20, "B82": 20, "C182": 20}),
    }
    assert (train.label.value_counts().to_dict(), len(cal), evaluation.label.value_counts().to_dict()) == counts[split]
    assert cal.label.eq("normal").all()
    used = pd.concat(roles, verify_integrity=True)
    assert not used.duplicated(["source", "No3_begin_ns", "No3_end_ns"]).any()
    role_tags = np.repeat(["train", "calibration", "evaluation"], [len(part) for part in roles])
    used = used.assign(split_id=split, role=role_tags)
    assert used.groupby("session_id").role.nunique().eq(1).all()
    assert used.groupby("campaign").role.nunique().eq(1).all()
    # Every selected capture is whole relative to the earlier audited eligibility rules.
    source = pd.concat([old[["session_id"]], bolt[["session_id"]]])
    selected_sessions = source[source.session_id.isin(used.session_id)]
    assert set(selected_sessions.index) == set(used.index)
    assert not np.isinf(used[features].to_numpy(float)).any()
    assert not train[features].isna().all().any()
    return used.drop(columns=features)


def p_scores(pipeline, x, ordered_features):
    if list(x.columns) != ordered_features:
        raise ValueError("Feature order does not match frozen V4 feature schema")
    probs = pd.DataFrame(pipeline.predict_proba(x), columns=pipeline.classes_, index=x.index)
    assert np.isfinite(probs).all().all()
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-12, rtol=0)
    return probs


def rows_with_scores(data, model_id, scores, cutoff, probs=None):
    columns = ["split_id", "role", "campaign", "session_id", "cohort", "label", "pair", "mount", "start_pose"]
    frame = data[columns].assign(model_id=model_id, score=scores, threshold=cutoff)
    frame = frame.assign(margin=frame.score-cutoff, alarm=frame.score.gt(cutoff))
    if probs is not None:
        for label in ("normal", *TARGETS):
            frame[f"p_{label}"] = probs[label]
        frame["prediction"] = probs[list(TARGETS)].idxmax(axis=1).where(frame.alarm, "normal")
    return frame


def summarize(scored):
    totals, per_session, pairs, confusion = [], [], [], []
    evaluation = scored[scored.role.eq("evaluation")]
    for (split, model), rows in evaluation.groupby(["split_id", "model_id"], sort=False):
        keys = dict(split_id=split, model_id=model)
        normal = rows[rows.label.eq("normal")]
        total = dict(**keys, normal_n=len(normal), fp=int(normal.alarm.sum()), fpr=normal.alarm.mean())
        for (group, session, cohort), batch in normal.groupby(["campaign", "session_id", "cohort"], sort=False):
            per_session.append(dict(**keys, campaign=group, session_id=session, cohort=cohort,
                                    n=len(batch), fp=int(batch.alarm.sum()), fpr=batch.alarm.mean()))
        for scope, batch in [("all", normal), *normal.groupby("campaign", sort=False)]:
            per_session.append(dict(**keys, campaign=scope, session_id="", cohort="all_normal",
                                    n=len(batch), fp=int(batch.alarm.sum()), fpr=batch.alarm.mean()))
        for target in TARGETS:
            positive = rows[rows.label.eq(target)]
            control = normal[normal.pair.eq(target)]
            pair = pd.concat([positive, control])
            y = pair.label.eq(target)
            typed = model != V4
            pair_result = dict(**keys, target=target, anomaly_n=len(positive), tp=int(positive.alarm.sum()),
                               tpr=positive.alarm.mean(), paired_normal_n=len(control),
                               paired_fp=int(control.alarm.sum()), paired_fpr=control.alarm.mean(),
                               alarm_auroc=roc_auc_score(y, pair.score), alarm_ap=average_precision_score(y, pair.score),
                               typed_tp=int(positive.prediction.eq(target).sum()) if typed else None,
                               typed_vs_normal_auroc=roc_auc_score(y, pair[f"p_{target}"]) if typed else None)
            pairs.append(pair_result)
            total.update({f"{target}_{key}": pair_result[key] for key in
                          ("anomaly_n", "tp", "tpr", "paired_normal_n", "paired_fp", "alarm_auroc", "typed_tp")})
        totals.append(total)
        if model != V4:
            for actual in ("normal", *TARGETS):
                for predicted in ("normal", *TARGETS):
                    count = int((rows.label.eq(actual) & rows.prediction.eq(predicted)).sum())
                    confusion.append(dict(**keys, actual=actual, predicted=predicted, n=count))
    return {
        "COMPARISON": totals, "NORMAL_SESSION_FPR": per_session, "LOAD_METRICS": pairs, "TYPE_CONFUSION": confusion,
    }


def run(project, output):
    protocol_path = Path(__file__).with_name("PROTOCOL.json")
    protocol = read_json(protocol_path)
    old_source, bolt_source, baseline, v4, reference, bolt_roles, paths, hashes = load_inputs(project)
    features = v4["features"]
    old, bolt = input_frames(old_source, bolt_source, features)
    splits = evaluation_splits(old, bolt, bolt_roles)
    manifest = {split: verify_roles(split, parts, old_source, bolt_source, features) for split, parts in splits.items()}
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    inputs = {str(path.relative_to(project)): hashes[key] for key, path in paths.items()}
    write_json(output / "RUN_CONFIG.json", dict(protocol=protocol, protocol_sha256=digest(protocol_path),
               code_sha256=digest(Path(__file__)), features=features, source_sha256=inputs,
               sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__))
    pd.concat(manifest.values()).reset_index().to_csv(output / "SPLIT_MANIFEST.csv", index=False)
    scored, model_info = [], []
    max_reference_error = 0.0
    for split, (train, calibration, evaluation) in splits.items():
        data = manifest[split]
        eval_meta = data.loc[evaluation.index]
        eval_x = evaluation[features]
        old_score = baseline.loc[evaluation.index, "score"]
        replay = reference["detector"].score(eval_x.to_numpy(float))
        np.testing.assert_allclose(replay, old_score, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(replay > v4["threshold"], old_score.gt(v4["threshold"]))
        max_reference_error = max(max_reference_error, float(np.max(np.abs(replay-old_score))))
        scored.append(rows_with_scores(eval_meta, V4, old_score, v4["threshold"]))
        for model_id, settings in protocol["models"].items():
            pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                     CLASSIFIER_TYPES[model_id](**settings))
            with warnings.catch_warnings(record=True) as issued, threadpool_limits(limits=1):
                warnings.simplefilter("always")
                pipeline.fit(train[features], train.label)
                if issued:
                    raise RuntimeError(f"{split}/{model_id} fit warning: {issued[0].message}")
                assert set(pipeline.classes_) == {"normal", *TARGETS}
                cal_probs = p_scores(pipeline, calibration[features], features)
                cal_score = 1-cal_probs.normal
                cutoff = float(np.quantile(cal_score, .95, method="higher"))
                eval_probs = p_scores(pipeline, eval_x, features)
                path = output / "models" / f"{split}_{model_id}.joblib"
                joblib.dump(dict(model=pipeline, features=features, threshold=cutoff, split_id=split,
                                 classes=list(pipeline.classes_), model_id=model_id,
                                 train_ids=train.index.tolist(), calibration_ids=calibration.index.tolist()), path)
                restored = joblib.load(path)
                np.testing.assert_array_equal(p_scores(restored["model"], eval_x, features), eval_probs)
                np.testing.assert_allclose(p_scores(pipeline, eval_x.iloc[::-1], features).iloc[::-1],
                                           eval_probs, atol=1e-12, rtol=0)
                assert restored["threshold"] == cutoff
            scored.extend([
                rows_with_scores(data.loc[calibration.index], model_id, cal_score, cutoff, cal_probs),
                rows_with_scores(eval_meta, model_id, 1-eval_probs.normal, cutoff, eval_probs),
            ])
            model_info.append(dict(split_id=split, model_id=model_id, train_n=len(train), calibration_n=len(calibration),
                                   calibration_fp=int(cal_score.gt(cutoff).sum()), threshold=cutoff,
                                   path=str(path.relative_to(output)), sha256=digest(path)))
    rows = pd.concat(scored).reset_index()
    assert not rows.duplicated(["split_id", "model_id", "cycle_id"]).any()
    assert rows.score.notna().all() and rows.alarm.eq(rows.margin.gt(0)).all()
    rows.to_csv(output / "CYCLE_SCORES.csv", index=False)
    summary = summarize(rows)
    for name, records in summary.items():
        pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(model_info).to_csv(output / "MODEL_INDEX.csv", index=False)
    assert {name: digest(path) for name, path in paths.items()} == hashes
    write_json(output / "VERIFICATION.json", dict(
        verified=True, no_cycle_session_or_campaign_crosses_roles_within_split=True,
        whole_eligible_captures=True, v4_eval_disjoint_from_v4_fit_and_cal_campaign=True,
        frozen_v4_replay_max_abs_error=max_reference_error, frozen_v4_alarm_replay_identical=True,
        input_hashes_unchanged=True, probabilities_reload_exact=True,
        evaluation_scores=len(rows[rows.role.eq("evaluation")]), capture_roles=len(pd.concat(manifest.values())),
        missing_values_train_only_imputation={split: [int(part[features].isna().sum().sum()) for part in parts]
                                             for split, parts in splits.items()},
    ))
    print(pd.DataFrame(summary["COMPARISON"]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    run(args.project, args.output)
