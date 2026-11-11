"""Evaluate two fixed binary objectives against archived V4 and multinomial logistic scores."""

import argparse
import importlib.util
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits


V4 = "V4_PCA_Q"
MULTI = "LOGISTIC_4CLASS"
MODELS = ("BINARY_BALANCED", "BINARY_KEEP_4CLASS_WEIGHTS")
TARGETS = ("A121", "B82", "C182")


def load_study(root):
    spec = importlib.util.spec_from_file_location("supervised_source", root / "run_supervised.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_binary(frozen, train, calibration, evaluation, mode, output, split):
    features = frozen["features"]
    stored_pipeline = frozen["model"]
    original = stored_pipeline.steps[-1][1]
    assert original.class_weight == "balanced"
    preprocessing = stored_pipeline[:-1]
    z_train = preprocessing.transform(train[features])
    y = train.label.ne("normal").to_numpy()
    weight_target = y if mode == "BINARY_BALANCED" else train.label.to_numpy()
    weights = compute_sample_weight("balanced", weight_target)
    classifier = clone(original).set_params(class_weight=None)
    with warnings.catch_warnings(record=True) as issued, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        classifier.fit(z_train, y, sample_weight=weights)
        if issued:
            raise RuntimeError(f"{split}/{mode}: {issued[0].message}")
        assert np.array_equal(classifier.classes_, [False, True])
        model = Pipeline([*stored_pipeline.steps[:-1], ("logisticregression", classifier)])
        calibration_score = model.predict_proba(calibration[features])[:, 1]
        evaluation_score = model.predict_proba(evaluation[features])[:, 1]
        assert np.isfinite(calibration_score).all() and np.isfinite(evaluation_score).all()
        threshold = float(np.quantile(calibration_score, .95, method="higher"))
        file = output / "models" / f"{split}_{mode}.joblib"
        joblib.dump({
            "model": model, "threshold": threshold, "features": features,
            "train_ids": train.index.tolist(), "calibration_ids": calibration.index.tolist(),
            "split_id": split, "model_id": mode, "positive_label": "known_load",
            "weight_by_original_label": {
                label: float(np.sum(weights[train.label.eq(label)]))
                for label in ("normal", *TARGETS)
            },
        }, file)
        saved = joblib.load(file)
        np.testing.assert_array_equal(saved["model"].predict_proba(evaluation[features])[:, 1], evaluation_score)
        reversed_scores = model.predict_proba(evaluation[features].iloc[::-1])[:, 1][::-1]
        np.testing.assert_allclose(reversed_scores, evaluation_score, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(reversed_scores > threshold, evaluation_score > threshold)
    return calibration_score, evaluation_score, saved, file


def score_frame(metadata, model_id, scores, threshold):
    return metadata.assign(model_id=model_id, score=scores, threshold=threshold,
                           alarm=np.asarray(scores) > threshold,
                           margin=np.asarray(scores)-threshold)


def metrics(scores):
    evaluation = scores[scores.role.eq("evaluation")]
    total, normal_scopes, load_rows, transitions = [], [], [], []
    for (split, model), rows in evaluation.groupby(["split_id", "model_id"], sort=False):
        key = dict(split_id=split, model_id=model)
        normal = rows[rows.label.eq("normal")]
        matched = normal[normal.pair.isin(TARGETS)]
        combined = dict(**key, normal_n=len(normal), fp=int(normal.alarm.sum()), fpr=normal.alarm.mean(),
                        matched_normal_n=len(matched), matched_fp=int(matched.alarm.sum()),
                        matched_fpr=matched.alarm.mean())
        for name, batch in [("all", normal), ("matched", matched), *normal.groupby("campaign", sort=False)]:
            normal_scopes.append(dict(**key, scope=name, session_id="", cohort="all",
                                      fp=int(batch.alarm.sum()), n=len(batch), fpr=batch.alarm.mean()))
        for (campaign, session, cohort), batch in normal.groupby(["campaign", "session_id", "cohort"], sort=False):
            normal_scopes.append(dict(**key, scope=campaign, session_id=session, cohort=cohort,
                                      fp=int(batch.alarm.sum()), n=len(batch), fpr=batch.alarm.mean()))
        for target in TARGETS:
            positive = rows[rows.label.eq(target)]
            negative = matched[matched.pair.eq(target)]
            pair = pd.concat([positive, negative])
            labels = pair.label.eq(target)
            result = dict(**key, target=target, tp=int(positive.alarm.sum()), anomaly_n=len(positive),
                          paired_fp=int(negative.alarm.sum()), paired_normal_n=len(negative),
                          tpr=positive.alarm.mean(), paired_fpr=negative.alarm.mean(),
                          auroc=roc_auc_score(labels, pair.score), ap=average_precision_score(labels, pair.score),
                          tpr_minus_paired_fpr=positive.alarm.mean()-negative.alarm.mean())
            load_rows.append(result)
            combined.update({f"{target}_{k}": result[k] for k in ("tp", "anomaly_n", "auroc", "paired_fp", "paired_normal_n")})
        total.append(combined)
        if model in MODELS:
            baseline = evaluation.loc[evaluation.split_id.eq(split) & evaluation.model_id.eq(MULTI)]
            baseline = baseline.set_index("cycle_id").loc[rows.cycle_id, "alarm"].to_numpy()
            change = rows[["cycle_id", "split_id", "model_id", "campaign", "cohort", "label", "pair", "alarm"]].copy()
            change["multiclass_alarm"] = baseline
            change["transition"] = np.select(
                [baseline & change.alarm, baseline & ~change.alarm, ~baseline & change.alarm],
                ["both_alarm", "multiclass_only", "binary_only"], default="neither",
            )
            transitions.append(change)
    return {
        "COMPARISON": pd.DataFrame(total), "NORMAL_SESSION_FPR": pd.DataFrame(normal_scopes),
        "LOAD_METRICS": pd.DataFrame(load_rows), "ALARM_TRANSITIONS": pd.concat(transitions),
    }


def run(project, output):
    study = project / "hardware/xarm_model_versions/supervised_load_session_holdout_20261111"
    prior = load_study(study)
    digest, read_csv = prior.digest, prior.read_csv
    source_output = study / "results"
    previous = prior.read_json(source_output / "RUN_CONFIG.json")
    assert digest(study / "run_supervised.py") == previous["code_sha256"]
    assert digest(study / "PROTOCOL.json") == previous["protocol_sha256"]
    protocol_file = Path(__file__).with_name("PROTOCOL.json")
    protocol = prior.read_json(protocol_file)
    old_src, bolt_src, archived_v4, freeze, _, bolt_roles, original_paths, original_hashes = prior.load_inputs(project)
    features = freeze["features"]
    old, bolt = prior.input_frames(old_src, bolt_src, features)
    splits = prior.evaluation_splits(old, bolt, bolt_roles)
    assert list(splits) == protocol["splits"]
    prior_manifest = read_csv(source_output / "SPLIT_MANIFEST.csv")
    archived = read_csv(source_output / "CYCLE_SCORES.csv")
    index = read_csv(source_output / "MODEL_INDEX.csv").query("model_id == 'LOGISTIC'").set_index("split_id")
    protected = [study / "run_supervised.py", study / "PROTOCOL.json",
                 source_output / "RUN_CONFIG.json", source_output / "SPLIT_MANIFEST.csv",
                 source_output / "CYCLE_SCORES.csv", source_output / "MODEL_INDEX.csv",
                 *(source_output / path for path in index.path)]
    protected_hashes = {path: digest(path) for path in protected}
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    prior.write_json(output / "RUN_CONFIG.json", {
        "protocol": protocol, "features": features, "libraries": previous["sklearn"],
        "protocol_sha256": digest(protocol_file), "code_sha256": digest(Path(__file__)),
        "protected_source_hashes": {str(path.relative_to(project)): value for path, value in protected_hashes.items()},
    })
    collected, info = [], []
    for split, (train, calibration, evaluation) in splits.items():
        manifest = prior.verify_roles(split, (train, calibration, evaluation), old_src, bolt_src, features)
        original_roles = prior_manifest[prior_manifest.split_id.eq(split)].set_index("cycle_id")
        columns = ["role", "session_id", "campaign", "label", "pair", "cohort"]
        assert set(manifest.index) == set(original_roles.index)
        pd.testing.assert_frame_equal(manifest[columns], original_roles.loc[manifest.index, columns].fillna(""))
        frozen_path = source_output / index.loc[split, "path"]
        assert protected_hashes[frozen_path] == index.loc[split, "sha256"]
        frozen = joblib.load(frozen_path)
        assert frozen["features"] == features and frozen["threshold"] == index.loc[split, "threshold"]
        assert frozen["train_ids"] == train.index.tolist() and frozen["calibration_ids"] == calibration.index.tolist()
        preprocess_hash = joblib.hash(frozen["model"][:-1])
        held_out = archived[archived.split_id.eq(split)]
        for before, name in ((V4, V4), ("LOGISTIC", MULTI)):
            rows = held_out[held_out.model_id.eq(before)].set_index("cycle_id")
            rows = rows.loc[evaluation.index] if before == V4 else rows.loc[calibration.index.append(evaluation.index)]
            if before == V4:
                np.testing.assert_array_equal(rows.score, archived_v4.loc[rows.index, "score"])
            else:
                x = pd.concat([calibration[features], evaluation[features]])
                with threadpool_limits(limits=1):
                    reproduced = 1-prior.p_scores(frozen["model"], x, features).normal
                np.testing.assert_allclose(reproduced, rows.score, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(reproduced.gt(frozen["threshold"]), rows.alarm)
            cols = ["split_id", "role", "campaign", "session_id", "cohort", "label", "pair", "mount", "start_pose"]
            rebuilt = score_frame(manifest.loc[rows.index, cols], name, rows.score, rows.threshold)
            np.testing.assert_array_equal(rebuilt.alarm, rows.alarm)
            collected.append(rebuilt)
        for model_id in MODELS:
            cal_score, eval_score, saved, path = fixed_binary(frozen, train, calibration, evaluation, model_id, output, split)
            cutoff = saved["threshold"]
            collected.extend([
                score_frame(manifest.loc[calibration.index, cols], model_id, cal_score, cutoff),
                score_frame(manifest.loc[evaluation.index, cols], model_id, eval_score, cutoff),
            ])
            info.append(dict(split_id=split, model_id=model_id, train_n=len(train), calibration_n=len(calibration),
                             calibration_fp=int(np.count_nonzero(cal_score > cutoff)), threshold=cutoff,
                             path=str(path.relative_to(output)), sha256=digest(path),
                             **{f"weight_total_{label}": value for label, value in saved["weight_by_original_label"].items()}))
        assert joblib.hash(frozen["model"][:-1]) == preprocess_hash
    scored = pd.concat(collected).reset_index()
    assert not scored.duplicated(["split_id", "model_id", "cycle_id"]).any()
    assert np.isfinite(scored.score).all() and scored.alarm.eq(scored.margin.gt(0)).all()
    scored.to_csv(output / "CYCLE_SCORES.csv", index=False)
    results = metrics(scored)
    for name, frame in results.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(info).to_csv(output / "MODEL_INDEX.csv", index=False)
    assert {path: digest(path) for path in protected} == protected_hashes
    assert {key: digest(path) for key, path in original_paths.items()} == original_hashes
    prior.write_json(output / "VERIFICATION.json", dict(
        old_model_and_data_hashes_preserved=True, reused_exact_split_and_preprocessing=True,
        archived_reference_scores_and_alarms_verified=True, binary_model_reload_exact=True,
        reverse_order_alarm_invariant=True, row_count=len(scored),
        evaluated_cycle_counts={split: len(parts[2]) for split, parts in splits.items()},
    ))
    print(results["COMPARISON"].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    run(args.project, args.output)
