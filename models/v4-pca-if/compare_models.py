"""Compare frozen V4 Q with two Isolation Forest representations on its original split."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score


BASELINE = "V4_PCA_Q"
VARIANTS = {"IF_24": "standardized_v4_features", "IF_PC13_Q": "frozen_v4_pc13_plus_q"}
FOREST_PARAMS = dict(n_estimators=200, random_state=20261031, n_jobs=1)
Q_BATCHES = [("train200", "validation100", "test99"), ("nov10_normal40",),
             ("extra_normal17", "extra_normal13"), ("preA",), ("A",), ("preB",), ("B",),
             ("preC",), ("C",), ("nov10_restart40",)]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def unique_cycles(path):
    result = read_csv(path).set_index("cycle_id")
    if not result.index.is_unique:
        raise ValueError(f"Duplicate cycle ID: {path}")
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def q_cutoff(values):
    return float(np.quantile(values, .95, method="higher"))


def represent(matrix, imputer, scaler, pca=None):
    scaled = scaler.transform(imputer.transform(matrix))
    if pca is None:
        return scaled
    coordinates = pca.transform(scaled)
    q = np.mean((scaled - pca.inverse_transform(coordinates)) ** 2, axis=1)
    return np.column_stack((coordinates, q))


def load_experiment(root):
    versions = root / "hardware/xarm_model_versions"
    v4 = versions / "v4_bolt_fixed"
    v51 = versions / "v5_1_multisession_normal_study"
    paths = dict(split=v4 / "SPLIT_MANIFEST.csv", freeze=v4 / "FROZEN_V4_CANDIDATES.json",
                 feature_definitions=v4 / "FEATURE_DEFINITIONS.json", model=v4 / "models/V4_PCA.joblib",
                 model_code=v4 / "analysis_code/normal_models.py", manifest=v51 / "DATA_USAGE_MANIFEST.csv",
                 features=v51 / "metrics/CYCLE_FEATURES.csv", prior_scores=v51 / "metrics/BASELINE_CYCLE_SCORES.csv",
                 audit=v51 / "verification/DATA_VERIFICATION.json")
    hashes = {key: digest(path) for key, path in paths.items()}
    frozen = read_json(paths["freeze"])
    audit = read_json(paths["audit"])
    spec = next(item for item in frozen["models"] if item["model_id"] == "V4_PCA")
    require(hashes["model"] == spec["sha256"] and hashes["split"] == frozen["split_manifest_sha256"]
            and hashes["feature_definitions"] == frozen["feature_definitions_sha256"], "Frozen V4 hash mismatch")
    require(hashes["features"] == audit["features_sha256"] and hashes["manifest"] == audit["data_sha256"], "V5.1 source hash mismatch")
    usage, features, split = (unique_cycles(paths[k]) for k in ("manifest", "features", "split"))
    require(len(usage) == len(features) == 649 and set(usage.index) == set(features.index), "Unexpected dataset")
    features = features.loc[usage.index]
    for column in ("role", "cohort", "condition", "pair_condition", "sensor_mount", "session_id"):
        np.testing.assert_array_equal(usage[column].fillna(""), features[column].fillna(""))
    require(usage.quality_valid.eq(True).all() and usage.sensor_mount.eq("bolt_and_nut_user_reported").all(), "Mixed or unreviewed data")
    for role, size in (("train", 200), ("validation", 100), ("test", 99)):
        current = usage.index[usage.role.eq(role)]
        original = split.index[split.role.eq(role)]
        require(len(current) == size and current.equals(original), f"Original {role} membership or order changed")
        require(usage.loc[current].condition.eq("normal").all(), f"Non-normal row in {role}")
    require(usage.role.isin(["train", "validation", "test", "evaluation_only"]).all(), "Unknown role")
    require(usage.groupby("session_id").role.nunique().max() == 1, "Capture crosses data roles")
    evaluated = usage.loc[usage.role.isin(["test", "evaluation_only"])]
    require(evaluated.condition.value_counts().to_dict() == {"normal": 259, "A": 30, "B": 30, "C": 30}, "Wrong evaluation denominator")
    require(evaluated.loc[evaluated.condition.eq("normal")].pair_condition.value_counts().to_dict() == {"A": 30, "B": 10, "C": 10},
            "Wrong paired normal denominator")
    sys.path.insert(0, str(paths["model_code"].parent))
    bundle = joblib.load(paths["model"])
    require(bundle["features"] == spec["features"] and len(spec["features"]) == 24, "V4 feature order differs")
    require(bundle["detector"].model.n_components_ == 13 and bundle["threshold"] == spec["threshold"], "Frozen V4 settings differ")
    return paths, hashes, usage, features[bundle["features"]], bundle


def compare_groups(evaluation):
    normal = evaluation.loc[evaluation.condition.eq("normal")]
    groups = {"all_normal": normal}
    groups.update({cohort: group for cohort, group in normal.groupby("cohort", sort=False)})
    restart = groups["nov10_restart40"]
    require(restart.session_cycle_index.tolist() == list(range(1, 41)), "Restart order differs")
    for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 40), (11, 40), (21, 40)):
        groups[f"restart_{lo:02}-{hi:02}"] = restart.loc[restart.session_cycle_index.between(lo, hi)]
    for label in "ABC":
        groups[f"anomaly_{label}"] = evaluation.loc[evaluation.condition.eq(label)]
    return groups


def summaries(usage, scored):
    groups = compare_groups(usage.loc[usage.role.isin(["test", "evaluation_only"])])
    normal = groups["all_normal"]
    baseline = scored[BASELINE]
    rows, transitions, ranking = [], [], []
    for model, current in scored.items():
        for name, group in groups.items():
            frame = current.loc[group.index]
            ratio = frame.score / frame.threshold
            rows.append(dict(model=model, group=name, label=group.condition.unique().item(), n=len(frame),
                             alarms=int(frame.flagged.sum()), rate=float(frame.flagged.mean()),
                             ratio_p10=ratio.quantile(.1), ratio_median=ratio.median(), ratio_p90=ratio.quantile(.9)))
            if model != BASELINE:
                old, new = baseline.loc[group.index].flagged, frame.flagged
                transitions.append(dict(model=model, group=name, n=len(frame), new_alarm=int((~old & new).sum()),
                                        lost_alarm=int((old & ~new).sum()), common_alarm=int((old & new).sum()),
                                        common_no_alarm=int((~old & ~new).sum())))
        for label in "ABC":
            ids = groups[f"anomaly_{label}"].index
            pair = normal.loc[normal.pair_condition.eq(label)].index
            binary = np.r_[np.ones(len(ids), dtype=int), np.zeros(len(pair), dtype=int)]
            values = current.loc[ids.append(pair), "score"]
            ranking.append(dict(model=model, condition=label, anomaly_n=len(ids), paired_normal_n=len(pair),
                                tp=int(current.loc[ids].flagged.sum()), paired_fp=int(current.loc[pair].flagged.sum()),
                                auroc=roc_auc_score(binary, values), ap=average_precision_score(binary, values)))
    return pd.DataFrame(rows), pd.DataFrame(transitions), pd.DataFrame(ranking)


def run(root, output):
    paths, hashes, usage, features, bundle = load_experiment(root)
    output.mkdir(parents=True, exist_ok=False)
    models = output / "models"
    models.mkdir()
    config = dict(input_sha256=hashes, input_paths={k: str(p.relative_to(root)) for k, p in paths.items()},
                  code_sha256=digest(Path(__file__)), features=bundle["features"], variant_representations=VARIANTS,
                  frozen_pca_components=13, forest_parameters=IsolationForest(**FOREST_PARAMS).get_params(),
                  calibration_rule="original normal 100, 95th percentile higher, strict score > threshold",
                  evaluation_scope="original V4 roles and all V5.1 reviewed evaluation rows, no preparation removal",
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  sklearn=sklearn.__version__, joblib=joblib.__version__)
    write_json(output / "RUN_CONFIG.json", config)
    detector = bundle["detector"]
    matrix = features.to_numpy(float)
    train = usage.role.eq("train")
    calibration = usage.role.eq("validation")
    scaled = represent(matrix, detector.imputer, detector.scaler)
    latent = represent(matrix, detector.imputer, detector.scaler, detector.model)
    q = pd.Series(index=usage.index, dtype=float)
    for cohorts in Q_BATCHES:
        selected = usage.cohort.isin(cohorts)
        q.loc[selected] = detector.score(features.loc[selected].to_numpy(float))
    cutoff = float(bundle["threshold"])
    recorded = read_csv(paths["prior_scores"])
    recorded = recorded.loc[recorded.model_id.eq("V4_PCA")].set_index("cycle_id").loc[usage.index]
    np.testing.assert_allclose(q, recorded.score, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(q, latent[:, -1], atol=1e-13, rtol=1e-12)
    np.testing.assert_array_equal(q.gt(cutoff), recorded.flagged)
    require(q_cutoff(q.loc[calibration]) == cutoff and recorded.threshold.eq(cutoff).all(), "Original cutoff mismatch")
    scored = {BASELINE: pd.DataFrame(dict(score=q, threshold=cutoff, flagged=q.gt(cutoff)))}
    model_rows = [dict(model=BASELINE, dimensions=24, threshold=cutoff, fit_n=200, cal_n=100,
                       calibration_fp=int(q.loc[calibration].gt(cutoff).sum()), path=str(paths["model"].relative_to(root)), sha256=hashes["model"])]
    for name, transformed in zip(VARIANTS, (scaled, latent)):
        forest = IsolationForest(**FOREST_PARAMS).fit(transformed[train])
        values = -forest.score_samples(transformed)
        limit = q_cutoff(values[calibration])
        artifact = models / f"{name}.joblib"
        joblib.dump(dict(model_id=name, estimator=forest, features=bundle["features"], imputer=detector.imputer,
                         scaler=detector.scaler, pca=detector.model if name == "IF_PC13_Q" else None, threshold=limit,
                         train_cycle_ids=usage.index[train].tolist(), calibration_cycle_ids=usage.index[calibration].tolist(),
                         source_v4_model_sha256=hashes["model"], run_config_sha256=digest(output / "RUN_CONFIG.json")), artifact)
        reloaded = joblib.load(artifact)
        replay = represent(matrix, reloaded["imputer"], reloaded["scaler"], reloaded["pca"])
        np.testing.assert_array_equal(values, -reloaded["estimator"].score_samples(replay))
        require(limit == reloaded["threshold"], "Saved IF cutoff differs")
        scored[name] = pd.DataFrame(dict(score=values, threshold=limit, flagged=values > limit), index=usage.index)
        model_rows.append(dict(model=name, dimensions=transformed.shape[1], threshold=limit, fit_n=200, cal_n=100,
                               calibration_fp=int((values[calibration] > limit).sum()),
                               path=artifact.relative_to(output).as_posix(), sha256=digest(artifact)))
    results, transitions, ranking = summaries(usage, scored)
    metadata = usage[["role", "session_id", "session_cycle_index", "cohort", "condition", "pair_condition", "leakage_group_id"]]
    cycles = pd.concat([metadata.join(frame).assign(model=model) for model, frame in scored.items()])
    for name, table in (("MODEL_INDEX", pd.DataFrame(model_rows)), ("EVALUATION", results),
                        ("PAIRED_TRANSITIONS", transitions), ("PAIRED_CONDITION_RANKING", ranking), ("CYCLE_SCORES", cycles.reset_index())):
        table.to_csv(output / f"{name}.csv", encoding="utf-8-sig", index=False)
    require(all(digest(path) == hashes[key] for key, path in paths.items()), "Original input changed during comparison")
    write_json(output / "VERIFICATION.json", dict(verified=True, original_files_unchanged=True,
               frozen_v4_score_max_error=float(np.max(np.abs(q - recorded.score))), frozen_v4_flags_identical=True,
               reloaded_if_scores_exact=True, split_counts=usage.role.value_counts().to_dict(), score_rows=len(cycles)))
    print(results.loc[results.group.isin(["all_normal", "nov10_restart40", "restart_11-40", "anomaly_A", "anomaly_B", "anomaly_C"]),
                      ["model", "group", "n", "alarms", "rate"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    run(args.project_root.resolve(strict=True), args.output.resolve())
