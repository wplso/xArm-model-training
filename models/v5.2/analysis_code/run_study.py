"""Replay fixed V4 Q and refit the three predeclared V5.2 preparation variants."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from summarize_study import build_summaries


STUDY = Path(__file__).resolve().parents[1]
POLICY = "PREP50_SHORT10_KEEP_PRENORMAL10_V1"
SEED = 20261031
MODELS = {"B0": ("V4_PCA", 24), "B1": ("V4_1_PCA", 25), "B2": ("V4_2A_PCA", 24)}
MAIN = "nov06_main400"
EXCEPTIONS = {"nov06_preB", "nov06_preC"}
SHORT = {"nov06_extra_normal30", "nov06_preA", "nov06_A", "nov06_B", "nov06_C", "nov10_normal40", "nov10_restart40"}
V4_BATCHES = [("train200", "validation100", "test99"), ("nov10_normal40",), ("extra_normal17", "extra_normal13"),
              ("preA",), ("A",), ("preB",), ("B",), ("preC",), ("C",), ("nov10_restart40",)]
REF_SCOPES = {"training": ("train200",), "calibration": ("validation100",), "historical_test": ("test99",),
              "prior_normal40": ("nov10_normal40",), "restart_normal40": ("nov10_restart40",),
              "normal_abc_controls": ("preA", "preB", "preC"), "anomaly_abc": ("A", "B", "C"),
              "related_normal30": ("extra_normal17", "extra_normal13")}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def unique_index(frame, key):
    indexed = frame.set_index(key)
    if not indexed.index.is_unique:
        raise ValueError(f"Duplicate values in {key}")
    return indexed


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def load_frozen_mask(mask_path, hash_path, split_path):
    expected_hash = hash_path.read_text(encoding="utf-8-sig").split()[0]
    assert digest(mask_path) == expected_hash, "Preparation mask differs from its pre-score receipt"
    mask = unique_index(read_csv(mask_path), "cycle_id")
    split = unique_index(read_csv(split_path), "cycle_id")
    assert len(mask) == 650 and set(mask.index) == set(split.index)
    split = split.loc[mask.index]
    for name in ("original_role", "quality_eligible", "preparation_excluded", "post_preparation_eligible"):
        np.testing.assert_array_equal(mask[name], split[name])
    boolean_names = ("quality_eligible", "preparation_excluded", "post_preparation_eligible", "is_pre_anomaly_normal10")
    assert all(mask[name].dtype == bool for name in boolean_names)
    assert set(mask.condition) == {"normal", "A", "B", "C"}
    assert mask.mount_id.nunique() == 1
    logical = mask.logical_collection_id
    order = mask.original_cycle_order
    assert not mask.duplicated(["logical_collection_id", "original_cycle_order"]).any()
    assert set(logical) == {MAIN} | EXCEPTIONS | SHORT
    assert len(mask[logical.eq(MAIN)]) == 399
    assert set(order[logical.eq(MAIN)]) == set(range(1, 400))
    assert mask.loc[~mask.quality_eligible, "logical_collection_id"].tolist() == ["nov06_extra_normal30"]
    assert order[~mask.quality_eligible].tolist() == [18]
    extra = mask[logical.eq("nov06_extra_normal30")].sort_values("original_cycle_order")
    assert extra.original_cycle_order.tolist() == list(range(1, 32)) and extra.acquisition_session_id.nunique() == 2
    np.testing.assert_array_equal(mask.is_pre_anomaly_normal10, logical.isin(EXCEPTIONS))
    expected_preparation = (logical.eq(MAIN) & order.le(50)) | (logical.isin(SHORT) & order.le(10))
    np.testing.assert_array_equal(mask.preparation_excluded, expected_preparation)
    np.testing.assert_array_equal(mask.post_preparation_eligible, mask.quality_eligible & ~expected_preparation)
    eval_roles = mask.original_role.isin(["test", "evaluation_only"])
    flags = {
        "fit_train": mask.original_role.eq("train") & mask.post_preparation_eligible,
        "fit_calibration": mask.original_role.eq("validation") & mask.post_preparation_eligible,
        "eval_e0": eval_roles & mask.quality_eligible,
        "eval_e1": eval_roles & mask.post_preparation_eligible,
        "eval_e2": eval_roles & mask.post_preparation_eligible,
    }
    for key, val in flags.items():
        assert split[key].dtype == bool
        np.testing.assert_array_equal(val, split[key])
        mask[key] = val
    assert mask.quality_eligible.sum() == 649 and expected_preparation.sum() == 120
    assert mask[mask.fit_train | mask.fit_calibration].condition.eq("normal").all()
    assert not (mask.fit_train & mask.fit_calibration).any()
    assert not ((mask.fit_train | mask.fit_calibration) & mask.eval_e0).any()
    assert (mask.fit_train.sum(), mask.fit_calibration.sum(), mask.original_role.eq("test").sum()) == (150, 100, 99)
    for column, expected in (("eval_e0", (259, 30, 30, 30)), ("eval_e1", (219, 20, 20, 20))):
        subset = mask[mask[column]]
        assert tuple(subset.condition.value_counts().loc[["normal", "A", "B", "C"]]) == expected
        pair_n = (30 if column == "eval_e0" else 20, 10, 10)
        assert tuple(subset[subset.condition.eq("normal")].pair_condition.value_counts().loc[list("ABC")]) == pair_n
    return mask, expected_hash


def validate_inputs(mask, features, legacy_split):
    eligible = mask[mask.quality_eligible]
    assert len(features) == 649 and set(eligible.index) == set(features.index)
    aligned = features.loc[eligible.index]
    assert aligned.sensor_mount.eq("bolt_and_nut_user_reported").all()
    for mask_name, feature_name in (("original_role", "role"), ("condition", "condition"),
                                    ("pair_condition", "pair_condition"), ("cohort", "cohort"),
                                    ("leakage_group_id", "leakage_group_id")):
        np.testing.assert_array_equal(eligible[mask_name].fillna(""), aligned[feature_name].fillna(""))
    legacy_roles = unique_index(legacy_split, "cycle_id").role
    main = mask[mask.logical_collection_id.eq(MAIN)]
    assert set(main.index) == set(legacy_roles.index)
    np.testing.assert_array_equal(main.original_role, legacy_roles.loc[main.index])
    return {role: legacy_roles[legacy_roles.eq(role)].index.tolist() for role in ("train", "validation", "test")}


def q_values(detector, matrix):
    pca = detector.model
    assert detector.kind == "PCA" and pca.n_components == .95 and pca.svd_solver == "full" and not pca.whiten
    scaled = detector.scaler.transform(detector.imputer.transform(matrix))
    manual = np.mean((scaled - pca.inverse_transform(pca.transform(scaled))) ** 2, axis=1)
    scores = detector.score(matrix)
    np.testing.assert_array_equal(scores, manual)
    return scores


def replay_v4(bundle, features, historical, legacy_ids, prior_scopes):
    order = bundle["features"]
    threshold = float(bundle["threshold"])
    frames = []
    for cohorts in V4_BATCHES:
        frame = features[features.cohort.isin(cohorts)]
        frames.append(pd.Series(q_values(bundle["detector"], frame[order].to_numpy(float)), index=frame.index))
    actual = pd.concat(frames, verify_integrity=True).loc[features.index]
    recorded = historical.loc[actual.index]
    np.testing.assert_allclose(actual, recorded.score, rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(actual > threshold, recorded.flagged)
    np.testing.assert_array_equal(recorded.threshold, threshold)
    calibration = actual.loc[legacy_ids["validation"]].to_numpy(float)
    q95 = float(np.quantile(calibration, .95, method="higher"))
    assert q95 == threshold == np.sort(calibration)[int(np.ceil(.95 * (len(calibration) - 1)))]
    entries = []
    for scope, cohorts in {**REF_SCOPES, "all649": tuple(features.cohort.unique())}.items():
        selected = features.index[features.cohort.isin(cohorts)]
        a, r = actual.loc[selected], recorded.loc[selected]
        flags = int((a > threshold).sum())
        if scope != "all649":
            assert len(a) == int(prior_scopes.loc[scope, "n"]) and flags == int(prior_scopes.loc[scope, "flagged_n"])
        entries.append(dict(scope=scope, n=len(a), flagged_n=flags, max_abs_score_difference=float((a - r.score).abs().max()),
                            same_flags=True, stored_threshold=threshold, original_calibration_q95_higher=q95,
                            threshold_exact=True))
    return actual, entries


def validate_fit(detector, train):
    n = len(train)
    assert detector.model.n_samples_ == detector.scaler.n_samples_seen_ == n
    assert detector.imputer.strategy == "median" and detector.imputer.keep_empty_features
    np.testing.assert_allclose(detector.imputer.statistics_, np.median(train, axis=0))
    np.testing.assert_allclose(detector.scaler.mean_, np.mean(train, axis=0))
    np.testing.assert_allclose(detector.scaler.var_, np.var(train, axis=0))


def build_cycle_rows(mask, scores, cutoff, family, evaluation, model):
    frame = mask.reset_index().copy()
    primary = frame[f"eval_{evaluation.lower()}"].to_numpy(bool)
    score = scores.reindex(mask.index).to_numpy(float)
    assert np.isfinite(score).sum() == frame.quality_eligible.sum()
    flagged = pd.array(score > cutoff, dtype="boolean")
    flagged[~frame.quality_eligible] = pd.NA
    status = np.select([~frame.quality_eligible, primary, ~frame.original_role.isin(["test", "evaluation_only"])],
                       ["NOT_EVALUATED_QUALITY", "SCORED", "NOT_EVALUATION_ROLE"], default="NOT_EVALUATED_PREPARATION")
    frame = frame.assign(family=family, evaluation=evaluation, model_id=model, score=score, threshold=cutoff,
                         score_over_threshold=score / cutoff, flagged=flagged, model_alarm=flagged,
                         policy_status=status, eligible_for_primary_metric=primary)
    return frame


def verify_matched_rows(cycles, summaries, prior_abc):
    keys = ["family", "cycle_id"]
    e0, e1, e2 = (cycles[cycles.evaluation.eq(item)].set_index(keys) for item in ("E0", "E1", "E2"))
    assert len(e0) == len(e1) == len(e2) == 1950 and all(x.index.is_unique for x in (e0, e1, e2))
    pd.testing.assert_frame_equal(e0[["score", "threshold", "flagged"]], e1[["score", "threshold", "flagged"]])
    assert e1[e1.eligible_for_primary_metric].index.equals(e2[e2.eligible_for_primary_metric].index)
    normal = summaries["NORMAL_FULL_VS_POST_PREP"]
    normal = normal[normal.group_type.eq("overall")]
    anomaly = summaries["ANOMALY_FULL_VS_POST_PREP"]
    for (_, _), subset in normal.groupby(["family", "evaluation"]):
        scopes = subset.set_index("scope")
        assert int(scopes.loc["full", "n"]) == 259 and int(scopes.loc["post_preparation", "n"]) == 219
        for key in ("n", "fp"):
            assert scopes.loc["full", key] == scopes.loc["post_preparation", key] + scopes.loc["excluded_preparation", key]
    for (_, _, _), subset in anomaly.groupby(["family", "evaluation", "condition"]):
        scopes = subset.set_index("scope")
        assert int(scopes.loc["full", "anomaly_n"]) == 30 and int(scopes.loc["post_preparation", "anomaly_n"]) == 20
        for key in ("anomaly_n", "tp", "paired_normal_n", "fp"):
            assert scopes.loc["full", key] == scopes.loc["post_preparation", key] + scopes.loc["excluded_preparation", key]
    e0_abc = anomaly[anomaly.evaluation.eq("E0") & anomaly.scope.eq("full")].set_index(["model_id", "condition"])
    expected = prior_abc.set_index(["model_id", "condition"]).loc[e0_abc.index]
    for key, prior in (("anomaly_n", "anomaly_n"), ("paired_normal_n", "normal_n"), ("fp", "FP"), ("tp", "TP")):
        np.testing.assert_array_equal(e0_abc[key], expected[prior])
    for key in ("auroc", "ap"):
        np.testing.assert_allclose(e0_abc[key], expected[key.upper()], rtol=1e-12, atol=1e-13)
    diagnostics = summaries["EXCLUDED_SEGMENT_DIAGNOSTICS"]
    np.testing.assert_array_equal(diagnostics.preparation_rule_excluded_n, diagnostics.n)
    np.testing.assert_array_equal(diagnostics.policy_not_evaluated_n,
                                  np.where(diagnostics.evaluation.eq("E0"), 0, diagnostics.n))
    calibration = summaries["THRESHOLD_AND_SCORE_DISTRIBUTIONS"]
    calibration = calibration[calibration.group_id.eq("calibration")]
    np.testing.assert_array_equal(calibration.threshold, calibration.score_q95_higher)


def run(source_root, out):
    if out.is_relative_to(source_root):
        raise ValueError("Output must be outside the original project tree")
    if (out / "VERIFICATION.json").exists() or (out / "MODEL_INDEX.csv").exists():
        raise FileExistsError("Preserve the completed run and select a new output directory")
    v4 = source_root / "hardware/xarm_model_versions/v4_bolt_fixed"
    v51 = source_root / "hardware/xarm_model_versions/v5_1_multisession_normal_study"
    legacy_path = v4 / "SPLIT_MANIFEST.csv"
    frozen_path = v4 / "FROZEN_V4_CANDIDATES.json"
    features_path = v51 / "metrics/CYCLE_FEATURES.csv"
    history_path = v51 / "metrics/BASELINE_CYCLE_SCORES.csv"
    prior_scopes_path = v51 / "metrics/BASELINE_REPRODUCTION.csv"
    abc_path = source_root / "hardware/No3_bolt_ABC_validation_20261106/results/MATCHED_CONDITION_METRICS.csv"
    audit_path = v51 / "verification/DATA_VERIFICATION.json"
    baseline_audit_path = v51 / "verification/baseline.json"
    mask_path = out / "PREPARATION_MASK.csv"
    mask_hash_path = out / "PREPARATION_MASK.sha256"
    split_path = out / "SPLIT_MANIFEST.csv"
    source = v4 / "analysis_code/normal_models.py"
    tracked = [legacy_path, frozen_path, features_path, history_path, prior_scopes_path, abc_path,
               audit_path, baseline_audit_path, mask_path, mask_hash_path, split_path, source,
               v4 / "FEATURE_DEFINITIONS.json", v4 / "TRAINING_CONFIG.json", Path(__file__), Path(__file__).with_name("summarize_study.py")]
    mask, mask_sha = load_frozen_mask(mask_path, mask_hash_path, split_path)
    features = unique_index(read_csv(features_path), "cycle_id")
    prior_audit = read_json(audit_path)
    prior_baseline_audit = read_json(baseline_audit_path)
    assert prior_audit["features_sha256"] == digest(features_path)
    assert prior_baseline_audit["verified"] and prior_baseline_audit["study_features_verified"]
    assert prior_audit["old_raw_hashes_reverified"] and prior_audit["restart_raw_feature_max_abs_error"] == 0
    legacy_ids = validate_inputs(mask, features, read_csv(legacy_path))
    frozen = read_json(frozen_path)
    specs = {item["model_id"]: item for item in frozen["models"]}
    assert frozen["split_manifest_sha256"] == digest(legacy_path)
    assert frozen["feature_definitions_sha256"] == digest(v4 / "FEATURE_DEFINITIONS.json")
    source_paths = {model: v4 / "models" / Path(specs[model]["path"]).name for model, _ in MODELS.values()}
    tracked.extend(source_paths.values())
    before = {path: digest(path) for path in tracked}
    for model, path in source_paths.items():
        assert before[path] == specs[model]["sha256"]
    training_configs = read_json(v4 / "TRAINING_CONFIG.json")["configs"]
    libraries = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     scikit_learn=sklearn.__version__, scipy=scipy.__version__, joblib=joblib.__version__)
    config = dict(study="V5.2 preparation exclusion", policy_id=POLICY, preparation_mask_sha256=mask_sha,
                  source_feature_table=str(features_path), source_feature_definitions=str(v4 / "FEATURE_DEFINITIONS.json"),
                  split_manifest_sha256=before[split_path], libraries=libraries,
                  roles={"old": {"train": 200, "calibration": 100, "test": 99}, "E2": {"train": 150, "calibration": 100, "test": 99}},
                  masks={"long": "one logical unit, original orders 1-50", "short": "orders 1-10 per predeclared logical unit",
                         "exceptions": sorted(EXCEPTIONS), "short_logical_collections": sorted(SHORT), "long_logical_collection": MAIN},
                  experiments={"E0": "saved model and cutoff, full quality-eligible evaluation",
                               "E1": "same E0 model/scores and cutoff, post-preparation evaluation",
                               "E2": "refit on retained original Train; recalibrate on unchanged original Calibration; same evaluation IDs as E1"},
                  model_families={family: {"baseline_id": model, "feature_count": n, "features": specs[model]["features"]}
                                  for family, (model, n) in MODELS.items()},
                  pca={"n_components": .95, "svd_solver": "full", "whiten": False, "seed": SEED,
                       "preprocessing": "train-only median imputation and StandardScaler", "score": "Q = mean((z - reconstructed_z)^2)",
                       "threshold": "unchanged calibration role q95 higher, strict score > threshold"},
                  inputs_sha256={p.relative_to(out).as_posix() if p.is_relative_to(out) else str(p): value
                                 for p, value in before.items()},
                  main_research_use="same existing exposed data, offline development only")
    config_path = out / "EXPERIMENT_CONFIG.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("The existing pre-score configuration disagrees with current inputs")
    if not config_path.exists():
        save_json(config_path, config)
    for subdir in ("metrics", "models"):
        (out / subdir).mkdir(exist_ok=True)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source.parent))
    from normal_models import Detector, threshold

    prior_scores, prior_scopes, prior_abc = (read_csv(p) for p in (history_path, prior_scopes_path, abc_path))
    metric_frames, index_rows, candidates, baseline_rows, scales, pca_rows = [], [], [], [], [], []
    train_ids = [cid for cid in legacy_ids["train"] if mask.loc[cid, "fit_train"]]
    calibration_ids = legacy_ids["validation"]
    assert len(train_ids) == 150 and set(train_ids).isdisjoint(calibration_ids)
    for family, (model, n_features) in MODELS.items():
        spec = specs[model]
        old = joblib.load(source_paths[model])
        names = spec["features"]
        assert len(names) == n_features and len(set(names)) == n_features
        assert old["model_id"] == model and old["features"] == names == training_configs[model]
        assert old["threshold"] == spec["threshold"] and old["training_cycle_ids"] == legacy_ids["train"]
        assert old["validation_cycle_ids"] == calibration_ids
        assert old["feature_definitions_sha256"] == frozen["feature_definitions_sha256"]
        assert old["split_manifest_sha256"] == frozen["split_manifest_sha256"]
        expected = unique_index(prior_scores[prior_scores.model_id.eq(model)], "cycle_id")
        expected_scopes = unique_index(prior_scopes[prior_scopes.model_id.eq(model)], "scope")
        assert set(expected.index) == set(features.index)
        fixed_scores, checks = replay_v4(old, features, expected, legacy_ids, expected_scopes)
        baseline_rows.extend(dict(family=family, model_id=model, model_sha256=before[source_paths[model]], **row) for row in checks)
        train = features.loc[train_ids, names].to_numpy(float)
        assert np.isfinite(features[names]).to_numpy().all()
        fit = Detector("PCA", SEED).fit(train)
        validate_fit(fit, train)
        new_scores = pd.Series(q_values(fit, features[names].to_numpy(float)), index=features.index)
        calibration = new_scores.loc[calibration_ids].to_numpy(float)
        cutoff = threshold(calibration)
        assert np.isfinite(cutoff) and cutoff > 0
        assert cutoff == float(np.sort(calibration)[int(np.ceil(.95 * (len(calibration) - 1)))])
        new_id = f"V52_{family}_E2_Q"
        artifact_path = out / "models" / f"{new_id}.joblib"
        bundle = dict(model_id=new_id, family=family, detector=fit, features=names, threshold=cutoff,
                      policy_id=POLICY, score_direction="higher is more anomalous", comparison="strict >",
                      training_cycle_ids=train_ids, validation_cycle_ids=calibration_ids,
                      source_model_sha256=before[source_paths[model]], preparation_mask_sha256=mask_sha,
                      split_manifest_sha256=before[split_path], features_sha256=before[features_path],
                      feature_definitions_sha256=frozen["feature_definitions_sha256"],
                      code_sha256={p.name: before[p] for p in (Path(__file__), Path(__file__).with_name("summarize_study.py"), source)},
                      experiment_config_sha256=digest(config_path), libraries=libraries)
        joblib.dump(bundle, artifact_path)
        reload = joblib.load(artifact_path)
        assert reload["threshold"] == cutoff and reload["features"] == names
        assert reload["training_cycle_ids"] == train_ids and reload["validation_cycle_ids"] == calibration_ids
        loaded_scores = q_values(reload["detector"], features[names].to_numpy(float))
        np.testing.assert_array_equal(loaded_scores, new_scores)
        np.testing.assert_array_equal(loaded_scores > cutoff, new_scores > cutoff)
        candidates.append({"model_id": new_id, "path": artifact_path.relative_to(out).as_posix(), "sha256": digest(artifact_path),
                           "family": family, "features": names, "threshold": cutoff, "fit_train_n": 150, "calibration_n": 100})
        for evaluation, artifact, obj, values in (("E0", source_paths[model], old, fixed_scores),
                                                  ("E1", source_paths[model], old, fixed_scores),
                                                  ("E2", artifact_path, bundle, new_scores)):
            detector = obj["detector"]
            cut = float(obj["threshold"])
            metric_frames.append(build_cycle_rows(mask, values, cut, family, evaluation, obj["model_id"]))
            index_rows.append(dict(family=family, evaluation=evaluation, model_id=obj["model_id"],
                                   path=artifact.relative_to(out).as_posix() if evaluation == "E2" else str(artifact),
                                   sha256=digest(artifact), feature_count=len(names), features="|".join(names), threshold=cut,
                                   fit_train_n=detector.model.n_samples_, calibration_n=len(calibration_ids),
                                   pca_components=detector.model.n_components_,
                                   explained_variance_fraction=detector.model.explained_variance_ratio_.sum(),
                                   calibration_fp=int((values.loc[calibration_ids] > cut).sum()),
                                   deployment_status="comparison_only" if evaluation == "E2" else "unchanged_baseline"))
            if evaluation == "E1":
                continue
            for j, name in enumerate(names):
                scales.append(dict(family=family, evaluation=evaluation, model_id=obj["model_id"], feature_order=j + 1,
                                   feature=name, median=detector.imputer.statistics_[j], center=detector.scaler.mean_[j],
                                   scale=detector.scaler.scale_[j], variance=detector.scaler.var_[j]))
            cumulative = np.cumsum(detector.model.explained_variance_ratio_)
            for j, component in enumerate(detector.model.explained_variance_ratio_):
                pca_rows.append(dict(family=family, evaluation=evaluation, model_id=obj["model_id"], component=j + 1,
                                     explained_variance=detector.model.explained_variance_[j], variance_ratio=component,
                                     cumulative_variance_ratio=cumulative[j]))

    cycles = pd.concat(metric_frames, ignore_index=True)
    summaries = build_summaries(cycles)
    verify_matched_rows(cycles, summaries, prior_abc)
    assert before == {path: digest(path) for path in tracked}, "A frozen source input changed during the study"
    for name, table in {**summaries, "CYCLE_SCORES": cycles, "BASELINE_REPRODUCTION": baseline_rows,
                        "STANDARDIZATION_PARAMETERS": scales, "PCA_COMPONENTS": pca_rows}.items():
        save_csv(out / "metrics" / f"{name}.csv", table)
    save_csv(out / "MODEL_INDEX.csv", index_rows)
    save_json(out / "FROZEN_V52_CANDIDATES.json", {"policy_id": POLICY, "preparation_mask_sha256": mask_sha,
              "promotion_approved": False, "operational_model_changed": False,
              "status": "Saved as fixed research comparisons; no candidate promoted to operations", "candidates": candidates})
    verified = dict(verified=True, policy_id=POLICY, frozen_mask_hash_checked=mask_sha,
                    scored_quality_cycles=649, mask_rows=len(mask), excluded_quality_cycles=int((~mask.quality_eligible).sum()),
                    score_rows=len(cycles), original_and_fit_train_n=[200, 150], unchanged_calibration_n=100, unchanged_test_n=99,
                    matched_post_preparation_cycles_per_family=279, same_original_role=True, no_evaluation_data_fitted=True,
                    all_three_replays_match_saved_scores_and_flags=True, original_calibration_q95_exact=True,
                    unchanged_baselines_e0_e1=True, matched_e1_e2_cycle_ids=True, e0_paired_abc_matches_original_counts_and_ranks=True,
                    no_preparation_policy_tuning=True, original_inputs_hashes_unchanged=True,
                    refit_models_reloaded_scores_and_flags_exact=True, score_input_audit="V5.1 frozen features, prior raw verification, and baseline replay",
                    all_fit_cutoffs_from_original_calibration=True, no_device_access=True,
                    max_v4_replay_abs_error=max(x["max_abs_score_difference"] for x in baseline_rows))
    save_json(out / "VERIFICATION.json", verified)
    print(json.dumps(verified, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Original project tree with saved V4 and V5.1")
    parser.add_argument("--out", type=Path, default=STUDY, help="Study directory with frozen mask and split inputs")
    args = parser.parse_args()
    run(args.source_root.resolve(), args.out.resolve())
