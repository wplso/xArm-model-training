"""Run the preregistered 24-feature normal-only xArm algorithm benchmark."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from model_runtime import (ATOL, RTOL, FOLD_IDS, FORMULAS, FROZEN_IDS, GAMMA_MULTIPLIER,
                           MAIN_IDS, SEED, calibration_cutoff, check_reload_and_invariance,
                           estimator_for, gamma_scale, prepare_estimator, require_feature_order,
                           score_z, solver_details, verify_self_exclusion)


STUDY = Path(__file__).resolve().parents[1]
MAIN_BASIC = ("LOF_K20", "OCSVM_G1", "KNN_K20_MEAN")
BENCHMARK_SPLITS = ("PRIMARY", "A_MAIN", "B_MAIN")
MODEL_IDS = {"PRIMARY": FROZEN_IDS + MAIN_IDS, "A_MAIN": FOLD_IDS, "B_MAIN": FOLD_IDS}
ROLE_MAP = {"T0": "train", "T_ADD": "train", "C0": "calibration", "C_ADD": "calibration", "E": "evaluation",
            "TEST99": "historical_test", "UNUSED": "unused"}
EXPECTED_ROLES = {"PRIMARY": {"train": 200, "calibration": 100, "evaluation": 349},
                  "A_MAIN": {"train": 280, "calibration": 140, "evaluation": 40, "historical_test": 99, "unused": 90},
                  "B_MAIN": {"train": 240, "calibration": 140, "evaluation": 170, "historical_test": 99}}
META = ["cohort", "condition", "pair_condition", "session_id", "session_cycle_index", "cycle_index", "eval_session",
        "leakage_group_id", "start_pose", "capture_uuid", "calendar_date"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint_array(array):
    matrix = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def fingerprint_preprocessor(imputer, scaler):
    return dict(imputer_statistics_sha256=fingerprint_array(imputer.statistics_),
                scaler_center_sha256=fingerprint_array(scaler.mean_),
                scaler_scale_sha256=fingerprint_array(scaler.scale_),
                scaler_variance_sha256=fingerprint_array(scaler.var_))


def csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def save_csv(path, value):
    pd.DataFrame(value).to_csv(path, encoding="utf-8-sig", index=False)


def import_previous(path):
    spec = importlib.util.spec_from_file_location("frozen_if_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_splits(usage, prior_split, group_index):
    observed_groups = usage.groupby("leakage_group_id").agg(n_cycles=("condition", "size"),
                        n_normal=("condition", lambda x: x.eq("normal").sum())).reset_index()
    observed_groups["n_anomaly"] = observed_groups.n_cycles - observed_groups.n_normal
    pd.testing.assert_frame_equal(observed_groups, group_index[observed_groups.columns], check_dtype=False)
    base = usage[META].assign(original_role=usage.role, acquisition_session_id=usage.acquisition_session_id)
    roles = usage.role.map({"train": "train", "validation": "calibration", "test": "evaluation", "evaluation_only": "evaluation"})
    output = {"PRIMARY": base.assign(role=roles, split_role=usage.role)}
    orig = usage.role.map({"train": "T0", "validation": "C0", "test": "TEST99"})
    groups = usage.leakage_group_id
    normal = usage.condition.eq("normal")
    by_group = {"A_MAIN": {"G1_nov06_related": np.where(normal, "T_ADD", "UNUSED"), "G2_nov10_first40": "C_ADD", "G3_nov10_restart40": "E"},
                "B_MAIN": {"G1_nov06_related": "E", "G2_nov10_first40": "T_ADD", "G3_nov10_restart40": "C_ADD"}}
    for split, definitions in by_group.items():
        observed = prior_split[prior_split.fold_id.eq(split)].set_index("cycle_id")
        assert observed.index.is_unique and set(observed.index) == set(usage.index)
        observed = observed.loc[usage.index]
        for column in ("leakage_group_id", "capture_uuid", "cohort", "eval_session", "condition"):
            np.testing.assert_array_equal(observed[column], usage[column])
        expected = orig.copy()
        for group, value in definitions.items():
            rows = groups.eq(group).to_numpy()
            expected.loc[rows] = value[rows] if isinstance(value, np.ndarray) else value
        np.testing.assert_array_equal(observed.split_role, expected)
        output[split] = base.assign(role=observed.split_role.map(ROLE_MAP), split_role=observed.split_role)
    for split, frame in output.items():
        assert frame.role.value_counts().to_dict() == EXPECTED_ROLES[split]
        assert frame[frame.role.isin(["train", "calibration"])].condition.eq("normal").all()
        assert frame.index.is_unique and len(frame) == len(usage)
        test = frame[frame.role.eq("evaluation")]
        expected = {"PRIMARY": {"normal": 259, "A": 30, "B": 30, "C": 30},
                    "A_MAIN": {"normal": 40}, "B_MAIN": {"normal": 80, "A": 30, "B": 30, "C": 30}}
        assert test.condition.value_counts().to_dict() == expected[split]
        if split != "PRIMARY":
            fit_groups = set(frame.loc[frame.role.isin(["train", "calibration"]), "leakage_group_id"])
            assert fit_groups.isdisjoint(set(test.leakage_group_id))
        if split != "A_MAIN":
            assert test.loc[test.condition.eq("normal"), "pair_condition"].value_counts().to_dict() == {"A": 30, "B": 10, "C": 10}
    return output


def calibrate_scores(bundle, z, metadata):
    calibration = metadata.role.eq("calibration").to_numpy()
    used = metadata.role.isin(["train", "calibration", "evaluation"]).to_numpy()
    train_novelty_na = bundle["kind"] in {"lof", "knn_mean"}
    to_score = used & ~metadata.role.eq("train").to_numpy() if train_novelty_na else used
    values = np.full(len(z), np.nan)
    values[to_score] = score_z(bundle, z[to_score])
    assert np.isfinite(values[to_score]).all()
    return values, calibration_cutoff(values[calibration])


def frozen_score_replay(previous, result_dir, root, source_paths, source_hashes, usage, features, x, z, baseline):
    index = csv(result_dir / "MODEL_INDEX.csv").set_index("model")
    old_scores = csv(result_dir / "CYCLE_SCORES.csv")
    old_summary = csv(result_dir / "EVALUATION.csv").set_index(["model", "group"])
    old_rank = csv(result_dir / "PAIRED_CONDITION_RANKING.csv").set_index(["model", "condition"])
    train_ids = usage.index[usage.role.eq("train")].tolist()
    cal_ids = usage.index[usage.role.eq("validation")].tolist()
    det = baseline["detector"]
    q = pd.Series(index=usage.index, dtype=float)
    for cohorts in previous.Q_BATCHES:
        subset = usage.cohort.isin(cohorts)
        q.loc[subset] = det.score(features.loc[subset].to_numpy(float))
    original_v4 = csv(source_paths["prior_scores"])
    original_v4 = original_v4[original_v4.model_id.eq("V4_PCA")].set_index("cycle_id").loc[usage.index]
    np.testing.assert_allclose(q, original_v4.score, atol=1e-13, rtol=RTOL)
    np.testing.assert_array_equal(q.gt(baseline["threshold"]), original_v4.flagged)
    scored, models, reproduction = {}, {}, []
    preprocessing = fingerprint_preprocessor(det.imputer, det.scaler)
    for model_id in FROZEN_IDS:
        path = root / index.loc[model_id, "path"] if model_id == "V4_PCA_Q" else result_dir / index.loc[model_id, "path"]
        assert digest(path) == index.loc[model_id, "sha256"]
        threshold = float(index.loc[model_id, "threshold"])
        if model_id == "V4_PCA_Q":
            assert path == source_paths["model"] and threshold == baseline["threshold"]
            loaded = {"estimator": det.model, "imputer": det.imputer, "scaler": det.scaler}
            values, kind = q.to_numpy(), "pca_q"
        else:
            loaded = joblib.load(path)
            assert loaded["model_id"] == model_id and loaded["features"] == baseline["features"] and loaded["threshold"] == threshold
            assert loaded["train_cycle_ids"] == train_ids and loaded["calibration_cycle_ids"] == cal_ids
            assert loaded["source_v4_model_sha256"] == source_hashes["model"]
            assert loaded["run_config_sha256"] == digest(result_dir / "RUN_CONFIG.json")
            assert loaded["estimator"].get_params() == previous.read_json(result_dir / "RUN_CONFIG.json")["forest_parameters"]
            np.testing.assert_array_equal(previous.represent(x, loaded["imputer"], loaded["scaler"]), z)
            kind = "if_pc13_q" if model_id == "IF_PC13_Q" else "if"
            if model_id == "IF_PC13_Q":
                assert loaded["pca"].n_components_ == 13 and not loaded["pca"].whiten
                np.testing.assert_array_equal(loaded["pca"].components_, det.model.components_)
            values = score_z(dict(**loaded, kind=kind), z)
        assert fingerprint_preprocessor(loaded["imputer"], loaded["scaler"]) == preprocessing
        stored = old_scores[old_scores.model.eq(model_id)].set_index("cycle_id").loc[usage.index]
        assert stored.index.is_unique and len(stored) == len(usage)
        for field in ("role", "session_id", "session_cycle_index", "cohort", "condition", "pair_condition", "leakage_group_id"):
            np.testing.assert_array_equal(stored[field].fillna(""), usage[field].fillna(""))
        np.testing.assert_allclose(values, stored.score, atol=1e-13, rtol=RTOL)
        np.testing.assert_array_equal(values > threshold, stored.flagged)
        np.testing.assert_array_equal(stored.threshold, threshold)
        assert calibration_cutoff(values[usage.role.eq("validation")]) == threshold
        scored[model_id] = pd.DataFrame(dict(score=values, threshold=threshold, flagged=values > threshold), index=usage.index)
        models[model_id] = (dict(**loaded, kind=kind, model_id=model_id, split_id="PRIMARY", features=baseline["features"],
                                score_formula=FORMULAS[kind], train_cycle_ids=train_ids, calibration_cycle_ids=cal_ids,
                                preprocessor_sha256=preprocessing, threshold=threshold)
                            if model_id == "V4_PCA_Q" else
                            dict(loaded, kind=kind, split_id="PRIMARY", score_formula=FORMULAS[kind], preprocessor_sha256=preprocessing), path)
        reproduction.append(dict(model_id=model_id, group_type="cycle_replay", group_id="all649", n=len(values), threshold=threshold,
                                 max_abs_error=float(np.max(np.abs(values-stored.score))), alarms_match=True, calibration_threshold_exact=True,
                                 frozen_model_sha256=digest(path)))
    replayed_groups, _, replayed_pairs = previous.summaries(usage, scored)
    for row in replayed_groups.itertuples(index=False):
        legacy = old_summary.loc[(row.model, row.group)]
        assert (row.n, row.alarms) == (legacy.n, legacy.alarms)
        reproduction.append(dict(model_id=row.model, group_type="normal_or_anomaly", group_id=row.group,
                                 n=row.n, alarms=row.alarms, reference_alarms=legacy.alarms, alarms_match=True))
    for row in replayed_pairs.itertuples(index=False):
        legacy = old_rank.loc[(row.model, row.condition)]
        assert (row.anomaly_n, row.paired_normal_n, row.tp, row.paired_fp) == (legacy.anomaly_n, legacy.paired_normal_n, legacy.tp, legacy.paired_fp)
        np.testing.assert_allclose([row.auroc, row.ap], [legacy.auroc, legacy.ap], atol=1e-13, rtol=RTOL)
        reproduction.append(dict(model_id=row.model, group_type="paired_condition", group_id=row.condition,
                                 n=row.anomaly_n, alarms=row.tp, reference_alarms=legacy.tp, alarms_match=True,
                                 paired_normal_n=row.paired_normal_n, paired_fp=row.paired_fp, auroc=row.auroc, ap=row.ap))
    return scored, models, reproduction


def export_scores(metadata, values, threshold, split_id, model_id):
    used = metadata.role.isin(["train", "calibration", "evaluation"])
    frame = metadata.loc[used].copy()
    values = values[used]
    flags = pd.array(values > threshold, dtype="boolean")
    flags[~np.isfinite(values)] = pd.NA
    assert frame.role.loc[~np.isfinite(values)].eq("train").all()
    return frame.assign(split_id=split_id, model_id=model_id, score=values, threshold=threshold,
                        margin=values-threshold, flagged=flags,
                        score_status=np.where(np.isfinite(values), "scored", "training_novelty_not_scored")).reset_index()


def diagnose(bundle, split, data, features, z, values, warnings_, saved_path, refit):
    eval_ = data.role.eq("evaluation").to_numpy()
    qa = check_reload_and_invariance(bundle, features.loc[eval_], values[eval_], saved_path if refit else None)
    own = verify_self_exclusion(bundle)
    scopes = {role: data.role.eq(role).to_numpy() for role in ("train", "calibration", "evaluation")}
    scopes["evaluation_normal"] = eval_ & data.condition.eq("normal").to_numpy()
    scopes.update({f"evaluation_{label}": eval_ & data.condition.eq(label).to_numpy() for label in "ABC"})
    output = []
    for scope, rows in scopes.items():
        if rows.any():
            status = "scored" if np.isfinite(values[rows]).all() else "training_novelty_not_scored"
            entry = dict(row_kind="model_scope", split_id=split, model_id=bundle["model_id"], kind=bundle["kind"],
                         score_scope=scope, n=int(rows.sum()), status=status, threshold=bundle["threshold"],
                         warning_count=len(warnings_), warnings=" | ".join(warnings_), self_neighbor_matches=own)
            if status == "scored":
                entry.update(solver_details(bundle, z[rows], values[rows]))
            if scope == "evaluation":
                entry.update(qa)
            output.append(entry)
    return qa, output


def gamma_diagnostics(scored):
    out = []
    primary = scored[scored.split_id.eq("PRIMARY") & scored.model_id.isin(GAMMA_MULTIPLIER)]
    for scope, frame in primary.groupby("role", sort=False):
        for condition in ("all", "normal", "A", "B", "C"):
            sub = frame if condition == "all" else frame[frame.condition.eq(condition)]
            if sub.empty:
                continue
            values = sub.pivot(index="cycle_id", columns="model_id", values="score")
            flags = sub.pivot(index="cycle_id", columns="model_id", values="flagged")
            for a, b in (("OCSVM_G025", "OCSVM_G1"), ("OCSVM_G1", "OCSVM_G4"), ("OCSVM_G025", "OCSVM_G4")):
                out.append(dict(row_kind="gamma_pair", split_id="PRIMARY", model_id=a, comparison_model_id=b, kind="ocsvm",
                                score_scope=f"{scope}_{condition}", n=len(values), gamma_rank_spearman=spearmanr(values[a], values[b]).statistic,
                                gamma_alarm_disagreements=int((flags[a] != flags[b]).sum())))
    return out


def run(root, output):
    if output.is_relative_to(root):
        raise ValueError("Use a fresh benchmark output outside the original project tree")
    if (output / "MODEL_INDEX.csv").exists() or (output / "metrics/VERIFICATION.json").exists():
        raise FileExistsError("Use a new output directory for a new run")
    sys.dont_write_bytecode = True
    versions = root / "hardware/xarm_model_versions"
    comparison = versions / "v4_pca_if_comparison_20261111"
    v51 = versions / "v5_1_multisession_normal_study"
    previous_path = comparison / "compare_models.py"
    previous = import_previous(previous_path)
    source_paths, hashes, usage, features, baseline = previous.load_experiment(root)
    prior_results = comparison / "results"
    prior_config = previous.read_json(prior_results / "RUN_CONFIG.json")
    assert prior_config["input_sha256"] == hashes and prior_config["code_sha256"] == digest(previous_path)
    fold_path = v51 / "SPLIT_MANIFEST.csv"
    group_path = v51 / "GROUP_MANIFEST.csv"
    fold_receipt = v51 / "verification/VERIFICATION.json"
    raw_receipt = versions / "v5_2_preparation_exclusion_study/SOURCE_PROVENANCE_AFTER.json"
    referenced = previous.read_json(fold_receipt)["prefit_input_sha256"]
    for path in (fold_path, group_path):
        assert next(val for key, val in referenced.items() if Path(key).name == path.name and "v5_1_multisession_normal_study" in key) == digest(path)
    assert previous.read_json(raw_receipt)["unchanged"]
    metadata = build_splits(usage, csv(fold_path), csv(group_path))
    feature_source = previous.read_json(source_paths["feature_definitions"])
    names = baseline["features"]
    assert names == feature_source["ordered_features"]["V4_PCA"]
    x = require_feature_order(features, names)
    detector = baseline["detector"]
    libraries = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     scikit_learn=sklearn.__version__, scipy=scipy.__version__, joblib=joblib.__version__)
    source_tracked = set(source_paths.values()) | {previous_path, fold_path, group_path, fold_receipt, raw_receipt} | set(prior_results.rglob("*.csv")) | set(prior_results.rglob("*.json")) | set(prior_results.rglob("*.joblib"))
    source_before = {p: digest(p) for p in sorted(source_tracked)}
    code_hashes = {Path(__file__).name: digest(Path(__file__)), "model_runtime.py": digest(Path(__file__).with_name("model_runtime.py"))}
    cfg = dict(study="Normal-only algorithm benchmark", libraries=libraries, seed=SEED,
               preprocessing="Primary reuses the stored frozen V4 B0 transform; A_MAIN and B_MAIN fit their common transforms on normal Train only",
               ordered_features=names, data_roles=EXPECTED_ROLES, model_ids=MODEL_IDS, fixed_main_candidates=MAIN_BASIC,
               fixed_sensitivity=list(MAIN_IDS[3:]), prefit_estimator_settings={k: estimator_for(k, 1/24)[1].get_params() for k in dict.fromkeys((*MAIN_IDS, *FOLD_IDS))},
               svm_gamma_rule="Train only: base=1/(24*Z_train.var(ddof=0)); saved parameters above use the nominal 1/24 scale",
               svm_gamma_multiplier=GAMMA_MULTIPLIER, score_formulas=FORMULAS,
               threshold_rule="normal calibration q95(higher), strict score > threshold; display margin=score-threshold",
               numerical_score_tolerance={"absolute": ATOL, "relative": RTOL}, native_thread_limit=1,
               scopes={"PRIMARY": "all quality-valid evaluation, includes all preparation cycles", "A_MAIN": "restart group held out", "B_MAIN": "retrospective whole load campaign held out"},
               original_input_sha256={p.relative_to(root).as_posix(): sha for p, sha in source_before.items()}, code_sha256=code_hashes,
               no_online_training_or_deployment=True)
    cfg_path = output / "EXPERIMENT_CONFIG.json"
    output.mkdir(parents=True, exist_ok=True)
    if cfg_path.exists():
        assert previous.read_json(cfg_path) == previous.json.loads(previous.json.dumps(cfg)), "Presaved experiment differs"
    else:
        previous.write_json(cfg_path, cfg)
    feature_schema = dict(version="V4_PCA B0", ordered_features=names, input_size=len(names),
                          feature_definitions=[feature_source["definitions"][key] for key in names],
                          input_order="exact; reject reordered or extra columns", nonfinite_policy="raise on unexpected NaN or Inf",
                          source_sha256=hashes["feature_definitions"], physical_axis_orientation=feature_source["physical_axis_orientation"],
                          physical_calibration=feature_source["physical_calibration"])
    previous.write_json(output / "FEATURE_SCHEMA.json", feature_schema)
    save_csv(output / "DATA_USAGE_MANIFEST.csv", usage.reset_index().assign(original_role=usage.role.to_numpy(),
             benchmark_primary_role=metadata["PRIMARY"].role.to_numpy(), preparation_exclusion_applied=False))
    save_csv(output / "SPLIT_MANIFEST.csv", pd.concat([part.assign(split_id=split).reset_index() for split, part in metadata.items()], ignore_index=True))
    models_dir, metrics = output / "models", output / "metrics"
    models_dir.mkdir(exist_ok=True)
    metrics.mkdir(exist_ok=True)
    cfg_sha = digest(cfg_path)

    score_frames, index_rows, numerics = [], [], []
    fit_settings = {}
    frozen_z = previous.represent(x, detector.imputer, detector.scaler)
    with threadpool_limits(limits=1):
        # Inherited Q is reproduced on its original batches; score gates use the saved cutoffs.
        old_scored, old_models, reproduction = frozen_score_replay(previous, prior_results, root, source_paths, hashes, usage, features, x, frozen_z, baseline)
        for split, data in metadata.items():
            train = data.role.eq("train").to_numpy()
            cal = data.role.eq("calibration").to_numpy()
            transform_start = time.perf_counter_ns()
            if split == "PRIMARY":
                imputer, scaler, z = detector.imputer, detector.scaler, frozen_z
            else:
                imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[train])
                scaler = StandardScaler().fit(imputer.transform(x[train]))
                z = scaler.transform(imputer.transform(x))
                assert scaler.n_samples_seen_ == int(train.sum())
                np.testing.assert_allclose(imputer.statistics_, np.median(x[train], axis=0))
                np.testing.assert_allclose(scaler.mean_, x[train].mean(axis=0))
            transform_fit_ms = (time.perf_counter_ns() - transform_start) / 1e6 if split != "PRIMARY" else None
            base = gamma_scale(z[train])
            scaler_digest = fingerprint_preprocessor(imputer, scaler)
            fit_settings[split] = dict(gamma_base=base, standardized_matrix_sha256=fingerprint_array(z), preprocessing=scaler_digest)
            for model_id in MODEL_IDS[split]:
                archived = model_id in FROZEN_IDS
                if archived:
                    bundle, path = old_models[model_id]
                    values = old_scored[model_id].score.to_numpy()
                    fit_ms, messages = None, []
                else:
                    kind, estimator, fit_ms, messages = prepare_estimator(model_id, z[train], base)
                    bundle = dict(model_id=model_id, split_id=split, kind=kind, estimator=estimator, imputer=imputer, scaler=scaler,
                                  features=names, parameters=estimator.get_params(), gamma_base=base if kind == "ocsvm" else None,
                                  score_formula=FORMULAS[kind], train_cycle_ids=data.index[train].tolist(), calibration_cycle_ids=data.index[cal].tolist(),
                                  preprocessor_sha256=scaler_digest, preprocessor_source="V4 frozen B0" if split == "PRIMARY" else f"{split} Train only",
                                  source_v4_model_sha256=hashes["model"], features_sha256=hashes["features"], split_manifest_sha256=digest(output / "SPLIT_MANIFEST.csv"),
                                  calibration_rule="q95(higher); strict >", experiment_config_sha256=cfg_sha, code_sha256=code_hashes, libraries=libraries)
                    values, bundle["threshold"] = calibrate_scores(bundle, z, data)
                    path = models_dir / f"{split}_{model_id}.joblib"
                    joblib.dump(bundle, path)
                threshold = bundle["threshold"]
                assert calibration_cutoff(values[cal]) == threshold
                score_frames.append(export_scores(data, values, threshold, split, model_id))
                qa, details = diagnose(bundle, split, data, features, z, values, messages, path, not archived)
                numerics.extend(details)
                role = "frozen_baseline" if archived else ("group_fold" if split != "PRIMARY" else ("main" if model_id in MAIN_BASIC else "sensitivity"))
                estimator = bundle["estimator"]
                status = "attention" if messages or getattr(estimator, "fit_status_", 0) else "verified"
                index_rows.append(dict(split_id=split, model_id=model_id, model_role=role, kind=bundle["kind"], status=status,
                                       path=(path.relative_to(root) if archived else path.relative_to(output)).as_posix(),
                                       path_base="project_root" if archived else "study_root", sha256=digest(path), bytes=path.stat().st_size,
                                       feature_count=24, estimator_input_dim=getattr(estimator, "n_features_in_", 24),
                                       pca_components=getattr(estimator, "n_components_", 13 if model_id == "IF_PC13_Q" else None),
                                       threshold=threshold, fit_n=int(train.sum()), calibration_n=int(cal.sum()),
                                       calibration_fp=int((values[cal] > threshold).sum()),
                                       svm_gamma_base=base if bundle["kind"] == "ocsvm" else None,
                                       svm_actual_gamma=estimator.get_params().get("gamma"),
                                       estimator_fit_ms=fit_ms, shared_preprocessing_fit_ms=transform_fit_ms,
                                       score_formula=bundle["score_formula"], **scaler_digest, **qa))

    scores = pd.concat(score_frames, ignore_index=True)
    assert len(scores) == 10530 and not scores.duplicated(["split_id", "model_id", "cycle_id"]).any()
    assert len(index_rows) == 18 and sum(row["model_role"] != "frozen_baseline" for row in index_rows) == 15
    assert scores.score.isna().equals(scores.role.eq("train") & scores.model_id.str.startswith(("LOF_", "KNN_")))
    np.testing.assert_array_equal(scores.loc[scores.score.notna(), "margin"].gt(0), scores.loc[scores.score.notna(), "flagged"])
    numerics.extend(gamma_diagnostics(scores))
    source_unchanged = all(digest(path) == sha for path, sha in source_before.items())
    assert source_unchanged, "The frozen source changed during this run"
    # Exercise the same entry guard used for a serialized model's feature input.
    for broken in (features.iloc[:, ::-1], features.assign(**{names[0]: float("nan")})):
        try:
            require_feature_order(broken, names)
        except ValueError:
            continue
        raise AssertionError("Malformed feature input was accepted")
    for file, table in ((metrics / "CYCLE_SCORES.csv", scores), (metrics / "BASELINE_REPRODUCTION.csv", reproduction),
                        (metrics / "NUMERICAL_DIAGNOSTICS.csv", numerics), (output / "MODEL_INDEX.csv", index_rows)):
        save_csv(file, table)
    qa = dict(verified=True, old_files_unchanged=True, raw_reverification_reused=True, preparation_excluded_n=0,
              unique_cycles=649, primary_train_n=200, primary_calibration_n=100, primary_eval_normal_n=259, primary_eval_anomaly_n=90,
              expected_roles=EXPECTED_ROLES, score_rows=len(scores), frozen_replays=3, new_models=15,
              old_if_standardized_input_equal=True, frozen_replay_max_abs_score_error=max(r.get("max_abs_error", 0) for r in reproduction),
              baseline_scores_flags_thresholds_and_pair_ranks_verified=True, original_and_group_train_roles_verified=True,
              new_models_reload_identical=True, all_group_fold_models_share_train_only_transform=True,
              strict_margin_sign_verified=True, feature_order_and_nonfinite_input_rejected=True,
              lof_and_knn_train_novelty_missing=True, calibration_only_thresholds_verified=True,
              single_batch_shuffle_reverse_same_flags=True,
              max_score_error_by_mode={mode: max(r.get(f"{mode}_max_abs_error", 0) for r in index_rows) for mode in ("batch", "single", "shuffled", "reversed")},
              library_versions=libraries, fit_settings=fit_settings, source_files_checked=len(source_before),
              experiment_config_sha256=cfg_sha, cycle_scores_sha256=digest(metrics / "CYCLE_SCORES.csv"), no_device_access=True)
    previous.write_json(metrics / "VERIFICATION.json", qa)
    print(f"Verified {len(scores)} rows and {len(index_rows)} split/model combinations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=STUDY)
    args = ap.parse_args()
    run(args.project_root.resolve(strict=True), args.output.resolve())
