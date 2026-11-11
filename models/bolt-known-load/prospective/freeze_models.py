"""Fit only the predeclared historical bolt split and freeze three research artifacts."""

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import shutil
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from threadpoolctl import threadpool_limits


REFERENCE_STUDY = "supervised_load_session_holdout_20261111"
FROZEN_FILES = {
    "V4_PCA_Q": "V4_PCA_Q.joblib",
    "LOGISTIC_MONSTER_GEL": "LOGISTIC_MONSTER_GEL.joblib",
    "LOGISTIC_BOLT": "LOGISTIC_BOLT.joblib",
}
HASH_COLUMNS = ["ina_csv_sha256", "adxl_csv_sha256", "position_csv_sha256"]
CLASSES = {"normal", "A121", "B82", "C182"}


def archived_helpers(study):
    spec = importlib.util.spec_from_file_location("previous_supervised", study / "run_supervised.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_roles(bolt, bolt_source, v4_roles, old_model, protocol):
    expected = protocol["bolt_fit"]
    campaign = bolt[bolt.campaign.eq(expected["campaign"])]
    assert len(campaign) == 170
    fit = campaign[campaign.cohort.isin(expected["cohorts"])]
    assert fit.cohort.value_counts().to_dict() == expected["cohorts"]
    assert fit.label.value_counts().to_dict() == {
        label: expected[f"{label}_n"] for label in CLASSES
    }
    assert len(campaign) - len(fit) == expected["excluded_same_campaign_normal_n"]
    assert campaign.loc[campaign.index.difference(fit.index), "label"].eq("normal").all()
    assert fit.start_pose.eq("A_release").all()
    raw_fit = bolt_source.loc[fit.index]
    assert raw_fit.sensor_mount.eq("bolt_and_nut_user_reported").all()
    assert raw_fit.base_mount.str.startswith("rigid_fixed").all()
    for label, mass, site, frame_mass in (
        ("normal", 41, "none", 0), ("A121", 121, "none", 0),
        ("B82", 41, "upper", 82), ("C182", 41, "lower", 182),
    ):
        subset = raw_fit.loc[fit.label.eq(label)]
        assert subset.payload_mass_g.eq(mass).all() and subset.frame_added_mass_g.eq(frame_mass).all()
        assert subset.frame_site.fillna("none").eq(site).all()
    cal_ids = v4_roles.index[v4_roles.role.eq("validation")]
    assert old_model["calibration_ids"] == cal_ids.tolist()
    cal = bolt.loc[cal_ids]
    assert len(cal) == protocol["calibration"]["normal_n"] and cal.label.eq("normal").all()
    assert cal.campaign.eq(protocol["calibration"]["campaign"]).all()
    for column in ("campaign", "session_id", "source"):
        assert set(fit[column]).isdisjoint(cal[column])
    assert set(fit.index).isdisjoint(cal.index)
    selected = pd.concat([fit, cal], verify_integrity=True)
    assert set(bolt.index[bolt.session_id.isin(selected.session_id)]) == set(selected.index)
    return fit, cal


def historical_captures(old, bolt):
    columns = ["session_id", "source"]
    earlier = old[columns].assign(mount="monster_gel")
    current = bolt[[*columns, *HASH_COLUMNS]].assign(mount="bolt")
    captures = pd.concat([earlier, current], ignore_index=True).drop_duplicates()
    assert captures.session_id.is_unique and len(captures) == 60
    assert captures.loc[captures.mount.eq("bolt"), HASH_COLUMNS].notna().all().all()
    return captures


def check_calibration(values, threshold, original):
    assert np.isfinite(values).all()
    assert float(np.quantile(values, .95, method="higher")) == threshold
    assert int(np.count_nonzero(values > threshold)) == 4
    if original is not None:
        np.testing.assert_allclose(values, original, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(values > threshold, np.asarray(original) > threshold)


def run(project, output):
    study = project / "hardware/xarm_model_versions" / REFERENCE_STUDY
    prior = archived_helpers(study)
    digest, read_json, write_json = prior.digest, prior.read_json, prior.write_json
    prior_run = read_json(study / "results/RUN_CONFIG.json")
    assert digest(study / "run_supervised.py") == prior_run["code_sha256"]
    protocol_path = Path(__file__).with_name("PROTOCOL.json")
    protocol = read_json(protocol_path)
    old_source, bolt_source, v4_scores, v4_spec, v4, roles, paths, source_hashes = prior.load_inputs(project)
    _, bolt = prior.input_frames(old_source, bolt_source, v4_spec["features"])
    old_index = prior.read_csv(study / "results/MODEL_INDEX.csv")
    old_spec = old_index.query("split_id == 'OLD_TO_BOLT' and model_id == 'LOGISTIC'").iloc[0]
    old_path = study / "results" / old_spec.path
    assert digest(old_path) == old_spec.sha256
    old_model = joblib.load(old_path)
    assert old_model["features"] == v4_spec["features"] and old_model["threshold"] == old_spec.threshold
    assert len(old_model["train_ids"]) == old_spec.train_n == 200
    fit, cal = data_roles(bolt, bolt_source, roles, old_model, protocol)
    assert v4["validation_cycle_ids"] == cal.index.tolist()
    assert set(fit.index).isdisjoint(old_model["train_ids"])
    assert set(fit.index).isdisjoint(v4["training_cycle_ids"])
    features = v4_spec["features"]
    assert len(features) == 24
    assert np.isfinite(fit[features]).all().all() and np.isfinite(cal[features]).all().all()

    original_estimator = old_model["model"].steps[-1][1]
    for name, value in protocol["learner"].items():
        assert original_estimator.get_params()[name] == value
    bolt_model = clone(old_model["model"])
    historical = historical_captures(old_source, bolt_source)
    old_scores_path = study / "results/CYCLE_SCORES.csv"
    original_lr_cal = prior.read_csv(old_scores_path).query(
        "split_id == 'OLD_TO_BOLT' and model_id == 'LOGISTIC' and role == 'calibration'"
    ).set_index("cycle_id")
    assert set(original_lr_cal.index) == set(cal.index)
    protected = {**{path: source_hashes[key] for key, path in paths.items()},
                 old_path: old_spec.sha256, old_scores_path: digest(old_scores_path)}
    feature_extractor = paths["v4_code"].with_name("v4_common.py")
    protected[feature_extractor] = digest(feature_extractor)
    output.mkdir(parents=True, exist_ok=True)
    models = output / "models"
    if models.exists() or (output / "FROZEN_MODELS.json").exists():
        raise FileExistsError("Choose a new output directory to preserve frozen artifacts")
    models.mkdir()
    used = pd.concat([fit.assign(role="train"), cal.assign(role="calibration")], verify_integrity=True)
    fit_manifest = output / "FIT_CAL_MANIFEST.csv"
    used.drop(columns=features).reset_index().to_csv(fit_manifest, index=False)
    history_path = output / "HISTORICAL_CAPTURE_IDS.csv"
    historical.to_csv(history_path, index=False)

    with warnings.catch_warnings(record=True) as issued, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        bolt_model.fit(fit[features], fit.label)
        if issued:
            raise RuntimeError(f"Unexpected training warning: {issued[0].message}")
        assert set(bolt_model.classes_) == CLASSES
        x_cal = cal[features]
        new_score = 1-prior.p_scores(bolt_model, x_cal, features).normal
        new_threshold = float(np.quantile(new_score, .95, method="higher"))
        check_calibration(new_score, new_threshold, None)
        old_score = 1-prior.p_scores(old_model["model"], x_cal, features).normal
        check_calibration(old_score, old_spec.threshold, original_lr_cal.loc[cal.index, "score"])
        q_score = v4["detector"].score(x_cal.to_numpy(float))
        check_calibration(q_score, v4_spec["threshold"], v4_scores.loc[cal.index, "score"])

        shutil.copyfile(paths["v4_model"], models / FROZEN_FILES["V4_PCA_Q"])
        shutil.copyfile(old_path, models / FROZEN_FILES["LOGISTIC_MONSTER_GEL"])
        new_path = models / FROZEN_FILES["LOGISTIC_BOLT"]
        joblib.dump({
            "model": bolt_model, "features": features, "threshold": new_threshold,
            "classes": list(bolt_model.classes_), "model_id": "LOGISTIC_BOLT",
            "train_ids": fit.index.tolist(), "calibration_ids": cal.index.tolist(),
        }, new_path)
        restored = joblib.load(new_path)
        assert restored["threshold"] == new_threshold
        np.testing.assert_array_equal(1-prior.p_scores(restored["model"], x_cal, features).normal, new_score)
    assert digest(models / FROZEN_FILES["V4_PCA_Q"]) == v4_spec["sha256"]
    assert digest(models / FROZEN_FILES["LOGISTIC_MONSTER_GEL"]) == old_spec.sha256
    assert all(digest(path) == value for path, value in protected.items())
    items = [
        ("V4_PCA_Q", "pca_q", paths["v4_model"], v4_spec["threshold"], len(v4["training_cycle_ids"])),
        ("LOGISTIC_MONSTER_GEL", "logistic_4class", old_path, old_spec.threshold, len(old_model["train_ids"])),
        ("LOGISTIC_BOLT", "logistic_4class", None, new_threshold, len(fit)),
    ]
    specs = [
        dict(model_id=name, kind=kind, artifact=f"models/{FROZEN_FILES[name]}",
             sha256=digest(models / FROZEN_FILES[name]), threshold=threshold,
             original_artifact=str(source.relative_to(project)) if source else None,
             train_n=train_n, calibration_n=len(cal), calibration_alarm_n=4)
        for name, kind, source, threshold, train_n in items
    ]
    scores = pd.DataFrame({"cycle_id": cal.index, "V4_PCA_Q": q_score,
                           "LOGISTIC_MONSTER_GEL": old_score, "LOGISTIC_BOLT": new_score})
    scores.to_csv(output / "CALIBRATION_SCORES.csv", index=False)
    frozen_utc = datetime.now(timezone.utc).isoformat()
    write_json(output / "FROZEN_MODELS.json", dict(
        frozen_utc=frozen_utc, status="frozen_for_future_evaluation", features=features, models=specs,
        protocol_sha256=digest(protocol_path), fit_cal_manifest_sha256=digest(fit_manifest),
        historical_capture_ids_sha256=digest(history_path),
        v4_model_code_sha256=source_hashes["v4_code"],
        v4_feature_extractor_sha256=protected[feature_extractor],
        library_versions=dict(python=sys.version.split()[0], sklearn=sklearn.__version__,
                              numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__),
    ))
    write_json(output / "FREEZE_VERIFICATION.json", dict(
        references_byte_identical=True, reference_calibration_reproduced=True, bolt_reload_exact=True,
        role_disjoint_by_cycle_capture_and_campaign=True, frozen_models=3,
        train_n=len(fit), calibration_n=len(cal), historical_capture_n=len(historical),
        historical_eval_scored_n=0, prospective_captures_n=0,
        frozen_manifest_sha256=digest(output / "FROZEN_MODELS.json"),
        protected_source_sha256={str(path.relative_to(project)): value for path, value in protected.items()},
    ))
    for spec in specs:
        print(spec["model_id"], spec["threshold"], spec["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.project, args.output)
