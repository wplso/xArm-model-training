"""Fit a separate bolt logistic candidate on preassigned normal blocks."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits


MODELS = Path("hardware/xarm_model_versions")
STUDY = MODELS / "bolt_known_load_prospective_20261111"
CAMPAIGN = Path("hardware/No3_bolt_normal_retrain_20261111")
PREVIOUS = Path("hardware/No3_bolt_ABC_interleaved_20261111")
CAPTURE_PLAN_SHA256 = "3c52560785b5648502c64fb0c409ba234604e97898fb3849fed083669061f4ac"
STAGES = {"N01": "train", "N02": "train", "N03": "calibration", "N04": "evaluation"}
BLOCK_N = 60
NEW_MODEL = "LOGISTIC_BOLT_NORMAL_EXPANDED"
BOLT = "LOGISTIC_BOLT"
FEATURE_CONDITION = {
    "payload_mass_g": 41, "frame_site": "none", "frame_added_mass_g": 0,
    "accelerometer_mount": "bolt_and_nut_user_reported", "base_mount": "rigid_fixed_operator_reported",
    "automatic_return": True, "object_present": True,
}
AUDIT_FLAGS = (
    "raw_uart_csv_identical", "raw_hid_csv_identical", "backup_sha256_verified",
    "backup_CRC_verified", "backup_members_identical", "source_hashes_unchanged",
)
AUDIT_ZERO = ("position_missing", "position_timeout", "ADXL_encoding_bad")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_reference(project):
    evaluator = runpy.run_path(str(project / STUDY / "evaluate_frozen.py"))
    common, inputs, frozen, bundles = evaluator["load_frozen"](project)
    inputs.track(project / STUDY / "FROZEN_MODELS.json")
    inputs.track(project / STUDY / "evaluate_frozen.py")
    inputs.track(Path(__file__))
    history_path = project / MODELS / "v5_1_multisession_normal_study/metrics/CYCLE_FEATURES.csv"
    expected_hash = inputs.json(project / STUDY / "FREEZE_VERIFICATION.json")["protected_source_sha256"]
    expected = next(value for name, value in expected_hash.items()
                    if Path(name).as_posix().replace("\\", "/") == history_path.relative_to(project).as_posix())
    require(common.sha(inputs.track(history_path)) == expected, "Original bolt features differ from the frozen source")
    original = inputs.csv(project / STUDY / "FIT_CAL_MANIFEST.csv", keep_default_na=False)
    original = original.loc[original.role.eq("train")].set_index("cycle_id")
    require(original.index.is_unique and original.label.value_counts().to_dict()
            == {"normal": 50, "A121": 30, "B82": 30, "C182": 30}, "Original bolt training counts changed")
    require(original.index.tolist() == bundles[BOLT]["train_ids"] and original.mount.eq("bolt").all()
            and original.campaign.nunique() == 1, "Original fit cycles differ from the frozen bolt artifact")
    features = frozen["features"]
    original_x = inputs.csv(history_path).set_index("cycle_id")
    require(original_x.index.is_unique, "Historical features contain duplicate cycles")
    original = original.join(original_x[features], validate="one_to_one")

    previous = project / PREVIOUS / "analysis/frozen_evaluation"
    prior_evaluator_path = inputs.track(project / PREVIOUS / "pc/evaluate_interleaved.py")
    prior_evaluator = runpy.run_path(str(prior_evaluator_path))
    _, prior_rows, _, evidence = prior_evaluator["collect_manifest"](
        project / PREVIOUS, project / STUDY / "FROZEN_MODELS.json")
    for path, sha256 in evidence.items():
        require(common.sha(inputs.track(path)) == sha256, f"Previous evaluation evidence changed: {path.name}")
    seen = inputs.csv(previous / "CYCLE_FEATURES.csv", keep_default_na=False).set_index("cycle_id")
    recorded = inputs.csv(previous / "CYCLE_SCORES.csv", keep_default_na=False)
    evaluated = pd.DataFrame(prior_rows).set_index("cycle_id")
    require(seen.index.is_unique and seen.index.equals(evaluated.index)
            and seen.label.value_counts().to_dict() == {"normal": 90, "A121": 30, "B82": 30, "C182": 30},
            "Prior evaluation cycles changed")
    for field in ("session_id", "source", "No3_begin_ns", "No3_end_ns", "label", "pair_condition", "campaign_id"):
        require(seen[field].astype(str).equals(evaluated[field].astype(str)), f"Prior evaluation {field} changed")
    seen = pd.concat([seen.drop(columns=features), seen[features].apply(pd.to_numeric, errors="raise")],
                     axis=1).reset_index()
    verify = inputs.json(previous / "VERIFICATION.json")
    model_hashes = {spec["model_id"]: spec["sha256"] for spec in frozen["models"]}
    require(verify["model_hashes"] == model_hashes and not verify["refit"]
            and not verify["threshold_change"], "Prior evaluation provenance differs")
    with threadpool_limits(limits=1):
        replay = evaluator["score_models"](seen, frozen, bundles)
    key = ["model_id", "cycle_id"]
    replay = replay.set_index(key).sort_index()
    recorded = recorded.set_index(key).sort_index()
    require(replay.index.equals(recorded.index) and replay.index.is_unique, "Prior score identities differ")
    np.testing.assert_allclose(replay[["score", "threshold"]], recorded[["score", "threshold"]],
                               atol=2e-13, rtol=0)
    for field in ("alarm", "predicted_label", "label", "pair_condition"):
        require(replay[field].equals(recorded[field]), f"Prior score replay differs on {field}")
    return evaluator, common, inputs, frozen, bundles, original, seen


def read_new_manifest(project, manifest_path, inputs, common, evaluator, frozen, previous_sessions):
    campaign = project / CAMPAIGN
    plan_path = campaign / "CAPTURE_PLAN.json"
    plan = inputs.json(plan_path)
    expected_stages = [dict(id=key, role=role, count=BLOCK_N) for key, role in STAGES.items()]
    require(common.sha(inputs.track(plan_path)) == CAPTURE_PLAN_SHA256
            and common.sha(inputs.track(project / STUDY / "FROZEN_MODELS.json")) == plan.get("frozen_models_sha256")
            and plan.get("campaign_id") == campaign.name and plan.get("sequence") == expected_stages,
            "Capture plan does not match the preassigned four blocks")
    review = inputs.json(campaign / "OPERATOR_REVIEW.json")
    require(review.get("all_cycles_confirmed") is True and review.get("accepted_cycle_n") == 240
            and isinstance(review.get("operator_confirmation"), str) and review["operator_confirmation"].strip()
            and review.get("stage_cycles") == dict.fromkeys(STAGES, BLOCK_N)
            and review.get("capture_plan_sha256") == CAPTURE_PLAN_SHA256
            and review.get("normal_manifest_sha256") == common.sha(inputs.track(manifest_path)),
            "Complete operator confirmation and manifest identity are required before training")
    data = inputs.csv(manifest_path, dtype=str, keep_default_na=False)
    columns = [*evaluator["REQUIRED"], "stage"]
    require(list(data.columns) == columns and len(data) == 240
            and data.cycle_id.is_unique and data.stage.value_counts().to_dict() == dict.fromkeys(STAGES, BLOCK_N)
            and data.label.eq("normal").all() and data.campaign_id.eq(campaign.name).all()
            and data.pair_condition.eq("").all() and data.quality_valid.str.lower().eq("true").all(),
            "Reviewed manifest schema, conditions or cycle count differs from the plan")
    for field in ("cycle_index", "No3_begin_ns", "No3_end_ns"):
        require(data[field].str.fullmatch(r"[0-9]+").all(), f"{field} must contain integer strings")
        data[field] = data[field].astype(np.int64)
    require(data.cycle_id.eq(data.session_id + "_" + data.cycle_index.astype(str)).all(),
            "A cycle ID differs from its session and index")
    history = inputs.csv(project / STUDY / "HISTORICAL_CAPTURE_IDS.csv", dtype=str, keep_default_na=False)
    banned_sessions = previous_sessions | set(history.session_id)
    raw_hashes = {field: set(history[field]) for field in evaluator["RAW_HASH_COLUMNS"]}
    expected_return = inputs.json(project / STUDY / "CAPTURE_PLAN.json")["capture_policy"]["return_sha256"]
    last_end = 0
    last_created = evaluator["aware_time"](frozen["frozen_utc"], "model frozen_utc")
    stage_summary = []
    for stage, role in STAGES.items():
        rows = data.loc[data.stage.eq(stage)].sort_values("cycle_index")
        require(rows.role.eq(role).all() and rows.cycle_index.tolist() == list(range(1, BLOCK_N + 1))
                and rows.session_id.nunique() == rows.source.nunique() == 1
                and rows.No3_end_ns.gt(rows.No3_begin_ns).all()
                and rows.No3_begin_ns.gt(rows.No3_end_ns.shift(fill_value=last_end)).all(),
                f"{stage}: roles or windows overlap or differ from the plan")
        last_end = int(rows.No3_end_ns.iloc[-1])
        session_id = rows.session_id.iloc[0]
        require(session_id not in banned_sessions, f"{stage}: session is reused or belongs to earlier data")
        banned_sessions.add(session_id)
        source = Path(rows.source.iloc[0]).resolve()
        require(source.is_relative_to(campaign.resolve()), f"{stage}: source belongs to another campaign")
        audit_path = campaign / stage / "analysis" / stage / "audit.json"
        audit = inputs.json(audit_path)
        condition = {**FEATURE_CONDITION, "campaign_id": campaign.name,
                     "label": "normal", "dataset_role": role, "training_registration": False}
        require(audit.get("stage") == stage and audit.get("cycles") == audit.get("quality_valid") == BLOCK_N
                and Path(audit.get("source", "")).resolve() == source
                and all(audit["condition"].get(field) == value for field, value in condition.items())
                and audit.get("models_updated") is False and audit.get("thresholds_updated") is False
                and all(audit.get(field) is True for field in AUDIT_FLAGS)
                and all(audit.get(field) == 0 for field in AUDIT_ZERO)
                and all(value == 0 for field in ("sequence_INA", "sequence_ADXL")
                        for value in audit[field].values()), f"{stage}: data quality or audit evidence mismatch")
        saved = inputs.json(source / "manifest.json")
        conditions = saved.get("conditions", {})
        created = evaluator["aware_time"](saved.get("created_utc"), f"{stage} created_utc")
        require(created > last_created and saved.get("session_id") == session_id
                and saved.get("source_sha256") == expected_return
                and saved.get("split_role") == role
                and all(conditions.get(field) == value for field, value in condition.items()),
                f"{stage}: capture conditions, timing or return file mismatch")
        last_created = created
        for field, filename in evaluator["RAW_HASH_COLUMNS"].items():
            digest = common.sha(inputs.track(source / filename))
            require(digest not in raw_hashes[field], f"{stage}: reused or historical raw file {filename}")
            raw_hashes[field].add(digest)
        for filename in ("sensors/measurements.csv", "sensors/adxl345_1.csv"):
            ids = inputs.csv(source / filename, usecols=["session_id"], dtype=str).session_id
            require(len(ids) and ids.eq(session_id).all(), f"{stage}: raw sensor session differs")
        events = {}
        with inputs.track(source / "events.jsonl").open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("kind") == "cycle_completed":
                    index = event["cycle_index"]
                    require(event["session_id"] == session_id and index not in events,
                            f"{stage}: duplicate or incorrect raw completion event")
                    result = event["result"]
                    events[index] = (result["No3_write_completed_ns"], result["No3_confirmed_ns"])
        expected = {row.cycle_index: (row.No3_begin_ns, row.No3_end_ns) for row in rows.itertuples(index=False)}
        require(events == expected, f"{stage}: exact cycle windows differ from completion events")
        stage_summary.append(dict(stage=stage, role=role, n=BLOCK_N, source=str(source), session_id=session_id,
                                  created_utc=created.isoformat(), audit_sha256=common.sha(audit_path)))
    return data.sort_values(["stage", "cycle_index"]), stage_summary


def score_rollup(scores):
    summary = []
    for (cohort, model, label), rows in scores.groupby(["cohort", "model_id", "label"], sort=False):
        n, alarms = len(rows), int(rows.alarm.sum())
        summary.append(dict(cohort=cohort, model_id=model, label=label, n=n, alarms=alarms,
                            alarm_rate=alarms / n, correct_load_type=(
                                int(rows.predicted_label.eq(label).sum()) if label != "normal" else None),
                            score_median=rows.score.median(), score_p90=rows.score.quantile(.9),
                            threshold=rows.threshold.iloc[0]))
    return pd.DataFrame(summary)


def run(project, manifest_path, output, verify_reference_only):
    evaluator, common, inputs, frozen, bundles, original, seen = read_reference(project)
    if verify_reference_only:
        require(inputs.unchanged(), "Existing reference changed during validation")
        print("Verified original bolt fit and replayed frozen models on the earlier 180 cycles without fitting")
        return
    require(not output.exists(), "Select a new output directory")
    data, stages = read_new_manifest(project, manifest_path, inputs, common, evaluator, frozen,
                                     set(seen.session_id) | set(original.session_id))
    new, quality = common.extract_features(data, project, inputs)
    features = frozen["features"]
    optional = evaluator["OPTIONAL_IQR"]
    require(new.cycle_id.tolist() == data.cycle_id.tolist() and new.cycle_id.is_unique
            and np.isfinite(new[[key for key in features if key != optional]]).all().all()
            and not np.isinf(new[optional]).any() and np.isfinite(original[features]).all().all(),
            "Nonfinite feature other than the frozen optional IQR was found")
    missing_iqr = new[optional].isna()
    quality = quality.merge(new[["cycle_id"]].assign(missing_id1_iqr=missing_iqr),
                            on="cycle_id", validate="one_to_one")
    fit_new = new.loc[new.role.eq("train")]
    fit = pd.concat([original.reset_index(), fit_new], ignore_index=True)
    require(fit.cycle_id.is_unique and fit.label.value_counts().to_dict()
            == {"normal": 170, "A121": 30, "B82": 30, "C182": 30}, "Unexpected expanded fit counts")
    cal = new.loc[new.role.eq("calibration")]
    holdout = new.loc[new.role.eq("evaluation")]
    for first, second in ((fit, cal), (fit, holdout), (cal, holdout), (fit, seen), (cal, seen), (holdout, seen)):
        require(set(first.session_id).isdisjoint(second.session_id), "A session crosses model roles")

    pipeline = clone(bundles[BOLT]["model"])
    with warnings.catch_warnings(record=True) as issued, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        pipeline.fit(fit[features], fit.label)
        if issued:
            raise RuntimeError(f"Unexpected training warning: {issued[0].message}")
        normal_index = list(pipeline.classes_).index("normal")
        cal_scores = 1 - pipeline.predict_proba(cal[features])[:, normal_index]
    threshold = float(np.quantile(cal_scores, .95, method="higher"))
    candidate = dict(model=pipeline, features=features, threshold=threshold, classes=list(pipeline.classes_),
                     model_id=NEW_MODEL, train_ids=fit.cycle_id.tolist(), calibration_ids=cal.cycle_id.tolist())
    new_spec = dict(model_id=NEW_MODEL, kind="logistic_4class", threshold=threshold,
                    artifact=f"models/{NEW_MODEL}.joblib", train_n=len(fit), calibration_n=len(cal))
    score_specs = {**frozen, "models": [*frozen["models"], new_spec]}
    all_bundles = {**bundles, NEW_MODEL: candidate}
    by_scope = []
    for name, rows in (("new_calibration", cal), ("new_normal_holdout", holdout),
                       ("previously_seen_development_regression", seen)):
        table = evaluator["score_models"](rows, score_specs, all_bundles)
        by_scope.append(table.assign(cohort=name))
    scores = pd.concat(by_scope, ignore_index=True)
    require(len(scores) == 4 * (60 + 60 + 180)
            and not scores.duplicated(["model_id", "cycle_id"]).any(), "Incomplete or duplicate score rows")
    metrics = score_rollup(scores)
    regression = evaluator["evaluate"](by_scope[-1])
    bolt = scores.loc[scores.model_id.eq(BOLT)].set_index("cycle_id")
    updated = scores.loc[scores.model_id.eq(NEW_MODEL)].set_index("cycle_id").loc[bolt.index]
    changes = bolt[["session_id", "label", "cohort"]].assign(
        frozen_bolt_alarm=bolt.alarm, updated_bolt_alarm=updated.alarm,
        frozen_bolt_type=bolt.predicted_label, updated_bolt_type=updated.predicted_label,
        transition=np.select([bolt.alarm & updated.alarm, bolt.alarm, updated.alarm],
                             ["both_alarm", "frozen_only", "updated_only"], default="neither"),
    ).reset_index()
    series = scores.loc[scores.cohort.eq("new_normal_holdout")].merge(
        holdout[["cycle_id", "cycle_index"]], on="cycle_id", validate="many_to_one")
    series["group_10"] = (series.cycle_index - 1) // 10 + 1
    window_metrics = series.groupby(["model_id", "group_10"], as_index=False).agg(
        n=("alarm", "size"), fp=("alarm", "sum"), fpr=("alarm", "mean"),
        median_score=("score", "median"), p90_score=("score", lambda values: values.quantile(.9)))
    require(inputs.unchanged(), "A frozen or source input changed during model development")

    roles = pd.concat([fit[["cycle_id", "session_id", "source", "label", "role", "stage"]],
                       cal[["cycle_id", "session_id", "source", "label", "role", "stage"]],
                       holdout[["cycle_id", "session_id", "source", "label", "role", "stage"]]], ignore_index=True)
    tables = (
        ("NEW_NORMAL_FEATURES.csv", new), ("SAMPLE_QUALITY.csv", quality),
        ("FIT_CAL_HOLDOUT_MANIFEST.csv", roles), ("CYCLE_SCORES.csv", scores),
        ("COMPARISON.csv", metrics), ("NEW_HOLDOUT_10_CYCLE_BLOCKS.csv", window_metrics),
        ("PAIRED_OLD_NEW_BOLT.csv", changes), ("PREVIOUS_PAIRED_LOAD_METRICS.csv", regression["PAIRED_LOAD_METRICS.csv"]),
        ("PREVIOUS_TYPE_CONFUSION.csv", regression["TYPE_CONFUSION.csv"]),
    )
    metadata = dict(
        candidate=new_spec, features=features, references=frozen["models"],
        historical_train_n=len(original), new_train_n=len(fit_new), new_calibration_n=len(cal),
        new_normal_holdout_n=len(holdout), previous_development_eval_n=len(seen), stages=stages,
        calibration_rule="higher empirical q95 of N03 only; strict score > threshold",
        model_and_preprocessing_settings="clone frozen bolt logistic pipeline; fit on expanded train only",
        coefficient_or_threshold_tuning_on_holdout=False, reference_models_unchanged=True,
        new_normal_holdout_scope="later held-out block of the same monitored acquisition campaign",
        previous_evaluation_scope="previously analyzed comparison; no training/calibration rows from this set",
        missing_required_features=0,
        optional_id1_iqr_imputed_by_role=new.loc[missing_iqr, "role"].value_counts().to_dict(),
        reload_calibration_scores_exact=True, input_sha256=inputs.hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".expanded-bolt-") as temporary:
        staging = Path(temporary) / "candidate"
        path = staging / new_spec["artifact"]
        path.parent.mkdir(parents=True)
        joblib.dump(candidate, path)
        restored = joblib.load(path)
        require(restored["threshold"] == threshold and restored["features"] == features, "Saved candidate differs")
        with threadpool_limits(limits=1):
            restored_scores = 1 - restored["model"].predict_proba(cal[features])[:, normal_index]
            np.testing.assert_array_equal(cal_scores, restored_scores)
        new_spec["sha256"] = common.sha(path)
        for name, table in tables:
            common.savecsv(staging / name, table)
        common.savej(staging / "VERIFICATION.json", metadata)
        require(inputs.unchanged(), "A frozen or source input changed while saving")
        staging.rename(output)
    print(metrics[["cohort", "model_id", "label", "alarms", "n", "alarm_rate", "correct_load_type"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-reference-only", action="store_true")
    args = parser.parse_args()
    if args.verify_reference_only == (args.output is not None):
        parser.error("Supply either --verify-reference-only or --output")
    project = args.project.resolve()
    run(project, args.manifest or project / CAMPAIGN / "NORMAL_MANIFEST.csv",
        args.output, args.verify_reference_only)
