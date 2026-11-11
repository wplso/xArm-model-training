"""Score quality-reviewed post-freeze captures using the three frozen models."""

import argparse
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits


STUDY = Path(__file__).resolve().parent
LOADS = ("A121", "B82", "C182")
LABELS = ("normal", *LOADS)
PHYSICAL_CONDITIONS = {
    "normal": (41, "none", 0),
    "A121": (121, "none", 0),
    "B82": (41, "upper", 82),
    "C182": (41, "lower", 182),
}
MODELS = {
    "V4_PCA_Q": "pca_q",
    "LOGISTIC_MONSTER_GEL": "logistic_4class",
    "LOGISTIC_BOLT": "logistic_4class",
}
REQUIRED = (
    "cycle_id", "session_id", "cycle_index", "source", "No3_begin_ns", "No3_end_ns",
    "quality_valid", "label", "pair_condition", "campaign_id", "role",
)
RAW_HASH_COLUMNS = {
    "ina_csv_sha256": "sensors/measurements.csv",
    "adxl_csv_sha256": "sensors/adxl345_1.csv",
    "position_csv_sha256": "robot/single_queries.csv",
}
OPTIONAL_IQR = "ID1_3_5s_IQR"


def aware_time(value, name):
    try:
        instant = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp with a timezone") from exc
    if instant.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return instant


def load_frozen(project):
    analysis = project / "hardware/xarm_model_versions/v4_bolt_fixed/analysis_code"
    sys.path.insert(0, str(analysis))
    try:
        common = importlib.import_module("v4_common")
        inputs = common.Inputs()
        frozen = inputs.json(STUDY / "FROZEN_MODELS.json")
        aware_time(frozen["frozen_utc"], "frozen_utc")
        features = frozen["features"]
        if len(features) != 24 or len(set(features)) != 24 or OPTIONAL_IQR not in features:
            raise ValueError("Expected the 24 unique V4 features in frozen order")
        specs = frozen["models"]
        if len(specs) != len(MODELS) or {s["model_id"]: s["kind"] for s in specs} != MODELS:
            raise ValueError("Expected exactly the three prespecified model IDs and kinds")
        if frozen["library_versions"]["sklearn"] != sklearn.__version__:
            raise ValueError("Use the frozen scikit-learn version to load these model artifacts")
        for file, key in (
            (analysis / "normal_models.py", "v4_model_code_sha256"),
            (analysis / "v4_common.py", "v4_feature_extractor_sha256"),
            (STUDY / "PROTOCOL.json", "protocol_sha256"),
            (STUDY / "FIT_CAL_MANIFEST.csv", "fit_cal_manifest_sha256"),
            (STUDY / "HISTORICAL_CAPTURE_IDS.csv", "historical_capture_ids_sha256"),
        ):
            path = str(inputs.track(file))
            if inputs.hashes[path] != frozen[key]:
                raise ValueError(f"Frozen dependency hash mismatch: {file.name}")
        bundles = {}
        for spec in specs:
            model_id = spec["model_id"]
            model_path = (STUDY / spec["artifact"]).resolve()
            if not model_path.is_relative_to(STUDY) or model_path.suffix != ".joblib":
                raise ValueError(f"Invalid artifact path for {model_id}")
            if inputs.hashes[str(inputs.track(model_path))] != spec["sha256"]:
                raise ValueError(f"Artifact hash mismatch for {model_id}")
            bundle = joblib.load(model_path)
            threshold = spec["threshold"]
            if not math.isfinite(threshold) or threshold != bundle["threshold"]:
                raise ValueError(f"Frozen threshold mismatch for {model_id}")
            if list(bundle["features"]) != features:
                raise ValueError(f"Frozen feature order mismatch for {model_id}")
            if spec["kind"] == "logistic_4class":
                model = bundle["model"]
                if set(model.classes_) != set(LABELS) or list(model.feature_names_in_) != features:
                    raise ValueError(f"Frozen classes or pipeline feature order mismatch for {model_id}")
            bundles[model_id] = bundle
    finally:
        sys.path.remove(str(analysis))
    return common, inputs, frozen, bundles


def validate_manifest(path, project, frozen, inputs):
    data = inputs.csv(path, dtype=str, keep_default_na=False)
    missing = set(REQUIRED) - set(data.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    data = data[list(REQUIRED)].copy()
    if data.empty or any(data[c].eq("").any() for c in REQUIRED if c != "pair_condition"):
        raise ValueError("The prospective manifest contains blank required values")
    if not data["role"].eq("evaluation").all() or not data["quality_valid"].str.lower().eq("true").all():
        raise ValueError("All rows must be quality_valid=true and role=evaluation")
    if set(data["label"]) != set(LABELS):
        raise ValueError("The manifest must contain normal, A121, B82 and C182")
    if data["campaign_id"].nunique() != 1:
        raise ValueError("Evaluate one new acquisition campaign at a time")
    is_normal = data["label"].eq("normal")
    if set(data.loc[is_normal, "pair_condition"]) - {"", *LOADS}:
        raise ValueError("A normal row can be unpaired or paired to A121, B82 or C182")
    if not set(LOADS).issubset(data.loc[is_normal, "pair_condition"]):
        raise ValueError("Provide a separate normal control for each of A121, B82 and C182")
    anomalous = data.loc[~is_normal]
    if not (anomalous["pair_condition"].eq("") | anomalous["pair_condition"].eq(anomalous["label"])).all():
        raise ValueError("A load row's pair_condition must be empty or equal to its label")
    for field in ("cycle_index", "No3_begin_ns", "No3_end_ns"):
        if not data[field].str.fullmatch(r"[0-9]+").all():
            raise ValueError(f"{field} must contain integers")
        data[field] = data[field].astype(np.int64)
    if not data["cycle_id"].eq(data["session_id"] + "_" + data["cycle_index"].astype(str)).all():
        raise ValueError("cycle_id must equal session_id plus the cycle index")
    if (data["cycle_index"] <= 0).any() or (data["No3_end_ns"] <= data["No3_begin_ns"]).any():
        raise ValueError("Cycle indices and No.3 window lengths must be positive")
    data["source"] = data["source"].map(lambda folder: str((project / folder).resolve()))
    if data.duplicated("cycle_id").any() or data.duplicated(["source", "No3_begin_ns", "No3_end_ns"]).any():
        raise ValueError("Duplicate cycle or No.3 raw window")
    capture_columns = ["source", "label", "pair_condition", "campaign_id"]
    if not data.groupby("session_id")[capture_columns].nunique().eq(1).all().all():
        raise ValueError("A capture may have only one source, condition, pair and campaign")
    if not data.groupby("source")["session_id"].nunique().eq(1).all():
        raise ValueError("A capture source may have only one session ID")
    windows = data.sort_values(["source", "No3_begin_ns"])
    previous_end = windows.groupby("source")["No3_end_ns"].shift(fill_value=-1)
    if windows["No3_begin_ns"].le(previous_end).any():
        raise ValueError("Overlapping No.3 windows within a capture")
    history = inputs.csv(STUDY / "HISTORICAL_CAPTURE_IDS.csv", dtype=str, keep_default_na=False)
    if "session_id" not in history or not history["session_id"].ne("").any():
        raise ValueError("The historical capture ledger has no session IDs")
    overlap = set(data["session_id"]) & set(history["session_id"])
    if overlap:
        raise ValueError(f"Historical capture IDs are in the prospective manifest: {sorted(overlap)}")
    frozen_utc = aware_time(frozen["frozen_utc"], "frozen_utc")
    capture_plan = inputs.json(STUDY / "CAPTURE_PLAN.json")
    return_sha256 = capture_plan["capture_policy"]["return_sha256"]
    raw_hashes = {column: set(history[column]) for column in RAW_HASH_COLUMNS if column in history}
    for (source_path, session_id), rows in data.groupby(["source", "session_id"], sort=False):
        source = Path(source_path)
        saved = inputs.json(source / "manifest.json")
        if saved.get("session_id") != session_id:
            raise ValueError(f"Source manifest session mismatch for {session_id}")
        if aware_time(saved.get("created_utc"), "capture created_utc") <= frozen_utc:
            raise ValueError(f"Capture must be created after the model freeze: {session_id}")
        conditions = saved.get("conditions") or {}
        actual = tuple(conditions.get(name) for name in ("payload_mass_g", "frame_site", "frame_added_mass_g"))
        if actual != PHYSICAL_CONDITIONS[rows.label.iloc[0]]:
            raise ValueError(f"Capture mass or frame site differs from its declared label: {session_id}")
        if (conditions.get("accelerometer_mount") != "bolt_and_nut_user_reported"
                or conditions.get("base_mount") != "rigid_fixed_operator_reported"
                or conditions.get("dataset_role") != "evaluation_only"
                or conditions.get("automatic_return") is not True
                or saved.get("source_sha256") != return_sha256):
            raise ValueError(f"Capture mount, return, or role metadata mismatch: {session_id}")
        for column, previous in raw_hashes.items():
            raw_path = str(inputs.track(source / RAW_HASH_COLUMNS[column]))
            if inputs.hashes[raw_path] in previous:
                raise ValueError(f"Historical raw sensor or position CSV: {raw_path}")
        for sensor in ("sensors/measurements.csv", "sensors/adxl345_1.csv"):
            sensor_ids = inputs.csv(source / sensor, usecols=["session_id"], dtype=str)["session_id"]
            if sensor_ids.empty or not sensor_ids.eq(session_id).all():
                raise ValueError(f"Raw sensor session mismatch in {source / sensor}")
        completed = {}
        with inputs.track(source / "events.jsonl").open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("kind") != "cycle_completed":
                    continue
                index = event["cycle_index"]
                if event.get("session_id") != session_id or index in completed:
                    raise ValueError(f"Invalid completed event for {session_id}_{index}")
                result = event["result"]
                completed[index] = (result["No3_write_completed_ns"], result["No3_confirmed_ns"])
        for row in rows.itertuples(index=False):
            if completed.get(row.cycle_index) != (row.No3_begin_ns, row.No3_end_ns):
                raise ValueError(f"No.3 bounds differ from the completed event for {row.cycle_id}")
    return data


def score_models(data, frozen, bundles):
    x = data[frozen["features"]]
    values = x.to_numpy(float)
    valid = np.isfinite(values)
    iqr_column = x.columns.get_loc(OPTIONAL_IQR)
    valid[:, iqr_column] |= np.isnan(values[:, iqr_column])
    bad = np.argwhere(~valid)
    if bad.size:
        i, j = bad[0]
        raise ValueError(f"Nonfinite frozen feature {x.columns[j]} at {data.iloc[i]['cycle_id']}")
    labels = ["cycle_id", "session_id", "campaign_id", "label", "pair_condition"]
    results = []
    with threadpool_limits(limits=1):
        for spec in frozen["models"]:
            model_id, threshold = spec["model_id"], spec["threshold"]
            bundle = bundles[model_id]
            rows = data[labels].copy()
            if spec["kind"] == "pca_q":
                score = bundle["detector"].score(x.to_numpy(float))
                prediction = np.where(score > threshold, "known_load_unspecified", "normal")
            else:
                model = bundle["model"]
                probabilities = pd.DataFrame(model.predict_proba(x), columns=model.classes_, index=x.index)
                score = 1 - probabilities["normal"].to_numpy()
                prediction = probabilities[list(LOADS)].idxmax(axis=1).where(score > threshold, "normal")
                rows = rows.join(probabilities.add_prefix("p_"))
            if not np.isfinite(score).all():
                raise ValueError(f"Nonfinite score for {model_id}")
            results.append(rows.assign(model_id=model_id, score=score, threshold=threshold,
                                       missing_id1_iqr=x[OPTIONAL_IQR].isna(),
                                       margin=score - threshold, alarm=score > threshold,
                                       predicted_label=prediction))
    return pd.concat(results, ignore_index=True)


def evaluate(scores):
    overall, normal_by_session, paired, confusion = [], [], [], []
    for model_id, group in scores.groupby("model_id", sort=False):
        normal = group[group.label.eq("normal")]
        anomalies = group[~group.label.eq("normal")]
        overall.append(dict(model_id=model_id, fp=int(normal.alarm.sum()), normal_n=len(normal),
                            fpr=normal.alarm.mean(), tp=int(anomalies.alarm.sum()),
                            anomaly_n=len(anomalies), tpr=anomalies.alarm.mean()))
        for (campaign, session, pair), rows in normal.groupby(
            ["campaign_id", "session_id", "pair_condition"], sort=False
        ):
            normal_by_session.append(dict(model_id=model_id, campaign_id=campaign, session_id=session,
                                          pair_condition=pair, fp=int(rows.alarm.sum()),
                                          normal_n=len(rows), fpr=rows.alarm.mean()))
        for target in LOADS:
            positives = anomalies[anomalies.label.eq(target)]
            controls = normal[normal.pair_condition.eq(target)]
            combined = pd.concat([positives, controls])
            actual = combined.label.eq(target)
            typed = int(positives.predicted_label.eq(target).sum()) if model_id != "V4_PCA_Q" else None
            paired.append(dict(model_id=model_id, label=target, fp=int(controls.alarm.sum()),
                               normal_n=len(controls), fpr=controls.alarm.mean(),
                               tp=int(positives.alarm.sum()), anomaly_n=len(positives), tpr=positives.alarm.mean(),
                               correct_type=typed, auroc=roc_auc_score(actual, combined.score),
                               average_precision=average_precision_score(actual, combined.score)))
        if model_id != "V4_PCA_Q":
            for actual in LABELS:
                for predicted in LABELS:
                    count = int((group.label.eq(actual) & group.predicted_label.eq(predicted)).sum())
                    confusion.append(dict(model_id=model_id, actual=actual, predicted=predicted, n=count))
    old = scores[scores.model_id.eq("LOGISTIC_MONSTER_GEL")].set_index("cycle_id")
    new = scores[scores.model_id.eq("LOGISTIC_BOLT")].set_index("cycle_id").loc[old.index]
    changes = old[["session_id", "label", "pair_condition"]].assign(
        monster_gel_alarm=old.alarm, bolt_alarm=new.alarm,
        transition=np.select([old.alarm & new.alarm, old.alarm & ~new.alarm, ~old.alarm & new.alarm],
                             ["both_alarm", "monster_gel_only", "bolt_only"], default="neither"),
    ).reset_index()
    return {
        "OVERALL.csv": pd.DataFrame(overall),
        "NORMAL_SESSION_FPR.csv": pd.DataFrame(normal_by_session),
        "PAIRED_LOAD_METRICS.csv": pd.DataFrame(paired),
        "TYPE_CONFUSION.csv": pd.DataFrame(confusion),
        "PAIRED_MODEL_CHANGES.csv": changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify_only and (args.manifest is not None or args.output is not None):
        parser.error("--verify-only does not accept --manifest or --output")
    if not args.verify_only and (args.manifest is None or args.output is None):
        parser.error("--manifest and --output are required to evaluate")
    project = args.project.resolve()
    common, inputs, frozen, bundles = load_frozen(project)
    if args.verify_only:
        if not inputs.unchanged():
            raise ValueError("At least one frozen input changed during verification")
        print("Verified frozen hashes, feature order and thresholds:")
        for spec in frozen["models"]:
            print(f"{spec['model_id']}  {spec['threshold']:.17g}")
        return
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    manifest = validate_manifest(args.manifest, project, frozen, inputs)
    features, quality = common.extract_features(manifest, project, inputs)
    scores = score_models(features, frozen, bundles)
    quality = quality.merge(
        features[["cycle_id"]].assign(missing_id1_iqr=features[OPTIONAL_IQR].isna()),
        on="cycle_id", validate="one_to_one",
    )
    if not inputs.unchanged():
        raise ValueError("At least one tracked input changed during evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    metrics = evaluate(scores)
    for name, table in {**metrics, "CYCLE_SCORES.csv": scores, "CYCLE_FEATURES.csv": features,
                        "SAMPLE_QUALITY.csv": quality}.items():
        common.savecsv(args.output / name, table)
    common.savej(args.output / "VERIFICATION.json", dict(
        frozen_utc=frozen["frozen_utc"], prospective_cycles=len(manifest),
        capture_count=manifest.session_id.nunique(),
        missing_id1_iqr_count=int(quality.missing_id1_iqr.sum()),
        counts_by_label=manifest.label.value_counts().to_dict(),
        counts_by_normal_pair=manifest.loc[manifest.label.eq("normal"), "pair_condition"].value_counts().to_dict(),
        model_hashes={s["model_id"]: s["sha256"] for s in frozen["models"]},
        reviewed_manifest_sha256=common.sha(args.manifest), tracked_input_hashes=inputs.hashes,
        old_inputs_unchanged=True, refit=False, threshold_change=False,
    ))
    print(metrics["OVERALL.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
