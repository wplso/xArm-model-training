"""Score the reviewed B/C campaign once with the frozen expanded bolt model."""

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits


CAMPAIGN = "No3_bolt_BC_independent_20261112"
PLAN_SHA256 = "a2ced3c928d84c6793201e3367775d43fd50c75dcef000950673b2117cde2ec1"
FROZEN_SPEC_SHA256 = "fa22f2bdcc6df511b153ea878be8d0944c4540b04020aea3da5d55df7c3e575e"
MODEL_ID = "LOGISTIC_BOLT_NORMAL_EXPANDED"
MODEL_SHA256 = "d13f356a01973a4fbb188396301b9c98f9062a36d24d35b7421031907bde548d"
THRESHOLDS = {"stored": 0.8214447226813868, "candidate": 0.40}
FIELDS = (
    "cycle_id", "session_id", "cycle_index", "source", "No3_begin_ns", "No3_end_ns",
    "quality_valid", "label", "pair_condition", "campaign_id", "role",
)
LOAD_TYPES = ("A121", "B82", "C182")
SEQUENCE = (
    ("N0", "normal", 10), ("B1", "B82", 15), ("C1", "C182", 15),
    ("N1", "normal", 10), ("C2", "C182", 15), ("B2", "B82", 15), ("N2", "normal", 10),
)
AUDIT_FLAGS = (
    "raw_uart_csv_identical", "raw_hid_csv_identical", "backup_sha256_verified",
    "backup_CRC_verified", "backup_members_identical", "source_hashes_unchanged",
)
RAW_FIELDS = {
    "ina_csv_sha256": "sensors/measurements.csv",
    "adxl_csv_sha256": "sensors/adxl345_1.csv",
    "position_csv_sha256": "robot/single_queries.csv",
}
OPTIONAL = "ID1_3_5s_IQR"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capture_utc(value):
    date = datetime.fromisoformat(value)
    require(date.utcoffset() is not None, "Capture timestamp must include a timezone")
    return date


def load_contract(project, campaign):
    study = project / "hardware/xarm_model_versions/bolt_known_load_prospective_20261111"
    reference = project / "hardware/No3_bolt_normal_retrain_20261111/analysis/retrained_bolt_normal"
    extractor = project / "hardware/xarm_model_versions/v4_bolt_fixed/analysis_code/v4_common.py"
    spec = importlib.util.spec_from_file_location("v4_common", extractor)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    inputs = common.Inputs()
    plan_path = campaign / "CAPTURE_PLAN.json"
    plan = inputs.json(plan_path)
    require(inputs.hashes[str(plan_path)] == PLAN_SHA256, "Capture plan hash differs")
    require(plan["campaign_id"] == CAMPAIGN and plan["model_id"] == MODEL_ID
            and plan["model_sha256"] == MODEL_SHA256 and plan["evaluation_role"] == "evaluation_only"
            and plan["stored_threshold"] == THRESHOLDS["stored"]
            and plan["candidate_threshold"] == THRESHOLDS["candidate"]
            and plan["alarm_rule"] == "1 - P(normal) > threshold"
            and tuple((row["id"], row["label"], row["count"]) for row in plan["sequence"]) == SEQUENCE,
            "Unexpected fixed campaign configuration")
    frozen_path = study / "FROZEN_MODELS.json"
    frozen = inputs.json(frozen_path)
    require(inputs.hashes[str(frozen_path)] == FROZEN_SPEC_SHA256, "Feature contract differs")
    require(common.sha(inputs.track(extractor)) == frozen["v4_feature_extractor_sha256"]
            and sklearn.__version__ == frozen["library_versions"]["sklearn"],
            "Frozen feature extractor or scikit-learn version differs")
    ver = inputs.json(reference / "VERIFICATION.json")
    model_spec = ver["candidate"]
    features = frozen["features"]
    require(len(features) == len(set(features)) == 24 and OPTIONAL in features
            and ver["features"] == features and model_spec["model_id"] == MODEL_ID
            and model_spec["sha256"] == MODEL_SHA256 and model_spec["threshold"] == THRESHOLDS["stored"],
            "Expanded model specification differs")
    model_path = reference / "models" / f"{MODEL_ID}.joblib"
    require(common.sha(inputs.track(model_path)) == MODEL_SHA256, "Expanded model SHA-256 differs")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    require(bundle["model_id"] == MODEL_ID and bundle["threshold"] == THRESHOLDS["stored"]
            and list(bundle["features"]) == list(model.feature_names_in_) == features
            and set(model.classes_) == {"normal", *LOAD_TYPES}, "Saved expanded pipeline differs")

    ledger = inputs.csv(study / "HISTORICAL_CAPTURE_IDS.csv", dtype=str, keep_default_na=False)
    frozen_roles = inputs.csv(reference / "FIT_CAL_HOLDOUT_MANIFEST.csv", dtype=str, keep_default_na=False)
    previous = inputs.csv(project / "hardware/No3_bolt_ABC_interleaved_20261111/analysis/frozen_evaluation/EVALUATION_MANIFEST.csv",
                          dtype=str, keep_default_na=False)
    require(frozen_roles.loc[frozen_roles.role.eq("train"), "cycle_id"].tolist() == bundle["train_ids"]
            and frozen_roles.loc[frozen_roles.role.eq("calibration"), "cycle_id"].tolist() == bundle["calibration_ids"],
            "Expanded model fit/calibration ledger differs")
    old_sessions = set(ledger.session_id) | set(frozen_roles.session_id) | set(previous.session_id)
    old_sources = {Path(value).resolve() for series in (ledger.source, frozen_roles.source, previous.source)
                   for value in series if value}
    old_raw_hashes = {key: set(ledger[key]) for key in RAW_FIELDS}
    for raw_path, checksum in ver["input_sha256"].items():
        for key, filename in RAW_FIELDS.items():
            if raw_path.replace("\\", "/").endswith("/" + filename):
                old_raw_hashes[key].add(checksum)
    return_plan = inputs.json(study / "CAPTURE_PLAN.json")
    return_sha256 = return_plan["capture_policy"]["return_sha256"]
    freeze_after = max(capture_utc(frozen["frozen_utc"]),
                       *(capture_utc(stage["created_utc"]) for stage in ver["stages"]))
    return common, inputs, plan, features, model, old_sessions, old_sources, old_raw_hashes, return_sha256, freeze_after


def validate_windows(source, session, rows, inputs):
    completed = {}
    with inputs.track(source / "events.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("kind") != "cycle_completed":
                continue
            index = event["cycle_index"]
            require(event["session_id"] == session and index not in completed,
                    f"{session}: duplicate or foreign completion event")
            result = event["result"]
            completed[index] = (result["No3_write_completed_ns"], result["No3_confirmed_ns"])
    accepted = {row.cycle_index: (row.No3_begin_ns, row.No3_end_ns) for row in rows.itertuples(index=False)}
    require(completed == accepted, f"{session}: No.3 windows differ from recorded completed events")


def collect_manifest(campaign, common, inputs, plan, old_sessions, old_sources, old_raw_hashes,
                     return_sha256, freeze_after):
    missing = [
        str(path.relative_to(campaign))
        for stage, _, _ in SEQUENCE
        for path in (
            campaign / stage / f"{stage}_OPERATOR_REVIEW.json",
            campaign / stage / f"{stage}_EVALUATION_MANIFEST.csv",
            campaign / stage / "analysis" / stage / "audit.json",
        )
        if not path.is_file()
    ]
    require(not missing, "Campaign lacks reviewed blocks: " + ", ".join(missing))
    collected = []
    sessions, sources = set(old_sessions), set(old_sources)
    seen_raw_hashes = {key: set(values) for key, values in old_raw_hashes.items()}
    for order, (stage, label, count) in enumerate(SEQUENCE):
        folder = campaign / stage
        audit_folder = folder / "analysis" / stage
        review = inputs.json(folder / f"{stage}_OPERATOR_REVIEW.json")
        audit_path = audit_folder / "audit.json"
        audit = inputs.json(audit_path)
        execution = inputs.json(folder / f"EXECUTION_USED_{stage}.json")
        next_stage = SEQUENCE[order + 1][0] if order + 1 < len(SEQUENCE) else "EVALUATION"
        session = review["session_id"]
        require(session and session not in sessions and review["accepted_cycle_n"] == count
                and review["operator_confirmation"].strip()
                and review["capture_plan_sha256"] == PLAN_SHA256
                and review["audit_sha256"] == inputs.hashes[str(audit_path)]
                and review["next_stage"] == next_stage,
                f"{stage}: operator review or audit checksum differs")
        sessions.add(session)
        pair = "" if label == "normal" else label
        expected = {
            **plan["conditions"][label],
            "condition_id": f"{CAMPAIGN}_{stage}", "campaign_id": CAMPAIGN,
            "label": label, "pair_condition": pair, "automatic_return": True,
            "object_present": True, "dataset_role": "evaluation_only", "training_registration": False,
            "accelerometer_mount": "bolt_and_nut_user_reported", "base_mount": "rigid_fixed_operator_reported",
        }
        require(audit["stage"] == stage and audit["cycles"] == audit["quality_valid"] == count
                and audit["models_updated"] is False and audit["thresholds_updated"] is False
                and all(audit[flag] is True for flag in AUDIT_FLAGS)
                and all(audit["condition"].get(key) == value for key, value in expected.items())
                and all(audit[key] == 0 for key in ("position_missing", "position_timeout", "ADXL_encoding_bad"))
                and all(audit[key] == {"missing": 0, "duplicates": 0, "resets": 0}
                        for key in ("sequence_INA", "sequence_ADXL")),
                f"{stage}: quality or physical condition did not match the fixed plan")
        require(execution["capture_plan_sha256"] == PLAN_SHA256 and execution["model_sha256"] == MODEL_SHA256
                and execution["conditions"] == audit["condition"] and execution["folder"] == audit["source"],
                f"{stage}: recorded execution differs from the audited capture")
        source = Path(audit["source"])
        require(source.is_absolute(), f"{stage}: raw source must be absolute")
        source = source.resolve()
        require(source.is_dir() and source.is_relative_to(folder / "runs") and source not in sources,
                f"{stage}: invalid or previously used raw source")
        sources.add(source)

        manifest_path = folder / f"{stage}_EVALUATION_MANIFEST.csv"
        rows = inputs.csv(manifest_path, dtype=str, keep_default_na=False)
        require(review["manifest_sha256"] == inputs.hashes[str(manifest_path)]
                and tuple(rows.columns) == FIELDS and len(rows) == count
                and rows.session_id.eq(session).all() and rows.source.eq(audit["source"]).all()
                and rows.quality_valid.str.lower().eq("true").all() and rows.label.eq(label).all()
                and rows.pair_condition.eq(pair).all() and rows.campaign_id.eq(CAMPAIGN).all()
                and rows.role.eq("evaluation").all(), f"{stage}: evaluation manifest differs")
        for field in ("cycle_index", "No3_begin_ns", "No3_end_ns"):
            require(rows[field].str.fullmatch(r"[0-9]+").all(), f"{stage}: invalid numeric field {field}")
            rows[field] = rows[field].astype(np.int64)
        rows = rows.sort_values("cycle_index").reset_index(drop=True)
        require(rows.cycle_index.tolist() == list(range(1, count + 1))
                and rows.cycle_id.eq(session + "_" + rows.cycle_index.astype(str)).all()
                and (rows.No3_begin_ns > 0).all()
                and (rows.No3_end_ns > rows.No3_begin_ns).all()
                and (rows.No3_begin_ns > rows.No3_end_ns.shift(fill_value=0)).all(),
                f"{stage}: invalid or overlapping No.3 cycle windows")
        raw_manifest = inputs.json(source / "manifest.json")
        created = capture_utc(raw_manifest["created_utc"])
        require(raw_manifest["session_id"] == session and raw_manifest["source_sha256"] == return_sha256
                and freeze_after < created <= capture_utc(review["confirmed_utc"])
                and all(raw_manifest["conditions"].get(key) == value for key, value in expected.items()),
                f"{stage}: raw capture manifest differs")
        raw_evidence = inputs.json(audit_folder / "source_hashes_before.json")
        raw_evidence = {key.replace("\\", "/"): value for key, value in raw_evidence.items()}
        checked = ("manifest.json", "events.jsonl", *RAW_FIELDS.values())
        for filename in checked:
            require(common.sha(inputs.track(source / filename)) == raw_evidence[filename],
                    f"{stage}: raw {filename} differs from its audit")
        for hash_key, filename in RAW_FIELDS.items():
            checksum = raw_evidence[filename]
            require(checksum not in seen_raw_hashes[hash_key], f"{stage}: duplicate {filename} bytes")
            seen_raw_hashes[hash_key].add(checksum)
        for filename in ("sensors/measurements.csv", "sensors/adxl345_1.csv"):
            sensor_ids = inputs.csv(source / filename, usecols=["session_id"], dtype=str).session_id
            require(not sensor_ids.empty and sensor_ids.eq(session).all(), f"{stage}: {filename} session differs")
        validate_windows(source, session, rows, inputs)
        collected.append(rows.assign(stage_id=stage))
    data = pd.concat(collected, ignore_index=True)
    require(data.cycle_id.is_unique and data.label.value_counts().to_dict() == {"normal": 30, "B82": 30, "C182": 30},
            "Campaign does not contain the 90 expected unique evaluation cycles")
    return data


def score_once(features, model, table):
    x = table[features]
    values = x.to_numpy(float)
    accepted = np.isfinite(values)
    optional_index = x.columns.get_loc(OPTIONAL)
    accepted[:, optional_index] |= np.isnan(values[:, optional_index])
    require(accepted.all(), "Nonfinite required feature")
    with threadpool_limits(limits=1):
        predictions = pd.DataFrame(model.predict_proba(x), columns=model.classes_, index=x.index)
    require(np.isfinite(predictions).all().all(), "Nonfinite model probability")
    score = 1 - predictions.normal
    load_type = predictions[list(LOAD_TYPES)].idxmax(axis=1)
    rows = table[["cycle_id", "session_id", "cycle_index", "stage_id", "label", "pair_condition", "campaign_id"]].copy()
    rows = rows.assign(model_id=MODEL_ID, score=score, missing_id1_iqr=x[OPTIONAL].isna()).join(predictions.add_prefix("p_"))
    for key, threshold in THRESHOLDS.items():
        rows[f"{key}_threshold"] = threshold
        rows[f"{key}_alarm"] = score > threshold
        rows[f"{key}_type"] = load_type.where(rows[f"{key}_alarm"], "normal")
    require(not (rows.stored_alarm & ~rows.candidate_alarm).any(), "Lower threshold lost a stored alarm")
    rows["transition"] = np.select([rows.stored_alarm, rows.candidate_alarm],
                                   ["both_alarm", "candidate_only"], default="neither")
    return rows


def summarize(rows, scope):
    metrics = []
    for key, threshold in THRESHOLDS.items():
        alarm, predicted = f"{key}_alarm", f"{key}_type"
        for group, block in rows.groupby("stage_id" if scope == "block" else "label", sort=False):
            label = block.label.iloc[0]
            count = int(block[alarm].sum())
            metrics.append(dict(
                scope=scope, group=group, actual=label, threshold_name=key, threshold=threshold,
                count=len(block), alarms=count, rate=count / len(block),
                metric="FPR" if label == "normal" else "TPR",
                correct_type=int(block[predicted].eq(label).sum()) if label != "normal" else None,
                score_p10=block.score.quantile(.1), score_median=block.score.median(), score_p90=block.score.quantile(.9),
            ))
    return pd.DataFrame(metrics)


def independent_ranking(rows):
    normal = rows.loc[rows.label.eq("normal")]
    records = []
    for label in ("B82", "C182"):
        load = rows.loc[rows.label.eq(label)]
        combined = pd.concat([normal, load], ignore_index=True)
        target = combined.label.eq(label)
        records.append(dict(label=label, common_normal_n=len(normal), load_n=len(load),
                            auroc=roc_auc_score(target, combined.score),
                            average_precision=average_precision_score(target, combined.score)))
    return pd.DataFrame(records)


def evaluate(project, campaign, output, verify_only):
    common, inputs, plan, features, model, *provenance = load_contract(project, campaign)
    data = collect_manifest(campaign, common, inputs, plan, *provenance)
    if verify_only:
        require(inputs.unchanged(), "An input changed during verification")
        print("Verified 7 reviewed blocks and 90 new evaluation cycles.")
        return
    require(not output.exists(), f"Evaluation output already exists: {output}")
    table, quality = common.extract_features(data, project, inputs)
    require(table.cycle_id.tolist() == data.cycle_id.tolist(), "Cycle order changed during extraction")
    scores = score_once(features, model, table)
    quality = quality.merge(scores[["cycle_id", "stage_id", "missing_id1_iqr"]], on="cycle_id", validate="one_to_one")
    block = summarize(scores, "block")
    pooled = summarize(scores, "pooled")
    thresholds = tuple(THRESHOLDS)
    labels = ["normal", *LOAD_TYPES]
    confusion = pd.concat([
        pd.crosstab(scores.label, scores[f"{key}_type"])
        .reindex(index=["normal", "B82", "C182"], columns=labels, fill_value=0)
        .rename_axis(index="actual", columns="predicted")
        .stack().rename("count").reset_index().assign(threshold_name=key)
        for key in thresholds
    ], ignore_index=True)
    transitions = scores.groupby(["stage_id", "label", "transition"], sort=False).size().rename("count").reset_index()
    totals = scores.groupby(["label", "transition"], sort=False).size().rename("count").reset_index()
    ranking = independent_ranking(scores)
    metadata = dict(
        campaign_id=CAMPAIGN, plan_sha256=PLAN_SHA256, model_id=MODEL_ID, model_sha256=MODEL_SHA256,
        threshold_values=THRESHOLDS, rule=plan["alarm_rule"], n=len(scores),
        class_counts=scores.label.value_counts().to_dict(), sessions=int(scores.session_id.nunique()),
        common_normal_n=30, missing_id1_iqr_n=int(scores.missing_id1_iqr.sum()),
        models_updated=False, thresholds_updated=False, training_registration=False,
        input_sha256=inputs.hashes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".fixed-bc-") as temp:
        staged = Path(temp) / "evaluation"
        staged.mkdir()
        for name, frame in (
            ("EVALUATION_MANIFEST.csv", data), ("CYCLE_FEATURES.csv", table), ("SAMPLE_QUALITY.csv", quality),
            ("CYCLE_SCORES.csv", scores), ("BLOCK_METRICS.csv", block), ("POOLED_METRICS.csv", pooled),
            ("TYPE_CONFUSION.csv", confusion), ("BLOCK_TRANSITIONS.csv", transitions),
            ("POOLED_TRANSITIONS.csv", totals), ("COMMON_NORMAL_RANKING.csv", ranking),
        ):
            common.savecsv(staged / name, frame)
        require(inputs.unchanged(), "An input changed during fixed-threshold evaluation")
        common.savej(staged / "VERIFICATION.json", metadata)
        staged.rename(output)
    print(pooled[["threshold_name", "actual", "alarms", "count", "metric", "rate", "correct_type"]].to_string(index=False))
    print(f"Saved fixed-threshold scores and evidence for {len(scores)} cycles to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only == (args.output is not None):
        parser.error("Supply either --verify-only or --output")
    project = args.project.resolve()
    campaign = (args.campaign or project / "hardware" / CAMPAIGN).resolve()
    evaluate(project, campaign, args.output.resolve() if args.output else None, args.verify_only)


if __name__ == "__main__":
    main()
