"""Evaluate the complete, operator-reviewed interleaved bolt campaign offline."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


CAMPAIGN_ID = "No3_bolt_ABC_interleaved_20261111"
PLAN_SHA256 = "62e636da0eefe115e3e47b737e03e20690b167e3c67afa0c210bf749d3925870"
STUDY = Path("hardware/xarm_model_versions/bolt_known_load_prospective_20261111")
MODELS = ("V4_PCA_Q", "LOGISTIC_MONSTER_GEL", "LOGISTIC_BOLT")
MANIFEST_FIELDS = (
    "cycle_id", "session_id", "cycle_index", "source", "No3_begin_ns", "No3_end_ns",
    "quality_valid", "label", "pair_condition", "campaign_id", "role",
)
AUDIT_FLAGS = (
    "raw_uart_csv_identical", "raw_hid_csv_identical", "backup_sha256_verified",
    "backup_CRC_verified", "backup_members_identical", "source_hashes_unchanged",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_manifest(campaign, frozen_manifest):
    plan_path = campaign / "CAPTURE_PLAN.json"
    if digest(plan_path) != PLAN_SHA256:
        raise ValueError("Capture plan differs from the pre-acquisition plan")
    plan = read_json(plan_path)
    if plan["campaign_id"] != CAMPAIGN_ID or len(plan["sequence"]) != 13:
        raise ValueError("Expected the 13-block interleaved bolt campaign")
    if digest(frozen_manifest) != plan["frozen_models_sha256"]:
        raise ValueError("Frozen model manifest differs from the capture plan")

    blocks = {}
    for stage in plan["sequence"]:
        block = stage["id"]
        folder = campaign if block == "N0" else campaign / block
        audit_stage = "normal30" if block == "N0" else block
        blocks[block] = (
            folder / f"{block}_OPERATOR_REVIEW.json",
            folder / "analysis" / audit_stage / "audit.json",
            folder / f"{block}_EVALUATION_MANIFEST.csv",
            audit_stage,
        )
    missing = [f"{block}: {path.name}" for block, files in blocks.items()
               for path in files[:3] if not path.is_file()]
    if missing:
        raise ValueError("Incomplete campaign: " + ", ".join(missing))

    manifest, stage_map = [], []
    session_ids, sources = set(), set()
    evidence = {plan_path: PLAN_SHA256, frozen_manifest: plan["frozen_models_sha256"]}
    for order, stage in enumerate(plan["sequence"], 1):
        block, label, count = stage["id"], stage["label"], stage["count"]
        pair = stage.get("pair_condition", label)
        review_path, audit_path, manifest_path, audit_stage = blocks[block]
        evidence.update({path: digest(path) for path in (review_path, audit_path, manifest_path)})
        review, audit = read_json(review_path), read_json(audit_path)
        session = review.get("session_id")
        if (not session or session in session_ids or not review.get("confirmed_utc")
                or not review.get("operator_confirmation", "").strip()
                or review.get("accepted_cycle_n") != count
                or review.get("capture_plan_sha256") != PLAN_SHA256
                or review.get("audit_sha256") != evidence[audit_path]):
            raise ValueError(f"{block}: missing or inconsistent operator review")
        session_ids.add(session)

        condition = {
            **plan["conditions"][label],
            "condition_id": f"{CAMPAIGN_ID}_{block}",
            "campaign_id": CAMPAIGN_ID, "label": label, "pair_condition": pair,
            "dataset_role": "evaluation_only", "training_registration": False,
            "automatic_return": True, "object_present": True,
            "accelerometer_mount": "bolt_and_nut_user_reported",
            "base_mount": "rigid_fixed_operator_reported",
        }
        if (audit.get("stage") != audit_stage or audit.get("cycles") != count
                or audit.get("quality_valid") != count
                or audit.get("models_updated") is not False
                or audit.get("thresholds_updated") is not False
                or any(audit.get(flag) is not True for flag in AUDIT_FLAGS)
                or any(audit.get("condition", {}).get(key) != value
                       for key, value in condition.items())):
            raise ValueError(f"{block}: data quality or physical condition audit did not match the plan")
        source = Path(audit.get("source", ""))
        if not source.is_absolute() or not source.is_dir() or source.resolve() in sources:
            raise ValueError(f"{block}: invalid or duplicate audited source")
        sources.add(source.resolve())

        with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise ValueError(f"{block}: unexpected evaluation manifest columns")
            rows = list(reader)
        if len(rows) != count:
            raise ValueError(f"{block}: expected {count} rows, found {len(rows)}")
        expected = {
            "session_id": session, "source": audit["source"], "quality_valid": "True",
            "label": label, "pair_condition": pair, "campaign_id": CAMPAIGN_ID, "role": "evaluation",
        }
        indices = set()
        for row in rows:
            index = int(row["cycle_index"])
            cycle_id = row["cycle_id"]
            if (None in row or index in indices or not 1 <= index <= count
                    or cycle_id != f"{session}_{index}"
                    or any(row.get(key) != value for key, value in expected.items())
                    or not 0 < int(row["No3_begin_ns"]) < int(row["No3_end_ns"])):
                raise ValueError(f"{block}: invalid evaluation cycle {cycle_id}")
            indices.add(index)
            stage_map.append({"cycle_id": cycle_id, "stage_id": block, "stage_order": order})
        manifest.extend(rows)
    return plan, manifest, stage_map, evidence


def neighboring_metrics(scores_path, plan, manifest, stage_map):
    cycles = {row["cycle_id"]: row for row in manifest}
    stages = {row["cycle_id"]: row["stage_id"] for row in stage_map}
    block_alarms = {(model, stage["id"]): [] for model in MODELS for stage in plan["sequence"]}
    seen = set()
    with scores_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            model, cycle = row["model_id"], row["cycle_id"]
            if (model not in MODELS or cycle not in cycles or (model, cycle) in seen
                    or row["alarm"] not in ("True", "False")
                    or any(row[key] != cycles[cycle][key]
                           for key in ("session_id", "label", "pair_condition", "campaign_id"))):
                raise ValueError(f"Invalid frozen score key, labels or alarm: {model}, {cycle}")
            seen.add((model, cycle))
            block_alarms[model, stages[cycle]].append(row["alarm"] == "True")
    if len(seen) != len(MODELS) * len(cycles):
        raise ValueError("Frozen score file lacks one or more model-cycle scores")

    sequence = plan["sequence"]
    metrics = []
    for model in MODELS:
        for order, stage in enumerate(sequence):
            if stage["label"] == "normal":
                continue
            block = stage["id"]
            alarms = block_alarms[model, block]
            result = {"model_id": model, "load_block": block, "load_label": stage["label"],
                      "tp": sum(alarms), "load_n": len(alarms), "tpr": sum(alarms) / len(alarms)}
            for side, neighbor in (("before", sequence[order - 1]), ("after", sequence[order + 1])):
                normal_block = neighbor["id"]
                normal_alarms = block_alarms[model, normal_block]
                result.update({f"{side}_normal_block": normal_block,
                               f"{side}_fp": sum(normal_alarms), f"{side}_normal_n": len(normal_alarms),
                               f"{side}_fpr": sum(normal_alarms) / len(normal_alarms)})
            metrics.append(result)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="Check all 13 blocks without scoring")
    args = parser.parse_args()
    if args.verify_only == (args.output is not None):
        parser.error("Supply either --verify-only or --output")
    project = args.project.resolve()
    campaign = (args.campaign or project / "hardware" / CAMPAIGN_ID).resolve()
    study = project / STUDY
    plan, manifest, stage_map, evidence = collect_manifest(campaign, study / "FROZEN_MODELS.json")
    if args.verify_only:
        print(f"Verified {len(plan['sequence'])} reviewed blocks and {len(manifest)} evaluation cycles.")
        return

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".interleaved-") as temporary:
        staging = Path(temporary)
        combined = staging / "EVALUATION_MANIFEST.csv"
        results = staging / "results"
        write_csv(combined, MANIFEST_FIELDS, manifest)
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(study / "evaluate_frozen.py"),
                        "--project", str(project), "--manifest", str(combined), "--output", str(results)],
                       check=True)
        metrics = neighboring_metrics(results / "CYCLE_SCORES.csv", plan, manifest, stage_map)
        write_csv(results / "LOAD_BLOCK_AND_NEIGHBOR_METRICS.csv", metrics[0].keys(), metrics)
        write_csv(results / "CYCLE_STAGE_MAP.csv", stage_map[0].keys(), stage_map)
        shutil.copyfile(combined, results / combined.name)
        if any(digest(path) != expected for path, expected in evidence.items()):
            raise ValueError("Capture plan, frozen manifest or reviewed block evidence changed during evaluation")
        (results / "CAMPAIGN_EVIDENCE_SHA256.json").write_text(
            json.dumps({str(path): value for path, value in evidence.items()}, indent=2) + "\n", encoding="utf-8",
        )
        results.rename(output)
    print(f"Saved frozen evaluation for {len(manifest)} cycles to {output}")


if __name__ == "__main__":
    main()
