"""Replay audited bolt features and compare source bytes before/after the experiment."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


OLD_VERSIONS = ("v4_bolt_fixed", "v5_position_score_study", "v5_1_multisession_normal_study")
RAW_FILES = {
    "raw_uart_sha256": "sensors/raw_rx.bin",
    "raw_hid_sha256": "raw/robot_hid_input.bin",
    "ina_csv_sha256": "sensors/measurements.csv",
    "adxl_csv_sha256": "sensors/adxl345_1.csv",
    "position_csv_sha256": "robot/single_queries.csv",
}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def collect_sources(root, versions, captures):
    hashes = {}
    for version in OLD_VERSIONS:
        for path in (versions / version).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                hashes[path.relative_to(root).as_posix()] = sha256(path)
    for row in captures.itertuples(index=False):
        capture = Path(row.source).resolve(strict=True)
        for column, relative in RAW_FILES.items():
            path = capture / relative
            actual = sha256(path)
            if actual != getattr(row, column):
                raise ValueError(f"V5.1 raw hash mismatch: {path}")
            hashes[path.relative_to(root).as_posix()] = actual
        for relative in ("manifest.json", "events.jsonl", "summary.json", "sensors/quality_report.json"):
            path = capture / relative
            hashes[path.relative_to(root).as_posix()] = sha256(path)
    return dict(sorted(hashes.items()))


def verify_windows(cycles):
    for source, rows in cycles.groupby("source", sort=False):
        directory = Path(source)
        metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
        if metadata["conditions"]["accelerometer_mount"] != "bolt_and_nut_user_reported":
            raise ValueError(f"Unexpected accelerometer mount: {directory}")
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8-sig").splitlines()]
        complete = [event["result"] for event in events if event["kind"] == "cycle_completed"]
        windows = {
            f"{metadata['session_id']}_{item['cycle_index']}": (item["No3_write_completed_ns"], item["No3_confirmed_ns"])
            for item in complete
        }
        if len(complete) != len(windows) or set(rows.cycle_id) != windows.keys():
            raise ValueError(f"Completed cycle ID mismatch: {directory}")
        for row in rows.itertuples(index=False):
            if windows[row.cycle_id] != (row.No3_begin_ns, row.No3_end_ns):
                raise ValueError(f"No.3 window mismatch: {row.cycle_id}")


def replay(root, versions, output):
    old = versions / "v5_1_multisession_normal_study"
    cycles = pd.read_csv(old / "DATA_USAGE_MANIFEST.csv", float_precision="round_trip")
    expected = pd.read_csv(old / "metrics/CYCLE_FEATURES.csv", float_precision="round_trip").set_index("cycle_id")
    if len(cycles) != 649 or not cycles.cycle_id.is_unique or not expected.index.is_unique:
        raise ValueError("Expected 649 unique V5.1 bolt cycles")
    verify_windows(cycles)

    original = versions / "v4_bolt_fixed/analysis_code/v4_common.py"
    spec = importlib.util.spec_from_file_location("v52_frozen_feature_replay", original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extracted, quality = module.extract_features(cycles, root, module.Inputs())
    extracted = extracted.set_index("cycle_id")
    if not extracted.index.equals(expected.index):
        raise ValueError("Raw feature cycle alignment mismatch")
    frozen = json.loads((versions / "v4_bolt_fixed/FROZEN_V4_CANDIDATES.json").read_text(encoding="utf-8-sig"))
    features = list(dict.fromkeys(name for model in frozen["models"] for name in model["features"]))
    if len(features) != 31:
        raise ValueError("Unexpected original V4 feature count")
    actual_values = extracted[features].to_numpy(dtype=float)
    expected_values = expected[features].to_numpy(dtype=float)
    equal = np.isclose(actual_values, expected_values, atol=1e-10, rtol=1e-10, equal_nan=True)
    if not equal.all():
        unequal = np.argwhere(~equal)[0]
        raise ValueError(f"Raw replay differs: {extracted.index[unequal[0]]} / {features[unequal[1]]}")
    quality = quality.set_index("cycle_id").join(extracted[features]).reset_index()
    quality.to_csv(output / "RAW_FEATURE_REPLAY.csv", index=False, encoding="utf-8-sig")
    return {
        "matched_original_event_windows": len(cycles),
        "replayed_feature_count": len(features),
        "matched_feature_cells": int(equal.size),
        "matched_missing_cells": int(np.isnan(actual_values).sum()),
        "maximum_absolute_error": float(np.nanmax(np.abs(actual_values - expected_values))),
        "cohorts": dict(Counter(cycles.cohort)),
        "original_feature_extractor_sha256": sha256(original),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("mode", choices=("before", "after"))
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    versions = root / "hardware/xarm_model_versions"
    captures = pd.read_csv(versions / "v5_1_multisession_normal_study/metrics/CAPTURE_BOUNDARY_EVIDENCE.csv")
    if len(captures) != 18 or not captures.capture_uuid.is_unique or not captures.source.is_unique:
        raise ValueError("Expected 18 audited captures")
    hashes = collect_sources(root, versions, captures)
    initial = args.output / "SOURCE_PROVENANCE_BEFORE.json"
    if args.mode == "before":
        if initial.exists():
            raise FileExistsError(initial)
        report = {"files": hashes, "capture_count": len(captures), "raw_hashes_matched": len(captures) * len(RAW_FILES)}
        report.update(replay(root, versions, args.output))
        save_json(initial, report)
        print(f"{report['matched_feature_cells']} feature values verified, {len(hashes)} source hashes recorded")
        return
    before = json.loads(initial.read_text(encoding="utf-8"))
    changes = sorted(key for key in set(hashes) | set(before["files"]) if hashes.get(key) != before["files"].get(key))
    save_json(args.output / "SOURCE_PROVENANCE_AFTER.json", {
        "unchanged": not changes,
        "source_files_checked": len(hashes),
        "captures_checked": len(captures),
        "changed_paths": changes,
        "before_manifest_sha256": sha256(initial),
    })
    if changes:
        raise ValueError(f"Source changes: {changes}")
    print(f"All {len(hashes)} source files unchanged")


if __name__ == "__main__":
    main()
