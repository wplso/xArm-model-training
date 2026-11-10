"""Verify archive files and exercise the saved detectors against the included features."""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(ARCHIVE / "model_runtime"))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rows(name):
    with (ARCHIVE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "models":
        raise ValueError(f"Invalid archive path {value!r}")
    return ARCHIVE.parent / path


def thresholds(bundle):
    stored = bundle["thresholds"] if "thresholds" in bundle else {"default": bundle["threshold"]}
    return {k: float(v) if v is not None and math.isfinite(v) else None for k, v in stored.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums-only", action="store_true", help="verify hashes without opening joblib files")
    parser.add_argument("--exact-scores", action="store_true", help="require bitwise detector score parity in the original runtime")
    args = parser.parse_args()
    artifacts = read_rows("ARTIFACT_SHA256.csv")
    for item in artifacts:
        path = relative_path(item["path"])
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise ValueError(f"Artifact hash mismatch {item['path']}")
    if args.checksums_only:
        print(json.dumps({"verified_artifacts": len(artifacts)}))
        return
    data, version_count, score_count, exact_score_count = {}, Counter(), 0, 0
    for item in read_rows("MANIFEST.csv"):
        version = item["version"]
        path = relative_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Model hash mismatch {path.name}")
        bundle = joblib.load(path)
        if bundle["features"] != json.loads(item["features_json"]) or thresholds(bundle) != json.loads(item["thresholds_json"]):
            raise ValueError(f"Saved schema or threshold mismatch {path.name}")
        if version not in data:
            data[version] = pd.read_csv(ARCHIVE / version / "CYCLE_FEATURES.csv", float_precision="round_trip")
        features = data[version][bundle["features"]].to_numpy(float)
        if len(features) != int(item["checked_feature_rows"]):
            raise ValueError(f"Feature count mismatch {path.name}")
        scores = np.asarray(bundle["detector"].score(features), dtype="<f8")
        if scores.shape != (len(features),) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid detector scores {path.name}")
        score_hash = hashlib.sha256(scores.tobytes()).hexdigest()
        exact = score_hash == item["raw_detector_scores_sha256"]
        if args.exact_scores and not exact:
            raise ValueError(f"Bitwise score mismatch {path.name}")
        exact_score_count += int(exact)
        score_count += len(scores)
        version_count[version] += 1
    print(json.dumps({"verified_artifacts": len(artifacts), "verified_models": sum(version_count.values()),
                      "models_by_version": version_count, "checked_scores": score_count,
                      "exact_score_models": exact_score_count}))


if __name__ == "__main__":
    main()
