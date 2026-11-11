"""Score cycle features with a saved xArm follow-up model."""

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


MODELS = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS / "model_runtime"))


def read_features(path):
    return pd.read_csv(path, float_precision="round_trip", encoding="utf-8-sig")


def score(bundle, cycles):
    features = cycles.loc[:, bundle["features"]]
    with threadpool_limits(limits=1):
        if "detector" in bundle:
            values = bundle["detector"].score(features.to_numpy(dtype=float))
        elif "model" in bundle:
            model = bundle["model"]
            probabilities = model.predict_proba(features)
            classes = model.classes_.tolist()
            if bundle.get("positive_label") == "known_load":
                values = probabilities[:, classes.index(True)]
            else:
                values = 1 - probabilities[:, classes.index("normal")]
        else:
            estimator = bundle["estimator"]
            scaled = bundle["scaler"].transform(bundle["imputer"].transform(features.to_numpy(dtype=float)))
            kind = bundle.get("kind")
            if kind == "knn_mean":
                values = estimator.kneighbors(scaled, return_distance=True)[0].mean(axis=1)
            elif kind == "pca_q":
                reconstructed = estimator.inverse_transform(estimator.transform(scaled))
                values = np.mean((scaled - reconstructed) ** 2, axis=1)
            else:
                pca = bundle.get("pca")
                if pca is not None:
                    coordinates = pca.transform(scaled)
                    reconstructed = pca.inverse_transform(coordinates)
                    residual = np.mean((scaled - reconstructed) ** 2, axis=1)
                    scaled = np.column_stack((coordinates, residual))
                values = -estimator.score_samples(scaled)
    if values.shape != (len(cycles),) or not np.isfinite(values).all():
        raise ValueError("Model returned invalid scores")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Trusted joblib bundle")
    parser.add_argument("features", type=Path, help="CSV with cycle_id and the model feature columns")
    parser.add_argument("output", type=Path, help="New score CSV")
    parser.add_argument("--threshold", type=float, help="Optional comparison threshold")
    args = parser.parse_args()

    cycles = read_features(args.features)
    if cycles.empty or cycles["cycle_id"].isna().any() or cycles["cycle_id"].duplicated().any():
        raise ValueError("Expected at least one cycle and unique, nonempty cycle_id values")
    bundle = joblib.load(args.model)
    threshold = bundle["threshold"] if args.threshold is None else args.threshold
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    values = score(bundle, cycles)
    output = pd.DataFrame({"cycle_id": cycles["cycle_id"], "score": values,
                           "threshold": threshold, "alarm": values > threshold})
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        output.to_csv(stream, index=False)


if __name__ == "__main__":
    main()
