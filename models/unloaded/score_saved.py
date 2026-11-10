"""Score completed-cycle inputs with the archived xArm unloaded models."""

import argparse
import csv
import json
from pathlib import Path
import sys
from threading import Lock

import numpy as np

from normal_models import Detector


_PICKLE_LOCK = Lock()
_MISSING = object()


def load_classical(path):
    """Load an original joblib from this trusted archive only."""
    import joblib

    # The single-action trainer serialized Detector as __main__.Detector.
    # Make that name resolve without changing the saved files.
    with _PICKLE_LOCK:
        main = sys.modules["__main__"]
        previous = getattr(main, "Detector", _MISSING)
        main.Detector = Detector
        try:
            return joblib.load(path)
        finally:
            if previous is _MISSING:
                del main.Detector
            else:
                main.Detector = previous


def score_classical(bundle, frame):
    values = frame.loc[:, bundle["features"]].to_numpy(dtype=float)
    if "detector" in bundle:
        scores = bundle["detector"].score(values)
        return scores, np.full(len(scores), bundle["threshold"])

    groups = frame["action_group_id"].to_numpy()
    if not set(groups).issubset(bundle["groups"]):
        raise ValueError("The archive supports only its fitted action groups")
    preprocessors = bundle["group_preprocessors"]
    if preprocessors is not None:
        for group in np.unique(groups):
            keep = groups == group
            imputer, scaler = preprocessors[group]
            values[keep] = scaler.transform(imputer.transform(values[keep]))

    if bundle["strategy"] != "independent_per_group":
        scores = bundle["detectors"]["pooled"].score(values)
        return scores, np.full(len(scores), bundle["thresholds"]["pooled"])

    scores = np.empty(len(values))
    limits = np.empty(len(values))
    for group in np.unique(groups):
        keep = groups == group
        scores[keep] = bundle["detectors"][group].score(values[keep])
        limits[keep] = bundle["thresholds"][group]
    return scores, limits


def score_autoencoder(directory, values):
    import torch

    from ae_models import Normalizer, make_model, score

    directory = Path(directory)
    config = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    expected = tuple(config["shape"])
    if values.ndim != len(expected) + 1 or values.shape[1:] != expected or not len(values):
        raise ValueError(f"Expected an array with shape (cycles, {', '.join(map(str, expected))})")
    normalizer = Normalizer()
    with np.load(directory / "normalizer.npz", allow_pickle=False) as params:
        normalizer.fill = params["fill"]
        normalizer.center = params["center"]
        normalizer.scale = params["scale"]
        normalizer.sequence = bool(params["sequence"])
    standardized, mask = normalizer.transform(values)
    model = make_model(config["architecture"], expected, config["cfg"])
    weights = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    scores = score(model, standardized, mask)
    return scores, np.full(len(scores), config["threshold"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("classic", "ae"))
    parser.add_argument("model", type=Path, help="Trusted joblib or autoencoder directory")
    parser.add_argument("input", type=Path, help="Feature CSV (classic) or NPY tensor (AE)")
    args = parser.parse_args()
    if args.kind == "classic":
        import pandas as pd

        scores, limits = score_classical(load_classical(args.model), pd.read_csv(args.input))
    else:
        import torch

        torch.set_num_threads(1)
        scores, limits = score_autoencoder(args.model, np.load(args.input, allow_pickle=False))
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("row_index", "score", "threshold", "flagged"))
    for index, (value, limit) in enumerate(zip(scores, limits, strict=True)):
        writer.writerow((index, value, limit, int(value > limit)))


if __name__ == "__main__":
    main()
