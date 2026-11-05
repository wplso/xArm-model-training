"""Train the original V3 configurations from a cycle-feature CSV (offline only)."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from v3.detectors import Detector, threshold

ROOT = Path(__file__).resolve().parent
FORBIDDEN = {"cycle_id", "session_id", "cohort", "stage", "classification", "anomaly",
             "category", "setting", "sample", "payload_mass_g", "frame_added_mass_g",
             "source", "start_pose", "automatic_return", "condition_id"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(data, config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; existing models are never overwritten")
    if not {"cycle_id", "cohort"}.issubset(data):
        raise ValueError("cycle_id and cohort columns are required")
    if data.cycle_id.isna().any() or data.cycle_id.duplicated().any() or data.cohort.isna().any():
        raise ValueError("Unique, nonempty cycle IDs and nonempty split labels are required")
    training, validation = data.cohort.eq("train"), data.cohort.eq("validation")
    if training.sum() < 2 or validation.sum() < 1:
        raise ValueError("Need separate normal train and validation cycles")
    if "classification" in data:
        if not data.loc[training | validation, "classification"].eq("confirmed_normal").all():
            raise ValueError("Training and validation must be confirmed normal")
    for names in config["configurations"].values():
        if FORBIDDEN.intersection(names) or len(names) != len(set(names)):
            raise ValueError("Metadata or duplicate columns in predictor whitelist")
        if set(names) - set(data):
            raise ValueError("Missing feature columns: " + str(sorted(set(names) - set(data))))
        matrix = data[names].to_numpy(float)
        if np.isinf(matrix).any():
            raise ValueError("Infinite input is invalid; missing features must be NaN")
    if "V3-C-fall" in config["configurations"]:
        for mask in [training, validation]:
            if data.loc[mask, "S5_current_fall_time"].notna().mean() < .9:
                raise ValueError("Original fall configuration requires >=90% train/validation coverage")
    output.mkdir(parents=True)
    (output / "models").mkdir()
    indices, score_tables, summaries = [], [], []
    for group, names in config["configurations"].items():
        matrix = data[names].to_numpy(float)
        for algorithm in config["algorithms"]:
            model_id = f"{group}_{algorithm}"
            begin = time.perf_counter()
            detector = Detector(algorithm, config["seed"]).fit(matrix[training])
            fit_s = time.perf_counter() - begin
            values = detector.score(matrix)
            cut = threshold(values[validation])
            # Verify fitting boundaries rather than learning preprocessing from evaluation data.
            np.testing.assert_allclose(detector.imputer.statistics_, np.nanmedian(matrix[training], axis=0))
            np.testing.assert_allclose(detector.scaler.mean_, detector.imputer.transform(matrix[training]).mean(axis=0))
            bundle = dict(detector=detector, features=names, threshold=cut, seed=config["seed"],
                          version="V3", configuration=group,
                          training_cycle_ids=data.loc[training, "cycle_id"].tolist(),
                          validation_cycle_ids=data.loc[validation, "cycle_id"].tolist(),
                          evaluation_cycle_ids=data.loc[~(training | validation), "cycle_id"].tolist())
            path = output / "models" / f"{model_id}.joblib"
            joblib.dump(bundle, path)
            reopened = joblib.load(path)
            np.testing.assert_array_equal(values, reopened["detector"].score(matrix))
            np.testing.assert_array_equal(values > cut, reopened["detector"].score(matrix) > reopened["threshold"])
            rank = int(detector.model.n_components_) if algorithm == "PCA" else None
            indices.append(dict(model_id=model_id, config_id=group, algorithm=algorithm,
                features=";".join(names), threshold=cut, train_n=int(training.sum()),
                validation_n=int(validation.sum()), PCA_rank=rank, invalid_PCA=rank == len(names),
                fit_s=fit_s, sha256=sha(path), path=str(path.relative_to(output))))
            table = data[["cycle_id", "cohort"]].assign(model_id=model_id, score=values,
                                                       threshold=cut, flagged=values > cut)
            score_tables.append(table)
            for cohort, rows in table.groupby("cohort", sort=False):
                summaries.append(dict(model_id=model_id, cohort=cohort, n=len(rows),
                    flagged=int(rows.flagged.sum()), flag_rate=float(rows.flagged.mean()),
                    score_median=float(rows.score.median()), score_q10=float(rows.score.quantile(.1)),
                    score_q90=float(rows.score.quantile(.9)), threshold=cut))
    pd.DataFrame(indices).to_csv(output / "MODEL_INDEX.csv", index=False, encoding="utf-8-sig")
    pd.concat(score_tables).to_csv(output / "CYCLE_SCORES.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output / "SCORE_SUMMARIES.csv", index=False, encoding="utf-8-sig")
    (output / "EXPERIMENT.json").write_text(json.dumps(dict(config=config,
        python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
        counts=data.cohort.value_counts().to_dict(), hardware_access=0,
        threshold_rule="normal validation quantile95 higher; score > threshold",
        preprocessing_fit="normal train only", original_data_supplied_by_user=True), indent=2), encoding="utf-8")
    return pd.DataFrame(indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/V3_TRAINING_CONFIG.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = pd.read_csv(args.features, float_precision="round_trip")
    index = train(data, config, args.output)
    (args.output / "INPUT_HASHES.json").write_text(json.dumps(dict(features_sha256=sha(args.features),
        config_sha256=sha(args.config), training_entrypoint_sha256=sha(__file__)), indent=2), encoding="utf-8")
    print(f"Saved {len(index)} V3 models to {args.output}")


if __name__ == "__main__":
    main()
