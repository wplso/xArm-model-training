"""Describe held-out LOF/kNN neighborhoods using the frozen Train200 reference."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits


def alarm_label(is_anomaly, flagged):
    return np.select(
        [is_anomaly & flagged, is_anomaly, flagged], ["TP", "FN", "FP"], default="TN"
    )


def describe_neighbors(run_dir, project_root):
    source_path = (
        project_root / "hardware/xarm_model_versions/v5_1_multisession_normal_study"
        / "metrics/CYCLE_FEATURES.csv"
    )
    source = pd.read_csv(source_path, float_precision="round_trip").set_index("cycle_id")
    metrics_dir = run_dir / "metrics"
    scores = pd.read_csv(metrics_dir / "CYCLE_SCORES.csv", float_precision="round_trip")
    scores = scores.loc[(scores.split_id == "PRIMARY") & (scores.role == "evaluation")]
    knn = scores.loc[scores.model_id == "KNN_K20_MEAN"].set_index("cycle_id")
    lof = scores.loc[scores.model_id == "LOF_K20"].set_index("cycle_id")
    if not all(frame.index.is_unique for frame in (source, knn, lof)):
        raise ValueError("Duplicate cycle ID in features or evaluation scores")
    lof = lof.loc[knn.index]
    model = joblib.load(run_dir / "models/PRIMARY_KNN_K20_MEAN.joblib")
    train_ids = list(model["train_cycle_ids"])
    if len(train_ids) != 200 or len(knn) != 349 or set(train_ids) & set(knn.index):
        raise ValueError("Expected 200 train-only references and 349 disjoint evaluation IDs")

    train = source.loc[train_ids]
    feature_names = list(model["features"])
    reference = model["scaler"].transform(
        model["imputer"].transform(train[feature_names].to_numpy(float))
    )
    queries = source.loc[knn.index, feature_names].to_numpy(float)
    scaled = model["scaler"].transform(model["imputer"].transform(queries))
    estimator = model["estimator"]
    np.testing.assert_array_equal(reference, estimator._fit_X)
    with threadpool_limits(limits=1):
        distances, indices = estimator.kneighbors(scaled)
    np.testing.assert_allclose(distances.mean(axis=1), knn.score, atol=1e-10, rtol=1e-10)

    blocks = train.block_id.to_numpy()[indices]
    cycle_order = train.session_cycle_index.to_numpy()[indices]
    row = knn[["session_id", "session_cycle_index", "cohort", "condition", "pair_condition"]].copy()
    row["knn_score"] = knn.score
    row["knn_margin"] = knn.margin
    row["lof_score"] = lof.score
    row["lof_margin"] = lof.margin
    is_anomaly = row.condition.ne("normal").to_numpy()
    row["knn_outcome"] = alarm_label(is_anomaly, knn.margin.gt(0).to_numpy())
    row["lof_outcome"] = alarm_label(is_anomaly, lof.margin.gt(0).to_numpy())
    neighbor_ids = np.asarray(train_ids)[indices]
    row["nearest_train_cycle_id"] = neighbor_ids[:, 0]
    row["nearest_train_block"] = blocks[:, 0]
    row["nearest_train_cycle_index"] = cycle_order[:, 0]
    row["median_neighbor_train_cycle_index"] = np.median(cycle_order, axis=1)
    row["neighbor_train_cycle_ids"] = ["|".join(ids) for ids in neighbor_ids]
    for block in sorted(train.block_id.unique()):
        row[f"neighbor_count_{block}"] = (blocks == block).sum(axis=1)
    row["distance_min"] = distances[:, 0]
    row["distance_mean"] = distances.mean(axis=1)
    row["distance_std"] = distances.std(axis=1)
    for quantile in (10, 50, 90):
        row[f"distance_p{quantile}"] = np.percentile(distances, quantile, axis=1)
    row["distance_max"] = distances[:, -1]
    row["zero_distances"] = (distances == 0).sum(axis=1)
    row["adjacent_distance_ties_1e12"] = (
        np.abs(np.diff(distances, axis=1)) <= 1e-12
    ).sum(axis=1)

    summaries = []

    def summarize(group, mask, comparison=None):
        mask = mask.to_numpy()
        subset = row.loc[mask]
        counts = np.bincount(indices[mask].ravel(), minlength=len(train_ids))
        overlap = np.nan
        if comparison is not None and len(subset):
            normal_counts = np.bincount(
                indices[comparison.to_numpy()].ravel(), minlength=len(train_ids)
            )
            overlap = np.minimum(counts / counts.sum(), normal_counts / normal_counts.sum()).sum()
        summaries.append({
            "group": group, "n": len(subset),
            "knn_alarms": subset.knn_margin.gt(0).sum(),
            "lof_alarms": subset.lof_margin.gt(0).sum(),
            "nearest_distance_median": subset.distance_min.median(),
            "mean_distance_median": subset.distance_mean.median(),
            "mean_distance_p10": subset.distance_mean.quantile(0.1),
            "mean_distance_p90": subset.distance_mean.quantile(0.9),
            "lof_score_median": subset.lof_score.median(),
            "lof_score_p10": subset.lof_score.quantile(0.1),
            "lof_score_p90": subset.lof_score.quantile(0.9),
            "lof_knn_spearman": spearmanr(subset.lof_score, subset.knn_score).statistic
            if len(subset) > 1 else np.nan,
            "neighbor_frequency_overlap_with_paired_normal": overlap,
            "median_neighbor_train_cycle_index": np.median(cycle_order[mask])
            if len(subset) else np.nan,
            **{
                f"neighbor_fraction_{block}": (blocks[mask] == block).mean()
                if len(subset) else np.nan
                for block in sorted(train.block_id.unique())
            },
        })

    summarize("all_evaluation", row.condition.notna())
    restart = row.cohort.str.contains("restart", case=False, na=False)
    if restart.sum() != 40:
        raise ValueError("Expected 40 evaluation cycles in the restart cohort")
    summarize("restart_all", restart)
    for method in ("knn", "lof"):
        for outcome in ("FP", "TN"):
            summarize(f"restart_{method}_{outcome}", restart & row[f"{method}_outcome"].eq(outcome))
    for condition in ("A", "B", "C"):
        paired = row.condition.eq("normal") & row.pair_condition.eq(condition)
        anomaly = row.condition.eq(condition)
        if (paired.sum(), anomaly.sum()) != ({"A": 30, "B": 10, "C": 10}[condition], 30):
            raise ValueError(f"Unexpected paired population for condition {condition}")
        summarize(f"{condition}_paired_normal", paired)
        summarize(f"{condition}_anomaly", anomaly, paired)
        for method in ("knn", "lof"):
            for outcome in ("TP", "FN"):
                summarize(
                    f"{condition}_{method}_{outcome}",
                    anomaly & row[f"{method}_outcome"].eq(outcome),
                    paired,
                )

    row.reset_index().to_csv(metrics_dir / "NEIGHBOR_DIAGNOSTICS.csv", index=False)
    pd.DataFrame(summaries).to_csv(metrics_dir / "NEIGHBOR_GROUP_SUMMARY.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    describe_neighbors(args.run_dir, args.project_root)
