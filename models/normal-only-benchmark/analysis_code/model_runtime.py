"""Fixed model specifications and batch-independent scoring for this benchmark."""

import time
import warnings

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM


SEED = 20261031
ATOL = 1e-11
RTOL = 1e-12
FROZEN_IDS = ("V4_PCA_Q", "IF_24", "IF_PC13_Q")
MAIN_IDS = ("LOF_K20", "OCSVM_G1", "KNN_K20_MEAN", "LOF_K10", "LOF_K40", "OCSVM_G025", "OCSVM_G4")
FOLD_IDS = ("PCA_Q", "LOF_K20", "OCSVM_G1", "KNN_K20_MEAN")
GAMMA_MULTIPLIER = {"OCSVM_G025": .25, "OCSVM_G1": 1., "OCSVM_G4": 4.}
FORMULAS = {
    "pca_q": "mean((Z - PCA.inverse_transform(PCA.transform(Z)))**2, axis=1)",
    "if": "-estimator.score_samples(Z)",
    "if_pc13_q": "-estimator.score_samples([13 non-whitened frozen PCs, Q])",
    "lof": "-estimator.score_samples(Z), novelty=True; valid on unseen rows",
    "ocsvm": "-estimator.score_samples(Z)",
    "knn_mean": "mean of the 20 Euclidean distances to nearest normal Train rows",
}


def estimator_for(model_id, gamma_base):
    if model_id.startswith("LOF_K"):
        return "lof", LocalOutlierFactor(n_neighbors=int(model_id[5:]), metric="euclidean", algorithm="brute",
                                        novelty=True, contamination="auto", n_jobs=1)
    if model_id.startswith("OCSVM_"):
        return "ocsvm", OneClassSVM(kernel="rbf", gamma=GAMMA_MULTIPLIER[model_id] * gamma_base,
                                    nu=.05, tol=1e-3, shrinking=True, max_iter=-1)
    if model_id == "KNN_K20_MEAN":
        return "knn_mean", NearestNeighbors(n_neighbors=20, metric="euclidean", algorithm="brute", n_jobs=1)
    if model_id == "PCA_Q":
        return "pca_q", PCA(n_components=.95, svd_solver="full", whiten=False)
    raise ValueError(f"Unknown benchmark model: {model_id}")


def gamma_scale(z_train):
    value = 1 / (z_train.shape[1] * z_train.var(ddof=0))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Non-positive or nonfinite SVM gamma scale")
    return float(value)


def require_feature_order(frame, features):
    if list(frame.columns) != list(features):
        raise ValueError("Feature order differs from the frozen schema")
    x = frame.to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Unexpected missing or nonfinite benchmark feature")
    return x


def q_score(estimator, z):
    return np.mean((z - estimator.inverse_transform(estimator.transform(z))) ** 2, axis=1)


def score_z(bundle, z):
    estimator = bundle["estimator"]
    kind = bundle["kind"]
    if kind == "pca_q":
        return q_score(estimator, z)
    if kind == "knn_mean":
        return estimator.kneighbors(z, return_distance=True)[0].mean(axis=1)
    if kind == "if_pc13_q":
        pca = bundle["pca"]
        z = np.column_stack([pca.transform(z), q_score(pca, z)])
    return -estimator.score_samples(z)


def score_features(bundle, frame):
    x = require_feature_order(frame, bundle["features"])
    return score_z(bundle, bundle["scaler"].transform(bundle["imputer"].transform(x)))


def calibration_cutoff(scores):
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be finite and nonempty")
    cutoff = float(np.quantile(scores, .95, method="higher"))
    assert cutoff == np.sort(scores)[int(np.ceil(.95 * (len(scores) - 1)))]
    return cutoff


def prepare_estimator(model_id, z_train, gamma_base):
    kind, estimator = estimator_for(model_id, gamma_base)
    start = time.perf_counter_ns()
    with warnings.catch_warnings(record=True) as issued:
        warnings.simplefilter("always")
        estimator.fit(z_train)
    fit_ms = (time.perf_counter_ns() - start) / 1e6
    warning_messages = [f"{w.category.__name__}: {w.message}" for w in issued]
    if kind in {"lof", "knn_mean"}:
        np.testing.assert_array_equal(estimator._fit_X, z_train)
    return kind, estimator, fit_ms, warning_messages


def check_reload_and_invariance(bundle, frame, expected_eval, new_path=None):
    """Use actual one-row calls and both shuffled/reversed batches on every evaluation row."""
    x = require_feature_order(frame, bundle["features"])
    imputer, scaler = bundle["imputer"], bundle["scaler"]
    z = scaler.transform(imputer.transform(x))
    limit = bundle["threshold"]
    variants = {"batch": score_z(bundle, z)}
    permutation = np.random.default_rng(SEED).permutation(len(x))
    variants["shuffled"] = score_z(bundle, z[permutation])[np.argsort(permutation)]
    variants["reversed"] = score_z(bundle, z[::-1])[::-1]
    duration, single = [], []
    for i in range(len(x)):
        start = time.perf_counter_ns()
        value = score_z(bundle, scaler.transform(imputer.transform(x[i:i+1])))
        duration.append((time.perf_counter_ns() - start) / 1e6)
        single.append(float(value[0]))
    variants["single"] = np.asarray(single)
    if new_path is not None:
        reloaded = joblib.load(new_path)
        assert reloaded["features"] == bundle["features"] and reloaded["threshold"] == limit
        assert reloaded["preprocessor_sha256"] == bundle["preprocessor_sha256"]
        variants["reload"] = score_features(reloaded, frame)
        np.testing.assert_array_equal(variants["reload"], variants["batch"])
    metrics = dict(score_tolerance_atol=ATOL, score_tolerance_rtol=RTOL,
                   scored_evaluation_rows=len(x), timing_single_preprocessed_ms_median=float(np.median(duration)),
                   timing_single_preprocessed_ms_p95=float(np.quantile(duration, .95)))
    for tag, values in variants.items():
        metrics[f"{tag}_max_abs_error"] = float(np.max(np.abs(values - expected_eval)))
        metrics[f"{tag}_alarm_mismatches"] = int(np.count_nonzero((values > limit) != (expected_eval > limit)))
        np.testing.assert_allclose(values, expected_eval, atol=ATOL, rtol=RTOL)
        assert metrics[f"{tag}_alarm_mismatches"] == 0, (bundle["model_id"], tag)
    return metrics


def solver_details(bundle, query_z, values):
    estimator = bundle["estimator"]
    limit = bundle["threshold"]
    unique_n = np.unique(values).size
    details = dict(unique_score_n=unique_n, score_tie_n=len(values)-unique_n,
                   threshold_equal_n=int(np.count_nonzero(values == limit)),
                   min_score=float(np.min(values)), max_score=float(np.max(values)))
    kind = bundle["kind"]
    if kind == "ocsvm":
        support = estimator.support_vectors_
        gamma = estimator.get_params()["gamma"]
        kernel = np.exp(-gamma * pairwise_distances(query_z, support, metric="sqeuclidean"))
        max_error = float(np.max(np.abs(-(kernel @ estimator.dual_coef_.ravel()) - values)))
        np.testing.assert_allclose(-(kernel @ estimator.dual_coef_.ravel()), values, atol=ATOL, rtol=RTOL)
        details.update(fit_status=int(estimator.fit_status_), n_iter=int(estimator.n_iter_), support_n=len(support),
                       support_fraction=len(support) / len(bundle["train_cycle_ids"]), actual_gamma=gamma,
                       kernel_min=float(kernel.min()), kernel_max=float(kernel.max()), kernel_median=float(np.median(kernel)),
                       kernel_underflow_zero_n=int(np.count_nonzero(kernel == 0)),
                       kernel_subnormal_nonzero_n=int(np.count_nonzero((kernel > 0) & (kernel < np.finfo(float).tiny))),
                       kernel_near_zero_n=int(np.count_nonzero(kernel < 1e-12)),
                       kernel_near_one_n=int(np.count_nonzero(kernel > 1 - 1e-12)), kernel_value_n=kernel.size,
                       rbf_dual_score_max_error=max_error)
    elif kind in {"knn_mean", "lof"}:
        k = estimator.n_neighbors
        d, idx = estimator.kneighbors(query_z, n_neighbors=k)
        assert d.shape == idx.shape == (len(query_z), k)
        assert idx.min() >= 0 and idx.max() < len(bundle["train_cycle_ids"])
        details.update(n_neighbors=k, nearest_zero_n=int(np.count_nonzero(d[:, 0] == 0)),
                       distance_zero_n=int(np.count_nonzero(d == 0)),
                       within_neighbor_list_tie_n=int(np.count_nonzero(np.diff(d, axis=1) == 0)))
    return details


def verify_self_exclusion(bundle):
    if bundle["kind"] not in {"knn_mean", "lof"}:
        return None
    estimator = bundle["estimator"]
    idx = estimator.kneighbors(X=None, n_neighbors=estimator.n_neighbors, return_distance=False)
    self_count = int(np.count_nonzero(idx == np.arange(len(idx))[:, None]))
    assert self_count == 0
    return self_count
