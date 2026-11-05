"""Offline V3.1 ablation and provenance audit. No device interfaces."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DOSE = {
    "normal_a": ("normal_a", 10), "normal_a2": ("normal_a2", 10),
    "normal_c": ("normal_c", 10), "normal_c2": ("normal_c2", 10),
    "A121": ("A", 20), "A193": ("A2", 20),
    "C182": ("C_retry", 20), "C462": ("C462_retry", 20),
}
REFERENCES = dict.fromkeys(["A", "B", "C", "D", "E", "F", "B232"], "previous20")
REFERENCES.update(A121="normal_a", A193="normal_a2", C182="normal_c", C462="normal_c2")
NORMALS = ["holdout", "previous20", "normal_a", "normal_a2", "normal_c", "normal_c2"]
S3, S4, S5 = (f"S{i}_current_mean" for i in (3, 4, 5))
RMS, STD, P95 = ("S4_current_rms", "S4_current_std", "S4_current_abs_p95")
DIFF, RATIO = ("S4_minus_S3_current_mean", "S4_div_S3_current_mean")
SIGNALS = ["current_A_rms", S3, S4, S5, RMS, STD, P95, DIFF, RATIO,
           "S3_current_rms", "S5_current_rms", "ID1_3_5s_IQR", "position_ID1_raw_range"]
REPRESENTATIVES = ["R0_IsolationForest", "R1_IsolationForest", "V3-A_PCA",
                   "V3-D_IsolationForest", "V3-C-S34_PCA"]


def configurations(base, grip):
    options = [[], [S3], [S4], [S5], [S3, S4], [S4, S5], [S3, S4, S5],
               [S4, RMS], [S4, STD], [S3, S4, RMS], [S3, S4, DIFF],
               [S3, S4, RATIO], [S3, S4, RMS, STD], [S4, P95], [S3, S4, grip]]
    result = {f"V31-{i}": [*base, *cols] for i, cols in enumerate(options)}
    for number, source in [(15, 14), (16, 4), (17, 2)]:
        result[f"V31-{number}"] = [c for c in result[f"V31-{source}"] if c != "position_ID1_raw_range"]
    return result


def combined_data(v3, src, h, old, recent):
    original, excluded = v3.load_data(h, src)
    manifest = src.csv(old / "DATA_MANIFEST.csv")
    pd.testing.assert_frame_equal(original[manifest.columns].reset_index(drop=True), manifest,
                                  check_dtype=False)
    dose = src.csv(recent / "analysis/fixed_model_comparison/CYCLE_FEATURES.csv")
    dose = dose.assign(cohort=dose.cohort.replace({"A": "A121", "A2": "A193", "C": "C182"})).copy()
    metadata = []
    for cohort, (stage, count) in DOSE.items():
        rows = dose[dose.cohort.eq(cohort)]
        result = src.json(recent / f"result_{stage}.json")
        audit = src.json(recent / f"analysis/{stage}/audit.json")
        review = src.json(recent / f"physical_review_{stage}.json")
        archive = src.track(result["archive"])
        assert len(rows) == result["completed_count"] == audit["quality_valid"] == count
        assert review["confirmed_completed_transport_and_return"] == count
        assert review["status"] == "confirmed"
        assert src.hashes[archive] == result["archive_sha256"]
        with zipfile.ZipFile(archive) as packed:
            assert packed.testzip() is None
        assert all(audit[key] for key in ("raw_uart_csv_identical", "raw_hid_csv_identical",
                                         "backup_members_identical", "source_hashes_unchanged"))
        assert not any(audit[key] for key in ("position_missing", "position_timeout", "ADXL_encoding_bad"))
        assert not any(audit["sequence_INA"].values()) and not any(audit["sequence_ADXL"].values())
        folder = Path(result["folder"])
        assert rows.source.map(Path).eq(folder).all()
        capture = src.json(folder / "manifest.json")
        assert rows.session_id.eq(capture["session_id"]).all()
        events = [json.loads(line) for line in src.track(folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        first, = [event["pose"] for event in events if event["kind"] == "verified_initial_pose"]
        dose.loc[rows.index, "start_pose"] = np.where(rows.cycle_index.eq(rows.cycle_index.min()), first, "A_release")
        dose.loc[rows.index, "capture_stage"] = stage
        metadata.append(dict(cohort=cohort, capture_stage=stage, session_id=capture["session_id"],
                             count=count, first_pose=first, archive_sha256=src.hashes[archive],
                             audit_position_missing=audit["position_missing"],
                             ID5_home_tolerance_raw=result["stage"].get("ID5_home_tolerance_raw", 5)))
    dose = dose.assign(automatic_return=True, base_condition="fixed",
                       classification=np.where(dose.normal_condition, "confirmed_normal_41g", "confirmed_anomaly"),
                       physical_outcome="operator confirmed transport and return")
    for stage, reason in [("C", "interrupted, tape and return timeout"), ("C2_pilot", "282g pilot"),
                          ("C462", "first attempt, incomplete")]:
        result = src.json(recent / f"result_{stage}.json")
        capture = src.json(Path(result["folder"]) / "manifest.json")
        excluded.loc[len(excluded)] = [capture["session_id"] + "_1", stage, "excluded",
                                       result["reason"], reason]
    original["capture_stage"] = original.stage
    original["base_condition"] = "fixed"
    combined = pd.concat([original, dose.reindex(columns=original.columns)], ignore_index=True)
    combined["ID4_home_tolerance_override"] = np.where(combined.cohort.eq("B232"), "954±10", "")
    combined["ID5_home_tolerance_override"] = np.where(combined.cohort.eq("C462"), "781±10", "")
    assert len(combined) == 788 and combined.cycle_id.is_unique
    assert not combined.cycle_id.isin(excluded.cycle_id).any()
    return combined, dose, excluded, pd.DataFrame(metadata)


def raw_extras(data, wave, v3, src):
    stats = []
    valid = data.set_index("cycle_id")
    ina = wave[wave.stream.eq("INA")]
    for cid, samples in ina.groupby("cycle_id", sort=False):
        out = {"cycle_id": cid}
        for phase, (lo, hi) in v3.PHASES.items():
            values = samples.loc[samples.elapsed_s.ge(lo) & samples.elapsed_s.lt(hi), "current_A"].to_numpy()
            if pd.notna(valid.at[cid, phase + "_current_mean"]):
                out[phase + "_current_rms"] = np.sqrt(np.mean(values**2))
                if phase == "S4":
                    out[STD] = values.std(ddof=0)
                    out[P95] = np.quantile(abs(values), .95, method="inverted_cdf")
        stats.append(out)
    new = pd.DataFrame(stats)
    data = data.merge(new, on="cycle_id", validate="one_to_one")
    data[DIFF] = data[S4] - data[S3]
    data[RATIO] = data[S4] / data[S3].where(data[S3].abs().ge(.05))
    observed = []
    for folder, cycles in data.groupby("source", sort=False):
        cols = ["cycle_id", "phase", "servo_id", "response_ns", "position_raw", "status", "value_valid"]
        pos = src.csv(Path(folder) / "robot/single_queries.csv", usecols=cols)
        pos = pos[pos.phase.eq("No3_A_to_B") & pos.status.eq("complete") & pos.value_valid.eq(1)
                  & pos.servo_id.isin([4, 5])]
        pos = pos.merge(cycles[["cycle_id", "cohort", v3.T0, v3.T1]], on="cycle_id", validate="many_to_one")
        pos = pos[pos.response_ns.ge(pos[v3.T0]) & pos.response_ns.le(pos[v3.T1])].copy()
        pos["elapsed_s"] = (pos.response_ns - pos[v3.T0]) / 1e9
        observed.append(pos[["cycle_id", "cohort", "servo_id", "elapsed_s", "position_raw"]])
    position = pd.concat(observed).sort_values("elapsed_s")
    ids = data.cycle_id
    rows, windows = [], {}
    for servo in (4, 5):
        p = position[position.servo_id.eq(servo)]
        train_ids = data.loc[data.cohort.eq("train"), "cycle_id"]
        options = []
        for lo in (9., 9.5, 10., 10.5, 11.):
            in_window = p[p.elapsed_s.ge(lo) & p.elapsed_s.lt(lo + .5)]
            grouped = in_window.groupby("cycle_id").position_raw
            medians = grouped.median().reindex(train_ids)
            options.append((int(medians.notna().sum()), float(medians.quantile(.75)-medians.quantile(.25)),
                            lo, grouped, medians))
        coverage, _, lo, grouped, medians = min(options, key=lambda option: (-option[0], option[1], option[2]))
        ref = float(medians.median())
        windows[str(servo)] = dict(window=[lo, lo+.5], train_median=ref, train_coverage=coverage,
                                    rule="max train coverage, then min train IQR, then earliest")
        last = p.drop_duplicates("cycle_id", keep="last").set_index("cycle_id")
        rows.append(pd.DataFrame(dict(cycle_id=ids, cohort=data.cohort, servo_id=servo,
                                      fixed_window_start_s=lo, fixed_window_end_s=lo+.5,
                                      fixed_n=ids.map(grouped.size()).fillna(0).astype(int),
                                      fixed_position_raw=ids.map(grouped.median()),
                                      fixed_time_deviation_raw=(ids.map(grouped.median())-ref).abs(),
                                      last_position_raw=ids.map(last.position_raw),
                                      last_elapsed_s=ids.map(last.elapsed_s), duration_s=data.duration_s)))
    return data, position, pd.concat(rows), windows


def frozen_scores(data, src, v3, index_paths, archives):
    all_frames, provenance = [], []
    for idx_path in index_paths:
        for row in src.csv(idx_path).itertuples(index=False):
            path = src.track(row.path)
            assert src.hashes[path] == row.sha256
            bundle = joblib.load(path)
            assert bundle["threshold"] == row.threshold
            cols = row.features.split(";")
            assert list(bundle["features"]) == cols
            values = bundle["detector"].score(data[cols].to_numpy(float))
            frame = v3.score_frame(data, row.version, "fixed_base", row.model_id, row.config_id,
                                   row.algorithm, bool(row.invalid_PCA), row.threshold, values)
            for i, saved in enumerate(archives):
                matching = saved[saved.version.eq(row.version) & saved.model_id.eq(row.model_id)]
                check = frame.merge(matching[["cycle_id", "score", "flagged", "threshold"]],
                                    on="cycle_id", suffixes=("", "_stored"), validate="one_to_one")
                expected = 668 if i == 0 else 120
                assert len(check) == expected
                np.testing.assert_allclose(check.score, check.score_stored, rtol=1e-10, atol=1e-12)
                assert check.flagged.eq(check.flagged_stored).all() and check.threshold.eq(check.threshold_stored).all()
                provenance.append(dict(version=row.version, model_id=row.model_id, n=len(check),
                                       max_score_error=float(abs(check.score-check.score_stored).max())))
            all_frames.append(frame)
    return pd.concat(all_frames), pd.DataFrame(provenance)


def distributions(data, value_cols, groups):
    rows = []
    for keys, frame in data.groupby(groups, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for column in value_cols:
            value = frame[column].dropna()
            rows.append(dict(zip(groups, keys), feature=column, total=len(frame), n=len(value),
                             median=value.median(), q10=value.quantile(.1), q90=value.quantile(.9),
                             iqr=value.quantile(.75)-value.quantile(.25)))
    return pd.DataFrame(rows)


def evaluate(scores):
    case_rows, normal_rows, missed = [], [], []
    for keys, frame in scores.groupby(["version", "model_id", "config_id", "algorithm"], sort=False):
        version, model, cfg, algorithm = keys
        common = dict(version=version, model_id=model, config_id=cfg, algorithm=algorithm,
                      threshold=frame.threshold.iloc[0], invalid_PCA=frame.invalid_PCA.iloc[0])
        indexed = dict(tuple(frame.groupby("cohort")))
        normal = frame[frame.cohort.isin([*NORMALS, "train", "validation"])]
        for field in ("cohort", "stage", "session_id"):
            for label, group in normal.groupby(field):
                normal_rows.append(dict(common, grouping=field, group=label, cohort=group.cohort.iloc[0],
                                        n=len(group), FP=int(group.flagged.sum()), FPR=group.flagged.mean(),
                                        median=group.score.median(), q10=group.score.quantile(.1),
                                        q90=group.score.quantile(.9), median_margin=group.margin.median()))
        for condition, reference in REFERENCES.items():
            for ref in (reference, "holdout") if reference == "previous20" else (reference,):
                pos, neg = indexed[condition], indexed[ref]
                labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                values = np.r_[pos.score, neg.score]
                tpr, fpr = pos.flagged.mean(), neg.flagged.mean()
                q95 = float(np.quantile(neg.score, .95, method="higher"))
                fn = pos[~pos.flagged]
                case_rows.append(dict(common, condition=condition, reference=ref,
                                      n_anomaly=len(pos), n_normal=len(neg), TP=int(pos.flagged.sum()),
                                      FP=int(neg.flagged.sum()), FN=len(fn), TPR=tpr, FPR=fpr, FNR=1-tpr,
                                      AUROC=roc_auc_score(labels, values), AP=average_precision_score(labels, values),
                                      balanced_accuracy=(1+tpr-fpr)/2, TPR_minus_FPR=tpr-fpr,
                                      median=pos.score.median(), q10=pos.score.quantile(.1), q90=pos.score.quantile(.9),
                                      median_margin=pos.margin.median(), normal_q95=q95,
                                      FN_above_reference_q95=int(fn.score.gt(q95).sum()),
                                      FN_median_margin=fn.margin.median()))
                if ref == reference and len(fn):
                    missed.append(fn.assign(condition=condition, reference=ref, normal_q95=q95,
                                            above_reference_q95=fn.score.gt(q95)))
    return pd.DataFrame(case_rows), pd.DataFrame(normal_rows), pd.concat(missed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    h, out = args.project.resolve() / "hardware", args.output.resolve()
    if out.exists() and any(p.is_file() and "analysis_code" not in p.relative_to(out).parts
                            for p in out.rglob("*")):
        raise RuntimeError("Choose an empty experiment directory.")
    for directory in ("models", "metrics", "figures", "verification"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    old = h / "xarm_model_versions/v3_phase_features"
    fixed = h / "xarm_model_versions/v1_fixed_base_20261105"
    recent = h / "No3_AC_dose_20261105"
    sys.path[:0] = [str(old / "analysis_code"),
                   str(h / "No3_41g_main_20261101/analysis_code/baselines_v1"),
                   str(h / "No3_waveform_comparison_20261103/analysis_code")]
    import train_xarm_v3 as v3
    from normal_models import Detector, threshold
    from compare_no3_waveforms import current_fall
    from summarize_xarm_v31 import summarize

    src = v3.Sources()
    stats_file = src.track(h / "normal_main_recheck_v2_20261031/pc/pilot_analysis.py")
    spec = importlib.util.spec_from_file_location("v31_feature_statistics", stats_file)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    for module in ("train_xarm_v3", "normal_models", "compare_no3_waveforms", "summarize_xarm_v31"):
        src.track(sys.modules[module].__file__)
    old_config = src.json(old / "TRAINING_CONFIG.json")
    base = old_config["configurations"]["R1"]
    configs = configurations(base, v3.GRIP)
    data, dose, excluded, audit = combined_data(v3, src, h, old, recent)
    data, wave, raw_parity = v3.extract(data, src, stats.describe, current_fall, base)
    data[v3.END] = (data.id4_end_raw - data.loc[data.cohort.eq("train"), "id4_end_raw"].median()).abs()
    for saved in [src.csv(old / "features/CYCLE_FEATURES.csv"), dose]:
        cols = [*base, *v3.CURRENT, v3.GRIP, v3.STEP, v3.FALL, v3.END]
        replay = data.set_index("cycle_id").loc[saved.cycle_id, cols]
        np.testing.assert_allclose(replay, saved[cols], rtol=1e-10, atol=1e-12, equal_nan=True)
    data, position, pos_features, windows = raw_extras(data, wave, v3, src)
    print("788 raw cycles reproduced, replaying frozen models", flush=True)
    archives = [src.csv(old / "metrics/CYCLE_SCORES.csv"),
                src.csv(recent / "analysis/fixed_model_comparison/CYCLE_SCORES.csv")]
    prior, frozen_audit = frozen_scores(data, src, v3, [old / "MODEL_INDEX.csv", fixed / "MODEL_INDEX.csv"], archives)
    all_train = data.loc[data.cohort.eq("train")]
    assert all_train[[*{c for cols in configs.values() for c in cols}]].notna().any().all()
    new, index, attribution, _ = v3.train(data, configs, old_config["seed"], Detector, threshold,
                                          out, prior, v3.sha(__file__))
    index["version"] = "V3.1"
    new["version"] = "V3.1"
    for row in index.itertuples(index=False):
        bundle = joblib.load(row.path)
        bundle["version"] = "V3.1"
        joblib.dump(bundle, row.path)
        index.loc[index.model_id.eq(row.model_id), "sha256"] = v3.sha(row.path)
        replay = joblib.load(row.path)
        values = replay["detector"].score(data[replay["features"]].to_numpy(float))
        np.testing.assert_array_equal(values, new.loc[new.model_id.eq(row.model_id), "score"])
    control = new[new.config_id.eq("V31-0")].merge(prior[prior.config_id.eq("R1")],
                                                   on=["cycle_id", "algorithm"], suffixes=("", "_prior"))
    assert len(control) == 3*len(data)
    np.testing.assert_allclose(control.score, control.score_prior, rtol=1e-10, atol=1e-12)
    assert control.flagged.eq(control.flagged_prior).all() and control.threshold.eq(control.threshold_prior).all()
    scores = pd.concat([prior, new], ignore_index=True)
    cases, normals, misses = evaluate(scores)
    print(f"{len(index)} models trained and replayed, writing comparisons", flush=True)
    manifest_cols = [c for c in data if c in ["cycle_id", "session_id", "cycle_index", "cohort", "stage",
                     "capture_stage", "classification", "physical_outcome", "condition_id", "payload_mass_g",
                     "frame_added_mass_g", "start_pose", "automatic_return", "base_condition", "source",
                     v3.T0, v3.T1, "ID4_home_tolerance_override", "ID5_home_tolerance_override"]]
    outputs = {"DATA_MANIFEST.csv": data[manifest_cols], "MODEL_INDEX.csv": index,
               "metrics/CYCLE_FEATURES.csv": data.drop(columns="source"), "metrics/CYCLE_SCORES.csv": scores,
               "ANOMALY_TYPE_METRICS.csv": cases, "NORMAL_FPR_COMPARISON.csv": normals,
               "ID4_ID5_POSITION_ANALYSIS.csv": pos_features, "verification/EXCLUDED_CYCLES.csv": excluded,
               "verification/RAW_FEATURE_PARITY.csv": raw_parity, "verification/FROZEN_REPLAY.csv": frozen_audit,
               "verification/DOSE_AUDIT.csv": audit, "metrics/MISSED_CYCLES.csv": misses}
    for name, table in outputs.items():
        v3.save_csv(out, name, table)
    v3.save_json(out, "FEATURE_DEFINITIONS.json", dict(windows=v3.PHASES, clock="No.3 command write completion / host monotonic",
                 minimum_samples=[8, 8, 12], max_gap_s=.25, no_sequence_breaks=True, s4_std_ddof=0,
                 abs_p95="nearest rank", ratio_min_abs_denominator_A=.05, positions=windows,
                 positions_in_model=False, id1_IQR="unchanged V3 4-sample IQR with +/-100ms sensitivity"))
    v3.save_json(out, "TRAINING_CONFIG.json", dict(configurations=configs, seed=old_config["seed"],
                 threshold="normal validation q95 higher, score > threshold", algorithms=list(v3.ALGOS),
                 imputer="train median", scaler="train StandardScaler", trees=200, pca_variance=.95,
                 training_cycle_ids=all_train.cycle_id.tolist(),
                 validation_cycle_ids=data.loc[data.cohort.eq("validation"), "cycle_id"].tolist(),
                 input_sha256={str(p): digest for p, digest in src.hashes.items()}, code_sha256=v3.sha(__file__)))
    summarize(out, h, v3, data, cases, normals, scores, attribution, position, pos_features, index, old, windows, wave)
    unchanged = all(v3.sha(p) == digest for p, digest in src.hashes.items())
    assert unchanged
    v3.save_json(out, "VERIFICATION.json", dict(n=len(data), cohort_counts=data.cohort.value_counts().to_dict(),
                 exclusions=len(excluded), source_and_prior_hashes_unchanged=unchanged,
                 cycle_ids_unique=True, native_features_replayed=788,
                 raw_23_max_abs_error=float(raw_parity.max_absolute_error.max()),
                 archived_V3_models=33, frozen_score_replay_rows=int(frozen_audit.n.sum()),
                 v31_models_saved_and_reloaded=len(index), V31_0_R1_parity=True,
                 dose_pairing="group only, no cycle pairs", train=250, validation=100,
                 independent_validation_executed=False, V3_replaced=False))


if __name__ == "__main__":
    main()
