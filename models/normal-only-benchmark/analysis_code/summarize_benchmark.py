"""Create and independently reconcile the normal-only benchmark evaluation tables."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


BASELINE = "V4_PCA_Q"
FROZEN_MODELS = (BASELINE, "IF_24", "IF_PC13_Q")
DEFAULT_MODELS = ("LOF_K20", "OCSVM_G1", "KNN_K20_MEAN")
SENSITIVITY_MODELS = ("LOF_K10", "LOF_K40", "OCSVM_G025", "OCSVM_G4")
PRIMARY_MODELS = FROZEN_MODELS + DEFAULT_MODELS + SENSITIVITY_MODELS
FOLD_MODELS = ("PCA_Q",) + DEFAULT_MODELS
MAIN = "PRIMARY"
FOLDS = ("A_MAIN", "B_MAIN")
SESSIONS = {
    "historical_test99": 99, "extra_normal17": 17, "extra_normal13": 13, "preA": 30,
    "preB": 10, "preC": 10, "nov10_normal40": 40, "nov10_restart40": 40,
}
RESTART_SEGMENTS = ((1, 10), (11, 20), (21, 30), (31, 40), (11, 40), (21, 40))
PAIR_NORMAL = {"A": 30, "B": 10, "C": 10}
TRANSITIONS = ("resolved_fp", "new_fp", "common_fp", "both_normal",
               "added_tp", "lost_tp", "common_tp", "both_missed")
DEFAULT_PROJECT = Path.cwd()


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def validate_scores(study, project):
    archive = project / "hardware/xarm_model_versions"
    canonical = archive / "v5_1_multisession_normal_study"
    usage = read_csv(canonical / "DATA_USAGE_MANIFEST.csv").set_index("cycle_id")
    assert usage.index.is_unique and len(usage) == 649 and usage.quality_valid.all()
    assert usage.sensor_mount.eq("bolt_and_nut_user_reported").all()
    split = read_csv(archive / "v4_bolt_fixed/SPLIT_MANIFEST.csv").set_index("cycle_id")
    assert split.index.is_unique and len(split) == 399
    assert usage.loc[split.index, "role"].equals(split.role)
    assert usage.role.value_counts().to_dict() == {"evaluation_only": 250, "train": 200, "validation": 100, "test": 99}
    # The V5.1 fold rules deliberately exclude the historical 99 from E in the group comparisons.
    old_folds = read_csv(canonical / "SPLIT_MANIFEST.csv")
    expected = {MAIN: usage.index[usage.role.isin(["test", "evaluation_only"])]}
    expected_fit = {MAIN: (usage.index[usage.role.eq("train")], usage.index[usage.role.eq("validation")])}
    for fold in FOLDS:
        part = old_folds[old_folds.fold_id.eq(fold)].set_index("cycle_id")
        assert part.index.is_unique and set(part.index) == set(usage.index)
        expected[fold] = part.index[part.split_role.eq("E")]
        expected_fit[fold] = (part.index[part.split_role.isin(["T0", "T_ADD"])],
                              part.index[part.split_role.isin(["C0", "C_ADD"])])
    assert {k: len(v) for k, v in expected.items()} == {MAIN: 349, "A_MAIN": 40, "B_MAIN": 170}
    assert {k: tuple(map(len, v)) for k, v in expected_fit.items()} == {MAIN: (200, 100), "A_MAIN": (280, 140), "B_MAIN": (240, 140)}

    scores = read_csv(study / "metrics/CYCLE_SCORES.csv")
    assert not scores.duplicated(["split_id", "model_id", "cycle_id"]).any()
    assert set(scores.split_id) == {MAIN, *FOLDS}
    assert set(scores.role) == {"train", "calibration", "evaluation"}
    scores["flagged"] = scores.flagged.astype("string").str.lower().map({"true": True, "false": False}).astype("boolean")
    scores["pair_condition"] = scores.pair_condition.fillna("")
    for split_id, group in scores.groupby("split_id", sort=False):
        assert set(group.model_id) == set(PRIMARY_MODELS if split_id == MAIN else FOLD_MODELS)
        for model, part in group.groupby("model_id", sort=False):
            evaluation = part[part.role.eq("evaluation")]
            train, calibration = expected_fit[split_id]
            assert set(evaluation.cycle_id) == set(expected[split_id]), (split_id, model, "evaluation")
            assert set(part.loc[part.role.eq("train"), "cycle_id"]) == set(train), (split_id, model, "train")
            assert set(part.loc[part.role.eq("calibration"), "cycle_id"]) == set(calibration), (split_id, model, "calibration")
            assert part.threshold.nunique() == 1 and np.isfinite(part.threshold).all()
            scored = part[part.score.notna()]
            assert np.isfinite(scored.score).all() and scored.flagged.notna().all()
            np.testing.assert_allclose(scored.margin, scored.score - scored.threshold, rtol=1e-12, atol=1e-14)
            np.testing.assert_array_equal(scored.flagged.to_numpy(bool), scored.score.gt(scored.threshold))
            assert part.loc[part.score.isna(), "role"].eq("train").all()
    aligned = usage.loc[scores.cycle_id]
    for a, b in (("original_role", "role"), ("condition", "condition"), ("eval_session", "eval_session"),
                 ("pair_condition", "pair_condition"), ("leakage_group_id", "leakage_group_id")):
        np.testing.assert_array_equal(scores[a].fillna("").to_numpy(), aligned[b].fillna("").to_numpy())
    scores["session_cycle_index"] = aligned.session_cycle_index.to_numpy()
    return scores, expected_fit


def score_stats(series, prefix):
    if series.empty:
        return {f"{prefix}_{name}": np.nan for name in ("min", "q05", "q25", "median", "q75", "q95", "max")}
    values = series.quantile([0, .05, .25, .5, .75, .95, 1])
    return dict(zip((f"{prefix}_{name}" for name in ("min", "q05", "q25", "median", "q75", "q95", "max")), values, strict=True))


def alarm_count(part):
    assert part.flagged.notna().all()
    return int(part.flagged.sum())


def normal_subsets(normal):
    yield "overall", "all", normal
    external = normal[normal.original_role.eq("evaluation_only")]
    if 0 < len(external) < len(normal):
        yield "external", "external160", external
    sessions = normal.assign(group=np.where(normal.original_role.eq("test"), "historical_test99", normal.eval_session))
    for key, group in sessions.groupby("group", sort=False):
        yield "session", key, group
    restart = normal[normal.eval_session.eq("nov10_restart40")]
    if len(restart):
        for lo, hi in RESTART_SEGMENTS:
            yield "restart_segment", f"restart_{lo}_{hi}", restart[restart.session_cycle_index.between(lo, hi)]


def normal_results(base, normal):
    records, sessions = [], []
    for kind, identifier, part in normal_subsets(normal):
        fp, n = alarm_count(part), len(part)
        item = dict(**base, session_kind=kind, session_id=identifier, fp=fp, n=n, fpr=fp/n,
                    session_count=1 if kind == "session" else "", source_session_id=identifier,
                    **score_stats(part.score, "score"), **score_stats(part.margin, "margin"))
        records.append(item)
        if kind == "session":
            sessions.append(item)
    worst = max(sessions, key=lambda row: row["fpr"])
    tied = [f"{row['session_id']}:{row['fp']}/{row['n']}" for row in sessions if row["fpr"] == worst["fpr"]]
    records.append(dict(worst, session_kind="worst_session", session_id="worst_session", tied_sessions="|".join(tied)))
    # The macro rate is computed across session rates. A pooled denominator is misleading here.
    records.append(dict(**base, session_kind="session_macro",
                        session_id="macro_eight_sessions" if len(sessions) == 8 else "macro_available_sessions",
                        fp=np.nan, n=np.nan, fpr=np.mean([r["fpr"] for r in sessions]),
                        session_count=len(sessions), source_session_id="|".join(r["session_id"] for r in sessions)))
    return records


def pair_subset(evaluation, condition):
    return evaluation[evaluation.condition.eq(condition) |
                      (evaluation.condition.eq("normal") & evaluation.pair_condition.eq(condition))]


def anomaly_results(base, evaluation):
    records = []
    for condition, expected_controls in PAIR_NORMAL.items():
        part = pair_subset(evaluation, condition)
        labels = part.condition.eq(condition)
        if not labels.any():
            continue
        abnormal, normal = part[labels], part[~labels]
        anomaly_n, normal_n = len(abnormal), len(normal)
        assert (anomaly_n, normal_n) == (30, expected_controls)
        tp, fp = alarm_count(abnormal), alarm_count(normal)
        records.append(dict(
            **base, condition=condition, tp=tp, anomaly_n=anomaly_n, tpr=tp/anomaly_n,
            fn=anomaly_n-tp, fp=fp, normal_n=normal_n, fpr=fp/normal_n, tn=normal_n-fp,
            auroc=roc_auc_score(labels, part.score), ap=average_precision_score(labels, part.score),
            ap_prevalence=anomaly_n/(anomaly_n+normal_n), tpr_minus_fpr=tp/anomaly_n-fp/normal_n,
            **score_stats(abnormal.score, "anomaly_score"), **score_stats(normal.score, "normal_score"),
            **score_stats(abnormal.margin, "anomaly_margin"), **score_stats(normal.margin, "normal_margin"),
        ))
    return records


def model_role(split_id, model):
    if split_id != MAIN:
        return "refit_group_comparator" if model == "PCA_Q" else "refit_group_primary"
    if model in FROZEN_MODELS:
        return "frozen_baseline"
    return "primary" if model in DEFAULT_MODELS else "sensitivity"


def model_comparison(base, normal, anomalies, evaluation, calibration):
    by_session = {row["session_id"]: row for row in normal}
    overall = by_session["all"]
    actual_normal_sessions = {r["session_id"]: r["n"] for r in normal if r["session_kind"] == "session"}
    assert actual_normal_sessions == SESSIONS and overall["n"] == 259
    result = dict(
        **base, calibration_fp=alarm_count(calibration), calibration_n=len(calibration),
        normal_fp=overall["fp"], normal_n=overall["n"], normal_fpr=overall["fpr"],
        normal_macro_fpr=by_session["macro_eight_sessions"]["fpr"],
        worst_session_id=by_session["worst_session"]["source_session_id"],
        worst_session_ties=by_session["worst_session"]["tied_sessions"],
        worst_session_fp=by_session["worst_session"]["fp"], worst_session_n=by_session["worst_session"]["n"],
        worst_session_fpr=by_session["worst_session"]["fpr"],
    )
    for label, session in (("historical", "historical_test99"), ("external", "external160"), ("first", "nov10_normal40"),
                           ("restart", "nov10_restart40"), ("restart_11_40", "restart_11_40"), ("restart_21_40", "restart_21_40")):
        for key in ("fp", "n", "fpr"):
            result[f"{label}_{key}"] = by_session[session][key]
    for row in anomalies:
        condition = row["condition"]
        for key in ("tp", "tpr", "fp", "fpr", "normal_n", "auroc", "ap", "tpr_minus_fpr"):
            result[f"{condition}_{key}"] = row[key]
        result[f"{condition}_n"] = row["anomaly_n"]
    result["anomaly_tp"] = sum(row["tp"] for row in anomalies)
    result["anomaly_n"] = sum(row["anomaly_n"] for row in anomalies)
    result["anomaly_tpr"] = result["anomaly_tp"] / result["anomaly_n"]
    result["macro_matched_auroc"] = float(np.mean([row["auroc"] for row in anomalies]))
    assert len(evaluation) == overall["n"] + result["anomaly_n"]
    # The attachment defines these as observed criteria, with no numerical 'clear deterioration' bound.
    result["normal_nonworse_vs_v4"] = result["normal_fp"] <= 45
    result["restart_improved_vs_v4"] = result["restart_fp"] < 23
    for condition, reference in (("A", 10), ("B", 12), ("C", 21)):
        result[f"{condition}_retained_vs_v4"] = result[f"{condition}_tp"] >= reference
    result["meets_joint_count_criteria"] = all(result[key] for key in (
        "normal_nonworse_vs_v4", "restart_improved_vs_v4", "A_retained_vs_v4", "B_retained_vs_v4", "C_retained_vs_v4",
    ))
    return result


def matched_transitions(main):
    rows = []
    baseline = main[main.model_id.eq(BASELINE)].set_index("cycle_id")
    for model, part in main.groupby("model_id", sort=False):
        compare = part.set_index("cycle_id").loc[baseline.index]
        np.testing.assert_array_equal(compare.condition, baseline.condition)
        before, after = baseline.flagged.to_numpy(bool), compare.flagged.to_numpy(bool)
        normal = compare.condition.eq("normal").to_numpy(bool)
        transition = np.select(
            [normal & before & ~after, normal & ~before & after, normal & before & after, normal & ~before & ~after,
             ~normal & ~before & after, ~normal & before & ~after, ~normal & before & after],
            TRANSITIONS[:-1], default="both_missed",
        )
        frame = compare[["model_id", "condition", "pair_condition", "eval_session", "cohort", "session_id",
                         "cycle_index", "session_cycle_index", "leakage_group_id", "score", "threshold", "margin"]].copy()
        frame = frame.assign(baseline_model_id=BASELINE, baseline_score=baseline.score,
                             baseline_threshold=baseline.threshold, baseline_margin=baseline.margin,
                             baseline_flagged=before, model_flagged=after, transition=transition)
        assert len(frame) == len(baseline) == 349
        assert set(frame.transition.unique()).issubset(TRANSITIONS)
        for condition in ("normal", "A", "B", "C"):
            where = frame.condition.eq(condition).to_numpy(bool)
            counts = frame.loc[where, "transition"].value_counts()
            added, lost = ("new_fp", "resolved_fp") if condition == "normal" else ("added_tp", "lost_tp")
            assert int(after[where].sum()) - int(before[where].sum()) == counts.get(added, 0) - counts.get(lost, 0)
        rows.append(frame.reset_index())
    return pd.concat(rows, ignore_index=True)


def sensitivity_scopes(part):
    yield "evaluation_all", part
    normal = part[part.condition.eq("normal")]
    yield "normal_all", normal
    restart = normal[normal.eval_session.eq("nov10_restart40")]
    yield "normal_restart40", restart
    yield "normal_restart_11_40", restart[restart.session_cycle_index.ge(11)]
    yield "normal_restart_21_40", restart[restart.session_cycle_index.ge(21)]
    for condition in PAIR_NORMAL:
        yield f"pair_{condition}", pair_subset(part, condition)


def parameter_sensitivity(main):
    rows = []
    for family, members in (("LOF", ("LOF_K20", "LOF_K10", "LOF_K40")),
                            ("OCSVM", ("OCSVM_G1", "OCSVM_G025", "OCSVM_G4"))):
        for reference, model in combinations(members, 2):
            ref = main[main.model_id.eq(reference)].set_index("cycle_id")
            compare = main[main.model_id.eq(model)].set_index("cycle_id").loc[ref.index]
            for scope, original in sensitivity_scopes(ref):
                alternative = compare.loc[original.index]
                before, after = original.flagged.to_numpy(bool), alternative.flagged.to_numpy(bool)
                new, lost = int((~before & after).sum()), int((before & ~after).sum())
                common, neither = int((before & after).sum()), int((~before & ~after).sum())
                assert new + lost + common + neither == len(original)
                label = original.condition.ne("normal")
                rank_measurable = label.any() and (~label).any()
                rows.append(dict(
                    family=family, model_id=model, reference_model_id=reference, scope=scope, n=len(original),
                    spearman_r=original.score.corr(alternative.score, method="spearman"),
                    reference_alarms=int(before.sum()), model_alarms=int(after.sum()),
                    added_alarms=new, lost_alarms=lost, common_alarms=common, shared_unflagged=neither,
                    changed_alarm_fraction=(new+lost)/len(original),
                    reference_auroc=roc_auc_score(label, original.score) if rank_measurable else np.nan,
                    model_auroc=roc_auc_score(label, alternative.score) if rank_measurable else np.nan,
                    reference_ap=average_precision_score(label, original.score) if rank_measurable else np.nan,
                    model_ap=average_precision_score(label, alternative.score) if rank_measurable else np.nan,
                ))
    assert len(rows) == 48
    return rows


def group_results(base, normal, anomalies, sizes):
    assert base["split_id"] in FOLDS
    assert len(anomalies) == (3 if base["split_id"] == "B_MAIN" else 0)
    rows = []
    for row in normal:
        scope = {"overall": "normal_overall", "session": "normal_session", "restart_segment": "restart_segment"}.get(row["session_kind"])
        if scope:
            rows.append(dict(**base, scope=scope, condition="normal", session_id=row["session_id"],
                             fp=row["fp"], normal_n=row["n"], fpr=row["fpr"], train_n=sizes[0], calibration_n=sizes[1]))
    rows.extend(dict(row, scope="paired_anomaly", session_id=f"paired_{row['condition']}",
                     train_n=sizes[0], calibration_n=sizes[1]) for row in anomalies)
    return rows


def generate(study, project):
    scores, expected_fit = validate_scores(study, project)
    normal_rows, anomaly_rows, primary_rows, fold_rows = [], [], [], []
    for (split_id, model), part in scores.groupby(["split_id", "model_id"], sort=False):
        evaluation = part[part.role.eq("evaluation")]
        calibration = part[part.role.eq("calibration")]
        base = dict(split_id=split_id, model_id=model, model_role=model_role(split_id, model), threshold=part.threshold.iloc[0])
        normal = normal_results(base, evaluation[evaluation.condition.eq("normal")])
        anomalies = anomaly_results(base, evaluation)
        normal_rows.extend(normal)
        anomaly_rows.extend(anomalies)
        if split_id == MAIN:
            primary_rows.append(model_comparison(base, normal, anomalies, evaluation, calibration))
        else:
            fold_rows.extend(group_results(base, normal, anomalies, tuple(map(len, expected_fit[split_id]))))

    main = scores[scores.split_id.eq(MAIN) & scores.role.eq("evaluation")]
    primary = pd.DataFrame(primary_rows).set_index("model_id").loc[list(PRIMARY_MODELS)].reset_index()
    replay_columns = ["normal_fp", "restart_fp", "restart_11_40_fp", "A_tp", "B_tp", "C_tp"]
    expected_counts = [[45, 23, 15, 10, 12, 21], [26, 11, 7, 1, 3, 2], [51, 23, 15, 2, 5, 14]]
    np.testing.assert_array_equal(primary.iloc[:3][replay_columns], expected_counts)
    normal_sessions = pd.DataFrame(normal_rows)
    unique_main = normal_sessions[normal_sessions.split_id.eq(MAIN) & normal_sessions.session_kind.eq("session")]
    fp_by_session = unique_main.pivot(index="model_id", columns="session_id", values="fp").loc[:, list(SESSIONS)]
    changes = fp_by_session.subtract(fp_by_session.loc[BASELINE], axis="columns")
    primary["normal_sessions_with_more_fp_vs_v4"] = primary.model_id.map(changes.gt(0).sum(axis="columns"))
    primary["normal_sessions_with_fewer_fp_vs_v4"] = primary.model_id.map(changes.lt(0).sum(axis="columns"))
    primary["normal_session_fp_deltas_vs_v4"] = primary.model_id.map({
        model: "|".join(f"{session}:{int(change):+d}" for session, change in row.items() if change)
        for model, row in changes.iterrows()
    })
    outputs = {
        "PRIMARY_MODEL_COMPARISON": primary,
        "NORMAL_SESSION_METRICS": normal_sessions,
        "PAIRED_ANOMALY_METRICS": pd.DataFrame(anomaly_rows),
        "PARAMETER_SENSITIVITY": pd.DataFrame(parameter_sensitivity(main)),
        "PAIRED_ALARM_TRANSITIONS": matched_transitions(main),
        "GROUP_HELDOUT_COMPARISON": pd.DataFrame(fold_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(study / f"metrics/{name}.csv", index=False, lineterminator="\n")
    print(primary[["model_id", *replay_columns, "A_auroc", "B_auroc", "C_auroc", "meets_joint_count_criteria"]].to_string(index=False))
    print({key: len(value) for key, value in outputs.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    args = parser.parse_args()
    generate(args.study, args.project)
