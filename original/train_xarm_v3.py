"""Offline V3 feature extraction and frozen-baseline comparison."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


T0, T1 = "No3_begin_ns", "No3_end_ns"
PHASES = {"S3": (3, 4), "S4": (4, 5), "S5": (5, 6.5)}
GRIP = "ID1_3_5s_IQR"
STEP = "ID1_3_5s_max_step"
FALL = "S5_current_fall_time"
END = "ID4_end_position_deviation"
CURRENT = [phase + "_current_mean" for phase in PHASES]
NEW = [GRIP, STEP, *CURRENT, FALL, END]
ALGOS = ("Statistical", "IsolationForest", "PCA")
CASES = [*list("ABCDEF"), "B232"]
NORMALS = ["holdout", "previous20"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Sources:
    def __init__(self):
        self.hashes = {}

    def track(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            self.hashes[path] = sha(path)
        return path

    def csv(self, path, **kwargs):
        return pd.read_csv(self.track(path), float_precision="round_trip", **kwargs)

    def json(self, path):
        return json.loads(self.track(path).read_text(encoding="utf-8-sig"))


def save_csv(out, name, frame):
    frame.to_csv(out / name, index=False, encoding="utf-8-sig")


def save_json(out, name, obj):
    (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_data(h, src):
    fixed = h / "xarm_model_versions/v1_fixed_base_20261105"
    data450 = h / "No3_41g_fixed_base_main450_20261104"
    six = h / "xarm_anomaly_evaluation_v21/20261104_confirmed_outcomes"
    upper_root = h / "No3_fixed_base_upper232_20261105"
    upper = upper_root / "retry02"
    split = src.csv(fixed / "DATA_SPLIT_MANIFEST.csv")
    base = src.csv(data450 / "derived/cycle_features.csv")
    review = src.csv(data450 / "derived/reviewed_normal_cycle_manifest.csv")
    audit450 = src.json(data450 / "verification/FINAL_COLLECTION_REVIEW.json")
    assert audit450 and review.quality_valid.all() and review.physical_label.eq("confirmed_normal_41g").all()
    assert base.cycle_id.tolist() == review.cycle_id.tolist()
    base = base.merge(split[["cycle_id", "cohort", "classification"]], on="cycle_id", validate="one_to_one").copy()
    assert base.groupby("cohort").size().to_dict() == {"holdout": 100, "train": 250, "validation": 100}
    expected = dict(zip([f"b{i:02}" for i in range(1, 10)], ["train"] * 5 + ["validation"] * 2 + ["holdout"] * 2))
    assert base.stage.map(expected).equals(base.cohort)
    raw_blocks = {p.name: str(p) for p in (data450 / "runs").glob("*/*") if (p / "robot/single_queries.csv").is_file()}
    base["source"] = base.stage.map(raw_blocks)
    base["start_pose"] = np.where(base.cycle_index.eq(1), "home", "A_release")
    base["automatic_return"] = True
    base["physical_outcome"] = "confirmed_normal_41g"

    cases = src.csv(six / "REBUILT_CYCLE_FEATURES.csv")
    labels = src.csv(six / "CYCLE_LABEL_REVIEW.csv")
    assert cases[["cycle_id", "classification", "physical_outcome"]].equals(
        labels[["cycle_id", "classification", "physical_outcome"]])
    src.json(six / "DATA_QUALITY_AUDIT.json")
    excluded = cases[cases.cohort.eq("E_pilot") | cases.classification.eq("confirmed_other_outcome")]
    excluded = excluded[["cycle_id", "cohort", "classification", "physical_outcome"]].copy()
    excluded["reason"] = np.where(excluded.cohort.eq("E_pilot"), "separate pilot outside A–F comparison", "E30 successful transport")
    cases = cases[~cases.cycle_id.isin(excluded.cycle_id)].rename(columns={"begin_ns": T0, "end_ns": T1}).copy()
    cases["cohort"] = cases.cohort.replace({"N": "previous20"})
    cases["stage"] = cases.cohort
    assert cases.groupby("cohort").size().to_dict() == dict(A=30, B=30, C=30, D=30, E=29, F=30, previous20=20)

    completed = src.json(upper / "result_remaining.json")
    audited = src.json(upper / "analysis/remaining/audit.json")
    src.track(upper_root / "CLOSEOUT_KO.md")
    assert completed["completed_count"] == audited["quality_valid"] == 19
    assert completed["archive_crc_ok"] and audited["backup_members_identical"]
    more = src.csv(upper / "analysis/remaining/cycle_features.csv").copy()
    assert more.cycle_index.tolist() == list(range(2, 21))
    more = more.assign(cohort="B232", classification="confirmed_anomaly", source=completed["folder"],
                       start_pose="A_release", automatic_return=True, physical_outcome="operator confirmed completion")
    # The actual first start pose is recorded by the collector.
    events = [json.loads(line) for line in src.track(Path(completed["folder"]) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    first_pose, = [e["pose"] for e in events if e["kind"] == "verified_initial_pose"]
    more.loc[more.cycle_index.eq(2), "start_pose"] = first_pose
    failed = src.json(upper_root / "retry01/result_first.json")
    failed_manifest = src.json(Path(failed["folder"]) / "manifest.json")
    excluded.loc[len(excluded)+1000] = [failed_manifest["session_id"]+"_1", "B232", "incomplete",
                                         "No3 only, return blocked", "ID4 position completion timeout"]
    data = pd.concat([base, cases, more], ignore_index=True)
    assert len(data) == 668 and data.cycle_id.is_unique and data.source.notna().all()
    assert (data[T0] < data[T1]).all()
    return data, excluded.reset_index(drop=True)


def subset(df, clock, begin, end, include_end=False):
    return df[df[clock].ge(begin) & (df[clock].le(end) if include_end else df[clock].lt(end))]


def full_features(ina, adxl, position, describe):
    values = {}
    for key in ("current_A", "voltage_V"):
        values.update(describe(ina[key].tolist(), key))
    for key in ("x_g", "y_g", "z_g"):
        values.update(describe(adxl[key].tolist(), key))
    xyz = adxl[["x_g", "y_g", "z_g"]].to_numpy()
    values.update(describe(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1).tolist(), "mean_removed_vector_g"))
    for joint in range(1, 7):
        values.update(describe(position.loc[position.servo_id.eq(joint), "position_raw"].tolist(),
                               f"position_ID{joint}_raw"))
    return values


def max_gap(times):
    return float(np.diff(times).max()) if len(times) >= 2 else np.nan


def extract(data, src, describe, current_fall, baseline_features):
    extras, waves, parity = [], [], []
    receive = "host_receive_monotonic_ns"
    for folder, group in data.groupby("source", sort=False):
        folder = Path(folder)
        raw_meta = src.json(folder / "manifest.json")
        assert group.session_id.eq(raw_meta["session_id"]).all()
        ina = src.csv(folder / "sensors/measurements.csv",
                      usecols=[receive, "current_A", "voltage_V", "valid", "status", "seq"])
        adxl = src.csv(folder / "sensors/adxl345_1.csv", usecols=[receive, "x_g", "y_g", "z_g"])
        pos = src.csv(folder / "robot/single_queries.csv",
                      usecols=["cycle_id", "phase", "servo_id", "response_ns", "position_raw", "status", "value_valid"])
        assert ina.valid.eq(1).all() and ina.status.eq("OK").all()
        pos = pos[pos.phase.eq("No3_A_to_B") & pos.status.eq("complete") & pos.value_valid.eq(1)]
        queries = dict(tuple(pos.groupby("cycle_id")))
        for row in group.to_dict("records"):
            cid, start, end = (row[k] for k in ("cycle_id", T0, T1))
            i = subset(ina, receive, start, end, True)
            a = subset(adxl, receive, start, end, True)
            p = subset(queries[cid], "response_ns", start, end, True)
            computed = full_features(i, a, p, describe)
            difference = max(abs(computed[k]-row[k]) for k in baseline_features)
            np.testing.assert_allclose([computed[k] for k in baseline_features],
                                       [row[k] for k in baseline_features], rtol=1e-11, atol=1e-11,
                                       err_msg=f"Raw replay {cid}")
            parity.append(dict(cycle_id=cid, max_absolute_error=difference))
            t = (i[receive].to_numpy() - start) / 1e9
            out = dict(cycle_id=cid, duration_s=(end-start)/1e9,
                       id4_end_raw=p[p.servo_id.eq(4)].sort_values("response_ns").position_raw.iloc[-1],
                       id4_end_t_s=(p[p.servo_id.eq(4)].response_ns.max()-start)/1e9)
            for phase, (lo, hi) in PHASES.items():
                mask = (t >= lo) & (t < hi)
                n = int(mask.sum())
                gap = max_gap(t[mask])
                seq_gap = int((np.diff(i.seq.to_numpy()[mask]) != 1).sum())
                valid = n >= (8 if phase != "S5" else 12) and gap <= .25 and seq_gap == 0
                out.update({phase+"_n": n, phase+"_max_gap_s": gap, phase+"_sequence_breaks": seq_gap,
                            phase+"_missing_reason": "" if valid else "insufficient samples or spacing/sequence failure",
                            phase+"_current_mean": float(i.current_A.to_numpy()[mask].mean()) if valid else np.nan})
            p1 = p[p.servo_id.eq(1)].sort_values("response_ns")
            pt = (p1.response_ns.to_numpy()-start)/1e9
            select = (pt >= 3) & (pt < 5)
            gt = pt[select]
            grip = p1.position_raw.to_numpy()[select]
            gap = max_gap(gt)
            valid = len(grip) == 4 and gap <= .75 and np.all(np.diff(gt)>0)
            out.update(id1_n=len(grip), id1_max_gap_s=gap, id1_times_s=";".join(f"{v:.6f}" for v in gt),
                       id1_missing_reason="" if valid else "expected four distinct samples with max gap <=0.75s",
                       **{GRIP: np.diff(np.quantile(grip, [.25, .75]))[0] if valid else np.nan,
                          STEP: abs(np.diff(grip)).max() if valid else np.nan})
            for shift, suffix in [(-.1,"minus_100ms"),(.1,"plus_100ms")]:
                moved = (pt >= 3+shift) & (pt < 5+shift)
                values = p1.position_raw.to_numpy()[moved]
                accepted = len(values)==4 and max_gap(pt[moved])<=.75
                out["id1_window_"+suffix+"_n"] = len(values)
                out["id1_window_"+suffix+"_iqr"] = np.diff(np.quantile(values,[.25,.75]))[0] if accepted else np.nan
            grip3 = p1.position_raw.to_numpy()[(pt >= 3) & (pt < 4)]
            grip4 = p1.position_raw.to_numpy()[(pt >= 4) & (pt < 5)]
            out["ID1_S4_minus_S3_mean"] = grip4.mean()-grip3.mean() if len(grip3) and len(grip4) else np.nan
            mask = (t >= 4) & (t < 7.2)
            ref_ok = np.sum((t>=4.5)&(t<5.3)) >= 5 and np.sum((t>=6.5)&(t<6.9)) >= 3
            value, bound = current_fall(t[mask], i.current_A.to_numpy()[mask]) if ref_ok and max_gap(t[mask]) <= .25 else (np.nan,np.nan)
            out.update(**{FALL:value}, s5_fall_threshold_A=bound,
                       s5_fall_missing_reason="" if np.isfinite(value) else
                       "no two-sample crossing" if np.isfinite(bound) else "insufficient reference or excessive gap")
            extras.append(out)
            waves.extend((cid, "INA", stamp, current, volts, np.nan) for stamp, current, volts
                         in zip(t, i.current_A, i.voltage_V))
            waves.extend((cid, "ID1", stamp, np.nan, np.nan, y) for stamp, y in zip(pt, p1.position_raw))
    wave = pd.DataFrame(waves, columns=["cycle_id","stream","elapsed_s","current_A","voltage_V","ID1_raw"])
    return data.merge(pd.DataFrame(extras), on="cycle_id", validate="one_to_one"), wave, pd.DataFrame(parity)


def transformations(detector, x):
    return detector.scaler.transform(detector.imputer.transform(x))


def score_frame(data, version, strategy, model, cfg, algo, invalid, cut, values):
    return data[["cycle_id","session_id","cohort","stage","start_pose","automatic_return"]].assign(
        version=version,strategy=strategy,model_id=model,config_id=cfg,algorithm=algo,
        invalid_PCA=invalid,score=values,threshold=cut,margin=values-cut,flagged=values>cut)


def existing(h, src, data, fixed):
    v21 = h/"xarm_model_versions/v2_1_confirmed450"
    old = src.csv(fixed/"CYCLE_SCORES.csv")
    frames, checks = [], []
    for version, index_path in [("V1-베이스고정",fixed/"MODEL_INDEX.csv"),("V2.1",v21/"MODEL_INDEX.csv"),("V1",v21/"MODEL_INDEX.csv")]:
        idx = src.csv(index_path)
        if version == "V2.1":
            idx = idx[idx.strategy.isin(["A","B"])]
        if version == "V1":
            idx = idx[idx.strategy.eq("reference")]
        for r in idx.itertuples(index=False):
            path = src.track(r.path)
            assert src.hashes[path] == r.sha256
            bundle = joblib.load(path)
            cols = r.features.split(";")
            assert cols == list(bundle["features"])
            cut = bundle["threshold"]
            assert cut == r.threshold
            values = bundle["detector"].score(data[cols].to_numpy(float))
            strategy = getattr(r,"strategy","fixed_base")
            invalid = bool(getattr(r,"invalid_PCA",getattr(r,"invalid_full_rank_PCA",False)))
            result = score_frame(data,version,strategy,r.model_id,r.config_id,r.algorithm,invalid,cut,values)
            saved = old[old.model_id.eq(r.model_id) & old.strategy.eq(strategy) & old.version.eq(version)]
            joined = result.merge(saved[["cycle_id","score","flagged"]],on="cycle_id",suffixes=("","_old"))
            if not joined.empty:
                assert len(joined) == len(data)-19
                np.testing.assert_allclose(joined.score,joined.score_old,rtol=1e-10,atol=1e-12)
                assert joined.flagged.eq(joined.flagged_old).all()
            if version=="V1-베이스고정":
                assert cut == float(np.quantile(values[data.cohort.eq("validation")],.95,method="higher"))
            checks.append(dict(version=version,strategy=strategy,model_id=r.model_id,stored_cycle_scores_verified=len(joined),
                               hash_verified=True))
            frames.append(result)
    return pd.concat(frames,ignore_index=True), pd.DataFrame(checks)


def train(data, configs, seed, detector_class, threshold, out, prior, source_hash):
    train_mask = data.cohort.eq("train")
    val_mask = data.cohort.eq("validation")
    scores, index, attribution, controls = [], [], [], []
    for cfg, cols in configs.items():
        x = data[cols].to_numpy(float)
        x_train = data.loc[train_mask,cols].to_numpy(float)
        for algo in ALGOS:
            model = f"{cfg}_{algo}"
            det = detector_class(algo,seed).fit(x_train)
            z = transformations(det,x)
            np.testing.assert_allclose(det.imputer.statistics_, np.nanmedian(x_train,axis=0))
            np.testing.assert_allclose(det.scaler.mean_, det.imputer.transform(x_train).mean(axis=0))
            values = det.score(x)
            cut = threshold(values[val_mask])
            assert cut == float(np.quantile(values[val_mask],.95,method="higher"))
            bundle = dict(detector=det, features=cols, threshold=cut, seed=seed, version="V3",
                          configuration=cfg, training_code_sha256=source_hash,
                          training_cycle_ids=data.loc[train_mask,"cycle_id"].tolist(),
                          validation_cycle_ids=data.loc[val_mask,"cycle_id"].tolist(),
                          evaluation_cycle_ids=data.loc[~(train_mask|val_mask),"cycle_id"].tolist())
            path = out/"models"/f"{model}.joblib"
            joblib.dump(bundle,path)
            reopened = joblib.load(path)
            np.testing.assert_array_equal(values,reopened["detector"].score(x))
            np.testing.assert_array_equal(values>cut,reopened["detector"].score(x)>reopened["threshold"])
            if cfg in ("R0","R1"):
                old = prior[prior.version.eq("V1-베이스고정") & prior.model_id.eq(
                    f"{'A2' if cfg=='R0' else 'A7'}_{algo}")].set_index("cycle_id").loc[data.cycle_id]
                np.testing.assert_allclose(values,old.score.to_numpy(),rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(cut,old.threshold.iloc[0],rtol=1e-10,atol=1e-12)
                np.testing.assert_array_equal(values>cut,old.flagged.to_numpy())
                controls.append(dict(model_id=model,verified_cycles=len(values),
                                     score_max_abs_error=float(np.max(np.abs(values-old.score.to_numpy()))),
                                     threshold_abs_error=abs(cut-old.threshold.iloc[0]),exact_flags=True))
            rank = int(det.model.n_components_) if algo=="PCA" else None
            variance = float(det.model.explained_variance_ratio_.sum()) if algo=="PCA" else None
            invalid = rank==len(cols)
            index.append(dict(version="V3",model_id=model,config_id=cfg,algorithm=algo,
                              features=";".join(cols),feature_count=len(cols),threshold=cut,
                              seed=seed,train_n=int(train_mask.sum()),validation_n=int(val_mask.sum()),
                              PCA_rank=rank,PCA_variance=variance,invalid_PCA=invalid,sha256=sha(path),path=str(path)))
            frame = score_frame(data,"V3","fixed_base",model,cfg,algo,invalid,cut,values)
            scores.append(frame)
            if algo in ("Statistical","PCA"):
                if algo == "PCA":
                    z = z - det.model.inverse_transform(det.model.transform(z))
                parts = z**2/len(cols)
                np.testing.assert_allclose(parts.sum(axis=1),values)
                a = pd.DataFrame(parts,columns=cols).assign(cycle_id=data.cycle_id,cohort=data.cohort,
                                                           model_id=model,flagged=frame.flagged,score=values)
                attribution.append(a.melt(id_vars=["cycle_id","cohort","model_id","flagged","score"],
                                          var_name="feature",value_name="contribution"))
    return pd.concat(scores),pd.DataFrame(index),pd.concat(attribution),pd.DataFrame(controls)


def metrics(scores):
    summaries, comparison, blocks = [], [], []
    keys = ["version","strategy","model_id","config_id","algorithm","invalid_PCA"]
    for key,g in scores.groupby(keys):
        meta = dict(zip(keys,key))
        cut = g.threshold.iloc[0]
        for name, x in g.groupby("cohort"):
            summaries.append(dict(**meta,cohort=name,n=len(x),flagged=int(x.flagged.sum()),rate=x.flagged.mean(),
                                  score_median=x.score.median(),score_q10=x.score.quantile(.1),score_q90=x.score.quantile(.9),
                                  threshold=cut,median_margin=x.margin.median()))
        for (name,session), x in g[g.cohort.isin(NORMALS)].groupby(["cohort","session_id"]):
            blocks.append(dict(**meta,cohort=name,session_id=session,n=len(x),FP=int(x.flagged.sum()),FPR=x.flagged.mean()))
        for normal in NORMALS:
            n = g[g.cohort.eq(normal)]
            for name in CASES:
                a = g[g.cohort.eq(name)]
                truth = np.r_[np.zeros(len(n)),np.ones(len(a))]
                raw = np.r_[n.score,a.score]
                tp,fp = int(a.flagged.sum()),int(n.flagged.sum())
                recall,fpr = a.flagged.mean(),n.flagged.mean()
                comparison.append(dict(**meta,cohort=name,normal=normal,TP=tp,FN=len(a)-tp,FP=fp,TN=len(n)-fp,
                                       anomaly_n=len(a),normal_n=len(n),TPR=recall,FPR=fpr,
                                       AUROC=roc_auc_score(truth,raw),PR_AUC_AP=average_precision_score(truth,raw),
                                       TPR_minus_FPR=recall-fpr,balanced_accuracy=(recall+1-fpr)/2,
                                       precision=tp/(tp+fp) if tp+fp else np.nan,
                                       F1=2*tp/(len(a)+tp+fp),positive_fraction=truth.mean()))
    return pd.DataFrame(summaries),pd.DataFrame(comparison),pd.DataFrame(blocks)


def diagnostics(data, scores, attribution, index, out):
    base = data[data.cohort.eq("train")]
    val = data[data.cohort.eq("validation")]
    limits = val[NEW].quantile([.05,.95])
    shift = []
    for name,g in data.groupby("cohort"):
        for c in NEW+["position_ID1_raw_range","position_ID1_raw_std","id4_end_raw","id4_end_t_s","duration_s",
                      "id1_n","id1_max_gap_s","ID1_S4_minus_S3_mean"]:
            x = g[c]
            shift.append(dict(cohort=name,feature=c,n=len(x),valid=x.count(),q10=x.quantile(.1),median=x.median(),
                              q90=x.quantile(.9),train_min=base[c].min(),train_max=base[c].max(),
                              below_train=int(x.lt(base[c].min()).sum()),above_train=int(x.gt(base[c].max()).sum())))
    save_csv(out,"features/DISTRIBUTIONS.csv",pd.DataFrame(shift))
    save_csv(out,"features/IQR_MAX_STEP_CORRELATION.csv",pd.DataFrame([
        dict(cohort=name,pearson=g[[GRIP,STEP]].corr().iloc[0,1],spearman=g[[GRIP,STEP]].corr(method="spearman").iloc[0,1])
        for name,g in data.groupby("cohort")]))
    primary = attribution.copy()
    primary["share"] = primary.contribution.div(primary.score.where(primary.score.gt(0)))
    save_csv(out,"metrics/CONTRIBUTIONS.csv",primary)
    save_csv(out,"metrics/CONTRIBUTION_SUMMARY.csv",primary.groupby(["model_id","cohort","flagged","feature"])
             .agg(n=("share","size"),mean_contribution=("contribution","mean"),mean_share=("share","mean")).reset_index())
    primary["group"] = np.select([primary.feature.str.startswith("ID"),primary.feature.str.startswith("S"),
                                  primary.feature.str.startswith("position"),primary.feature.str.contains("_g_")],
                                 ["new_position","phase_current","whole_position","acceleration"],default="electrical")
    save_csv(out,"metrics/CONTRIBUTION_GROUPS.csv",primary.groupby(["model_id","cohort","flagged","cycle_id","group"])
             .contribution.sum().groupby(level=[0,1,2,4]).mean().rename("mean_group_contribution").reset_index())
    miss = scores[scores.version.eq("V3") & ~scores.invalid_PCA & scores.cohort.isin(CASES) & ~scores.flagged].copy()
    p95 = scores[scores.version.eq("V3") & scores.cohort.eq("holdout")].groupby("model_id").score.quantile(.95)
    miss = miss.join(p95.rename("holdout_p95"),on="model_id").merge(data[["cycle_id",*NEW]],on="cycle_id")
    outside = miss[NEW].lt(limits.loc[.05]) | miss[NEW].gt(limits.loc[.95])
    used = index.set_index("model_id").features.map(lambda s:set(s.split(";")))
    miss["outside_validation_input"] = [";".join(c for c,b in zip(NEW,row) if b and c in used[model])
                                        for model,row in zip(miss.model_id,outside.to_numpy())]
    miss["diagnostic_class"] = np.select([miss.score.gt(miss.holdout_p95),miss.outside_validation_input.ne("")],
                                         ["score above normal P95 but below cutoff","new input shifted, score within normal P95"],
                                         default="no large change in monitored new inputs")
    save_csv(out,"metrics/FALSE_NEGATIVES.csv",miss)
    save_csv(out,"metrics/POSE_STRATIFIED.csv",scores.groupby(["version","strategy","model_id","cohort","start_pose","automatic_return"])
             .agg(n=("score","size"),flagged=("flagged","sum"),score_median=("score","median")).reset_index())
    return miss


def ablations(distribution, comparison):
    normal = distribution[distribution.version.eq("V3") & distribution.cohort.isin(NORMALS)]
    anomalies = comparison[comparison.version.eq("V3")]
    pairs = [("R1","V3-A"),("R1","V3-B"),("V3-B","V3-C"),("V3-C","V3-D"),
             ("V3-D","V3-E"),("V3-C","V3-F"),("V3-C","V3-G"),("V3-C-S34","V3-C"),
             ("V3-C","V3-C-fall")]
    rows = []
    for old,new in pairs:
        x = anomalies[anomalies.config_id.eq(old)]
        y = anomalies[anomalies.config_id.eq(new)]
        joined = x.merge(y,on=["algorithm","cohort","normal"],suffixes=("_old","_new"))
        for r in joined.itertuples(index=False):
            rows.append(dict(before=old,after=new,algorithm=r.algorithm,condition=r.cohort,normal=r.normal,
                             delta_FPR=r.FPR_new-r.FPR_old,delta_TPR=r.TPR_new-r.TPR_old,
                             delta_AUROC=r.AUROC_new-r.AUROC_old))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    h = args.project.resolve()/"hardware"
    out = args.output.resolve()
    if out.exists() and any(p.name not in ("analysis_code",) for p in out.iterdir()):
        raise RuntimeError("Use a new output directory")
    for folder in ("models","metrics","features","figures","verification"):
        (out/folder).mkdir(parents=True,exist_ok=True)
    src = Sources()
    source_hash = sha(__file__)
    fixed = h/"xarm_model_versions/v1_fixed_base_20261105"
    config = src.json(fixed/"TRAINING_CONFIG.json")
    legacy = h/"No3_41g_main_20261101/analysis_code/baselines_v1"
    phase_module = h/"No3_waveform_comparison_20261103/analysis_code"
    sys.path[:0] = [str(legacy),str(phase_module)]
    src.track(legacy/"normal_models.py")
    src.track(phase_module/"compare_no3_waveforms.py")
    from normal_models import Detector, threshold
    from compare_no3_waveforms import current_fall
    stats_path = src.track(h/"normal_main_recheck_v2_20261031/pc/pilot_analysis.py")
    spec = importlib.util.spec_from_file_location("v3_legacy_feature_statistics",stats_path)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    base, excluded = load_data(h,src)
    external, multisensor = (config["configurations"][c] for c in ("A2","A7"))
    data,wave,parity = extract(base,src,stats.describe,current_fall,multisensor)
    data = data.copy()
    train_data = data[data.cohort.eq("train")]
    id4ref = float(train_data.id4_end_raw.median())
    data[END] = (data.id4_end_raw-id4ref).abs()
    old_features = src.csv(h/"xarm_position_features_20261105/POSITION_FEATURES.csv",
                           usecols=["cycle_id","joint","carry_3_5_iqr"])
    compare = data.merge(old_features[old_features.joint.eq(1)][["cycle_id","carry_3_5_iqr"]],on="cycle_id")
    assert len(compare) == 530
    np.testing.assert_allclose(compare[GRIP],compare.carry_3_5_iqr,equal_nan=True)
    old_changes = src.csv(h/"xarm_position_features_20261105/ID1_3_TO_5_CHANGES.csv",
                          usecols=["cycle_id","biggest_increase_raw","biggest_decrease_raw"])
    c = data.merge(old_changes,on="cycle_id")
    np.testing.assert_allclose(c[STEP],c[["biggest_increase_raw","biggest_decrease_raw"]].abs().max(axis=1))
    assert len(c) == 80
    core = [*multisensor,GRIP,*CURRENT]
    replace = [c for c in core if c!="position_ID1_raw_range"]
    configs = {"R0":external,"R1":multisensor,"V3-A":[*multisensor,GRIP],"V3-B":[*multisensor,*CURRENT],
               "V3-C":core,"V3-D":replace,"V3-E":[c for c in replace if c!="position_ID1_raw_std"],
               "V3-F":[*multisensor,STEP,*CURRENT],"V3-G":[*core,STEP],
               "V3-C-S34":[c for c in core if c!="S5_current_mean"]}
    eligible_fall = data[data.cohort.isin(["train","validation"])].groupby("cohort")[FALL].apply(lambda s:s.notna().mean())
    if eligible_fall.ge(.9).all():
        configs["V3-C-fall"] = [*core,FALL]
    prior,reference_checks = existing(h,src,data,fixed)
    print("Existing frozen scores and original thresholds checked",flush=True)
    new,model_index,attribution,control_checks = train(data,configs,config["seed"],Detector,threshold,out,prior,source_hash)
    scores = pd.concat([prior,new],ignore_index=True)
    distribution,comparison,session = metrics(scores)
    save_csv(out,"DATA_MANIFEST.csv",data[["cycle_id","session_id","cycle_index","cohort","stage","classification",
               "physical_outcome","condition_id","payload_mass_g","frame_added_mass_g","start_pose","automatic_return",
               "source",T0,T1]])
    save_csv(out,"features/CYCLE_FEATURES.csv",data[["cycle_id","cohort",*multisensor,*NEW,*[
        c for c in data.columns if c.startswith(("id1_","id4_","s5_fall_","S3_","S4_","S5_")) and c not in NEW],
        "ID1_S4_minus_S3_mean","duration_s"]])
    save_csv(out,"MODEL_INDEX.csv",model_index)
    save_csv(out,"metrics/CYCLE_SCORES.csv",scores)
    save_csv(out,"metrics/SCORE_DISTRIBUTIONS.csv",distribution)
    save_csv(out,"metrics/ANOMALY_METRICS.csv",comparison)
    save_csv(out,"metrics/ABLATION_DELTAS.csv",ablations(distribution,comparison))
    save_csv(out,"metrics/NORMAL_SESSION_FPR.csv",session)
    maxima = session.groupby(["version","strategy","model_id"]).FPR.max().rename("max_observed_normal_group_FPR").reset_index()
    save_csv(out,"metrics/MAX_NORMAL_FPR.csv",maxima)
    save_csv(out,"verification/EXCLUDED_CYCLES.csv",excluded)
    save_csv(out,"verification/RAW_FEATURE_PARITY.csv",parity)
    save_csv(out,"verification/FROZEN_MODEL_REPLAY.csv",reference_checks)
    save_csv(out,"verification/R0_R1_PARITY.csv",control_checks)
    misses = diagnostics(data,scores,attribution,model_index,out)
    from plot_xarm_v3 import figures
    figures(out,data,scores,distribution,comparison,wave,misses)
    save_json(out,"FEATURE_DEFINITIONS.json",dict(clock="host monotonic, No.3 command write completion",
              existing="Native samples, inclusive No.3 end; matches archived 23 features",
              windows=PHASES,id1=dict(window=[3,5],required_samples=4,maximum_gap_s=.75,
                  quantile="linear Q75-Q25",max_step="maximum absolute adjacent observed step"),
              current=dict(windows="half-open",minimum_samples=[8,8,12],max_gap_s=.25,
                           no_sequence_breaks=True), fall=dict(raw_window=[4,7.2],high_reference=[4.5,5.3],
                  low_reference=[6.5,6.9],first_crossing=[5.2,6.6],rule="two consecutive samples below midpoint",
                  minimum_reference_samples=[5,3]),id4=dict(normal_train_median_raw=id4ref,
                  reference_pose="No.3 home endpoint",included_in_models=False,
                  reason="completion gate censors normal range and differs for B232")))
    hashes={str(p):digest for p,digest in src.hashes.items()}
    save_json(out,"TRAINING_CONFIG.json",dict(version="V3",seed=config["seed"],blocks=config["blocks"],
              configurations=configs,algorithms=list(ALGOS),IF_n_estimators=200,PCA_explained_variance=.95,
              scaler="StandardScaler fit on train only",imputer="median train only",
              threshold="validation q95 higher, score > threshold",fall_minimum_train_and_validation_coverage=.9,
              fall_coverage=eligible_fall.to_dict(),training_code_sha256=source_hash,python=sys.version,
              sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,input_sha256=hashes))
    assert all(sha(path)==digest for path,digest in src.hashes.items())
    save_json(out,"verification/VERIFICATION.json",dict(cycles=len(data),existing_raw_feature_replays=len(parity),
              id1_iqr_replayed=len(compare),id1_max_step_replayed=len(c),model_count=len(model_index),
              r0_r1_models_with_identical_flags=len(control_checks),input_hashes_unchanged=len(hashes),
              stored_models_verified=len(reference_checks),reloaded_models=len(model_index),
              hardware_access=0,thresholds_fit_on_normal_validation_only=True))
    print(model_index.groupby("algorithm").size().to_string(),flush=True)


if __name__=="__main__":
    main()
