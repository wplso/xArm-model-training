"""Offline normal-only baselines. Requires operator-reviewed cycles and frozen split."""
from pathlib import Path
import argparse,hashlib,json,time,sys
import joblib,numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
W=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Detector:
    def __init__(self,kind,seed=20261031):
        self.kind=kind;self.seed=seed
    def fit(self,X):
        X=np.asarray(X,float)
        if not np.isfinite(X).any(axis=0).all():raise ValueError("All-missing training feature")
        self.imputer=SimpleImputer(strategy="median",keep_empty_features=True)
        self.scaler=StandardScaler();Z=self.scaler.fit_transform(self.imputer.fit_transform(X))
        if self.kind=="IsolationForest":self.model=IsolationForest(n_estimators=200,random_state=self.seed,n_jobs=1).fit(Z)
        elif self.kind=="PCA":self.model=PCA(n_components=.95,svd_solver="full").fit(Z)
        elif self.kind!="Statistical":raise ValueError(self.kind)
        return self
    def score(self,X):
        Z=self.scaler.transform(self.imputer.transform(np.asarray(X,float)))
        if self.kind=="Statistical":return np.mean(Z**2,axis=1)
        if self.kind=="IsolationForest":return -self.model.score_samples(Z)
        return np.mean((Z-self.model.inverse_transform(self.model.transform(Z)))**2,axis=1)
def threshold(scores,target=.05):return float(np.quantile(scores,1-target,method="higher"))
def load_reviewed(run):
    review=json.loads((run/"metadata/operator_review.json").read_text(encoding="utf-8-sig"))
    confirmed=set(review["normal_cycle_ids"])
    sessions=json.loads((run/"metadata/session_split.json").read_text(encoding="utf-8-sig"))
    data=[]
    for s in sessions:
        root=Path(s["analysis"])
        f=pd.read_csv(root/"derived/cycle_features.csv")
        q=pd.read_csv(root/"quality/cycle_quality.csv")
        merged=f.merge(q[["cycle_id","cycle_valid"]],on="cycle_id",validate="one_to_one")
        merged=merged[merged.cycle_valid.eq(True)&merged.cycle_id.isin(confirmed)].copy()
        merged["dataset_role"]=s["dataset_role"];data.append(merged)
    result=pd.concat(data,ignore_index=True)
    if result.cycle_id.duplicated().any():raise ValueError("Duplicate cycle ID across split")
    if result.groupby("session_id").dataset_role.nunique().max()>1:raise ValueError("Session split leakage")
    return result,review
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("run",type=Path);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    cfg=json.loads((W/"config/model_experiment.json").read_text(encoding="utf-8"))
    data,review=load_reviewed(args.run)
    parts={role:data[data.dataset_role.eq(role)] for role in ("train","validation","evaluation")}
    counts={k:len(v) for k,v in parts.items()}
    if min(counts.values())<25:raise ValueError("Need >=25 reviewed normal cycles per prospective role; available="+str(counts))
    if args.output.exists():raise FileExistsError("New model output required; no overwriting prior evaluation")
    args.output.mkdir(parents=True)
    (args.output/"models").mkdir()
    # Freeze all choices before evaluating. No data-dependent tuning or selector.
    (args.output/"experiment_freeze.json").write_text(json.dumps({"config":cfg,"counts":counts,
        "config_sha256":sha(W/"config/model_experiment.json"),"source_sha256":sha(Path(__file__)),
        "review_sha256":sha(args.run/"metadata/operator_review.json"),"model_selection_on_test":False,
        "evaluation_scope":"same-date held-out sessions; independent new-date FPR pending"},ensure_ascii=False,indent=2),encoding="utf-8")
    metrics=[];score_rows=[]
    full_train=counts["train"]
    n_values=sorted(set([n for n in cfg["learning_curve_train_counts"] if n<=full_train]+[full_train]))
    for feature_set,names in cfg["feature_sets"].items():
        forbidden={"session_id","cycle_id","dataset_role","action_group_id","anomaly","category","setting","sample"}
        if forbidden.intersection(names):raise ValueError("Metadata entered predictor whitelist")
        for n in n_values:
            train=parts["train"].iloc[:n]
            for kind in cfg["algorithms"]:
                begin=time.perf_counter()
                detector=Detector(kind,cfg["seed"]).fit(train[names].to_numpy())
                fit_ms=(time.perf_counter()-begin)*1000
                validation_scores=detector.score(parts["validation"][names].to_numpy())
                limit=threshold(validation_scores,cfg["target_normal_FPR"])
                joblib.dump({"detector":detector,"features":names,"threshold":limit,"score_direction":"larger is more anomalous",
                    "normal_condition":"unloaded No.3 only","split":"session-level, same date","anomaly_recall_validated":False},
                    args.output/"models"/f"{feature_set}_{kind}_train{n}.joblib")
                for role,frame in parts.items():
                    begin=time.perf_counter();scores=detector.score(frame[names].to_numpy())
                    elapsed_ms=(time.perf_counter()-begin)*1000
                    flagged=scores>limit
                    metrics.append({"feature_set":feature_set,"algorithm":kind,"train_n":n,"split":role,
                        "normal_n":len(scores),"normal_false_positives":int(flagged.sum()),"normal_FPR":float(flagged.mean()),
                        "threshold":limit,"score_mean":float(scores.mean()),"score_std":float(scores.std()),
                        "score_p95":float(np.quantile(scores,.95)),"features":len(names),"fit_ms":fit_ms,
                        "mean_score_ms_per_cycle":elapsed_ms/len(scores),"AUROC":"not_available_normal_only",
                        "Recall":"not_available_normal_only","independent_date_validation":False})
                    for (_,r),score,flag in zip(frame.iterrows(),scores,flagged):
                        score_rows.append({"session_id":r.session_id,"cycle_id":r.cycle_id,"feature_set":feature_set,
                            "algorithm":kind,"train_n":n,"split":role,"score":float(score),"flagged":bool(flag)})
    pd.DataFrame(metrics).to_csv(args.output/"normal_model_metrics.csv",index=False,encoding="utf-8-sig")
    score_table=pd.DataFrame(score_rows);score_table.to_csv(args.output/"normal_scores.csv",index=False,encoding="utf-8-sig")
    score_table.groupby(["feature_set","algorithm","train_n","split","session_id"],as_index=False).agg(
        normal_n=("flagged","size"),normal_FPR=("flagged","mean"),score_mean=("score","mean")).to_csv(args.output/"session_FPR.csv",index=False,encoding="utf-8-sig")
    data.to_csv(args.output/"reviewed_cycle_features.csv",index=False,encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m=pd.DataFrame(metrics);fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,(feature_set,_) in zip(axs,cfg["feature_sets"].items()):
        subset=m[(m.feature_set==feature_set)&(m.split=="evaluation")]
        for kind,grp in subset.groupby("algorithm"):ax.plot(grp.train_n,grp.normal_FPR,"o-",label=kind)
        ax.axhline(cfg["target_normal_FPR"],ls="--",color="gray",label="validation target 5%")
        ax.set(title=feature_set,xlabel="Train cycles",ylabel="Held-out normal FPR");ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("Unloaded No.3 normal-only baselines: same-date sessions")
    fig.savefig(args.output/"normal_FPR_learning_curve.png",dpi=160);plt.close(fig)
    final=m[(m.train_n==full_train)&(m.split=="evaluation")]
    lines=["# 무부하 No.3 정상 기반 Baseline 비교","","동일 날짜의 사전 분리 세션에서 정상 FPR과 점수 분포만 평가했다. 이상 데이터가 없어 AUROC·Recall·진단 정확도를 계산하지 않았다. 다른 날짜 성능·물체 운반 일반화는 미검증이다.","",
           "| 입력 | 모델 | 정상 평가 수 | FPR |","|---|---|---:|---:|"]
    for _,x in final.iterrows():lines.append(f"| {x.feature_set} | {x.algorithm} | {x.normal_n} | {x.normal_FPR:.3f} |")
    lines+=["","학습 데이터만으로 대치/정규화를 적합했다. 정상 validation의 상위 95% 분위수(higher)로 임계값을 고정하고 score > threshold를 이상으로 표시했다. 평가값으로 특징·모델·임계값을 조정하지 않았다.",
        "","모든 특징은 사이클 종료 후 계산되는 요약값이다. 실시간 조기 탐지 결과가 아니다. 관절 위치는 실제 응답 raw의 저주파 통계이며 실제 목표 대비 추종 오차/속도가 아니다. 통합 입력은 공개 voraus-AD의 B/C 센서 구성과 같지 않다.",
        "",f"검토된 정상 수: {counts}. 학습 곡선 실행 수: {n_values}. 300/500 학습은 train 확보량 부족으로 미실행이다.",
        "","같은 날짜의 연속 블록 간 상관성과 정상 조건 범위를 고려해야 한다. 독립 날짜 정상 FPR·실제 이상 Recall을 확인하기 전 주력 모델을 확정하지 않는다.",
        "","다중 동작 데이터가 없어 동작별/통합/조건부 모델 비교는 미실행이다. No.4/5는 반복 종료·경로 검증 후 별도 수집한다. 작은 MLP Autoencoder는 기본 모델과 데이터량 검토 이후 선택적으로 검토하며 이번 필수 범위에서 제외했다."]
    (args.output/"NORMAL_MODEL_BASELINE_REPORT_KO.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"counts":counts,"models":len(metrics)//3,"results":str(args.output)},ensure_ascii=False))
if __name__=="__main__":
    sys.modules.setdefault("normal_models",sys.modules[__name__])
    Detector.__module__="normal_models"
    main()
