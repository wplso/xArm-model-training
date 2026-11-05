"""Generate V3.1 comparison artifacts and freeze later evaluation candidates."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train_xarm_v31 import (NORMALS, REFERENCES, REPRESENTATIVES, SIGNALS, S3, S4, S5,
                            RMS, STD, P95, DIFF, RATIO, distributions)


DOSE_CASES = ["A121", "A193", "C182", "C462"]
DOSE_PLOT = [["previous20","A","normal_a","A121","normal_a2","A193"],
             ["previous20","C","normal_c","C182","normal_c2","C462"]]


def comparisons(normals, primary, index):
    n = normals[normals.grouping.eq("cohort") & normals.group.isin(NORMALS)]
    table = n.pivot(index=["version","model_id"], columns="group", values="FPR")
    worst = normals[normals.grouping.eq("stage") & normals.cohort.isin(NORMALS)]
    table["worst_normal_FPR"] = worst.groupby(["version","model_id"]).FPR.max()
    tpr = primary.pivot(index=["version","model_id"], columns="condition", values="TPR")
    table = table.join(tpr).reset_index().merge(index[["version","model_id","algorithm","config_id",
        "invalid_PCA","feature_count"]], on=["version","model_id"], validate="one_to_one")
    table["macro_A_to_F_TPR"] = table[list("ABCDEF")].mean(axis=1)
    table["weak_load_TPR"] = table[["A121","C182"]].mean(axis=1)
    table["external_mean_FPR"] = table[NORMALS[1:]].mean(axis=1)
    table["balanced_development_score"] = (
        table[["macro_A_to_F_TPR","weak_load_TPR","A193","C462"]].mean(axis=1)
        - table[["holdout","external_mean_FPR","worst_normal_FPR"]].mean(axis=1))
    return table


def contrasts(data):
    rows = []
    for condition in ["A","B","C","B232",*DOSE_CASES]:
        reference = REFERENCES[condition]
        abnormal, normal = (data[data.cohort.eq(c)] for c in (condition, reference))
        for feature in SIGNALS:
            a, n = abnormal[feature].dropna(), normal[feature].dropna()
            if a.empty or n.empty:
                continue
            auc = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(n))], np.r_[a, n])
            rows.append(dict(condition=condition, reference=reference, feature=feature, anomaly_n=len(a),
                             normal_n=len(n), anomaly_median=a.median(), normal_median=n.median(),
                             median_delta=a.median()-n.median(), increasing_value_AUROC=auc,
                             direction_free_AUROC=max(auc,1-auc),
                             anomaly_in_normal_q10_q90=int(a.between(n.quantile(.1), n.quantile(.9)).sum())))
    return pd.DataFrame(rows)


def figures(out, data, cases, scores, pos, endpoints, table, picked, wave):
    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(out/"figures"/f"{filename}.png", dpi=140)
        plt.close(fig)

    def label(c):
        name = {"A":"A82","B":"B82","C":"C82"}.get(c,c)
        return f"{name}\n(n={int(data.cohort.eq(c).sum())})"

    def ticks(ax, labels):
        ax.set_xticks(range(len(labels)), [label(c) for c in labels], fontsize=8)

    def box(ax, frame, value, cohorts):
        ax.boxplot([frame.loc[frame.cohort.eq(c),value].dropna() for c in cohorts],
                   positions=range(len(cohorts)), showfliers=False)
        ticks(ax, cohorts)

    fig, axes = plt.subplots(2,2,figsize=(13,7),sharex="col")
    for i, cohorts in enumerate(DOSE_PLOT):
        for ax, features in zip(axes[:,i], [[S3,S4,S5],["current_A_rms","S3_current_rms",RMS,"S5_current_rms"]]):
            for f in features:
                group = data.groupby("cohort")[f]
                m = group.median().reindex(cohorts)
                ax.errorbar(range(len(cohorts)), m,
                            yerr=[m-group.quantile(.1).reindex(cohorts),group.quantile(.9).reindex(cohorts)-m],
                            marker="o", label=f)
            ax.legend(fontsize=8)
            ax.set_ylabel("A, median and Q10–Q90")
            ticks(ax, cohorts)
    axes[0,0].set_title("Payload 41/82/121/193 g")
    axes[0,1].set_title("Lower frame added 0/82/182/462 g")
    save(fig,"01_current_by_mass")

    models = [*REPRESENTATIVES[2:], *picked.values()]
    fig, axes = plt.subplots(len(models),2,figsize=(14,2.35*len(models)))
    for row, model in zip(axes, models):
        frame = scores[scores.model_id.eq(model)]
        for ax, cohorts in zip(row, DOSE_PLOT):
            box(ax,frame,"score",cohorts)
            ax.axhline(frame.threshold.iloc[0],color="firebrick",linestyle="--")
            ax.set_title(model,fontsize=9)
    save(fig,"02_scores_by_mass_and_paired_controls")

    new = table[table.version.eq("V3.1")].set_index("model_id")
    auroc = cases.pivot(index="model_id",columns="condition",values="AUROC")
    fig, axes = plt.subplots(1,3,figsize=(14,11),sharey=True)
    for ax, matrix, title in zip(axes, [new[["holdout","previous20","worst_normal_FPR"]],
            new[DOSE_CASES],auroc.loc[new.index,DOSE_CASES]], ["Normal FPR","Dose TPR","Dose AUROC (20+10)"]):
        im = ax.imshow(matrix,aspect="auto",vmin=0,vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=25)
        ax.set_yticks(range(len(matrix)),matrix.index,fontsize=6)
        fig.colorbar(im,ax=ax,fraction=.034)
    save(fig,"03_S4_ablation")

    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    for c, color in [("normal_c","tab:blue"),("C182","tab:orange")]:
        ids = data.loc[data.cohort.eq(c),"cycle_id"]
        traces = wave[wave.cycle_id.isin(ids)&wave.stream.eq("INA")&wave.elapsed_s.between(2.5,7)]
        for _, tr in traces.groupby("cycle_id"):
            axes[0].plot(tr.elapsed_s,tr.current_A,color=color,alpha=.18)
        axes[0].plot([],[],color=color,label=label(c).replace("\n"," "))
    axes[0].legend()
    axes[0].set(xlabel="No.3 elapsed (s)",ylabel="A",title="Native INA226 traces")
    for k,f in enumerate([S3,S4,S5]):
        for j,c in enumerate(["normal_c","C182"]):
            d = data.loc[data.cohort.eq(c),f].dropna()
            axes[1].scatter(np.full(len(d),k+j*.15),d,color=f"C{j}",alpha=.5)
    axes[1].set_xticks([.075,1.075,2.075],["S3","S4","S5"])
    axes[1].set(title="Phase current means",ylabel="A")
    frame = scores[scores.model_id.eq("V3-D_IsolationForest")]
    cs = ["normal_c","C182","validation","previous20"]
    for i,c in enumerate(cs):
        part = frame[frame.cohort.eq(c)]
        axes[2].scatter(np.full(len(part),i),part.score,c=np.where(part.flagged,"firebrick","steelblue"),alpha=.5)
    axes[2].axhline(frame.threshold.iloc[0],linestyle="--",color="black")
    ticks(axes[2],cs)
    axes[2].set(title="V3-D IF: red = flagged",ylabel="Score")
    save(fig,"04_C182_false_negatives")

    fig, axes = plt.subplots(2,3,figsize=(15,8))
    cs = ["normal_c","C182","normal_c2","C462","B232"]
    for servo, row in zip([4,5],axes):
        for i,c in enumerate(cs):
            for _, tr in pos[pos.servo_id.eq(servo)&pos.cohort.eq(c)].groupby("cycle_id"):
                row[0].plot(tr.elapsed_s,tr.position_raw,color=f"C{i}",alpha=.15)
            row[0].plot([],[],color=f"C{i}",label=c)
        row[0].legend(fontsize=7)
        row[0].set(xlabel="No.3 elapsed (s)",ylabel="Position (raw)",title=f"ID{servo} native")
        for ax,key,title in zip(row[1:],["fixed_position_raw","last_position_raw"],
                               ["Train-selected window","Last valid; B232/C462 gates differ"]):
            box(ax,endpoints[endpoints.servo_id.eq(servo)],key,cs)
            ax.set_title(title)
    save(fig,"05_ID4_ID5_feedback")

    cohorts = ["holdout","previous20",*list("ABCDEF"),*DOSE_CASES]
    frame = table.set_index("model_id").loc[models,cohorts]
    fig,ax = plt.subplots(figsize=(15,4.8))
    im = ax.imshow(frame,aspect="auto",vmin=0,vmax=1)
    ticks(ax,cohorts)
    ax.set_yticks(range(len(models)),models)
    for i,j in np.ndindex(frame.shape):
        ax.text(j,i,f"{frame.iat[i,j]:.0%}",ha="center",va="center",
                color="white" if frame.iat[i,j]<.52 else "black")
    ax.set_title("V3 versus V3.1: FPR first two columns, TPR thereafter")
    fig.colorbar(im,ax=ax)
    save(fig,"06_V3_V31_comparison")


def summarize(out, h, v3, data, cases, normals, scores, attribution, position, endpoints,
              index, old, windows, wave):
    indexes = [old/"MODEL_INDEX.csv",h/"xarm_model_versions/v1_fixed_base_20261105/MODEL_INDEX.csv"]
    all_index = pd.concat([index,*[pd.read_csv(p,float_precision="round_trip") for p in indexes]])
    primary = cases[cases.reference.eq(cases.condition.map(REFERENCES))]
    table = comparisons(normals,primary,all_index)
    contrast = contrasts(data)
    dist = distributions(data,SIGNALS,["cohort"])
    s4 = [S4,RMS,STD,P95,DIFF,RATIO]
    attribution["sensor_group"] = attribution.feature.map(
        lambda k: "phase_current" if k.startswith(("S3_","S4_","S5_")) else "cycle_current"
        if k.startswith("current_") else "voltage" if k.startswith("voltage_") else "position"
        if k.startswith(("ID1_","position_")) else "vibration")
    group = attribution.groupby(["model_id","cohort","cycle_id","sensor_group"],as_index=False).contribution.sum()
    total = group.groupby(["model_id","cycle_id"]).contribution.transform("sum")
    group["fraction"] = group.contribution/total.where(total.gt(0))
    outputs = {
        "FEATURE_CONFIGURATIONS.csv": index[["config_id","algorithm","features","feature_count"]],
        "FEATURE_ABLATION_RESULTS.csv": table[table.version.eq("V3.1")],
        "S4_FEATURE_COMPARISON.csv": dist[dist.feature.isin(s4)],
        "metrics/FEATURE_DISTRIBUTIONS.csv": dist,
        "metrics/DOSE_FEATURE_CONTRASTS.csv": contrast,
        "metrics/S4_CORRELATIONS.csv": pd.concat([g[s4].corr(method="spearman").assign(cohort=c)
                                         for c,g in data.groupby("cohort")]).rename_axis("feature").reset_index(),
        "A_C_PAIRED_EVALUATION.csv": primary[primary.condition.isin(DOSE_CASES)],
        "LOAD_DOSE_RESPONSE.csv": primary.merge(contrast,on=["condition","reference"]),
        "THRESHOLD_VS_AUROC_ANALYSIS.csv": primary.merge(
            normals[normals.grouping.eq("cohort")&normals.group.eq("validation")][
                ["model_id","median","q10","q90"]].rename(columns=lambda c:c if c=="model_id" else "validation_"+c),
            on="model_id",validate="many_to_one"),
        "V3_V31_MATCHED_COMPARISON.csv": table,
        "metrics/FEATURE_CONTRIBUTIONS.csv": attribution.groupby(
            ["model_id","cohort","feature","sensor_group"]).contribution.agg(["median","mean"]).reset_index(),
        "metrics/SENSOR_GROUP_CONTRIBUTIONS.csv": group.groupby(["model_id","cohort","sensor_group"],
            as_index=False)[["contribution","fraction"]].median(),
        "metrics/PCA_COMPONENTS.csv": index[index.algorithm.eq("PCA")],
        "metrics/ID4_ID5_COHORT_SUMMARY.csv": distributions(endpoints,["fixed_position_raw",
             "fixed_time_deviation_raw","last_position_raw","last_elapsed_s","duration_s"],["cohort","servo_id"])}
    for name,frame in outputs.items():
        v3.save_csv(out,name,frame)
    valid = table[table.version.eq("V3.1") & ~table.invalid_PCA]
    keys = {
        "balanced": ["balanced_development_score","worst_normal_FPR","feature_count"],
        "weak_load": ["weak_load_TPR","worst_normal_FPR","balanced_development_score"],
        "low_FPR": ["worst_normal_FPR","external_mean_FPR","balanced_development_score"]}
    picked = {}
    for role,cols in keys.items():
        ascending = [c.endswith("FPR") or c=="feature_count" for c in cols]
        remain = valid[~valid.model_id.isin(picked.values())]
        picked[role] = remain.sort_values([*cols,"model_id"],ascending=[*ascending,True]).model_id.iloc[0]
    frozen = all_index[all_index.model_id.isin([*picked.values(),*REPRESENTATIVES])]
    v3.save_json(out,"FROZEN_CANDIDATES.json",dict(status="development_only_V3_retained",candidates=picked,
        models=[dict(model_id=r.model_id,version=r.version,features=r.features.split(";"),
                     threshold=r.threshold,path=r.path,sha256=r.sha256) for r in frozen.itertuples(index=False)],
        excluded_cycle_ids=data.cycle_id.tolist(),threshold_rule="strict > frozen cutoff; no refitting",
        artifacts_sha256={f:v3.sha(out/f) for f in ("TRAINING_CONFIG.json","FEATURE_DEFINITIONS.json","DATA_MANIFEST.csv")}))
    figures(out,data,primary,scores,position,endpoints,table,picked,wave)
    report(out,table,contrast,primary,data,endpoints,picked,windows)


def report(out, table, contrast, primary, data, endpoints, picked, windows):
    def save(name,content):
        (out/name).write_text(content,encoding="utf-8")

    save("INDEPENDENT_VALIDATION_PLAN_KO.md", """# 다음 날짜 독립 검증

1. 새 날짜와 세션 ID를 기록한다. 개발용 788회와 제외 회차는 재사용하지 않는다.
2. 같은 베이스·센서 고정, 동일 시작 자세, 물체 위치, 명령 간 대기, B→A 자동 원복 파일·설정을 확인한다. home과 A_release를 섞어 해석하지 않는다. 자동 원복이 불가능한 파지 실패·이탈·위치 오류는 원복 정책이 맞는 별도 평가로 둔다.
3. 41g 정상과 A121·C182를 우선 비교하고 A193·C462 및 기존 하중 조건의 범위도 확인한다. C462 종료 허용폭 차이를 별도 기록한다. 실제 운반·복귀·통신 상태와 제외 기준을 채점 전에 확정한다.
4. FROZEN_CANDIDATES.json의 특징 순서·모델 해시·임계값과 전처리 객체를 고정한다. 기존 V3와 V3.1을 같은 유효 Cycle에서 비교한다. 평가 전에 허용 FPR과 최소 TPR을 합의한다.
5. 세션별 FP/n과 TP/n, AUROC·AP 및 점수 분포를 비교한다. 결과를 확인하기 전까지 V3를 유지한다. 이번 독립 자료는 학습이나 임계값 결정에 쓰지 않는다.
""")
    save("README_KO.md", """# xArm V3.1 오프라인 재현

프로젝트 루트에서 기존 .venv/Scripts/python.exe로 analysis_code/train_xarm_v31.py를 실행하고 인수로 프로젝트 루트와 새 출력 디렉터리를 준다. V3와 동일한 Python 환경을 사용한다.

- V31_REPORT_KO.md에 결과와 해석을 정리했다.
- DATA_MANIFEST.csv는 실제 Cycle 경계와 분할, MODEL_INDEX.csv는 모델과 임계값을 기록한다.
- ANOMALY_TYPE_METRICS.csv는 정상 대조를 별도 열로 표시한다.
- VERIFICATION.json 및 verification에는 원시 재생, 저장 모델 대조, 제외 기록이 있다.
- 독립 검증은 INDEPENDENT_VALIDATION_PLAN_KO.md와 FROZEN_CANDIDATES.json을 따른다.
""")
    t = table.set_index("model_id")
    header = ["| 모델 | 정상 100 FPR | 별도 20 FPR | A121 | A193 | C182 | C462 | 기존 A/B/C/D/E/F TPR |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for model in ["A2_IsolationForest","A7_IsolationForest",*REPRESENTATIVES[2:],*picked.values()]:
        row = t.loc[model]
        values = [f"{row[c]:.0%}" for c in ["holdout","previous20",*DOSE_CASES]]
        header.append("| " + " | ".join([f"{row.version} {model}",*values,"/".join(f"{row[c]:.0%}" for c in "ABCDEF")])+" |")
    shift = ["| 특징 | A121 | A193 | C182 | C462 |","|---|---:|---:|---:|---:|"]
    for name in SIGNALS[:9]:
        values = contrast[contrast.feature.eq(name)].set_index("condition").loc[DOSE_CASES].median_delta
        shift.append("| " + " | ".join([name,*[f"{v:+.4f}" for v in values]])+" |")
    position = []
    for c,s,n in [("B232",4,"previous20"),("C182",5,"normal_c"),("C462",5,"normal_c2")]:
        a,b = (endpoints[endpoints.servo_id.eq(s)&endpoints.cohort.eq(k)] for k in (c,n))
        position.append(f"- {c} ID{s} 고정창 {a.fixed_position_raw.median():.1f} 대 {b.fixed_position_raw.median():.1f} raw "
                        f"(유효 {a.fixed_n.gt(0).sum()}/{len(a)}, 정상 {b.fixed_n.gt(0).sum()}/{len(b)}). "
                        f"마지막 위치 {a.last_position_raw.median():.1f} 대 {b.last_position_raw.median():.1f} raw, "
                        f"종료 대리 시각 {a.duration_s.median():.2f} 대 {b.duration_s.median():.2f}초.")
    s4_corr = data.loc[data.cohort.eq("train"),[S4,RMS]].corr(method="spearman").iloc[0,1]
    def compare(x,y,alg="IsolationForest"):
        a,b = (t.loc[f"V31-{i}_{alg}"] for i in (x,y))
        return f"A121 {a.A121:.0%}→{b.A121:.0%}, C182 {a.C182:.0%}→{b.C182:.0%}, 별도 정상 FPR {a.previous20:.0%}→{b.previous20:.0%}"

    lines = []
    for x,y,name in [(0,2,"S4 평균"),(4,6,"S5 추가"),(2,7,"S4 RMS"),(2,8,"S4 std"),
                     (2,13,"S4 P95"),(4,10,"구간 차이"),(4,11,"구간 비율"),(4,14,"ID1 IQR"),(14,15,"ID1 범위 제거")]:
        lines.append(f"- {name} ({x}→{y}) · IF {compare(x,y)}. PCA {compare(x,y,'PCA')}. Statistical {compare(x,y,'Statistical')}.")
    f = primary[primary.model_id.eq("V3-D_IsolationForest")&primary.condition.eq("C182")].iloc[0]
    c = primary[primary.model_id.eq(picked["balanced"])&primary.condition.eq("C182")].iloc[0]
    save("V31_REPORT_KO.md",f"""# xArm V3.1 하중 민감도 개발 평가

V3를 유지한다. V3.1은 18개 특징 구성의 Statistical·Isolation Forest·PCA 54개를 학습했다. 저장된 V3 33개와 V1-베이스고정 21개를 동일 데이터로 재평가했다.

## 데이터와 검증

788회는 고정 베이스 정상 학습 250, 검증 100, 홀드아웃 100, 별도 정상 20, 기존 이상 198, 신규 정상 40와 신규 이상 80으로 구성된다. 전처리는 학습 250회에서 적합했다. 정상 검증 100회의 q95(higher)를 초과한 경우를 이상으로 판정했다. 기존 A~F/B232는 별도 정상 20회를 우선 대조하며 홀드아웃 100회 비교는 분리했다. 신규 정상 10회와 이상 20회는 그룹 단위 대조다.

C182 최초 중단, C282 파일럿, C462 첫 미완료, B232 첫 미완료 및 E의 별도 파일럿·성공 회차를 제외했다. C462는 ID5 781±10, B232는 ID4 954±10 종료 정책을 사용했다. 신규 120회의 실제 운반·원복 확인 기록과 원시 UART/HID 감사를 확인하고 아카이브 해시·CRC를 대조했다. 원시 자료로 기존 23특징과 S3~S5, ID1 파지, 기타 V3 특징을 재현했다.

## 전류 변화

직전 정상 대비 중앙값 변화량이다. 전류 단위는 A이며 S4/S3 비율은 무단위다.

{chr(10).join(shift)}

A121→A193에서는 S3·S4가 함께 증가했다. C182→C462는 전체 RMS와 S4가 증가했으나 S3 증가폭은 감소했다. 각 하중은 순서대로 수집한 한 블록이다. 물체 형상·파지 조건·운전 이력·완료 정책의 영향을 분리할 수 없어 질량의 물리적 인과나 통계적 유의성을 확정하기 어렵다.

## Ablation

{chr(10).join(lines)}

학습 정상에서 S4 평균–RMS 순위 상관은 {s4_corr:.3f}였다. 평균에 RMS를 추가할 때의 성능 차이는 알고리즘별로 따로 확인한다. Std·P95 및 구간 차이·비율도 같은 정상 대조와 고정 임계값 규칙으로 비교했다. 파지 IQR은 네 위치 응답과 시간 정렬에 민감하다. D/E/F 및 네 신규 정상 블록의 결과를 CSV에 함께 보존했다.

Statistical의 z² 평균, PCA의 재구성 잔차 제곱 평균을 특징별·센서군별로 분해했다. PCA 성분 수·설명분산·전체 랭크 여부는 metrics/PCA_COMPONENTS.csv에 기록했다. IF는 입력을 바꿔 재학습한 Ablation으로 비교했다.

## C182의 순위와 경보

V3-D IF의 C182는 AUROC {f.AUROC:.3f}, 탐지 {f.TP}/20, 짝 정상 오탐 {f.FP}/10이다. 미탐 {f.FN}회 가운데 {f.FN_above_reference_q95}회는 짝 정상 q95보다 높았지만 검증 기준 임계값을 넘지 못했다. 이상 점수 중앙값과 임계값 차이는 {f.median_margin:+.4f}였다. 이 모델의 낮은 경보율에는 점수와 검증 기준 사이 간격이 직접 작용했다. S4의 작은 변화와 구성별 격차를 고려하면 입력과 점수 함수의 영향도 있다. 종합 후보는 {c.TP}/20, AUROC {c.AUROC:.3f}였다. 짝 정상 q95는 해석에만 썼다.

## 관절 위치

정상 학습의 표본 확보율과 IQR로 ID4 {windows["4"]["window"]}초, ID5 {windows["5"]["window"]}초를 선택했다. 실제 응답만 사용했다.

{chr(10).join(position)}

완료 허용폭과 시각이 마지막 위치에 영향을 줄 수 있다. 같은 시각의 관찰과 함께 비교했으나 한 블록씩 수집한 자료이고, 2Hz 순차 응답에서 고정창 표본이 빠질 수 있으므로 보조 특징으로 남겼다.

## 후보 비교

{chr(10).join(header)}

- 종합 후보 **{picked["balanced"]}**. A~F·증량 탐지와 여러 정상 그룹의 오탐을 함께 반영했다.
- 미세 하중 후보 **{picked["weak_load"]}**. A121·C182 탐지를 우선했다.
- 저오탐 후보 **{picked["low_FPR"]}**. 최악 정상 블록 FPR을 우선했다.

동일 데이터에서 특징과 후보를 비교했으므로 개발 결과다. 다음 날짜에는 동일 베이스·시작 자세·B→A 자동 원복 조건의 41g 정상과 A121·C182를 먼저 검증하고 A193·C462와 기존 이상으로 범위를 확인한다. 저장한 V3와 V3.1 모델 및 임계값을 고정한다. 독립 검증을 검토하기 전까지 V3를 유지한다.
""")
