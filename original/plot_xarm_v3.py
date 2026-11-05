"""Static diagnostics for the V3 offline experiment."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ["Statistical","IsolationForest","PCA"]
CASES = list("ABCDEF")+["B232"]


def figures(out, data, scores, distribution, comparison, wave, misses):
    root = out/"figures"

    def save(fig, name):
        fig.savefig(root/f"{name}.png",dpi=145)
        plt.close(fig)

    d = distribution[distribution.version.eq("V3")]
    cfg = d.config_id.unique()
    fig, axes = plt.subplots(3,1,figsize=(11,8),sharex=True,layout="constrained")
    for ax,algo in zip(axes,ALGORITHMS):
        sub = d[d.algorithm.eq(algo)&d.cohort.isin(["holdout","previous20"])]
        table = sub.pivot(index="config_id",columns="cohort",values="rate").reindex(cfg)
        table.plot.bar(ax=ax,rot=0,color=["#397ca4","#bf6b40"])
        ax.set(title=algo,ylabel="False positive rate",ylim=(0,1))
    save(fig,"01_normal_FPR")

    fig, axes = plt.subplots(1,3,figsize=(14,5),sharey=True,layout="constrained")
    cmp = comparison[comparison.version.eq("V3") & comparison.normal.eq("holdout")]
    for ax,algo in zip(axes,ALGORITHMS):
        table = cmp[cmp.algorithm.eq(algo)].pivot(index="config_id",columns="cohort",values="TPR").reindex(index=cfg,columns=CASES)
        image = ax.imshow(table,aspect="auto",vmin=0,vmax=1,cmap="Blues")
        ax.set(xticks=range(len(CASES)),xticklabels=CASES,yticks=range(len(cfg)),yticklabels=cfg,title=algo)
        for (i,j),v in np.ndenumerate(table.to_numpy()):
            ax.text(j,i,f"{v:.0%}",ha="center",va="center",fontsize=8,color="white" if v>.6 else "black")
    fig.colorbar(image,ax=axes,label="TPR")
    save(fig,"02_anomaly_TPR_heatmap")

    def boxes(ax, feature, names):
        values = [data[data.cohort.eq(name)][feature].dropna() for name in names]
        ax.boxplot(values,tick_labels=names,showfliers=False)
        ax.set_title(feature)
        ax.tick_params(axis="x",labelrotation=18)

    names = ["holdout","previous20","D","E","F"]
    fig, axes = plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for ax,feature in zip(axes.ravel(),["ID1_3_5s_IQR","ID1_3_5s_max_step","id1_n","id1_max_gap_s"]):
        boxes(ax,feature,names)
    save(fig,"03_grip_features_and_sampling")

    fig,axes = plt.subplots(1,3,figsize=(13,4.6),layout="constrained")
    for ax,phase in zip(axes,["S3","S4","S5"]):
        boxes(ax,phase+"_current_mean",["holdout","previous20","A","B","C","B232"])
        ax.set_ylabel("A")
    save(fig,"04_phase_current")

    fig,axes = plt.subplots(1,3,figsize=(13,4.4),layout="constrained")
    for ax,feature in zip(axes,["position_ID1_raw_range","ID1_3_5s_IQR","ID1_3_5s_max_step"]):
        boxes(ax,feature,names)
    save(fig,"05_old_and_new_position_features")

    fig,axes = plt.subplots(1,2,figsize=(10,4.6),layout="constrained")
    valid = comparison[comparison.version.eq("V3") & ~comparison.invalid_PCA]
    for ax,normal in zip(axes,["holdout","previous20"]):
        for case in CASES:
            g = valid[valid.normal.eq(normal)&valid.cohort.eq(case)]
            ax.scatter(g.AUROC,g.TPR,label=case,s=14,alpha=.6)
        ax.set(xlim=(0,1.02),ylim=(0,1.02),xlabel="AUROC",ylabel="Frozen cutoff TPR",title=normal)
        ax.legend(fontsize=7,ncol=2)
    save(fig,"06_ranking_vs_detection")

    fig,axes = plt.subplots(3,1,figsize=(11,9),layout="constrained")
    group_names = ["validation","holdout","previous20",*CASES]
    for ax,algo in zip(axes,ALGORITHMS):
        sample = scores[scores.version.eq("V3")&scores.model_id.eq("V3-C_"+algo)]
        ax.boxplot([sample[sample.cohort.eq(n)].score for n in group_names],tick_labels=group_names)
        ax.axhline(sample.threshold.iloc[0],color="#a63734",linestyle="--")
        ax.set(title="V3-C "+algo,ylabel="Score")
    save(fig,"07_score_distributions")

    # Match only for illustrating failure modes, without assigning a paired trial.
    g = misses[misses.model_id.eq("V3-C_IsolationForest")]
    normal = scores[scores.version.eq("V3")&scores.model_id.eq("V3-C_IsolationForest")&scores.cohort.eq("holdout")]
    normal_id = normal.iloc[(normal.score-normal.score.median()).abs().to_numpy().argmin()].cycle_id
    fig,axes = plt.subplots(2,3,figsize=(12,7),layout="constrained")
    for ax,case in zip(axes.ravel(),list("ABCDEF")):
        candidates = g[g.cohort.eq(case)].sort_values("score")
        if candidates.empty:
            ax.set_title(case+" | no misses")
            ax.axis("off")
            continue
        cid = candidates.iloc[len(candidates)//2].cycle_id
        for label,cycle,color in [("normal holdout",normal_id,"#367899"),("missed "+case,cid,"#b34832")]:
            c = wave[wave.cycle_id.eq(cycle)]
            for stream,feature,style,axis in [("INA","current_A","-",ax),("INA","voltage_V",":",None),("ID1","ID1_raw","--",None)]:
                trace = c[c.stream.eq(stream)]
                if axis is None:
                    continue
                axis.plot(trace.elapsed_s,trace[feature],style,color=color,label=label)
        ax.set(xlim=(2.5,6.5),title=case,ylabel="Current (A)",xlabel="No.3 time (s)")
        ax.legend(fontsize=7)
    save(fig,"08_missed_current")

    fig,axes = plt.subplots(3,1,figsize=(10,8),layout="constrained")
    for ax,(stream,col) in zip(axes,[("INA","current_A"),("INA","voltage_V"),("ID1","ID1_raw")]):
        for name in ["A","C","D","E","F"]:
            candidate = g[g.cohort.eq(name)]
            if candidate.empty:
                continue
            trace = wave[wave.cycle_id.eq(candidate.iloc[0].cycle_id)&wave.stream.eq(stream)]
            ax.plot(trace.elapsed_s,trace[col],label="missed "+name,alpha=.8)
        n = wave[wave.cycle_id.eq(normal_id)&wave.stream.eq(stream)]
        ax.plot(n.elapsed_s,n[col],label="normal holdout",color="black",linewidth=1.5)
        ax.set(xlim=(2.5,6.5),ylabel=col,xlabel="No.3 time (s)")
        ax.legend(fontsize=7,ncol=4)
    save(fig,"09_missed_current_voltage_position")
