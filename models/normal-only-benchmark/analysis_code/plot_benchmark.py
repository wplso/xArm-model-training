"""Plot the seven predeclared normal-only algorithm benchmark comparisons."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np
import pandas as pd

MODELS = ("V4_PCA_Q", "IF_24", "IF_PC13_Q", "LOF_K20", "OCSVM_G1", "KNN_K20_MEAN",
          "LOF_K10", "LOF_K40", "OCSVM_G025", "OCSVM_G4")
MODEL_LABEL = {"V4_PCA_Q": "V4 PCA Q", "IF_24": "IF 24특징", "IF_PC13_Q": "PCA 13+Q → IF",
               "LOF_K20": "LOF k=20", "OCSVM_G1": "SVM γ×1", "KNN_K20_MEAN": "kNN k=20 평균",
               "LOF_K10": "LOF k=10", "LOF_K40": "LOF k=40", "OCSVM_G025": "SVM γ×0.25",
               "OCSVM_G4": "SVM γ×4", "PCA_Q": "PCA Q"}
FOLD_MODELS = ("PCA_Q", *MODELS[3:6])
COLORS = {"normal": "#344f68", "restart": "#aa526c", "A": "#b36a39", "B": "#3d8272", "C": "#896c9e"}
CONDITIONS = ("A", "B", "C")
LOAD_NAMES = ("A 121g", "B 상부 82g", "C 하부 182g")
SESSIONS = {
    "historical_test99": "Test99", "extra_normal17": "중단 전17", "extra_normal13": "재개13",
    "preA": "A 짝 정상", "preB": "B 짝 정상", "preC": "C 짝 정상",
    "nov10_normal40": "첫 정상40", "nov10_restart40": "재가동40",
}


def read(study, name):
    return pd.read_csv(study / "metrics" / name, float_precision="round_trip", low_memory=False)


def title(fig, heading, detail):
    fig.suptitle(heading, x=.018, y=.995, ha="left", fontsize=16, fontweight="bold")
    fig.text(.018, .995 - .32 / fig.get_figheight(), detail, va="top", fontsize=9, color="#455870")


def save(fig, out, name):
    path = out / name
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)
    assert path.stat().st_size > 1024
    print(path.name)


def axes_grid(ax, rows, columns, breaks=True):
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    if breaks:
        for y in (2.5, 5.5):
            ax.axhline(y, color="white", lw=2)


def rate_heatmap(ax, counts, totals, rows, columns, cmap, rates=False, breaks=True, rate_digits=1):
    counts, totals = np.asarray(counts, dtype=float), np.asarray(totals, dtype=float)
    assert np.isfinite(counts).all() and (totals > 0).all()
    fraction = counts / totals
    if isinstance(cmap, tuple):
        data = np.stack([plt.get_cmap(palette)(fraction[:, col])
                         for col, palette in enumerate(cmap)], axis=1)
        ax.imshow(data, aspect="auto", interpolation="nearest")
    else:
        ax.imshow(fraction, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i, j in np.ndindex(counts.shape):
        text = f"{int(counts[i,j])}/{int(totals[i,j])}"
        if rates:
            text += f"\n{fraction[i,j]:.{rate_digits}%}"
        ax.text(j, i, text, ha="center", va="center", fontsize=8.5,
                color="white" if fraction[i,j] >= .68 else "#142d44")
    axes_grid(ax, rows, columns, breaks)


def overview(primary, out):
    table = primary.set_index("model_id").loc[list(MODELS)]
    labels = [MODEL_LABEL[x] for x in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.8), sharey=True,
                             gridspec_kw={"width_ratios": [2, 3]})
    rate_heatmap(axes[0], table[["normal_fp", "restart_fp"]], table[["normal_n", "restart_n"]],
                 labels, ["전체 정상", "재가동40"], "OrRd", rates=True)
    rate_heatmap(axes[1], table[[f"{key}_tp" for key in CONDITIONS]],
                 table[[f"{key}_n" for key in CONDITIONS]], labels, LOAD_NAMES, "YlGn", rates=True)
    axes[0].set_title("정상 오탐률 · FP/n", loc="left", fontsize=10)
    axes[1].set_title("하중 이상 탐지율 · TP/n", loc="left", fontsize=10)
    title(fig, "동일 24특징에서의 정상 오탐과 하중 탐지",
          "주 실험 · 고유 정상 259회(재가동 40회 포함) · 이상 유형별 30회 · 위부터 동결 3개, 기본 3개, 민감도 4개")
    fig.subplots_adjust(left=.20, right=.99, top=.84, bottom=.1, wspace=.08)
    save(fig, out, "01_PRIMARY_FPR_TPR.png")


def session_chart(normal, out):
    frame = normal[normal.split_id.eq("PRIMARY") & normal.session_kind.eq("session")]
    table = frame.set_index(["model_id", "session_id"])
    keys = pd.MultiIndex.from_product([MODELS, SESSIONS])
    counts = table.loc[keys, "fp"].to_numpy().reshape(len(MODELS), len(SESSIONS))
    totals = table.loc[keys, "n"].to_numpy().reshape(len(MODELS), len(SESSIONS))
    fig, ax = plt.subplots(figsize=(12.3, 5.8))
    rate_heatmap(ax, counts, totals, [MODEL_LABEL[x] for x in MODELS], list(SESSIONS.values()), "OrRd")
    ax.axvline(6.5, color="#8fa3b1", lw=1.1)
    title(fig, "정상 평가 8개 수집 단위의 오탐",
          "셀은 FP/n · 수집 단위는 독립 날짜와 구분 · 재가동 첫 10회도 전체에 포함")
    fig.subplots_adjust(left=.18, right=.99, top=.86, bottom=.10)
    save(fig, out, "02_NORMAL_SESSION_FPR.png")


def paired_roc(paired, out):
    table = paired[paired.split_id.eq("PRIMARY")].set_index(["model_id", "condition"])
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 5.7), sharey=True)
    positions = np.arange(len(MODELS))
    labels = [MODEL_LABEL[x] for x in MODELS]
    for key, condition, ax in zip(CONDITIONS, LOAD_NAMES, axes, strict=True):
        frame = table.loc[pd.MultiIndex.from_product([MODELS, [key]])]
        values = frame["auroc"].to_numpy()
        ax.axvline(.5, lw=.8, color="#8296a6", ls="--")
        ax.scatter(values, positions, s=36, color=COLORS[key], zorder=3)
        for y, value in zip(positions, values, strict=True):
            dx = -6 if value >= .85 else 6
            ax.annotate(f"{value:.3f}", (value, y), xytext=(dx, 0), textcoords="offset points",
                        ha="right" if dx < 0 else "left", va="center", fontsize=8)
        ax.set_yticks(positions, labels)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlabel("AUROC")
        ax.set_title(f"{condition} · 이상 {frame.anomaly_n.iat[0]} / 정상 {frame.normal_n.iat[0]}",
                     loc="left", fontsize=9)
        ax.grid(axis="x", alpha=.15)
        for y in (2.5, 5.5):
            ax.axhline(y, color="#d8e0e8", lw=.8)
    axes[0].invert_yaxis()
    title(fig, "같은 A/B/C와 짝 정상에서의 AUROC",
          "주 실험 · 판정 임계값과 독립적인 순위 지표 · 세로 점선은 AUROC 0.5")
    fig.subplots_adjust(left=.18, right=.99, top=.83, bottom=.12, wspace=.17)
    save(fig, out, "03_PAIRED_CONDITION_AUROC.png")


def evaluation(scores, split):
    return scores.loc[scores.split_id.eq(split) & scores.role.eq("evaluation")]


def restart_margin(scores, out):
    restart = evaluation(scores, "PRIMARY")
    restart = restart[restart.eval_session.eq("nov10_restart40")]
    fig, axes = plt.subplots(5, 2, figsize=(12, 10.5), sharex=True, squeeze=False)
    for model_id, ax in zip(MODELS, axes.flat, strict=True):
        frame = restart[restart.model_id.eq(model_id)].sort_values("cycle_index")
        assert frame.cycle_index.tolist() == list(range(1, 41))
        margin = frame["margin"].to_numpy()
        np.testing.assert_allclose(margin, (frame.score - frame.threshold).to_numpy(),
                                   rtol=1e-10, atol=1e-13)
        flags = frame.flagged.eq(True)
        np.testing.assert_array_equal(margin > 0, flags)
        x = frame.cycle_index
        ax.plot(x, margin, color="#32658d", lw=1.15, marker=".", ms=3)
        ax.scatter(x[flags], margin[flags], s=17, color="#b36b46", linewidth=0, zorder=3)
        ax.axhline(0, color="#596d81", lw=.85, ls="--")
        for cut in (10.5, 20.5):
            ax.axvline(cut, color="#a9b5bf", lw=.7, ls=":")
        ax.set_xlim(.6, 40.4)
        ax.set_title(f"{MODEL_LABEL[model_id]} · {int(flags.sum())}/40", loc="left", fontsize=9)
        formatter = ScalarFormatter(useOffset=False, useMathText=False)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(axis="y", alpha=.14)
    for ax in axes[-1]:
        ax.set_xticks([1, 10, 20, 30, 40])
    fig.supylabel("이상 점수 − 모델 임계값", x=.036, fontsize=9.5)
    fig.supxlabel("11/10 재가동 정상 회차", y=.035, fontsize=9.5)
    title(fig, "재가동 정상40회의 임계값 초과분",
          "모델별 y축 스케일 · 0 초과가 경보(주황 점) · 첫 10회 포함 · 세로 점선은 하위 집계 경계")
    fig.subplots_adjust(left=.13, right=.99, top=.90, bottom=.085, hspace=.64, wspace=.26)
    save(fig, out, "04_RESTART_SCORE_MINUS_THRESHOLD.png")


def transition_heatmap(ax, changes, totals, row_names, col_names, normal):
    added, lost = changes
    benefit = (lost - added if normal else added - lost) / totals
    ax.imshow(benefit, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-.75, vcenter=0, vmax=.75),
              aspect="auto", interpolation="nearest")
    for i, j in np.ndindex(benefit.shape):
        color = "white" if abs(benefit[i, j]) >= .57 else "#153048"
        ax.text(j, i, f"−{lost[i,j]} / +{added[i,j]}", ha="center", va="center",
                fontsize=8.5, color=color)
    axes_grid(ax, row_names, col_names, breaks=False)
    for y in (1.5, 4.5):
        ax.axhline(y, lw=2, color="white")


def transitions(scores, out):
    evals = evaluation(scores, "PRIMARY")
    metadata = evals.loc[evals.model_id.eq("V4_PCA_Q"), ["cycle_id", "condition", "eval_session"]].set_index("cycle_id")
    flags = evals.pivot(index="cycle_id", columns="model_id", values="flagged").loc[metadata.index]
    assert flags.notna().all().all(), "All evaluated cycles need a saved alarm decision"
    flags = flags.eq(True)
    cohorts = [metadata.condition.eq("normal"), metadata.eval_session.eq("nov10_restart40"),
               *(metadata.condition.eq(c) for c in CONDITIONS)]
    added, lost = [], []
    for model in MODELS[1:]:
        a = flags[model] & ~flags["V4_PCA_Q"]
        b = ~flags[model] & flags["V4_PCA_Q"]
        added.append([int(a[mask].sum()) for mask in cohorts])
        lost.append([int(b[mask].sum()) for mask in cohorts])
    added, lost = np.asarray(added), np.asarray(lost)
    counts = np.array([int(mask.sum()) for mask in cohorts])
    labels = [MODEL_LABEL[x] for x in MODELS[1:]]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.9), sharey=True,
                             gridspec_kw={"width_ratios": [2, 3]})
    columns = [f"{name}\n(n={n})" for name, n in zip(("전체 정상", "재가동", *LOAD_NAMES), counts, strict=True)]
    transition_heatmap(axes[0], (added[:, :2], lost[:, :2]), counts[:2], labels, columns[:2], True)
    transition_heatmap(axes[1], (added[:, 2:], lost[:, 2:]), counts[2:], labels, columns[2:], False)
    axes[0].set_title("정상 · 해소 FP / 신규 FP", loc="left", fontsize=10)
    axes[1].set_title("이상 · 소실 TP / 추가 TP", loc="left", fontsize=10)
    title(fig, "동일 회차에서 기존 V4 PCA Q 대비 경보 변화",
          "주 실험 · 같은 회차를 짝지어 비교 · 색은 분모로 나눈 순변화(초록은 FP 감소 또는 TP 증가)")
    fig.subplots_adjust(left=.19, right=.99, top=.82, bottom=.12, wspace=.09)
    save(fig, out, "05_PAIRED_ALARM_TRANSITIONS.png")


def sensitivity_values(data, model, reference):
    if model == reference:
        return 1., 0, 0
    a = data.loc[data.model_id.eq(model) & data.reference_model_id.eq(reference)]
    b = data.loc[data.reference_model_id.eq(model) & data.model_id.eq(reference)]
    rows, inverse = (a, False) if len(a) else (b, True)
    rank = rows.loc[rows.scope.eq("evaluation_all"), "spearman_r"].iat[0]
    alarm = rows.loc[rows.scope.eq("normal_restart40")].iloc[0]
    if inverse:
        return rank, int(alarm.lost_alarms), int(alarm.added_alarms)
    return rank, int(alarm.added_alarms), int(alarm.lost_alarms)


def sensitivity(primary, metrics, out):
    table = primary.set_index("model_id")
    table["restart_fpr"] = table["restart_fp"] / table["restart_n"]
    fig, axes = plt.subplots(1, 2, figsize=(11.9, 5.9), sharey=True)
    lines = (("normal_fpr", "전체 정상 FPR", "normal"), ("restart_fpr", "재가동 FPR", "restart"),
             *((f"{key}_tpr", f"{key} TPR", key) for key in CONDITIONS))
    panels = ((("LOF_K10", "LOF_K20", "LOF_K40"), ("k=10", "k=20", "k=40"), "LOF 이웃 수"),
              (("OCSVM_G025", "OCSVM_G1", "OCSVM_G4"), ("γ×0.25", "γ×1", "γ×4"), "SVM gamma 배율"))
    for ax, (ids, ticks, label) in zip(axes, panels, strict=True):
        ordered = table.loc[list(ids)]
        for column, name, color in lines:
            ax.plot(range(3), ordered[column], "o-", lw=1.5, ms=4, color=COLORS[color], label=name)
        for x, model in enumerate(ids):
            rank, add, lose = sensitivity_values(metrics, model, ids[1])
            rank_label = f"{rank:.2f}" if np.isfinite(rank) else "N/A"
            ax.text(x, -.18, f"ρ {rank_label} · FP +{add}/−{lose}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        ax.set_xticks(range(3), ticks)
        ax.set_xlim(-.29, 2.29)
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=.14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.018, .995 - .54 / fig.get_figheight()),
               frameon=False, ncol=5, fontsize=8)
    title(fig, "사전에 고정한 LOF와 SVM 설정 민감도",
          "ρ는 기본 설정 대비 전체 평가의 점수 순위 상관 · FP 증감은 재가동40에서 같은 회차 비교")
    fig.subplots_adjust(left=.095, right=.98, top=.79, bottom=.23, wspace=.17)
    save(fig, out, "06_LOF_SVM_PARAMETER_SENSITIVITY.png")


def group_heldout(scores, out):
    a = evaluation(scores, "A_MAIN")
    b = evaluation(scores, "B_MAIN")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.55), sharey=True,
                             gridspec_kw={"width_ratios": [3, 4]})
    assert a.condition.eq("normal").all()
    masks_a = [a.condition.eq("normal"), a.cycle_index.ge(11), a.cycle_index.ge(21)]
    masks_b = [b.condition.eq(c) for c in ("normal", *CONDITIONS)]
    labels = [MODEL_LABEL[x] for x in FOLD_MODELS]
    group_info = ((a, masks_a, ("재가동 전체", "11–40회", "21–40회"), "OrRd"),
                  (b, masks_b, ("정상 FP", *LOAD_NAMES), ("OrRd", "YlGn", "YlGn", "YlGn")))
    for ax, (frame, masks, headings, cmap) in zip(axes, group_info, strict=True):
        values = [[frame.loc[mask & frame.model_id.eq(m), "flagged"].eq(True) for mask in masks]
                  for m in FOLD_MODELS]
        counts = np.array([[x.sum() for x in row] for row in values])
        totals = np.array([[len(x) for x in row] for row in values])
        rate_heatmap(ax, counts, totals, labels, headings, cmap, rates=True, breaks=False, rate_digits=2)
    axes[0].set_title("A_MAIN · 재가동 정상 제외 평가 (FP/n)", loc="left", fontsize=9.5)
    axes[1].set_title("B_MAIN · 하중 캠페인 제외 평가 (정상 FP/n · 이상 TP/n)", loc="left", fontsize=9.5)
    axes[1].axvline(.5, color="white", lw=2)
    title(fig, "두 그룹 제외 분할의 개별 결과",
          "A와 B는 별도 적합 · B는 11/10 정상으로 11/06 하중 캠페인을 평가한 회고 비교")
    fig.subplots_adjust(left=.18, right=.99, top=.78, bottom=.15, wspace=.11)
    save(fig, out, "07_SEPARATE_GROUP_HELDOUT_RESULTS.png")


def main(study):
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "text.color": "#17283b", "axes.labelcolor": "#263c53", "font.size": 9,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    out = study / "figures"
    out.mkdir(parents=True, exist_ok=True)
    primary = read(study, "PRIMARY_MODEL_COMPARISON.csv")
    scores = read(study, "CYCLE_SCORES.csv")
    overview(primary, out)
    session_chart(read(study, "NORMAL_SESSION_METRICS.csv"), out)
    paired_roc(read(study, "PAIRED_ANOMALY_METRICS.csv"), out)
    restart_margin(scores, out)
    transitions(scores, out)
    sensitivity(primary, read(study, "PARAMETER_SENSITIVITY.csv"), out)
    group_heldout(scores, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().study_dir)
