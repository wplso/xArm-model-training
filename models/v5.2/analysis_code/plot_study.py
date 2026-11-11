"""Render the six V5.2 preparation exclusion charts from saved study results."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

FAMILIES = ("B0", "B1", "B2")
EVALUATIONS = ("E0", "E1", "E2")
CONDITIONS = ("A", "B", "C")
COLLECTIONS = {
    "nov06_main400": "볼트 고정 본수집",
    "nov06_extra_normal30": "추가 정상 17+13",
    "nov06_preA": "A 직전 정상",
    "nov06_A": "A 이상 · 운반 121g",
    "nov06_preB": "B 직전 정상 (예외)",
    "nov06_B": "B 이상 · 상부 82g",
    "nov06_preC": "C 직전 정상 (예외)",
    "nov06_C": "C 이상 · 하부 182g",
    "nov10_normal40": "11/10 첫 정상40",
    "nov10_restart40": "11/10 재가동40",
}
COLORS = {"retained": "#45796e", "preparation": "#dba775", "quality": "#929ca7",
          "frozen": "#326188", "retrained": "#a55c37"}
CONDITION_LABELS = ("A 121g", "B 상부 82g", "C 하부 182g")
MATCHED = ("E1", "E2")


def read(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def header(fig, title, subtitle):
    fig.suptitle(title, x=.018, y=.995, ha="left", fontsize=16, fontweight="bold")
    fig.text(.018, .995 - .32 / fig.get_figheight(), subtitle, va="top",
             fontsize=9, color="#455870")


def save(fig, folder, filename):
    path = folder / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)
    return path


def heatmap(ax, count, total, row_labels, col_labels, cmap="OrRd", show_rate=False):
    count = np.asarray(count, dtype=float)
    total = np.asarray(total, dtype=float)
    assert np.isfinite(count).all() and (total > 0).all()
    rates = count / total
    ax.imshow(rates, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i, j in np.ndindex(rates.shape):
        label = f"{int(count[i, j])}/{int(total[i, j])}"
        if show_rate:
            label += f"\n{rates[i, j]:.1%}"
        ax.text(j, i, label, ha="center", va="center", fontsize=9,
                color="white" if rates[i, j] >= .68 else "#182e44")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.spines[["top", "bottom", "left", "right"]].set_visible(False)


def retention(mask, out):
    assert mask["cycle_id"].is_unique
    assert set(mask["logical_collection_id"]) == set(COLLECTIONS)
    counts = mask.assign(
        retained=mask["quality_eligible"] & mask["post_preparation_eligible"],
        preparation=mask["quality_eligible"] & mask["preparation_excluded"],
        quality=~mask["quality_eligible"],
    ).groupby("logical_collection_id").agg(
        retained=("retained", "sum"), preparation=("preparation", "sum"),
        quality=("quality", "sum"), total=("cycle_id", "size"),
    ).loc[list(COLLECTIONS)]
    np.testing.assert_array_equal(counts[["retained", "preparation", "quality"]].sum(axis=1), counts["total"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.3), gridspec_kw={"height_ratios": [1, 5.4]})
    for ax, part in zip(axes, (counts.iloc[:1], counts.iloc[1:]), strict=True):
        y = np.arange(len(part))
        left = np.zeros(len(part))
        for key in ("retained", "preparation", "quality"):
            values = part[key].to_numpy()
            ax.barh(y, values, left=left, height=.64, color=COLORS[key], linewidth=0)
            for i, number in enumerate(values):
                if number and key != "quality":
                    ax.text(left[i] + number / 2, i, str(int(number)), color="#132c41" if
                            key == "preparation" else "white", ha="center", va="center", fontsize=9)
            left += values
        for i, row in enumerate(part.itertuples()):
            quality = f" (품질 제외 {row.quality})" if row.quality else ""
            ax.annotate(f"전체 {row.total}{quality}", (row.total, i), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(y, [COLLECTIONS[key] for key in part.index])
        ax.invert_yaxis()
        ax.set_xlim(0, part["total"].max() * 1.25)
        ax.set_xlabel("회차 수")
        ax.grid(axis="x", alpha=.16)
    keys = ("retained", "preparation", "quality")
    fig.legend([Patch(color=COLORS[k]) for k in keys], ["유지", "준비구간 분리", "기존 품질 제외"],
               loc="upper left", bbox_to_anchor=(.018, .995 - .54 / fig.get_figheight()),
               frameon=False, ncol=3)
    header(fig, "논리적 수집별 유지·제외·전체 회차",
           "본수집 첫 50회 한 번 · 짧은 수집 첫 10회 · B/C 직전 정상10회 전부 유지 · 미송신 #400 제외")
    fig.subplots_adjust(left=.24, top=.84, bottom=.09, hspace=.44)
    return save(fig, out, "01_COLLECTION_RETENTION.png")


def normal_fpr(scores, out):
    normal = scores.loc[scores["condition"].eq("normal") & scores["eligible_for_primary_metric"]]
    groups = [group for group in COLLECTIONS if group in set(normal["logical_collection_id"])]
    labels = ["전체 고유 정상"] + [
        "본수집 후반 Test" if group == "nov06_main400" else COLLECTIONS[group] for group in groups
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.7), sharey=True)
    for family, ax in zip(FAMILIES, axes, strict=True):
        rows = normal[normal["family"].eq(family)]
        assert not rows.duplicated(["evaluation", "cycle_id"]).any()
        full = rows.groupby("evaluation")["flagged"].agg(["sum", "size"]).reindex(EVALUATIONS)
        by_group = rows.groupby(["logical_collection_id", "evaluation"])["flagged"].agg(["sum", "size"])
        order = pd.MultiIndex.from_product([groups, EVALUATIONS])
        parts = by_group.reindex(order)
        count = np.vstack([full["sum"].to_numpy(), parts["sum"].to_numpy().reshape(-1, 3)])
        total = np.vstack([full["size"].to_numpy(), parts["size"].to_numpy().reshape(-1, 3)])
        np.testing.assert_array_equal(total[:, 1], total[:, 2])
        heatmap(ax, count, total, labels, EVALUATIONS, show_rate=True)
        ax.axhline(.5, color="white", lw=2)
        ax.axvline(.5, color="white", lw=2)
        ax.set_title(family, loc="left", fontsize=11)
    header(fig, "정상 FPR · 전체 평가와 준비 이후 평가",
           "셀은 FP/정상 회차 · E0는 전체 · E1/E2는 같은 유지 회차 · E0와 E1은 같은 모델·경계")
    fig.subplots_adjust(left=.24, right=.99, top=.83, bottom=.1, wspace=.16)
    return save(fig, out, "02_NORMAL_FPR_FULL_AND_RETAINED.png")


def matched_anomaly(anomaly, out):
    table = anomaly[anomaly["scope"].eq("post_preparation") & anomaly["evaluation"].isin(MATCHED)]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.3), sharey="row")
    for col, family in enumerate(FAMILIES):
        rows = table[table["family"].eq(family)].set_index(["condition", "evaluation"])
        rows = rows.reindex(pd.MultiIndex.from_product([CONDITIONS, MATCHED]))
        for row, (num, den, cmap, measure) in enumerate((
            ("tp", "anomaly_n", "YlGn", "이상 탐지율"),
            ("fp", "paired_normal_n", "OrRd", "짝 정상 오탐률"),
        )):
            ax = axes[row, col]
            total = rows[den].to_numpy().reshape(3, 2)
            np.testing.assert_array_equal(total[:, 0], total[:, 1])
            heatmap(ax, rows[num].to_numpy().reshape(3, 2), total, CONDITION_LABELS, MATCHED, cmap)
            ax.set_title(f"{family} · {measure}", loc="left", fontsize=10)
    header(fig, "E1과 E2의 같은 유지 회차 비교",
           "위는 TP/이상, 아래는 FP/짝 정상 · E1 기존 모델, E2 재학습 · B/C 짝 정상10회 예외 유지")
    fig.subplots_adjust(left=.15, top=.83, bottom=.1, hspace=.47, wspace=.19)
    return save(fig, out, "03_MATCHED_ANOMALY_AND_PAIRED_NORMAL.png")


def chronology(scores, out):
    evals = (("E1", "frozen", "E1 기존"), ("E2", "retrained", "E2 재학습"))
    cohorts = ("nov10_normal40", "nov10_restart40")
    fig, axes = plt.subplots(3, 2, figsize=(11.7, 7.8), sharex=True)
    for row, family in enumerate(FAMILIES):
        for col, collection in enumerate(cohorts):
            ax = axes[row, col]
            block = scores.loc[scores["logical_collection_id"].eq(collection) & scores["family"].eq(family)]
            for evaluation, color_key, _ in evals:
                subset = block[block["evaluation"].eq(evaluation)].sort_values("original_cycle_order")
                numbers = subset["original_cycle_order"]
                assert numbers.tolist() == list(range(1, 41))
                excluded = subset["preparation_excluded"]
                assert numbers[excluded].tolist() == list(range(1, 11))
                color = COLORS[color_key]
                ratios = subset["score_over_threshold"]
                ax.plot(numbers, ratios, color=color, alpha=.82, lw=1.15)
                alerts = subset["flagged"].eq(True) & ~excluded
                ax.scatter(numbers[alerts], ratios[alerts], facecolor="white", edgecolors=color,
                           s=24, lw=1.25, zorder=3, marker="o" if evaluation == "E1" else "D")
            ax.axvspan(.5, 10.5, color="#dac8b5", alpha=.43, lw=0)
            ax.axhline(1, color="#52647a", ls="--", lw=.85)
            ax.set_xlim(.5, 40.5)
            ax.set_title(f"{family} · {COLLECTIONS[collection]} (유지 30회)", loc="left", fontsize=9)
            ax.set_ylabel("Q / 모델 임계값")
            ax.grid(axis="y", alpha=.14)
    for ax in axes[-1]:
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
        ax.set_xlabel("원래 수행 회차")
    handles = [Line2D([], [], color=COLORS[key]) for _, key, _ in evals]
    handles.append(Patch(color="#dac8b5", alpha=.43))
    fig.legend(handles, [label for _, _, label in evals] + ["준비구간 판정 보류"],
               loc="upper left", bbox_to_anchor=(.018, .995 - .54 / fig.get_figheight()),
               frameon=False, ncol=3)
    header(fig, "두 정상40회 시간순 점수와 준비 이후 오탐",
           "각 열은 별도 세션 · 1–10회 판정 보류 · 유지 구간 경보는 원/마름모 · 경계는 1")
    fig.subplots_adjust(top=.81, bottom=.09, hspace=.43, wspace=.25)
    return save(fig, out, "04_NORMAL40_SCORE_TIMELINES.png")


def excluded_anomaly(anomaly, out):
    scopes = ("excluded_preparation", "post_preparation")
    policies = ("E0", "E2")
    table = anomaly[anomaly["scope"].isin(scopes) & anomaly["evaluation"].isin(policies)]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.2), sharey="row")
    for col, family in enumerate(FAMILIES):
        for row, (scope, title) in enumerate(zip(scopes, ("준비구간 · 판정 보류", "준비 이후 · 유지"), strict=True)):
            ax = axes[row, col]
            rows = table[table["family"].eq(family) & table["scope"].eq(scope)].set_index(
                ["condition", "evaluation"],
            ).reindex(pd.MultiIndex.from_product([CONDITIONS, policies]))
            heatmap(ax, rows["tp"].to_numpy().reshape(3, 2), rows["anomaly_n"].to_numpy().reshape(3, 2),
                    CONDITION_LABELS, ("기존 모델", "재학습 모델"), "OrRd" if row == 0 else "YlGn")
            ax.set_title(f"{family} · {title}", loc="left", fontsize=9)
    header(fig, "이상 첫 10회와 유지 20회의 모델 경보",
           "셀은 모델 경보/해당 이상 회차 · 위쪽 경보는 참고 진단이며 실제 정책에서는 판정 보류")
    fig.subplots_adjust(left=.15, top=.83, bottom=.1, hspace=.47, wspace=.19)
    return save(fig, out, "05_EXCLUDED_AND_RETAINED_ANOMALY.png")


def calibration(scores, out):
    table = scores[scores["original_role"].eq("validation")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.7), sharey=True)
    for row, family in enumerate(FAMILIES):
        model_data = table[table["family"].eq(family)]
        old_ids = set(model_data.loc[model_data["evaluation"].eq("E0"), "cycle_id"])
        new_ids = set(model_data.loc[model_data["evaluation"].eq("E2"), "cycle_id"])
        assert old_ids == new_ids
        for col, (evaluation, color_key, label) in enumerate((
            ("E0", "frozen", "기존 학습"), ("E2", "retrained", "준비 제외 재학습"),
        )):
            ax = axes[row, col]
            block = model_data[model_data["evaluation"].eq(evaluation)]
            values = np.sort(block["score"])
            n = len(values)
            assert n == len(old_ids) and np.isfinite(values).all()
            cutoff = block["threshold"].iat[0]
            assert np.isclose(np.quantile(values, .95, method="higher"), cutoff, rtol=1e-12, atol=1e-14)
            ax.step(values, np.arange(1, n + 1) / n, where="post", color=COLORS[color_key], lw=1.6)
            ax.axvline(cutoff, color="#334961", lw=1.1, ls="--", label=f"임계값 {cutoff:.4g}")
            ax.axhline(.95, color="#9cabb9", lw=.75)
            ax.legend(loc="lower right", frameon=False, fontsize=8)
            ax.set_title(f"{family} · {label} (n={n})", loc="left", fontsize=9)
            ax.set_ylabel("정상 누적 비율")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.set_xlabel("모델의 재구성 Q")
            ax.set_ylim(0, 1.04)
            ax.grid(axis="x", alpha=.13)
    header(fig, "준비 제외 재학습 전후의 정상 교정 분포",
           "양쪽은 동일 정상 Calibration ID · 각 모델의 Q에서 q95(higher) 산출 · 평가는 별도")
    fig.subplots_adjust(top=.86, bottom=.08, hspace=.57, wspace=.23)
    return save(fig, out, "06_CALIBRATION_BEFORE_AND_AFTER.png")


def main(study):
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "text.color": "#18283a", "axes.labelcolor": "#283a51", "font.size": 9,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    out = study / "figures"
    out.mkdir(parents=True, exist_ok=True)
    mask = read(study / "PREPARATION_MASK.csv")
    scores = read(study / "metrics" / "CYCLE_SCORES.csv")
    anomaly = read(study / "metrics" / "ANOMALY_FULL_VS_POST_PREP.csv")
    charts = [retention(mask, out), normal_fpr(scores, out), matched_anomaly(anomaly, out),
              chronology(scores, out), excluded_anomaly(anomaly, out), calibration(scores, out)]
    for path in charts:
        assert path.stat().st_size > 1024, f"Empty chart {path}"
        print(path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().study_dir)
