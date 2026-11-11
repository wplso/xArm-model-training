"""Compare saved bolt-model scores on the same previous evaluation cycles."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import quantiles


MODELS = ("LOGISTIC_BOLT", "LOGISTIC_BOLT_NORMAL_EXPANDED")
NORMAL_COUNTS = {"A121": 40, "B82": 20, "C182": 20}
PREVIOUS_COHORT = "previously_seen_development_regression"
STUDY = "hardware/xarm_model_versions/bolt_known_load_prospective_20261111"
CAMPAIGN = "hardware/No3_bolt_normal_retrain_20261111/analysis/retrained_bolt_normal"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def plot(panels, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 4.8), sharex=True, layout="constrained")
    for row, model in enumerate(MODELS):
        for col, label in enumerate(NORMAL_COUNTS):
            ax = axes[row, col]
            normal, load, result = panels[model, label]
            for y, color, scores in ((0, "#467c98", normal), (1, "#bb5c36", load)):
                ax.scatter(scores, y + rng.uniform(-.19, .19, len(scores)), s=16, alpha=.64,
                           color=color, edgecolors="none", zorder=3)
                q1, median, q3 = quantiles(scores, n=4, method="inclusive")
                ax.hlines(y, q1, q3, linewidth=4, color=color, zorder=2)
                ax.vlines(median, y - .24, y + .24, linewidth=2, color=color, zorder=4)
            ax.axvline(result["threshold"], color="#222222", ls="--", lw=1.2, zorder=5)
            ax.set_title(f"{label}  ·  AUROC {result['auroc']:.3f}  ·  {result['tp']}/30 detected",
                         fontsize=10, pad=7)
            ax.set_yticks([0, 1], ["Paired normal", "Load"] if col == 0 else ["", ""])
            ax.set(xlim=(-.02, 1.02), ylim=(-.4, 1.4), xticks=[0, .2, .4, .6, .8, 1])
            ax.grid(axis="x", color="#dfe4e8", linewidth=.6)
            ax.tick_params(axis="y", length=0)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
        axes[row, 0].set_ylabel(("Original" if row == 0 else "Expanded") + " bolt model", labelpad=12)
    fig.suptitle("Bolt model scores on the earlier load evaluation", fontsize=15, weight="bold")
    fig.supxlabel("Anomaly score 1 − P(normal)    ·    Dashed lines show the frozen thresholds", fontsize=10)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def summarize(project, output, render):
    study, source = project / STUDY, project / CAMPAIGN
    frozen_path, verification_path = study / "FROZEN_MODELS.json", source / "VERIFICATION.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8-sig"))
    verification = json.loads(verification_path.read_text(encoding="utf-8-sig"))
    specs = {
        MODELS[0]: (study, next(spec for spec in frozen["models"] if spec["model_id"] == MODELS[0])),
        MODELS[1]: (source, verification["candidate"]),
    }
    score_path = source / "CYCLE_SCORES.csv"
    metrics_path = source / "PREVIOUS_PAIRED_LOAD_METRICS.csv"
    tracked = {path: digest(path) for path in (frozen_path, verification_path, score_path, metrics_path)}
    for model, (folder, spec) in specs.items():
        path = folder / spec["artifact"]
        tracked[path] = digest(path)
        if spec["model_id"] != model or tracked[path] != spec["sha256"]:
            raise ValueError(f"Model hash mismatch for {model}")

    reference_metrics = {(row["model_id"], row["label"]): row for row in read_csv(metrics_path)}
    by_model = {model: {} for model in MODELS}
    for row in read_csv(score_path):
        if row["cohort"] != PREVIOUS_COHORT or row["model_id"] not in MODELS:
            continue
        model = row["model_id"]
        score, threshold = float(row["score"]), float(row["threshold"])
        if (not 0 <= score <= 1 or threshold != specs[model][1]["threshold"]
                or row["alarm"] != str(score > threshold)):
            raise ValueError("Saved threshold, alarm, or score mismatch")
        key = row["cycle_id"]
        if key in by_model[model]:
            raise ValueError("Duplicate score row")
        by_model[model][key] = (row["label"], row["pair_condition"], score)
    old, new = (by_model[model] for model in MODELS)
    if (len(old) != 180 or old.keys() != new.keys()
            or any(old[key][:2] != new[key][:2] for key in old)):
        raise ValueError("Evaluation cycle mismatch")

    panels, summary = {}, []
    for model, rows in by_model.items():
        threshold = specs[model][1]["threshold"]
        for label, expected_normals in NORMAL_COUNTS.items():
            normal = [score for actual, pair, score in rows.values() if actual == "normal" and pair == label]
            load = [score for actual, _, score in rows.values() if actual == label]
            if (len(normal), len(load)) != (expected_normals, 30):
                raise ValueError(f"Unexpected class counts for {label}")
            wins = sum((a > n) + .5 * (a == n) for a in load for n in normal)
            auc = wins / (len(normal) * len(load))
            fp = sum(value > threshold for value in normal)
            tp = sum(value > threshold for value in load)
            ref = reference_metrics[model, label]
            if fp != int(ref["fp"]) or tp != int(ref["tp"]) or not math.isclose(auc, float(ref["auroc"])):
                raise ValueError(f"Metric differs from saved reference for {model}/{label}")
            normal_q1, normal_median, normal_q3 = quantiles(normal, n=4, method="inclusive")
            load_q1, load_median, load_q3 = quantiles(load, n=4, method="inclusive")
            max_normal = max(normal)
            result = dict(model_id=model, label=label, normal_n=len(normal), load_n=len(load),
                          threshold=threshold, fp=fp, tp=tp, auroc=auc,
                          normal_q1=normal_q1, normal_median=normal_median, normal_q3=normal_q3,
                          normal_max=max_normal, load_q1=load_q1, load_median=load_median, load_q3=load_q3,
                          load_above_all_normals=sum(value > max_normal for value in load),
                          missed_above_all_normals=sum(max_normal < value <= threshold for value in load))
            panels[model, label] = (normal, load, result)
            summary.append(result)

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "SCORE_DISTRIBUTIONS.csv"
    if not render:
        with summary_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary[0])
            writer.writeheader()
            writer.writerows(summary)
    else:
        if len(read_csv(summary_path)) != len(summary):
            raise ValueError("Missing summary")
        plot(panels, output / "SCORE_DISTRIBUTIONS.png")
    if any(digest(path) != expected for path, expected in tracked.items()):
        raise ValueError("An input changed during analysis")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true", help="Render after writing the CSV")
    args = parser.parse_args()
    summarize(args.project.resolve(), args.output.resolve(), args.render)
