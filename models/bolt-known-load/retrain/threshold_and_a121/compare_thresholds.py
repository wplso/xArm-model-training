"""Replay a fixed threshold grid on archived scores without changing a model."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


STUDY = "hardware/xarm_model_versions/bolt_known_load_prospective_20261111"
RESULTS = "hardware/No3_bolt_normal_retrain_20261111/analysis/retrained_bolt_normal"
OLD = "LOGISTIC_BOLT"
NEW = "LOGISTIC_BOLT_NORMAL_EXPANDED"
GRID = tuple(i / 100 for i in range(80, 9, -5))
SEEN = "previously_seen_development_regression"
GROUPS = {
    "new_normal": ("new_normal_holdout", "normal", 60),
    "calibration_normal": ("new_calibration", "normal", 60),
    "earlier_normal": (SEEN, "normal", 90),
    "A121": (SEEN, "A121", 30),
    "B82": (SEEN, "B82", 30),
    "C182": (SEEN, "C182", 30),
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [row for row in rows if row["model_id"] == NEW]
    x = [row["threshold"] for row in points]
    fig, ax = plt.subplots(figsize=(8.1, 4.4), layout="constrained")
    for group, label, color in (
        ("new_normal", "New normal false positives", "#273f56"),
        ("B82", "B82 detection", "#167a72"),
        ("C182", "C182 detection", "#a84b32"),
        ("A121", "A121 detection", "#7d687d"),
    ):
        values = [100 * row[f"{group}_rate"] for row in points]
        ax.plot(x, values, "o-", color=color, linewidth=1.6, markersize=3.5, label=label)
    ax.axvline(x[0], color="#506070", ls="--", lw=1, alpha=.8)
    ax.set(title="Offline threshold comparison for the retrained bolt model",
           xlabel="Threshold (left to right: lower threshold)", ylabel="Rate (%)",
           ylim=(-2, 103), xlim=(.84, .075),
           xticks=[.80, .70, .60, .50, .40, .30, .20, .10])
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.grid(axis="y", color="#dfe4e8", lw=.65)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(output, dpi=175)
    plt.close(fig)


def run(project, output, render):
    study, results = project / STUDY, project / RESULTS
    frozen_path, verification_path = study / "FROZEN_MODELS.json", results / "VERIFICATION.json"
    specs = {
        OLD: (study, next(row for row in json.loads(frozen_path.read_text(encoding="utf-8-sig"))["models"]
                          if row["model_id"] == OLD)),
        NEW: (results, json.loads(verification_path.read_text(encoding="utf-8-sig"))["candidate"]),
    }
    score_path, reference_path = results / "CYCLE_SCORES.csv", results / "COMPARISON.csv"
    tracked = {path: sha(path) for path in (frozen_path, verification_path, score_path, reference_path)}
    for model, (folder, spec) in specs.items():
        path = folder / spec["artifact"]
        tracked[path] = sha(path)
        if spec["model_id"] != model or spec["sha256"] != tracked[path]:
            raise ValueError(f"Model mismatch for {model}")

    scores = {model: {group: {} for group in GROUPS} for model in specs}
    group_keys = {(cohort, label): group for group, (cohort, label, _) in GROUPS.items()}
    with score_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            model, cycle = row["model_id"], row["cycle_id"]
            if model not in specs:
                continue
            group = group_keys[row["cohort"], row["label"]]
            value, threshold = float(row["score"]), specs[model][1]["threshold"]
            if (cycle in scores[model][group] or not 0 <= value <= 1
                    or float(row["threshold"]) != threshold or row["alarm"] != str(value > threshold)):
                raise ValueError(f"Duplicate row or score/threshold mismatch for {model}/{cycle}")
            scores[model][group][cycle] = value
    for group, (_, _, count) in GROUPS.items():
        if len(scores[OLD][group]) != count or scores[OLD][group].keys() != scores[NEW][group].keys():
            raise ValueError(f"Paired cycle set mismatch for {group}")

    with reference_path.open(encoding="utf-8-sig", newline="") as stream:
        archived = {(row["model_id"], row["cohort"], row["label"]): row for row in csv.DictReader(stream)}
    comparisons = []
    for model, (_, spec) in specs.items():
        frozen_threshold = spec["threshold"]
        thresholds = (frozen_threshold,) if model == OLD else (frozen_threshold, *GRID)
        if thresholds[0] != max(thresholds):
            raise ValueError("The experimental grid must be below the frozen threshold")
        for threshold in thresholds:
            row = dict(model_id=model, threshold=threshold, frozen_threshold=threshold == frozen_threshold)
            for group, (cohort, label, count) in GROUPS.items():
                alarms = sum(value > threshold for value in scores[model][group].values())
                if threshold == frozen_threshold:
                    ref = archived[model, cohort, label]
                    if alarms != int(ref["alarms"]) or count != int(ref["n"]):
                        raise ValueError(f"Frozen result mismatch for {model}/{group}")
                row.update({f"{group}_n": count, f"{group}_alarms": alarms, f"{group}_rate": alarms / count})
            comparisons.append(row)
    expanded = [row for row in comparisons if row["model_id"] == NEW]
    for group in GROUPS:
        alarm_counts = [row[f"{group}_alarms"] for row in expanded]
        if alarm_counts != sorted(alarm_counts):
            raise ValueError(f"Nonmonotonic sweep for {group}")

    output.mkdir(parents=True, exist_ok=True)
    table = output / "THRESHOLD_SWEEP.csv"
    if render:
        with table.open(encoding="utf-8-sig", newline="") as stream:
            archived_sweep = list(csv.DictReader(stream))
        if archived_sweep != [{key: str(value) for key, value in row.items()} for row in comparisons]:
            raise ValueError("Sweep table differs from the recalculated values")
        plot(comparisons, output / "THRESHOLD_SWEEP.png")
    else:
        with table.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=comparisons[0])
            writer.writeheader()
            writer.writerows(comparisons)
    if any(sha(path) != value for path, value in tracked.items()):
        raise ValueError("An input changed during replay")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    run(args.project.resolve(), args.output.resolve(), args.render)
