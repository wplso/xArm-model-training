"""Independently count primary and paired results from saved per-cycle scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FAMILIES = ("B0", "B1", "B2")
CONDITIONS = ("normal", "A", "B", "C")
EVALUATIONS = {"E0": (259, 30, 30, 30), "E1": (219, 20, 20, 20), "E2": (219, 20, 20, 20)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def transition(left, right):
    require(left.index.equals(right.index), "Comparison IDs differ")
    return {
        "n": len(left),
        "additional": int((~left & right).sum()),
        "resolved_or_lost": int((left & ~right).sum()),
        "common_alarm": int((left & right).sum()),
        "common_no_alarm": int((~left & ~right).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    output = parser.parse_args().output
    mask = pd.read_csv(output / "PREPARATION_MASK.csv").set_index("cycle_id")
    scores = pd.read_csv(output / "metrics/CYCLE_SCORES.csv", float_precision="round_trip")
    require(len(mask) == 650 and mask.index.is_unique and int(mask.quality_eligible.sum()) == 649, "Candidate audit count differs")
    require(not scores.duplicated(["cycle_id", "family", "evaluation"]).any(), "Duplicate score key")
    scores = scores.set_index("cycle_id")
    pair_counts = {}
    primary = {}
    for family in FAMILIES:
        frames = {
            code: scores.loc[(scores.family == family) & (scores.evaluation == code)].sort_index()
            for code in EVALUATIONS
        }
        for code, frame in frames.items():
            require(frame.index.equals(mask.sort_index().index), f"Incomplete per-cycle table: {family}/{code}")
            valid = frame.loc[frame.quality_eligible]
            require(np.isfinite(valid.score).all(), f"Missing valid score: {family}/{code}")
            require((valid.score > valid.threshold).equals(valid.flagged.astype(bool)), f"Strict threshold mismatch: {family}/{code}")
            require(frame.loc[~frame.quality_eligible, "policy_status"].eq("NOT_EVALUATED_QUALITY").all(), "Quality status differs")
            selected = frame.loc[frame.eligible_for_primary_metric]
            require(selected.quality_eligible.all(), f"Ineligible primary cycle: {family}/{code}")
            if code != "E0":
                require(not selected.preparation_excluded.any(), f"Preparation leakage: {family}/{code}")
                held = valid.preparation_excluded & valid.original_role.isin(["test", "evaluation_only"])
                require(held.sum() == 70 and valid.loc[held, "policy_status"].eq("NOT_EVALUATED_PREPARATION").all(), "Preparation status differs")
            counts = tuple(int((selected.condition == condition).sum()) for condition in CONDITIONS)
            require(counts == EVALUATIONS[code] and len(selected) == sum(counts), f"Wrong primary counts: {family}/{code}")
            primary[f"{family}/{code}"] = {condition: int(selected.loc[selected.condition == condition, "flagged"].sum())
                                            for condition in CONDITIONS}
        left, retained, retrained = (frames[key] for key in ("E0", "E1", "E2"))
        for name in ("score", "threshold", "flagged"):
            require(left[name].equals(retained[name]), f"Frozen E0/E1 {name} differs: {family}")
        for condition in CONDITIONS:
            after = retained.eligible_for_primary_metric & (retained.condition == condition)
            new = retrained.eligible_for_primary_metric & (retrained.condition == condition)
            result = transition(retained.loc[after, "flagged"].astype(bool), retrained.loc[new, "flagged"].astype(bool))
            require(result["additional"] + result["common_alarm"] == primary[f"{family}/E2"][condition], "E2 sum mismatch")
            require(result["resolved_or_lost"] + result["common_alarm"] == primary[f"{family}/E1"][condition], "E1 sum mismatch")
            pair_counts[f"{family}/{condition}"] = result

    normal = pd.read_csv(output / "metrics/NORMAL_FULL_VS_POST_PREP.csv")
    anomaly = pd.read_csv(output / "metrics/ANOMALY_FULL_VS_POST_PREP.csv")
    for row in normal.loc[normal.primary_scope & normal.group_type.eq("overall")].itertuples(index=False):
        require(primary[f"{row.family}/{row.evaluation}"]["normal"] == row.fp, "Published FP count differs")
    for row in anomaly.loc[anomaly.primary_scope].itertuples(index=False):
        require(primary[f"{row.family}/{row.evaluation}"][row.condition] == row.tp, "Published TP count differs")
    diagnostics = pd.read_csv(output / "metrics/EXCLUDED_SEGMENT_DIAGNOSTICS.csv")
    require((diagnostics.policy_not_evaluated_n == diagnostics.n.where(diagnostics.evaluation.ne("E0"), 0)).all(),
            "Published policy hold count differs")
    checks = {"passed": True, "primary_alarms": primary, "same_cycle_transitions": pair_counts}
    (output / "COMPARISON_CROSSCHECK.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {len(scores)} per-cycle rows, {len(primary)} primary summaries and {len(pair_counts)} exact-ID comparisons")


if __name__ == "__main__":
    main()
