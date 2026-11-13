"""Exact V1 window and target reuse with a compact causal bin cache."""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[3]
DEFAULT_PROJECT = WORKSPACE if (WORKSPACE / "docs/xarm/DOCUMENT_INDEX.md").is_file() else WORKSPACE
ORIGINALS = Path(os.environ.get("XARM_FORECAST_PROJECT", DEFAULT_PROJECT)) / "hardware/xarm_forecasting"
V1_ARCHIVES = Path(os.environ.get("XARM_FORECAST_V1_CACHE", WORKSPACE / "forecast_multitarget_work"))
V1_MULTI = Path(os.environ.get("XARM_FORECAST_V1_MULTI", ORIGINALS / "forecast_multitarget_v1"))
V1_ELECTRIC = Path(os.environ.get("XARM_FORECAST_V1_E", ORIGINALS / "forecast_v1"))
NS = 1_000_000_000
GRID_S = .1
STRIDE_S = .2
HORIZONS = (.5, 1., 2.)
HISTORIES = (1., 2., 5.)
ROLES = ("train", "development", "calibration", "evaluation")
CACHE_ROLES = {"train": "fit", "development": "fit", "calibration": "calibration", "evaluation": "evaluation"}
TARGET_KEYS = ("y", "available_ns", "sample_ns", "target_seq", "target_mcu_read_end_ms",
               "target_saturated", "target_duplicate_h", "target_duplicate_prior_issue")
TARGETS = ("I_A", "V_V", *(f"ID{i}_raw" for i in range(1, 7)), "ax_g", "ay_g", "az_g", "RMS_g", "P95_g", "Peak_g")
UNITS = ("A", "V", *("raw",) * 6, *("g",) * 6)
TARGET_GROUPS = {"E": np.arange(2), "P": np.arange(2, 8), "A_RAW": np.arange(8, 11), "A_STAT": np.arange(11, 14)}
V1_FEATURE_SPEC = ROOT / "INPUT_GROUPS.json"
if not V1_FEATURE_SPEC.exists():
    V1_FEATURE_SPEC = V1_MULTI / "INPUT_GROUP_DEFINITIONS.json"
SPEC = json.loads(V1_FEATURE_SPEC.read_text(encoding="utf-8"))
CHANNELS = tuple(SPEC["channels"])
GROUPS = {name: [CHANNELS.index(column) for column in columns] for name, columns in SPEC["groups"].items()}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, contents):
    path.write_text(json.dumps(contents, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_archive(role):
    """Load a validated local pickle produced by the project V1 data builder."""
    path = V1_ARCHIVES / "data" / f"{role}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def _history_indexes(issue_rows, length, origin):
    """Map each requested issue to bins of its own cycle only."""
    offsets = np.arange(round(length / GRID_S) - 1, -1, -1)
    indices = 2 * issue_rows[:, None] + 1 - offsets
    return indices, indices >= 2 * origin[issue_rows, None]


def _encode_role(role, root):
    source = V1_ARCHIVES / "data" / f"{role}.pkl"
    started = perf_counter()
    expected = json.loads((V1_MULTI / "verification" / f"TIMING_{role}.json").read_text(encoding="utf-8"))
    source_hash = sha(source)
    assert source_hash == expected["cache_sha256"], f"V1 cache hash differs: {source}"
    archived = read_archive(role)
    issue = archived["issue_meta"].reset_index(drop=True)
    match = archived["meta"].reset_index(drop=True)
    assert tuple(archived["targets"]) == TARGETS and tuple(archived["channels"]) == CHANNELS
    assert set(archived["history_valid"]) == {.5, 1., 2.}
    assert np.array_equal(issue.feature_row, np.arange(len(issue)))
    assert not issue.duplicated(["cycle_id", "issue_ns"]).any()
    assert np.all((issue.issue_ns > issue.No3_begin_ns) & (issue.issue_ns < issue.No3_end_ns))
    starts = issue.cycle_id.ne(issue.cycle_id.shift()).to_numpy()
    origins = np.maximum.accumulate(np.where(starts, np.arange(len(issue)), 0))
    rel_ns = issue.issue_ns.to_numpy() - issue.No3_begin_ns.to_numpy()
    np.testing.assert_array_equal(rel_ns, (np.arange(len(issue)) - origins + 1) * round(STRIDE_S * NS))
    bins = archived["x"][:, -2:, :].reshape(-1, len(CHANNELS)).astype(np.float32, copy=False)
    for history in (.5, 1., 2.):
        indexes, inside = _history_indexes(np.arange(len(issue)), history, origins)
        replay = bins[np.maximum(indexes, 0)].copy()
        replay[~inside] = np.nan
        np.testing.assert_array_equal(replay, archived["x"][:, -round(history / GRID_S):])
        reconstructed_valid = inside.all(axis=1) & np.isfinite(replay[:, :, :2]).all(axis=(1, 2))
        np.testing.assert_array_equal(reconstructed_valid, archived["history_valid"][history])
    issue_ids = match.feature_row.to_numpy()
    horizon_ids = np.array([HORIZONS.index(h) for h in match.horizon_s])
    assert len(match) == len(issue) * len(HORIZONS)
    assert len(set(zip(issue_ids, horizon_ids))) == len(match)
    assert len(issue) == expected["issue_rows"] and len(match) == expected["issue_horizon_rows"]
    np.testing.assert_array_equal(match.issue_ns, issue.issue_ns.to_numpy()[issue_ids])
    np.testing.assert_array_equal(match.nominal_target_ns, match.issue_ns + (match.horizon_s * NS).astype(np.int64))
    target_valid = np.isfinite(archived["y"])
    assert np.array_equal(target_valid, archived["available_ns"] >= 0)
    sample_ns = archived["sample_ns"]
    available = archived["available_ns"]
    assert np.all(~target_valid | ((sample_ns >= match.nominal_target_ns.to_numpy()[:, None]) & (sample_ns <= available)))
    assert np.all(~target_valid | (available <= match.end_ns.to_numpy()[:, None]))
    assert np.all(~target_valid[:, 11:] | (available[:, 11:] == match.nominal_target_ns.to_numpy()[:, None] + 200_000_000))

    folder = root / "cache" / role
    folder.mkdir(parents=True, exist_ok=False)
    issue.to_csv(folder / "issues.csv", index=False)
    np.save(folder / "bins.npy", bins)
    np.save(folder / "origins.npy", origins)
    np.save(folder / "persistence.npy", archived["persistence"])
    for key in TARGET_KEYS:
        array = archived[key]
        compact = np.empty((len(issue), len(HORIZONS), len(TARGETS)), dtype=array.dtype)
        compact[issue_ids, horizon_ids] = array
        np.testing.assert_array_equal(compact[issue_ids, horizon_ids], array)
        np.save(folder / f"{key}.npy", compact)
    issue_bytes = sum(path.stat().st_size for path in folder.iterdir())
    return {
        "role": role, "archive_path": str(source), "archive_sha256": source_hash,
        "archive_bytes": source.stat().st_size, "new_cache_bytes": issue_bytes,
        "cycles": int(issue.cycle_id.nunique()), "issues": len(issue),
        "elapsed_s": perf_counter() - started,
        "windows_and_validity_exact_for_lengths_s": [.5, 1., 2.],
        "all_horizon_targets_and_flags_exact": True,
        "causal_prefix_points_verified_in_V1": archived["verification"]["tested_input_prefix_points"],
        "no_synthetic_adxl_sample_time": True,
    }


def prepare(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    if any((root / "cache" / role).exists() for role in ("fit", "calibration", "evaluation")):
        raise FileExistsError("Use --output to create a cache without overwriting an existing run")
    audits = []
    for role in ("fit", "calibration", "evaluation"):
        result = _encode_role(role, root)
        audits.append(result)
        print(f"{role}: {result['cycles']} cycles, {result['issues']} issues", flush=True)
    source_files = {
        "SPLIT_MANIFEST.csv": V1_MULTI / "SPLIT_MANIFEST.csv",
        "INPUT_GROUPS.json": V1_MULTI / "INPUT_GROUP_DEFINITIONS.json",
        "TARGET_DEFINITIONS.json": V1_MULTI / "TARGET_DEFINITIONS.json",
        "TARGET_ALIGNMENT_POLICY.json": V1_MULTI / "TARGET_ALIGNMENT_POLICY.json",
        "V1_SOURCE_FILE_HASHES.csv": V1_MULTI / "verification" / "SOURCE_FILE_HASHES.csv",
    }
    for name, source in source_files.items():
        shutil.copyfile(source, root / name)
    manifest = pd.read_csv(root / "SPLIT_MANIFEST.csv", dtype={"cycle_id": str})
    assert not manifest.cycle_id.duplicated().any()
    assert manifest.groupby("role").size().to_dict() == {"train": 429, "development": 120, "calibration": 40, "evaluation": 930}
    assert manifest.group_id.nunique() == 10
    assert manifest.groupby("group_id").role.nunique().eq(1).all()
    assert manifest.loc[manifest.role.ne("evaluation"), "label"].eq("normal").all()
    for role, batch in CACHE_ROLES.items():
        issues = pd.read_csv(root / "cache" / batch / "issues.csv", usecols=["role", "cycle_id"])
        assert set(issues.loc[issues.role.eq(role), "cycle_id"]) == set(manifest.loc[manifest.role.eq(role), "cycle_id"])
    save_json(root / "CACHE_AUDIT.json", {
        "cache_archive_audits": audits, "cycle_manifest_sha256": sha(root / "SPLIT_MANIFEST.csv"),
        "input_groups_sha256": sha(root / "INPUT_GROUPS.json"), "target_definitions_sha256": sha(root / "TARGET_DEFINITIONS.json"),
        "raw_manifest_sha256": sha(V1_MULTI / "verification" / "SOURCE_FILE_HASHES.csv"),
        "raw_files_reread_for_cache": False, "role_counts": manifest.groupby("role").size().to_dict(),
        "group_count": int(manifest.group_id.nunique()),
    })


class Data:
    """Select causal windows and aligned targets from memory-mapped cache files."""
    def __init__(self, root=ROOT):
        self.root = Path(root)

    @lru_cache(maxsize=None)
    def _issue(self, role):
        return pd.read_csv(self.root / "cache" / CACHE_ROLES[role] / "issues.csv", low_memory=False)

    @lru_cache(maxsize=None)
    def _array(self, role, key):
        return np.load(self.root / "cache" / CACHE_ROLES[role] / f"{key}.npy", mmap_mode="r")

    @lru_cache(maxsize=16)
    def _indexes(self, role, length, require_ina=True):
        issue = self._issue(role)
        rows = np.flatnonzero(issue.role.eq(role))
        indexes, inside = _history_indexes(rows, length, self._array(role, "origins"))
        valid = inside.all(axis=1)
        if require_ina:
            finite = np.isfinite(self._array(role, "bins")[indexes[valid], :2]).all(axis=(1, 2))
            valid[valid] = finite
        return rows[valid], indexes[valid]

    def _horizons(self, horizons, w):
        if not np.isclose(w, .2):
            raise ValueError("V1-compatible cache preserves w=.2 only")
        return [HORIZONS.index(float(h)) for h in np.atleast_1d(horizons)]

    def get(self, role, L, horizons, group, w=.2):
        rows, indexes = self._indexes(role, L)
        columns = GROUPS[group]
        # Index bins and sensor columns together to avoid copying the full input group.
        x = self._array(role, "bins")[indexes[..., None], columns]
        h = self._horizons(horizons, w)
        y = self._array(role, "y")[rows[:, None], h]
        return x, y, self._issue(role).iloc[rows].reset_index(drop=True)

    def targets(self, role, L, horizons, w=.2):
        rows, _ = self._indexes(role, L)
        h = self._horizons(horizons, w)
        out = {key: self._array(role, key)[rows[:, None], h] for key in TARGET_KEYS}
        out["meta"] = self._issue(role).iloc[rows].reset_index(drop=True)
        out["nominal_target_ns"] = out["meta"].issue_ns.to_numpy()[:, None] + (np.asarray(horizons) * NS).astype(np.int64)
        out["valid_mask"] = np.isfinite(out["y"])
        return out

    def persistence(self, role, L):
        rows, _ = self._indexes(role, L)
        return self._array(role, "persistence")[rows]

    def direct(self, role):
        """Established direct RMS/Peak monitor at all available 0.2 s issue points."""
        issue = self._issue(role)
        rows = np.flatnonzero(issue.role.eq(role))
        return issue.iloc[rows].reset_index(drop=True), self._array(role, "persistence")[rows[:, None], [11, 13]]

    def legacy_cache(self, role):
        return read_archive(CACHE_ROLES.get(role, role))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT, help="Independent directory for regenerated compact cache")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output)
